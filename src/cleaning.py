"""Turn clean order lines into purchases and adjustments.

`lines` is the output of notebook 01 (`data/processed/lines.parquet`): one row per
order line with `customer_id`, `invoice`, `stock_code`, `quantity`, `price`,
`invoice_date`, `revenue`, `line_id` and `kind` (sale | cancellation | correction).

Cancelled product sales are removed in matched pairs. Corrections (manual entries,
discounts) cannot be tied reliably to an order (many are reverse-and-rebill entries),
so they only change net revenue.

`split_purchases(..., as_of=cutoff)` only uses lines up to the cutoff, so a
cancellation made after it cannot erase an earlier purchase (no look-ahead in the
churn features of 04).
"""

from datetime import date, datetime, time
from typing import NamedTuple

import polars as pl

PRODUCT_CODE = r"^\d{5}[A-Za-z]*$"
CORRECTION_CODES = ["M", "D", "ADJUST", "ADJUST2"]
# The two yearly sheets both contain 1-9 Dec 2010 (same invoices): keep it from the 2010-2011 sheet only.
OVERLAP_START = datetime(2010, 12, 1)
FIRST_SHEET = "Year 2009-2010"


class Purchases(NamedTuple):
    transactions: pl.DataFrame
    adjustments: pl.DataFrame
    cancelled_lines: pl.DataFrame


def match_cancelled_lines(lines: pl.DataFrame) -> pl.DataFrame:
    """Pair each product cancellation with an earlier sale line of the same customer,
    product, quantity and price, one to one.

    Each cancellation takes the latest earlier unmatched sale. When several
    cancellations compete for the same sale, the earliest cancellation gets it and
    the others fall back to the next earlier sale, if there is one.

    Returns one row per pair: `cancel_line`, `sale_line`, `cancel_date`, `sale_date`.
    """
    keys = ["customer_id", "stock_code", "units", "price"]
    cancels = (
        lines.filter(pl.col("kind") == "cancellation")
        .select("customer_id", "stock_code", "price", "invoice_date",
                units=-pl.col("quantity"), cancel_line=pl.col("line_id"))
        .sort("invoice_date", "cancel_line")
    )
    sales = (
        lines.filter(pl.col("kind") == "sale")
        .select("customer_id", "stock_code", "price", units=pl.col("quantity"),
                sale_date=pl.col("invoice_date"), sale_line=pl.col("line_id"))
        .sort("sale_date", "sale_line")
    )
    rounds = []
    while True:
        pairs = (
            cancels.join_asof(
                sales,
                left_on="invoice_date",
                right_on="sale_date",
                by=keys,
                strategy="backward",
                check_sortedness=False,  # both sides are sorted by date, so every `by` group is too
            )
            .drop_nulls("sale_line")
            .unique("sale_line", keep="first", maintain_order=True)
        )
        rounds.append(pairs)
        if pairs.is_empty():
            break
        cancels = cancels.join(pairs.select("cancel_line"), on="cancel_line", how="anti")
        sales = sales.join(pairs.select("sale_line"), on="sale_line", how="anti")
    return pl.concat(rounds).select(
        "cancel_line", "sale_line", cancel_date=pl.col("invoice_date"), sale_date="sale_date"
    )


def split_purchases(lines: pl.DataFrame, as_of: date | datetime | None = None) -> Purchases:
    """Remove cancelled sales and return purchases (`transactions`) and the remaining
    `adjustments` (unmatched cancellations and all corrections).

    `as_of` is inclusive. A `date` covers that whole day, and a `datetime` is used
    as is.
    """
    if as_of is not None:
        if not isinstance(as_of, datetime):
            as_of = datetime.combine(as_of, time.max)
        lines = lines.filter(pl.col("invoice_date") <= as_of)

    cancelled = match_cancelled_lines(lines)
    remaining = lines.filter(
        ~pl.col("line_id").is_in(cancelled["sale_line"].implode()),
        ~pl.col("line_id").is_in(cancelled["cancel_line"].implode()),
    )
    return Purchases(
        transactions=remaining.filter(pl.col("kind") == "sale"),
        adjustments=remaining.filter(pl.col("kind") != "sale"),
        cancelled_lines=cancelled,
    )


# ---- Raw rows -> clean order lines (notebook 01) ----
# Each step takes and returns the raw frame (`sheet`, `invoice`, `stock_code`, `quantity`,
# `invoice_date`, `price`, `customer_id`, `country`, ...), so the notebook can measure its effect.


def drop_sheet_overlap(df: pl.DataFrame) -> pl.DataFrame:
    """Keep 1-9 Dec 2010 only from the 2010-2011 sheet."""
    return df.filter(~((pl.col("sheet") == FIRST_SHEET) & (pl.col("invoice_date") >= OVERLAP_START)))


def drop_duplicate_lines(df: pl.DataFrame) -> pl.DataFrame:
    return df.unique(maintain_order=True)


def drop_missing_customer(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("customer_id").is_not_null())


def drop_charges_and_tests(df: pl.DataFrame) -> pl.DataFrame:
    """Keep products and the codes that correct what the customer paid for goods."""
    return df.filter(pl.col("stock_code").str.contains(PRODUCT_CODE) | pl.col("stock_code").is_in(CORRECTION_CODES))


def keep_valid_lines(df: pl.DataFrame) -> pl.DataFrame:
    """Product sales with positive quantity and price, product cancellations with negative
    quantity, and corrections with a non-zero value."""
    is_product = pl.col("stock_code").str.contains(PRODUCT_CODE)
    is_cancellation = pl.col("invoice").str.starts_with("C")
    valid_sale = is_product & ~is_cancellation & (pl.col("quantity") > 0) & (pl.col("price") > 0)
    valid_cancellation = is_product & is_cancellation & (pl.col("quantity") < 0)
    valid_correction = ~is_product & (pl.col("quantity") * pl.col("price") != 0)
    return df.filter(valid_sale | valid_cancellation | valid_correction)


STEPS = [drop_sheet_overlap, drop_duplicate_lines, drop_missing_customer, drop_charges_and_tests, keep_valid_lines]


def build_lines(df: pl.DataFrame) -> pl.DataFrame:
    """Add `revenue`, `kind` (sale | cancellation | correction) and `line_id` to the filtered
    rows. A non-product code is a correction; an invoice starting with `C` is a cancellation."""
    is_product = pl.col("stock_code").str.contains(PRODUCT_CODE)
    return df.drop("sheet").with_columns(
        revenue=pl.col("quantity") * pl.col("price"),
        kind=pl.when(~is_product)
        .then(pl.lit("correction"))
        .when(pl.col("invoice").str.starts_with("C"))
        .then(pl.lit("cancellation"))
        .otherwise(pl.lit("sale")),
        line_id=pl.int_range(pl.len()),
    )


def build_customers(lines: pl.DataFrame, transactions: pl.DataFrame, adjustments: pl.DataFrame) -> pl.DataFrame:
    """One row per customer: product `sales`, `adjustments`, `net_revenue`, main `country` (the
    most frequent one, ties broken alphabetically) and `included` (product purchases and
    positive net revenue)."""
    customer_country = (
        lines.group_by("customer_id", "country").len()
        .sort(["customer_id", "len", "country"], descending=[False, True, False])
        .unique("customer_id", keep="first", maintain_order=True)
        .select("customer_id", "country")
    )
    return (
        lines.select("customer_id").unique()
        .join(transactions.group_by("customer_id").agg(sales=pl.col("revenue").sum()), on="customer_id", how="left")
        .join(adjustments.group_by("customer_id").agg(adjustments=pl.col("revenue").sum()), on="customer_id", how="left")
        .with_columns(pl.col("sales", "adjustments").fill_null(0))
        .with_columns(net_revenue=pl.col("sales") + pl.col("adjustments"))
        .join(customer_country, on="customer_id", how="left")
        .with_columns(included=(pl.col("sales") > 0) & (pl.col("net_revenue") > 0))
        .sort("customer_id")
    )
