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
        .sort("invoice_date")
    )
    sales = (
        lines.filter(pl.col("kind") == "sale")
        .select("customer_id", "stock_code", "price", units=pl.col("quantity"),
                sale_date=pl.col("invoice_date"), sale_line=pl.col("line_id"))
        .sort("sale_date")
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
