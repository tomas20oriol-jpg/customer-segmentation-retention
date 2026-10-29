"""Churn dataset at a cutoff date, without look-ahead.

Population: customers with at least one purchase in the `lookback_days` up to and
including the cutoff. Label: `churned` = 1 if they make no purchase in the
`horizon_days` after the cutoff.

Features come only from `split_purchases(lines, as_of=cutoff)`, so a cancellation made
after the cutoff cannot erase a purchase made before it. The label uses
`split_purchases(lines, as_of=cutoff + horizon)`, so an order placed and cancelled
inside the horizon does not count as a purchase.
"""

from datetime import date, timedelta

import polars as pl

from src.cleaning import split_purchases

LOOKBACK_DAYS = 180
HORIZON_DAYS = 180
PEAK_MONTHS = (9, 10, 11)

# `sales` is not a model feature: it is almost orders x average order value on the log
# scale, and the collinearity makes the coefficients unreadable. It is kept as a column
# to value the revenue at risk.
FEATURES = [
    "recency_days",
    "orders",
    "avg_order_value",
    "products",
    "active_months",
    "cancelled_share",
    "is_uk",
    "peak_days_ahead",
]


def peak_days_ahead(cutoff: date, horizon_days: int = HORIZON_DAYS) -> int:
    """Days of the autumn peak (Sep-Nov) inside the label window. It is known at the
    cutoff (it is the calendar), and it lets the model shift its baseline by season."""
    return sum((cutoff + timedelta(days=d)).month in PEAK_MONTHS for d in range(1, horizon_days + 1))


def churn_features(lines: pl.DataFrame, cutoff: date, lookback_days: int = LOOKBACK_DAYS,
                   horizon_days: int = HORIZON_DAYS) -> pl.DataFrame:
    """One row per customer with a purchase in the lookback window, using only what was
    known at the cutoff. `horizon_days` only sets `peak_days_ahead` (a calendar fact)."""
    window_start = cutoff - timedelta(days=lookback_days - 1)  # lookback = [window_start, cutoff]
    past = split_purchases(lines, as_of=cutoff)
    in_window = pl.col("invoice_date").dt.date() >= window_start

    sales = (
        past.transactions.filter(in_window)
        .group_by("customer_id")
        .agg(
            last_purchase=pl.col("invoice_date").max().dt.date(),
            orders=pl.col("invoice").n_unique(),
            sales=pl.col("revenue").sum(),
            products=pl.col("stock_code").n_unique(),
            active_months=pl.col("invoice_date").dt.truncate("1mo").n_unique(),
            country=pl.col("country").mode().sort().first(),
        )
    )
    cancelled = (
        past.adjustments.filter(in_window, pl.col("kind") == "cancellation")
        .group_by("customer_id")
        .agg(cancelled=-pl.col("revenue").sum())
    )
    return (
        sales.join(cancelled, on="customer_id", how="left")
        .with_columns(
            cutoff=pl.lit(cutoff),
            recency_days=(pl.lit(cutoff) - pl.col("last_purchase")).dt.total_days(),
            avg_order_value=pl.col("sales") / pl.col("orders"),
            cancelled_share=(pl.col("cancelled").fill_null(0) / pl.col("sales")).clip(0, 1),
            is_uk=(pl.col("country") == "United Kingdom").cast(pl.Int8),
            peak_days_ahead=pl.lit(peak_days_ahead(cutoff, horizon_days)),
        )
        .select("customer_id", "cutoff", *FEATURES, "sales")
        .sort("customer_id")
    )


def churn_labels(lines: pl.DataFrame, cutoff: date, horizon_days: int = HORIZON_DAYS) -> pl.DataFrame:
    """Customers who make at least one purchase in (cutoff, cutoff + horizon]. Cancellations
    made inside the horizon are applied, so a cancelled order does not count."""
    future = split_purchases(lines, as_of=cutoff + timedelta(days=horizon_days)).transactions
    return future.filter(pl.col("invoice_date").dt.date() > cutoff).select("customer_id").unique()


def churn_dataset(
    lines: pl.DataFrame, cutoff: date, lookback_days: int = LOOKBACK_DAYS, horizon_days: int = HORIZON_DAYS
) -> pl.DataFrame:
    """Features at the cutoff plus `churned` (1 = no purchase in the horizon)."""
    returning = churn_labels(lines, cutoff, horizon_days)
    return churn_features(lines, cutoff, lookback_days, horizon_days).with_columns(
        churned=(~pl.col("customer_id").is_in(returning["customer_id"].implode())).cast(pl.Int8)
    )
