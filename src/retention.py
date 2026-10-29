"""Cohort activity and purchase intervals for retention analysis.

A customer's cohort is the calendar month of their first purchase in the data. A
customer is active in a month if they bought at least once in it. Intervals between
purchase days feed survival analysis: an interval that is still open when the data
ends is right-censored (event = 0).
"""

from datetime import date

import polars as pl


def months_between(start: pl.Expr, end: pl.Expr) -> pl.Expr:
    return (end.dt.year() - start.dt.year()) * 12 + end.dt.month() - start.dt.month()


def cohort_activity(transactions: pl.DataFrame, last_month: date) -> pl.DataFrame:
    """One row per customer and active month up to `last_month` (inclusive): `cohort`,
    `month` and `months_since_first` (0 = the acquisition month)."""
    active = (
        transactions.select("customer_id", month=pl.col("invoice_date").dt.truncate("1mo").dt.date())
        .filter(pl.col("month") <= last_month)
        .unique()
    )
    return active.with_columns(cohort=pl.col("month").min().over("customer_id")).with_columns(
        months_since_first=months_between(pl.col("cohort"), pl.col("month"))
    ).sort("customer_id", "month")


def retention_table(activity: pl.DataFrame) -> pl.DataFrame:
    """Share of each cohort active `months_since_first` months after acquisition.
    Only observed combinations appear, so the right-censored corner is simply absent."""
    sizes = activity.filter(pl.col("months_since_first") == 0).group_by("cohort").agg(cohort_size=pl.len())
    return (
        activity.group_by("cohort", "months_since_first").agg(active=pl.len())
        .join(sizes, on="cohort")
        .with_columns(retention=pl.col("active") / pl.col("cohort_size"))
        .sort("cohort", "months_since_first")
    )


def purchase_intervals(transactions: pl.DataFrame, data_end: date) -> pl.DataFrame:
    """Every interval between a customer's consecutive purchase days, plus the open
    interval from their last purchase to `data_end`, which is censored.

    Columns: `customer_id`, `start`, `duration` (days), `event` (1 = next purchase seen,
    0 = censored), `interval_number` (1 = from first to second purchase day) and
    `weight` (1 / intervals of that customer, so each customer weighs 1 in total).
    """
    days = (
        transactions.select("customer_id", start=pl.col("invoice_date").dt.date())
        .filter(pl.col("start") <= data_end)
        .unique()
        .sort("customer_id", "start")
    )
    return (
        days.with_columns(next_day=pl.col("start").shift(-1).over("customer_id"))
        .with_columns(
            event=pl.col("next_day").is_not_null().cast(pl.Int8),
            duration=(pl.coalesce("next_day", pl.lit(data_end)) - pl.col("start")).dt.total_days(),
            interval_number=pl.int_range(1, pl.len() + 1).over("customer_id"),
        )
        .filter(pl.col("duration") > 0)  # a purchase on the last day has no follow-up
        .with_columns(weight=1 / pl.len().over("customer_id"))
        .select("customer_id", "start", "duration", "event", "interval_number", "weight")
    )
