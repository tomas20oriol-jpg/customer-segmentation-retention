"""Customer-level RFM features.

Recency is measured in whole days from the snapshot date to the customer's last
purchase day. The snapshot is usually the day after the last transaction in the data,
or the cutoff date in the churn model of 04. Frequency counts distinct invoices, and
Monetary is net revenue (sales plus adjustments).
"""

from datetime import date

import polars as pl

N_SCORES = 5


def rfm_table(transactions: pl.DataFrame, adjustments: pl.DataFrame, snapshot: date) -> pl.DataFrame:
    """One row per customer with purchases before `snapshot`: recency (days),
    frequency (invoices), monetary (net revenue, £) and average order value.

    `snapshot` is exclusive: for a churn cutoff, pass `cutoff + 1 day`. The inputs must
    match the snapshot. `transactions.parquet` uses cancellations from the whole period,
    so at an earlier cutoff, build the inputs with `split_purchases(lines, as_of=cutoff)`.
    The date filter here does not undo cancellations made after the cutoff.

    Customers with net revenue <= 0 are dropped, because RFM needs a positive spend.
    """
    transactions = transactions.filter(pl.col("invoice_date").dt.date() < snapshot)
    adjustments = adjustments.filter(pl.col("invoice_date").dt.date() < snapshot)
    return (
        transactions.group_by("customer_id")
        .agg(
            last_purchase=pl.col("invoice_date").max().dt.date(),
            frequency=pl.col("invoice").n_unique(),
            sales=pl.col("revenue").sum(),
        )
        .join(adjustments.group_by("customer_id").agg(adjustments=pl.col("revenue").sum()),
              on="customer_id", how="left")
        .with_columns(
            recency=(pl.lit(snapshot) - pl.col("last_purchase")).dt.total_days(),
            monetary=pl.col("sales") + pl.col("adjustments").fill_null(0),
        )
        .filter(pl.col("monetary") > 0)
        .with_columns(avg_order_value=pl.col("monetary") / pl.col("frequency"))
        .select("customer_id", "recency", "frequency", "monetary", "avg_order_value")
        .sort("customer_id")
    )


def _score(column: str, higher_is_better: bool = True) -> pl.Expr:
    """1-5 score from the percentile rank. Ties share a score, so the many customers
    with one order all get the same frequency score, instead of being split at random."""
    rank = pl.col(column).rank(method="average", descending=not higher_is_better)
    # Multiply before dividing: rank / n * 5 can land just above an integer (3.0000000000000004)
    # and ceil() would push a customer on a quintile boundary up one score.
    return (rank * N_SCORES / pl.len()).ceil().clip(1, N_SCORES).cast(pl.Int8)


def rfm_scores(rfm: pl.DataFrame) -> pl.DataFrame:
    """Add R, F and M scores (5 = best) and a rule-based segment.

    The segment uses recency and the combined frequency-monetary score (FM):
    - Champions: R 4-5 and FM 4-5
    - Promising: R 4-5 and FM 1-3
    - Needs attention: R 3
    - At risk: R 1-2 and FM 3-5
    - Lost: R 1-2 and FM 1-2
    """
    return rfm.with_columns(
        r_score=_score("recency", higher_is_better=False),
        f_score=_score("frequency"),
        m_score=_score("monetary"),
    ).with_columns(
        fm_score=((pl.col("f_score") + pl.col("m_score") + 1) // 2).cast(pl.Int8),  # mean, halves round up
    ).with_columns(
        rfm_segment=pl.when(pl.col("r_score") >= 4)
        .then(pl.when(pl.col("fm_score") >= 4).then(pl.lit("Champions")).otherwise(pl.lit("Promising")))
        .when(pl.col("r_score") == 3)
        .then(pl.lit("Needs attention"))
        .otherwise(pl.when(pl.col("fm_score") >= 3).then(pl.lit("At risk")).otherwise(pl.lit("Lost")))
    )
