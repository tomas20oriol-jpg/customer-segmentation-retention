from datetime import date, datetime

import polars as pl
import pytest

from src.retention import cohort_activity, purchase_intervals, retention_table


def tx(rows):
    return pl.DataFrame(
        {"customer_id": [r[0] for r in rows], "invoice_date": [r[1] for r in rows]},
        schema={"customer_id": pl.Int64, "invoice_date": pl.Datetime},
    )


# ---------- cohort_activity ----------

def test_cohort_is_first_active_month():
    t = tx([(1, datetime(2010, 3, 15)), (1, datetime(2010, 5, 2))])
    out = cohort_activity(t, date(2010, 12, 1))
    assert out["cohort"].unique().to_list() == [date(2010, 3, 1)]
    assert out["month"].to_list() == [date(2010, 3, 1), date(2010, 5, 1)]
    assert out["months_since_first"].to_list() == [0, 2]


def test_months_since_first_across_year_boundary():
    t = tx([(1, datetime(2010, 11, 20)), (1, datetime(2011, 2, 1))])
    out = cohort_activity(t, date(2011, 12, 1))
    assert out["months_since_first"].to_list() == [0, 3]


def test_several_purchases_in_a_month_count_once():
    t = tx([(1, datetime(2010, 3, 1)), (1, datetime(2010, 3, 2)), (1, datetime(2010, 3, 30, 23))])
    out = cohort_activity(t, date(2010, 12, 1))
    assert out.height == 1


def test_months_after_last_month_excluded_and_last_month_inclusive():
    t = tx([(1, datetime(2010, 1, 5)), (1, datetime(2010, 4, 5)), (1, datetime(2010, 5, 5))])
    out = cohort_activity(t, date(2010, 4, 1))
    assert out["month"].to_list() == [date(2010, 1, 1), date(2010, 4, 1)]


def test_cohort_activity_empty_input():
    out = cohort_activity(tx([]), date(2010, 12, 1))
    assert out.height == 0
    assert {"cohort", "month", "months_since_first"} <= set(out.columns)


# ---------- retention_table ----------

def test_month_zero_is_100_percent_and_cohort_size_from_month_zero():
    t = tx([
        (1, datetime(2010, 1, 10)), (2, datetime(2010, 1, 11)), (3, datetime(2010, 1, 12)),
        (1, datetime(2010, 2, 10)),
    ])
    r = retention_table(cohort_activity(t, date(2010, 3, 1)))
    m0 = r.filter(pl.col("months_since_first") == 0).row(0, named=True)
    assert m0["retention"] == 1.0 and m0["cohort_size"] == 3
    m1 = r.filter(pl.col("months_since_first") == 1).row(0, named=True)
    assert m1["active"] == 1 and m1["retention"] == pytest.approx(1 / 3)
    assert m1["cohort_size"] == 3


def test_unobserved_months_are_absent_not_zero():
    t = tx([(1, datetime(2010, 1, 10)), (2, datetime(2010, 3, 10))])
    r = retention_table(cohort_activity(t, date(2010, 3, 1)))
    jan = r.filter(pl.col("cohort") == date(2010, 1, 1))
    mar = r.filter(pl.col("cohort") == date(2010, 3, 1))
    # Jan cohort: nobody is active in months 1 or 2 -> those rows are absent, not 0
    assert jan["months_since_first"].to_list() == [0]
    assert mar["months_since_first"].to_list() == [0]
    assert (r["retention"] > 0).all()


def test_retention_table_empty_input():
    r = retention_table(cohort_activity(tx([]), date(2010, 12, 1)))
    assert r.height == 0


# ---------- purchase_intervals ----------

END = date(2010, 12, 31)


def test_one_interval_per_consecutive_pair_plus_censored_last():
    t = tx([(1, datetime(2010, 1, 1)), (1, datetime(2010, 1, 11)), (1, datetime(2010, 2, 10))])
    out = purchase_intervals(t, END).sort("interval_number")
    assert out["duration"].to_list() == [10, 30, (END - date(2010, 2, 10)).days]
    assert out["event"].to_list() == [1, 1, 0]
    assert out["start"].to_list() == [date(2010, 1, 1), date(2010, 1, 11), date(2010, 2, 10)]


def test_same_day_invoices_collapse():
    t = tx([
        (1, datetime(2010, 1, 1, 9)), (1, datetime(2010, 1, 1, 17)),
        (1, datetime(2010, 1, 6, 10)),
    ])
    out = purchase_intervals(t, END).sort("interval_number")
    assert out["duration"].to_list()[0] == 5
    assert out.height == 2


def test_one_time_buyer_gives_single_censored_interval():
    t = tx([(1, datetime(2010, 12, 1))])
    out = purchase_intervals(t, END)
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["duration"] == 30 and row["event"] == 0
    assert row["interval_number"] == 1 and row["weight"] == 1.0


def test_purchase_on_data_end_gives_no_zero_length_interval():
    t = tx([(1, datetime(2010, 12, 1)), (1, datetime(2010, 12, 31, 15))])
    out = purchase_intervals(t, END)
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["duration"] == 30 and row["event"] == 1
    assert (out["duration"] > 0).all()


def test_interval_number_starts_at_one_per_customer():
    t = tx([
        (1, datetime(2010, 1, 1)), (1, datetime(2010, 2, 1)),
        (2, datetime(2010, 3, 1)), (2, datetime(2010, 4, 1)), (2, datetime(2010, 5, 1)),
    ])
    out = purchase_intervals(t, END)
    for cid, n in [(1, 2), (2, 3)]:
        nums = out.filter(pl.col("customer_id") == cid).sort("interval_number")["interval_number"].to_list()
        assert nums == list(range(1, n + 1))


def test_weights_sum_to_one_per_customer():
    t = tx([
        (1, datetime(2010, 1, 1)), (1, datetime(2010, 2, 1)), (1, datetime(2010, 3, 1)),
        (2, datetime(2010, 6, 1)),
    ])
    out = purchase_intervals(t, END)
    sums = out.group_by("customer_id").agg(w=pl.col("weight").sum())
    assert sums["w"].to_list() == pytest.approx([1.0] * 2)


def test_purchases_after_data_end_ignored():
    t = tx([(1, datetime(2010, 1, 1)), (1, datetime(2011, 3, 1))])
    out = purchase_intervals(t, END)
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["event"] == 0 and row["duration"] == (END - date(2010, 1, 1)).days


def test_customer_with_only_post_end_purchases_absent():
    t = tx([(1, datetime(2011, 3, 1))])
    assert purchase_intervals(t, END).height == 0


def test_purchase_intervals_empty_input():
    assert purchase_intervals(tx([]), END).height == 0


def test_only_purchase_on_data_end_gives_no_rows():
    t = tx([(1, datetime(2010, 12, 31))])
    assert purchase_intervals(t, END).height == 0
