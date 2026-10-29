from datetime import date, datetime

import polars as pl
import pytest

from src.features import rfm_scores, rfm_table

SNAPSHOT = date(2011, 12, 10)

TX_SCHEMA = {"invoice": pl.String, "customer_id": pl.String, "invoice_date": pl.Datetime, "revenue": pl.Float64}
ADJ_SCHEMA = {"customer_id": pl.String, "invoice_date": pl.Datetime, "revenue": pl.Float64}


def tx(rows):
    """rows: (customer_id, invoice, datetime, revenue)"""
    return pl.DataFrame(
        {
            "invoice": [r[1] for r in rows],
            "customer_id": [r[0] for r in rows],
            "invoice_date": [r[2] for r in rows],
            "revenue": [float(r[3]) for r in rows],
        },
        schema=TX_SCHEMA,
    )


def adj(rows):
    """rows: (customer_id, datetime, revenue)"""
    return pl.DataFrame(
        {
            "customer_id": [r[0] for r in rows],
            "invoice_date": [r[1] for r in rows],
            "revenue": [float(r[2]) for r in rows],
        },
        schema=ADJ_SCHEMA,
    )


NO_ADJ = adj([])


def row(table, customer):
    return table.filter(pl.col("customer_id") == customer).row(0, named=True)


# ---------------- rfm_table ----------------
def test_recency_is_whole_days_from_snapshot_to_last_purchase_day():
    t = tx([
        ("A", "1", datetime(2011, 11, 30, 23, 59), 10),
        ("A", "2", datetime(2011, 12, 1, 0, 1), 10),
    ])
    r = row(rfm_table(t, NO_ADJ, SNAPSHOT), "A")
    assert r["recency"] == 9  # 1 Dec -> 10 Dec; time of day is irrelevant


def test_recency_is_one_for_purchase_on_the_day_before_snapshot():
    t = tx([("A", "1", datetime(2011, 12, 9, 23, 30), 10)])
    assert row(rfm_table(t, NO_ADJ, SNAPSHOT), "A")["recency"] == 1


def test_single_purchase_customer_has_frequency_one_and_aov_equal_to_monetary():
    t = tx([("A", "1", datetime(2011, 6, 1, 10), 50)])
    r = row(rfm_table(t, NO_ADJ, SNAPSHOT), "A")
    assert r["frequency"] == 1
    assert r["monetary"] == pytest.approx(50)
    assert r["avg_order_value"] == pytest.approx(50)


def test_frequency_counts_distinct_invoices_not_lines():
    t = tx([
        ("A", "1", datetime(2011, 6, 1, 10), 10),
        ("A", "1", datetime(2011, 6, 1, 10), 20),
        ("A", "1", datetime(2011, 6, 1, 10), 30),
    ])
    r = row(rfm_table(t, NO_ADJ, SNAPSHOT), "A")
    assert r["frequency"] == 1
    assert r["monetary"] == pytest.approx(60)


def test_several_invoices_on_the_same_day_count_separately():
    t = tx([
        ("A", "1", datetime(2011, 6, 1, 9), 10),
        ("A", "2", datetime(2011, 6, 1, 15), 10),
        ("A", "3", datetime(2011, 6, 1, 18), 10),
    ])
    r = row(rfm_table(t, NO_ADJ, SNAPSHOT), "A")
    assert r["frequency"] == 3
    assert r["avg_order_value"] == pytest.approx(10)


def test_monetary_is_net_of_adjustments_and_aov_uses_net():
    t = tx([
        ("A", "1", datetime(2011, 6, 1), 100),
        ("A", "2", datetime(2011, 7, 1), 100),
    ])
    a = adj([("A", datetime(2011, 7, 5), -40), ("A", datetime(2011, 7, 6), 10)])
    r = row(rfm_table(t, a, SNAPSHOT), "A")
    assert r["monetary"] == pytest.approx(170)
    assert r["avg_order_value"] == pytest.approx(85)


def test_adjustments_do_not_change_frequency_or_recency():
    t = tx([("A", "1", datetime(2011, 6, 1), 100)])
    a = adj([("A", datetime(2011, 12, 1), -10)])
    r = row(rfm_table(t, a, SNAPSHOT), "A")
    assert (r["frequency"], r["recency"]) == (1, (SNAPSHOT - date(2011, 6, 1)).days)


def test_customer_with_net_revenue_zero_or_negative_is_dropped():
    t = tx([
        ("ZERO", "1", datetime(2011, 6, 1), 100),
        ("NEG", "2", datetime(2011, 6, 1), 100),
        ("OK", "3", datetime(2011, 6, 1), 100),
    ])
    a = adj([("ZERO", datetime(2011, 6, 2), -100), ("NEG", datetime(2011, 6, 2), -150)])
    out = rfm_table(t, a, SNAPSHOT)
    assert out["customer_id"].to_list() == ["OK"]


def test_customer_with_only_adjustments_is_not_in_table():
    t = tx([("A", "1", datetime(2011, 6, 1), 100)])
    a = adj([("B", datetime(2011, 6, 2), 50)])
    assert rfm_table(t, a, SNAPSHOT)["customer_id"].to_list() == ["A"]


def test_transaction_on_snapshot_day_is_ignored():
    t = tx([
        ("A", "1", datetime(2011, 12, 1), 100),
        ("A", "2", datetime(2011, 12, 10, 0, 0), 900),
        ("A", "3", datetime(2011, 12, 10, 15, 0), 900),
    ])
    r = row(rfm_table(t, NO_ADJ, SNAPSHOT), "A")
    assert (r["recency"], r["frequency"], r["monetary"]) == (9, 1, pytest.approx(100))


def test_customer_whose_only_purchases_are_on_or_after_snapshot_is_absent():
    t = tx([
        ("FUTURE", "1", datetime(2011, 12, 10), 100),
        ("LATER", "2", datetime(2011, 12, 20), 100),
        ("OK", "3", datetime(2011, 12, 9), 100),
    ])
    assert rfm_table(t, NO_ADJ, SNAPSHOT)["customer_id"].to_list() == ["OK"]


def test_adjustments_on_or_after_snapshot_are_ignored():
    t = tx([("A", "1", datetime(2011, 6, 1), 100)])
    a = adj([
        ("A", datetime(2011, 12, 9, 23, 59), -30),   # counts
        ("A", datetime(2011, 12, 10, 0, 0), -500),   # on snapshot: ignored
        ("A", datetime(2012, 1, 1), -500),           # future: ignored
    ])
    assert row(rfm_table(t, a, SNAPSHOT), "A")["monetary"] == pytest.approx(70)


def test_future_adjustment_cannot_drop_a_customer():
    t = tx([("A", "1", datetime(2011, 6, 1), 100)])
    a = adj([("A", datetime(2011, 12, 20), -1000)])
    assert rfm_table(t, a, SNAPSHOT)["customer_id"].to_list() == ["A"]


def test_empty_transactions_give_empty_table():
    out = rfm_table(tx([]), NO_ADJ, SNAPSHOT)
    assert out.height == 0
    assert {"customer_id", "recency", "frequency", "monetary", "avg_order_value"} <= set(out.columns)


def test_one_row_per_customer_sorted_by_id():
    t = tx([
        ("B", "1", datetime(2011, 6, 1), 10),
        ("A", "2", datetime(2011, 6, 1), 10),
        ("A", "3", datetime(2011, 7, 1), 10),
    ])
    out = rfm_table(t, NO_ADJ, SNAPSHOT)
    assert out["customer_id"].to_list() == ["A", "B"]


# ---------------- rfm_scores ----------------
def rfm(recency, frequency, monetary):
    n = len(recency)
    return pl.DataFrame({
        "customer_id": [f"c{i}" for i in range(n)],
        "recency": recency,
        "frequency": frequency,
        "monetary": [float(m) for m in monetary],
        "avg_order_value": [1.0] * n,
    })


def test_scores_range_1_to_5_with_five_distinct_customers():
    out = rfm_scores(rfm([50, 40, 30, 20, 10], [1, 2, 3, 4, 5], [10, 20, 30, 40, 50]))
    assert out["f_score"].to_list() == [1, 2, 3, 4, 5]
    assert out["m_score"].to_list() == [1, 2, 3, 4, 5]


def test_exact_quintile_boundary_is_not_pushed_to_next_score():
    # 5 distinct values: percentile ranks 0.2..1.0 map exactly to scores 1..5
    out = rfm_scores(rfm([5, 4, 3, 2, 1], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]))
    assert out["f_score"].to_list() == [1, 2, 3, 4, 5]


def test_scores_stay_within_1_and_5_for_many_customers():
    n = 137
    out = rfm_scores(rfm(list(range(1, n + 1)), list(range(1, n + 1)), list(range(1, n + 1))))
    for c in ("r_score", "f_score", "m_score", "fm_score"):
        assert out[c].min() >= 1 and out[c].max() <= 5
    assert set(out["f_score"].unique()) == {1, 2, 3, 4, 5}


def test_low_recency_gets_high_r_score():
    out = rfm_scores(rfm([50, 40, 30, 20, 10], [1] * 5, [10] * 5))
    assert out["r_score"].to_list() == [1, 2, 3, 4, 5]


def test_high_frequency_and_monetary_get_high_scores():
    out = rfm_scores(rfm([10, 10, 10, 10, 10], [5, 4, 3, 2, 1], [1, 2, 3, 4, 5]))
    assert out["f_score"].to_list() == [5, 4, 3, 2, 1]
    assert out["m_score"].to_list() == [1, 2, 3, 4, 5]


def test_customers_with_the_same_frequency_share_a_score():
    # six one-order customers, four more frequent ones
    out = rfm_scores(rfm(list(range(1, 11)), [1] * 6 + [2, 3, 4, 5], list(range(10, 110, 10))))
    ones = out.filter(pl.col("frequency") == 1)["f_score"]
    assert ones.n_unique() == 1
    # and they score below every more frequent customer
    assert ones[0] < out.filter(pl.col("frequency") > 1)["f_score"].min()


def test_all_equal_values_share_one_score():
    out = rfm_scores(rfm([7] * 8, [1] * 8, [100] * 8))
    for c in ("r_score", "f_score", "m_score"):
        assert out[c].n_unique() == 1


def test_scores_are_int8():
    out = rfm_scores(rfm([1, 2], [1, 2], [1, 2]))
    assert all(out.schema[c] == pl.Int8 for c in ("r_score", "f_score", "m_score", "fm_score"))


def test_input_columns_and_row_count_are_preserved():
    data = rfm([50, 40, 30], [1, 2, 3], [10, 20, 30])
    out = rfm_scores(data)
    assert out.height == 3
    assert out.select(data.columns).equals(data)


def test_fm_score_is_mean_of_f_and_m_with_halves_rounded_up():
    # f_score = 1..5, m_score = [2, 1, 3, 5, 4] -> sums 3, 3, 6, 9, 9
    out = rfm_scores(rfm([10] * 5, [1, 2, 3, 4, 5], [20, 10, 30, 50, 40]))
    assert out["f_score"].to_list() == [1, 2, 3, 4, 5]
    assert out["m_score"].to_list() == [2, 1, 3, 5, 4]
    assert out["fm_score"].to_list() == [2, 2, 3, 5, 5]  # 1.5 -> 2, 4.5 -> 5


def test_fm_score_rounds_halves_up_at_each_boundary():
    # sums: (1,4)=5 -> 2.5 -> 3 ; (2,3)=5 -> 3 ; (4,5)=9 -> 5 ; (3,2)... all distinct per column
    out = rfm_scores(rfm([10] * 5, [1, 2, 3, 4, 5], [4, 3, 1, 2, 5]))
    assert out["f_score"].to_list() == [1, 2, 3, 4, 5]
    assert out["m_score"].to_list() == [4, 3, 1, 2, 5]
    assert out["fm_score"].to_list() == [3, 3, 2, 3, 5]  # 2.5->3, 2.5->3, 2.0, 3.0, 5.0


def test_segments_champions_promising_needs_attention_at_risk_lost():
    # r_score 1..5 (recency 50..10); f and m scores equal, giving fm = [1, 4, 3, 2, 5]
    out = rfm_scores(rfm([50, 40, 30, 20, 10], [1, 4, 3, 2, 5], [1, 4, 3, 2, 5]))
    assert out["r_score"].to_list() == [1, 2, 3, 4, 5]
    assert out["fm_score"].to_list() == [1, 4, 3, 2, 5]
    assert out["rfm_segment"].to_list() == ["Lost", "At risk", "Needs attention", "Promising", "Champions"]


def test_segment_boundaries_r4_fm4_champions_r2_fm2_lost():
    # r = 1..5, fm = [5, 2, 1, 4, 3]
    out = rfm_scores(rfm([50, 40, 30, 20, 10], [5, 2, 1, 4, 3], [5, 2, 1, 4, 3]))
    assert out["fm_score"].to_list() == [5, 2, 1, 4, 3]
    assert out["rfm_segment"].to_list() == ["At risk", "Lost", "Needs attention", "Champions", "Promising"]


def test_segment_boundary_fm3_is_at_risk_for_low_recency_and_promising_for_high_recency():
    # r = 1..5; fm = [1, 3, 3, 3, 5] (r2: f4/m2, r4: f2/m4 -> mean 3)
    out = rfm_scores(rfm([50, 40, 30, 20, 10], [1, 4, 3, 2, 5], [1, 2, 3, 4, 5]))
    assert out["fm_score"].to_list() == [1, 3, 3, 3, 5]
    assert out["rfm_segment"].to_list() == ["Lost", "At risk", "Needs attention", "Promising", "Champions"]


def test_needs_attention_ignores_fm_score():
    out = rfm_scores(rfm([50, 40, 30, 20, 10], [1, 2, 5, 4, 3], [1, 2, 5, 4, 3]))
    assert out["r_score"][2] == 3 and out["fm_score"][2] == 5
    assert out["rfm_segment"][2] == "Needs attention"


def test_every_customer_gets_one_of_the_five_segments():
    n = 60
    out = rfm_scores(rfm(list(range(1, n + 1)), [(i * 7) % 11 + 1 for i in range(n)], [(i * 13) % 17 + 1 for i in range(n)]))
    assert out["rfm_segment"].null_count() == 0
    assert set(out["rfm_segment"].unique()) <= {"Champions", "Promising", "Needs attention", "At risk", "Lost"}


def test_rfm_table_feeds_rfm_scores_end_to_end():
    t = tx([
        ("A", "1", datetime(2011, 12, 9), 500),
        ("A", "2", datetime(2011, 12, 8), 500),
        ("B", "3", datetime(2011, 1, 5), 20),
        ("C", "4", datetime(2011, 6, 5), 100),
    ])
    out = rfm_scores(rfm_table(t, NO_ADJ, SNAPSHOT))
    a, b = row(out, "A"), row(out, "B")
    assert a["r_score"] > b["r_score"] and a["m_score"] > b["m_score"]
    assert a["rfm_segment"] == "Champions"
    assert b["r_score"] < a["r_score"]
    assert b["rfm_segment"] in {"At risk", "Lost"}
