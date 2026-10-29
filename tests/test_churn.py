from datetime import date, datetime, timedelta

import polars as pl

from src.churn import FEATURES, churn_dataset, churn_features, churn_labels, peak_days_ahead

SCHEMA = {
    "invoice": pl.String, "stock_code": pl.String, "quantity": pl.Int64,
    "invoice_date": pl.Datetime("us"), "price": pl.Float64, "customer_id": pl.String,
    "revenue": pl.Float64, "kind": pl.String, "line_id": pl.Int64, "country": pl.String,
}

CUTOFF = date(2010, 6, 30)
H = 180
WINDOW_START = CUTOFF - timedelta(days=179)  # lookback 180 days, inclusive at both ends
HORIZON_END = CUTOFF + timedelta(days=H)


def at(d, hour=12):
    return datetime(d.year, d.month, d.day, hour)


def make_lines(rows):
    """rows: (line_id, kind, customer, invoice, code, qty, price, date, country)"""
    return pl.DataFrame(
        [
            dict(line_id=i, kind=k, customer_id=c, invoice=inv, stock_code=code, quantity=q,
                 price=p, invoice_date=d, revenue=q * p, country=ctry)
            for i, k, c, inv, code, q, p, d, ctry in rows
        ],
        schema=SCHEMA,
    )


def sale(i, d, cust="A", qty=5, price=2.0, code="10001", inv=None, country="United Kingdom"):
    return (i, "sale", cust, inv or f"S{i}", code, qty, price, d, country)


def cancel(i, d, cust="A", qty=-5, price=2.0, code="10001", country="United Kingdom"):
    return (i, "cancellation", cust, f"C{i}", code, qty, price, d, country)


def correction(i, d, cust="A", code="M", country="United Kingdom"):
    return (i, "correction", cust, f"C{i}", code, 1, 3.0, d, country)


def feats(rows, **kw):
    return churn_features(make_lines(rows), CUTOFF, **kw)


def row(df, cust="A"):
    return df.filter(pl.col("customer_id") == cust).row(0, named=True)


# ---- (1) no look-ahead in features ----

def test_purchase_after_cutoff_does_not_change_any_feature():
    base = [
        sale(1, at(date(2010, 3, 1)), qty=5, price=2.0, code="10001"),
        sale(2, at(date(2010, 5, 10)), qty=3, price=4.0, code="10002"),
    ]
    future = [
        sale(3, at(CUTOFF + timedelta(days=1)), qty=100, price=50.0, code="10003", country="France"),
        sale(4, at(CUTOFF + timedelta(days=60)), qty=1, price=1.0, code="10004"),
    ]
    assert feats(base).equals(feats(base + future))


def test_customer_with_only_future_purchases_is_not_in_features():
    out = feats([sale(1, at(date(2010, 3, 1)), cust="A"),
                 sale(2, at(CUTOFF + timedelta(days=5)), cust="B")])
    assert out["customer_id"].to_list() == ["A"]


# ---- (2) cancellations and the cutoff ----

def test_cancellation_after_cutoff_does_not_remove_purchase_from_features():
    rows = [sale(1, at(date(2010, 5, 1))), cancel(2, at(CUTOFF + timedelta(days=10)))]
    out = feats(rows)
    assert row(out)["orders"] == 1
    assert row(out)["sales"] == 10.0
    assert row(out)["cancelled_share"] == 0.0


def test_cancellation_before_cutoff_removes_purchase_from_features():
    rows = [sale(1, at(date(2010, 5, 1))),
            cancel(2, at(date(2010, 5, 2))),
            sale(3, at(date(2010, 6, 1)), qty=1, price=1.0, code="10002")]
    out = feats(rows)
    assert row(out)["orders"] == 1
    assert row(out)["sales"] == 1.0
    assert row(out)["products"] == 1


def test_customer_whose_only_purchase_was_cancelled_before_cutoff_is_excluded():
    out = feats([sale(1, at(date(2010, 5, 1))), cancel(2, at(date(2010, 5, 2)))])
    assert out.is_empty()


# ---- (3) label window ----

def labels(rows, **kw):
    return churn_labels(make_lines(rows), CUTOFF, **kw)["customer_id"].to_list()


def test_label_counts_purchase_inside_horizon():
    assert labels([sale(1, at(CUTOFF + timedelta(days=1)))]) == ["A"]
    assert labels([sale(1, at(HORIZON_END))]) == ["A"]  # last day, inclusive


def test_label_ignores_purchase_on_cutoff_day():
    assert labels([sale(1, at(CUTOFF, hour=23))]) == []


def test_label_ignores_purchase_after_horizon():
    assert labels([sale(1, at(HORIZON_END + timedelta(days=1), hour=0))]) == []


def test_purchase_on_cutoff_day_belongs_to_features_not_label():
    rows = [sale(1, at(CUTOFF, hour=23))]
    out = churn_dataset(make_lines(rows), CUTOFF)
    r = row(out)
    assert r["recency_days"] == 0
    assert r["churned"] == 1


def test_customer_with_purchase_in_horizon_is_not_churned_and_others_are():
    rows = [
        sale(1, at(date(2010, 6, 1)), cust="A"),
        sale(2, at(date(2010, 6, 2)), cust="B"),
        sale(3, at(CUTOFF + timedelta(days=30)), cust="B"),
    ]
    out = churn_dataset(make_lines(rows), CUTOFF)
    assert dict(zip(out["customer_id"], out["churned"])) == {"A": 1, "B": 0}


# ---- (4) placed and cancelled inside the horizon ----

def test_order_placed_and_cancelled_inside_horizon_does_not_count_as_return():
    rows = [sale(1, at(date(2010, 6, 1))),
            sale(2, at(CUTOFF + timedelta(days=10)), qty=7, price=3.0, code="10009"),
            cancel(3, at(CUTOFF + timedelta(days=12)), qty=-7, price=3.0, code="10009")]
    out = churn_dataset(make_lines(rows), CUTOFF)
    assert row(out)["churned"] == 1


def test_cancellation_after_horizon_does_not_undo_a_return():
    rows = [sale(1, at(date(2010, 6, 1))),
            sale(2, at(CUTOFF + timedelta(days=10)), qty=7, price=3.0, code="10009"),
            cancel(3, at(HORIZON_END + timedelta(days=5)), qty=-7, price=3.0, code="10009")]
    out = churn_dataset(make_lines(rows), CUTOFF)
    assert row(out)["churned"] == 0


# ---- (5) population ----

def test_population_lookback_is_inclusive_at_both_ends():
    rows = [
        sale(1, at(WINDOW_START, hour=0), cust="first_day"),
        sale(2, at(WINDOW_START - timedelta(days=1), hour=23), cust="too_old"),
        sale(3, at(CUTOFF, hour=23), cust="last_day"),
    ]
    out = feats(rows)
    assert sorted(out["customer_id"].to_list()) == ["first_day", "last_day"]


def test_purchases_before_the_window_do_not_count_in_features():
    rows = [
        sale(1, at(WINDOW_START - timedelta(days=30)), qty=100, price=10.0, code="10005"),
        sale(2, at(date(2010, 5, 1)), qty=1, price=2.0, code="10001"),
    ]
    r = row(feats(rows))
    assert r["orders"] == 1
    assert r["sales"] == 2.0
    assert r["products"] == 1


def test_empty_input_gives_empty_dataset():
    out = churn_dataset(make_lines([]), CUTOFF)
    assert out.is_empty()
    assert "churned" in out.columns


# ---- (6) feature definitions ----

def test_recency_is_zero_for_purchase_on_cutoff_day_and_counts_days_otherwise():
    rows = [sale(1, at(CUTOFF), cust="A"),
            sale(2, at(CUTOFF - timedelta(days=10)), cust="B")]
    out = feats(rows)
    assert row(out, "A")["recency_days"] == 0
    assert row(out, "B")["recency_days"] == 10


def test_recency_uses_last_purchase_day():
    rows = [sale(1, at(date(2010, 3, 1))), sale(2, at(date(2010, 6, 20)))]
    assert row(feats(rows))["recency_days"] == 10


def test_active_months_counts_distinct_calendar_months_not_days():
    rows = [
        sale(1, at(date(2010, 3, 1))), sale(2, at(date(2010, 3, 31))),
        sale(3, at(date(2010, 4, 1))), sale(4, at(date(2010, 6, 30))),
    ]
    r = row(feats(rows))
    assert r["active_months"] == 3  # Mar, Apr, Jun
    assert r["orders"] == 4


def test_several_invoices_same_day_count_as_separate_orders_one_month():
    rows = [sale(1, at(date(2010, 5, 5), 9)), sale(2, at(date(2010, 5, 5), 15), code="10002")]
    r = row(feats(rows))
    assert r["orders"] == 2
    assert r["active_months"] == 1
    assert r["recency_days"] == (CUTOFF - date(2010, 5, 5)).days


def test_orders_counts_distinct_invoices_not_lines():
    rows = [sale(1, at(date(2010, 5, 5)), inv="X", code="10001"),
            sale(2, at(date(2010, 5, 5)), inv="X", code="10002")]
    r = row(feats(rows))
    assert r["orders"] == 1
    assert r["products"] == 2


def test_avg_order_value_is_sales_over_orders():
    rows = [sale(1, at(date(2010, 5, 5)), qty=5, price=2.0),          # 10
            sale(2, at(date(2010, 5, 6)), qty=10, price=3.0, code="10002")]  # 30
    r = row(feats(rows))
    assert r["sales"] == 40.0
    assert r["avg_order_value"] == 20.0


def test_cancelled_share_is_unmatched_cancellation_over_sales():
    # The cancelled item was never sold in the data (different price), so it stays unmatched.
    rows = [sale(1, at(date(2010, 5, 5)), qty=10, price=10.0),  # 100
            cancel(2, at(date(2010, 5, 6)), qty=-1, price=25.0, code="10002")]  # -25
    assert row(feats(rows))["cancelled_share"] == 0.25


def test_cancelled_share_is_clipped_to_one_when_cancellations_exceed_sales():
    rows = [sale(1, at(date(2010, 5, 5)), qty=1, price=10.0),
            cancel(2, at(date(2010, 5, 6)), qty=-10, price=5.0, code="10002")]
    assert row(feats(rows))["cancelled_share"] == 1.0


def test_cancelled_share_is_zero_without_cancellations_and_within_unit_range():
    rows = [sale(1, at(date(2010, 5, 5)), cust="A"),
            sale(2, at(date(2010, 5, 5)), cust="B"),
            cancel(3, at(date(2010, 5, 6)), cust="B", qty=-1, price=0.5, code="10002")]
    out = feats(rows)
    assert row(out, "A")["cancelled_share"] == 0.0
    assert out["cancelled_share"].is_between(0, 1).all()


def test_corrections_do_not_count_as_cancellations_or_orders():
    rows = [sale(1, at(date(2010, 5, 5))), correction(2, at(date(2010, 5, 6)))]
    r = row(feats(rows))
    assert r["cancelled_share"] == 0.0
    assert r["orders"] == 1


def test_is_uk_flag():
    rows = [sale(1, at(date(2010, 5, 5)), cust="UK"),
            sale(2, at(date(2010, 5, 5)), cust="FR", country="France")]
    out = feats(rows)
    assert row(out, "UK")["is_uk"] == 1
    assert row(out, "FR")["is_uk"] == 0


def test_is_uk_uses_main_country_of_customer():
    rows = [sale(1, at(date(2010, 5, 5)), country="France"),
            sale(2, at(date(2010, 5, 6)), country="United Kingdom", code="10002"),
            sale(3, at(date(2010, 5, 7)), country="United Kingdom", code="10003")]
    assert row(feats(rows))["is_uk"] == 1


def test_sales_column_is_present_but_not_a_model_feature():
    out = feats([sale(1, at(date(2010, 5, 5)))])
    assert "sales" in out.columns
    assert "sales" not in FEATURES
    assert set(FEATURES) <= set(out.columns)


def test_dataset_has_one_row_per_customer_and_binary_label():
    rows = [sale(1, at(date(2010, 5, 5)), cust="A"), sale(2, at(date(2010, 5, 6)), cust="A", code="10002"),
            sale(3, at(date(2010, 5, 6)), cust="B")]
    out = churn_dataset(make_lines(rows), CUTOFF)
    assert out["customer_id"].is_unique().all()
    assert set(out["churned"].to_list()) <= {0, 1}


# ---- (7) peak_days_ahead ----

def test_peak_days_zero_for_dec_1_cutoff_with_180_days():
    # Dec 2 .. May 30: no Sep-Nov day
    assert peak_days_ahead(date(2010, 12, 1), 180) == 0


def test_peak_days_for_jun_1_cutoff_with_180_days():
    # Jun 2 .. Nov 28. Sep 1 is day 92, Nov 28 is day 180 -> 180 - 91 = 89 peak days.
    assert peak_days_ahead(date(2010, 6, 1), 180) == 89


def test_peak_days_full_autumn_is_91():
    # Aug 31 + 91 days = Sep 1 .. Nov 30
    assert peak_days_ahead(date(2010, 8, 31), 91) == 91


def test_peak_days_excludes_the_cutoff_day_itself():
    # Cutoff on Sep 30, 1 day ahead = Oct 1 -> 1; cutoff Nov 30, 1 day ahead = Dec 1 -> 0
    assert peak_days_ahead(date(2010, 9, 30), 1) == 1
    assert peak_days_ahead(date(2010, 11, 30), 1) == 0
    assert peak_days_ahead(date(2010, 8, 31), 1) == 1  # Sep 1 counts: window starts the day after


def test_peak_days_zero_horizon_is_zero():
    assert peak_days_ahead(date(2010, 9, 15), 0) == 0


def test_peak_days_feature_follows_horizon_argument():
    rows = [sale(1, at(date(2010, 8, 30)))]
    out = churn_features(make_lines(rows), date(2010, 8, 31), horizon_days=30)
    assert row(out)["peak_days_ahead"] == 30
