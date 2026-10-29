from datetime import date, datetime

import polars as pl

from src.cleaning import match_cancelled_lines, split_purchases

SCHEMA = {
    "invoice": pl.String, "stock_code": pl.String, "quantity": pl.Int64,
    "invoice_date": pl.Datetime("us"), "price": pl.Float64, "customer_id": pl.String,
    "revenue": pl.Float64, "kind": pl.String, "line_id": pl.Int64,
}


def make_lines(rows):
    """rows: (line_id, kind, customer, invoice, code, qty, price, date)"""
    return pl.DataFrame(
        [
            dict(line_id=i, kind=k, customer_id=c, invoice=inv, stock_code=code, quantity=q,
                 price=p, invoice_date=d, revenue=q * p)
            for i, k, c, inv, code, q, p, d in rows
        ],
        schema=SCHEMA,
    )


D1, D2, D3, D4 = (datetime(2010, 1, d) for d in (1, 2, 3, 4))


def sale(i, d, qty=5, price=2.0, cust="A", inv=None, code="10001"):
    return (i, "sale", cust, inv or f"S{i}", code, qty, price, d)


def cancel(i, d, qty=-5, price=2.0, cust="A", code="10001"):
    return (i, "cancellation", cust, f"C{i}", code, qty, price, d)


def correction(i, d, revenue, cust="A", code="M"):
    return (i, "correction", cust, f"C{i}", code, 1 if revenue > 0 else -1, abs(revenue), d)


def ids(df):
    return sorted(df["line_id"].to_list())


# ---- match_cancelled_lines ----

def test_cancellation_pairs_with_earlier_identical_sale():
    pairs = match_cancelled_lines(make_lines([sale(1, D1), cancel(2, D2)]))
    assert pairs.select("cancel_line", "sale_line").rows() == [(2, 1)]


def test_cancellation_before_sale_does_not_match():
    assert match_cancelled_lines(make_lines([cancel(1, D1), sale(2, D2)])).is_empty()


def test_cancellation_on_same_timestamp_as_sale_matches():
    pairs = match_cancelled_lines(make_lines([sale(1, D1), cancel(2, D1)]))
    assert pairs.height == 1


def test_cancellation_with_different_quantity_does_not_match():
    assert match_cancelled_lines(make_lines([sale(1, D1, qty=5), cancel(2, D2, qty=-3)])).is_empty()


def test_cancellation_with_different_price_does_not_match():
    assert match_cancelled_lines(make_lines([sale(1, D1, price=2.0), cancel(2, D2, price=2.5)])).is_empty()


def test_cancellation_with_different_customer_or_product_does_not_match():
    lines = make_lines([sale(1, D1), cancel(2, D2, cust="B"), cancel(3, D2, code="10002")])
    assert match_cancelled_lines(lines).is_empty()


def test_cancellation_takes_latest_earlier_sale():
    lines = make_lines([sale(1, D1), sale(2, D2), cancel(3, D3)])
    assert match_cancelled_lines(lines).select("cancel_line", "sale_line").rows() == [(3, 2)]


def test_two_cancellations_competing_for_one_sale_only_one_wins():
    lines = make_lines([sale(1, D1), cancel(2, D2), cancel(3, D3)])
    pairs = match_cancelled_lines(lines)
    assert pairs.select("cancel_line", "sale_line").rows() == [(2, 1)]  # earliest cancellation keeps it


def test_two_cancellations_with_two_sales_fall_back_to_the_older_sale():
    lines = make_lines([sale(1, D1), sale(2, D2), cancel(3, D3), cancel(4, D4)])
    pairs = match_cancelled_lines(lines)
    assert sorted(pairs.select("cancel_line", "sale_line").rows()) == [(3, 2), (4, 1)]
    assert pairs["sale_line"].n_unique() == 2


def test_fallback_does_not_pair_with_a_sale_later_than_the_cancellation():
    lines = make_lines([sale(1, D1), cancel(2, D2), cancel(3, D3), sale(4, D4)])
    pairs = match_cancelled_lines(lines)
    assert pairs.select("cancel_line", "sale_line").rows() == [(2, 1)]


def test_match_cancelled_lines_empty_input():
    assert match_cancelled_lines(make_lines([])).is_empty()


# ---- split_purchases ----

def test_cancelled_sale_and_cancellation_both_leave_transactions_and_adjustments():
    res = split_purchases(make_lines([sale(1, D1), sale(2, D1, code="10002"), cancel(3, D2)]))
    assert ids(res.transactions) == [2]
    assert res.adjustments.is_empty()


def test_unmatched_cancellation_goes_to_adjustments_and_sale_stays():
    res = split_purchases(make_lines([sale(1, D1, qty=5), cancel(2, D2, qty=-7)]))
    assert ids(res.transactions) == [1]
    assert ids(res.adjustments) == [2]


def test_customer_with_only_cancellations_has_no_transactions():
    res = split_purchases(make_lines([cancel(1, D1), cancel(2, D2, code="10002")]))
    assert res.transactions.is_empty()
    assert ids(res.adjustments) == [1, 2]


def test_single_purchase_customer_is_kept():
    res = split_purchases(make_lines([sale(1, D1)]))
    assert ids(res.transactions) == [1]


def test_several_invoices_same_day_are_all_kept():
    res = split_purchases(make_lines([sale(1, D1, inv="I1"), sale(2, D1, inv="I2", code="10002")]))
    assert res.transactions["invoice"].n_unique() == 2


def test_negative_correction_matching_an_invoice_total_does_not_remove_the_invoice():
    lines = make_lines([sale(1, D1, qty=2, price=5.0, inv="I1"), correction(2, D2, -10.0)])
    res = split_purchases(lines)
    assert ids(res.transactions) == [1]
    assert ids(res.adjustments) == [2]


def test_reverse_and_rebill_corrections_keep_sale_and_both_corrections():
    lines = make_lines([sale(1, D1, qty=2, price=5.0), correction(2, D2, -10.0), correction(3, D3, 10.0)])
    res = split_purchases(lines)
    assert ids(res.transactions) == [1]
    assert ids(res.adjustments) == [2, 3]
    assert res.adjustments["revenue"].sum() == 0


def test_positive_correction_stays_in_adjustments():
    res = split_purchases(make_lines([sale(1, D1), correction(2, D2, 4.0)]))
    assert ids(res.adjustments) == [2]
    assert ids(res.transactions) == [1]


def test_as_of_ignores_cancellation_made_after_cutoff():
    lines = make_lines([sale(1, D1), cancel(2, D3)])
    res = split_purchases(lines, as_of=D2)
    assert ids(res.transactions) == [1]
    assert res.adjustments.is_empty()
    assert res.cancelled_lines.is_empty()


def test_without_as_of_same_cancellation_removes_sale():
    assert split_purchases(make_lines([sale(1, D1), cancel(2, D3)])).transactions.is_empty()


def test_as_of_drops_sales_after_cutoff():
    res = split_purchases(make_lines([sale(1, D1), sale(2, D3, code="10002")]), as_of=D2)
    assert ids(res.transactions) == [1]


def test_line_exactly_on_as_of_is_included():
    res = split_purchases(make_lines([sale(1, D2)]), as_of=D2)
    assert ids(res.transactions) == [1]


def test_cancellation_exactly_on_as_of_is_applied():
    res = split_purchases(make_lines([sale(1, D1), cancel(2, D2)]), as_of=D2)
    assert res.transactions.is_empty()


def test_as_of_ignores_correction_made_after_cutoff():
    lines = make_lines([sale(1, D1, qty=2, price=5.0), correction(2, D3, -10.0)])
    res = split_purchases(lines, as_of=D2)
    assert ids(res.transactions) == [1]
    assert res.adjustments.is_empty()


def test_competing_cancellations_both_survive_correctly_in_split():
    res = split_purchases(make_lines([sale(1, D1), cancel(2, D2), cancel(3, D3)]))
    assert res.transactions.is_empty()
    assert ids(res.adjustments) == [3]


def test_split_purchases_empty_input():
    res = split_purchases(make_lines([]))
    assert res.transactions.is_empty() and res.adjustments.is_empty()


def test_split_purchases_empty_after_as_of():
    res = split_purchases(make_lines([sale(1, D3)]), as_of=D1)
    assert res.transactions.is_empty()


def test_as_of_date_includes_sale_late_on_that_day():
    late = datetime(2010, 1, 2, 23, 59, 59)
    res = split_purchases(make_lines([sale(1, late)]), as_of=date(2010, 1, 2))
    assert ids(res.transactions) == [1]


def test_as_of_date_excludes_next_day():
    res = split_purchases(make_lines([sale(1, D3)]), as_of=date(2010, 1, 2))
    assert res.transactions.is_empty()


def test_as_of_date_excludes_next_day_midnight_sale_but_applies_same_day_cancellation():
    lines = make_lines([sale(1, D1), cancel(2, datetime(2010, 1, 2, 18, 0)),
                        sale(3, datetime(2010, 1, 3, 0, 0), code="10002")])
    res = split_purchases(lines, as_of=date(2010, 1, 2))
    assert res.transactions.is_empty()
    assert res.adjustments.is_empty()


def test_as_of_datetime_is_used_as_is():
    lines = make_lines([sale(1, datetime(2010, 1, 2, 10, 0)), sale(2, datetime(2010, 1, 2, 12, 0), code="10002")])
    res = split_purchases(lines, as_of=datetime(2010, 1, 2, 11, 0))
    assert ids(res.transactions) == [1]


def test_purchases_has_no_corrected_invoices_field():
    assert "corrected_invoices" not in split_purchases(make_lines([]))._fields


def test_matching_is_the_same_whatever_the_row_order_with_tied_timestamps():
    rows = [sale(1, D1), sale(2, D1), cancel(3, D2), cancel(4, D2)]
    forward = match_cancelled_lines(make_lines(rows)).sort("cancel_line").select("cancel_line", "sale_line")
    backward = match_cancelled_lines(make_lines(rows[::-1])).sort("cancel_line").select("cancel_line", "sale_line")
    assert forward.rows() == backward.rows()
    assert len(forward.rows()) == 2
