from datetime import datetime

import polars as pl

from src.cleaning import (
    OVERLAP_START, STEPS, build_customers, build_lines, drop_charges_and_tests, drop_duplicate_lines,
    drop_missing_customer, drop_sheet_overlap, keep_valid_lines, split_purchases,
)

SCHEMA = {
    "invoice": pl.String, "stock_code": pl.String, "description": pl.String, "quantity": pl.Int64,
    "invoice_date": pl.Datetime("us"), "price": pl.Float64, "customer_id": pl.String,
    "country": pl.String, "sheet": pl.String,
}
FIRST, SECOND = "Year 2009-2010", "Year 2010-2011"
D = datetime(2010, 6, 1, 12)


def raw(*rows):
    """rows: dicts overriding a valid product sale."""
    base = dict(invoice="500001", stock_code="85123A", description="x", quantity=2, invoice_date=D,
                price=3.0, customer_id="A", country="United Kingdom", sheet=FIRST)
    return pl.DataFrame([{**base, **r} for r in rows], schema=SCHEMA)


# ---- drop_sheet_overlap ----

def test_overlap_rows_come_only_from_the_second_sheet():
    df = raw(
        dict(invoice="1", sheet=FIRST, invoice_date=datetime(2010, 12, 5)),
        dict(invoice="2", sheet=SECOND, invoice_date=datetime(2010, 12, 5)),
    )
    assert drop_sheet_overlap(df)["invoice"].to_list() == ["2"]


def test_overlap_starts_at_midnight_on_1_december_and_keeps_the_day_before():
    df = raw(
        dict(invoice="1", sheet=FIRST, invoice_date=OVERLAP_START),
        dict(invoice="2", sheet=FIRST, invoice_date=datetime(2010, 11, 30, 23, 59)),
    )
    assert drop_sheet_overlap(df)["invoice"].to_list() == ["2"]


def test_second_sheet_is_never_dropped_by_the_overlap_rule():
    df = raw(dict(sheet=SECOND, invoice_date=datetime(2010, 12, 1)), dict(sheet=SECOND, invoice_date=datetime(2011, 5, 1)))
    assert drop_sheet_overlap(df).height == 2


# ---- drop_duplicate_lines ----

def test_exact_duplicates_are_removed_and_first_occurrence_order_is_kept():
    df = raw(dict(invoice="2"), dict(invoice="1"), dict(invoice="2"))
    assert drop_duplicate_lines(df)["invoice"].to_list() == ["2", "1"]


def test_lines_that_differ_in_one_column_are_not_duplicates():
    assert drop_duplicate_lines(raw(dict(quantity=2), dict(quantity=3))).height == 2


# ---- drop_missing_customer ----

def test_lines_without_customer_are_dropped():
    df = raw(dict(customer_id=None), dict(customer_id="B"))
    assert drop_missing_customer(df)["customer_id"].to_list() == ["B"]


# ---- drop_charges_and_tests ----

def test_products_and_correction_codes_are_kept_and_charges_dropped():
    codes = ["85123A", "10002", "M", "D", "ADJUST", "ADJUST2", "POST", "DOT", "BANK CHARGES", "TEST001", "AMAZONFEE", "m"]
    kept = drop_charges_and_tests(raw(*[dict(stock_code=c) for c in codes]))["stock_code"].to_list()
    assert kept == ["85123A", "10002", "M", "D", "ADJUST", "ADJUST2"]


def test_product_code_is_exactly_five_digits_with_an_optional_letter_suffix():
    kept = drop_charges_and_tests(raw(*[dict(stock_code=c) for c in ["1234", "123456", "12345", "12345AB", "1234A"]]))
    assert kept["stock_code"].to_list() == ["12345", "12345AB"]


# ---- keep_valid_lines ----

def test_valid_sale_needs_positive_quantity_and_price():
    df = raw(dict(invoice="1"), dict(invoice="2", quantity=0), dict(invoice="3", quantity=-1),
             dict(invoice="4", price=0.0), dict(invoice="5", price=-1.0))
    assert keep_valid_lines(df)["invoice"].to_list() == ["1"]


def test_cancellation_needs_negative_quantity_and_a_c_invoice():
    df = raw(dict(invoice="C1", quantity=-2), dict(invoice="C2", quantity=2), dict(invoice="3", quantity=-2))
    assert keep_valid_lines(df)["invoice"].to_list() == ["C1"]


def test_cancellation_is_valid_even_with_zero_price_but_not_a_sale():
    df = raw(dict(invoice="C1", quantity=-2, price=0.0), dict(invoice="2", quantity=2, price=0.0))
    assert keep_valid_lines(df)["invoice"].to_list() == ["C1"]


def test_correction_needs_a_non_zero_value_whatever_its_sign():
    df = raw(dict(invoice="1", stock_code="M", quantity=1, price=5.0), dict(invoice="2", stock_code="M", quantity=-1, price=5.0),
             dict(invoice="3", stock_code="M", quantity=1, price=0.0), dict(invoice="4", stock_code="D", quantity=0, price=5.0))
    assert keep_valid_lines(df)["invoice"].to_list() == ["1", "2"]


def test_steps_run_in_the_documented_order_and_never_add_rows():
    assert [s.__name__ for s in STEPS] == [
        "drop_sheet_overlap", "drop_duplicate_lines", "drop_missing_customer", "drop_charges_and_tests", "keep_valid_lines"
    ]
    df = raw(dict(invoice="1"), dict(invoice="2", customer_id=None), dict(invoice="3", stock_code="POST"),
             dict(invoice="4", sheet=FIRST, invoice_date=datetime(2010, 12, 2)))
    for step in STEPS:
        before = df.height
        df = step(df)
        assert df.height <= before
    assert df["invoice"].to_list() == ["1"]


# ---- build_lines ----

def test_kind_and_revenue():
    df = raw(dict(invoice="1"), dict(invoice="C2", quantity=-2), dict(invoice="3", stock_code="M", quantity=1, price=-4.0))
    out = build_lines(df)
    assert out["kind"].to_list() == ["sale", "cancellation", "correction"]
    assert out["revenue"].to_list() == [6.0, -6.0, -4.0]
    assert "sheet" not in out.columns


def test_line_id_is_unique_and_follows_row_order():
    out = build_lines(raw(dict(invoice="1"), dict(invoice="2"), dict(invoice="3")))
    assert out["line_id"].to_list() == [0, 1, 2]


def test_a_c_invoice_with_a_non_product_code_is_a_correction_not_a_cancellation():
    out = build_lines(raw(dict(invoice="C9", stock_code="M", quantity=-1)))
    assert out["kind"].to_list() == ["correction"]


# ---- build_customers ----

def customers_from(*rows):
    lines = build_lines(raw(*rows))
    purchases = split_purchases(lines)
    return build_customers(lines, purchases.transactions, purchases.adjustments)


def test_customer_with_only_a_cancelled_order_has_no_sales_and_is_excluded():
    out = customers_from(dict(invoice="1"), dict(invoice="C2", quantity=-2, invoice_date=datetime(2010, 6, 2)))
    row = out.row(0, named=True)
    assert row["sales"] == 0 and row["net_revenue"] == 0 and row["included"] is False


def test_customer_with_only_corrections_is_excluded_even_with_positive_net_revenue():
    out = customers_from(dict(stock_code="M", quantity=1, price=10.0))
    row = out.row(0, named=True)
    assert row["sales"] == 0 and row["net_revenue"] == 10.0 and row["included"] is False


def test_customer_with_negative_net_revenue_is_excluded():
    out = customers_from(dict(invoice="1"), dict(invoice="2", stock_code="M", quantity=-1, price=100.0))
    row = out.row(0, named=True)
    assert row["sales"] == 6.0 and row["net_revenue"] == -94.0 and row["included"] is False


def test_normal_customer_is_included_and_net_revenue_is_sales_plus_adjustments():
    out = customers_from(dict(invoice="1"), dict(invoice="2", stock_code="M", quantity=1, price=-1.0))
    row = out.row(0, named=True)
    assert row["sales"] == 6.0 and row["adjustments"] == -1.0 and row["net_revenue"] == 5.0 and row["included"] is True


def test_main_country_is_the_most_frequent_with_alphabetical_tie_break():
    out = customers_from(dict(invoice="1", country="Spain"), dict(invoice="2", country="France"),
                         dict(invoice="3", country="Spain"), dict(invoice="4", customer_id="B", country="Spain"),
                         dict(invoice="5", customer_id="B", country="France"))
    assert dict(zip(out["customer_id"], out["country"])) == {"A": "Spain", "B": "France"}


def test_one_row_per_customer_sorted_by_id():
    out = customers_from(dict(customer_id="B"), dict(customer_id="A"), dict(customer_id="B", invoice="2"))
    assert out["customer_id"].to_list() == ["A", "B"]
