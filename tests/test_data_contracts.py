import pytest
import polars as pl


from pathlib import Path

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
PRODUCT = r"^\d{5}[A-Za-z]*$"


def needs(name):
    return pytest.mark.skipif(not (PROCESSED / name).exists(), reason=f"{name} not available")


def load(name):
    return pl.read_parquet(PROCESSED / name)


COLUMNS = ["invoice", "stock_code", "quantity", "invoice_date", "price", "customer_id", "revenue", "kind", "line_id"]


# ---- lines ----
@needs("lines.parquet")
def test_lines_has_expected_columns_and_kinds():
    df = load("lines.parquet")
    assert set(COLUMNS) <= set(df.columns)
    assert set(df["kind"].unique()) <= {"sale", "cancellation", "correction"}


@needs("lines.parquet")
def test_lines_line_id_unique_and_no_null_customer():
    df = load("lines.parquet")
    assert df["line_id"].is_unique().all()
    assert df["customer_id"].null_count() == 0


@needs("lines.parquet")
def test_lines_sales_positive_and_cancellations_negative():
    df = load("lines.parquet")
    sales = df.filter(pl.col("kind") == "sale")
    canc = df.filter(pl.col("kind") == "cancellation")
    assert (sales["quantity"] > 0).all() and (sales["price"] > 0).all()
    assert (canc["quantity"] < 0).all()
    assert canc["invoice"].str.starts_with("C").all()
    assert not sales["invoice"].str.starts_with("C").any()


@needs("lines.parquet")
def test_lines_revenue_equals_quantity_times_price():
    df = load("lines.parquet")
    assert ((df["revenue"] - df["quantity"] * df["price"]).abs() < 1e-6).all()


# ---- transactions ----
@needs("transactions.parquet")
def test_transactions_no_null_customer_id():
    assert load("transactions.parquet")["customer_id"].null_count() == 0


@needs("transactions.parquet")
def test_transactions_quantity_and_price_positive():
    df = load("transactions.parquet")
    assert (df["quantity"] > 0).all()
    assert (df["price"] > 0).all()


@needs("transactions.parquet")
def test_transactions_revenue_equals_quantity_times_price():
    df = load("transactions.parquet")
    assert ((df["revenue"] - df["quantity"] * df["price"]).abs() < 1e-6).all()


@needs("transactions.parquet")
def test_transactions_no_exact_duplicate_rows():
    df = load("transactions.parquet")
    assert df.is_duplicated().sum() == 0


@needs("transactions.parquet")
def test_transactions_no_cancellation_invoices():
    assert not load("transactions.parquet")["invoice"].str.starts_with("C").any()


@needs("transactions.parquet")
def test_transactions_stock_codes_are_products():
    assert load("transactions.parquet")["stock_code"].str.contains(PRODUCT).all()


@needs("transactions.parquet")
def test_transactions_are_all_sales():
    assert set(load("transactions.parquet")["kind"].unique()) == {"sale"}


# ---- adjustments ----
@needs("adjustments.parquet")
def test_adjustments_kind_is_cancellation_or_correction():
    assert set(load("adjustments.parquet")["kind"].unique()) <= {"cancellation", "correction"}


@needs("adjustments.parquet")
def test_adjustment_cancellations_are_negative_product_lines_with_c_invoice():
    c = load("adjustments.parquet").filter(pl.col("kind") == "cancellation")
    assert c["invoice"].str.starts_with("C").all()
    assert (c["quantity"] < 0).all()
    assert c["stock_code"].str.contains(PRODUCT).all()


@needs("adjustments.parquet")
def test_adjustment_corrections_have_known_codes_and_nonzero_revenue():
    c = load("adjustments.parquet").filter(pl.col("kind") == "correction")
    assert c["stock_code"].is_in(["M", "D", "ADJUST", "ADJUST2"]).all()
    assert (c["revenue"] != 0).all()


# ---- customers ----
@needs("customers.parquet")
def test_customers_id_unique():
    assert load("customers.parquet")["customer_id"].is_unique().all()


@needs("customers.parquet")
def test_customers_net_revenue_is_sales_plus_adjustments():
    df = load("customers.parquet")
    assert ((df["net_revenue"] - (df["sales"] + df["adjustments"])).abs() < 1e-6).all()


@needs("customers.parquet")
def test_customers_included_implies_positive_sales_and_net_revenue():
    inc = load("customers.parquet").filter(pl.col("included"))
    assert (inc["sales"] > 0).all()
    assert (inc["net_revenue"] > 0).all()


# ---- segments ----
@needs("segments.parquet")
def test_segments_customer_id_unique():
    assert load("segments.parquet")["customer_id"].is_unique().all()


@needs("segments.parquet")
def test_segments_are_the_four_kmeans_segments():
    seg = load("segments.parquet")["segment"]
    assert seg.null_count() == 0
    assert set(seg.unique()) == {"Champions", "Promising", "At risk", "Lost"}


@needs("segments.parquet")
def test_segments_scores_between_1_and_5():
    df = load("segments.parquet")
    for c in ("r_score", "f_score", "m_score", "fm_score"):
        assert df[c].null_count() == 0
        assert df[c].min() >= 1 and df[c].max() <= 5


@needs("segments.parquet")
@needs("customers.parquet")
def test_segments_cover_exactly_the_included_customers():
    seg = set(load("segments.parquet")["customer_id"])
    inc = set(load("customers.parquet").filter(pl.col("included"))["customer_id"])
    assert seg == inc


# ---- cross-file ----
@needs("customers.parquet")
@needs("transactions.parquet")
def test_customers_sales_match_transactions_revenue():
    cust = load("customers.parquet")
    tx = load("transactions.parquet").group_by("customer_id").agg(t=pl.col("revenue").sum())
    j = cust.join(tx, on="customer_id", how="left").with_columns(pl.col("t").fill_null(0))
    assert ((j["sales"] - j["t"]).abs() < 1e-4).all()
