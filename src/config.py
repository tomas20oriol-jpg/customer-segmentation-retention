from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

RAW_EXCEL = RAW_DATA_DIR / "online_retail_II.xlsx"
RAW_PARQUET = INTERIM_DATA_DIR / "online_retail_II.parquet"
LINES = PROCESSED_DATA_DIR / "lines.parquet"
TRANSACTIONS = PROCESSED_DATA_DIR / "transactions.parquet"
ADJUSTMENTS = PROCESSED_DATA_DIR / "adjustments.parquet"
CUSTOMERS = PROCESSED_DATA_DIR / "customers.parquet"
SEGMENTS = PROCESSED_DATA_DIR / "segments.parquet"
CHURN_SCORES = PROCESSED_DATA_DIR / "churn_scores.parquet"

RANDOM_STATE = 42
