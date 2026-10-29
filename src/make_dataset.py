"""Convert the UCI Excel file (two yearly sheets) into a single parquet file.

Reading the 45 MB workbook takes minutes, so this runs once and the notebooks
read the parquet. Run from the project root: `.venv/bin/python -m src.make_dataset`.
"""

import pandas as pd
import polars as pl

from src.config import RAW_EXCEL, RAW_PARQUET

# Customer ID is read as text: it has missing values and must not become a float.
DTYPES = {"Invoice": str, "StockCode": str, "Description": str, "Customer ID": str}


def main() -> None:
    sheets = pd.read_excel(RAW_EXCEL, sheet_name=None, dtype=DTYPES)
    df = pl.concat(
        [pl.from_pandas(sheet).with_columns(pl.lit(name).alias("Sheet")) for name, sheet in sheets.items()]
    )
    df = df.rename(
        {
            "Invoice": "invoice",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "Price": "price",
            "Customer ID": "customer_id",
            "Country": "country",
            "Sheet": "sheet",
        }
    )
    RAW_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(RAW_PARQUET)
    print(f"{df.height:,} rows written to {RAW_PARQUET}")


if __name__ == "__main__":
    main()
