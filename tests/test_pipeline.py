"""
test_pipeline.py
-----------------
Lightweight sanity checks for the cleaned dataset. Not a full test suite,
but enough to prove the pipeline's cleaning rules actually hold.

Run with:  python tests/test_pipeline.py
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_PATH = BASE_DIR / "data" / "processed" / "tea_coffee_sales_clean.csv"


def main() -> int:
    if not CLEAN_PATH.exists():
        print("Clean dataset not found — run `python src/pipeline.py` first.")
        return 1

    df = pd.read_csv(CLEAN_PATH)
    failures = []

    if df["order_id"].duplicated().any():
        failures.append("Found duplicate order_id values.")

    if (df["units_sold"] < 0).any():
        failures.append("Found negative units_sold values.")

    if df[["unit_price", "units_sold", "revenue"]].isnull().any().any():
        failures.append("Found unexpected nulls in numeric columns.")

    expected_revenue = (df["units_sold"] * df["unit_price"]).round(2)
    if not (expected_revenue == df["revenue"]).all():
        failures.append("Revenue column does not equal units_sold * unit_price.")

    if not set(df["category"].unique()).issubset({"Tea", "Coffee"}):
        failures.append(f"Unexpected category values: {df['category'].unique()}")

    if failures:
        print("FAILED:")
        for f in failures:
            print(f" - {f}")
        return 1

    print(f"All checks passed on {len(df)} rows. Data quality looks good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
