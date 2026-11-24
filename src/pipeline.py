"""
pipeline.py
-----------
Tea & Coffee Sales Data Pipeline
A standalone capstone project applying every module of Kaggle's
"Pandas" micro-course to a single, coherent dataset.

Kaggle module -> where it's used in this file:
  1. Creating, Reading & Writing   -> load_raw_data(), each export step
  2. Indexing, Selecting & Assigning -> clean_data(), feature engineering
  3. Summary Functions & Maps      -> profile_data(), apply()/map() usage
  4. Grouping & Sorting            -> aggregate_sales()
  5. Data Types & Missing Values   -> clean_data()
  6. Renaming & Combining          -> enrich_with_region_targets(), rename steps

Run with:  python src/pipeline.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for chart export
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------
# Paths & logging
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "tea_coffee_sales_raw.csv"
CLEAN_PATH = BASE_DIR / "data" / "processed" / "tea_coffee_sales_clean.csv"
REPORT_DIR = BASE_DIR / "output" / "reports"
CHART_DIR = BASE_DIR / "output" / "charts"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tea_coffee_pipeline")


# --------------------------------------------------------------------------
# 1. Creating, Reading & Writing
# --------------------------------------------------------------------------

def load_raw_data(path: Path) -> pd.DataFrame:
    """Read the raw CSV export into a DataFrame."""
    log.info("Loading raw data from %s", path)
    df = pd.read_csv(path, parse_dates=["order_date"])
    log.info("Loaded %d rows, %d columns", *df.shape)
    return df


# --------------------------------------------------------------------------
# 2 & 5. Indexing/Selecting/Assigning + Data Types & Missing Values
# --------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data: fix text, dtypes, duplicates, missing & invalid values."""
    df = df.copy()

    # Normalize text columns: strip whitespace, consistent casing
    df["category"] = df["category"].str.strip().str.title()
    df["region"] = df["region"].str.strip().str.title()
    df["product"] = df["product"].str.strip()

    # Drop exact duplicate orders
    before = len(df)
    df = df.drop_duplicates(subset=["order_id"])
    log.info("Dropped %d duplicate rows", before - len(df))

    # Fix invalid negative units_sold (data entry errors) -> take absolute value
    neg_mask = df["units_sold"] < 0
    log.info("Fixing %d negative units_sold values", neg_mask.sum())
    df.loc[neg_mask, "units_sold"] = df.loc[neg_mask, "units_sold"].abs()

    # Impute missing numeric values with the per-product median (more
    # representative than a single global median)
    df["unit_price"] = df.groupby("product")["unit_price"].transform(
        lambda s: s.fillna(s.median())
    )
    df["units_sold"] = df.groupby("product")["units_sold"].transform(
        lambda s: s.fillna(s.median())
    )

    # Fill missing payment_method with an explicit "Unknown" category rather
    # than silently dropping rows
    df["payment_method"] = df["payment_method"].fillna("Unknown")

    # Correct dtypes
    df["units_sold"] = df["units_sold"].round().astype("int64")
    df["unit_price"] = df["unit_price"].round(2).astype("float64")
    df["category"] = df["category"].astype("category")
    df["region"] = df["region"].astype("category")
    df["customer_type"] = df["customer_type"].astype("category")
    df["payment_method"] = df["payment_method"].astype("category")

    # Feature engineering (assigning new columns)
    df["revenue"] = (df["units_sold"] * df["unit_price"]).round(2)
    df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["order_year"] = df["order_date"].dt.year
    df["day_of_week"] = df["order_date"].dt.day_name()

    df = df.reset_index(drop=True)
    log.info("Clean dataset ready: %d rows, %d columns", *df.shape)
    return df


# --------------------------------------------------------------------------
# 3. Summary Functions & Maps
# --------------------------------------------------------------------------

def profile_data(df: pd.DataFrame) -> dict:
    """Produce descriptive-statistics summaries, using map/apply-style logic."""
    summary = {
        "total_orders": len(df),
        "total_revenue": round(df["revenue"].sum(), 2),
        "average_order_value": round(df["revenue"].mean(), 2),
        "most_popular_product": df["product"].value_counts().idxmax(),
        "top_region_by_revenue": df.groupby("region")["revenue"].sum().idxmax(),
    }

    # map(): classify each order by size using a lookup function
    def size_band(units: int) -> str:
        if units <= 5:
            return "Small"
        if units <= 15:
            return "Medium"
        return "Large"

    df["order_size_band"] = df["units_sold"].map(size_band)

    log.info("Profile summary: %s", summary)
    return summary


# --------------------------------------------------------------------------
# 4. Grouping & Sorting
# --------------------------------------------------------------------------

def aggregate_sales(df: pd.DataFrame):
    """Build the core aggregated views used for reporting."""
    by_category = (
        df.groupby("category", observed=True)["revenue"]
        .agg(total_revenue="sum", avg_order_value="mean", orders="count")
        .round(2)
        .sort_values("total_revenue", ascending=False)
    )

    by_region = (
        df.groupby("region", observed=True)["revenue"]
        .sum()
        .round(2)
        .sort_values(ascending=False)
    )

    by_month = (
        df.groupby("order_month")["revenue"]
        .sum()
        .round(2)
        .sort_index()
    )

    top_products = (
        df.groupby("product")["revenue"]
        .sum()
        .round(2)
        .sort_values(ascending=False)
        .head(5)
    )

    pivot_region_category = pd.pivot_table(
        df,
        values="revenue",
        index="region",
        columns="category",
        aggfunc="sum",
        observed=True,
        fill_value=0,
    ).round(2)

    return {
        "by_category": by_category,
        "by_region": by_region,
        "by_month": by_month,
        "top_products": top_products,
        "pivot_region_category": pivot_region_category,
    }


# --------------------------------------------------------------------------
# 6. Renaming & Combining
# --------------------------------------------------------------------------

def enrich_with_region_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Combine sales data with a small lookup table of regional revenue
    targets, demonstrating merge + rename."""
    targets = pd.DataFrame(
        {
            "region": ["North", "South", "East", "West", "Central"],
            "quarterly_target": [15000, 12000, 13000, 14000, 11000],
        }
    ).rename(columns={"quarterly_target": "region_quarterly_target"})

    region_actuals = (
        df.groupby("region", observed=True)["revenue"].sum().reset_index()
        .rename(columns={"revenue": "region_actual_revenue"})
    )

    merged = region_actuals.merge(targets, on="region", how="left")
    merged["pct_of_target"] = (
        merged["region_actual_revenue"] / merged["region_quarterly_target"] * 100
    ).round(1)
    return merged.sort_values("pct_of_target", ascending=False)


# --------------------------------------------------------------------------
# Reporting & charts
# --------------------------------------------------------------------------

def export_reports(summary, agg, targets, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    agg["by_category"].to_csv(out_dir / "revenue_by_category.csv")
    agg["by_region"].to_csv(out_dir / "revenue_by_region.csv")
    agg["by_month"].to_csv(out_dir / "revenue_by_month.csv")
    agg["top_products"].to_csv(out_dir / "top_5_products.csv")
    agg["pivot_region_category"].to_csv(out_dir / "pivot_region_by_category.csv")
    targets.to_csv(out_dir / "region_vs_target.csv", index=False)

    with open(out_dir / "summary.md", "w") as f:
        f.write("# Tea & Coffee Sales — Summary Report\n\n")
        for key, value in summary.items():
            f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")

    log.info("Reports written to %s", out_dir)


def export_charts(agg, chart_dir: Path) -> None:
    chart_dir.mkdir(parents=True, exist_ok=True)

    # Monthly revenue trend
    fig, ax = plt.subplots(figsize=(9, 4.5))
    agg["by_month"].plot(kind="line", marker="o", ax=ax, color="#6f4e37")
    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue ($)")
    plt.xticks(rotation=60, fontsize=7)
    fig.tight_layout()
    fig.savefig(chart_dir / "monthly_revenue_trend.png", dpi=150)
    plt.close(fig)

    # Revenue by category
    fig, ax = plt.subplots(figsize=(5, 4))
    agg["by_category"]["total_revenue"].plot(kind="bar", ax=ax, color=["#6f4e37", "#8bc34a"])
    ax.set_title("Total Revenue: Tea vs Coffee")
    ax.set_ylabel("Revenue ($)")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(chart_dir / "revenue_by_category.png", dpi=150)
    plt.close(fig)

    # Top 5 products
    fig, ax = plt.subplots(figsize=(6, 4))
    agg["top_products"].sort_values().plot(kind="barh", ax=ax, color="#4a7c59")
    ax.set_title("Top 5 Products by Revenue")
    ax.set_xlabel("Revenue ($)")
    fig.tight_layout()
    fig.savefig(chart_dir / "top_5_products.png", dpi=150)
    plt.close(fig)

    # Revenue by region
    fig, ax = plt.subplots(figsize=(6, 4))
    agg["by_region"].plot(kind="bar", ax=ax, color="#b08968")
    ax.set_title("Revenue by Region")
    ax.set_ylabel("Revenue ($)")
    plt.xticks(rotation=30)
    fig.tight_layout()
    fig.savefig(chart_dir / "revenue_by_region.png", dpi=150)
    plt.close(fig)

    log.info("Charts written to %s", chart_dir)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_pipeline() -> None:
    raw_df = load_raw_data(RAW_PATH)
    clean_df = clean_data(raw_df)
    summary = profile_data(clean_df)
    agg = aggregate_sales(clean_df)
    targets = enrich_with_region_targets(clean_df)

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEAN_PATH, index=False)
    log.info("Clean data written to %s", CLEAN_PATH)

    export_reports(summary, agg, targets, REPORT_DIR)
    export_charts(agg, CHART_DIR)

    log.info("Pipeline complete. See output/reports and output/charts.")


if __name__ == "__main__":
    run_pipeline()
