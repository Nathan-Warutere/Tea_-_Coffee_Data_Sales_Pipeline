"""
generate_data.py
-----------------
Creates a realistic (and deliberately a little messy) raw sales dataset
for a fictional "Tea & Coffee" retailer, so the pipeline has something
authentic to clean, transform, and analyze.

Run once to (re)create data/raw/tea_coffee_sales_raw.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

PRODUCTS = {
    "Earl Grey Tea": ("Tea", 4.50),
    "Green Tea": ("Tea", 3.80),
    "Chamomile Tea": ("Tea", 4.20),
    "Masala Chai": ("Tea", 4.00),
    "English Breakfast Tea": ("Tea", 4.10),
    "Espresso": ("Coffee", 3.50),
    "Cappuccino": ("Coffee", 4.75),
    "Latte": ("Coffee", 4.95),
    "Americano": ("Coffee", 3.25),
    "Cold Brew": ("Coffee", 5.25),
}

REGIONS = ["North", "South", "East", "West", "Central"]
CUSTOMER_TYPES = ["Retail", "Wholesale", "Online"]
PAYMENT_METHODS = ["Cash", "Card", "Mobile Money", None]  # None -> missing on purpose

N_ROWS = 3000

dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")

rows = []
for i in range(N_ROWS):
    product = np.random.choice(list(PRODUCTS.keys()))
    category, base_price = PRODUCTS[product]
    date = np.random.choice(dates)
    region = np.random.choice(REGIONS)
    customer_type = np.random.choice(CUSTOMER_TYPES, p=[0.55, 0.15, 0.30])
    units = int(np.random.gamma(shape=2.0, scale=4.0)) + 1
    # small random price jitter to mimic real-world promos/discounts
    unit_price = round(base_price * np.random.uniform(0.9, 1.1), 2)
    payment = np.random.choice(PAYMENT_METHODS, p=[0.25, 0.55, 0.15, 0.05])

    rows.append(
        {
            "order_id": f"ORD{10000 + i}",
            "order_date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "product": product,
            "category": category,
            "region": region,
            "customer_type": customer_type,
            "units_sold": units,
            "unit_price": unit_price,
            "payment_method": payment,
        }
    )

df = pd.DataFrame(rows)

# --- Deliberately introduce real-world messiness for the pipeline to handle ---

# 1. Duplicate a handful of rows
dupes = df.sample(25, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

# 2. Null out some unit_price and units_sold values
null_idx = df.sample(frac=0.03, random_state=2).index
df.loc[null_idx, "unit_price"] = np.nan

null_idx2 = df.sample(frac=0.02, random_state=3).index
df.loc[null_idx2, "units_sold"] = np.nan

# 3. Inconsistent text casing / whitespace in category and region
inconsistent_idx = df.sample(frac=0.05, random_state=4).index
df.loc[inconsistent_idx, "category"] = df.loc[inconsistent_idx, "category"].str.upper()

inconsistent_idx2 = df.sample(frac=0.05, random_state=5).index
df.loc[inconsistent_idx2, "region"] = " " + df.loc[inconsistent_idx2, "region"].str.lower() + " "

# 4. A few negative units_sold (data entry errors) to filter out later
error_idx = df.sample(5, random_state=6).index
df.loc[error_idx, "units_sold"] = -abs(df.loc[error_idx, "units_sold"].fillna(3))

df.to_csv("data/raw/tea_coffee_sales_raw.csv", index=False)
print(f"Generated {len(df)} rows -> data/raw/tea_coffee_sales_raw.csv")
