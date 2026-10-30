
"""ROLE: EDA ENGINEER
Inspect data quality and operational patterns.
"""

from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA = Path("data")
INPUT_FILE = DATA / "prepared_dataset.csv"

TARGET = "delay_status"

NUMERIC_COLUMNS = [
    "dwell_time_hours",
    "throughput_teu_per_hr",
    "capacity_utilization_pct",
    "customs_hold_hours",
    "weather_severity_index",
    "labor_deficit_ratio",
    "equipment_availability_pct",
    "historical_delay_rate",
]

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Prepared dataset not found: {INPUT_FILE}"
    )

DATA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

if df.empty:
    raise ValueError("The prepared dataset contains no rows.")

print("=" * 60)
print("EDA ENGINEER: LOGISTICS TELEMETRY ANALYSIS")
print("=" * 60)

print("Dataset shape:", df.shape)
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

# --------------------------------------------------
# 3. DATA QUALITY
# --------------------------------------------------

print("\nMissing values per column:")
print(df.isna().sum().to_string())

print("\nDuplicate rows:", int(df.duplicated().sum()))

# Save a reusable data-quality summary.
quality = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing_count": df.isna().sum(),
    "missing_percent": (
        df.isna().mean() * 100
    ).round(2),
    "unique_values": df.nunique(dropna=True),
})

quality.to_csv(DATA / "eda_data_quality.csv")

# --------------------------------------------------
# 4. DELAY STATUS DISTRIBUTION
# --------------------------------------------------

if TARGET in df.columns:
    counts = df[TARGET].value_counts(dropna=False)
    percentages = (
        df[TARGET]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
    )

    distribution = pd.DataFrame({
        "count": counts,
        "percentage": percentages,
    })

    print("\nDelay status distribution:")
    print(distribution.to_string())

    distribution.to_csv(
        DATA / "eda_delay_status_distribution.csv"
    )

    # Plot class distribution.
    ax = counts.plot(
        kind="bar",
        title="Shipment Delay Status Distribution",
        xlabel="Delay Status",
        ylabel="Number of Records",
        figsize=(8, 5),
    )

    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(
        DATA / "eda_delay_status.png",
        dpi=140,
    )
    plt.close()

else:
    print(f"\nWARNING: Target '{TARGET}' not found.")

# --------------------------------------------------
# 5. NUMERIC SUMMARY
# --------------------------------------------------

numeric_df = df.select_dtypes(include="number")

if not numeric_df.empty:
    summary = numeric_df.describe().T

    summary.to_csv(DATA / "eda_numeric_summary.csv")

    print("\nNumeric summary:")
    print(summary.to_string())

    # Exclude identifier-like numeric columns from
    # correlation analysis.
    identifier_columns = [
        col for col in numeric_df.columns
        if col.lower() in {
            "shipment_id",
            "container_id",
            "current_node_id",
        }
    ]

    correlation_df = numeric_df.drop(
        columns=identifier_columns,
        errors="ignore",
    )

    correlation_df.corr().to_csv(
        DATA / "eda_correlations.csv"
    )

    print("\nSaved numeric summary and correlations.")

# --------------------------------------------------
# 6. RANGE VALIDATION
# --------------------------------------------------

checks = {}

def flag_count(column, condition):
    if column in df.columns:
        values = pd.to_numeric(
            df[column], errors="coerce"
        )
        return int(condition(values).sum())

    return None

checks["negative_dwell"] = flag_count(
    "dwell_time_hours",
    lambda x: x < 0,
)

checks["negative_throughput"] = flag_count(
    "throughput_teu_per_hr",
    lambda x: x < 0,
)

checks["negative_customs_hold"] = flag_count(
    "customs_hold_hours",
    lambda x: x < 0,
)

checks["capacity_outside_0_100"] = flag_count(
    "capacity_utilization_pct",
    lambda x: (x < 0) | (x > 100),
)

checks["equipment_outside_0_100"] = flag_count(
    "equipment_availability_pct",
    lambda x: (x < 0) | (x > 100),
)

checks["labor_ratio_outside_0_1"] = flag_count(
    "labor_deficit_ratio",
    lambda x: (x < 0) | (x > 1),
)

checks["historical_delay_rate_outside_0_1"] = flag_count(
    "historical_delay_rate",
    lambda x: (x < 0) | (x > 1),
)

print("\nPotential range violations (review):")

for name, count in checks.items():
    print(f"{name}: {count if count is not None else 'Column missing'}")

pd.DataFrame(
    [
        {"check": name, "issue_count": count}
        for name, count in checks.items()
    ]
).to_csv(DATA / "eda_range_violations.csv", index=False)

# --------------------------------------------------
# 7. FEATURE DISTRIBUTIONS
# --------------------------------------------------

for col in [
    "dwell_time_hours",
    "capacity_utilization_pct",
    "throughput_teu_per_hr",
]:
    if col not in df.columns:
        continue

    values = pd.to_numeric(df[col], errors="coerce").dropna()

    if values.empty:
        continue

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=30, edgecolor="black")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(
        DATA / f"eda_{col}_distribution.png",
        dpi=140,
    )

    plt.close()

# --------------------------------------------------
# 8. FINAL SUMMARY
# --------------------------------------------------

print("\nEDA artifacts saved under:", DATA.resolve())
print("EDA analysis completed successfully.")
