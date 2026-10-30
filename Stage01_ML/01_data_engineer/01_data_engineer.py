\
"""
ROLE: DATA ENGINEER
Project: AI-Powered Prediction of Global Supply Chain
         Logistics Bottlenecks

Purpose:
1. Load raw shipment telemetry.
2. Validate the actual dataset schema.
3. Clean column names and data types.
4. Detect duplicates, missing values, and suspicious values.
5. Preserve delay_status and shipment identifiers.
6. Save prepared_dataset.csv and data_quality_report.csv.
"""

from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

DATA = Path("data")

INPUT_FILE = DATA / "shipment_telemetry_raw.csv"
OUTPUT_FILE = DATA / "prepared_dataset.csv"
REPORT_FILE = DATA / "data_quality_report.csv"

# --------------------------------------------------
# 2. EXPECTED DATASET SCHEMA
# --------------------------------------------------

REQUIRED_COLUMNS = [
    "timestamp",
    "shipment_id",
    "container_id",
    "carrier_code",
    "origin_port",
    "destination_port",
    "current_node_id",
    "dwell_time_hours",
    "throughput_teu_per_hr",
    "capacity_utilization_pct",
    "customs_hold_hours",
    "weather_severity_index",
    "labor_deficit_ratio",
    "equipment_availability_pct",
    "historical_delay_rate",
    "priority_tier",
    "delay_status",
    "is_synthetic",
]

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

NON_NEGATIVE_COLUMNS = [
    "dwell_time_hours",
    "throughput_teu_per_hr",
    "customs_hold_hours",
    "weather_severity_index",
]

VALID_DELAY_STATUSES = {
    "ON_TIME",
    "AT_RISK",
    "CRITICAL_DELAY",
}

# --------------------------------------------------
# 3. DATA QUALITY REPORT HELPER
# --------------------------------------------------

quality_issues = []


def record_issue(column, check, count, details):
    quality_issues.append({
        "column": column,
        "check": check,
        "issue_count": int(count),
        "details": details,
    })


# --------------------------------------------------
# 4. LOAD RAW DATA
# --------------------------------------------------

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {INPUT_FILE}\n"
            "Run this script from the project root."
        )

    DATA.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    raw_rows = len(df)

    print("=" * 60)
    print("DATA ENGINEER: SHIPMENT TELEMETRY PREPARATION")
    print("=" * 60)
    print(f"Raw rows: {raw_rows}")
    print(f"Raw columns: {len(df.columns)}")

    # --------------------------------------------------
    # 5. NORMALIZE COLUMN NAMES
    # --------------------------------------------------

    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    if df.columns.duplicated().any():
        raise ValueError(
            "Duplicate column names found after normalization."
        )

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    unexpected_columns = [
        col for col in df.columns
        if col not in REQUIRED_COLUMNS
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing:\n"
            + ", ".join(missing_columns)
        )

    print("Schema validation: PASSED")
    print("Unexpected columns:", unexpected_columns)

    # --------------------------------------------------
    # 6. REMOVE EXACT DUPLICATE ROWS
    # --------------------------------------------------

    duplicate_count = int(df.duplicated().sum())

    record_issue(
        "all_columns",
        "exact_duplicate_rows",
        duplicate_count,
        "Exact duplicate records removed.",
    )

    df = df.drop_duplicates().copy()

    print(f"Duplicate rows removed: {duplicate_count}")

    # --------------------------------------------------
    # 7. RECORD MISSING VALUES BEFORE CONVERSION
    # --------------------------------------------------

    for col in df.columns:
        missing_count = int(df[col].isna().sum())

        if missing_count > 0:
            record_issue(
                col,
                "missing_before_conversion",
                missing_count,
                "Missing values in source data.",
            )

    # --------------------------------------------------
    # 8. CONVERT TIMESTAMP
    # --------------------------------------------------

    original_timestamp = df["timestamp"].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    invalid_timestamps = int(
        (
            original_timestamp.notna()
            & df["timestamp"].isna()
        ).sum()
    )

    missing_timestamps = int(df["timestamp"].isna().sum())

    record_issue(
        "timestamp",
        "invalid_timestamp",
        invalid_timestamps,
        "Non-empty values that could not be parsed.",
    )

    record_issue(
        "timestamp",
        "missing_timestamp",
        missing_timestamps,
        "Missing or unparseable timestamps.",
    )

    # --------------------------------------------------
    # 9. CONVERT NUMERIC COLUMNS
    # --------------------------------------------------

    for col in NUMERIC_COLUMNS:
        original = df[col].copy()

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        invalid_count = int(
            (
                original.notna()
                & df[col].isna()
            ).sum()
        )

        record_issue(
            col,
            "invalid_numeric",
            invalid_count,
            "Non-empty values that could not be converted.",
        )

        record_issue(
            col,
            "missing_after_conversion",
            df[col].isna().sum(),
            "Review missing values during EDA.",
        )

    # --------------------------------------------------
    # 10. CHECK NUMERIC BOUNDARIES
    # --------------------------------------------------

    for col in NON_NEGATIVE_COLUMNS:
        invalid_count = int((df[col] < 0).sum())

        record_issue(
            col,
            "negative_value",
            invalid_count,
            "Negative values flagged; records preserved.",
        )

    # Percentages should normally be between 0 and 100.
    # Values outside this range are flagged, not deleted.
    percentage_columns = [
        "capacity_utilization_pct",
        "equipment_availability_pct",
    ]

    for col in percentage_columns:
        invalid_count = int(
            (
                (df[col] < 0)
                | (df[col] > 100)
            ).sum()
        )

        record_issue(
            col,
            "outside_percentage_range",
            invalid_count,
            "Verify measurement definitions and source data.",
        )

    # Ratios and rates are expected to be in [0, 1].
    ratio_columns = [
        "labor_deficit_ratio",
        "historical_delay_rate",
    ]

    for col in ratio_columns:
        invalid_count = int(
            (
                (df[col] < 0)
                | (df[col] > 1)
            ).sum()
        )

        record_issue(
            col,
            "outside_ratio_range",
            invalid_count,
            "Verify that this field uses a 0-to-1 scale.",
        )

    # --------------------------------------------------
    # 11. VALIDATE TARGET LABELS
    # --------------------------------------------------

    df["delay_status"] = (
        df["delay_status"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    invalid_labels = int(
        (
            df["delay_status"].notna()
            & ~df["delay_status"].isin(
                VALID_DELAY_STATUSES
            )
        ).sum()
    )

    record_issue(
        "delay_status",
        "invalid_target_label",
        invalid_labels,
        "Expected ON_TIME, AT_RISK, or CRITICAL_DELAY.",
    )

    print("\nDelay status distribution:")
    print(
        df["delay_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    # --------------------------------------------------
    # 12. CHECK SYNTHETIC DATA INDICATOR
    # --------------------------------------------------

    synthetic_values = df["is_synthetic"].astype("string").str.strip().str.lower()

    invalid_synthetic = int(
        (
            synthetic_values.notna()
            & ~synthetic_values.isin(
                ["true", "false", "1", "0"]
            )
        ).sum()
    )

    record_issue(
        "is_synthetic",
        "invalid_boolean_value",
        invalid_synthetic,
        "Expected true/false or 1/0.",
    )

    synthetic_count = int(
        synthetic_values.isin(["true", "1"]).sum()
    )

    print(
        f"\nRows marked synthetic: "
        f"{synthetic_count}/{len(df)}"
    )

    if len(df) and synthetic_count == len(df):
        print(
            "WARNING: Every row is marked synthetic. "
            "Document this limitation in the project."
        )

    # --------------------------------------------------
    # 13. SORT BY TIMESTAMP
    # --------------------------------------------------

    df = df.sort_values(
        "timestamp",
        na_position="last",
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 14. SAVE PREPARED DATASET
    # --------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S%z",
    )

    # --------------------------------------------------
    # 15. SAVE DATA QUALITY REPORT
    # --------------------------------------------------

    report = pd.DataFrame(
        quality_issues,
        columns=[
            "column",
            "check",
            "issue_count",
            "details",
        ],
    )

    report.to_csv(
        REPORT_FILE,
        index=False,
    )

    # --------------------------------------------------
    # 16. FINAL SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("DATA ENGINEER SUMMARY")
    print("=" * 60)

    print(f"Original rows: {raw_rows}")
    print(f"Prepared rows: {len(df)}")
    print(f"Prepared columns: {len(df.columns)}")

    print("\nMissing values after preparation:")
    print(df.isna().sum().to_string())

    print("\nSaved files:")
    print(f"Prepared dataset: {OUTPUT_FILE}")
    print(f"Quality report:   {REPORT_FILE}")

    print("\nData Engineer pipeline completed.")


if __name__ == "__main__":
    main()
