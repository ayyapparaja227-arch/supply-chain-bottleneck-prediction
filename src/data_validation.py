"""Basic CSV validation for the Stage 1 supply-chain project."""
import argparse
from pathlib import Path
import pandas as pd

def inspect_csv(path: str) -> None:
    df = pd.read_csv(path)
    print(f"File: {path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumns and dtypes:")
    print(df.dtypes.to_string())
    print("\nMissing values (top 30):")
    print(df.isna().sum().sort_values(ascending=False).head(30).to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print("\nSample:")
    print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a shipment-level CSV")
    args = parser.parse_args()
    if not Path(args.input).exists():
        raise SystemExit(f"File not found: {args.input}")
    inspect_csv(args.input)
