#!/usr/bin/env python3
"""
Lookup Table Filter Tool (robust to '#file_name' vs 'file_name' headers, etc.)
- Auto-discovers lookup tables under common roots
- Normalizes headers (strips leading '#', trims whitespace)
- Simple interactive filters (by file_name regex + per-column conditions)
- Writes a filtered CSV alongside the source

Usage: python svo_spectra_filter.py
"""

import glob
import os
import re
import sys
from datetime import datetime

import pandas as pd

# Try these roots automatically before prompting
DEFAULT_BASE_DIRS = [
    "data/stellar_models",
    "./data/stellar_models",
    "../data/stellar_models",
]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lstrip("#").strip() for c in df.columns]
    return df

def _load_lookup(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df)

    # Identify a filename column and canonicalize to 'file_name'
    fname_col = None
    for cand in ("file_name", "filename", "file"):
        if cand in df.columns:
            fname_col = cand
            break
    if fname_col is None:
        raise KeyError(
            f"Could not find a filename column in: {path}\n"
            f"Columns found: {list(df.columns)}\n"
            f"(Expected one of: 'file_name', 'filename', 'file')"
        )

    # Always ensure string dtype and a 'file_name' alias
    df[fname_col] = df[fname_col].astype(str)
    if "file_name" not in df.columns:
        df["file_name"] = df[fname_col]
    return df

def _find_base_dir() -> str:
    for cand in DEFAULT_BASE_DIRS:
        if os.path.isdir(cand):
            return cand
    print("Enter the path to the base directory containing lookup tables")
    print(f"(Tried: {', '.join(DEFAULT_BASE_DIRS)})")
    base = input("Base directory path: ").strip()
    if not base:
        sys.exit("Error: No base directory provided.")
    if not os.path.isdir(base):
        sys.exit(f"Error: Base directory does not exist: {base}")
    return base

def _discover_lookup_tables(base_dir: str):
    # Search up to 2 levels deep for lookup_table*.csv
    found = []
    for root, dirs, files in os.walk(base_dir):
        rel_depth = os.path.relpath(root, start=base_dir).count(os.sep)
        if rel_depth > 2:
            dirs[:] = []
            continue
        for fn in files:
            low = fn.lower()
            if low.startswith("lookup_table") and low.endswith(".csv"):
                found.append(os.path.join(root, fn))
    return sorted(found)

def _pick_one(items, prompt="Choose an item: "):
    for i, it in enumerate(items, 1):
        print(f"{i}. {it}")
    choice = input(prompt).strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(items)):
        sys.exit("Invalid selection.")
    return items[int(choice) - 1]

def _apply_file_regex(df: pd.DataFrame) -> pd.DataFrame:
    inc = input("Regex to INCLUDE by file_name (Enter to skip): ").strip()
    if inc:
        try:
            rx = re.compile(inc)
            df = df[df["file_name"].map(lambda s: bool(rx.search(s)))]
        except re.error as e:
            sys.exit(f"Invalid include regex: {e}")

    exc = input("Regex to EXCLUDE by file_name (Enter to skip): ").strip()
    if exc:
        try:
            rx = re.compile(exc)
            df = df[~df["file_name"].map(lambda s: bool(rx.search(s)))]
        except re.error as e:
            sys.exit(f"Invalid exclude regex: {e}")
    return df

def _apply_column_filters(df: pd.DataFrame) -> pd.DataFrame:
    print("\nColumns available:\n", ", ".join(df.columns))
    print(
        "\nAdd per-column filters (press Enter at column prompt to finish). "
        "Supported ops: ==, !=, >, >=, <, <=, contains (regex)."
    )
    while True:
        col = input("\nColumn to filter (Enter to finish): ").strip()
        if not col:
            break
        if col not in df.columns:
            print(f"  ! '{col}' not in columns.")
            continue
        op = input("Operator [==, !=, >, >=, <, <=, contains]: ").strip()
        val = input("Value (string/number; regex for 'contains'): ").strip()

        try:
            if op == "contains":
                rx = re.compile(val)
                df = df[df[col].astype(str).map(lambda s: bool(rx.search(s)))]
            else:
                # Try numeric comparison when possible, else string compare
                series = df[col]
                try:
                    lhs = pd.to_numeric(series, errors="coerce")
                    rhs = pd.to_numeric(pd.Series([val]*len(df)), errors="coerce").iloc[0]
                    is_numeric = not pd.isna(rhs)
                except Exception:
                    is_numeric = False

                if is_numeric:
                    if op == "==": df = df[lhs == rhs]
                    elif op == "!=": df = df[lhs != rhs]
                    elif op == ">":  df = df[lhs > rhs]
                    elif op == ">=": df = df[lhs >= rhs]
                    elif op == "<":  df = df[lhs < rhs]
                    elif op == "<=": df = df[lhs <= rhs]
                    else: print("  ! Unknown operator."); continue
                else:
                    sval = val
                    colstr = series.astype(str)
                    if op == "==": df = df[colstr == sval]
                    elif op == "!=": df = df[colstr != sval]
                    elif op == ">":  df = df[colstr >  sval]
                    elif op == ">=": df = df[colstr >= sval]
                    elif op == "<":  df = df[colstr <  sval]
                    elif op == "<=": df = df[colstr <= sval]
                    else: print("  ! Unknown operator."); continue
        except re.error as e:
            sys.exit(f"Invalid regex: {e}")
        except Exception as e:
            sys.exit(f"Filter error: {e}")
        print(f"  -> rows remaining: {len(df):,}")
    return df

def main():
    print("Welcome to the Lookup Table Filter Tool!")
    base_dir = _find_base_dir()
    tables = _discover_lookup_tables(base_dir)
    if not tables:
        sys.exit("No lookup tables found in the specified directory.")

    print("\nAvailable lookup tables:")
    table_path = _pick_one(tables, "Enter the number of the lookup table you want to filter: ")

    print(f"\nSelected lookup table: {table_path}")
    df = _load_lookup(table_path)
    print(f"Rows: {len(df):,} | Columns: {list(df.columns)}")

    # Quick file_name regex filters
    df = _apply_file_regex(df)

    # Optional per-column filters
    df = _apply_column_filters(df)

    # Write output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(os.path.dirname(table_path), f"lookup_table_filtered_{ts}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}\nDone.")

if __name__ == "__main__":
    main()
