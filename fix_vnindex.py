"""
fix_vnindex.py — One-time repair of VNINDEX.csv.

THE BUG
daily_update.py's job_update_index appends new rows formatted '%Y-%m-%d' to a
file whose historical rows are DD/MM/YYYY. Three things then go wrong:

  1. MIXED FORMATS. No single parse works:
       dayfirst=True  -> newest 15 rows become NaT (all of July silently lost)
       default        -> 2971 NaT (history mangled)
       format='mixed' -> 0 NaT but reads 07/10/2026 as 7 Oct, a future date
     archive/4sectors.py reads this file and parses dayfirst, so every backtest
     has been running without the most recent weeks of index data.

  2. STRING SORT. `old_df.sort_values(by="date")` sorts DD/MM/YYYY text
     lexicographically, so 31/12/2019 lands next to 31/12/2025 and the file is
     not in chronological order at all.

  3. BROKEN DEDUP. The check `d_str in old_df['date'].values` compares an ISO
     string against DD/MM/YYYY rows, so it never matches history and re-runs
     can append duplicates.

THE FIX
Normalise every row to ISO (YYYY-MM-DD), drop fully-empty rows, de-duplicate
on date keeping the last occurrence, and sort chronologically. ISO sorts
correctly as text, so the existing string-sort in daily_update.py becomes
harmless once the whole file is ISO.

Ambiguity note: bare DD/MM/YYYY is ambiguous for day<=12. Vietnamese sources
write day-first, and the file's own history confirms it (values like 31/12
exist, which can only be day-first), so day-first is applied to non-ISO rows.

Writes a .bak next to the original before touching anything.

Usage:  python fix_vnindex.py            (repair in place, keeps a .bak)
        python fix_vnindex.py --check    (report only, no writes)
"""

import os
import shutil
import sys

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE, "VNINDEX.csv")


def parse_mixed(series: pd.Series) -> pd.Series:
    """ISO first, then day-first for anything left. Never guesses month-first,
    which is what produced the future dates."""
    s = series.astype(str).str.strip()
    out = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    todo = out.isna()
    if todo.any():
        out.loc[todo] = pd.to_datetime(s[todo], format="%d/%m/%Y", errors="coerce")
    todo = out.isna()
    if todo.any():   # last resort for any other separator/ordering
        out.loc[todo] = pd.to_datetime(s[todo], dayfirst=True, errors="coerce")
    return out


def main():
    check_only = "--check" in sys.argv
    if not os.path.exists(PATH):
        raise SystemExit(f"not found: {PATH}")

    df = pd.read_csv(PATH)
    n0 = len(df)

    empty = df.isna().all(axis=1).sum()
    df = df.dropna(how="all")
    df = df[df["date"].notna()]

    parsed = parse_mixed(df["date"])
    unparsed = int(parsed.isna().sum())
    if unparsed:
        print(f"  WARNING: {unparsed} rows have unparseable dates, dropping:")
        print(df.loc[parsed.isna(), "date"].head(10).to_string())
    df = df[parsed.notna()].copy()
    df["date"] = parsed[parsed.notna()]

    dupes = int(df["date"].duplicated().sum())
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date")

    future = int((df["date"] > pd.Timestamp.now().normalize()).sum())

    print(f"  rows in            : {n0:,}")
    print(f"  fully-empty removed: {empty}")
    print(f"  unparseable removed: {unparsed}")
    print(f"  duplicate dates    : {dupes}")
    print(f"  future-dated rows  : {future}")
    print(f"  rows out           : {len(df):,}")
    print(f"  range              : {df['date'].min().date()} -> {df['date'].max().date()}")

    if check_only:
        print("\n--check: no files written")
        return

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    bak = PATH + ".bak"
    shutil.copy2(PATH, bak)
    df.to_csv(PATH, index=False)
    print(f"\n  backup  -> {bak}")
    print(f"  written -> {PATH}  (all dates ISO, chronological)")


if __name__ == "__main__":
    main()
