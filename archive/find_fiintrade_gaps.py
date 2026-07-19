"""
find_fiintrade_gaps.py — For each ticker with harvested FiinTrade investor-
classification data (fiintrade_combined.parquet), find trading dates that
fall WITHIN that ticker's own captured date range but have no data (gaps
from harvest interruptions), so a follow-up script can re-fetch just those.

Reference trading calendar per ticker = data/price/{ticker}.parquet's own
"time" column (real trading days that ticker actually traded, not a generic
weekday calendar — avoids flagging holidays/listing gaps as "missing").

Output: fiintrade_missing_dates.csv, columns: ticker, from_date, to_date,
missing_days — missing dates per ticker collapsed into <=30-calendar-day
ranges (greedy chunking: extend a range while last-first <= RANGE_MAX_DAYS,
start a new one once that would be exceeded). Matches the harvest script's
setDateRangeAndApply(from, to) — each output row is directly one call.

Usage: python archive/find_fiintrade_gaps.py
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMBINED_PATH = os.path.join(BASE, "fiintrade_combined.parquet")
PRICE_DIR = os.path.join(BASE, "data", "price")
OUT_PATH = os.path.join(BASE, "fiintrade_missing_dates.csv")
RANGE_MAX_DAYS = 30

# Any one of these being non-null marks a row as "has real investor data"
# (a captured response, not just page metadata / other endpoint noise).
INVESTOR_COLS = [
    "item_localIndividualBuyValue",
    "item_localInstitutionalBuyValue",
    "item_foreignIndividualBuyTradingMatchValue",
    "item_foreignInstitutionalBuyTradingMatchValue",
    "item_proprietaryTotalBuyTradeValue",
]


def chunk_into_ranges(dates: list, max_days: int = RANGE_MAX_DAYS) -> list:
    """Greedily group sorted dates into ranges spanning at most `max_days`
    calendar days each. Returns list of (from_date, to_date, count)."""
    if not dates:
        return []
    ranges = []
    start = dates[0]
    prev = dates[0]
    count = 1
    for d in dates[1:]:
        if (d - start).days <= max_days:
            prev = d
            count += 1
        else:
            ranges.append((start, prev, count))
            start = d
            prev = d
            count = 1
    ranges.append((start, prev, count))
    return ranges


def main():
    print("Loading fiintrade_combined.parquet...")
    df = pd.read_parquet(COMBINED_PATH, columns=["item_code", "item_tradingDate", "capturedAt"] + INVESTOR_COLS)
    df["item_tradingDate"] = pd.to_datetime(df["item_tradingDate"], errors="coerce")
    df = df.dropna(subset=["item_code", "item_tradingDate"])

    has_data = df[INVESTOR_COLS].notna().any(axis=1)
    df = df[has_data].copy()
    df = df.sort_values("capturedAt").drop_duplicates(subset=["item_code", "item_tradingDate"], keep="last")
    print(f"  {df['item_code'].nunique()} tickers with investor-data captures, {len(df):,} ticker-days")

    rows = []
    no_price_file = []
    for ticker, g in df.groupby("item_code"):
        captured = set(g["item_tradingDate"].dt.normalize())
        cap_min, cap_max = min(captured), max(captured)

        price_path = os.path.join(PRICE_DIR, f"{ticker}.parquet")
        if not os.path.exists(price_path):
            no_price_file.append(ticker)
            continue
        try:
            pdf = pd.read_parquet(price_path, columns=["time"])
            pdf["time"] = pd.to_datetime(pdf["time"], errors="coerce")
            trading_dates = set(pdf["time"].dropna().dt.normalize())
        except Exception:
            no_price_file.append(ticker)
            continue

        expected = {d for d in trading_dates if cap_min <= d <= cap_max}
        missing = sorted(expected - captured)
        for from_d, to_d, cnt in chunk_into_ranges(missing):
            rows.append({
                "ticker": ticker,
                "from_date": from_d.strftime("%Y-%m-%d"),
                "to_date": to_d.strftime("%Y-%m-%d"),
                "missing_days": cnt,
            })

    out = pd.DataFrame(rows).sort_values(["ticker", "from_date"])
    out.to_csv(OUT_PATH, index=False)

    n_tickers_with_gaps = out["ticker"].nunique() if not out.empty else 0
    n_missing_days = out["missing_days"].sum() if not out.empty else 0
    print(f"\n  {len(out):,} ranges ({n_missing_days:,} missing days) across {n_tickers_with_gaps} tickers")
    print(f"  Saved to {OUT_PATH}")
    if no_price_file:
        print(f"  {len(no_price_file)} tickers skipped (no data/price/*.parquet to use as reference calendar): "
              f"{no_price_file[:20]}{'...' if len(no_price_file) > 20 else ''}")

    if not out.empty:
        print("\n  Top 20 tickers by number of ranges:")
        print(out["ticker"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
