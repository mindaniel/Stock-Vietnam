"""
merge_fiintrade_into_investor_flow.py — Merge the validated FiinTrade-derived
investor-flow data (data/investor_flow_fiintrade, built by
build_fiintrade_investor_flow.py) into data/investor_flow (nguoiquansat,
Sep 2024+), extending each ticker's history backward.

Validated in-session: after fixing the Match-field bug, all 5 flow columns
match nguoiquansat at rounding-level precision (median diff ~0.000-0.002 tỷ)
across a 40-ticker random sample, ~16,500 overlapping ticker-days. On
overlapping dates, nguoiquansat is kept as-is (it's the source everything
else in the codebase was built/tested against) — FiinTrade only fills in
dates BEFORE nguoiquansat's own coverage starts for that ticker.

data/investor_flow is git-tracked, so this is fully reversible via
`git checkout -- data/investor_flow` if anything looks wrong after.

Usage: python archive/merge_fiintrade_into_investor_flow.py
"""

import glob
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NQ_DIR = os.path.join(BASE, "data", "investor_flow")
FIIN_DIR = os.path.join(BASE, "data", "investor_flow_fiintrade")


def main():
    fiin_tickers = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(FIIN_DIR, "*.parquet"))}
    nq_tickers = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(NQ_DIR, "*.parquet"))}
    all_tickers = sorted(fiin_tickers | nq_tickers)

    print(f"nguoiquansat: {len(nq_tickers)} tickers, fiintrade: {len(fiin_tickers)} tickers, "
          f"union: {len(all_tickers)} tickers")

    n_extended = 0
    n_new = 0
    n_unchanged = 0
    total_days_added = 0

    for ticker in all_tickers:
        nq_path = os.path.join(NQ_DIR, f"{ticker}.parquet")
        fiin_path = os.path.join(FIIN_DIR, f"{ticker}.parquet")

        nq = pd.read_parquet(nq_path) if os.path.exists(nq_path) else pd.DataFrame()
        fiin = pd.read_parquet(fiin_path) if os.path.exists(fiin_path) else pd.DataFrame()

        if not nq.empty:
            nq["date"] = pd.to_datetime(nq["date"])
        if not fiin.empty:
            fiin["date"] = pd.to_datetime(fiin["date"])
            fiin["ticker"] = ticker

        if nq.empty and fiin.empty:
            continue

        if nq.empty:
            # No nguoiquansat coverage at all — fiintrade-only ticker.
            merged = fiin.sort_values("date").reset_index(drop=True)
            n_new += 1
        elif fiin.empty:
            merged = nq  # nothing to add
            n_unchanged += 1
        else:
            nq_min_date = nq["date"].min()
            fiin_before = fiin[fiin["date"] < nq_min_date]
            if fiin_before.empty:
                merged = nq
                n_unchanged += 1
            else:
                merged = pd.concat([fiin_before, nq], ignore_index=True)
                merged = merged.sort_values("date").reset_index(drop=True)
                total_days_added += len(fiin_before)
                n_extended += 1

        merged.to_parquet(nq_path, index=False)

    print(f"\nExtended (fiintrade added pre-nguoiquansat history): {n_extended} tickers, "
          f"{total_days_added:,} ticker-days added")
    print(f"New (fiintrade-only, no nguoiquansat coverage):       {n_new} tickers")
    print(f"Unchanged (no earlier fiintrade data available):     {n_unchanged} tickers")
    print(f"\nSaved to {NQ_DIR} (git-tracked — revert with: git checkout -- data/investor_flow)")


if __name__ == "__main__":
    main()
