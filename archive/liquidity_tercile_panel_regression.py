"""
liquidity_tercile_panel_regression.py — Does any investor-flow signal show
up specifically in LESS liquid stocks, where a given flow imbalance is a
bigger fraction of daily volume and should have more price impact, even if
it washes out in liquid large-caps (where the same flow is easily absorbed)?

Reuses all_players_panel_regression.py's panel-building (identical universe,
accumulation-score construction, fundamentals attachment) — the only change
is splitting the pooled full-sample panel into liquidity TERCILES (by each
ticker's own median daily traded value) and running the same 5-player
regression within each tercile separately, instead of by calendar period.

Usage:  python archive/liquidity_tercile_panel_regression.py
"""

import glob, os, sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "archive"))
sys.path.insert(0, os.path.join(BASE, "lib"))

from all_players_panel_regression import (
    build_ticker_frame, attach_fundamentals, run_regression,
    FLOW_DIR, PRICE_DIR, FORWARD_DAYS, SAMPLE_STRIDE,
)
from factor_stock_ranker import build_factor_features


def liquidity_by_ticker() -> dict:
    """Median daily traded value (VND) per ticker over its own price history —
    continuous metric for tercile-splitting for the WHOLE regression sample
    (not just a >MIN_LIQUIDITY_VND yes/no cutoff like the other script uses)."""
    out = {}
    for fpath in glob.glob(os.path.join(PRICE_DIR, "*.parquet")):
        ticker = os.path.splitext(os.path.basename(fpath))[0].upper()
        try:
            df = pd.read_parquet(fpath)
            df.columns = [c.strip().lower() for c in df.columns]
            if "close" not in df.columns or "volume" not in df.columns:
                continue
            med_to = (df["close"] * df["volume"] * 1000).median()
            if med_to > 0:
                out[ticker] = med_to
        except Exception:
            pass
    return out


def main():
    print("Computing per-ticker liquidity (median daily traded value)...")
    liq = liquidity_by_ticker()
    flow_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet"))}
    mapping = pd.read_csv(os.path.join(BASE, "ticker_sectors.csv"))
    mapping.columns = [c.strip().lower() for c in mapping.columns]
    mapping = mapping[mapping["exchange"].isin(["HOSE", "HNX"])]
    ticker_to_sector = dict(zip(mapping["ticker"].str.upper(), mapping["industry"]))

    # Use a wider net than the other script's fixed MIN_LIQUIDITY_VND cutoff —
    # we WANT the illiquid tail here, that's the point of this test. Just
    # require it actually trades (median > 0) and has flow + sector data.
    universe = sorted((set(liq) & flow_tickers) & set(ticker_to_sector))
    print(f"  {len(universe)} tickers with liquidity + flow + sector data")

    print("Building per-ticker panels...")
    frames = []
    for i, t in enumerate(universe):
        sec = ticker_to_sector.get(t)
        if sec is None or sec == "Unknown":
            continue
        f = build_ticker_frame(t, sec)
        if not f.empty:
            frames.append(f)
        if (i + 1) % 300 == 0:
            print(f"  ...{i + 1}/{len(universe)} processed, {len(frames)} usable so far")

    all_df = pd.concat(frames, ignore_index=True)
    print(f"  {all_df['ticker'].nunique()} tickers, date range "
          f"{all_df['date'].min().date()} to {all_df['date'].max().date()}")

    sample_dates = sorted(all_df["date"].unique())[::SAMPLE_STRIDE]
    panel = all_df[all_df["date"].isin(sample_dates)].copy()
    print(f"  {len(sample_dates)} periods, {len(panel):,} panel rows (pre-fundamentals)")

    print("Loading fundamentals...")
    qfeat = build_factor_features(symbols=universe)
    panel = attach_fundamentals(panel, qfeat)
    panel = panel.rename(columns={f"fwd_{FORWARD_DAYS}": "fwd_ret"})
    panel["period"] = panel["date"].dt.strftime("%Y-%m")

    # Assign liquidity tercile PER TICKER (not per row) so a ticker doesn't
    # jump terciles across its own history just from short-term volume noise.
    liq_series = pd.Series({t: liq[t] for t in panel["ticker"].unique() if t in liq})
    tercile_labels = pd.qcut(liq_series, 3, labels=["Q1 (illiquid)", "Q2 (mid)", "Q3 (liquid)"])
    panel["liq_tercile"] = panel["ticker"].map(tercile_labels)

    print("\nTercile boundaries (median daily traded value, VND):")
    for label in ["Q1 (illiquid)", "Q2 (mid)", "Q3 (liquid)"]:
        tickers_in = tercile_labels[tercile_labels == label].index
        vals = liq_series[tickers_in]
        print(f"  {label}: {len(tickers_in)} tickers, range "
              f"{vals.min()/1e9:.3f} - {vals.max()/1e9:.3f} tỷ VND/day")

    for label in ["Q1 (illiquid)", "Q2 (mid)", "Q3 (liquid)"]:
        sub = panel[panel["liq_tercile"] == label].copy()
        run_regression(sub, f"LIQUIDITY TERCILE: {label}")


if __name__ == "__main__":
    main()
