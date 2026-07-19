"""
player_clustering_analysis.py — Does each player CLUSTER their buying
around a specific point in the ride (conviction, concentrated timing) or
SPREAD it out evenly across the whole move (indiscriminate accumulation)?
And does clustered buying actually land at a better (lower) price within
the ride, and precede bigger gains — testing the hypothesis that spread-out
buyers pay a less "appealing" average price?

Reuses player_sequencing_analysis.py's ride detection (swing-low to
swing-high, price-defined, no fixed calendar window) and universe-building.
For each (ride, player), computes THREE numbers instead of just one timing
estimate:

  1. center_of_mass  — same as before: flow-weighted mean day (0=low, 1=high)
  2. dispersion       — flow-weighted STANDARD DEVIATION of day, normalized
                         to [0,1]. Low dispersion = buying concentrated in a
                         tight window (conviction); high dispersion = spread
                         evenly across the whole ride (indiscriminate).
  3. entry_price_pct — flow-weighted average CLOSE PRICE paid, expressed as
                         a percentile of the ride's own low-high range
                         (0 = bought exactly at the low, 1 = bought exactly
                         at the high). This is the direct "how appealing was
                         their average price" measure the hypothesis is
                         actually about — dispersion is a proxy, this is the
                         real thing.

Then: correlate dispersion with entry_price_pct (does spreading out really
mean a worse average price?), and compare ride outcomes (gain_pct) between
concentrated vs spread-out buyers, per player.

Usage:  python archive/player_clustering_analysis.py
"""

import glob, os, sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "archive"))
from player_sequencing_analysis import (
    find_rides, liquid_universe, PLAYER_COLS, FLOW_DIR, PRICE_DIR,
)

OUT_PATH = os.path.join(BASE, "backtest_reports", "player_clustering_rides.csv")


def flow_weighted_stats(flow: pd.Series, close: pd.Series) -> dict:
    """center_of_mass, dispersion (both normalized day-index), and
    entry_price_pct (percentile of close within [close.min, close.max] over
    the window), all weighted by NET-BUYING days only."""
    buys = flow.clip(lower=0)
    total = buys.sum()
    if total <= 0:
        return {"com": np.nan, "dispersion": np.nan, "entry_price_pct": np.nan}

    n = len(flow)
    idx = np.arange(n)
    w = buys.values
    com_raw = (idx * w).sum() / total
    var_raw = ((idx - com_raw) ** 2 * w).sum() / total
    denom = max(n - 1, 1)
    com = com_raw / denom
    dispersion = np.sqrt(var_raw) / denom

    lo, hi = close.min(), close.max()
    if hi > lo:
        avg_price = (close.values * w).sum() / total
        entry_price_pct = (avg_price - lo) / (hi - lo)
    else:
        entry_price_pct = np.nan

    return {"com": com, "dispersion": dispersion, "entry_price_pct": entry_price_pct}


def main():
    print("Building universe...")
    liquid = liquid_universe()
    flow_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet"))}
    mapping = pd.read_csv(os.path.join(BASE, "ticker_sectors.csv"))
    mapping.columns = [c.strip().lower() for c in mapping.columns]
    mapping = mapping[mapping["exchange"].isin(["HOSE", "HNX"])]
    ticker_to_sector = dict(zip(mapping["ticker"].str.upper(), mapping["industry"]))
    universe = sorted(liquid & flow_tickers & set(ticker_to_sector))
    print(f"  {len(universe)} tickers")

    rows = []
    for i, ticker in enumerate(universe):
        fpath = os.path.join(FLOW_DIR, f"{ticker}.parquet")
        if not os.path.exists(fpath):
            continue
        flow = pd.read_parquet(fpath)
        flow["date"] = pd.to_datetime(flow["date"])
        flow = flow.sort_values("date").reset_index(drop=True)
        if "close" in flow.columns:
            flow = flow.drop(columns=["close"])  # investor_flow has its own close; price's is authoritative here

        price_path = os.path.join(PRICE_DIR, f"{ticker}.parquet")
        if not os.path.exists(price_path):
            continue
        price = pd.read_parquet(price_path)
        price.columns = [c.strip().lower() for c in price.columns]
        date_col = "time" if "time" in price.columns else "date"
        price["date"] = pd.to_datetime(price[date_col])
        price = price.sort_values("date").reset_index(drop=True)
        if not {"high", "low", "close"}.issubset(price.columns):
            continue

        df = flow.merge(price[["date", "high", "low", "close"]], on="date", how="inner")
        if len(df) < 40:
            continue

        rides = find_rides(df)
        if not rides:
            continue
        sector = ticker_to_sector.get(ticker)

        for low_idx, high_idx, gain in rides:
            if not np.isfinite(gain):
                continue  # same bad-data guard as player_sequencing_analysis.py
            ride_id = f"{ticker}_{df['date'].iloc[low_idx].date()}"
            close_window = df["close"].iloc[low_idx:high_idx + 1]
            for label, col in PLAYER_COLS.items():
                if col not in df.columns:
                    continue
                flow_window = df[col].iloc[low_idx:high_idx + 1].fillna(0)
                stats = flow_weighted_stats(flow_window, close_window)
                rows.append({
                    "ride_id": ride_id, "ticker": ticker, "sector": sector,
                    "gain_pct": gain, "duration_days": high_idx - low_idx,
                    "player": label, **stats,
                })

        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(universe)} tickers processed")

    df = pd.DataFrame(rows).dropna(subset=["com", "dispersion", "entry_price_pct"])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n{len(df):,} (ride, player) rows, saved to {OUT_PATH}")

    print(f"\n{'='*95}")
    print("  DISPERSION vs ENTRY PRICE — does spreading out mean paying a worse price?")
    print(f"{'-'*95}")
    for player, g in df.groupby("player"):
        corr = g["dispersion"].corr(g["entry_price_pct"])
        print(f"  {player:<24} corr(dispersion, entry_price_pct) = {corr:+.3f}   n={len(g)}")

    print(f"\n{'='*95}")
    print("  CONCENTRATED (bottom tercile dispersion) vs SPREAD-OUT (top tercile), per player")
    print(f"{'-'*95}")
    print(f"  {'Player':<24} {'Conc. price%':>13} {'Spread price%':>14} {'Conc. gain':>11} {'Spread gain':>12}")
    for player, g in df.groupby("player"):
        terciles = pd.qcut(g["dispersion"], 3, labels=["low", "mid", "high"], duplicates="drop")
        conc = g[terciles == "low"]
        spread = g[terciles == "high"]
        print(f"  {player:<24} {conc['entry_price_pct'].mean():>12.1%} {spread['entry_price_pct'].mean():>13.1%} "
              f"{conc['gain_pct'].mean():>10.1%} {spread['gain_pct'].mean():>11.1%}")

    print(f"\n{'='*95}")
    print("  OVERALL: entry_price_pct and dispersion summary by player")
    print(f"{'-'*95}")
    print(df.groupby("player")[["com", "dispersion", "entry_price_pct"]].agg(["mean", "median"]).round(3))


if __name__ == "__main__":
    main()
