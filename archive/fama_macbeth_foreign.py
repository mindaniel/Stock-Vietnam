"""
fama_macbeth_foreign.py — Strip out market and sector (industry) premium from
the foreign-accumulation signal via Fama-MacBeth cross-sectional regression.

Why: the earlier sector-heatmap test found some sectors (Financial Services,
F&B for institutional; Utilities, Insurance for retail) where foreign
accumulation predicts positive forward returns. But that could just mean
"foreign investors happened to pile into the sectors that rallied during this
21-month window" — a market/sector-timing confound, not genuine stock-picking
skill. Fama-MacBeth isolates whether foreign accumulation predicts a stock's
return RELATIVE TO its own sector peers in the same period, which nets out
both the overall market move and the sector's average move that period.

Method (standard Fama-MacBeth, adapted with sector-neutralization):
  1. Sample monthly cross-sections (~20 trading days apart) to keep periods
     roughly independent — the earlier daily-overlapping tests overstate
     sample size for a formal significance test.
  2. At each period t: for every stock, compute the foreign-accumulation
     signal (60d trailing) and forward return over horizon h.
  3. RAW regression: cross-sectionally regress fwd_return ~ signal (one
     slope per period).
  4. SECTOR-NEUTRAL regression: demean both return and signal within each
     sector for that period (removes that period's market move AND each
     sector's average move), then regress the demeaned quantities.
  5. Collect the time series of per-period slopes for both regressions.
     Fama-MacBeth estimate = mean slope; t-stat = mean / (std / sqrt(T)).
     If RAW is significant but SECTOR-NEUTRAL collapses toward zero, the
     earlier finding was a sector/market-timing artifact, not stock-picking
     skill within sectors.

CAVEAT: ~21 months of data sampled monthly gives ~15-20 periods at best —
already a small sample for a t-test, and shrinks further at the 120d horizon
(fewer usable periods before running out of forward-return room). Treat
significance as directional, not a publishable result.

Usage:  python archive/fama_macbeth_foreign.py
"""

import glob, os, sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOW_DIR  = os.path.join(BASE, "data", "investor_flow")
PRICE_DIR = os.path.join(BASE, "data", "price")

MIN_LIQUIDITY_VND = 1_000_000_000
ACCUM_WINDOW = 60
FORWARD_HORIZONS = [20, 60, 120]
SAMPLE_STRIDE = 20     # trading days between cross-sections (~monthly)
MIN_STOCKS_PER_SECTOR_PERIOD = 3   # need at least this many to demean sensibly


def liquid_universe():
    liquid = set()
    for fpath in glob.glob(os.path.join(PRICE_DIR, "*.parquet")):
        ticker = os.path.splitext(os.path.basename(fpath))[0].upper()
        try:
            df = pd.read_parquet(fpath)
            df.columns = [c.strip().lower() for c in df.columns]
            if "close" not in df.columns or "volume" not in df.columns:
                continue
            med_to = (df["close"] * df["volume"] * 1000).tail(60).median()
            if med_to >= MIN_LIQUIDITY_VND:
                liquid.add(ticker)
        except Exception:
            pass
    return liquid


def build_ticker_frame(ticker: str, sector: str) -> pd.DataFrame:
    fpath = os.path.join(FLOW_DIR, f"{ticker}.parquet")
    df = pd.read_parquet(fpath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "close" not in df.columns or len(df) < ACCUM_WINDOW + max(FORWARD_HORIZONS) + 5:
        return pd.DataFrame()

    inst   = df.get("to_chuc_nuocngoai_net", pd.Series(0, index=df.index)).fillna(0)
    retail = df.get("ca_nhan_nuocngoai_net", pd.Series(0, index=df.index)).fillna(0)

    out = pd.DataFrame({"date": df["date"], "ticker": ticker, "sector": sector})
    for label, series in [("inst", inst), ("retail", retail)]:
        cum   = series.rolling(ACCUM_WINDOW).sum()
        scale = series.abs().rolling(ACCUM_WINDOW).mean() * ACCUM_WINDOW
        out[f"accum_{label}"] = cum / scale.replace(0, np.nan)
    for h in FORWARD_HORIZONS:
        out[f"fwd_{h}"] = df["close"].shift(-h) / df["close"] - 1
    out["bar_idx"] = np.arange(len(out))
    return out


def fama_macbeth(df: pd.DataFrame, signal_col: str, fwd_col: str, sector_neutral: bool):
    """Returns (mean_slope, t_stat, n_periods) across sampled cross-sections."""
    slopes = []
    sample_dates = sorted(df["date"].unique())[::SAMPLE_STRIDE]
    for d in sample_dates:
        cross = df[df["date"] == d].dropna(subset=[signal_col, fwd_col]).copy()
        if len(cross) < 15:
            continue
        if sector_neutral:
            sec_counts = cross.groupby("sector")["ticker"].transform("count")
            cross = cross[sec_counts >= MIN_STOCKS_PER_SECTOR_PERIOD]
            if len(cross) < 15:
                continue
            cross["y"] = cross[fwd_col] - cross.groupby("sector")[fwd_col].transform("mean")
            cross["x"] = cross[signal_col] - cross.groupby("sector")[signal_col].transform("mean")
        else:
            cross["y"] = cross[fwd_col]
            cross["x"] = cross[signal_col]

        x = cross["x"].values
        y = cross["y"].values
        if x.std() < 1e-9:
            continue
        # simple OLS slope (single regressor + intercept)
        xm, ym = x.mean(), y.mean()
        denom = ((x - xm) ** 2).sum()
        if denom < 1e-12:
            continue
        beta = ((x - xm) * (y - ym)).sum() / denom
        slopes.append(beta)

    if len(slopes) < 3:
        return np.nan, np.nan, len(slopes)
    slopes = np.array(slopes)
    mean_slope = slopes.mean()
    se = slopes.std(ddof=1) / np.sqrt(len(slopes))
    tstat = mean_slope / se if se > 0 else np.nan
    return mean_slope, tstat, len(slopes)


def main():
    print("Building liquid universe + sector map...")
    liquid = liquid_universe()
    flow_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet"))}

    mapping = pd.read_csv(os.path.join(BASE, "ticker_sectors.csv"))
    mapping.columns = [c.strip().lower() for c in mapping.columns]
    mapping = mapping[mapping["exchange"].isin(["HOSE", "HNX"])]
    ticker_to_sector = dict(zip(mapping["ticker"].str.upper(), mapping["industry"]))

    universe = sorted(liquid & flow_tickers)
    frames = []
    for t in universe:
        sec = ticker_to_sector.get(t)
        if sec is None or sec == "Unknown":
            continue
        f = build_ticker_frame(t, sec)
        if not f.empty:
            frames.append(f)
    all_df = pd.concat(frames, ignore_index=True)
    print(f"  {len(universe)} liquid tickers, {all_df['ticker'].nunique()} usable, "
          f"{all_df['sector'].nunique()} sectors, {len(all_df):,} ticker-day rows")

    print(f"\n{'='*95}")
    print("  FAMA-MACBETH: foreign accumulation -> forward return, RAW vs SECTOR-NEUTRAL")
    print(f"  (sampled every {SAMPLE_STRIDE} trading days, ~monthly cross-sections)")
    print(f"{'-'*95}")
    print(f"  {'Type':<8} {'Horizon':>8} {'RAW slope':>11} {'RAW t':>7} {'':>3} "
          f"{'Neutral slope':>14} {'Neutral t':>10} {'n_periods':>10}")
    print(f"  {'-'*93}")

    for label in ["inst", "retail"]:
        for h in FORWARD_HORIZONS:
            raw_slope, raw_t, n_raw = fama_macbeth(all_df, f"accum_{label}", f"fwd_{h}", sector_neutral=False)
            neu_slope, neu_t, n_neu = fama_macbeth(all_df, f"accum_{label}", f"fwd_{h}", sector_neutral=True)
            print(f"  {label:<8} {h:>7}d {raw_slope:>+11.4f} {raw_t:>+7.2f} {'':>3} "
                  f"{neu_slope:>+14.4f} {neu_t:>+10.2f} {n_neu:>10}")

    print(f"\n{'='*95}")
    print("  INTERPRETATION")
    print(f"{'-'*95}")
    print("  RAW slope/t = does the signal predict returns at all (market + sector + stock")
    print("  effects all mixed in). Sector-neutral slope/t = same test AFTER removing each")
    print("  period's market move and each sector's average move — isolates whether foreign")
    print("  accumulation predicts a stock beating its OWN sector peers.")
    print()
    print("  |t| > ~2 is the usual significance bar. If sector-neutral |t| collapses vs raw,")
    print("  the earlier sector-heatmap result was mostly foreign investors correctly timing")
    print("  WHICH SECTORS would rally, not picking better stocks within a sector.")
    print(f"{'='*95}")


if __name__ == "__main__":
    main()
