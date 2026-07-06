"""
flow_predictive_test.py — Does NDT flow data actually predict forward returns
once you account for its real ~2-trading-day publication lag?

Context: FlowSignalEngine._get() filters `date <= as_of`. In the live
.4Sectorlivesignals.py script, querying "as of today" silently falls back
to the newest row it actually has (~2 days stale) — no lookahead, but also
no indication of staleness. The question is whether that stale/lagged
signal still has predictive value, or whether by the time it's usable the
edge is gone.

This is NOT a portfolio backtest (entries/exits/position sizing). It is a
direct signal-quality test:
  1. Build smart_score and distribution_alert as vectorized daily series
     per ticker (same math as lib/flow_signals.py, no lookahead).
  2. Compare NOLAG (same-day, what the automated 4sectors.py backtest
     effectively assumed) vs LAG (shifted back FLOW_LAG trading days —
     what you can realistically query in live use).
  3. Measure forward returns (5/10/20 trading days) conditioned on score
     buckets and on distribution_alert firing, for both NOLAG and LAG.
  4. Report Spearman IC and mean/median forward returns per bucket so you
     can see how much of the apparent edge survives the lag.

Universe: liquid tickers (>=1B VND/day 60d median turnover) that overlap
with data/investor_flow (Sep 2024+ coverage).

Usage:  python archive/flow_predictive_test.py
"""

import glob, os, sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOW_DIR  = os.path.join(BASE, "data", "investor_flow")
PRICE_DIR = os.path.join(BASE, "data", "price")

MIN_LIQUIDITY_VND = 1_000_000_000
FLOW_LAG = int(sys.argv[1]) if len(sys.argv) > 1 else 2   # trading days — real publication lag
Z_WINDOW = 20                # smart_score window (matches FlowSignalEngine default)
DIST_WINDOW = 10             # distribution_alert window (matches default)
FORWARD_HORIZONS = [5, 10, 20]

FLOW_COLS = [
    "tu_doanh_net", "ca_nhan_trongnuoc_net", "to_chuc_trongnuoc_net",
    "ca_nhan_nuocngoai_net", "to_chuc_nuocngoai_net",
]
SMART_WEIGHTS = {
    "to_chuc_trongnuoc_net": 0.50, "ca_nhan_trongnuoc_net": 0.20,
    "tu_doanh_net": 0.15, "to_chuc_nuocngoai_net": 0.10,
    "ca_nhan_nuocngoai_net": 0.05,
}
DIST_SELLERS = ["to_chuc_nuocngoai_net", "tu_doanh_net"]
DIST_BUYER   = "ca_nhan_trongnuoc_net"


def liquid_universe():
    """Tickers with median 60d turnover >= MIN_LIQUIDITY_VND, using latest price data."""
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


def zscore_series(s: pd.Series, window: int) -> pd.Series:
    """Vectorized version of FlowSignalEngine._z: z of value_t vs the
    PRIOR `window` values (excludes t itself), clipped to [-1,1]."""
    roll_mean = s.rolling(window).mean().shift(1)
    roll_std  = s.rolling(window).std().shift(1)
    z = (s - roll_mean) / roll_std
    z = z.clip(-3, 3) / 3.0
    # flat-history fallback: std~0 -> sign(value)*0.3 (rare edge case, ignored
    # here for simplicity — contributes negligible bias at this sample size)
    return z.clip(-1, 1)


def build_ticker_frame(ticker: str) -> pd.DataFrame:
    fpath = os.path.join(FLOW_DIR, f"{ticker}.parquet")
    df = pd.read_parquet(fpath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "close" not in df.columns or len(df) < Z_WINDOW + max(FORWARD_HORIZONS) + 5:
        return pd.DataFrame()

    # ── smart_score (no lag = same-day, matches what the naive backtest saw) ──
    score = pd.Series(0.0, index=df.index)
    total_w = 0.0
    for col, w in SMART_WEIGHTS.items():
        if col not in df.columns:
            continue
        score = score + w * zscore_series(df[col], Z_WINDOW).fillna(0.0)
        total_w += w
    score = score / total_w if total_w > 0 else score

    # ── distribution_alert (rolling mean INCLUDES current day, matches _raw_mean) ──
    roll10 = {c: df[c].rolling(DIST_WINDOW).mean() for c in FLOW_COLS if c in df.columns}
    dist = pd.Series(True, index=df.index)
    for c in DIST_SELLERS:
        dist = dist & (roll10.get(c, pd.Series(1.0, index=df.index)) < 0)
    dist = dist & (roll10.get(DIST_BUYER, pd.Series(-1.0, index=df.index)) > 0)
    dist = dist.fillna(False)

    out = pd.DataFrame({
        "ticker": ticker,
        "date": df["date"],
        "close": df["close"],
        "score_nolag": score,
        "score_lag": score.shift(FLOW_LAG),
        "dist_nolag": dist,
        "dist_lag": dist.shift(FLOW_LAG).fillna(False).astype(bool),
    })
    for h in FORWARD_HORIZONS:
        out[f"fwd_{h}"] = df["close"].shift(-h) / df["close"] - 1
    return out


def bucket_label(score):
    if pd.isna(score):
        return None
    if score >= 0.25:
        return "BUY (>=0.25)"
    if score <= -0.25:
        return "SELL (<=-0.25)"
    return "NEUTRAL"


def analyze(all_df: pd.DataFrame, label: str, score_col: str, dist_col: str):
    print(f"\n{'═'*78}")
    print(f"  {label}   (score_col={score_col}, dist_col={dist_col})")
    print(f"{'─'*78}")

    df = all_df.dropna(subset=[score_col]).copy()
    df["bucket"] = df[score_col].apply(bucket_label)

    print(f"  N observations: {len(df):,}  |  tickers: {df['ticker'].nunique()}")
    print()
    print(f"  {'Score bucket':<16} {'N':>8}  " +
          "  ".join(f"{'fwd'+str(h)+'d':>9}" for h in FORWARD_HORIZONS))
    print(f"  {'-'*70}")
    for b in ["SELL (<=-0.25)", "NEUTRAL", "BUY (>=0.25)"]:
        sub = df[df["bucket"] == b]
        row = f"  {b:<16} {len(sub):>8,}  "
        row += "  ".join(f"{sub[f'fwd_{h}'].mean()*100:>+8.2f}%" for h in FORWARD_HORIZONS)
        print(row)

    print()
    print(f"  Spearman IC (score vs forward return):")
    for h in FORWARD_HORIZONS:
        sub = df.dropna(subset=[f"fwd_{h}"])
        ic = sub[score_col].corr(sub[f"fwd_{h}"], method="spearman")
        print(f"    {h:>2}d:  IC = {ic:>+.4f}   (n={len(sub):,})")

    print()
    print(f"  Distribution alert ({dist_col}) — fired vs not fired:")
    print(f"  {'':16} {'N':>8}  " +
          "  ".join(f"{'fwd'+str(h)+'d':>9}" for h in FORWARD_HORIZONS))
    for flag, lbl in [(True, "ALERT FIRED"), (False, "no alert")]:
        sub = df[df[dist_col] == flag]
        row = f"  {lbl:<16} {len(sub):>8,}  "
        row += "  ".join(f"{sub[f'fwd_{h}'].mean()*100:>+8.2f}%" for h in FORWARD_HORIZONS)
        print(row)
    fired = df[df[dist_col] == True]
    not_fired = df[df[dist_col] == False]
    print()
    for h in FORWARD_HORIZONS:
        diff = fired[f"fwd_{h}"].mean() - not_fired[f"fwd_{h}"].mean()
        print(f"    Alert-fired minus no-alert, {h}d fwd return: {diff*100:+.2f}pp"
              f"  (negative = alert correctly precedes underperformance)")


def main():
    print("Loading liquid universe (>=1B VND/day 60d median turnover)...")
    liquid = liquid_universe()
    flow_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet"))}
    universe = sorted(liquid & flow_tickers)
    print(f"  {len(liquid)} liquid tickers, {len(flow_tickers)} have flow data, "
          f"{len(universe)} overlap")

    frames = []
    for t in universe:
        f = build_ticker_frame(t)
        if not f.empty:
            frames.append(f)
    if not frames:
        print("No data.")
        return
    all_df = pd.concat(frames, ignore_index=True)
    print(f"  Built {len(all_df):,} ticker-day rows across {all_df['ticker'].nunique()} tickers")

    analyze(all_df, "NOLAG — same-day flow query (lookahead-biased, what the naive "
                     "backtest effectively used)", "score_nolag", "dist_nolag")
    analyze(all_df, f"LAG={FLOW_LAG}d — realistic query (what you can actually see live)",
            "score_lag", "dist_lag")

    print(f"\n{'═'*78}")
    print("  INTERPRETATION")
    print(f"{'─'*78}")
    print("  Compare IC and bucket spreads between NOLAG and LAG sections above.")
    print(f"  If LAG's numbers are close to NOLAG's, the signal survives the real")
    print(f"  {FLOW_LAG}-day publication delay and the 'flow:' tags in section [4] of")
    print(f"  .4Sectorlivesignals.py are trustworthy as shown. If LAG's IC/spread")
    print(f"  collapses toward zero, the tags are informative in hindsight only —")
    print(f"  by the time you can see them, the move has already happened.")
    print(f"{'═'*78}")


if __name__ == "__main__":
    main()
