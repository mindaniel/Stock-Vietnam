"""
foreign_longterm_test.py — If foreign investors are aggressive/momentum-driven
and visibly move price (established earlier: positive IC with past returns,
positive correlation with closing above session VWAP), they can't realistically
be scalping for short-term edge — everyone can see them coming. The thesis this
tests: their profit motive is a longer-horizon POSITION (quarters, not days),
and the aggressive day-to-day execution is just how they build/exit that
position, not the source of their edge.

Method: build a "sustained foreign accumulation" signal — trailing N-day
cumulative foreign net flow (institutional + retail foreign combined),
normalized by trailing turnover — and test it against LONG forward return
horizons (60/120/180/250 trading days, i.e. ~3/6/9/12 months), not the
short 1-20 day horizons tested earlier. If the long-term positioning thesis
is right, this should show a MUCH stronger signal at long horizons than the
short-horizon flow score did.

Also runs a long-only, T+2.5-respecting backtest: buy the stocks with the
strongest trailing foreign accumulation, hold for the matching long horizon,
vs a full-universe baseline.

CAVEAT UP FRONT: investor_flow data only covers Sep 2024-present (~22 months).
A 12-month forward-return test on ~22 months of data has only ~1-2 truly
independent (non-overlapping) windows — this is a directional read, not a
statistically robust result. Treat accordingly.

Usage:  python archive/foreign_longterm_test.py
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
ACCUM_WINDOWS = [60, 120]          # trading days of trailing accumulation to test as signal
FORWARD_HORIZONS = [20, 60, 120, 180, 250]   # ~1/3/6/9/12 months
FRICTION      = 0.0025
MIN_HOLD_DAYS = 2                  # T+2.5


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


def build_ticker_frame(ticker: str) -> pd.DataFrame:
    fpath = os.path.join(FLOW_DIR, f"{ticker}.parquet")
    df = pd.read_parquet(fpath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "close" not in df.columns or len(df) < max(ACCUM_WINDOWS) + max(FORWARD_HORIZONS) + 5:
        return pd.DataFrame()

    # "Foreigners" = institutional + retail foreign combined
    foreign_net = df.get("to_chuc_nuocngoai_net", 0).fillna(0) + df.get("ca_nhan_nuocngoai_net", 0).fillna(0)

    out = pd.DataFrame({"date": df["date"], "ticker": ticker, "close": df["close"]})
    # turnover proxy: use own traded value if we had volume; investor_flow lacks
    # volume, so normalize by trailing rolling std of foreign_net itself instead
    # (self-normalizing z-score, comparable across tickers of different size)
    for w in ACCUM_WINDOWS:
        cum = foreign_net.rolling(w).sum()
        scale = foreign_net.abs().rolling(w).mean() * w
        out[f"accum_{w}"] = cum / scale.replace(0, np.nan)
    for h in FORWARD_HORIZONS:
        out[f"fwd_{h}"] = df["close"].shift(-h) / df["close"] - 1
    out["bar_idx"] = np.arange(len(out))
    out["open"] = df["close"]  # investor_flow has no separate open; use close as entry proxy
    return out


def main():
    print("Building liquid universe...")
    liquid = liquid_universe()
    flow_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet"))}
    universe = sorted(liquid & flow_tickers)
    print(f"  {len(universe)} liquid tickers with flow data")

    frames = []
    for t in universe:
        f = build_ticker_frame(t)
        if not f.empty:
            frames.append(f)
    all_df = pd.concat(frames, ignore_index=True)
    print(f"  Built {len(all_df):,} ticker-day rows across {all_df['ticker'].nunique()} tickers")

    print(f"\n{'='*90}")
    print("  PART 1: IC of trailing foreign accumulation vs forward return, by horizon")
    print(f"{'-'*90}")
    for w in ACCUM_WINDOWS:
        print(f"\n  Signal: trailing {w}-day foreign net accumulation (normalized)")
        print(f"  {'Fwd horizon':>14} {'IC':>9} {'n':>9}")
        for h in FORWARD_HORIZONS:
            sub = all_df.dropna(subset=[f"accum_{w}", f"fwd_{h}"])
            ic = sub[f"accum_{w}"].corr(sub[f"fwd_{h}"], method="spearman")
            print(f"  {h:>10}d {ic:>+12.4f} {len(sub):>9,}")

    print(f"\n{'='*90}")
    print("  PART 2: Long-only backtest — buy top-quintile foreign accumulators, hold N days")
    print("  T+2.5 respected (entry+2 min), 0.25% friction each way, vs full-universe baseline")
    print(f"{'-'*90}")

    for w in ACCUM_WINDOWS:
        print(f"\n  Signal window: {w}d trailing foreign accumulation")
        print(f"  {'Hold':>6} {'Signal avg ret':>15} {'Baseline avg ret':>17} {'Excess':>9} {'Win%':>7} {'N':>8}")
        for h in FORWARD_HORIZONS:
            if h < MIN_HOLD_DAYS:
                continue
            sig_rets, base_rets = [], []
            # sample monthly (every ~20 trading days) per ticker to reduce
            # overlap/autocorrelation in this small sample, not daily
            for ticker, g in all_df.groupby("ticker"):
                g = g.reset_index(drop=True)
                for i in range(0, len(g) - h - 1, 20):
                    row = g.iloc[i]
                    sig = row[f"accum_{w}"]
                    if pd.isna(sig):
                        continue
                    entry_idx = i + 1
                    exit_idx  = entry_idx + h
                    if exit_idx >= len(g):
                        continue
                    entry_px = g.iloc[entry_idx]["open"]
                    exit_px  = g.iloc[exit_idx]["close"]
                    if entry_px <= 0:
                        continue
                    ret = exit_px / entry_px - 1
                    ret = (1 - FRICTION) * (1 + ret) * (1 - FRICTION) - 1
                    base_rets.append((ticker, row["date"], sig, ret))

            if not base_rets:
                continue
            bdf = pd.DataFrame(base_rets, columns=["ticker", "date", "sig", "ret"])
            # cross-sectional quintile by date to keep it fair (compare same-period stocks)
            bdf["q"] = bdf.groupby("date")["sig"].transform(
                lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") if x.nunique() >= 5 else np.nan)
            top_q = bdf["q"].max()
            sig_ret  = bdf[bdf["q"] == top_q]["ret"]
            base_ret = bdf["ret"]
            if len(sig_ret) == 0:
                continue
            print(f"  {h:>5}d {sig_ret.mean()*100:>+14.2f}% {base_ret.mean()*100:>+16.2f}% "
                  f"{(sig_ret.mean()-base_ret.mean())*100:>+8.2f}pp {np.mean(sig_ret>0)*100:>6.1f}% "
                  f"{len(sig_ret):>8,}")

    print(f"\n{'='*90}")
    print("  CAVEAT")
    print(f"{'-'*90}")
    print("  Only ~22 months of investor_flow history exist. The 250-day (~12mo) horizon")
    print("  has essentially 1-2 independent windows once you account for overlap — treat")
    print("  long-horizon results as a directional read, not a validated edge. Re-run as")
    print("  more history accumulates before sizing any real position on this.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
