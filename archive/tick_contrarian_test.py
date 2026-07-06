"""
tick_contrarian_test.py — Is same-day tick aggressor imbalance a TRADEABLE
contrarian signal (buy the panic, since institutions are quietly buying it)?

Prior findings (tick_size_proxy_test.py, tick_size_band_scan.py):
  - Tick net (aggressive buy value - aggressive sell value) correlates
    NEGATIVELY with real domestic institutional net flow (r=-0.19) — days
    of aggressive tick SELLING tend to be days institutions are buying.
  - Institutional accumulation (from lib/flow_signals.py's validated
    distribution_alert / smart_score work) tends to precede outperformance.

This test closes the loop: does tick_net on day T predict forward RETURNS
over the next 1/3/5/10 days? If tick_net is negatively related to forward
returns (net selling today -> higher returns tomorrow), that's a same-day,
zero-lag, tradeable contrarian signal — no need to wait for the lagged
nguoiquansat data at all.

Caveat printed at the end: tick_data only covers ~2 months (May-Jun 2026),
so this is a directional read, not a statistically robust backtest. Treat
results as "worth monitoring longer," not "trade tomorrow."

Usage:  python archive/tick_contrarian_test.py
"""

import glob, os, sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICK_DIR  = os.path.join(BASE, "data", "tick_data")
PRICE_DIR = os.path.join(BASE, "data", "price")

FORWARD_HORIZONS = [1, 3, 5, 10]
N_QUANTILES = 5   # quintile sort for the long-short test


def daily_tick_net(ticker: str) -> pd.DataFrame:
    fpath = os.path.join(TICK_DIR, f"{ticker}.parquet")
    df = pd.read_parquet(fpath)
    df["date"] = pd.to_datetime(df["td"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date"])
    df["value_bn"] = df["p"].astype(float) * df["v"].astype(float) / 1e9
    df["signed"] = np.where(df["s"] == "buy", df["value_bn"], -df["value_bn"])
    g = df.groupby("date").agg(
        tick_net=("signed", "sum"),
        day_value_bn=("value_bn", "sum"),
    ).reset_index()
    # normalize by day's total traded value so signal is comparable across stocks
    g["tick_net_norm"] = g["tick_net"] / g["day_value_bn"].replace(0, np.nan)
    g["ticker"] = ticker
    return g


def load_prices(ticker: str) -> pd.DataFrame:
    fpath = os.path.join(PRICE_DIR, f"{ticker}.parquet")
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_parquet(fpath)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "time" if "time" in df.columns else "date"
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if "close" not in df.columns:
        return pd.DataFrame()
    df = df[df["close"] > 0]   # drop bad/zero-price rows (e.g. delisted/halted tickers)
    out = df[["date", "close"]].copy()
    for h in FORWARD_HORIZONS:
        out[f"fwd_{h}"] = out["close"].shift(-h) / out["close"] - 1
    return out


def main():
    tick_tickers = sorted({os.path.splitext(os.path.basename(f))[0].upper()
                            for f in glob.glob(os.path.join(TICK_DIR, "*.parquet"))})
    print(f"Testing {len(tick_tickers)} tickers with tick data...")

    frames = []
    for t in tick_tickers:
        try:
            tick_df = daily_tick_net(t)
            price_df = load_prices(t)
            if tick_df.empty or price_df.empty:
                continue
            m = tick_df.merge(price_df, on="date", how="inner")
            if not m.empty:
                frames.append(m)
        except Exception:
            continue

    if not frames:
        print("No data.")
        return
    all_df = pd.concat(frames, ignore_index=True)
    print(f"Merged {len(all_df):,} ticker-day rows, {all_df['ticker'].nunique()} tickers, "
          f"{all_df['date'].nunique()} trading days "
          f"({all_df['date'].min().date()} to {all_df['date'].max().date()})")

    print(f"\n{'='*80}")
    print("  Spearman IC: tick_net_norm (day T) vs forward return")
    print(f"{'-'*80}")
    for h in FORWARD_HORIZONS:
        sub = all_df.dropna(subset=["tick_net_norm", f"fwd_{h}"])
        ic = sub["tick_net_norm"].corr(sub[f"fwd_{h}"], method="spearman")
        print(f"  {h:>2}d fwd:  IC = {ic:>+.4f}   (n={len(sub):,})  "
              f"{'[negative = contrarian works]' if h == FORWARD_HORIZONS[0] else ''}")

    print(f"\n{'='*80}")
    print(f"  Quintile sort: rank stocks each day by tick_net_norm (Q1=most sold, "
          f"Q{N_QUANTILES}=most bought), mean forward return per quintile")
    print(f"{'-'*80}")
    for h in FORWARD_HORIZONS:
        sub = all_df.dropna(subset=["tick_net_norm", f"fwd_{h}"]).copy()
        if sub.empty:
            continue
        sub["q"] = sub.groupby("date")["tick_net_norm"].transform(
            lambda x: pd.qcut(x, N_QUANTILES, labels=False, duplicates="drop") if x.nunique() >= N_QUANTILES else np.nan)
        sub = sub.dropna(subset=["q"])
        means = sub.groupby("q")[f"fwd_{h}"].mean() * 100
        q_low  = means.get(0.0, np.nan)
        q_high = means.get(float(N_QUANTILES - 1), np.nan)
        spread = q_low - q_high  # long-most-sold, short-most-bought
        print(f"  {h:>2}d fwd:  Q1(most sold)={q_low:>+6.2f}%   "
              f"Q{N_QUANTILES}(most bought)={q_high:>+6.2f}%   "
              f"long-short spread={spread:>+6.2f}pp  (n={len(sub):,})")

    print(f"\n{'='*80}")
    print("  CAVEAT")
    print(f"{'-'*80}")
    print(f"  Only {all_df['date'].nunique()} trading days of tick data exist "
          f"(~2 months). This is a")
    print("  directional read of whether the idea has legs, not a statistically")
    print("  robust backtest — treat as 'worth tracking as data accumulates,' not")
    print("  'trade this tomorrow.' Re-run this script periodically as tick_data")
    print("  grows; the IC/spread numbers should stabilize (or fall apart) with")
    print("  more days.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
