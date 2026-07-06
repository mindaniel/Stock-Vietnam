"""
tick_contrarian_backtest.py — Long-only, T+2.5-constrained backtest of the
tick-tape contrarian idea (buy stocks with heavy same-day aggressive SELLING,
on the theory that domestic institutions are quietly accumulating into it).

Rules enforced (per user instruction — this is not the free long/short IC
test from tick_contrarian_test.py):
  - LONG ONLY. No shorting the "most bought" side — VN retail accounts
    can't short. We only ever buy candidates and later sell them.
  - T+2.5 settlement: buy on day T's OPEN, earliest sell is the afternoon
    of T+2 (2 full trading days later) — approximated with daily bars as
    exit no earlier than entry_idx + 2 (i.e. 3 bars held: T, T+1, T+2).
  - Signal computed from tick data at the close of day T-1 (fully known,
    no lookahead) -> entry at T's open (next available print).
  - Round-trip friction: 0.25% each way (matches archive/4sectors.py's
    BROKER_FEE + SLIPPAGE), applied on both entry and exit.

Strategy: each day, rank stocks by tick_net_norm (aggressive buy minus
aggressive sell value, normalised by day's traded value). Buy the bottom
decile (most heavily tick-sold) equal-weight, hold for a fixed N trading
days (N >= 2, satisfying T+2.5), then sell. Compare against buying the
full universe equal-weight over the same days (baseline — what you'd get
with no signal at all).

Caveat: tick_data covers only ~2 months (57 trading days) — small sample,
directional read only.

Usage:  python archive/tick_contrarian_backtest.py
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

FRICTION       = 0.0025          # one-way, matches archive/4sectors.py
MIN_HOLD_DAYS  = 2               # T+2.5: earliest sell = entry_idx + 2 (afternoon of T+2)
HOLD_VARIANTS  = [2, 3, 5, 10]   # trading days held (>=2 to satisfy T+2.5)
BOTTOM_PCT     = 0.20            # buy the most heavily tick-sold 20% each day


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
    if "close" not in df.columns or "open" not in df.columns:
        return pd.DataFrame()
    df = df[(df["close"] > 0) & (df["open"] > 0)]
    df["ticker"] = ticker
    df = df.reset_index(drop=True)
    df["bar_idx"] = df.index
    return df[["date", "open", "close", "ticker", "bar_idx"]]


def main():
    tick_tickers = sorted({os.path.splitext(os.path.basename(f))[0].upper()
                            for f in glob.glob(os.path.join(TICK_DIR, "*.parquet"))})
    print(f"Loading {len(tick_tickers)} tickers with tick data...")

    tick_frames, price_frames = [], {}
    for t in tick_tickers:
        try:
            tick_df = daily_tick_net(t)
            price_df = load_prices(t)
            if tick_df.empty or price_df.empty:
                continue
            tick_frames.append(tick_df)
            price_frames[t] = price_df.set_index("date")
        except Exception:
            continue

    all_tick = pd.concat(tick_frames, ignore_index=True)
    print(f"Signal available for {all_tick['ticker'].nunique()} tickers, "
          f"{all_tick['date'].nunique()} trading days")

    print(f"\n{'='*90}")
    print(f"  LONG-ONLY, T+2.5-CONSTRAINED BACKTEST")
    print(f"  Buy bottom {BOTTOM_PCT*100:.0f}% by tick_net_norm each day, hold N trading "
          f"days (N>=2 per T+2.5), {FRICTION*100:.2f}% friction each way")
    print(f"{'-'*90}")
    print(f"  {'Hold(d)':>8} {'Signal avg ret':>15} {'Baseline avg ret':>17} "
          f"{'Excess':>9} {'Win rate':>9} {'N trades':>9}")
    print(f"  {'-'*88}")

    for hold in HOLD_VARIANTS:
        if hold < MIN_HOLD_DAYS:
            continue
        sig_rets, base_rets = [], []
        for date, day_group in all_tick.groupby("date"):
            day_group = day_group.dropna(subset=["tick_net_norm"])
            if len(day_group) < 10:
                continue
            n_buy = max(1, int(len(day_group) * BOTTOM_PCT))
            candidates = day_group.nsmallest(n_buy, "tick_net_norm")["ticker"].tolist()
            universe = day_group["ticker"].tolist()

            for group, out_list in [(candidates, sig_rets), (universe, base_rets)]:
                for ticker in group:
                    pdf = price_frames.get(ticker)
                    if pdf is None or date not in pdf.index:
                        continue
                    entry_bar = pdf.loc[date, "bar_idx"]
                    entry_bars = pdf[pdf["bar_idx"] == entry_bar + 1]  # T+1 open = entry
                    exit_bars  = pdf[pdf["bar_idx"] == entry_bar + 1 + hold]
                    if entry_bars.empty or exit_bars.empty:
                        continue
                    entry_px = float(entry_bars["open"].iloc[0])
                    exit_px  = float(exit_bars["close"].iloc[0])
                    if entry_px <= 0:
                        continue
                    ret = (exit_px / entry_px - 1)
                    ret = (1 - FRICTION) * (1 + ret) * (1 - FRICTION) - 1  # buy + sell friction
                    out_list.append(ret)

        if not sig_rets:
            continue
        sig_avg  = np.mean(sig_rets) * 100
        base_avg = np.mean(base_rets) * 100 if base_rets else np.nan
        win_rate = np.mean([r > 0 for r in sig_rets]) * 100
        print(f"  {hold:>8} {sig_avg:>+14.3f}% {base_avg:>+16.3f}% "
              f"{sig_avg-base_avg:>+8.3f}pp {win_rate:>8.1f}% {len(sig_rets):>9,}")

    print(f"\n{'='*90}")
    print("  CAVEAT")
    print(f"{'-'*90}")
    print(f"  Only {all_tick['date'].nunique()} trading days of tick data exist (~2 months).")
    print("  This backtest respects T+2.5 (no exit before entry+2 trading days) and is")
    print("  long-only (equal-weight buy of the signal bucket vs the full universe,")
    print("  no shorting). Friction of 0.25% is charged on both entry and exit.")
    print("  Given the short sample, treat results as directional, not conclusive —")
    print("  re-run as tick_data accumulates.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
