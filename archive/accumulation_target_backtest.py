"""
accumulation_target_backtest.py — Test the "ride the smart money" thesis
properly: it's a zero-sum game, so if domestic institutions are patient
value buyers accumulating into weakness, the trade isn't "buy today, sell
in N days" — it's "buy when they're accumulating, hold while they keep
accumulating, sell when THEY start distributing (taking profit off others)
or a price target is hit."

This uses the full Sep 2024-present investor_flow history (not the ~2-month
tick_data window — much bigger sample), because the entry/exit signals here
are the LAGGED-BUT-VALIDATED FlowSignalEngine signals (accumulation_signal /
distribution_alert), not raw same-day tick data.

Rules:
  - LONG ONLY.
  - T+2.5 settlement: buy at T+1 open (signal known at T's close, lag
    already applied), earliest sell = entry_idx + 2 trading days.
  - ENTRY: accumulation_signal fires (domestic institutional z-score > 0.2
    AND no active distribution alert), using FLOW_LAG_DAYS-lagged data —
    i.e. only using flow info that would actually be available live.
  - EXIT (first of):
      1. distribution_alert fires (institutions handing off to retail) —
         once past the T+2.5 minimum hold.
      2. Price target hit (tests +15% and +25%).
      3. Max hold timeout (safety net so trades don't run forever).
  - One position per ticker at a time (no pyramiding/overlap).
  - Friction 0.25% each way.

BASELINE (controls for the stock's own drift): for every realized signal
trade (ticker, hold_length), also compute K random "placebo" entries into
the SAME ticker held for the SAME number of days, at otherwise random
dates. This tests whether the specific TIMING of the accumulation signal
adds value versus just owning the stock for that long at a random time.

Usage:  python archive/accumulation_target_backtest.py
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
FLOW_LAG      = 2          # trading days — realistic publication lag
Z_WINDOW      = 20
DIST_WINDOW   = 10
ACCUM_Z_THRESHOLD = 0.2    # matches FlowSignalEngine.accumulation_signal()
MIN_HOLD_DAYS = 2          # T+2.5
MAX_HOLD_DAYS = 60         # safety-net timeout (~3 months)
TP_VARIANTS   = [0.15, 0.25]
N_PLACEBO     = 3          # random same-ticker, same-hold-length comparisons per trade

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
    roll_mean = s.rolling(window).mean().shift(1)
    roll_std  = s.rolling(window).std().shift(1)
    z = (s - roll_mean) / roll_std
    return (z.clip(-3, 3) / 3.0).clip(-1, 1)


def build_ticker_frame(ticker: str):
    flow_path  = os.path.join(FLOW_DIR, f"{ticker}.parquet")
    price_path = os.path.join(PRICE_DIR, f"{ticker}.parquet")
    if not os.path.exists(flow_path) or not os.path.exists(price_path):
        return None

    fdf = pd.read_parquet(flow_path)
    fdf["date"] = pd.to_datetime(fdf["date"])
    fdf = fdf.sort_values("date").reset_index(drop=True)
    if len(fdf) < Z_WINDOW + MAX_HOLD_DAYS + 5:
        return None

    pdf = pd.read_parquet(price_path)
    pdf.columns = [c.strip().lower() for c in pdf.columns]
    date_col = "time" if "time" in pdf.columns else "date"
    pdf["date"] = pd.to_datetime(pdf[date_col], errors="coerce")
    pdf = pdf.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if "open" not in pdf.columns or "close" not in pdf.columns:
        return None
    pdf = pdf[(pdf["open"] > 0) & (pdf["close"] > 0)]

    # ── domestic-institutional z-score (drives accumulation_signal) ──
    dom_z = zscore_series(fdf["to_chuc_trongnuoc_net"], Z_WINDOW).fillna(0.0)

    # ── distribution_alert (rolling mean includes current day) ──
    roll10 = {c: fdf[c].rolling(DIST_WINDOW).mean() for c in FLOW_COLS if c in fdf.columns}
    dist = pd.Series(True, index=fdf.index)
    for c in DIST_SELLERS:
        dist = dist & (roll10.get(c, pd.Series(1.0, index=fdf.index)) < 0)
    dist = dist & (roll10.get(DIST_BUYER, pd.Series(-1.0, index=fdf.index)) > 0)
    dist = dist.fillna(False)

    accum = (dom_z > ACCUM_Z_THRESHOLD) & (~dist)

    fdf["dom_z"] = dom_z
    fdf["dist"]  = dist
    fdf["accum"] = accum
    # apply realistic publication lag: what you'd actually see live on day T
    fdf["dist_lag"]  = fdf["dist"].shift(FLOW_LAG).fillna(False).astype(bool)
    fdf["accum_lag"] = fdf["accum"].shift(FLOW_LAG).fillna(False).astype(bool)

    merged = pdf.merge(fdf[["date", "accum_lag", "dist_lag"]], on="date", how="inner")
    merged = merged.reset_index(drop=True)
    merged["bar_idx"] = merged.index
    return merged


def simulate_ticker(df: pd.DataFrame, tp_pct: float, rng: np.random.Generator):
    """Returns (trades, placebo_returns) for one ticker."""
    n = len(df)
    open_arr  = df["open"].values
    close_arr = df["close"].values
    accum     = df["accum_lag"].values
    dist      = df["dist_lag"].values

    trades = []
    i = 0
    while i < n - 1:
        if not accum[i]:
            i += 1
            continue
        entry_idx = i + 1
        if entry_idx >= n:
            break
        entry_px = open_arr[entry_idx]
        if entry_px <= 0:
            i += 1
            continue

        exit_idx = None
        exit_reason = None
        j = entry_idx + MIN_HOLD_DAYS
        while j < n and j <= entry_idx + MAX_HOLD_DAYS:
            ret_so_far = close_arr[j] / entry_px - 1
            if dist[j]:
                exit_idx, exit_reason = j, "distribution"
                break
            if ret_so_far >= tp_pct:
                exit_idx, exit_reason = j, "target"
                break
            j += 1
        if exit_idx is None:
            exit_idx = min(entry_idx + MAX_HOLD_DAYS, n - 1)
            exit_reason = "timeout"

        exit_px = close_arr[exit_idx]
        ret = exit_px / entry_px - 1
        trades.append({
            "entry_idx": entry_idx, "exit_idx": exit_idx,
            "hold_days": exit_idx - entry_idx,
            "ret": ret, "reason": exit_reason,
        })
        i = exit_idx + 1   # no overlapping positions

    # placebo: same ticker, same hold length, random start
    placebo_rets = []
    for tr in trades:
        h = tr["hold_days"]
        if h <= 0 or h >= n - 1:
            continue
        valid_starts = np.arange(0, n - 1 - h)
        if len(valid_starts) == 0:
            continue
        picks = rng.choice(valid_starts, size=min(N_PLACEBO, len(valid_starts)), replace=False)
        for p in picks:
            e_px = open_arr[p]
            x_px = close_arr[p + h]
            if e_px > 0:
                placebo_rets.append(x_px / e_px - 1)

    return trades, placebo_rets


def main():
    print("Building liquid universe...")
    liquid = liquid_universe()
    flow_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet"))}
    universe = sorted(liquid & flow_tickers)
    print(f"  {len(universe)} liquid tickers with flow data")

    rng = np.random.default_rng(42)
    FRICTION = 0.0025

    for tp in TP_VARIANTS:
        print(f"\n{'='*90}")
        print(f"  TARGET = +{tp*100:.0f}%   (T+2.5 enforced, long-only, "
              f"{FRICTION*100:.2f}% friction each way, lag={FLOW_LAG}d)")
        print(f"{'-'*90}")

        all_trades, all_placebo = [], []
        reason_counts = {}
        for t in universe:
            df = build_ticker_frame(t)
            if df is None or df.empty:
                continue
            trades, placebo = simulate_ticker(df, tp, rng)
            for tr in trades:
                tr["ticker"] = t
                reason_counts[tr["reason"]] = reason_counts.get(tr["reason"], 0) + 1
            all_trades.extend(trades)
            all_placebo.extend(placebo)

        if not all_trades:
            print("  No trades generated.")
            continue

        rets = np.array([(1 - FRICTION) * (1 + tr["ret"]) * (1 - FRICTION) - 1
                          for tr in all_trades])
        placebo_arr = np.array([(1 - FRICTION) * (1 + r) * (1 - FRICTION) - 1
                                 for r in all_placebo])

        print(f"  N signal trades:      {len(rets):,}")
        print(f"  Exit reasons:          " + "  ".join(f"{k}={v}" for k, v in reason_counts.items()))
        print(f"  Avg hold (days):       {np.mean([tr['hold_days'] for tr in all_trades]):.1f}")
        print(f"  Signal avg return:     {rets.mean()*100:>+.3f}%   "
              f"median={np.median(rets)*100:>+.3f}%   win rate={np.mean(rets>0)*100:.1f}%")
        print(f"  Placebo avg return:    {placebo_arr.mean()*100:>+.3f}%   "
              f"(n={len(placebo_arr):,}, same ticker/hold-length, random timing)")
        excess = rets.mean() - placebo_arr.mean()
        # simple two-sample t-stat for a rough significance read
        se = np.sqrt(rets.var()/len(rets) + placebo_arr.var()/len(placebo_arr))
        tstat = excess / se if se > 0 else np.nan
        print(f"  Excess vs placebo:     {excess*100:>+.3f}pp   (t-stat ~ {tstat:.2f})")

    print(f"\n{'='*90}")
    print("  INTERPRETATION")
    print(f"{'-'*90}")
    print("  Excess = signal trades' avg return minus same-ticker/same-hold-length")
    print("  placebo trades at random timing. Positive & |t-stat|>~2 means the")
    print("  accumulation-signal timing genuinely adds value beyond just owning")
    print("  the stock. If excess is near zero, the 'ride the smart money' entries")
    print("  aren't better timed than chance once you control for each stock's own")
    print("  drift over the same holding period.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
