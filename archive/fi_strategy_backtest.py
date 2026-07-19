"""
fi_strategy_backtest.py — Foreign-Institutional follow strategy, emitting a
trade CSV in the exact schema backtest_dashboard.html expects.

Three variants so the dashboard can compare them side by side:

  FI_FlowExit   entry: FI flow rank>=80% & 5d>0 & close>MA20
                exit:  flow rank<50% | close<MA20-5% | stop -12%
                (the version that had the BEST risk-adjusted result:
                 +10.2% CAGR, -31.5% maxDD)

  FI_PriceExit  entry: SAME as above
                exit:  close<MA20-5% | stop -12%   (NO flow exit)
                Tests whether dropping the early flow exit — which was ending
                68% of trades at ~15 days, apparently cutting winners short —
                recovers return without giving back the drawdown benefit.

  PriceOnly     entry: close>MA20 (no flow at all)
                exit:  same price rules
                The honest benchmark: +15.9% CAGR but -60.7% maxDD.

Guards carried from the rest of this project, each of which caught a real
error earlier: validated tickers (data/price holds non-ticker panel files),
zero-price rows dropped, adjusted prices only (the flow parquet's close is
UNADJUSTED — using it corrupts every return), point-in-time expanding ranks,
one position per ticker at a time.

Output columns match the dashboard's ALL_TRADES objects exactly:
  ticker, strategy, entry_date, entry_price, exit_date, exit_price,
  return_pct, days_held, exit_reason, entry_volume, trend_at_entry,
  volatility_at_entry

Usage:  python archive/fi_strategy_backtest.py
"""

import glob, os, sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOW_DIR = os.path.join(BASE, "data", "investor_flow")
PRICE_DIR = os.path.join(BASE, "data", "price")
OUT_CSV = os.path.join(BASE, os.environ.get("FI_OUT", "fi_strategy_trades.csv"))

FI_COL = "to_chuc_nuocngoai_net"
ENTRY_PCT, EXIT_PCT = 0.80, 0.50
TRAIL, STOP = 0.05, -0.12
COST = 0.005
MIN_LIQ_BN, LIQ_WINDOW = float(__import__("os").environ.get("FI_LIQ_BN", 20)), 250
WARMUP = 260


def load(t):
    fp = os.path.join(FLOW_DIR, f"{t}.parquet")
    pp = os.path.join(PRICE_DIR, f"{t}.parquet")
    if not (os.path.exists(fp) and os.path.exists(pp)):
        return None
    try:
        f = pd.read_parquet(fp); p = pd.read_parquet(pp)
    except Exception:
        return None
    if "date" not in f.columns or FI_COL not in f.columns:
        return None
    f["date"] = pd.to_datetime(f["date"])
    if "close" in f.columns:
        f = f.drop(columns=["close"])          # UNADJUSTED — never use for returns
    p.columns = [c.strip().lower() for c in p.columns]
    dc = "time" if "time" in p.columns else "date"
    p["date"] = pd.to_datetime(p[dc])
    if not {"close", "volume"}.issubset(p.columns):
        return None
    p = p[p["close"] > 0]
    d = f[["date", FI_COL]].merge(p[["date", "close", "volume"]], on="date", how="inner")
    d = d.sort_values("date").reset_index(drop=True)
    if len(d) < 500:
        return None

    c = d["close"]
    d["value_bn"] = c * 1000 * d["volume"] / 1e9
    d["liq_pit"] = d["value_bn"].rolling(LIQ_WINDOW, min_periods=100).mean().shift(1)
    d["ma20"] = c.rolling(20).mean()
    d["ma60"] = c.rolling(60).mean()
    d["vol20"] = c.pct_change().rolling(20).std() * 100
    raw = d[FI_COL].astype(float).fillna(0)
    d["fi_5d"] = raw.rolling(5, min_periods=3).sum()
    d["fi_rank"] = raw.rolling(20, min_periods=10).sum().expanding(min_periods=120).rank(pct=True)
    return d


def trend_label(row):
    if not np.isfinite(row["ma20"]) or not np.isfinite(row["ma60"]):
        return "RANGE"
    if row["ma20"] > row["ma60"] and row["close"] > row["ma20"]:
        return "UP"
    if row["ma20"] < row["ma60"] and row["close"] < row["ma20"]:
        return "DOWN"
    return "RANGE"


def backtest(d, ticker, strategy):
    use_flow_entry = strategy in ("FI_FlowExit", "FI_PriceExit")
    use_flow_exit = strategy == "FI_FlowExit"

    c = d["close"].values
    ma = d["ma20"].values
    rk = d["fi_rank"].values
    f5 = d["fi_5d"].values
    liq = (d["liq_pit"] >= MIN_LIQ_BN).values
    vol = d["vol20"].values
    volu = d["volume"].values
    dates = d["date"].dt.strftime("%Y-%m-%d").values
    n = len(d)

    out = []
    i = WARMUP
    while i < n - 1:
        ok = liq[i] and np.isfinite(ma[i]) and c[i] > ma[i]
        if use_flow_entry:
            ok = ok and np.isfinite(rk[i]) and rk[i] >= ENTRY_PCT and f5[i] > 0
        if not ok:
            i += 1
            continue

        entry, ei = c[i], i
        j, reason = i + 1, "END_OF_DATA"
        while j < n:
            if c[j] / entry - 1 <= STOP:
                reason = "STOP_LOSS"; break
            if np.isfinite(ma[j]) and c[j] < ma[j] * (1 - TRAIL):
                reason = "BREAKDOWN"; break
            if use_flow_exit and np.isfinite(rk[j]) and rk[j] < EXIT_PCT:
                reason = "FLOW_LEFT"; break
            j += 1
        j = min(j, n - 1)

        gross = c[j] / entry - 1
        out.append({
            "ticker": ticker,
            "strategy": strategy,
            "entry_date": dates[ei],
            "entry_price": round(float(entry), 2),
            "exit_date": dates[j],
            "exit_price": round(float(c[j]), 2),
            "return_pct": round(float((gross - COST) * 100), 2),
            "days_held": int(j - ei),
            "exit_reason": reason,
            "entry_volume": int(volu[ei]) if np.isfinite(volu[ei]) else 0,
            "trend_at_entry": trend_label(d.iloc[ei]),
            "volatility_at_entry": round(float(vol[ei]), 2) if np.isfinite(vol[ei]) else 0.0,
        })
        i = j + 1
    return out


def summarise(df):
    print(f"\n{'='*94}")
    print(f"  {'strategy':<15}{'trades':>8}{'win%':>7}{'avg%':>8}{'med%':>8}"
          f"{'days':>7}{'PF':>7}{'CAGR':>9}{'maxDD':>9}")
    print(f"{'-'*94}")
    for s, g in df.groupby("strategy"):
        r = g["return_pct"] / 100
        wins, losses = r[r > 0], r[r <= 0]
        pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan
        m = g.copy()
        m["mo"] = pd.to_datetime(m["exit_date"]).dt.to_period("M")
        monthly = m.groupby("mo")["return_pct"].mean().sort_index() / 100
        eq = (1 + monthly).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        yrs = max(len(monthly) / 12, 1e-9)
        cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else np.nan
        print(f"  {s:<15}{len(g):>8,}{100*(r>0).mean():>6.1f}%{r.mean()*100:>+8.2f}"
              f"{r.median()*100:>+8.2f}{g['days_held'].mean():>7.0f}{pf:>7.2f}"
              f"{cagr:>+9.1%}{dd:>9.1%}")

    print(f"\n{'-'*94}")
    print("  EXIT REASON MIX")
    for s, g in df.groupby("strategy"):
        mix = (g["exit_reason"].value_counts(normalize=True) * 100).round(0).astype(int)
        print(f"  {s:<15} " + "  ".join(f"{k}={v}%" for k, v in mix.items()))


def main():
    sect = pd.read_csv(os.path.join(BASE, "ticker_sectors.csv"))
    sect.columns = [c.strip().lower() for c in sect.columns]
    valid = set(sect["ticker"].astype(str).str.upper())
    tickers = sorted(t for t in
                     (os.path.splitext(os.path.basename(f))[0].upper()
                      for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet")))
                     if t in valid)
    print(f"loading {len(tickers)} validated tickers...")

    rows = []
    kept = 0
    for t in tickers:
        d = load(t)
        if d is None or (d["liq_pit"] >= MIN_LIQ_BN).sum() < 200:
            continue
        kept += 1
        for s in ("FI_FlowExit", "FI_PriceExit", "PriceOnly"):
            rows.extend(backtest(d, t, s))
    print(f"  usable tickers: {kept}")

    df = pd.DataFrame(rows).sort_values(["ticker", "strategy", "entry_date"])
    df.to_csv(OUT_CSV, index=False)
    summarise(df)
    print(f"\nsaved {len(df):,} trades ({df['ticker'].nunique()} tickers) to:\n  {OUT_CSV}")
    print("\ncolumns:", ", ".join(df.columns))


if __name__ == "__main__":
    main()
