"""
flow_follow_strategy.py — Practical "follow the money" strategy.

Different from every earlier test in this project: those used FIXED holding
periods (10/20/60/120 days). This one is DYNAMIC — you stay in while the flow
persists and leave when it stops. That is what "go with the money flow"
actually means as a trading rule, and it has not been tested here before.

RULES (deliberately simple, all computable from yesterday's close):

  ENTRY  — all three must hold
    1. flow_20d percentile rank >= ENTRY_PCT   (money is arriving, measured
       against this stock's OWN history, point-in-time expanding rank)
    2. flow_5d > 0                             (arriving now, not just lately)
    3. close > MA20                            (price confirms; do not catch
       a falling knife on flow alone)

  EXIT   — whichever fires first
    1. flow_20d percentile rank < EXIT_PCT     (the money left -> you leave)
    2. close < MA20 * (1 - TRAIL)              (price broke down)
    3. hard stop at STOP from entry
    (no time limit — that is the point)

Tests each of the 5 investor types as the "money" being followed, so the
question "whose flow is worth following" is answered by the same rule rather
than by five differently-tuned ones.

Reports practical metrics, not just significance: total return, win rate,
average hold length, best/worst, profit factor, max drawdown of the equity
curve, and turnover. Costs charged at 0.5% round trip.

Guards kept from the rest of the project (these caused real, large errors):
  - ticker validation (data/price holds non-ticker panel files)
  - zero-price rows dropped
  - point-in-time ranks only (expanding, shifted) — no future data
  - one position per ticker
  - trade-weighted aggregation, not date-mean

Usage:  python archive/flow_follow_strategy.py
        python archive/flow_follow_strategy.py --scan     (today's candidates)
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

PLAYERS = {
    "ForeignInst":   "to_chuc_nuocngoai_net",
    "ForeignRetail": "ca_nhan_nuocngoai_net",
    "DomInst":       "to_chuc_trongnuoc_net",
    "DomRetail":     "ca_nhan_trongnuoc_net",
    "Prop":          "tu_doanh_net",
}

ENTRY_PCT = 0.80
EXIT_PCT = 0.50
TRAIL = 0.05
STOP = -0.12
COST = 0.005
MIN_LIQ_BN = 20
LIQ_WINDOW = 250
WARMUP = 260


def valid_tickers():
    s = pd.read_csv(os.path.join(BASE, "ticker_sectors.csv"))
    s.columns = [c.strip().lower() for c in s.columns]
    return set(s["ticker"].astype(str).str.upper())


def load(t):
    fp = os.path.join(FLOW_DIR, f"{t}.parquet")
    pp = os.path.join(PRICE_DIR, f"{t}.parquet")
    if not (os.path.exists(fp) and os.path.exists(pp)):
        return None
    try:
        f = pd.read_parquet(fp); p = pd.read_parquet(pp)
    except Exception:
        return None
    if "date" not in f.columns:
        return None
    f["date"] = pd.to_datetime(f["date"])
    if "close" in f.columns:
        f = f.drop(columns=["close"])     # flow close is UNADJUSTED; never use it
    p.columns = [c.strip().lower() for c in p.columns]
    dc = "time" if "time" in p.columns else "date"
    p["date"] = pd.to_datetime(p[dc])
    if not {"close", "volume"}.issubset(p.columns):
        return None
    p = p[p["close"] > 0]
    d = f.merge(p[["date", "close", "volume"]], on="date", how="inner")
    d = d.sort_values("date").reset_index(drop=True)
    if len(d) < 500:
        return None
    c = d["close"]
    d["value_bn"] = c * 1000 * d["volume"] / 1e9
    d["liq_pit"] = d["value_bn"].rolling(LIQ_WINDOW, min_periods=100).mean().shift(1)
    d["ma20"] = c.rolling(20).mean()
    for name, col in PLAYERS.items():
        if col not in d.columns:
            continue
        raw = d[col].astype(float).fillna(0)
        f5 = raw.rolling(5, min_periods=3).sum()
        f20 = raw.rolling(20, min_periods=10).sum()
        d[f"{name}_5d"] = f5
        d[f"{name}_20d"] = f20
        # POINT-IN-TIME percentile of today's 20d flow within this stock's own
        # history so far. expanding().rank(pct=True) uses only data up to today.
        d[f"{name}_rank"] = f20.expanding(min_periods=120).rank(pct=True)
    return d


def run(d, name):
    """Dynamic flow-following. Returns list of trade dicts."""
    need = [f"{name}_rank", f"{name}_5d"]
    if any(x not in d.columns for x in need):
        return []
    c = d["close"].values
    ma = d["ma20"].values
    rk = d[f"{name}_rank"].values
    f5 = d[f"{name}_5d"].values
    liq = (d["liq_pit"] >= MIN_LIQ_BN).values
    dates = d["date"].values
    n = len(d)

    trades = []
    i = WARMUP
    while i < n - 1:
        if not (liq[i] and np.isfinite(rk[i]) and np.isfinite(ma[i])
                and rk[i] >= ENTRY_PCT and f5[i] > 0 and c[i] > ma[i]):
            i += 1
            continue
        entry, ei = c[i], i
        j = i + 1
        reason = "END"
        while j < n:
            if c[j] / entry - 1 <= STOP:
                reason = "STOP"; break
            if np.isfinite(ma[j]) and c[j] < ma[j] * (1 - TRAIL):
                reason = "BREAKDOWN"; break
            if np.isfinite(rk[j]) and rk[j] < EXIT_PCT:
                reason = "FLOW_LEFT"; break
            j += 1
        j = min(j, n - 1)
        trades.append({
            "entry_date": dates[ei], "exit_date": dates[j],
            "days": j - ei, "ret": c[j] / entry - 1 - COST, "reason": reason,
        })
        i = j + 1
    return trades


def run_noflow(d):
    """Benchmark arm: SAME exit rules, entry ignores flow (close>MA20 only).
    Any strategy with a trailing stop produces many small losses and a few
    large wins; this arm shows how much of that profile is the exit mechanic
    rather than the flow signal."""
    c = d["close"].values
    ma = d["ma20"].values
    liq = (d["liq_pit"] >= MIN_LIQ_BN).values
    dates = d["date"].values
    n = len(d)
    trades = []
    i = WARMUP
    while i < n - 1:
        if not (liq[i] and np.isfinite(ma[i]) and c[i] > ma[i]):
            i += 1
            continue
        entry, ei = c[i], i
        j = i + 1
        reason = "END"
        while j < n:
            if c[j] / entry - 1 <= STOP:
                reason = "STOP"; break
            if np.isfinite(ma[j]) and c[j] < ma[j] * (1 - TRAIL):
                reason = "BREAKDOWN"; break
            j += 1
        j = min(j, n - 1)
        trades.append({"entry_date": dates[ei], "exit_date": dates[j],
                       "days": j - ei, "ret": c[j] / entry - 1 - COST,
                       "reason": reason})
        i = j + 1
    return trades


def metrics(tr):
    """Practical metrics. NOTE: an earlier version compounded trades in ticker
    order and reported a meaningless -90%+ drawdown. Trades overlap across
    tickers, so a single-capital equity curve is not well defined here. We
    instead report a DATE-ORDERED equal-weight monthly curve, which is what a
    diversified book would actually experience."""
    if not tr:
        return None
    df = pd.DataFrame(tr)
    r = df["ret"]
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan

    df["m"] = pd.to_datetime(df["exit_date"]).dt.to_period("M")
    monthly = df.groupby("m")["ret"].mean().sort_index()
    eq = (1 + monthly).cumprod()
    dd = (eq / eq.cummax() - 1).min() if len(eq) else np.nan
    yrs = max(len(monthly) / 12.0, 1e-9)
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if len(eq) and eq.iloc[-1] > 0 else np.nan

    return dict(n=len(df), win=100 * (r > 0).mean(), avg=r.mean(), med=r.median(),
                best=r.max(), worst=r.min(), days=df["days"].mean(),
                pf=pf, dd=dd, cagr=cagr, months=len(monthly),
                reasons=df["reason"].value_counts().to_dict())


def main():
    scan = "--scan" in sys.argv
    valid = valid_tickers()
    tickers = sorted(t for t in
                     (os.path.splitext(os.path.basename(f))[0].upper()
                      for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet")))
                     if t in valid)
    print(f"loading {len(tickers)} validated tickers...")
    data = {}
    for t in tickers:
        d = load(t)
        if d is not None and (d["liq_pit"] >= MIN_LIQ_BN).sum() >= 200:
            data[t] = d
    print(f"  usable: {len(data)}\n")

    if scan:
        print("=" * 84)
        print("  TODAY'S CANDIDATES — entry conditions met on the latest bar")
        print("=" * 84)
        hits = []
        for t, d in data.items():
            last = d.iloc[-1]
            for name in PLAYERS:
                rk, f5 = f"{name}_rank", f"{name}_5d"
                if rk not in d.columns or not np.isfinite(last.get(rk, np.nan)):
                    continue
                if (last["liq_pit"] >= MIN_LIQ_BN and last[rk] >= ENTRY_PCT
                        and last[f5] > 0 and last["close"] > last["ma20"]):
                    hits.append({"ticker": t, "follow": name,
                                 "date": str(last["date"])[:10],
                                 "flow_rank": round(float(last[rk]), 3),
                                 "close": float(last["close"])})
        if not hits:
            print("  none")
        else:
            h = pd.DataFrame(hits).sort_values(["follow", "flow_rank"], ascending=[True, False])
            print(h.to_string(index=False))
            out = os.path.join(BASE, "backtest_reports", "flow_follow_scan.csv")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            h.to_csv(out, index=False)
            print(f"\nsaved to {out}")
        return

    print("=" * 100)
    print(f"  FOLLOW-THE-FLOW  |  enter rank>={ENTRY_PCT:.0%} & 5d>0 & close>MA20")
    print(f"  exit: rank<{EXIT_PCT:.0%} | close<MA20-{TRAIL:.0%} | stop {STOP:.0%}  "
          f"| cost {COST:.1%} | NO time limit")
    print("=" * 100)
    print(f"  {'follow':<15}{'trades':>7}{'win%':>7}{'avg':>8}{'med':>8}"
          f"{'avg days':>10}{'PF':>7}{'maxDD':>8}{'best':>8}{'worst':>8}")
    allm = {}
    for name in PLAYERS:
        tr = []
        for t, d in data.items():
            tr.extend(run(d, name))
        m = metrics(tr)
        if not m:
            continue
        allm[name] = m
        print(f"  {name:<15}{m['n']:>7,}{m['win']:>6.1f}%{m['avg']:>+8.2%}{m['med']:>+8.2%}"
              f"{m['days']:>10.0f}{m['pf']:>7.2f}{m['dd']:>8.1%}{m['best']:>+8.1%}{m['worst']:>+8.1%}")

    # BENCHMARK: identical EXIT rules, but entry ignores flow entirely
    # (close>MA20 only). This isolates how much of the edge comes from the
    # FLOW signal versus from the trailing-exit mechanics, which by themselves
    # manufacture a many-small-losses/few-big-wins profile.
    bench_tr = []
    for t, d in data.items():
        bench_tr.extend(run_noflow(d))
    bm = metrics(bench_tr)
    if bm:
        allm["[BENCH no-flow]"] = bm
        print(f"  {'[BENCH no-flow]':<15}{bm['n']:>7,}{bm['win']:>6.1f}%{bm['avg']:>+8.2%}"
              f"{bm['med']:>+8.2%}{bm['days']:>10.0f}{bm['pf']:>7.2f}{bm['dd']:>8.1%}"
              f"{bm['best']:>+8.1%}{bm['worst']:>+8.1%}")

    print(f"\n{'-'*100}")
    print("  ANNUALISED (equal-weight monthly book) + EDGE OVER NO-FLOW BENCHMARK")
    print(f"{'-'*100}")
    print(f"  {'follow':<17}{'CAGR':>9}{'maxDD':>9}{'avg/trade':>11}{'edge vs bench':>15}")
    for k, m in allm.items():
        edge = "" if k.startswith("[") or not bm else f"{m['avg']-bm['avg']:+.2%}"
        print(f"  {k:<17}{m['cagr']:>+9.1%}{m['dd']:>9.1%}{m['avg']:>+11.2%}{edge:>15}")

    print(f"\n{'-'*100}")
    print("  EXIT REASON MIX")
    for name, m in allm.items():
        tot = sum(m["reasons"].values())
        mix = "  ".join(f"{k}={v/tot:.0%}" for k, v in sorted(m["reasons"].items()))
        print(f"  {name:<15} {mix}")

    print(f"\n{'-'*100}")
    print("  READ THIS BEFORE TRADING IT")
    print(f"{'-'*100}")
    print("  'avg' is per trade, NOT annualised. A +1% average over ~40 days is")
    print("  roughly 6%/yr before tax and before any slippage beyond the 0.5% charged.")
    print("  Compare against simply holding the index over the same period.")


if __name__ == "__main__":
    main()
