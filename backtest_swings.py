"""
Swing structure analysis
========================
No fixed TP/SL. Uses the swing points themselves as natural exits.

LONG : buy at HL breakout → hold until next swing HIGH forms → measure gain
SHORT: sell at LH breakdown → hold until next swing LOW forms → measure gain

Also:
  - Transition matrix: P(HH|HL), P(LH|HL), etc.
  - Logistic regression: do accumulation features predict HH vs LH?

Usage:
  python backtest_swings.py ACB
  python backtest_swings.py ACB VCB FPT --mode both
  python backtest_swings.py --all --mode long
"""

import pandas as pd
import numpy as np
import sys, argparse
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE_DIR  = Path(__file__).parent
PRICE_DIR = BASE_DIR / "data" / "price"
TICK_DIR  = BASE_DIR / "data" / "tick_data"
ORDER     = 5
MAX_MOVE  = 0.22
MIN_BARS  = 3
MAX_BARS  = 40
T25_BARS  = 3    # T+2.5 settlement: buy open day 0, earliest sell = open day 3


# ── Data ──────────────────────────────────────────────────────────────

def load_daily(ticker: str) -> pd.DataFrame:
    path = PRICE_DIR / f"{ticker}.parquet"
    if path.exists():
        df = pd.read_parquet(path, columns=["time","open","high","low","close","volume"])
        df["date"] = pd.to_datetime(df["time"], errors="coerce")
        df = (df.dropna(subset=["date"]).set_index("date")
                .rename(columns={"open":"Open","high":"High","low":"Low",
                                  "close":"Close","volume":"Volume"})
                [["Open","High","Low","Close","Volume"]].sort_index().dropna())
        for c in ["Open","High","Low","Close"]:
            df[c] = df[c] * 1000
        return df
    path = TICK_DIR / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No data for {ticker}")
    df = pd.read_parquet(path, columns=["td","p","v"])
    df["date"] = pd.to_datetime(df["td"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df["v"] = pd.to_numeric(df["v"], errors="coerce").fillna(0)
    ohlcv = (df.groupby("date")["p"]
               .agg(Open="first", High="max", Low="min", Close="last")
               .join(df.groupby("date")["v"].sum().rename("Volume")))
    return ohlcv.sort_index().dropna()


# ── Pivot detection ────────────────────────────────────────────────────

def find_pivots(df, order=ORDER):
    h, l = df["High"].values, df["Low"].values
    highs, lows = [], []
    for i in range(order, len(df) - order):
        if h[i] == h[i-order:i+order+1].max(): highs.append(i)
        if l[i] == l[i-order:i+order+1].min(): lows.append(i)
    return highs, lows


def label_pivots(df, highs, lows):
    """Label each pivot HH/LH/HL/LL relative to previous same-type pivot."""
    pts = [(i, df["High"].iloc[i], "H") for i in highs]
    pts += [(i, df["Low"].iloc[i],  "L") for i in lows]
    pts.sort(key=lambda x: x[0])
    last_h = last_l = None
    labeled = []
    for idx, price, ptype in pts:
        if ptype == "H":
            lbl = ("HH" if last_h and price > last_h else
                   "LH" if last_h else "H")
            last_h = price
        else:
            lbl = ("HL" if last_l and price > last_l else
                   "LL" if last_l else "L")
            last_l = price
        labeled.append({"idx": idx, "price": price, "type": ptype, "label": lbl,
                         "date": df.index[idx]})
    return labeled


# ── Transition matrix ──────────────────────────────────────────────────

def transition_matrix(labeled):
    """
    Count P(next_label | current_label) for same-type pivots.
    e.g. P(HH | previous swing high was HH) = ?
    """
    from collections import defaultdict
    counts = defaultdict(lambda: defaultdict(int))
    prev_h = prev_l = None
    for p in labeled:
        if p["type"] == "H":
            if prev_h:
                counts[prev_h][p["label"]] += 1
            prev_h = p["label"]
        else:
            if prev_l:
                counts[prev_l][p["label"]] += 1
            prev_l = p["label"]
    return counts


# ── Swing-to-swing return measurement ─────────────────────────────────

def _vol_slope(vol_slice):
    if len(vol_slice) < 3: return 0.0
    x = np.arange(len(vol_slice), dtype=float)
    return float(np.polyfit(x, vol_slice.astype(float), 1)[0])


def measure_swings(df, labeled, mode="long",
                   max_move=MAX_MOVE, min_bars=MIN_BARS, max_bars=MAX_BARS,
                   order=ORDER):
    """
    For each HL (long) or LH (short):
      1. Check if there was a qualifying accumulation pattern before the pivot
         (slow move with declining volume)
      2. Find the next pivot of opposite type (H after L, L after H)
      3. Measure the gain from entry (next open after signal) to that exit pivot

    Returns a list of measurement dicts.
    """
    n = len(df)
    records = []

    # Build fast lookup: labeled by position
    by_idx = {p["idx"]: p for p in labeled}

    # We need pairs: (HL, next_H) for long, (LH, next_L) for short
    # Walk through labeled sequence
    H_pivots = [p for p in labeled if p["type"] == "H"]
    L_pivots = [p for p in labeled if p["type"] == "L"]

    if mode == "long":
        # For each HL, find the next swing high
        for i, lp in enumerate(L_pivots):
            if lp["label"] not in ("HL", "L"):
                continue
            lo_idx    = lp["idx"]
            lo_price  = lp["price"]

            # Find the swing high that preceded this low (to measure the correction)
            prev_highs = [p for p in H_pivots if p["idx"] < lo_idx]
            if not prev_highs:
                continue
            prev_H = prev_highs[-1]

            # Measure correction: prev swing high → this low
            correction = (prev_H["price"] - lo_price) / prev_H["price"]
            if correction > max_move or correction < 0.01:
                continue

            bars_down = lo_idx - prev_H["idx"]
            if bars_down < min_bars or bars_down > max_bars:
                continue

            # Volume slope during correction
            vol_sl = df["Volume"].iloc[prev_H["idx"]+1 : lo_idx+1].values
            v_slope = _vol_slope(vol_sl)

            # Find next swing high after this low
            next_highs = [p for p in H_pivots if p["idx"] > lo_idx]
            if not next_highs:
                continue
            next_H = next_highs[0]
            next_H_label = next_H["label"]   # HH = continuation, LH = failure

            # Entry: pivot confirmed only after ORDER bars on the right side
            # → earliest realistic entry is open of bar lo_idx + order + 1
            entry_idx = lo_idx + order + 1
            if entry_idx >= n:
                continue
            entry_price = df["Open"].iloc[entry_idx]

            # Exit: next swing high confirmed at next_H["idx"] + order
            # → exit at open of bar after confirmation
            # T+2.5 settlement: can't sell before open of entry + T25_BARS
            exit_idx = max(next_H["idx"] + order + 1, entry_idx + T25_BARS)
            if exit_idx >= n:
                exit_price = df["Close"].iloc[-1]
            else:
                exit_price = df["Open"].iloc[exit_idx]

            if entry_price <= 0 or not np.isfinite(entry_price):
                continue
            gain = (exit_price - entry_price) / entry_price * 100
            if not np.isfinite(gain):
                continue

            records.append({
                "direction":     "LONG",
                "entry_date":    df.index[entry_idx],
                "entry_price":   round(entry_price / 1000, 2),
                "exit_date":     df.index[min(exit_idx, n-1)],
                "exit_price":    round(exit_price / 1000, 2),
                "gain_pct":      round(gain, 2),
                "next_label":    next_H_label,     # HH or LH
                "win":           next_H_label == "HH",
                "correction":    round(correction * 100, 2),
                "bars_down":     bars_down,
                "vol_slope_neg": v_slope < 0,      # True = vol declining = good
                "vol_slope":     round(v_slope, 0),
                "hold_bars":     next_H["idx"] - lo_idx,
                "low_label":     lp["label"],
            })

    else:  # short
        for i, hp in enumerate(H_pivots):
            if hp["label"] not in ("LH", "H"):
                continue
            hi_idx   = hp["idx"]
            hi_price = hp["price"]

            prev_lows = [p for p in L_pivots if p["idx"] < hi_idx]
            if not prev_lows:
                continue
            prev_L = prev_lows[-1]

            rally = (hi_price - prev_L["price"]) / prev_L["price"]
            if rally > max_move or rally < 0.01:
                continue

            bars_up = hi_idx - prev_L["idx"]
            if bars_up < min_bars or bars_up > max_bars:
                continue

            vol_sl  = df["Volume"].iloc[prev_L["idx"]+1 : hi_idx+1].values
            v_slope = _vol_slope(vol_sl)

            next_lows = [p for p in L_pivots if p["idx"] > hi_idx]
            if not next_lows:
                continue
            next_L = next_lows[0]
            next_L_label = next_L["label"]  # LL = continuation, HL = failure

            entry_idx = hi_idx + order + 1
            if entry_idx >= n:
                continue
            entry_price = df["Open"].iloc[entry_idx]

            exit_idx = max(next_L["idx"] + order + 1, entry_idx + T25_BARS)
            if exit_idx >= n:
                exit_price = df["Close"].iloc[-1]
            else:
                exit_price = df["Open"].iloc[exit_idx]

            if entry_price <= 0 or not np.isfinite(entry_price):
                continue
            gain = (entry_price - exit_price) / entry_price * 100
            if not np.isfinite(gain):
                continue

            records.append({
                "direction":     "SHORT",
                "entry_date":    df.index[entry_idx],
                "entry_price":   round(entry_price / 1000, 2),
                "exit_date":     df.index[min(exit_idx, n-1)],
                "exit_price":    round(exit_price / 1000, 2),
                "gain_pct":      round(gain, 2),
                "next_label":    next_L_label,
                "win":           next_L_label == "LL",
                "correction":    round(rally * 100, 2),
                "bars_down":     bars_up,
                "vol_slope_neg": v_slope < 0,
                "vol_slope":     round(v_slope, 0),
                "hold_bars":     next_L["idx"] - hi_idx,
                "high_label":    hp["label"],
            })

    return records


# ── Logistic regression ────────────────────────────────────────────────

def run_regression(records, direction="LONG"):
    """
    Simple logistic regression: predict win (HH for long, LL for short)
    from accumulation features.
    Features: correction%, bars_down, vol_slope_neg
    """
    if len(records) < 20:
        return

    df = pd.DataFrame(records)
    df["vol_neg"] = df["vol_slope_neg"].astype(int)
    X = df[["correction", "bars_down", "vol_neg"]].values
    y = df["win"].astype(int).values

    # Normalize features
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-9
    Xn     = (X - X_mean) / X_std

    # Mini logistic regression via gradient descent (no sklearn dependency)
    w = np.zeros(Xn.shape[1] + 1)   # +1 for bias
    Xb = np.column_stack([np.ones(len(Xn)), Xn])
    lr, epochs = 0.1, 500
    for _ in range(epochs):
        logits = Xb @ w
        probs  = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))
        grad   = Xb.T @ (probs - y) / len(y)
        w     -= lr * grad

    # Accuracy
    preds = (1 / (1 + np.exp(-np.clip(Xb @ w, -20, 20))) > 0.5).astype(int)
    acc   = (preds == y).mean() * 100
    base  = max(y.mean(), 1 - y.mean()) * 100

    lbl = "HH (uptrend continues)" if direction == "LONG" else "LL (downtrend continues)"
    print(f"\n  Logistic regression → predicting {lbl}")
    print(f"  Features: correction%, bars_in_phase, vol_declining")
    print(f"  Accuracy: {acc:.0f}%   Baseline (majority): {base:.0f}%   "
          f"Lift: {acc-base:+.0f}pp")
    print(f"  Coefficients (normalized):")
    feat_names = ["correction%", "bars_down", "vol_neg"]
    for nm, coef in zip(feat_names, w[1:]):
        direction_lbl = "↑ (helps)" if coef > 0 else "↓ (hurts)"
        print(f"    {nm:<14} {coef:>+6.3f}  {direction_lbl}")

    # Segment by vol_declining
    for vol_filter, label in [(True, "vol declining ✓"), (False, "vol rising  ✗")]:
        sub = [r for r in records if r["vol_slope_neg"] == vol_filter]
        if len(sub) < 5: continue
        wr = np.mean([r["win"] for r in sub]) * 100
        avg = np.mean([r["gain_pct"] for r in sub])
        print(f"  When {label}: n={len(sub)}  WR={wr:.0f}%  avg={avg:+.1f}%")


# ── Output ─────────────────────────────────────────────────────────────

def summarise(ticker, records, df, labeled, mode):
    print(f"\n{'═'*68}")
    print(f"  {ticker}   {len(df)} days  "
          f"({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")

    if not records:
        print("  No swings measured."); return

    # Overall stats
    gains = [r["gain_pct"] for r in records]
    wins  = [r for r in records if r["win"]]
    print(f"  Swings measured : {len(records)}")
    print(f"  Continuations   : {len(wins)} / {len(records)} "
          f"({len(wins)/len(records)*100:.0f}%)")
    print(f"  Avg gain (buy HL → next H): {np.mean(gains):+.1f}%  "
          f"Med: {np.median(gains):+.1f}%  "
          f"Std: {np.std(gains):.1f}%")

    # Win vs loss breakdown
    w_gains = [r["gain_pct"] for r in wins]
    l_gains = [r["gain_pct"] for r in records if not r["win"]]
    if w_gains: print(f"  Avg gain when HH (continuation): {np.mean(w_gains):+.1f}%")
    if l_gains: print(f"  Avg gain when LH (failure)     : {np.mean(l_gains):+.1f}%")

    # Transition matrix
    tm = transition_matrix(labeled)
    print(f"\n  Transition matrix (what follows each pivot type):")
    for from_lbl in ("HH", "LH", "HL", "LL"):
        row = tm.get(from_lbl, {})
        total = sum(row.values())
        if not total: continue
        parts = "  ".join(f"{to}:{cnt/total*100:.0f}%({cnt})"
                          for to, cnt in sorted(row.items()))
        print(f"    {from_lbl} → {parts}")

    # Regression
    run_regression(records, direction="LONG" if mode == "long" else "SHORT")

    # ── Current signal ────────────────────────────────────────────
    today      = df.index[-1]
    cur_price  = df["Close"].iloc[-1]
    n          = len(df)
    last_H     = [p for p in labeled if p["type"] == "H"][-1:]
    last_L     = [p for p in labeled if p["type"] == "L"][-1:]

    lH = last_H[0] if last_H else None
    lL = last_L[0] if last_L else None

    print(f"\n  ── Current signal  ({today.strftime('%Y-%m-%d')})  price={cur_price/1000:.2f}k ──")

    if lH:
        conf_date = df.index[min(lH["idx"] + ORDER, n - 1)]
        bars_ago  = n - 1 - lH["idx"]
        print(f"  Last HIGH : {lH['price']/1000:.2f}k  {lH['date'].strftime('%Y-%m-%d')}  "
              f"[{lH['label']}]  confirmed {conf_date.strftime('%Y-%m-%d')}  ({bars_ago}b ago)")
    if lL:
        conf_date = df.index[min(lL["idx"] + ORDER, n - 1)]
        bars_ago  = n - 1 - lL["idx"]
        print(f"  Last LOW  : {lL['price']/1000:.2f}k  {lL['date'].strftime('%Y-%m-%d')}  "
              f"[{lL['label']}]  confirmed {conf_date.strftime('%Y-%m-%d')}  ({bars_ago}b ago)")

    # Derive verdict from last HIGH + last LOW labels
    h_lbl = lH["label"] if lH else "?"
    l_lbl = lL["label"] if lL else "?"

    if   l_lbl == "HL" and h_lbl == "HH":
        verdict = "LONG   ▲  uptrend intact — HL + HH"
    elif l_lbl == "HL" and h_lbl == "LH":
        verdict = "CAUTION  HL low but LH high — possible distribution"
    elif l_lbl == "HL":
        verdict = "LONG   ▲  HL confirmed — watch for next HIGH label"
    elif l_lbl == "LL" and h_lbl == "LH":
        verdict = "REDUCE ▼  downtrend — LL + LH"
    elif l_lbl == "LL":
        verdict = "REDUCE ▼  LL confirmed — downtrend structure"
    else:
        verdict = "NEUTRAL  insufficient pivot history"

    print(f"  Signal    : {verdict}")

    # Flag if a fresh pivot was confirmed very recently (within ORDER+2 bars)
    fresh_signals = []
    if lL and (n - 1 - lL["idx"]) <= ORDER + 2:
        fresh_signals.append(f"fresh {l_lbl} low confirmed ~{lL['date'].strftime('%Y-%m-%d')}")
    if lH and (n - 1 - lH["idx"]) <= ORDER + 2:
        fresh_signals.append(f"fresh {h_lbl} high confirmed ~{lH['date'].strftime('%Y-%m-%d')}")
    if fresh_signals:
        print(f"  ** FRESH : {' | '.join(fresh_signals)}")

    # Recent trades table
    recent = records[-20:]
    print(f"\n  {'Dir':<6} {'Entry':11} {'E.P':>6} {'Exit':11} {'X.P':>6} "
          f"{'Gain%':>7} {'Next':<5} {'Corr%':>6} {'Bars':>4} {'VolDn':>6}")
    print(f"  {'-'*80}")
    for r in recent:
        icon = "✓" if r["win"] else "✗"
        vdn  = "✓" if r["vol_slope_neg"] else "✗"
        dir_lbl = "↑ LONG" if r["direction"] == "LONG" else "↓ SHRT"
        print(f"  {dir_lbl}  {r['entry_date'].strftime('%Y-%m-%d')}  {r['entry_price']:>6.2f}  "
              f"{r['exit_date'].strftime('%Y-%m-%d')}  {r['exit_price']:>6.2f}  "
              f"{r['gain_pct']:>+6.1f}%  {r['next_label']:<5} {icon}  "
              f"{r['correction']:>5.1f}%  {r['bars_down']:>4}  {vdn}")


# ── CLI ────────────────────────────────────────────────────────────────

def run_ticker(ticker, order, max_move, mode):
    try:
        df = load_daily(ticker)
    except FileNotFoundError as e:
        print(f"  {e}"); return []
    if len(df) < order * 2 + 10:
        return []

    highs, lows = find_pivots(df, order)
    labeled     = label_pivots(df, highs, lows)
    records     = []

    if mode in ("long", "both"):
        records += measure_swings(df, labeled, "long",  max_move, MIN_BARS, MAX_BARS, order)
    if mode in ("short", "both"):
        records += measure_swings(df, labeled, "short", max_move, MIN_BARS, MAX_BARS, order)

    records.sort(key=lambda x: x["entry_date"])
    summarise(ticker, records, df, labeled, mode)
    return records


# ═══════════════════════════════════════════════════════════════════════
# Aggregate analysis (--analyze)
# ═══════════════════════════════════════════════════════════════════════

def _logit_coefs(records):
    """Return (accuracy, baseline, lift, coefs_dict) from mini logistic regression."""
    if len(records) < 30:
        return None
    df = pd.DataFrame(records)
    df["vol_neg"] = df["vol_slope_neg"].astype(int)
    X = df[["correction", "bars_down", "vol_neg"]].values
    y = df["win"].astype(int).values
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Xb = np.column_stack([np.ones(len(Xn)), Xn])
    w  = np.zeros(Xb.shape[1])
    for _ in range(500):
        p  = 1 / (1 + np.exp(-np.clip(Xb @ w, -20, 20)))
        w -= 0.1 * Xb.T @ (p - y) / len(y)
    preds = (1 / (1 + np.exp(-np.clip(Xb @ w, -20, 20))) > 0.5).astype(int)
    acc   = (preds == y).mean() * 100
    base  = max(y.mean(), 1 - y.mean()) * 100
    return acc, base, acc - base, dict(zip(["correction", "bars_down", "vol_neg"], w[1:]))


def analyze_all(all_recs, order, max_move):
    """Full cross-ticker analysis: sector breakdown, best/worst stocks, feature importance."""
    if not all_recs:
        print("No records."); return

    # Load sector map
    sec_map = {}
    sec_path = BASE_DIR / "data" / "all_stocks_with_industries.parquet"
    if sec_path.exists():
        sm = pd.read_parquet(sec_path, columns=["ticker", "industry", "exchange"])
        sm = sm.drop_duplicates("ticker").set_index("ticker")
        sec_map = sm["industry"].to_dict()

    df_all = pd.DataFrame(all_recs)
    df_all["sector"] = df_all["ticker"].map(sec_map).fillna("Unknown")
    df_all["win_i"]  = df_all["win"].astype(int)

    n_total = len(df_all)
    wr_all  = df_all["win_i"].mean() * 100
    avg_all = df_all["gain_pct"].mean()

    print(f"\n{'═'*72}")
    print(f"  HOSE SWING ANALYSIS — LONG only  |  "
          f"{n_total:,} swings  |  order={order}  max_move={max_move*100:.0f}%")
    print(f"  Overall continuation rate: {wr_all:.1f}%   Avg gain/swing: {avg_all:+.2f}%")

    # ── 1. Sector breakdown ────────────────────────────────────────────
    print(f"\n  {'─'*72}")
    print(f"  SECTOR BREAKDOWN  (min 50 swings)")
    print(f"  {'Sector':<32} {'Swings':>6} {'Cont%':>6} {'AvgGain':>8} "
          f"{'AvgWin':>8} {'AvgLoss':>8}")
    print(f"  {'-'*68}")
    sec_stats = []
    for sec, grp in df_all.groupby("sector"):
        if len(grp) < 50: continue
        wr  = grp["win_i"].mean() * 100
        avg = grp["gain_pct"].mean()
        aw  = grp.loc[grp["win"], "gain_pct"].mean() if grp["win"].any() else 0
        al  = grp.loc[~grp["win"], "gain_pct"].mean() if (~grp["win"]).any() else 0
        sec_stats.append((sec, len(grp), wr, avg, aw, al))
    for sec, n, wr, avg, aw, al in sorted(sec_stats, key=lambda x: -x[2]):
        bar = "█" * int(wr / 5)
        print(f"  {sec:<32} {n:>6,}  {wr:>5.1f}%  {avg:>+7.2f}%  "
              f"{aw:>+7.2f}%  {al:>+7.2f}%  {bar}")

    # ── 2. Per-ticker stats ────────────────────────────────────────────
    ticker_stats = []
    for tkr, grp in df_all.groupby("ticker"):
        n = len(grp)
        if n < 15: continue
        wr  = grp["win_i"].mean() * 100
        avg = grp["gain_pct"].mean()
        aw  = grp.loc[grp["win"],  "gain_pct"].mean() if grp["win"].any()  else 0
        al  = grp.loc[~grp["win"], "gain_pct"].mean() if (~grp["win"]).any() else 0
        ev  = wr/100 * aw + (1 - wr/100) * al   # expected value
        sec = grp["sector"].iloc[0]
        avg_corr = grp["correction"].mean()
        avg_bars = grp["bars_down"].mean()
        ticker_stats.append({
            "ticker": tkr, "sector": sec, "n": n,
            "wr": wr, "avg": avg, "aw": aw, "al": al, "ev": ev,
            "avg_corr": avg_corr, "avg_bars": avg_bars,
        })
    ts = pd.DataFrame(ticker_stats).sort_values("ev", ascending=False)

    def _print_ticker_table(rows, title):
        print(f"\n  {title}")
        print(f"  {'Ticker':<7} {'Sector':<28} {'N':>4} {'Cont%':>6} "
              f"{'AvgGain':>8} {'EV':>6} {'Corr%':>6} {'Bars':>5}")
        print(f"  {'-'*76}")
        for _, r in rows.iterrows():
            print(f"  {r['ticker']:<7} {r['sector']:<28} {r['n']:>4}  "
                  f"{r['wr']:>5.1f}%  {r['avg']:>+7.2f}%  {r['ev']:>+5.2f}%  "
                  f"{r['avg_corr']:>5.1f}%  {r['avg_bars']:>5.1f}")

    print(f"\n  {'─'*72}")
    _print_ticker_table(ts.head(20), "TOP 20 by Expected Value (min 15 swings)")
    print(f"\n  {'─'*72}")
    _print_ticker_table(ts.tail(20).iloc[::-1], "BOTTOM 20 by Expected Value")

    # ── 3. Feature analysis across all trades ─────────────────────────
    print(f"\n  {'─'*72}")
    print(f"  WHAT SEPARATES ✓ (HH) FROM ✗ (LH) — all {n_total:,} trades")
    for feat, label in [("correction", "Correction%"), ("bars_down", "Bars down"),
                        ("vol_slope",  "Vol slope (raw)")]:
        if feat not in df_all.columns: continue
        wv = df_all.loc[df_all["win"],  feat].mean()
        lv = df_all.loc[~df_all["win"], feat].mean()
        print(f"  {label:<20}  ✓ avg={wv:>+7.2f}   ✗ avg={lv:>+7.2f}   "
              f"diff={wv-lv:>+6.2f}")

    # Correction depth buckets
    print(f"\n  Continuation rate by correction depth:")
    bins = [(0, 5), (5, 10), (10, 15), (15, 22)]
    for lo, hi in bins:
        sub = df_all[(df_all["correction"] >= lo) & (df_all["correction"] < hi)]
        if len(sub) < 20: continue
        wr  = sub["win_i"].mean() * 100
        avg = sub["gain_pct"].mean()
        print(f"    {lo:>2}-{hi}%  n={len(sub):>5,}  cont={wr:.1f}%  avg={avg:+.2f}%")

    # Bars-down buckets
    print(f"\n  Continuation rate by bars in pullback:")
    for lo, hi in [(3, 7), (7, 15), (15, 25), (25, 40)]:
        sub = df_all[(df_all["bars_down"] >= lo) & (df_all["bars_down"] < hi)]
        if len(sub) < 20: continue
        wr  = sub["win_i"].mean() * 100
        avg = sub["gain_pct"].mean()
        print(f"    {lo:>2}-{hi:>2} bars  n={len(sub):>5,}  cont={wr:.1f}%  avg={avg:+.2f}%")

    # ── 4. Combined logistic regression ───────────────────────────────
    result = _logit_coefs(all_recs)
    if result:
        acc, base, lift, coefs = result
        print(f"\n  {'─'*72}")
        print(f"  LOGISTIC REGRESSION  (all {n_total:,} trades)")
        print(f"  Accuracy {acc:.0f}%  Baseline {base:.0f}%  Lift {lift:+.0f}pp")
        for feat, coef in coefs.items():
            arrow = "↑ more likely HH" if coef > 0 else "↓ less likely HH"
            print(f"    {feat:<14} {coef:>+.3f}  {arrow}")

    print(f"\n{'═'*72}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tickers",     nargs="*")
    p.add_argument("--all",       action="store_true", help="All price tickers")
    p.add_argument("--hose",      action="store_true", help="HOSE tickers only (analysis mode)")
    p.add_argument("--analyze",   action="store_true", help="Run HOSE analysis, suppress per-ticker output")
    p.add_argument("--mode",      default="long", choices=["long","short","both"])
    p.add_argument("--order",     type=int,   default=ORDER)
    p.add_argument("--max-move",  type=float, default=MAX_MOVE * 100)
    args = p.parse_args()

    order    = args.order
    max_move = args.max_move / 100

    # Build ticker list
    if args.analyze or args.hose:
        sec_path = BASE_DIR / "data" / "all_stocks_with_industries.parquet"
        sm = pd.read_parquet(sec_path, columns=["ticker", "exchange"])
        hose = set(sm[sm["exchange"] == "HOSE"]["ticker"].unique())
        available = {f.stem for f in PRICE_DIR.glob("*.parquet")}
        tickers = sorted(hose & available)
        print(f"Running {len(tickers)} HOSE tickers …", flush=True)
    elif args.all:
        tickers = sorted(f.stem for f in PRICE_DIR.glob("*.parquet"))
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        p.print_help(); return

    all_recs = []
    for i, tkr in enumerate(tickers):
        if args.analyze or args.hose:
            # silent run — no per-ticker print
            try:
                df = load_daily(tkr)
            except FileNotFoundError:
                continue
            if len(df) < order * 2 + 10:
                continue
            highs, lows = find_pivots(df, order)
            labeled     = label_pivots(df, highs, lows)
            recs = []
            if args.mode in ("long", "both"):
                recs += measure_swings(df, labeled, "long",  max_move, MIN_BARS, MAX_BARS, order)
            if args.mode in ("short", "both"):
                recs += measure_swings(df, labeled, "short", max_move, MIN_BARS, MAX_BARS, order)
            for r in recs:
                r["ticker"] = tkr
            all_recs += recs
            if (i + 1) % 50 == 0:
                print(f"  … {i+1}/{len(tickers)}", flush=True)
        else:
            recs = run_ticker(tkr, order, max_move, args.mode)
            for r in recs:
                r["ticker"] = tkr
            all_recs += recs

    if args.analyze or args.hose:
        analyze_all(all_recs, order, max_move)
    elif len(tickers) > 1 and all_recs:
        print(f"\n{'═'*68}")
        print(f"  AGGREGATE  {len(tickers)} tickers  mode={args.mode}")
        for direction in ("LONG", "SHORT"):
            grp = [r for r in all_recs if r["direction"] == direction]
            if not grp: continue
            wr  = np.mean([r["win"] for r in grp]) * 100
            avg = np.mean([r["gain_pct"] for r in grp])
            print(f"  {direction}: {len(grp)} swings  "
                  f"Continuation: {wr:.0f}%  Avg gain: {avg:+.1f}%")
        print(f"{'═'*68}")


if __name__ == "__main__":
    main()
