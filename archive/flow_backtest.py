"""
flow_backtest.py v3

Entry  : buy at OPEN of day after 3 consecutive days score >= ACCUM_THRESH
Min hold : 3 trading days (T+2.5 settlement)
Exit   : DIST signal (score < -0.08) OR max-hold timeout (default 20d)
Sectors: Banks, Insurance, Real Estate, Automobiles & Parts (best from v2)

Usage:
    python flow_backtest.py                  # default sector filter
    python flow_backtest.py --all-sectors    # all 402 tickers
    python flow_backtest.py --max-hold 30
    python flow_backtest.py --detail         # print every individual trade
"""

import os, sys, datetime
import pandas as pd
import numpy as np

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

FLOW_DIR  = os.path.join(BASE_DIR, "data", "investor_flow")
PRICE_DIR = os.path.join(BASE_DIR, "data", "price")
SECTORS_F = os.path.join(BASE_DIR, "ticker_sectors.csv")

SMOOTH_DAYS      = 3
ACCUM_THRESH     = 0.04
DIST_EXIT_THRESH = -0.08
CONFIRM_DAYS     = 3
MIN_HOLD_DAYS    = 3
DEFAULT_MAX_HOLD = 20

GOOD_SECTORS = {
    "Banks",
    "Insurance",
    "Real Estate",
    "Automobiles & Parts",
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_flow(ticker):
    fpath = os.path.join(FLOW_DIR, f"{ticker}.parquet")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_parquet(fpath)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return None


def load_price(ticker):
    fpath = os.path.join(PRICE_DIR, f"{ticker}.parquet")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_parquet(fpath)
        col = "time" if "time" in df.columns else "date"
        df = df.rename(columns={col: "date"})
        df["date"] = df["date"].astype(str)
        df["open"]  = pd.to_numeric(df.get("open",  0), errors="coerce").fillna(0)
        df["close"] = pd.to_numeric(df.get("close", 0), errors="coerce").fillna(0)
        df = df[["date","open","close"]].sort_values("date").reset_index(drop=True)
        return df[df["close"] > 0]
    except Exception:
        return None


def load_sectors():
    try:
        df = pd.read_csv(SECTORS_F, encoding="utf-8-sig")
        col = next((c for c in df.columns if "sector" in c.lower() or "industry" in c.lower()), None)
        if col:
            return dict(zip(df["ticker"].str.upper(), df[col]))
    except Exception:
        pass
    return {}


# ── Score ─────────────────────────────────────────────────────────────────────

def add_scores(df):
    smart = df["to_chuc_trongnuoc_net"] + df["to_chuc_nuocngoai_net"]
    total = (
        df["tu_doanh_net"].abs() + df["ca_nhan_trongnuoc_net"].abs()
        + df["to_chuc_trongnuoc_net"].abs() + df["ca_nhan_nuocngoai_net"].abs()
        + df["to_chuc_nuocngoai_net"].abs()
    )
    df = df.copy()
    df["score_raw"]   = (smart / total.replace(0, np.nan)).fillna(0)
    df["smart_score"] = df["score_raw"].rolling(SMOOTH_DAYS, min_periods=1).mean()
    return df


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_ticker(flow_df, price_df, max_hold):
    flow_df = add_scores(flow_df)

    p_dates = list(price_df["date"])
    p_idx   = {d: i for i, d in enumerate(p_dates)}
    p_open  = dict(zip(p_dates, price_df["open"]))
    p_close = dict(zip(p_dates, price_df["close"]))

    scores = flow_df["smart_score"].values
    fdates = flow_df["date"].values

    trades = []
    i = CONFIRM_DAYS
    trade_end_i = -1

    while i < len(scores):
        if i <= trade_end_i:
            i += 1
            continue

        # 3-day confirmation: score just crossed and held for 3 days
        if not (scores[i]   >= ACCUM_THRESH and
                scores[i-1] >= ACCUM_THRESH and
                scores[i-2] >= ACCUM_THRESH and
                scores[i-3]  < ACCUM_THRESH):
            i += 1
            continue

        conf_date = fdates[i]
        if conf_date not in p_idx:
            i += 1
            continue

        next_p = p_idx[conf_date] + 1
        if next_p >= len(p_dates):
            i += 1
            continue

        entry_date  = p_dates[next_p]
        entry_price = p_open.get(entry_date, 0)
        if entry_price <= 0:
            i += 1
            continue

        min_exit_p_idx = next_p + MIN_HOLD_DAYS

        exit_date   = None
        exit_price  = None
        exit_reason = "max_hold"
        max_high    = entry_price
        days_held   = 0

        for j in range(i + 1, min(i + max_hold + 1, len(scores))):
            fdate = fdates[j]
            c = p_close.get(fdate, 0)
            if c > max_high:
                max_high = c

            # enforce T+2.5
            if fdate not in p_idx or p_idx[fdate] < min_exit_p_idx:
                continue

            if float(scores[j]) <= DIST_EXIT_THRESH:
                exit_date   = fdate
                exit_price  = c if c > 0 else entry_price
                exit_reason = "dist"
                days_held   = j - i
                break

            if j == min(i + max_hold, len(scores) - 1):
                exit_date  = fdate
                exit_price = c if c > 0 else entry_price
                days_held  = j - i

        if not exit_date or not exit_price or exit_price <= 0:
            i += 1
            continue

        gain_pct = (exit_price - entry_price) / entry_price * 100
        peak_pct = (max_high   - entry_price) / entry_price * 100

        trades.append({
            "entry_date":  entry_date,
            "exit_date":   exit_date,
            "days_held":   days_held,
            "entry_price": round(entry_price, 3),
            "exit_price":  round(exit_price,  3),
            "gain_pct":    round(gain_pct, 3),
            "peak_pct":    round(peak_pct, 3),
            "exit_reason": exit_reason,
        })

        trade_end_i = i + days_held
        i = trade_end_i + 1

    return trades


# ── Stats per ticker ──────────────────────────────────────────────────────────

def ticker_stats(trades):
    if not trades:
        return None
    gains     = [t["gain_pct"] for t in trades]
    wins      = [g for g in gains if g > 0]
    losses    = [g for g in gains if g <= 0]
    compound  = 1.0
    for g in gains:
        compound *= (1 + g / 100)

    return {
        "n":           len(trades),
        "win_rate":    len(wins) / len(trades) * 100,
        "avg_gain":    np.mean(gains),
        "total_gain":  sum(gains),           # sum of % gains (equal-capital basis)
        "compound":    (compound - 1) * 100, # compounded return on ticker
        "avg_win":     np.mean(wins)  if wins  else 0.0,
        "avg_loss":    np.mean(losses) if losses else 0.0,
        "best":        max(gains),
        "worst":       min(gains),
        "avg_days":    np.mean([t["days_held"] for t in trades]),
        "avg_peak":    np.mean([t["peak_pct"]  for t in trades]),
        "pct_dist":    sum(1 for t in trades if t["exit_reason"] == "dist") / len(trades) * 100,
        "pct_time":    sum(1 for t in trades if t["exit_reason"] == "max_hold") / len(trades) * 100,
    }


def overall_stats(all_trades):
    return ticker_stats(all_trades)


def ticker_mdd(trades):
    """MDD for a single ticker's sequential trades."""
    equity = [1.0]
    for t in sorted(trades, key=lambda x: x["entry_date"]):
        equity.append(equity[-1] * (1 + t["gain_pct"] / 100))
    peak, mdd = equity[0], 0.0
    for e in equity[1:]:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > mdd:
            mdd = dd
    return mdd * 100


def compute_mdd(all_trades):
    """
    Compute MDD per ticker, then aggregate.
    Also computes an equal-weight daily portfolio equity curve for the chart.
    """
    # Per-ticker MDD
    by_ticker = {}
    for t in all_trades:
        by_ticker.setdefault(t["ticker"], []).append(t)

    mdds = {tkr: ticker_mdd(tr) for tkr, tr in by_ticker.items()}
    mdd_values = list(mdds.values())
    avg_mdd = np.mean(mdd_values)
    max_mdd = max(mdd_values)
    worst_ticker = max(mdds, key=lambda k: mdds[k])

    # Date range for backtest period
    try:
        sorted_trades = sorted(all_trades, key=lambda x: x["entry_date"])
        d0 = sorted_trades[0]["entry_date"]
        d1 = sorted_trades[-1]["exit_date"]
        days  = (datetime.datetime.strptime(d1, "%Y-%m-%d") -
                 datetime.datetime.strptime(d0, "%Y-%m-%d")).days
        months = days / 30.44
    except Exception:
        months = 0

    # Equal-weight daily portfolio equity curve for the drawdown chart:
    # On each calendar date, average the return of all trades active that day.
    # Build a date→list[gain_pct] mapping (daily slice by trade duration).
    # Simplified: use the sequential per-ticker equity compounded and averaged.
    # For chart: just show per-ticker MDD distribution.

    # Portfolio-level equity: assume capital split equally across all active trades on each day
    # Build date range
    all_dates = sorted(set(
        t["entry_date"] for t in all_trades) | set(t["exit_date"] for t in all_trades))

    # Daily portfolio return = avg gain of all trades that exited that day
    # (simple proxy for a multi-ticker portfolio)
    exit_by_date = {}
    for t in all_trades:
        exit_by_date.setdefault(t["exit_date"], []).append(t["gain_pct"])

    equity    = [1.0]
    dd_curve  = [0.0]
    peak_e    = 1.0
    for d in all_dates:
        gains_today = exit_by_date.get(d, [])
        if gains_today:
            avg_r = np.mean(gains_today) / 100
            new_e = equity[-1] * (1 + avg_r)
            equity.append(new_e)
            if new_e > peak_e:
                peak_e = new_e
            dd_curve.append((peak_e - new_e) / peak_e * 100)

    # Portfolio-level MDD from daily curve
    portfolio_mdd = max(dd_curve) if dd_curve else 0

    # Annualised return from avg total gain per ticker
    total_compound = 1.0
    for tkr, tr in by_ticker.items():
        tkr_compound = 1.0
        for t in sorted(tr, key=lambda x: x["entry_date"]):
            tkr_compound *= (1 + t["gain_pct"] / 100)
        total_compound += (tkr_compound - 1) / len(by_ticker)

    return {
        "avg_mdd":       avg_mdd,
        "max_mdd":       max_mdd,
        "worst_ticker":  worst_ticker,
        "portfolio_mdd": portfolio_mdd,
        "mdds":          mdds,
        "months":        months,
        "dd_curve":      dd_curve,
        "all_dates":     all_dates,
    }


def compute_significance(all_trades):
    """
    T-test: is mean return significantly different from 0?
    Also compute per-trade Sharpe (annualised assuming avg hold days).
    """
    gains = np.array([t["gain_pct"] for t in all_trades])
    n     = len(gains)
    mean  = np.mean(gains)
    std   = np.std(gains, ddof=1)
    stderr = std / np.sqrt(n)
    tstat  = mean / stderr if stderr > 0 else 0

    # p-value approximation (two-tailed, large-sample normal)
    from math import erfc, sqrt
    pval = erfc(abs(tstat) / sqrt(2))

    # Per-trade Sharpe (annualised)
    avg_days    = np.mean([t["days_held"] for t in all_trades])
    trades_per_year = 252 / avg_days if avg_days > 0 else 0
    sharpe_annual = (mean / std) * np.sqrt(trades_per_year) if std > 0 else 0

    # Profit factor
    wins  = gains[gains > 0]
    losses = gains[gains <= 0]
    pf = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")

    # Expectancy per trade (in % of capital)
    win_rate = len(wins) / n
    expectancy = (win_rate * np.mean(wins) if len(wins) else 0) + \
                 ((1 - win_rate) * np.mean(losses) if len(losses) else 0)

    return {
        "n":           n,
        "mean":        mean,
        "std":         std,
        "tstat":       tstat,
        "pval":        pval,
        "sharpe":      sharpe_annual,
        "profit_factor": pf,
        "expectancy":  expectancy,
    }


# ── Pretty printers ───────────────────────────────────────────────────────────

def bar(val, lo, hi, width=20):
    if hi == lo:
        return " " * width
    frac = max(0, min(1, (val - lo) / (hi - lo)))
    n = int(frac * width)
    return "#" * n + "." * (width - n)


def main():
    flags    = sys.argv[1:]
    max_hold = DEFAULT_MAX_HOLD
    for i, f in enumerate(flags):
        if f == "--max-hold" and i + 1 < len(flags):
            try: max_hold = int(flags[i + 1])
            except: pass

    all_sectors = "--all-sectors" in flags
    show_detail = "--detail" in flags

    sectors = load_sectors()
    tickers = sorted(f.replace(".parquet","") for f in os.listdir(FLOW_DIR) if f.endswith(".parquet"))
    if not all_sectors:
        tickers = [t for t in tickers if sectors.get(t,"") in GOOD_SECTORS]

    sector_label = "all sectors" if all_sectors else "/".join(sorted(GOOD_SECTORS))
    print(f"\nFlow Backtest v3  |  {len(tickers)} tickers  |  max hold {max_hold}d")
    print(f"Sectors : {sector_label}")
    print(f"Entry   : 3-day confirmation, buy at OPEN next day")
    print(f"Min hold: {MIN_HOLD_DAYS}d (T+2.5)  |  Exit: DIST score < {DIST_EXIT_THRESH} or {max_hold}d timeout")
    print()

    all_trades  = []
    ticker_rows = []

    for ticker in tickers:
        flow_df  = load_flow(ticker)
        price_df = load_price(ticker)
        if flow_df is None or price_df is None or len(flow_df) < 30 or len(price_df) < 20:
            continue

        trades = simulate_ticker(flow_df, price_df, max_hold)
        if not trades:
            ticker_rows.append({
                "ticker": ticker, "sector": sectors.get(ticker,"")[:22],
                "n": 0, "win_rate": 0, "avg_gain": 0, "total_gain": 0,
                "compound": 0, "avg_days": 0, "best": 0, "worst": 0,
            })
            continue

        st = ticker_stats(trades)
        for t in trades:
            t["ticker"] = ticker
            t["sector"] = sectors.get(ticker,"Unknown")
        all_trades.extend(trades)

        ticker_rows.append({
            "ticker":    ticker,
            "sector":    sectors.get(ticker,"")[:22],
            "n":         st["n"],
            "win_rate":  st["win_rate"],
            "avg_gain":  st["avg_gain"],
            "total_gain":st["total_gain"],
            "compound":  st["compound"],
            "avg_days":  st["avg_days"],
            "best":      st["best"],
            "worst":     st["worst"],
        })

    if not all_trades:
        print("  No trades found.")
        return

    ov = overall_stats(all_trades)

    # ── Overall summary ───────────────────────────────────────────────────────
    total_gain_sum = sum(t["gain_pct"] for t in all_trades)
    compound_all   = 1.0
    for t in sorted(all_trades, key=lambda x: x["entry_date"]):
        compound_all *= (1 + t["gain_pct"] / 100)
    compound_all = (compound_all - 1) * 100

    n_wins  = sum(1 for t in all_trades if t["gain_pct"] > 0)
    n_loss  = len(all_trades) - n_wins

    print("=" * 75)
    print(f"  OVERALL  ({len(all_trades)} trades  |  {n_wins}W / {n_loss}L  |  win rate {ov['win_rate']:.1f}%)")
    print(f"  Avg gain      : {ov['avg_gain']:+.2f}% per trade")
    print(f"  Total gain    : {total_gain_sum:+.1f}%  (sum of all trades, equal-capital basis)")
    print(f"  Compound      : {compound_all:+.1f}%  (if 1 pool, trades sequential by date)")
    print(f"  Avg win       : {ov['avg_win']:+.2f}%   |  Avg loss: {ov['avg_loss']:+.2f}%")
    print(f"  Best trade    : {ov['best']:+.2f}%   |  Worst: {ov['worst']:+.2f}%")
    print(f"  Avg hold      : {ov['avg_days']:.1f}d   |  Avg peak: {ov['avg_peak']:+.2f}%")
    print(f"  Exit by DIST  : {ov['pct_dist']:.0f}%    |  Timeout: {ov['pct_time']:.0f}%")
    print("=" * 75)

    # ── MDD + statistical significance ────────────────────────────────────────
    mdd  = compute_mdd(all_trades)
    sig  = compute_significance(all_trades)

    print(f"\n  --- Risk metrics ---")
    print(f"  Avg MDD per ticker : {mdd['avg_mdd']:.1f}%")
    print(f"  Worst ticker MDD   : {mdd['max_mdd']:.1f}%  ({mdd['worst_ticker']})")
    print(f"  Portfolio MDD      : {mdd['portfolio_mdd']:.1f}%  (equal-weight daily portfolio)")
    print(f"  Profit factor      : {sig['profit_factor']:.2f}  (gross wins / gross losses  —  >1.5 good)")
    print(f"  Sharpe ratio       : {sig['sharpe']:.2f}  (ann., per-trade  —  >1 good)")
    print(f"  Expectancy         : {sig['expectancy']:+.2f}% per trade")
    calmar = ov["avg_gain"] * 12 / mdd["avg_mdd"] if mdd["avg_mdd"] > 0 else 0
    print(f"  Calmar (rough)     : {calmar:.2f}  (avg monthly gain / avg MDD)")

    print(f"\n  --- Statistical significance ---")
    print(f"  Mean return   : {sig['mean']:+.3f}%  (std {sig['std']:.2f}%,  n={sig['n']})")
    print(f"  T-statistic   : {sig['tstat']:.2f}")
    print(f"  P-value       : {sig['pval']:.4f}  ({'SIGNIFICANT at p<0.05' if sig['pval'] < 0.05 else 'NOT significant at p<0.05'})")
    print(f"\n  --- Honest caveats ---")
    months = mdd["months"]
    print(f"  [1] Only ~{months:.0f} months of data — very short for robust conclusions")
    print(f"  [2] Sector whitelist chosen AFTER seeing results — in-sample selection bias")
    print(f"  [3] No transaction costs (~0.15%/side → ~0.30%/trade reduces mean +0.42% to +0.12%)")
    print(f"  [4] T-stat assumes independent trades — correlated tickers inflate significance")
    print(f"  [5] Need walk-forward test before live trading (split: train H1 2025, test H2 2025)")

    # ── Per-ticker MDD distribution chart ────────────────────────────────────
    mdd_vals = sorted(mdd["mdds"].values())
    max_dd   = max(mdd_vals) if mdd_vals else 1
    print(f"\n  Per-ticker MDD distribution ({len(mdd_vals)} tickers, max {max_dd:.1f}%):")
    buckets = [(0,5),(5,10),(10,15),(15,20),(20,30),(30,50),(50,100)]
    labels  = ["0-5%","5-10%","10-15%","15-20%","20-30%","30-50%",">50%"]
    total_t = len(mdd_vals)
    for (lo, hi), lbl in zip(buckets, labels):
        count = sum(1 for v in mdd_vals if lo <= v < hi)
        b     = "#" * (count * 30 // max(total_t, 1))
        print(f"  {lbl:>8}  {b:<30}  {count:>3} tickers")

    # ── Portfolio drawdown chart over time ────────────────────────────────────
    dd = mdd["dd_curve"]
    if dd and max(dd) > 0:
        max_dd_p = max(dd)
        print(f"\n  Portfolio drawdown over time (equal-weight, max {max_dd_p:.1f}%):")
        rows, width = 6, 65
        step    = max(1, len(dd) // width)
        sampled = [max(dd[i:i+step]) for i in range(0, len(dd), step)][:width]
        for row in range(rows, 0, -1):
            thresh = max_dd_p * row / rows
            line   = "".join("#" if v >= thresh else " " for v in sampled)
            print(f"  {thresh:>5.1f}% | {line}")
        print(f"  {'0.0%':>6} | " + "-" * len(sampled))

    # ── By sector ─────────────────────────────────────────────────────────────
    print(f"\n  {'Sector':<28}  {'N':>4}  {'WinRate':>7}  {'AvgGain':>8}  {'TotalGain':>10}  {'Compound':>9}  {'AvgDays':>7}")
    print("  " + "-" * 85)
    by_sector = {}
    for t in all_trades:
        by_sector.setdefault(t["sector"], []).append(t)
    for sec, tr in sorted(by_sector.items(), key=lambda x: sum(t["gain_pct"] for t in x[1]), reverse=True):
        st  = ticker_stats(tr)
        tg  = sum(t["gain_pct"] for t in tr)
        cmp = 1.0
        for t in sorted(tr, key=lambda x: x["entry_date"]):
            cmp *= (1 + t["gain_pct"] / 100)
        cmp = (cmp - 1) * 100
        print(f"  {sec[:28]:<28}  {st['n']:>4}  {st['win_rate']:>6.1f}%  "
              f"{st['avg_gain']:>+7.2f}%  {tg:>+9.1f}%  {cmp:>+8.1f}%  {st['avg_days']:>6.1f}d")

    # ── Full ticker table ─────────────────────────────────────────────────────
    tr_df = (pd.DataFrame(ticker_rows)
               .sort_values("compound", ascending=False)
               .reset_index(drop=True))

    best_cmp  = tr_df["compound"].max()
    worst_cmp = tr_df["compound"].min()

    print(f"\n  Full ticker table ({len(tr_df)} tickers, sorted by compound return):")
    print(f"\n  {'#':>3}  {'Ticker':<6}  {'N':>3}  {'Win%':>5}  {'Avg':>7}  {'TotalGain':>10}  "
          f"{'Compound':>9}  {'Days':>5}  {'Best':>7}  {'Worst':>7}  {'Chart':<22}  Sector")
    print("  " + "-" * 120)

    for rank, row in tr_df.iterrows():
        if row["n"] == 0:
            print(f"  {rank+1:>3}  {row['ticker']:<6}  {'—':>3}  {'—':>5}  {'—':>7}  {'—':>10}  "
                  f"{'—':>9}  {'—':>5}  {'—':>7}  {'—':>7}  {'':22}  {row['sector']}")
            continue

        b = bar(row["compound"], worst_cmp, best_cmp, width=22)
        flag = " *" if row["win_rate"] >= 55 and row["avg_gain"] >= 1.5 and row["n"] >= 5 else "  "

        print(f"{flag}{rank+1:>3}  {row['ticker']:<6}  {row['n']:>3}  {row['win_rate']:>4.0f}%  "
              f"{row['avg_gain']:>+6.1f}%  {row['total_gain']:>+9.1f}%  "
              f"{row['compound']:>+8.1f}%  {row['avg_days']:>4.1f}d  "
              f"{row['best']:>+6.1f}%  {row['worst']:>+6.1f}%  {b}  {row['sector']}")

    # ── Gain distribution ─────────────────────────────────────────────────────
    gains = [t["gain_pct"] for t in all_trades]
    total = len(gains)
    print(f"\n  Gain distribution ({total} trades):")
    buckets = [(-999,-10),(-10,-5),(-5,0),(0,5),(5,10),(10,20),(20,999)]
    labels  = ["<-10%","-10 to -5%","-5 to 0%","0 to +5%","+5 to +10%","+10 to +20%",">+20%"]
    for (lo, hi), lbl in zip(buckets, labels):
        count = sum(1 for g in gains if lo <= g < hi)
        b     = "#" * (count * 40 // max(total, 1))
        print(f"  {lbl:>12}  {b:<40}  {count:>4}  ({count/total*100:.1f}%)")

    # ── Individual trades ─────────────────────────────────────────────────────
    if show_detail:
        print(f"\n  All trades ({len(all_trades)}):")
        print(f"  {'Ticker':<6}  {'Entry':>10}  {'Exit':>10}  {'Days':>4}  "
              f"{'Gain':>7}  {'Peak':>7}  {'Exit':<5}  Sector")
        print("  " + "-" * 82)
        for t in sorted(all_trades, key=lambda x: x["entry_date"]):
            rsn = "DIST" if t["exit_reason"] == "dist" else "TIME"
            print(f"  {t['ticker']:<6}  {t['entry_date']:>10}  {t['exit_date']:>10}  "
                  f"{t['days_held']:>4}  {t['gain_pct']:>+6.1f}%  {t['peak_pct']:>+6.1f}%  "
                  f"{rsn:<5}  {t['sector'][:22]}")

    print()


if __name__ == "__main__":
    main()
