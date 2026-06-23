"""
compare_flow_backtest.py
========================
Runs the 4-sector backtest on the Sep 2024 → today window with up to 3 passes:
  • Baseline      : FLOW_SIGNAL_ENABLED = False  (normal selection, no flow)
  • Peak-exit only: flow entry ranking OFF, FLOW_PEAK_EXIT = True
                    Tests exit timing only — does detecting the peak help?
  • Flow-full     : FLOW_SIGNAL_ENABLED = True + FLOW_PEAK_EXIT = True
                    Both entry ranking AND peak-exit signal active.

Reports side-by-side metrics + per-trade breakdown.

Usage:
  python compare_flow_backtest.py
  python compare_flow_backtest.py --no-plot
  python compare_flow_backtest.py --flow-top 8      # keep top-8 by smart_score
  python compare_flow_backtest.py --dist-exit        # legacy distribution-only exit
  python compare_flow_backtest.py --peak-exit        # composite peak-exit signal
  python compare_flow_backtest.py --window 10        # flow_rank window (default 20)
  python compare_flow_backtest.py --peak-exit --no-entry  # peak-exit only, no ranking
"""

import sys, argparse, importlib.util
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Import 4sectors as a named module so we can modify its globals ────────────
_here = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("_4s", _here / "4sectors.py")
_4s   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_4s)

FLOW_START = pd.Timestamp("2024-09-16")

# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers (duplicated from 4sectors.py to avoid re-import complexity)
# ─────────────────────────────────────────────────────────────────────────────

def metrics(ret):
    r   = ret.replace([np.inf, -np.inf], 0).dropna()
    cum = (1 + r).cumprod()
    tot = cum.iloc[-1] - 1
    n   = len(r) / 252
    ann = (1 + tot) ** (1 / max(n, 0.1)) - 1
    sh  = (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0
    mdd = ((cum / cum.cummax()) - 1).min()
    return tot*100, ann*100, sh, mdd*100


def monthly_returns(df_d):
    r = df_d["ret"].replace([np.inf, -np.inf], 0).dropna()
    r.index = pd.to_datetime(r.index)
    monthly = (1 + r).resample("ME").prod() - 1
    return monthly


def win_rate(df_t):
    if df_t is None or df_t.empty:
        return 0.0, 0.0
    wins = (df_t["pnl_pct"] > 0).sum()
    return float(wins / len(df_t) * 100), float(df_t["pnl_pct"].mean())


def avg_hold(df_t):
    if df_t is None or df_t.empty:
        return 0.0
    return float(df_t["hold_days"].mean())


# ─────────────────────────────────────────────────────────────────────────────
# Run one backtest pass with specific flow settings
# ─────────────────────────────────────────────────────────────────────────────

def run_one(df_all, gb, vn, stock_data,
            flow_entry: bool, peak_exit: bool, dist_exit: bool,
            top_n: int, window: int):
    """
    Run one backtest pass.

    flow_entry : enable FLOW_SIGNAL_ENABLED (entry re-ranking by smart_score)
    peak_exit  : enable FLOW_PEAK_EXIT (composite peak-exit signal)
    dist_exit  : enable FLOW_DIST_EXIT (legacy simple distribution-alert exit)
    """
    _4s.FLOW_SIGNAL_ENABLED = flow_entry
    _4s.FLOW_RANK_TOP_N     = top_n
    _4s.FLOW_RANK_WINDOW    = window
    _4s.FLOW_DIST_EXIT      = dist_exit
    _4s.FLOW_PEAK_EXIT      = peak_exit

    # Engine must be available whenever any flow feature is active
    needs_engine = flow_entry or peak_exit or dist_exit
    if not needs_engine:
        _saved_engine      = _4s._FLOW_ENGINE
        _4s._FLOW_ENGINE   = None

    df_d, df_t, _, _ = _4s.run_backtest(
        df_all, gb, vn, stock_data, FLOW_START)

    if not needs_engine:
        _4s._FLOW_ENGINE = _saved_engine

    # Build label
    parts = []
    if flow_entry:
        parts.append(f"entry(top{top_n or 'all'},w{window}d)")
    if peak_exit:
        parts.append("peak_exit")
    if dist_exit:
        parts.append("dist_exit")
    label = ("Flow(" + "+".join(parts) + ")") if parts else "Baseline (no flow)"

    return df_d, df_t, label


# ─────────────────────────────────────────────────────────────────────────────
# Comparison report
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(results: list):
    """results = [(df_d, df_t, label), ...]"""
    n = len(results)

    print(f"\n{'═'*72}")
    print(f"  FLOW SIGNAL COMPARISON  |  {FLOW_START.date()} → today  (~20 months)")
    print(f"{'─'*72}")

    # Header
    labels = [r[2] for r in results]
    hdr = f"  {'Metric':<20}"
    for lbl in labels:
        hdr += f"  {lbl[:22]:>22}"
    print(hdr)
    print(f"  {'─'*68}")

    all_met = []
    for df_d, df_t, lbl in results:
        tot, ann, sh, mdd = metrics(df_d["ret"])
        wr, avg_pnl = win_rate(df_t)
        ah = avg_hold(df_t)
        n_trades = len(df_t) if df_t is not None else 0
        inv = (df_d["active"] != "CASH").mean() * 100 if "active" in df_d.columns else 0
        end_val = float(df_d["port_value"].iloc[-1]) / 1e6
        all_met.append(dict(
            tot=tot, ann=ann, sh=sh, mdd=mdd,
            wr=wr, avg_pnl=avg_pnl, ah=ah, n_trades=n_trades,
            inv=inv, end_val=end_val
        ))

    def row(name, key, fmt="+.1f", suffix=""):
        line = f"  {name:<20}"
        vals = [m[key] for m in all_met]
        for i, v in enumerate(vals):
            cell = f"{v:{fmt}}{suffix}"
            if i > 0:
                delta = v - vals[0]
                sign  = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
                # For MDD and laggard-type metrics, sign is reversed
                if key == "mdd":
                    sign = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
                cell += f" ({sign}{abs(delta):.1f})"
            line += f"  {cell:>22}"
        print(line)

    row("Total return",  "tot",      "+.1f",  "%")
    row("CAGR",          "ann",      "+.1f",  "%")
    row("Sharpe",        "sh",       "+.2f",  "")
    row("Max drawdown",  "mdd",      "+.1f",  "%")
    row("End value (M)", "end_val",  ".2f",   "")
    row("Win rate",      "wr",       ".1f",   "%")
    row("Avg trade P&L", "avg_pnl",  "+.1f",  "%")
    row("Avg hold (d)",  "ah",       ".0f",   "d")
    row("N trades",      "n_trades", ".0f",   "")
    row("Time invested", "inv",      ".0f",   "%")
    print(f"  {'─'*68}")

    # Delta summary
    if n >= 2:
        base = all_met[0]
        for i, (_, _, lbl) in enumerate(results[1:], 1):
            m = all_met[i]
            print(f"\n  vs Baseline:  {lbl}")
            d_tot = m["tot"] - base["tot"]
            d_ann = m["ann"] - base["ann"]
            d_sh  = m["sh"]  - base["sh"]
            verdict = ("✅ BETTER" if d_ann > 0.5 and d_sh > 0
                       else "❌ WORSE"  if d_ann < -0.5 and d_sh < 0
                       else "≈ NEUTRAL")
            print(f"    CAGR: {d_ann:>+.1f}%  |  Sharpe: {d_sh:>+.2f}  |  "
                  f"Total return: {d_tot:>+.1f}%  →  {verdict}")
    print(f"{'═'*72}\n")


def print_monthly_table(results: list):
    """Side-by-side monthly returns."""
    print(f"\n  MONTHLY RETURNS COMPARISON")
    print(f"  {'Month':<8}", end="")
    for _, _, lbl in results:
        print(f"  {lbl[:18]:>18}", end="")
    print()
    print(f"  {'─'*60}")

    monthly_all = [monthly_returns(df_d) for df_d, _, _ in results]
    all_months = sorted(set().union(*[set(m.index) for m in monthly_all]))

    for dt in all_months:
        line = f"  {dt.strftime('%Y-%m'):<8}"
        vals = []
        for m in monthly_all:
            v = m.get(dt, np.nan)
            vals.append(v)
        for i, v in enumerate(vals):
            if np.isnan(v):
                line += f"  {'—':>18}"
            else:
                flag = ""
                if i > 0 and not np.isnan(vals[0]):
                    d = v - vals[0]
                    flag = f" ({d:>+.1f}%)" if abs(d) > 0.1 else ""
                line += f"  {v*100:>+12.1f}%{flag:>6}"
        print(line)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(results: list, no_plot: bool = False):
    matplotlib.use("TkAgg")
    fig = plt.figure(figsize=(14, 9), facecolor="#111827")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    DARK = "#1f2937";  GRID = "#374151";  TEXT = "#9ca3af"
    COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]

    def sax(ax):
        ax.set_facecolor(DARK)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.spines[:].set_color(GRID)
        ax.grid(color=GRID, lw=0.5, alpha=0.5)

    # 1. Equity curves
    ax1 = fig.add_subplot(gs[0, :])   # full width top
    sax(ax1)
    for i, (df_d, _, lbl) in enumerate(results):
        r   = df_d["ret"].replace([np.inf, -np.inf], 0).fillna(0)
        cum = (1 + r).cumprod() * 100
        ax1.plot(df_d.index, cum, color=COLORS[i % len(COLORS)],
                 lw=1.8 if i == 0 else 1.4,
                 linestyle="-" if i == 0 else "--",
                 label=lbl, alpha=0.9)
    ax1.axhline(100, color="#6b7280", lw=0.6, ls=":")
    ax1.set_title("Portfolio Value (start = 100)", color=TEXT, fontsize=9)
    ax1.legend(fontsize=8, framealpha=0.3, facecolor=DARK,
               edgecolor=GRID, labelcolor="#e5e7eb")
    ax1.set_ylabel("Value", color=TEXT, fontsize=8)

    # 2. Rolling 20d return diff (flow-on vs baseline)
    ax2 = fig.add_subplot(gs[1, 0])
    sax(ax2)
    if len(results) >= 2:
        base_r = results[0][0]["ret"].replace([np.inf,-np.inf], 0).fillna(0)
        flow_r = results[1][0]["ret"].replace([np.inf,-np.inf], 0).fillna(0)
        common = base_r.index.intersection(flow_r.index)
        diff   = flow_r.loc[common] - base_r.loc[common]
        roll   = diff.rolling(20).sum() * 100
        pos = roll.clip(lower=0)
        neg = roll.clip(upper=0)
        ax2.fill_between(roll.index, pos, color="#10b981", alpha=0.6, label="Flow > Base")
        ax2.fill_between(roll.index, neg, color="#ef4444", alpha=0.6, label="Base > Flow")
        ax2.axhline(0, color="#6b7280", lw=0.8)
        ax2.set_title("Rolling 20d excess return: Flow-ON vs Baseline", color=TEXT, fontsize=8)
        ax2.set_ylabel("%", color=TEXT, fontsize=8)
        ax2.legend(fontsize=7, framealpha=0.3, facecolor=DARK, edgecolor=GRID,
                   labelcolor="#e5e7eb")

    # 3. Trade P&L distribution
    ax3 = fig.add_subplot(gs[1, 1])
    sax(ax3)
    for i, (df_d, df_t, lbl) in enumerate(results):
        if df_t is None or df_t.empty:
            continue
        pnls = df_t["pnl_pct"].clip(-30, 60)
        ax3.hist(pnls, bins=25, color=COLORS[i % len(COLORS)],
                 alpha=0.55 if i > 0 else 0.75,
                 label=f"{lbl[:16]} (n={len(df_t)})",
                 edgecolor="none")
    ax3.axvline(0, color="#6b7280", lw=0.8)
    ax3.set_title("Trade P&L distribution", color=TEXT, fontsize=8)
    ax3.set_xlabel("P&L %", color=TEXT, fontsize=8)
    ax3.legend(fontsize=7, framealpha=0.3, facecolor=DARK, edgecolor=GRID,
               labelcolor="#e5e7eb")

    fig.suptitle(f"Flow Signal Comparison  |  {FLOW_START.date()} → today",
                 color="white", fontsize=11)
    out = _here / "compare_flow_backtest.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Chart saved: {out.name}")
    if not no_plot:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-plot",   action="store_true")
    p.add_argument("--flow-top",  type=int,   default=10,
                   help="FLOW_RANK_TOP_N (default 10)")
    p.add_argument("--window",    type=int,   default=20,
                   help="FLOW_RANK_WINDOW in days (default 20)")
    p.add_argument("--dist-exit", action="store_true",
                   help="Enable legacy distribution-alert exit")
    p.add_argument("--peak-exit", action="store_true",
                   help="Enable composite peak-exit signal (distribution+heavy_sell+flip)")
    p.add_argument("--no-entry",  action="store_true",
                   help="Disable entry re-ranking (test exit-only)")
    p.add_argument("--monthly",   action="store_true",
                   help="Print monthly returns table")
    args = p.parse_args()

    use_exit = args.peak_exit or args.dist_exit

    print(f"\n{'═'*72}")
    print(f"  Flow Signal Backtest Comparison")
    print(f"  Period : {FLOW_START.date()} → today  (~20 months)")
    print(f"  Config : top_n={args.flow_top}  window={args.window}d  "
          f"peak_exit={args.peak_exit}  dist_exit={args.dist_exit}")
    print(f"{'─'*72}")
    print(f"  Loading data (runs once for all passes)...")

    # ── Load shared data ────────────────────────────────────────────────────
    _4s.FUNDAMENTAL_FILTER_ENABLED = True

    stock_data     = _4s.load_individual_stocks(_4s.INDIVIDUAL_DATA_DIR)
    liquid_tickers = _4s.get_liquid_tickers(stock_data)

    if _4s.FUNDAMENTAL_FILTER_ENABLED:
        _4s._FUND_DATA = _4s.load_fundamental_data()

    df_all, gb, vn, approved = _4s.load_sector_data(
        _4s.STOCK_DATA_PATH, _4s.VNINDEX_PATH, liquid_tickers)
    stock_data = {t: d for t, d in stock_data.items()
                  if t in approved and t in liquid_tickers}

    if _4s.FACTOR_SELECTION_ENABLED and _4s._FACTOR_RANKER_AVAILABLE:
        print("  Loading factor features...")
        _4s._QFEAT = _4s.build_factor_features(symbols=list(approved))

    # Pre-load flow engine once (shared across all runs)
    try:
        from flow_signals import FlowSignalEngine
        _flow_dir = _here / "data" / "investor_flow"
        _4s._FLOW_ENGINE = FlowSignalEngine(_flow_dir).load()
        print(f"  [FLOW] {len(_4s._FLOW_ENGINE.tickers)} tickers loaded: "
              f"{', '.join(_4s._FLOW_ENGINE.tickers)}")
    except Exception as e:
        print(f"  [FLOW] Engine load failed: {e}")
        _4s._FLOW_ENGINE = None

    # ── Build run list ──────────────────────────────────────────────────────
    # Always start with baseline.  Add extra passes when flow flags are set.
    runs = []

    # Pass 1: Baseline — no flow at all
    runs.append(dict(flow_entry=False, peak_exit=False, dist_exit=False))

    # Pass 2: Exit-only — peak/dist exit with no entry re-ranking
    if use_exit and args.no_entry:
        runs.append(dict(flow_entry=False,
                         peak_exit=args.peak_exit,
                         dist_exit=args.dist_exit))

    # Pass 3: Full flow — entry ranking + exit signal (skip if --no-entry, same as pass 2)
    if not args.no_entry:
        runs.append(dict(flow_entry=True,
                         peak_exit=args.peak_exit,
                         dist_exit=args.dist_exit))

    results = []
    total = len(runs)
    for i, cfg in enumerate(runs, 1):
        lbl_hint = ("BASELINE" if not any(cfg.values())
                    else ("PEAK-EXIT ONLY" if cfg["peak_exit"] and not cfg["flow_entry"]
                    else "FLOW-FULL"))
        print(f"\n{'─'*72}")
        print(f"  [{i}/{total}] Running {lbl_hint}...")
        df_d, df_t, lbl = run_one(
            df_all, gb, vn, stock_data,
            top_n=args.flow_top,
            window=args.window,
            **cfg,
        )
        results.append((df_d, df_t, lbl))

    # ── Report ──────────────────────────────────────────────────────────────
    print_comparison(results)
    if args.monthly:
        print_monthly_table(results)

    # Per-trade breakdown vs baseline
    dt_base = results[0][1]
    if dt_base is not None and not dt_base.empty and len(results) > 1:
        for _, dt_alt, lbl_alt in results[1:]:
            if dt_alt is None or dt_alt.empty:
                continue
            print(f"\n  TRADE COUNT BY SECTOR  (vs {lbl_alt})")
            print(f"  {'Sector':<20}  {'Baseline':>9}  {'Alt':>9}  {'Delta':>7}")
            print(f"  {'─'*52}")
            base_sec = dt_base.groupby("sector")["pnl_pct"].agg(["count","mean"])
            alt_sec  = dt_alt.groupby("sector")["pnl_pct"].agg(["count","mean"])
            for sec in sorted(set(base_sec.index) | set(alt_sec.index)):
                bn = int(base_sec.loc[sec, "count"]) if sec in base_sec.index else 0
                fn = int(alt_sec.loc[sec,  "count"]) if sec in alt_sec.index  else 0
                bm = float(base_sec.loc[sec, "mean"]) if sec in base_sec.index else 0
                fm = float(alt_sec.loc[sec,  "mean"]) if sec in alt_sec.index  else 0
                print(f"  {sec:<20}  {bn:>4} ({bm:>+5.1f}%)  {fn:>4} ({fm:>+5.1f}%)  {fn-bn:>+4}")

            base_tickers = set(dt_base["ticker"].unique())
            alt_tickers  = set(dt_alt["ticker"].unique())
            added   = alt_tickers  - base_tickers
            removed = base_tickers - alt_tickers
            if added:
                print(f"\n  NEW in {lbl_alt}:     {', '.join(sorted(added))}")
            if removed:
                print(f"  REMOVED in {lbl_alt}: {', '.join(sorted(removed))}")

    plot_comparison(results, no_plot=args.no_plot)


if __name__ == "__main__":
    main()
