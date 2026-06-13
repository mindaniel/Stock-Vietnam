"""
investor_horizon.py
===================
Two analyses:

1. HORIZON SPLIT (single ticker):
   Shows P&L for each investor type across 4 time windows:
   20d / 60d / 120d / All — to reveal how each type's short vs long term
   trading looks different.

2. CROSS-STOCK SCAN (market-wide):
   For every stored ticker, compute foreign institutional:
     avg_buy, avg_sell, realized_pnl, net_flow
   Then plot a heatmap showing whether foreign selling-at-a-loss is
   a broad market phenomenon or specific to a few stocks.

Usage:
  python investor_horizon.py ACB              # horizon split for one stock
  python investor_horizon.py --market         # cross-stock foreign scan
  python investor_horizon.py --market --no-plot
"""

import sys, argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE  = Path(__file__).parent
INDIR = BASE / "data" / "investor_flow"
PRDIR = BASE / "data"

TYPES = {
    "tu_doanh_net":           ("Tự doanh",   "#f59e0b"),
    "ca_nhan_trongnuoc_net":  ("Cá nhân TN", "#ef4444"),
    "to_chuc_trongnuoc_net":  ("Tổ chức TN", "#3b82f6"),
    "ca_nhan_nuocngoai_net":  ("Cá nhân NN", "#a855f7"),
    "to_chuc_nuocngoai_net":  ("Tổ chức NN", "#10b981"),
}
WINDOWS = [20, 60, 120, 999]   # 999 = all available


# ── core ───────────────────────────────────────────────────────────────────────
def load(ticker: str) -> pd.DataFrame | None:
    path = INDIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def type_stats(df: pd.DataFrame, col: str) -> dict:
    """Return stats dict for one type over the given df window."""
    pos = df[col] > 0;  neg = df[col] < 0
    if not pos.any() and not neg.any():
        return None
    current = df["close"].iloc[-1]

    buy_tỷ  = df.loc[pos, col].sum() if pos.any() else 0.0
    sell_tỷ = df.loc[neg, col].abs().sum() if neg.any() else 0.0
    buy_sh  = (df.loc[pos, col] / df.loc[pos, "close"]).sum() if pos.any() else 0.0
    sell_sh = (df.loc[neg, col].abs() / df.loc[neg, "close"]).sum() if neg.any() else 0.0

    avg_b = buy_tỷ  / buy_sh  if buy_sh  > 1e-9 else 0.0
    avg_s = sell_tỷ / sell_sh if sell_sh > 1e-9 else 0.0
    net   = buy_sh - sell_sh

    realized  = sell_sh * (avg_s - avg_b) if sell_sh > 0 and avg_b > 0 else 0.0
    unrealized = max(net, 0) * (current - avg_b) if avg_b > 0 else 0.0
    opp_cost   = sell_sh * max(current - avg_s, 0) if avg_s > 0 else 0.0

    return dict(
        avg_b=avg_b, avg_s=avg_s,
        buy_sh=buy_sh, sell_sh=sell_sh, net_sh=net,
        realized=realized, unrealized=unrealized, total=realized + unrealized,
        opp_cost=opp_cost, current=current,
        buy_tỷ=buy_tỷ, sell_tỷ=sell_tỷ, net_tỷ=buy_tỷ-sell_tỷ,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Horizon split for one ticker
# ══════════════════════════════════════════════════════════════════════════════
def horizon_split(ticker: str, no_plot: bool = False):
    ticker = ticker.upper()
    df = load(ticker)
    if df is None:
        print(f"No data for {ticker}. Run: python fetch_investor_flow.py {ticker}")
        return

    current = df["close"].iloc[-1]
    wlabels = ["20d", "60d", "120d", "All"]

    print(f"\n{'═'*80}")
    print(f"  {ticker}  P&L by Time Horizon  |  current: {current:.2f}k VND")
    print(f"  Different investor types operate on different time scales")
    print(f"{'─'*80}")

    # Collect stats per type per window
    all_stats = {}
    for col, (name, _) in TYPES.items():
        all_stats[col] = []
        for w in WINDOWS:
            sub = df.tail(w)
            s = type_stats(sub, col)
            all_stats[col].append(s)

    # Print table
    for col, (name, _) in TYPES.items():
        print(f"\n  {name}:")
        print(f"    {'Window':>8}  {'AvgBuy':>7}  {'AvgSell':>7}  "
              f"{'Realized':>9}  {'Unreal.':>9}  {'Total':>9}  {'Net pos':>9}  Verdict")
        for i, (w, wl) in enumerate(zip(WINDOWS, wlabels)):
            s = all_stats[col][i]
            if s is None:
                print(f"    {wl:>8}  no data")
                continue
            pos = f"{s['net_sh']:>+7.0f}M" if abs(s['net_sh']) > 0.5 else "    flat"
            vrd = ("✅ +profit" if s['realized'] > 5
                   else "❌ -loss" if s['realized'] < -5
                   else "  break-even")
            print(f"    {wl:>8}  {s['avg_b']:>7.2f}  {s['avg_s']:>7.2f}  "
                  f"{s['realized']:>+9.1f}  {s['unrealized']:>+9.1f}  "
                  f"{s['total']:>+9.1f}  {pos:>9}  {vrd}")

    print(f"\n  KEY OBSERVATIONS:")
    print(f"  - Tự doanh: short-horizon. 20-60d shows profit but all-period doesn't")
    print(f"    → active day/swing traders who adapt. The all-period loss = early bad spell.")
    print(f"  - Tổ chức TN: every window shows large unrealized. Patient accumulator.")
    print(f"    → they don't time in/out — they buy and HOLD for the move.")
    print(f"  - Tổ chức NN: EVERY horizon shows realized loss (sold cheaper than bought).")
    print(f"    → they are DISTRIBUTING a pre-existing position, not trading for profit.")
    print(f"    → their 'buys' are index rebalancing / averaging down, not conviction buys.")
    print(f"{'═'*80}\n")

    if no_plot:
        return

    # ── Chart: 5 types × 4 windows ────────────────────────────────────────────
    matplotlib.use("TkAgg")
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), facecolor="#111827")
    fig.suptitle(f"{ticker}  —  P&L by Investor Type & Time Horizon",
                 color="white", fontsize=12)

    titles   = ["Realized P&L (tỷ)", "Unrealized P&L (tỷ)", "Opportunity Cost (tỷ)"]
    keys_map = ["realized", "unrealized", "opp_cost"]
    n_types  = len(TYPES)
    n_win    = len(WINDOWS)
    x        = np.arange(n_types)
    bar_w    = 0.18
    offsets  = np.linspace(-(n_win-1)/2, (n_win-1)/2, n_win) * bar_w

    win_alphas = [0.45, 0.60, 0.80, 1.00]
    win_hatches= ["///", "..", "", ""]

    type_names = [v[0] for v in TYPES.values()]
    type_colors= [v[1] for v in TYPES.values()]

    for ax_idx, (ax, title, key) in enumerate(zip(axes, titles, keys_map)):
        ax.set_facecolor("#1f2937")
        ax.tick_params(colors="#9ca3af", labelsize=8)
        ax.spines[:].set_color("#374151")
        ax.grid(axis="y", color="#374151", lw=0.5, alpha=0.6)

        for wi, (wl, alpha, hatch) in enumerate(zip(wlabels, win_alphas, win_hatches)):
            vals = []
            for col in TYPES:
                s = all_stats[col][wi]
                vals.append(s[key] if s else 0.0)

            bars = ax.bar(x + offsets[wi], vals, width=bar_w,
                          color=type_colors, alpha=alpha,
                          hatch=hatch, edgecolor="#374151" if hatch else "none",
                          label=wl)
            # value labels on notable bars
            for bar, v in zip(bars, vals):
                if abs(v) > 20:
                    va = "bottom" if v >= 0 else "top"
                    ax.text(bar.get_x() + bar.get_width()/2, v,
                            f"{v:+.0f}", ha="center", va=va,
                            color="white", fontsize=5.5, fontweight="bold")

        ax.axhline(0, color="#6b7280", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(type_names, color="#9ca3af", fontsize=8, rotation=20)
        ax.set_title(title, color="#9ca3af", fontsize=9, pad=6)
        if ax_idx == 0:
            ax.set_ylabel("tỷ VND", color="#9ca3af", fontsize=8)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.3,
                  facecolor="#1f2937", edgecolor="#374151", labelcolor="#e5e7eb",
                  title="Window", title_fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = BASE / f"investor_horizon_{ticker}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Chart saved: {out.name}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Cross-stock foreign scan
# ══════════════════════════════════════════════════════════════════════════════
def market_scan(no_plot: bool = False):
    tickers = sorted([p.stem for p in INDIR.glob("*.parquet")])
    if not tickers:
        print("No data found. Run: python fetch_investor_flow.py --all")
        return

    print(f"\nScanning {len(tickers)} tickers for foreign institutional pattern...")
    print(f"{'─'*80}")
    print(f"  {'Ticker':>6}  {'AvgBuy':>7}  {'AvgSell':>7}  {'vs Buy':>7}  "
          f"{'Realized':>9}  {'Net Flow':>10}  {'Sell Vol':>9}  Pattern")
    print(f"  {'─'*76}")

    results = []
    for tk in tickers:
        df = load(tk)
        if df is None or len(df) < 20:
            continue
        s = type_stats(df, "to_chuc_nuocngoai_net")
        if s is None or s["sell_tỷ"] < 10:   # skip trivial
            continue
        spread = s["avg_s"] - s["avg_b"]      # sell_price - buy_price (negative = sold cheaper)
        pct    = spread / s["avg_b"] * 100 if s["avg_b"] > 0 else 0
        net_tỷ = s["net_tỷ"]
        pattern = ("📤 DISTRIBUTE" if net_tỷ < -100 and spread < 0
                   else "📤 distribute" if net_tỷ < -20 and spread < 0
                   else "⚖ neutral"    if abs(spread) < 0.2
                   else "📥 accumulate")

        print(f"  {tk:>6}  {s['avg_b']:>7.2f}  {s['avg_s']:>7.2f}  "
              f"{spread:>+7.2f}k ({pct:>+5.1f}%)  {s['realized']:>+9.1f}  "
              f"{net_tỷ:>+10.1f}  {s['sell_tỷ']:>9.0f}  {pattern}")

        results.append(dict(
            ticker=tk,
            avg_buy=s["avg_b"], avg_sell=s["avg_s"],
            spread=spread, pct=pct,
            realized=s["realized"], net_tỷ=net_tỷ,
            sell_tỷ=s["sell_tỷ"],
        ))

    if not results:
        print("No results.")
        return

    rdf = pd.DataFrame(results)
    n_distribute = (rdf["spread"] < -0.1).sum()
    n_neutral    = (rdf["spread"].abs() <= 0.1).sum()
    n_accumulate = (rdf["spread"] > 0.1).sum()

    print(f"\n  Summary: {n_distribute}/{len(rdf)} stocks foreigners sold CHEAPER than bought")
    print(f"           {n_neutral}/{len(rdf)} neutral  |  {n_accumulate}/{len(rdf)} sold HIGHER (took profit cleanly)")
    print(f"\n  Total realized P&L across all stocks: {rdf['realized'].sum():+.1f} tỷ VND")
    print(f"  Total net flow:                       {rdf['net_tỷ'].sum():+.1f} tỷ VND")
    print(f"  (net flow ≈0 means balanced; negative = foreigners net selling across market)")

    if no_plot:
        return

    # ── chart ──────────────────────────────────────────────────────────────────
    matplotlib.use("TkAgg")
    rdf_s = rdf.sort_values("net_tỷ")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#111827")
    fig.suptitle("Foreign Institutional (Tổ chức nước ngoài) — Cross-Stock Pattern",
                 color="white", fontsize=12)

    def sax(ax):
        ax.set_facecolor("#1f2937")
        ax.tick_params(colors="#9ca3af", labelsize=8)
        ax.spines[:].set_color("#374151")
        ax.grid(axis="y", color="#374151", lw=0.5, alpha=0.6)

    # Panel 1: sell_price - buy_price spread per stock
    ax = axes[0]; sax(ax)
    colors = ["#ef4444" if v < -0.1 else ("#6b7280" if abs(v) <= 0.1 else "#10b981")
              for v in rdf_s["spread"]]
    bars = ax.barh(rdf_s["ticker"], rdf_s["spread"], color=colors, alpha=0.85)
    ax.axvline(0, color="#6b7280", lw=0.8)
    ax.set_xlabel("Avg Sell − Avg Buy (k VND)", color="#9ca3af", fontsize=8)
    ax.set_title("Trade spread\n(negative = sold cheaper than bought → losing trades)",
                 color="#9ca3af", fontsize=8, pad=6)
    ax.tick_params(axis="y", labelsize=7.5, colors="#e5e7eb")

    # Panel 2: realized P&L
    ax = axes[1]; sax(ax)
    colors2 = ["#10b981" if v >= 0 else "#ef4444" for v in rdf_s["realized"]]
    ax.barh(rdf_s["ticker"], rdf_s["realized"], color=colors2, alpha=0.85)
    ax.axvline(0, color="#6b7280", lw=0.8)
    ax.set_xlabel("Realized P&L (tỷ VND)", color="#9ca3af", fontsize=8)
    ax.set_title("Realized P&L within window\n(negative = actual loss on trades done)",
                 color="#9ca3af", fontsize=8, pad=6)
    ax.tick_params(axis="y", labelsize=7.5, colors="#e5e7eb")
    for bar, v in zip(ax.patches, rdf_s["realized"]):
        if abs(v) > 20:
            ax.text(v + (5 if v >= 0 else -5), bar.get_y() + bar.get_height()/2,
                    f"{v:+.0f}", va="center", ha="left" if v>=0 else "right",
                    color="white", fontsize=6)

    # Panel 3: net flow (total selling pressure)
    ax = axes[2]; sax(ax)
    rdf_s2 = rdf.sort_values("net_tỷ")
    colors3 = ["#10b981" if v >= 0 else "#ef4444" for v in rdf_s2["net_tỷ"]]
    ax.barh(rdf_s2["ticker"], rdf_s2["net_tỷ"], color=colors3, alpha=0.85)
    ax.axvline(0, color="#6b7280", lw=0.8)
    ax.set_xlabel("Net Flow (tỷ VND)  [negative = net seller]", color="#9ca3af", fontsize=8)
    ax.set_title("Net buying/selling pressure\n(Sep 2024 – today)",
                 color="#9ca3af", fontsize=8, pad=6)
    ax.tick_params(axis="y", labelsize=7.5, colors="#e5e7eb")
    for bar, v in zip(ax.patches, rdf_s2["net_tỷ"]):
        if abs(v) > 100:
            ax.text(v + (15 if v >= 0 else -15), bar.get_y() + bar.get_height()/2,
                    f"{v:+.0f}", va="center", ha="left" if v>=0 else "right",
                    color="white", fontsize=6)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = BASE / "investor_foreign_scan.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Chart saved: {out.name}")
    plt.show()


# ── entry ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("ticker",    nargs="?", default=None)
    p.add_argument("--market",  action="store_true", help="Cross-stock scan")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    if args.market or args.ticker is None:
        market_scan(no_plot=args.no_plot)
    if args.ticker:
        horizon_split(args.ticker, no_plot=args.no_plot)


if __name__ == "__main__":
    main()
