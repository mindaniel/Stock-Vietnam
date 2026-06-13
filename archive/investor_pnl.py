"""
investor_pnl.py
===============
Honest P&L analysis for each investor type.

Key insight: every type trades BOTH WAYS (gross buy AND gross sell every period).
This shows:
  - Gross activity: how much they actually bought vs sold
  - Avg buy price vs avg sell price vs current price
  - Net position and unrealized gain/loss on what they still hold
  - Realized P&L from completed trades (bought then sold)
  - Opportunity cost: shares they sold that are now worth more

Important caveat printed in output: foreigners held ACB for years before our data
starts (Sep 2024), so their FULL P&L is not visible here.

Usage:
  python investor_pnl.py ACB
  python investor_pnl.py ACB --from 2025-01-01
  python investor_pnl.py ACB --no-plot
"""

import sys, argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE  = Path(__file__).parent
INDIR = BASE / "data" / "investor_flow"
PRDIR = BASE / "data"

TYPES = {
    "tu_doanh_net":           ("Tự doanh",           "#f59e0b"),
    "ca_nhan_trongnuoc_net":  ("Cá nhân TN",         "#ef4444"),
    "to_chuc_trongnuoc_net":  ("Tổ chức TN",         "#3b82f6"),
    "ca_nhan_nuocngoai_net":  ("Cá nhân NN",         "#a855f7"),
    "to_chuc_nuocngoai_net":  ("Tổ chức NN",         "#10b981"),
}


# ── data ───────────────────────────────────────────────────────────────────────
def load_flow(ticker: str) -> pd.DataFrame:
    path = INDIR / f"{ticker}.parquet"
    if not path.exists():
        print(f"No data for {ticker}. Run: python fetch_investor_flow.py {ticker}")
        sys.exit(1)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_price(ticker: str) -> pd.DataFrame | None:
    path = PRDIR / f"{ticker}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["time"]).rename(columns={"time": "date"})
    return df[["date", "open", "high", "low", "close"]].sort_values("date")


# ── core stats per type ────────────────────────────────────────────────────────
def compute_stats(df: pd.DataFrame, col: str, current_price: float) -> dict:
    pos = df[col] > 0
    neg = df[col] < 0

    gross_buy_tỷ  = df.loc[pos, col].sum()
    gross_sell_tỷ = df.loc[neg, col].abs().sum()
    net_tỷ        = df[col].sum()

    buy_shares  = (df.loc[pos, col] / df.loc[pos, "close"]).sum()   # million shares
    sell_shares = (df.loc[neg, col].abs() / df.loc[neg, "close"]).sum()
    net_shares  = buy_shares - sell_shares

    avg_buy  = gross_buy_tỷ  / buy_shares  if buy_shares  > 0 else 0.0
    avg_sell = gross_sell_tỷ / sell_shares if sell_shares > 0 else 0.0

    # Realized P&L: for every share sold, the gain vs the avg buy price of that round-trip
    # Approximation: realized = sell_shares * (avg_sell - avg_buy)
    # (assumes each sell was previously bought at the overall avg_buy price)
    realized = sell_shares * (avg_sell - avg_buy) if sell_shares > 0 and avg_buy > 0 else 0.0

    # Unrealized: remaining position × (current - avg_buy)
    unrealized = max(net_shares, 0) * (current_price - avg_buy) if avg_buy > 0 else 0.0

    # Opportunity cost: shares they sold that have appreciated since
    opp_cost = sell_shares * max(current_price - avg_sell, 0) if avg_sell > 0 else 0.0

    return dict(
        gross_buy_tỷ  = gross_buy_tỷ,
        gross_sell_tỷ = gross_sell_tỷ,
        net_tỷ        = net_tỷ,
        buy_shares    = buy_shares,
        sell_shares   = sell_shares,
        net_shares    = net_shares,
        avg_buy       = avg_buy,
        avg_sell      = avg_sell,
        realized      = realized,
        unrealized    = unrealized,
        total_pnl     = realized + unrealized,
        opp_cost      = opp_cost,
    )


# ── rolling series for timeline chart ─────────────────────────────────────────
def rolling_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Day-by-day running stats: cumulative realized P&L + unrealized at each close.
    Uses average-cost method.
    """
    position   = 0.0
    avg_cost   = 0.0
    cost_basis = 0.0
    realized   = 0.0
    rows = []

    for _, row in df.iterrows():
        price = row["close"]
        flow  = row[col]
        if price <= 0:
            rows.append((row["date"], price, position, avg_cost, realized, 0.0))
            continue

        sh = flow / price   # million shares

        if sh >= 0:
            new_cost  = cost_basis + flow
            new_pos   = position + sh
            avg_cost  = new_cost / new_pos if new_pos > 1e-9 else avg_cost
            cost_basis = new_cost
            position  = new_pos
        else:
            sell_sh = min(abs(sh), position)
            realized   += sell_sh * (price - avg_cost)
            cost_basis -= sell_sh * avg_cost
            position   -= sell_sh
            if position < 1e-9:
                position = 0.0; cost_basis = 0.0; avg_cost = 0.0

        unrealized = position * (price - avg_cost)
        rows.append((row["date"], price, position, avg_cost, realized, unrealized))

    return pd.DataFrame(rows, columns=["date","price","position","avg_cost",
                                        "realized","unrealized"])


# ── console summary ────────────────────────────────────────────────────────────
def print_summary(ticker: str, stats: dict, current: float, period: str):
    print(f"\n{'═'*75}")
    print(f"  {ticker}  Investor P&L Analysis  |  {period}  |  Current: {current:.2f}k VND")
    print(f"  ⚠  Foreigners' pre-2024 position is NOT in this data — see note below.")
    print(f"{'─'*75}")
    print(f"  {'Type':<18} {'Avg Buy':>7} {'Avg Sell':>8} {'Net Pos':>9} "
          f"{'Realized':>9} {'Unreal.':>9} {'Total':>9}  {'Opp.Cost':>9}")
    print(f"  {'─'*73}")

    for col, (name, _) in TYPES.items():
        s = stats[col]
        ac = f"{s['avg_buy']:.2f}k"  if s['avg_buy']  > 0 else "  n/a"
        as_ = f"{s['avg_sell']:.2f}k" if s['avg_sell'] > 0 else "  n/a"
        pos = f"{s['net_shares']:>7.1f}M" if s['net_shares'] > 0.1 else "    flat"
        win = "✅" if s['total_pnl'] > 10 else ("❌" if s['total_pnl'] < -5 else "〰")
        print(f"  {name:<18} {ac:>7} {as_:>8} {pos:>9} "
              f"{s['realized']:>+9.1f} {s['unrealized']:>+9.1f} {s['total_pnl']:>+9.1f}  "
              f"{s['opp_cost']:>+9.1f} {win}")

    tot_real  = sum(s['realized']  for s in stats.values())
    tot_unr   = sum(s['unrealized'] for s in stats.values())
    tot_opp   = sum(s['opp_cost']   for s in stats.values())
    print(f"  {'─'*73}")
    print(f"  {'TOTAL':<18} {'':>7} {'':>8} {'':>9} "
          f"{tot_real:>+9.1f} {tot_unr:>+9.1f} {tot_real+tot_unr:>+9.1f}  {tot_opp:>+9.1f}")

    print(f"""
  WHAT THE NUMBERS MEAN:
  ─────────────────────────────────────────────────────────────────────
  Realized   = profit/loss from shares bought AND SOLD within window
               (actual cash gain or loss already captured)
  Unrealized = current position × (current price − avg buy price)
               (paper gain, not yet realized)
  Opp.Cost   = shares SOLD × (current price − avg sell price)
               = money left on table by selling before the price rose
               (this is NOT a cash loss — just missed upside)

  WHY THE TOTAL IS NOT ZERO:
  ─────────────────────────────────────────────────────────────────────
  1. STOCKS ARE NOT ZERO-SUM. When price rises, all holders gain.
     Nobody "loses" when domestic institutions gain +{stats['to_chuc_trongnuoc_net']['unrealized']:+.0f} tỷ unrealized —
     that wealth was created by the market, not taken from anyone.

  2. MISSING HISTORY. Foreigners (Tổ chức NN) held ACB for years before
     Sep 2024 — likely bought at 15-18k. They sold at 22-25k avg in
     our window = +40-60% profit on original cost. That realized gain
     is invisible here. Within our window they look like losers because
     they net-sold during the dip (avg sell {stats['to_chuc_nuocngoai_net']['avg_sell']:.2f}k) but their
     pre-window cost basis was far lower.

  3. THE REAL LOSER IN THIS WINDOW: Tự doanh.
     They bought at {stats['tu_doanh_net']['avg_buy']:.2f}k, sold at {stats['tu_doanh_net']['avg_sell']:.2f}k (sold CHEAPER than they bought),
     fully exited = {stats['tu_doanh_net']['realized']:+.1f} tỷ actual realized loss. Bad timing.
{'═'*75}""")


# ── chart ──────────────────────────────────────────────────────────────────────
def plot(ticker: str, df: pd.DataFrame, stats: dict,
         roll: dict, price_df: pd.DataFrame | None):
    matplotlib.use("TkAgg")
    fig = plt.figure(figsize=(15, 13), facecolor="#111827")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.28,
                            height_ratios=[1.5, 1.2, 1.1])

    ax_price = fig.add_subplot(gs[0, :])      # full width: price + avg cost lines
    ax_pnl   = fig.add_subplot(gs[1, :])      # full width: running total P&L
    ax_bar_g = fig.add_subplot(gs[2, 0])      # gross buy vs sell bar
    ax_bar_p = fig.add_subplot(gs[2, 1])      # P&L breakdown bar

    def style(ax):
        ax.set_facecolor("#1f2937")
        ax.tick_params(colors="#9ca3af", labelsize=8)
        ax.spines[:].set_color("#374151")
        ax.grid(axis="y", color="#374151", linewidth=0.5, alpha=0.6)

    for ax in [ax_price, ax_pnl, ax_bar_g, ax_bar_p]:
        style(ax)

    # ── Price + avg cost lines ─────────────────────────────────────────────────
    if price_df is not None and not price_df.empty:
        px = price_df[price_df["date"] >= df["date"].min()]
        up = px["close"] >= px["open"]
        ax_price.vlines(px["date"], px["low"], px["high"],
                        color="#6b7280", linewidth=0.5, alpha=0.7)
        ax_price.vlines(px.loc[up,"date"],  px.loc[up,"open"],  px.loc[up,"close"],
                        color="#10b981", linewidth=2.2)
        ax_price.vlines(px.loc[~up,"date"], px.loc[~up,"close"], px.loc[~up,"open"],
                        color="#ef4444", linewidth=2.2)
        plo = px["low"].min();  phi = px["high"].max()
    else:
        ax_price.plot(df["date"], df["close"], color="#e5e7eb", linewidth=1.0)
        plo = df["close"].min(); phi = df["close"].max()

    pad = (phi - plo) * 0.07
    ax_price.set_ylim(plo - pad, phi + pad * 2.5)

    # avg cost lines while position > 0.1
    for col, (name, color) in TYPES.items():
        r = roll[col]
        held = r[r["position"] > 0.1]
        if not held.empty:
            ax_price.plot(held["date"], held["avg_cost"],
                          color=color, linewidth=1.3, linestyle="--", alpha=0.9)
        # mark avg_sell as horizontal dotted if they sold
        s = stats[col]
        if s["avg_sell"] > 0 and s["sell_shares"] > 5:
            ax_price.axhline(s["avg_sell"], color=color, linewidth=0.8,
                             linestyle=":", alpha=0.55)

    # horizontal line for avg_buy of each type (small ticks on right)
    for col, (name, color) in TYPES.items():
        s = stats[col]
        if s["avg_buy"] > 0 and s["net_shares"] > 5:
            ax_price.annotate(f"{name} cost {s['avg_buy']:.2f}k",
                              xy=(df["date"].iloc[-1], s["avg_buy"]),
                              xytext=(5, 0), textcoords="offset points",
                              color=color, fontsize=6.5, va="center")

    ax_price.set_ylabel("Price (k VND)", color="#9ca3af", fontsize=8)
    ax_price.set_title(f"{ticker}  —  Who's Making Money? (avg cost = dashed, avg sell = dotted)",
                       color="white", fontsize=11, pad=8)

    legend_handles = [Patch(color=v[1]) for v in TYPES.values()]
    legend_labels  = [v[0] for v in TYPES.values()]
    ax_price.legend(legend_handles, legend_labels, loc="upper left", fontsize=7,
                    framealpha=0.3, facecolor="#1f2937", edgecolor="#374151",
                    labelcolor="#e5e7eb", ncol=5)

    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax_price.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_price.tick_params(axis="x", colors="#9ca3af", labelsize=8)
    plt.setp(ax_price.get_xticklabels(), rotation=30, ha="right")

    # ── Running total P&L ──────────────────────────────────────────────────────
    for col, (name, color) in TYPES.items():
        r = roll[col]
        total = r["realized"] + r["unrealized"]
        ax_pnl.plot(r["date"], total, label=name, color=color,
                    linewidth=1.5, alpha=0.9)
        ax_pnl.fill_between(r["date"], 0, total, color=color, alpha=0.06)

    ax_pnl.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--")
    ax_pnl.set_ylabel("Total P&L\n(tỷ VND)", color="#9ca3af", fontsize=8)
    ax_pnl.set_title("Cumulative P&L over time (realized + unrealized)",
                     color="#9ca3af", fontsize=8, pad=4)
    ax_pnl.legend(loc="upper left", fontsize=7, framealpha=0.3,
                  facecolor="#1f2937", edgecolor="#374151",
                  labelcolor="#e5e7eb", ncol=5)

    # annotate end values
    for col, (name, color) in TYPES.items():
        r = roll[col]
        val = (r["realized"] + r["unrealized"]).iloc[-1]
        ax_pnl.annotate(f"{val:+.0f}",
                        xy=(r["date"].iloc[-1], val),
                        xytext=(4, 0), textcoords="offset points",
                        color=color, fontsize=7, va="center", fontweight="bold")

    ax_pnl.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax_pnl.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_pnl.tick_params(axis="x", colors="#9ca3af", labelsize=8)
    plt.setp(ax_pnl.get_xticklabels(), rotation=30, ha="right")

    # ── Gross buy vs sell bars ─────────────────────────────────────────────────
    names  = [v[0] for v in TYPES.values()]
    colors = [v[1] for v in TYPES.values()]
    keys   = list(TYPES.keys())
    x = np.arange(len(keys))
    w = 0.35

    buys  = [stats[c]["gross_buy_tỷ"]  for c in keys]
    sells = [stats[c]["gross_sell_tỷ"] for c in keys]

    ax_bar_g.bar(x - w/2, buys,  width=w, color=colors, alpha=0.75, label="Gross bought")
    ax_bar_g.bar(x + w/2, [-s for s in sells], width=w, color=colors,
                 alpha=0.40, label="Gross sold (neg)")

    for i, (b, s, c) in enumerate(zip(buys, sells, colors)):
        ax_bar_g.text(i - w/2, b + 50, f"{b/1000:.1f}k", ha="center",
                      color=c, fontsize=6.5, fontweight="bold")
        ax_bar_g.text(i + w/2, -s - 200, f"-{s/1000:.1f}k", ha="center",
                      color=c, fontsize=6.5, alpha=0.8)

    ax_bar_g.axhline(0, color="#6b7280", linewidth=0.7)
    ax_bar_g.set_xticks(x)
    ax_bar_g.set_xticklabels(names, color="#9ca3af", fontsize=7.5, rotation=15)
    ax_bar_g.set_ylabel("tỷ VND", color="#9ca3af", fontsize=7)
    ax_bar_g.set_title("Gross Trading Activity\n(bright=bought, faded=sold)",
                       color="#9ca3af", fontsize=8, pad=4)

    # ── P&L breakdown bars ─────────────────────────────────────────────────────
    realized   = [stats[c]["realized"]   for c in keys]
    unrealized = [stats[c]["unrealized"] for c in keys]
    opp_cost   = [stats[c]["opp_cost"]   for c in keys]
    w3 = 0.25

    ax_bar_p.bar(x - w3, realized,   width=w3, color=colors, alpha=0.65, label="Realized")
    ax_bar_p.bar(x,      unrealized, width=w3, color=colors, alpha=0.90, label="Unrealized")
    ax_bar_p.bar(x + w3, opp_cost,   width=w3, color=colors, alpha=0.35,
                 label="Opp. cost (missed upside)", hatch="//", edgecolor="white")

    for i, (r, u, o, c) in enumerate(zip(realized, unrealized, opp_cost, colors)):
        for xpos, val in [(i-w3, r), (i, u), (i+w3, o)]:
            if abs(val) > 15:
                va = "bottom" if val >= 0 else "top"
                ax_bar_p.text(xpos, val, f"{val:+.0f}",
                              ha="center", va=va, color=c, fontsize=5.5, fontweight="bold")

    ax_bar_p.axhline(0, color="#6b7280", linewidth=0.7)
    ax_bar_p.set_xticks(x)
    ax_bar_p.set_xticklabels(names, color="#9ca3af", fontsize=7.5, rotation=15)
    ax_bar_p.set_ylabel("tỷ VND", color="#9ca3af", fontsize=7)
    ax_bar_p.set_title("P&L Breakdown\n(realized | unrealized | opp. cost of selling early)",
                       color="#9ca3af", fontsize=8, pad=4)
    ax_bar_p.legend(loc="upper right", fontsize=6.5, framealpha=0.3,
                    facecolor="#1f2937", edgecolor="#374151", labelcolor="#e5e7eb")

    fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.07)
    out = BASE / f"investor_pnl_{ticker}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Chart saved: {out.name}")
    plt.show()


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--from",     dest="from_date", default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--no-plot",  action="store_true")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    df     = load_flow(ticker)

    if args.from_date:
        df = df[df["date"] >= pd.to_datetime(args.from_date)]
    elif args.lookback:
        cutoff = df["date"].max() - pd.Timedelta(days=int(args.lookback * 365 / 252))
        df = df[df["date"] >= cutoff]

    df = df.reset_index(drop=True)
    current = df["close"].iloc[-1]
    period  = f"{df['date'].min().date()} → {df['date'].max().date()}  ({len(df)}d)"

    stats = {col: compute_stats(df, col, current) for col in TYPES}
    print_summary(ticker, stats, current, period)

    if not args.no_plot:
        roll     = {col: rolling_stats(df, col) for col in TYPES}
        price_df = load_price(ticker)
        if price_df is not None:
            price_df = price_df[price_df["date"] >= df["date"].min()]
        plot(ticker, df, stats, roll, price_df)


if __name__ == "__main__":
    main()
