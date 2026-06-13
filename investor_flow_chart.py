"""
investor_flow_chart.py
======================
Shows 5 investor type net flows for a stock alongside price.

Usage:
  python investor_flow_chart.py ACB
  python investor_flow_chart.py ACB --lookback 90
  python investor_flow_chart.py ACB --from 2025-01-01
  python investor_flow_chart.py ACB --no-plot   (print summary only)

Panels:
  1. Price (proper y-axis + colour-coded candle direction)
  2. Cumulative net flow + price overlay on secondary axis
  3. Daily net flow bars (stacked, so you see net buyer/seller each day)
  4. Rolling 20-day balance score

If data is missing, run:  python fetch_investor_flow.py <TICKER>
"""

import sys, argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE   = Path(__file__).parent
INDIR  = BASE / "data" / "investor_flow"
PRDIR  = BASE / "data"

# ── investor type colours ──────────────────────────────────────────────────────
TYPES = {
    "tu_doanh_net":           ("Tự doanh",           "#f59e0b"),   # amber
    "ca_nhan_trongnuoc_net":  ("Cá nhân trong nước", "#ef4444"),   # red
    "to_chuc_trongnuoc_net":  ("Tổ chức trong nước", "#3b82f6"),   # blue
    "to_chuc_nuocngoai_net":  ("Tổ chức nước ngoài", "#10b981"),   # teal
}
# ca_nhan_nuocngoai is included in stacked bar (rounds to ~0 net) but not in legend

# ── loaders ────────────────────────────────────────────────────────────────────
def load_flow(ticker: str) -> pd.DataFrame:
    path = INDIR / f"{ticker}.parquet"
    if not path.exists():
        print(f"No investor flow data for {ticker}.")
        print(f"Run:  python fetch_investor_flow.py {ticker}")
        sys.exit(1)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_price(ticker: str) -> pd.DataFrame | None:
    path = PRDIR / f"{ticker}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date")


# ── helpers ────────────────────────────────────────────────────────────────────
def balance_score(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """20-day rolling: (domestic_inst + foreign_inst) minus (retail + tu_doanh)."""
    smart  = df["to_chuc_trongnuoc_net"] + df["to_chuc_nuocngoai_net"]
    retail = df["ca_nhan_trongnuoc_net"] + df["tu_doanh_net"]
    return (smart - retail).rolling(window).sum()


def _ax_style(ax):
    ax.set_facecolor("#1f2937")
    ax.tick_params(colors="#9ca3af", labelsize=8)
    ax.spines[:].set_color("#374151")
    ax.grid(axis="y", color="#374151", linewidth=0.5, alpha=0.6)


def _fmt_xaxis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.tick_params(axis="x", colors="#9ca3af", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


# ── console summary ────────────────────────────────────────────────────────────
def print_summary(ticker: str, df: pd.DataFrame):
    last = df.iloc[-1]
    print(f"\n{'═'*65}")
    print(f"  {ticker}  Investor Flow  |  {df['date'].min().date()} → {df['date'].max().date()}  ({len(df)}d)")
    print(f"  Last close: {last['close']:.2f}k VND")
    print(f"{'─'*65}")
    print(f"  {'Type':<26} {'20d':>8} {'60d':>8} {'All-time':>10}  Trend")
    print(f"  {'─'*62}")

    cols_all  = list(TYPES.keys()) + ["ca_nhan_nuocngoai_net"]
    totals = {}
    for col, (name, _) in TYPES.items():
        t20  = df[col].tail(20).sum()
        t60  = df[col].tail(60).sum()
        tall = df[col].sum()
        totals[col] = (t20, t60, tall)
        # trend: compare 20d vs 60d slope
        slope = "▲" if t20 > 0 else "▼"
        print(f"  {name:<26} {t20:>+8.1f} {t60:>+8.1f} {tall:>+10.1f}  {slope}")

    nn = df["ca_nhan_nuocngoai_net"]
    print(f"  {'Cá nhân nước ngoài':<26} {nn.tail(20).sum():>+8.1f} {nn.tail(60).sum():>+8.1f} {nn.sum():>+10.1f}  {'▲' if nn.sum()>0 else '▼'}")

    # net buyer vs seller summary
    print(f"{'─'*65}")
    smart_20  = df["to_chuc_trongnuoc_net"].tail(20).sum() + df["to_chuc_nuocngoai_net"].tail(20).sum()
    retail_20 = df["ca_nhan_trongnuoc_net"].tail(20).sum() + df["tu_doanh_net"].tail(20).sum()
    total_buy_20  = sum(v for v in [df[c].tail(20).sum() for c in cols_all] if v > 0)
    total_sell_20 = sum(v for v in [df[c].tail(20).sum() for c in cols_all] if v < 0)

    print(f"  20d gross buying  : {total_buy_20:>+8.1f} tỷ")
    print(f"  20d gross selling : {total_sell_20:>+8.1f} tỷ")
    print(f"  20d net           : {total_buy_20+total_sell_20:>+8.1f} tỷ  (≈0 by definition)")
    print(f"{'─'*65}")

    if smart_20 > 0 and retail_20 < 0:
        signal = "✅ ACCUMULATION — institutions buying, retail selling"
    elif smart_20 < 0 and retail_20 > 0:
        signal = "⚠️  DISTRIBUTION — institutions selling, retail buying"
    elif smart_20 > 0 and retail_20 > 0:
        signal = "📈 BROAD BUYING"
    else:
        signal = "📉 BROAD SELLING"
    print(f"  20d signal: {signal}")
    print(f"{'═'*65}\n")


# ── main chart ─────────────────────────────────────────────────────────────────
def plot_investor_flow(ticker: str, lookback: int = 120,
                       from_date: str | None = None,
                       no_plot: bool = False):
    ticker = ticker.upper()
    df     = load_flow(ticker)

    # date filter
    if from_date:
        df = df[df["date"] >= pd.to_datetime(from_date)]
    else:
        cutoff = df["date"].max() - pd.Timedelta(days=int(lookback * 365 / 252))
        df = df[df["date"] >= cutoff]

    df = df.reset_index(drop=True)
    if df.empty:
        print("No data in selected range.")
        return

    price_df = load_price(ticker)
    if price_df is not None:
        price_df = price_df[price_df["date"] >= df["date"].min()].reset_index(drop=True)

    print_summary(ticker, df)
    if no_plot:
        return

    # ── figure layout ──────────────────────────────────────────────────────────
    matplotlib.use("TkAgg")
    fig = plt.figure(figsize=(14, 12), facecolor="#111827")
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.08,
                            height_ratios=[1.4, 1.2, 1.2, 0.8])

    ax_price = fig.add_subplot(gs[0])
    ax_cum   = fig.add_subplot(gs[1], sharex=ax_price)
    ax_daily = fig.add_subplot(gs[2], sharex=ax_price)
    ax_score = fig.add_subplot(gs[3], sharex=ax_price)

    for ax in [ax_price, ax_cum, ax_daily, ax_score]:
        _ax_style(ax)

    # ── Panel 1: Price (proper y-axis) ─────────────────────────────────────────
    pdata = price_df if (price_df is not None and not price_df.empty) else df.rename(columns={"close": "close"})
    # Use the flow df's close if no separate price file
    if price_df is not None and not price_df.empty:
        px = price_df
    else:
        px = df[["date", "close"]].copy()
        px["open"] = px["close"]

    # colour bars: green if close >= open, red if close < open
    if "open" in px.columns:
        up   = px["close"] >= px["open"]
        for _, row in px.iterrows():
            c = "#10b981" if row["close"] >= row.get("open", row["close"]) else "#ef4444"
            ax_price.vlines(row["date"], row.get("low", row["close"]),
                            row.get("high", row["close"]), color=c, linewidth=0.6, alpha=0.6)
        ax_price.vlines(px.loc[up,  "date"], px.loc[up,  "open"], px.loc[up,  "close"],
                        color="#10b981", linewidth=2.5)
        ax_price.vlines(px.loc[~up, "date"], px.loc[~up, "close"], px.loc[~up, "open"],
                        color="#ef4444", linewidth=2.5)
    else:
        ax_price.plot(px["date"], px["close"], color="#e5e7eb", linewidth=1.2)

    # ← KEY FIX: y-axis range based on actual price, not from 0
    plo = px["low"].min()  if "low"  in px.columns else px["close"].min()
    phi = px["high"].max() if "high" in px.columns else px["close"].max()
    pad = (phi - plo) * 0.06
    ax_price.set_ylim(plo - pad, phi + pad)

    ax_price.set_ylabel("Price (k VND)", color="#9ca3af", fontsize=8)
    ax_price.set_title(f"{ticker}  —  Investor Type Net Flow (tỷ VND)",
                       color="white", fontsize=12, pad=8)
    plt.setp(ax_price.get_xticklabels(), visible=False)

    # annotate latest price
    ax_price.annotate(f"{df['close'].iloc[-1]:.2f}k",
                      xy=(df["date"].iloc[-1], df["close"].iloc[-1]),
                      xytext=(-45, 6), textcoords="offset points",
                      color="white", fontsize=8, fontweight="bold")

    # ── Panel 2: Cumulative + price overlay ────────────────────────────────────
    for col, (name, color) in TYPES.items():
        cum = df[col].cumsum()
        ax_cum.plot(df["date"], cum, label=name, color=color, linewidth=1.5, alpha=0.9)
        ax_cum.fill_between(df["date"], 0, cum, color=color, alpha=0.06)

    ax_cum.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--")
    ax_cum.set_ylabel("Cumulative\n(tỷ VND)", color="#9ca3af", fontsize=8)

    # Price secondary axis on cumulative panel
    ax2 = ax_cum.twinx()
    ax2.set_facecolor("none")
    ax2.plot(df["date"], df["close"], color="white", linewidth=1.0,
             alpha=0.55, linestyle="-", zorder=2)
    ax2.set_ylabel("Price (k)", color="#6b7280", fontsize=7)
    ax2.tick_params(colors="#6b7280", labelsize=7)
    ax2.spines[:].set_color("#374151")
    lo2 = df["close"].min(); hi2 = df["close"].max()
    ax2.set_ylim(lo2 - (hi2 - lo2) * 0.1, hi2 + (hi2 - lo2) * 0.5)  # keep price near top

    ax_cum.legend(loc="upper left", fontsize=7, framealpha=0.3,
                  facecolor="#1f2937", edgecolor="#374151",
                  labelcolor="#e5e7eb", ncol=2)
    plt.setp(ax_cum.get_xticklabels(), visible=False)

    # ── Panel 3: Stacked daily bars ────────────────────────────────────────────
    # Separate positive and negative contributions per day → stacked
    all_cols = list(TYPES.keys()) + ["ca_nhan_nuocngoai_net"]
    all_colors = [v[1] for v in TYPES.values()] + ["#a855f7"]  # purple for foreign retail

    pos_bottom = np.zeros(len(df))
    neg_bottom = np.zeros(len(df))

    for col, color in zip(all_cols, all_colors):
        vals = df[col].values
        pos_vals = np.where(vals > 0, vals, 0)
        neg_vals = np.where(vals < 0, vals, 0)

        if pos_vals.any():
            ax_daily.bar(df["date"], pos_vals, bottom=pos_bottom,
                         color=color, alpha=0.85, linewidth=0, width=0.7)
        if neg_vals.any():
            ax_daily.bar(df["date"], neg_vals, bottom=neg_bottom,
                         color=color, alpha=0.85, linewidth=0, width=0.7)

        pos_bottom += pos_vals
        neg_bottom += neg_vals

    # Net line overlay (should be ~0 every day by definition)
    daily_net = df[all_cols].sum(axis=1)
    ax_daily.plot(df["date"], daily_net, color="white", linewidth=0.5, alpha=0.4, label="Net")

    ax_daily.axhline(0, color="#6b7280", linewidth=0.9)
    ax_daily.set_ylabel("Daily Flow\n(tỷ VND)", color="#9ca3af", fontsize=8)

    # legend for stacked bars
    type_labels = [v[0] for v in TYPES.values()] + ["Cá nhân NN"]
    handles = [Patch(color=c, alpha=0.85) for c in all_colors]
    ax_daily.legend(handles, type_labels, loc="upper left", fontsize=6,
                    framealpha=0.3, facecolor="#1f2937", edgecolor="#374151",
                    labelcolor="#e5e7eb", ncol=3)
    plt.setp(ax_daily.get_xticklabels(), visible=False)

    # ── Panel 4: Balance score ─────────────────────────────────────────────────
    score = balance_score(df, window=20)
    pos_m = score >= 0
    ax_score.fill_between(df["date"],  0, score, where=pos_m,
                          color="#10b981", alpha=0.75, label="Inst. dominant (+)")
    ax_score.fill_between(df["date"],  0, score, where=~pos_m,
                          color="#ef4444", alpha=0.75, label="Retail dominant (−)")
    ax_score.axhline(0, color="#6b7280", linewidth=0.8)
    ax_score.set_ylabel("20d Balance\n(tỷ VND)", color="#9ca3af", fontsize=8)
    ax_score.legend(loc="upper left", fontsize=7, framealpha=0.3,
                    facecolor="#1f2937", edgecolor="#374151",
                    labelcolor="#e5e7eb")

    _fmt_xaxis(ax_score)

    # ── annotations on last bar ───────────────────────────────────────────────
    last_d = df["date"].iloc[-1]
    for col, (name, color) in TYPES.items():
        v = df[col].iloc[-1]
        if abs(v) > 30:
            ax_daily.annotate(f"{v:+.0f}",
                              xy=(last_d, v),
                              xytext=(4, 0), textcoords="offset points",
                              color=color, fontsize=6.5, va="center")

    fig.subplots_adjust(left=0.08, right=0.93, top=0.95, bottom=0.06)
    out = BASE / f"investor_flow_{ticker}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Chart saved: {out.name}")
    plt.show()


# ── entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", help="Stock ticker (e.g. ACB)")
    parser.add_argument("--lookback", type=int, default=120,
                        help="Trading days to show (default 120 ≈ 6 months)")
    parser.add_argument("--from", dest="from_date", default=None,
                        help="Start date YYYY-MM-DD (overrides --lookback)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Print summary only, no chart")
    args = parser.parse_args()

    plot_investor_flow(
        ticker    = args.ticker,
        lookback  = args.lookback,
        from_date = args.from_date,
        no_plot   = args.no_plot,
    )


if __name__ == "__main__":
    main()
