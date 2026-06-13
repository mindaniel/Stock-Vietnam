"""
adaptive_backtest.py
Flexible factor-timing backtest for VN equities.

Each month the composite score weights shift based on which factors have been
working over a trailing lookback window.  Compares:
  1. Adaptive (factor-timed weights)
  2. Static best configs from the grid
  3. Equal-weight market portfolio
  4. VNINDEX buy-and-hold

Usage:
  python adaptive_backtest.py
  python adaptive_backtest.py --lookback 6
  python adaptive_backtest.py --lookback 12 --top 0.20 --cost-bps 15
  python adaptive_backtest.py --start 2018-01-01 --no-regime
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ff_vietnam import (
    load_prices, load_sector_map, load_fundamentals_latest_by_year,
    monthly_stock_panel, attach_accounting, build_factors,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

RF_ANNUAL   = 0.03
FACTORS     = ["quality", "investment", "value", "momentum"]
FACTOR_COLS = {"quality": "profitability", "investment": "asset_growth",
               "value": "bm", "momentum": "mom_12_1"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    m, sd = s.mean(), s.std()
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - m) / sd


def perf_stats(r: pd.Series, label: str) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {}
    eq    = (1 + r).cumprod()
    years = len(r) / 12
    cagr  = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol   = r.std() * np.sqrt(12)
    sharpe = (r.mean() * 12) / vol if vol > 0 else np.nan
    dd    = eq / eq.cummax() - 1
    return {"label": label, "months": len(r), "cagr": cagr,
            "vol": vol, "sharpe": sharpe, "maxdd": dd.min(),
            "calmar": cagr / abs(dd.min()) if dd.min() != 0 else np.nan}


def load_regime() -> pd.DataFrame:
    vn  = pd.read_csv(os.path.join(BASE_DIR, "VNINDEX.csv"))
    vn["date"]  = pd.to_datetime(vn["date"], errors="coerce")
    vn["close"] = pd.to_numeric(vn["close"], errors="coerce")
    vn  = vn.dropna(subset=["date", "close"]).sort_values("date")
    m   = vn.set_index("date")["close"].resample("ME").last().to_frame("close")
    m["ma10"]    = m["close"].rolling(10, min_periods=10).mean()
    m["risk_on"] = (m["close"] >= m["ma10"]).astype(int)
    m["mkt_ret"] = m["close"].pct_change(fill_method=None)
    return m.reset_index()


# ── Adaptive weight engine ────────────────────────────────────────────────────

def adaptive_weights(ff: pd.DataFrame, date: pd.Timestamp,
                     lookback: int, method: str = "momentum") -> dict:
    """
    Given factor returns up to (not including) `date`,
    return a weight dict {factor_name: weight} summing to 1.
    Only factors with positive recent performance get positive weight.
    """
    past = ff[ff["date"] < date].tail(lookback)
    if len(past) < max(3, lookback // 2):
        # Not enough history — use equal weight across all four
        return {f: 0.25 for f in FACTORS}

    # Map factor names to FF column names
    ff_col = {"quality": "RMW", "investment": "CMA",
               "value": "HML", "momentum": "MOM"}

    scores = {}
    for fname, col in ff_col.items():
        series = past[col].dropna()
        if len(series) < 3:
            scores[fname] = 0.0
            continue
        if method == "sharpe":
            mu = series.mean()
            sd = series.std()
            scores[fname] = (mu / sd) if sd > 0 else 0.0
        else:  # cumulative return (momentum on factors)
            scores[fname] = float((1 + series).prod() - 1)

    # Only keep positive scores; if all negative → equal weight
    pos = {k: v for k, v in scores.items() if v > 0}
    if not pos:
        return {f: 0.25 for f in FACTORS}

    total = sum(pos.values())
    return {f: (pos.get(f, 0.0) / total) for f in FACTORS}


# ── Single backtest run ───────────────────────────────────────────────────────

def run_backtest(panel: pd.DataFrame, ff: pd.DataFrame,
                 regime_df: pd.DataFrame | None,
                 static_weights: dict | None,
                 lookback: int, method: str,
                 top: float, min_liq: float, cost_bps: float) -> pd.DataFrame:
    """
    Run one backtest pass.
    If static_weights is None → adaptive mode.
    """
    rf_m    = (1 + RF_ANNUAL) ** (1 / 12) - 1
    rets    = []
    prev_w  = {}

    for dt, g in panel.groupby("date"):
        # Regime gate
        if regime_df is not None:
            row = regime_df.loc[regime_df["date"] == dt, "risk_on"]
            if len(row) == 0 or int(row.iloc[0]) == 0:
                prev_w = {}
                rets.append({"date": dt, "ret": 0.0, "n": 0, "turnover": 0.0,
                              **{f"w_{k}": 0.0 for k in FACTORS}})
                continue

        g = g[g["value"] >= min_liq].dropna(subset=["ret"])
        if len(g) < 20:
            continue

        # Factor weights for this month
        if static_weights is not None:
            w = static_weights
        else:
            w = adaptive_weights(ff, dt, lookback, method)

        # Composite z-score
        comp = pd.Series(0.0, index=g.index)
        wsum = pd.Series(0.0, index=g.index)
        for fname, fweight in w.items():
            if fweight == 0:
                continue
            col = FACTOR_COLS[fname]
            # For investment, higher is WORSE (more aggressive), so invert
            raw = -g[col] if fname == "investment" else g[col]
            z   = zscore(raw)
            ok  = z.notna()
            comp.loc[ok] += fweight * z.loc[ok]
            wsum.loc[ok] += fweight

        g = g.copy()
        g["score"] = comp / wsum.replace(0, np.nan)
        g = g.dropna(subset=["score"])
        if len(g) < 10:
            continue

        # Pick top quantile
        q      = g["score"].quantile(1 - top)
        picks  = g[g["score"] >= q]
        if picks.empty:
            continue

        w_new   = {s: 1 / len(picks) for s in picks["symbol"]}
        gross   = picks["ret"].mean()
        all_s   = set(prev_w) | set(w_new)
        turnover= sum(abs(w_new.get(s, 0) - prev_w.get(s, 0)) for s in all_s)
        cost    = turnover * cost_bps / 10_000
        net     = gross - cost
        prev_w  = w_new

        rets.append({"date": dt, "ret": net, "n": len(picks),
                     "turnover": turnover,
                     **{f"w_{k}": v for k, v in w.items()}})

    return pd.DataFrame(rets).sort_values("date")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",    default="2018-01-01")
    ap.add_argument("--lookback", type=int, default=6,
                    help="Months of factor history to use for timing (default 6)")
    ap.add_argument("--method",   default="momentum",
                    choices=["momentum", "sharpe"],
                    help="How to score factor recency: cumulative return or Sharpe")
    ap.add_argument("--top",      type=float, default=0.25)
    ap.add_argument("--min-liq",  type=float, default=1_000_000)
    ap.add_argument("--cost-bps", type=float, default=15.0)
    ap.add_argument("--no-regime", action="store_true")
    args = ap.parse_args()

    print("Loading data...")
    prices  = load_prices()
    sec_map = load_sector_map()
    monthly = monthly_stock_panel(prices).merge(sec_map, on="symbol", how="left")
    fin     = load_fundamentals_latest_by_year()
    panel   = attach_accounting(monthly, fin)
    panel   = panel[panel["date"] >= pd.Timestamp(args.start)].copy()

    print("Building FF factors...")
    ff = build_factors(panel, rf_annual=RF_ANNUAL)

    regime_df = None if args.no_regime else load_regime()

    print("Running backtests...")

    # ── 1. Adaptive ───────────────────────────────────────────────────────────
    adap = run_backtest(panel, ff, regime_df, None,
                        args.lookback, args.method,
                        args.top, args.min_liq, args.cost_bps)

    # ── 2. Static configs ─────────────────────────────────────────────────────
    static_configs = {
        "Quality+Invest (current best)": {"quality": 0.50, "investment": 0.50,
                                           "value": 0.00, "momentum": 0.00},
        "Quality+Invest+Mom (historical)": {"quality": 0.35, "investment": 0.15,
                                             "value": 0.00, "momentum": 0.50},
        "All factors equal":              {"quality": 0.25, "investment": 0.25,
                                           "value": 0.25, "momentum": 0.25},
    }
    static_results = {}
    for name, sw in static_configs.items():
        r = run_backtest(panel, ff, regime_df, sw,
                         args.lookback, args.method,
                         args.top, args.min_liq, args.cost_bps)
        static_results[name] = r

    # ── 3. VNINDEX ────────────────────────────────────────────────────────────
    vni = load_regime()
    vni = vni[vni["date"] >= pd.Timestamp(args.start)].dropna(subset=["mkt_ret"])

    # ── Align all series ──────────────────────────────────────────────────────
    series = {"Adaptive": adap.set_index("date")["ret"].replace(0, np.nan).dropna()}
    for name, r in static_results.items():
        series[name] = r.set_index("date")["ret"].replace(0, np.nan).dropna()
    series["VNINDEX"] = vni.set_index("date")["mkt_ret"]

    # Cumulative equity curves (include cash months as 0 for regime strategies)
    cum = {}
    for name, r in static_results.items():
        eq = (1 + r.set_index("date")["ret"]).cumprod()
        cum[name] = eq
    eq_adap = (1 + adap.set_index("date")["ret"]).cumprod()
    cum["Adaptive"] = eq_adap
    eq_vni = (1 + vni.set_index("date")["mkt_ret"]).cumprod()
    cum["VNINDEX"] = eq_vni

    # ── Print stats ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"ADAPTIVE FACTOR BACKTEST  (start={args.start}, lookback={args.lookback}M, "
          f"regime={'OFF' if args.no_regime else 'ON'})")
    print("=" * 72)
    print(f"{'Strategy':<38} {'Months':>6} {'CAGR':>7} {'Vol':>6} "
          f"{'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7}")
    print("-" * 72)

    all_stats = []
    for name, r in [("Adaptive", adap)] + list(static_results.items()):
        s = perf_stats(r["ret"], name)
        if not s:
            continue
        all_stats.append(s)
        print(f"{name:<38} {s['months']:>6} {s['cagr']:>7.1%} {s['vol']:>6.1%} "
              f"{s['sharpe']:>7.2f} {s['maxdd']:>7.1%} {s['calmar']:>7.2f}")

    # VNINDEX stats
    s = perf_stats(vni["mkt_ret"], "VNINDEX")
    all_stats.append(s)
    print(f"{'VNINDEX':38} {s['months']:>6} {s['cagr']:>7.1%} {s['vol']:>6.1%} "
          f"{s['sharpe']:>7.2f} {s['maxdd']:>7.1%} {s['calmar']:>7.2f}")
    print("=" * 72)

    # ── Adaptive weight evolution ─────────────────────────────────────────────
    if not adap.empty and "w_quality" in adap.columns:
        print("\nAdaptive weight evolution (recent 12 months):")
        w_cols = [c for c in adap.columns if c.startswith("w_")]
        recent_w = adap.set_index("date")[w_cols].tail(12)
        recent_w.columns = [c.replace("w_", "") for c in w_cols]
        print(recent_w.round(2).to_string())

    # ── Save results ──────────────────────────────────────────────────────────
    stats_df = pd.DataFrame(all_stats)
    tag = f"adaptive_lb{args.lookback}_{args.method}_reg{int(not args.no_regime)}"
    stats_path = os.path.join(OUT_DIR, f"adaptive_stats_{tag}.csv")
    stats_df.to_csv(stats_path, index=False)

    adap_path = os.path.join(OUT_DIR, f"adaptive_returns_{tag}.csv")
    adap.to_csv(adap_path, index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(13, 14))
    fig.suptitle(f"Adaptive Factor Strategy  (lookback={args.lookback}M, regime={'ON' if not args.no_regime else 'OFF'})",
                 fontsize=13, fontweight="bold")

    # Panel 1: Equity curves
    ax1 = axes[0]
    colors = {"Adaptive": "#e63946",
              "Quality+Invest (current best)": "#457b9d",
              "Quality+Invest+Mom (historical)": "#2a9d8f",
              "All factors equal": "#f4a261",
              "VNINDEX": "#999999"}
    lws   = {"Adaptive": 2.5, "VNINDEX": 1.5}

    for name, eq in cum.items():
        ax1.plot(eq.index, eq.values,
                 label=name,
                 color=colors.get(name, "#333333"),
                 linewidth=lws.get(name, 1.8),
                 linestyle="--" if name == "VNINDEX" else "-")

    # Shade regime-off periods (VNINDEX below MA)
    if regime_df is not None:
        regime_plot = regime_df[regime_df["date"] >= pd.Timestamp(args.start)]
        for _, row in regime_plot.iterrows():
            if row["risk_on"] == 0:
                ax1.axvspan(row["date"] - pd.offsets.MonthBegin(1),
                            row["date"],
                            alpha=0.12, color="red", linewidth=0)

    ax1.set_ylabel("Portfolio value (start=1)")
    ax1.set_title("Cumulative equity curves  (red shading = regime OFF / cash)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Panel 2: Adaptive factor weights over time
    ax2 = axes[1]
    if not adap.empty and "w_quality" in adap.columns:
        w_cols = [c for c in adap.columns if c.startswith("w_")]
        adap_plot = adap.set_index("date")[w_cols].copy()
        adap_plot.columns = [c.replace("w_", "") for c in w_cols]
        adap_plot = adap_plot[adap_plot.sum(axis=1) > 0]  # drop cash months
        fcolors   = {"quality": "#2a9d8f", "investment": "#457b9d",
                     "momentum": "#e63946", "value": "#f4a261"}
        ax2.stackplot(adap_plot.index,
                      [adap_plot[c] for c in adap_plot.columns],
                      labels=list(adap_plot.columns),
                      colors=[fcolors.get(c, "#aaa") for c in adap_plot.columns],
                      alpha=0.8)
        ax2.set_ylabel("Factor weight")
        ax2.set_title("Adaptive factor weights over time")
        ax2.legend(fontsize=8, loc="upper left")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

    # Panel 3: Rolling 12M Sharpe
    ax3 = axes[2]
    for name, r in [("Adaptive", adap)] + list(static_results.items()):
        ret_s = r.set_index("date")["ret"]
        roll  = ret_s.rolling(12).apply(
            lambda x: (x.mean() * 12) / (x.std() * np.sqrt(12)) if x.std() > 0 else np.nan
        )
        ax3.plot(roll.index, roll.values,
                 label=name,
                 color=colors.get(name, "#333"),
                 linewidth=lws.get(name, 1.5),
                 alpha=0.85)
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.axhline(1, color="gray",  linewidth=0.6, linestyle=":")
    ax3.set_ylabel("Rolling 12M Sharpe")
    ax3.set_title("Rolling Sharpe ratio (12-month window)")
    ax3.legend(fontsize=8, loc="upper left")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, f"adaptive_chart_{tag}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved:")
    print(f"  Chart:   {fig_path}")
    print(f"  Stats:   {stats_path}")
    print(f"  Returns: {adap_path}")


if __name__ == "__main__":
    main()
