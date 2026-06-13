"""
ff_explain.py
Fama-French explainability scan for VN equity portfolios.

Runs FF3 / FF5 / FF6 regressions on:
  - every sector (equal-weight, sectors with >= MIN_STOCKS stocks)
  - the best saved factor-strategy backtest return series
  - any individual tickers passed via --ticker

Output: results/ff_explain_summary.csv  +  console table

Usage:
  python ff_explain.py --start 2018-01-01
  python ff_explain.py --start 2018-01-01 --ticker VNM FPT ACB
  python ff_explain.py --start 2016-01-01 --min-stocks 3
"""

import argparse
import os
import sys
import glob

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ff_vietnam import (
    load_prices, load_sector_map, load_fundamentals_latest_by_year,
    monthly_stock_panel, attach_accounting, build_factors,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "results")
RF_ANNUAL = 0.03
MIN_STOCKS = 5
MIN_MONTHS = 24

# Best saved backtest file (quality+momentum, regime on)
BACKTEST_FILE = os.path.join(
    OUT_DIR,
    "factor_long_only_returns_s2018-01-01_liq1000000_top0.25_cost15.0_reg1_v0q1i0m1.csv",
)


# ── OLS with standard errors and t-stats ─────────────────────────────────────

def ols_stats(y: pd.Series, X: pd.DataFrame) -> dict | None:
    """
    OLS regression of y on X (with intercept).
    Returns alpha, betas, R², t-stat and p-value on alpha (two-tailed).
    Returns None if too few observations.
    """
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    n  = len(df)
    if n < MIN_MONTHS:
        return None

    yv  = df["y"].values
    Xv  = np.column_stack([np.ones(n), df[X.columns].values])
    k   = Xv.shape[1]

    beta = np.linalg.lstsq(Xv, yv, rcond=None)[0]
    yhat  = Xv @ beta
    resid = yv - yhat

    ss_res = resid @ resid
    ss_tot = ((yv - yv.mean()) ** 2).sum()
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    sigma2 = ss_res / (n - k)
    cov    = sigma2 * np.linalg.pinv(Xv.T @ Xv)
    se     = np.sqrt(np.diag(cov))

    alpha   = float(beta[0])
    t_alpha = alpha / se[0] if se[0] > 0 else np.nan
    p_alpha = float(2 * sp_stats.t.sf(abs(t_alpha), df=n - k)) if not np.isnan(t_alpha) else np.nan

    return {
        "n":         n,
        "alpha_m":   alpha,
        "alpha_ann": (1 + alpha) ** 12 - 1,
        "t_alpha":   t_alpha,
        "p_alpha":   p_alpha,
        "r2":        r2,
        "betas":     {c: float(beta[i + 1]) for i, c in enumerate(X.columns)},
    }


# ── Portfolio return helpers ──────────────────────────────────────────────────

def sector_portfolio(panel: pd.DataFrame, sector: str) -> pd.Series:
    g = panel[panel["sector"] == sector]
    return g.groupby("date")["ret"].mean().rename("port_ret").dropna()


def ticker_portfolio(panel: pd.DataFrame, ticker: str) -> pd.Series:
    g = panel[panel["symbol"] == ticker.upper()]
    return g.groupby("date")["ret"].mean().rename("port_ret").dropna()


def backtest_portfolio(path: str) -> pd.Series | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "net_ret"]).sort_values("date")
    df = df[df["net_ret"] != 0.0]        # drop cash-parked months
    return df.set_index("date")["net_ret"].rename("port_ret")


# ── Regression runner ─────────────────────────────────────────────────────────

def run_regressions(name: str, port: pd.Series, ff: pd.DataFrame) -> list[dict]:
    ff_idx = ff.set_index("date")
    rf     = ff_idx["RF"]
    excess = (port - rf).dropna()
    if len(excess) < MIN_MONTHS:
        return []

    models = {
        "FF3": ["MKT_RF", "SMB", "HML"],
        "FF5": ["MKT_RF", "SMB", "HML", "RMW", "CMA"],
        "FF6": ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"],
    }
    rows = []
    for mname, factors in models.items():
        X = ff_idx[factors].dropna()
        res = ols_stats(excess, X)
        if res is None:
            continue
        sig = "***" if res["p_alpha"] < 0.01 else "**" if res["p_alpha"] < 0.05 else "*" if res["p_alpha"] < 0.10 else ""
        rows.append({
            "portfolio":  name,
            "model":      mname,
            "n_months":   res["n"],
            "alpha_m":    round(res["alpha_m"],   6),
            "alpha_ann":  round(res["alpha_ann"],  4),
            "t_alpha":    round(res["t_alpha"],    2),
            "sig":        sig,
            "p_alpha":    round(res["p_alpha"],    4),
            "r2":         round(res["r2"],         4),
            **{f"b_{k}": round(v, 4) for k, v in res["betas"].items()},
        })
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",      default="2018-01-01")
    ap.add_argument("--min-stocks", type=int, default=MIN_STOCKS)
    ap.add_argument("--ticker",     nargs="*", default=[])
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading prices...")
    prices = load_prices()
    if prices.empty:
        sys.exit("No price data found in data/*.parquet")

    sec_map = load_sector_map()
    print("Building monthly panel...")
    monthly = monthly_stock_panel(prices).merge(sec_map, on="symbol", how="left")
    fin     = load_fundamentals_latest_by_year()
    panel   = attach_accounting(monthly, fin)
    panel   = panel[panel["date"] >= pd.Timestamp(args.start)].copy()

    print("Building FF factors...")
    ff = build_factors(panel, rf_annual=RF_ANNUAL)
    if ff.empty:
        sys.exit("Could not build factors. Check data coverage.")

    coverage = {col: ff[col].notna().sum() for col in ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]}
    print(f"Factor coverage ({len(ff)} months): " +
          "  ".join(f"{k}={v}" for k, v in coverage.items()))
    print(f"  Note: HML requires PB data ({coverage['HML']} months). "
          f"SMB uses {'mkt_cap_proxy' if coverage['HML'] > 12 else 'size_proxy (traded value)'} as size proxy.")

    all_rows = []

    # ── Sectors ──────────────────────────────────────────────────────────────
    for sector, grp in panel.groupby("sector"):
        if pd.isna(sector) or sector in ("", "Unknown"):
            continue
        n_syms = grp["symbol"].nunique()
        if n_syms < args.min_stocks:
            continue
        port = sector_portfolio(panel, sector)
        rows = run_regressions(f"SECTOR:{sector}", port, ff)
        all_rows.extend(rows)
        print(f"  {sector:<35} stocks={n_syms:>4}  FF6 R²={rows[-1]['r2']:.3f}  α_ann={rows[-1]['alpha_ann']:.2%}{rows[-1]['sig']}"
              if rows else f"  {sector}: insufficient data")

    # ── Factor strategy portfolio ─────────────────────────────────────────────
    bt_port = backtest_portfolio(BACKTEST_FILE)
    if bt_port is not None:
        rows = run_regressions("FACTOR_STRATEGY(Q+MOM+Regime)", bt_port, ff)
        all_rows.extend(rows)
        if rows:
            r = rows[-1]
            print(f"\n  Factor strategy (FF6)  months={r['n_months']}  R²={r['r2']:.3f}  α_ann={r['alpha_ann']:.2%}{r['sig']}")
    else:
        print(f"\n  [skip] Backtest file not found: {BACKTEST_FILE}")

    # ── Individual tickers ────────────────────────────────────────────────────
    for tk in (args.ticker or []):
        port = ticker_portfolio(panel, tk)
        rows = run_regressions(f"TICKER:{tk.upper()}", port, ff)
        all_rows.extend(rows)
        if rows:
            r = rows[-1]
            print(f"  {tk.upper():<10} FF6  months={r['n_months']}  R²={r['r2']:.3f}  α_ann={r['alpha_ann']:.2%}{r['sig']}")

    if not all_rows:
        print("No results generated.")
        return

    summary = pd.DataFrame(all_rows)
    out_path = os.path.join(OUT_DIR, "ff_explain_summary.csv")
    summary.to_csv(out_path, index=False)

    # ── Print clean FF6 table sorted by annualised alpha ─────────────────────
    ff6 = summary[summary["model"] == "FF6"].copy()
    ff6 = ff6.sort_values("alpha_ann", ascending=False)

    print("\n" + "=" * 90)
    print("FAMA-FRENCH 6-FACTOR EXPLAINABILITY  (sorted by annualised alpha)")
    print("=" * 90)
    print(f"{'Portfolio':<42} {'N':>5} {'α_ann':>7} {'t(α)':>6} {'sig':>3} {'R²':>6}  Top betas")
    print("-" * 90)
    for _, r in ff6.iterrows():
        betas = {k.replace("b_", ""): v for k, v in r.items() if str(k).startswith("b_")}
        top   = sorted(betas.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        beta_str = "  ".join(f"{k}={v:+.2f}" for k, v in top)
        print(f"{r['portfolio']:<42} {r['n_months']:>5} {r['alpha_ann']:>+7.2%} {r['t_alpha']:>6.2f} {r['sig']:>3} {r['r2']:>6.3f}  {beta_str}")

    print("=" * 90)
    print("Significance: *** p<0.01  ** p<0.05  * p<0.10")
    print(f"\nFull results saved to: {out_path}")

    # ── FF3 vs FF5 vs FF6: incremental R² for factor strategy ────────────────
    strat_rows = summary[summary["portfolio"].str.startswith("FACTOR_STRATEGY")]
    if not strat_rows.empty:
        print("\nFactor strategy — incremental R² across models:")
        for _, r in strat_rows.sort_values("model").iterrows():
            print(f"  {r['model']}  R²={r['r2']:.4f}  α_ann={r['alpha_ann']:+.2%}{r['sig']}")


if __name__ == "__main__":
    main()
