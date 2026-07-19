"""
all_players_panel_regression.py — Which of the 5 investor types (foreign
retail, foreign institutional, domestic retail, domestic institutional,
proprietary/tự doanh) actually predicts forward returns, and how has each
one's effect changed across market regimes?

Same core method validated earlier this session (foreign_flow_panel_regression.py,
which found foreign retail positive/significant, foreign institutional
negative — survived fundamentals + sector FE + time FE + clustered SEs), now:
  - all 5 players entered TOGETHER (each is a control for the others)
  - full ~1,500-ticker, up to ~10-year universe (data/investor_flow after the
    FiinTrade merge), not just the ~400-ticker Sep-2024+ nguoiquansat-only set
  - split into market-regime sub-periods to see whether each player's sign/
    significance is stable over time or regime-dependent (this is the part
    the single 22-month sample earlier in this session couldn't test)

Model (per period, and once for the full sample):
  fwd_ret_{i,t+60d} = b1*accum_fg_retail + b2*accum_fg_inst
                    + b3*accum_dom_retail + b4*accum_dom_inst + b5*accum_prop
                    + b6*np_yoy + b7*roe + b8*debt_equity
                    + sector_FE + time_FE + e_{i,t}
  Clustered SEs by ticker.

Usage:  python archive/all_players_panel_regression.py
"""

import glob, os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "lib"))
from factor_stock_ranker import build_factor_features

FLOW_DIR  = os.path.join(BASE, "data", "investor_flow")
PRICE_DIR = os.path.join(BASE, "data", "price")

MIN_LIQUIDITY_VND = 1_000_000_000
ACCUM_WINDOW  = 60
FORWARD_DAYS  = 60
SAMPLE_STRIDE = 20   # ~monthly cross-sections

PLAYER_COLS = {
    "fg_retail":   "ca_nhan_nuocngoai_net",
    "fg_inst":     "to_chuc_nuocngoai_net",
    "dom_retail":  "ca_nhan_trongnuoc_net",
    "dom_inst":    "to_chuc_trongnuoc_net",
    "prop":        "tu_doanh_net",
}

# Market-regime sub-periods (VN market history). Buckets need enough width
# for ACCUM_WINDOW (60d) + FORWARD_DAYS (60d) lookback/lookahead to fit
# comfortably, and enough cross-sectional tickers with coverage that period.
PERIODS = [
    ("2014-2017 (growth)",      "2014-01-01", "2017-12-31"),
    ("2018-2019 (correction)",  "2018-01-01", "2019-12-31"),
    ("2020-2021 (COVID boom)",  "2020-01-01", "2021-12-31"),
    ("2022 (bear market)",      "2022-01-01", "2022-12-31"),
    ("2023-2024 (recovery)",    "2023-01-01", "2024-12-31"),
    ("2025-2026 (recent)",      "2025-01-01", "2026-12-31"),
]


def liquid_universe():
    liquid = set()
    for fpath in glob.glob(os.path.join(PRICE_DIR, "*.parquet")):
        ticker = os.path.splitext(os.path.basename(fpath))[0].upper()
        try:
            df = pd.read_parquet(fpath)
            df.columns = [c.strip().lower() for c in df.columns]
            if "close" not in df.columns or "volume" not in df.columns:
                continue
            med_to = (df["close"] * df["volume"] * 1000).tail(60).median()
            if med_to >= MIN_LIQUIDITY_VND:
                liquid.add(ticker)
        except Exception:
            pass
    return liquid


def build_ticker_frame(ticker: str, sector: str) -> pd.DataFrame:
    fpath = os.path.join(FLOW_DIR, f"{ticker}.parquet")
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_parquet(fpath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "close" not in df.columns or len(df) < ACCUM_WINDOW + FORWARD_DAYS + 5:
        return pd.DataFrame()

    out = pd.DataFrame({"date": df["date"], "ticker": ticker, "sector": sector})
    for label, col in PLAYER_COLS.items():
        series = df.get(col, pd.Series(0, index=df.index)).fillna(0)
        cum   = series.rolling(ACCUM_WINDOW).sum()
        scale = series.abs().rolling(ACCUM_WINDOW).mean() * ACCUM_WINDOW
        out[f"accum_{label}"] = cum / scale.replace(0, np.nan)
    out[f"fwd_{FORWARD_DAYS}"] = df["close"].shift(-FORWARD_DAYS) / df["close"] - 1
    return out


def attach_fundamentals(panel: pd.DataFrame, qfeat: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    qfeat_by_sym = {sym: g.sort_values("avail_date") for sym, g in qfeat.groupby("symbol")}
    for _, row in panel.iterrows():
        sub = qfeat_by_sym.get(row["ticker"].upper())
        rec = row.to_dict()
        if sub is not None:
            q0 = sub[sub["avail_date"] <= row["date"]].tail(1)
            if not q0.empty:
                rec["np_yoy"] = q0["np_yoy"].iloc[0]
                rec["roe"] = q0["roe"].iloc[0]
                rec["debt_equity"] = q0["debt_equity"].iloc[0]
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def run_regression(sub: pd.DataFrame, label: str):
    accum_cols = [f"accum_{k}" for k in PLAYER_COLS]
    needed = ["fwd_ret"] + accum_cols + ["np_yoy", "roe", "debt_equity", "sector", "period"]
    sub = sub[["ticker"] + needed].dropna()
    print(f"\n{'='*100}")
    print(f"  {label}")
    print(f"{'-'*100}")
    print(f"  N = {len(sub):,}  tickers = {sub['ticker'].nunique()}  periods = {sub['period'].nunique()}")
    if len(sub) < 80 or sub["ticker"].nunique() < 15:
        print("  Not enough observations/cross-sectional breadth, skipping.")
        return

    sec_counts = sub["sector"].value_counts()
    ok_sectors = sec_counts[sec_counts >= 8].index
    sub = sub[sub["sector"].isin(ok_sectors)]
    if sub.empty or sub["ticker"].nunique() < 15:
        print("  Not enough after sector-count filter, skipping.")
        return

    formula = ("fwd_ret ~ " + " + ".join(accum_cols) +
               " + np_yoy + roe + debt_equity + C(sector) + C(period)")
    try:
        model = smf.ols(formula, data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["ticker"]}
        )
    except Exception as e:
        print(f"  Regression failed: {e}")
        return

    keep = ["Intercept"] + accum_cols + ["np_yoy", "roe", "debt_equity"]
    print(f"\n  {'Variable':<16} {'coef':>10} {'std err':>10} {'t':>8} {'P>|t|':>8}")
    print(f"  {'-'*60}")
    for name in keep:
        if name not in model.params.index:
            continue
        stars = ""
        p = model.pvalues[name]
        if p < 0.01: stars = "***"
        elif p < 0.05: stars = "**"
        elif p < 0.10: stars = "*"
        print(f"  {name:<16} {model.params[name]:>+10.5f} {model.bse[name]:>10.5f} "
              f"{model.tvalues[name]:>8.2f} {model.pvalues[name]:>8.3f} {stars}")
    print(f"\n  R-squared (incl. FE): {model.rsquared:.4f}   N={int(model.nobs)}")


def main():
    print("Building liquid universe + sector map...")
    liquid = liquid_universe()
    flow_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet"))}
    mapping = pd.read_csv(os.path.join(BASE, "ticker_sectors.csv"))
    mapping.columns = [c.strip().lower() for c in mapping.columns]
    mapping = mapping[mapping["exchange"].isin(["HOSE", "HNX"])]
    ticker_to_sector = dict(zip(mapping["ticker"].str.upper(), mapping["industry"]))

    universe = sorted(liquid & flow_tickers)
    print(f"  {len(universe)} liquid tickers with flow data")

    print("Building per-ticker panels (this may take a few minutes for ~1000+ tickers)...")
    frames = []
    for i, t in enumerate(universe):
        sec = ticker_to_sector.get(t)
        if sec is None or sec == "Unknown":
            continue
        f = build_ticker_frame(t, sec)
        if not f.empty:
            frames.append(f)
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(universe)} tickers processed, {len(frames)} usable so far")

    all_df = pd.concat(frames, ignore_index=True)
    print(f"  {all_df['ticker'].nunique()} tickers, {all_df['sector'].nunique()} sectors, "
          f"date range {all_df['date'].min().date()} to {all_df['date'].max().date()}")

    sample_dates = sorted(all_df["date"].unique())[::SAMPLE_STRIDE]
    panel = all_df[all_df["date"].isin(sample_dates)].copy()
    print(f"  {len(sample_dates)} periods, {len(panel):,} panel rows (pre-fundamentals)")

    print("Loading fundamentals...")
    qfeat = build_factor_features(symbols=universe)
    panel = attach_fundamentals(panel, qfeat)
    panel = panel.rename(columns={f"fwd_{FORWARD_DAYS}": "fwd_ret"})
    panel["period"] = panel["date"].dt.strftime("%Y-%m")

    run_regression(panel, f"FULL SAMPLE — all periods pooled ({panel['date'].min().date()} to {panel['date'].max().date()})")

    for label, start, end in PERIODS:
        sub = panel[(panel["date"] >= start) & (panel["date"] <= end)].copy()
        run_regression(sub, label)

    print(f"\n{'='*100}")
    print("  INTERPRETATION")
    print(f"{'-'*100}")
    print("  accum_fg_retail / accum_fg_inst / accum_dom_retail / accum_dom_inst / accum_prop")
    print("  are all entered TOGETHER, so each coefficient is net of the other four players,")
    print("  sector-period average return, and the growth/quality/leverage fundamentals.")
    print("  Compare sign/significance across periods: a player whose coefficient flips sign")
    print("  or loses significance across regimes is NOT a stable, regime-independent signal —")
    print("  that's the key test the single-regime (Sep2024+) sample earlier could never run.")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
