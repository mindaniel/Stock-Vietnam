"""
foreign_flow_long_panel_regression.py — Same panel regression as
foreign_flow_panel_regression.py, but on data/foreign_flow_long (FireAnt,
TOTAL foreign flow — institutional + retail combined, no split) instead of
data/investor_flow (nguoiquansat, has the split but only Sep 2024+).

Trade-off: lose the retail/institutional split (which had opposite signs —
this test can't distinguish them), gain ~15-19 years of history per ticker
instead of ~22 months, covering multiple market regimes (2018 correction,
2020 COVID crash, 2022 bear market, several bull runs) instead of one mostly-
recovering window. This is the robustness check flagged back when the long
data was first fetched: does "foreign flow predicts returns" hold up outside
the single regime everything else in this session was tested on?

Model:
  fwd_ret_{i,t+h} = b1*accum_foreign_{i,t}
                  + b2*np_yoy_{i,t} + b3*roe_{i,t} + b4*debt_equity_{i,t}
                  + sector_FE + time_FE + e_{i,t}

Standard errors clustered by ticker. Sample: ~monthly cross-sections across
full available history per ticker.

Usage:  python archive/foreign_flow_long_panel_regression.py
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

FFL_DIR   = os.path.join(BASE, "data", "foreign_flow_long")
PRICE_DIR = os.path.join(BASE, "data", "price")

MIN_LIQUIDITY_VND = 1_000_000_000
ACCUM_WINDOW  = 60
FORWARD_HORIZONS = [60, 120]
SAMPLE_STRIDE = 20   # ~monthly cross-sections
MIN_ROWS_PER_TICKER = ACCUM_WINDOW + max(FORWARD_HORIZONS) + 250   # want several years of usable history


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
    fpath = os.path.join(FFL_DIR, f"{ticker}.parquet")
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_parquet(fpath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[df["close"] > 0]
    if len(df) < MIN_ROWS_PER_TICKER:
        return pd.DataFrame()

    net = df["foreign_net_value"].fillna(0)
    out = pd.DataFrame({"date": df["date"], "ticker": ticker, "sector": sector, "close": df["close"]})
    cum   = net.rolling(ACCUM_WINDOW).sum()
    scale = net.abs().rolling(ACCUM_WINDOW).mean() * ACCUM_WINDOW
    out["accum_foreign"] = cum / scale.replace(0, np.nan)
    for h in FORWARD_HORIZONS:
        out[f"fwd_{h}"] = df["close"].shift(-h) / df["close"] - 1
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


def main():
    print("Building liquid universe + sector map...")
    liquid = liquid_universe()
    ffl_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                   for f in glob.glob(os.path.join(FFL_DIR, "*.parquet"))}
    mapping = pd.read_csv(os.path.join(BASE, "ticker_sectors.csv"))
    mapping.columns = [c.strip().lower() for c in mapping.columns]
    mapping = mapping[mapping["exchange"].isin(["HOSE", "HNX"])]
    ticker_to_sector = dict(zip(mapping["ticker"].str.upper(), mapping["industry"]))

    universe = sorted(liquid & ffl_tickers)
    frames = []
    for t in universe:
        sec = ticker_to_sector.get(t)
        if sec is None or sec == "Unknown":
            continue
        f = build_ticker_frame(t, sec)
        if not f.empty:
            frames.append(f)
    all_df = pd.concat(frames, ignore_index=True)
    print(f"  {all_df['ticker'].nunique()} tickers, {all_df['sector'].nunique()} sectors, "
          f"date range {all_df['date'].min().date()} to {all_df['date'].max().date()}")

    sample_dates = sorted(all_df["date"].unique())[::SAMPLE_STRIDE]
    panel = all_df[all_df["date"].isin(sample_dates)].copy()
    print(f"  {len(sample_dates)} periods, {len(panel):,} panel rows (pre-fundamentals)")

    print("Loading fundamentals (this may take a moment for the full universe)...")
    qfeat = build_factor_features(symbols=universe)
    panel = attach_fundamentals(panel, qfeat)

    sec_counts = panel["sector"].value_counts()
    ok_sectors = sec_counts[sec_counts >= 15].index
    panel = panel[panel["sector"].isin(ok_sectors)]
    panel["period"] = panel["date"].dt.strftime("%Y-%m")

    for h in FORWARD_HORIZONS:
        print(f"\n{'='*100}")
        print(f"  PANEL REGRESSION (LONG HISTORY, TOTAL foreign flow) — forward {h}-day return")
        print(f"  fwd_ret ~ accum_foreign + np_yoy + roe + debt_equity + sector FE + time FE, "
              f"clustered SE by ticker")
        print(f"{'-'*100}")

        cols = ["ticker", "sector", "period", f"fwd_{h}", "accum_foreign", "np_yoy", "roe", "debt_equity"]
        sub = panel[cols].rename(columns={f"fwd_{h}": "fwd_ret"}).dropna()
        print(f"  N = {len(sub):,}  (after dropping missing fundamentals)")
        if len(sub) < 50:
            print("  Not enough observations, skipping.")
            continue

        formula = "fwd_ret ~ accum_foreign + np_yoy + roe + debt_equity + C(sector) + C(period)"
        model = smf.ols(formula, data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["ticker"]}
        )

        keep = ["Intercept", "accum_foreign", "np_yoy", "roe", "debt_equity"]
        print(f"\n  {'Variable':<16} {'coef':>10} {'std err':>10} {'t':>8} {'P>|t|':>8}")
        print(f"  {'-'*60}")
        for name in keep:
            if name not in model.params.index:
                continue
            print(f"  {name:<16} {model.params[name]:>+10.5f} {model.bse[name]:>10.5f} "
                  f"{model.tvalues[name]:>8.2f} {model.pvalues[name]:>8.3f}")
        print(f"\n  R-squared (incl. FE): {model.rsquared:.4f}   N={int(model.nobs)}   "
              f"periods spanned: {sub['period'].nunique()}")

    print(f"\n{'='*100}")
    print("  INTERPRETATION")
    print(f"{'-'*100}")
    print("  This is TOTAL foreign flow (inst+retail mixed) across ~15-19 years, multiple")
    print("  market regimes. The short-sample panel regression (Sep 2024+, split by type)")
    print("  found retail positive/significant, institutional negative/weaker. If")
    print("  accum_foreign here comes out negative/significant, that's consistent with")
    print("  institutional dominating the aggregate (matches the earlier tick-tape and")
    print("  lead/lag findings that 'foreign' in aggregate behaves like momentum-chasing).")
    print("  If it's flat/insignificant, the aggregate cancels out — expected if retail and")
    print("  institutional effects are roughly offsetting in size across most periods.")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
