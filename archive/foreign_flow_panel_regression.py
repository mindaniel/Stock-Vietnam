"""
foreign_flow_panel_regression.py — Proper panel regression: forward returns
on foreign flow, WITH fundamental controls AND sector + time fixed effects,
clustered standard errors. This supersedes the earlier Fama-MacBeth test
(which only handled time effects implicitly, one cross-section at a time,
and couldn't control for fundamentals and flow simultaneously in one model).

Model:
  fwd_ret_{i,t+h} = b1*retail_accum_{i,t} + b2*inst_accum_{i,t}
                  + b3*np_yoy_{i,t} + b4*roe_{i,t} + b5*debt_equity_{i,t}
                  + sector_FE + time_FE + e_{i,t}

- retail_accum / inst_accum entered TOGETHER so each is a control for the
  other (earlier work showed they have opposite signs — a regression that
  omits one would bias the other's coefficient).
- np_yoy/roe/debt_equity as controls — isolates the flow signal's effect
  net of the "they just pick high-growth stocks" story from
  reverse_engineer_retail_picks.py, rather than confounding the two.
- Sector fixed effects absorb each sector's average return that period
  (same idea as the Fama-MacBeth sector-neutralization, but now inside a
  single joint regression with the other controls instead of a two-step
  procedure).
- Time (period) fixed effects absorb the whole market's move that period.
- Standard errors clustered by ticker — flow/return observations for the
  same stock across overlapping monthly samples are not independent;
  naive OLS SEs would overstate significance.

Sample: liquid tickers with investor_flow coverage (Sep 2024-present),
~monthly cross-sections (same construction as fama_macbeth_foreign.py).

Usage:  python archive/foreign_flow_panel_regression.py
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
FORWARD_HORIZONS = [60, 120]
SAMPLE_STRIDE = 20   # ~monthly cross-sections


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
    df = pd.read_parquet(fpath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "close" not in df.columns or len(df) < ACCUM_WINDOW + max(FORWARD_HORIZONS) + 5:
        return pd.DataFrame()

    retail = df.get("ca_nhan_nuocngoai_net", pd.Series(0, index=df.index)).fillna(0)
    inst   = df.get("to_chuc_nuocngoai_net", pd.Series(0, index=df.index)).fillna(0)

    out = pd.DataFrame({"date": df["date"], "ticker": ticker, "sector": sector})
    for label, series in [("retail", retail), ("inst", inst)]:
        cum   = series.rolling(ACCUM_WINDOW).sum()
        scale = series.abs().rolling(ACCUM_WINDOW).mean() * ACCUM_WINDOW
        out[f"accum_{label}"] = cum / scale.replace(0, np.nan)
    for h in FORWARD_HORIZONS:
        out[f"fwd_{h}"] = df["close"].shift(-h) / df["close"] - 1
    return out


def attach_fundamentals(panel: pd.DataFrame, qfeat: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time np_yoy/roe/debt_equity as of each row's own date."""
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
    flow_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet"))}
    mapping = pd.read_csv(os.path.join(BASE, "ticker_sectors.csv"))
    mapping.columns = [c.strip().lower() for c in mapping.columns]
    mapping = mapping[mapping["exchange"].isin(["HOSE", "HNX"])]
    ticker_to_sector = dict(zip(mapping["ticker"].str.upper(), mapping["industry"]))

    universe = sorted(liquid & flow_tickers)
    frames = []
    for t in universe:
        sec = ticker_to_sector.get(t)
        if sec is None or sec == "Unknown":
            continue
        f = build_ticker_frame(t, sec)
        if not f.empty:
            frames.append(f)
    all_df = pd.concat(frames, ignore_index=True)

    # sample ~monthly to keep observations closer to independent
    sample_dates = sorted(all_df["date"].unique())[::SAMPLE_STRIDE]
    panel = all_df[all_df["date"].isin(sample_dates)].copy()
    print(f"  {panel['ticker'].nunique()} tickers, {panel['sector'].nunique()} sectors, "
          f"{len(sample_dates)} periods, {len(panel):,} panel rows (pre-fundamentals)")

    print("Loading fundamentals...")
    qfeat = build_factor_features(symbols=universe)
    panel = attach_fundamentals(panel, qfeat)

    # keep only sectors with enough observations for a stable FE dummy
    sec_counts = panel["sector"].value_counts()
    ok_sectors = sec_counts[sec_counts >= 15].index
    panel = panel[panel["sector"].isin(ok_sectors)]

    panel["period"] = panel["date"].dt.strftime("%Y-%m")

    for h in FORWARD_HORIZONS:
        print(f"\n{'='*100}")
        print(f"  PANEL REGRESSION — forward {h}-day return")
        print(f"  fwd_ret ~ accum_retail + accum_inst + np_yoy + roe + debt_equity "
              f"+ sector FE + time FE, clustered SE by ticker")
        print(f"{'-'*100}")

        cols = ["ticker", "sector", "period", f"fwd_{h}", "accum_retail", "accum_inst",
                "np_yoy", "roe", "debt_equity"]
        sub = panel[cols].rename(columns={f"fwd_{h}": "fwd_ret"}).dropna()
        print(f"  N = {len(sub):,}  (after dropping missing fundamentals)")
        if len(sub) < 50:
            print("  Not enough observations, skipping.")
            continue

        formula = ("fwd_ret ~ accum_retail + accum_inst + np_yoy + roe + debt_equity "
                   "+ C(sector) + C(period)")
        model = smf.ols(formula, data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["ticker"]}
        )

        # only print the substantive (non-FE-dummy) coefficients
        keep = ["Intercept", "accum_retail", "accum_inst", "np_yoy", "roe", "debt_equity"]
        print(f"\n  {'Variable':<16} {'coef':>10} {'std err':>10} {'t':>8} {'P>|t|':>8}")
        print(f"  {'-'*60}")
        for name in keep:
            if name not in model.params.index:
                continue
            print(f"  {name:<16} {model.params[name]:>+10.5f} {model.bse[name]:>10.5f} "
                  f"{model.tvalues[name]:>8.2f} {model.pvalues[name]:>8.3f}")
        print(f"\n  R-squared (incl. FE): {model.rsquared:.4f}   N={int(model.nobs)}")
        print(f"  (sector and time FE dummy coefficients omitted from display, still in model)")

    print(f"\n{'='*100}")
    print("  INTERPRETATION")
    print(f"{'-'*100}")
    print("  accum_retail / accum_inst coefficients are now net of: the fundamental growth")
    print("  story (np_yoy/roe/debt_equity), each sector's average return that period, and")
    print("  each month's overall market move. If accum_retail stays positive/significant and")
    print("  accum_inst stays negative/significant here too, the earlier findings are not")
    print("  explained away by growth characteristics or sector-timing — they're independent")
    print("  effects. Compare np_yoy's own coefficient: if IT is also significant here, growth")
    print("  has its own separate predictive power alongside (not instead of) the flow signal.")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
