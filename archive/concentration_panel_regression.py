"""
concentration_panel_regression.py — Forward-looking test of the one finding
from today's session that showed real content: does CONCENTRATED (bursty)
buying predict better forward returns than the same amount of buying spread
out evenly, once accumulation size, fundamentals, sector, and time are all
controlled for?

This reformulates player_clustering_analysis.py's descriptive, ex-post
"ride" finding (concentrated buying preceded bigger gains, +2 to +6pp,
across all 5 players) as a genuinely POINT-IN-TIME predictive test:

  - accum_{player}: existing 60-day accumulation score (direction + size)
  - disp_{player}:  60-day trailing DISPERSION of that player's net-buying,
                     computed using ONLY the trailing window ending at the
                     sample date (no future price data, unlike the
                     ride-based version which used swing pivots confirmed
                     with hindsight). Low = concentrated bursts,
                     high = spread evenly. Normalized to [0, 1].

Model: fwd_ret_{i,t+60d} = sum_p (b1p*accum_p + b2p*disp_p)
                          + np_yoy + roe + debt_equity
                          + sector_FE + time_FE + e_{i,t}
       clustered SEs by ticker.

If disp_p is negative and significant (net of accum_p's own size effect),
concentration itself carries forward-looking information, not just "more
buying happened" — the actual test of whether the earlier descriptive
finding survives becoming a real predictive claim.

Usage:  python archive/concentration_panel_regression.py
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
SAMPLE_STRIDE = 20   # ~monthly cross-sections, same as the other panel scripts

PLAYER_COLS = {
    "fg_retail":   "ca_nhan_nuocngoai_net",
    "fg_inst":     "to_chuc_nuocngoai_net",
    "dom_retail":  "ca_nhan_trongnuoc_net",
    "dom_inst":    "to_chuc_trongnuoc_net",
    "prop":        "tu_doanh_net",
}


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


def dispersion_at(window: np.ndarray) -> float:
    """Flow-weighted dispersion (normalized std of day-index) among
    NET-BUYING days in a trailing window, using only that window's own
    data — point-in-time, no future information."""
    buys = np.clip(window, 0, None)
    total = buys.sum()
    if total <= 0:
        return np.nan
    n = len(window)
    idx = np.arange(n)
    com = (idx * buys).sum() / total
    var = ((idx - com) ** 2 * buys).sum() / total
    return np.sqrt(var) / max(n - 1, 1)


def compute_dispersion_at_samples(df: pd.DataFrame, sample_idx: np.ndarray) -> dict:
    """For each player, dispersion at each sampled row index using the
    trailing ACCUM_WINDOW ending at that row (point-in-time)."""
    result = {label: np.full(len(df), np.nan) for label in PLAYER_COLS}
    for label, col in PLAYER_COLS.items():
        if col not in df.columns:
            continue
        series = df[col].fillna(0).values
        for i in sample_idx:
            if i < ACCUM_WINDOW - 1:
                continue
            window = series[i - ACCUM_WINDOW + 1:i + 1]
            result[label][i] = dispersion_at(window)
    return result


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
    flow_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet"))}
    mapping = pd.read_csv(os.path.join(BASE, "ticker_sectors.csv"))
    mapping.columns = [c.strip().lower() for c in mapping.columns]
    mapping = mapping[mapping["exchange"].isin(["HOSE", "HNX"])]
    ticker_to_sector = dict(zip(mapping["ticker"].str.upper(), mapping["industry"]))
    universe = sorted(liquid & flow_tickers & set(ticker_to_sector))
    print(f"  {len(universe)} tickers")

    print("Building per-ticker panels (accum scores + point-in-time dispersion)...")
    frames = []
    for i, t in enumerate(universe):
        sec = ticker_to_sector.get(t)
        if sec is None or sec == "Unknown":
            continue
        fpath = os.path.join(FLOW_DIR, f"{t}.parquet")
        if not os.path.exists(fpath):
            continue
        df = pd.read_parquet(fpath)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        if "close" not in df.columns or len(df) < ACCUM_WINDOW + FORWARD_DAYS + 5:
            continue

        out = pd.DataFrame({"date": df["date"], "ticker": t, "sector": sec})
        for label, col in PLAYER_COLS.items():
            series = df.get(col, pd.Series(0, index=df.index)).fillna(0)
            cum   = series.rolling(ACCUM_WINDOW).sum()
            scale = series.abs().rolling(ACCUM_WINDOW).mean() * ACCUM_WINDOW
            out[f"accum_{label}"] = cum / scale.replace(0, np.nan)
        out[f"fwd_{FORWARD_DAYS}"] = df["close"].shift(-FORWARD_DAYS) / df["close"] - 1

        # Only compute the (more expensive) dispersion at the sampled rows
        n = len(df)
        sample_idx = np.arange(0, n, SAMPLE_STRIDE)
        disp = compute_dispersion_at_samples(df, sample_idx)
        for label in PLAYER_COLS:
            out[f"disp_{label}"] = disp[label]

        out = out.iloc[sample_idx]
        frames.append(out)

        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(universe)} tickers, {len(frames)} usable so far")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.rename(columns={f"fwd_{FORWARD_DAYS}": "fwd_ret"})
    panel["period"] = panel["date"].dt.strftime("%Y-%m")
    print(f"  {panel['ticker'].nunique()} tickers, {len(panel):,} panel rows (pre-fundamentals)")

    print("Loading fundamentals...")
    qfeat = build_factor_features(symbols=universe)
    panel = attach_fundamentals(panel, qfeat)

    accum_cols = [f"accum_{k}" for k in PLAYER_COLS]
    disp_cols = [f"disp_{k}" for k in PLAYER_COLS]
    needed = ["fwd_ret"] + accum_cols + disp_cols + ["np_yoy", "roe", "debt_equity", "sector", "period"]
    sub = panel[["ticker"] + needed].dropna()
    print(f"\nN = {len(sub):,}  tickers = {sub['ticker'].nunique()}  periods = {sub['period'].nunique()}")

    sec_counts = sub["sector"].value_counts()
    ok_sectors = sec_counts[sec_counts >= 15].index
    sub = sub[sub["sector"].isin(ok_sectors)]

    formula = ("fwd_ret ~ " + " + ".join(accum_cols) + " + " + " + ".join(disp_cols) +
               " + np_yoy + roe + debt_equity + C(sector) + C(period)")
    model = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["ticker"]})

    keep = ["Intercept"] + accum_cols + disp_cols + ["np_yoy", "roe", "debt_equity"]
    print(f"\n{'='*90}")
    print("  DOES CONCENTRATION (disp_*) PREDICT FORWARD RETURNS, NET OF ACCUMULATION SIZE?")
    print(f"{'-'*90}")
    print(f"  {'Variable':<16} {'coef':>10} {'std err':>10} {'t':>8} {'P>|t|':>8}")
    print(f"  {'-'*60}")
    for name in keep:
        if name not in model.params.index:
            continue
        p = model.pvalues[name]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        print(f"  {name:<16} {model.params[name]:>+10.5f} {model.bse[name]:>10.5f} "
              f"{model.tvalues[name]:>8.2f} {model.pvalues[name]:>8.3f} {stars}")
    print(f"\n  R-squared (incl. FE): {model.rsquared:.4f}   N={int(model.nobs):,}")

    print(f"\n{'='*90}")
    print("  INTERPRETATION")
    print(f"{'-'*90}")
    print("  disp_* coefficients being NEGATIVE and significant would mean: for a given")
    print("  amount of accumulation (accum_* held fixed), MORE spread-out buying (higher")
    print("  dispersion) predicts WORSE forward returns — i.e. concentration itself carries")
    print("  independent forward-looking information, confirming the ride-based descriptive")
    print("  finding as a genuine predictive signal, not just a restatement of accum size.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
