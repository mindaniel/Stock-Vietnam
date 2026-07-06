"""
market_control_test.py — Who actually moves price? A price-impact regression:
same-day return regressed on each investor type's net flow, normalized by
that day's total traded value. The type with the largest, most significant
coefficient is the one whose buying/selling has the biggest measurable
effect on price — i.e. who "controls" price-setting, as distinct from who
merely correlates with price direction (which the earlier lead/lag tests
already covered).

Also computes a same-day VWAP-based check using tick_data (the ~2-month
window where we have intraday prints): on days a given investor type is a
heavy net buyer, does the stock close ABOVE or BELOW its own volume-
weighted average price for that session? Closing above VWAP after buying
means that flow pushed price up through the session (aggressive, price-
setting). Closing at/below VWAP despite net buying means it was absorbed
passively without moving price (liquidity-providing, not price-setting).

Usage:  python archive/market_control_test.py
"""

import glob, os, sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOW_DIR  = os.path.join(BASE, "data", "investor_flow")
PRICE_DIR = os.path.join(BASE, "data", "price")
TICK_DIR  = os.path.join(BASE, "data", "tick_data")

MIN_LIQUIDITY_VND = 1_000_000_000

FLOW_TYPES = {
    "tu_doanh_net":          "Tu doanh (prop desk)",
    "to_chuc_nuocngoai_net": "Foreign institutional",
    "to_chuc_trongnuoc_net": "Domestic institutional",
    "ca_nhan_trongnuoc_net": "Domestic retail",
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


def load_price_with_turnover(ticker: str) -> pd.DataFrame:
    fpath = os.path.join(PRICE_DIR, f"{ticker}.parquet")
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_parquet(fpath)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "time" if "time" in df.columns else "date"
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if "close" not in df.columns or "volume" not in df.columns:
        return pd.DataFrame()
    df = df[df["close"] > 0]
    df["turnover_bn"] = df["close"] * df["volume"] * 1000 / 1e9
    df["ret"] = df["close"].pct_change()
    return df[["date", "ret", "turnover_bn"]]


def build_regression_frame(ticker: str) -> pd.DataFrame:
    fpath = os.path.join(FLOW_DIR, f"{ticker}.parquet")
    if not os.path.exists(fpath):
        return pd.DataFrame()
    fdf = pd.read_parquet(fpath)
    fdf["date"] = pd.to_datetime(fdf["date"])
    fdf = fdf.sort_values("date").reset_index(drop=True)

    pdf = load_price_with_turnover(ticker)
    if pdf.empty:
        return pd.DataFrame()

    m = fdf.merge(pdf, on="date", how="inner")
    if m.empty:
        return pd.DataFrame()
    for col in FLOW_TYPES:
        if col in m.columns:
            m[f"norm_{col}"] = m[col] / m["turnover_bn"].replace(0, np.nan)
    m["ticker"] = ticker
    return m


def daily_vwap(ticker: str) -> pd.DataFrame:
    fpath = os.path.join(TICK_DIR, f"{ticker}.parquet")
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_parquet(fpath)
    df["date"] = pd.to_datetime(df["td"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date"])
    df["p"] = df["p"].astype(float)
    df["v"] = df["v"].astype(float)
    df["pv"] = df["p"] * df["v"]
    g = df.groupby("date").agg(sum_pv=("pv", "sum"), sum_v=("v", "sum"))
    g["vwap"] = g["sum_pv"] / g["sum_v"]
    # last print of the day (data sorted desc by time within date group upstream, so
    # take max index by time string safely via idxmax on time-like sort key 't')
    last_px = df.sort_values("t").groupby("date")["p"].last()
    g["close_px"] = last_px
    g["close_vs_vwap"] = (g["close_px"] - g["vwap"]) / g["vwap"]
    g = g.reset_index()
    g["ticker"] = ticker
    return g[["date", "ticker", "vwap", "close_px", "close_vs_vwap"]]


def main():
    print("Building liquid universe...")
    liquid = liquid_universe()
    flow_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet"))}
    universe = sorted(liquid & flow_tickers)
    print(f"  {len(universe)} liquid tickers with flow data")

    print(f"\n{'='*90}")
    print("  PART 1: Price-impact regression")
    print("  same-day return ~ normalized net flow (net / day's turnover), pooled OLS")
    print(f"{'-'*90}")

    frames = []
    for t in universe:
        f = build_regression_frame(t)
        if not f.empty:
            frames.append(f)
    all_df = pd.concat(frames, ignore_index=True)

    norm_cols = [f"norm_{c}" for c in FLOW_TYPES]
    reg_df = all_df.dropna(subset=norm_cols + ["ret"]).copy()
    # clip extreme outliers (thin-volume days can blow up the normalized ratio)
    for c in norm_cols:
        lo, hi = reg_df[c].quantile([0.005, 0.995])
        reg_df[c] = reg_df[c].clip(lo, hi)
    reg_df["ret"] = reg_df["ret"].clip(*reg_df["ret"].quantile([0.005, 0.995]))

    X = reg_df[norm_cols].values
    X = np.column_stack([np.ones(len(X)), X])
    y = reg_df["ret"].values
    beta, resid, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid_arr = y - yhat
    n, k = X.shape
    dof = n - k
    sigma2 = (resid_arr @ resid_arr) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    tstats = beta / se
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = (resid_arr ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    print(f"  N = {n:,}   R^2 = {r2:.4f}")
    print(f"  {'Investor type':<26} {'coef':>10} {'t-stat':>9}   interpretation")
    print(f"  {'-'*80}")
    for i, (col, label) in enumerate(FLOW_TYPES.items(), start=1):
        c, ts = beta[i], tstats[i]
        strength = "STRONGEST price-setter" if abs(ts) == max(abs(tstats[1:])) else ""
        print(f"  {label:<26} {c:>+10.5f} {ts:>9.2f}   {strength}")

    print(f"\n{'='*90}")
    print("  PART 2: Close-vs-VWAP check (tick_data window only, ~2 months)")
    print("  Does that type's net buying coincide with closing ABOVE session VWAP")
    print("  (pushing price up through the day) or below (absorbed passively)?")
    print(f"{'-'*90}")

    tick_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(TICK_DIR, "*.parquet"))}
    vwap_frames = []
    for t in sorted(tick_tickers & set(universe)):
        f = daily_vwap(t)
        if not f.empty:
            vwap_frames.append(f)
    vwap_df = pd.concat(vwap_frames, ignore_index=True)

    merged = vwap_df.merge(all_df[["date", "ticker"] + list(FLOW_TYPES)], on=["date", "ticker"], how="inner")
    print(f"  N = {len(merged):,} ticker-days")
    print(f"  {'Investor type':<26} {'corr w/ close-vs-vwap':>22}   interpretation")
    print(f"  {'-'*80}")
    for col, label in FLOW_TYPES.items():
        sub = merged.dropna(subset=[col, "close_vs_vwap"])
        corr = sub[col].corr(sub["close_vs_vwap"], method="spearman")
        interp = ("buying pushes close ABOVE vwap (price-setter)" if corr > 0.05 else
                  "buying doesn't push close above vwap (passive)" if corr < 0.02 else
                  "weak/unclear")
        print(f"  {label:<26} {corr:>+22.4f}   {interp}")

    print(f"\n{'='*90}")
    print("  INTERPRETATION")
    print(f"{'-'*90}")
    print("  Part 1: largest |t-stat| coefficient = whose net flow best explains same-day")
    print("  price moves once you control for the others -> closest thing to 'who sets")
    print("  the price' in this data.")
    print("  Part 2: positive correlation with close-vs-vwap = that type's buying tends")
    print("  to coincide with the stock closing near its session high (aggressive,")
    print("  price-pushing). Near-zero/negative = that type is a passive counterparty,")
    print("  filled without moving the close.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
