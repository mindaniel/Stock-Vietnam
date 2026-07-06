"""
tick_size_band_scan.py — Which trade-size band (if any) actually tracks real
domestic institutional net flow?

Follow-up to tick_size_proxy_test.py, which showed "large trade = institutional"
is backwards (negative correlation, sign-agreement below 50%). Working theory:
institutions in VN typically slice orders into many small child prints (algo/
iceberg execution) specifically to avoid moving the price and avoid looking
like a block trade — so the real footprint should show up as a *persistent
directional bias across many small-to-medium trades*, not isolated big ones.

This scans discrete size bands (not cumulative >= threshold) and tests both:
  - value-net  (buy value - sell value within the band)
  - count-net  (buy count - sell count within the band)
against the ground-truth to_chuc_trongnuoc_net from nguoiquansat, for the
same May-Jun 2026 overlap window as before.

Usage:  python archive/tick_size_band_scan.py
"""

import glob, os, sys
import pandas as pd
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICK_DIR = os.path.join(BASE, "data", "tick_data")
FLOW_DIR = os.path.join(BASE, "data", "investor_flow")

# (low, high) in billion VND — low inclusive, high exclusive. None = open end.
BANDS = [
    (0.0,   0.02),
    (0.02,  0.05),
    (0.05,  0.10),
    (0.10,  0.20),
    (0.20,  0.50),
    (0.50,  1.00),
    (1.00,  2.00),
    (2.00,  None),
]


def band_label(lo, hi):
    return f"{lo}-{hi}B" if hi is not None else f">={lo}B"


def daily_band_flows(ticker: str) -> pd.DataFrame:
    fpath = os.path.join(TICK_DIR, f"{ticker}.parquet")
    df = pd.read_parquet(fpath)
    df["date"] = pd.to_datetime(df["td"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date"])
    df["value_bn"] = df["p"].astype(float) * df["v"].astype(float) / 1e9
    df["is_buy"] = (df["s"] == "buy").astype(int)

    out = None
    for lo, hi in BANDS:
        mask = (df["value_bn"] >= lo) if hi is None else \
               ((df["value_bn"] >= lo) & (df["value_bn"] < hi))
        b = df[mask].copy()
        lbl = band_label(lo, hi)
        b["signed_val"] = np.where(b["s"] == "buy", b["value_bn"], -b["value_bn"])
        b["signed_cnt"] = np.where(b["s"] == "buy", 1, -1)
        g = b.groupby("date").agg(
            **{f"val_{lbl}": ("signed_val", "sum"),
               f"cnt_{lbl}": ("signed_cnt", "sum")}
        )
        out = g if out is None else out.join(g, how="outer")
    out = out.fillna(0.0).reset_index()
    out["ticker"] = ticker
    return out


def load_flow_ground_truth(ticker: str) -> pd.DataFrame:
    fpath = os.path.join(FLOW_DIR, f"{ticker}.parquet")
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_parquet(fpath)
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "to_chuc_trongnuoc_net"]]


def main():
    tick_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(TICK_DIR, "*.parquet"))}
    flow_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet"))}
    universe = sorted(tick_tickers & flow_tickers)
    print(f"Universe: {len(universe)} tickers with both tick + flow data")

    frames = []
    for t in universe:
        try:
            band_df = daily_band_flows(t)
            flow_df = load_flow_ground_truth(t)
            if band_df.empty or flow_df.empty:
                continue
            m = band_df.merge(flow_df, on="date", how="inner")
            if not m.empty:
                frames.append(m)
        except Exception:
            continue

    if not frames:
        print("No data.")
        return
    all_df = pd.concat(frames, ignore_index=True)
    print(f"Merged {len(all_df):,} ticker-day rows, {all_df['ticker'].nunique()} tickers")

    print(f"\n{'='*86}")
    print("  Correlation vs REAL to_chuc_trongnuoc_net, by size band")
    print(f"{'-'*86}")
    print(f"  {'Band':<12} {'val pearson':>12} {'val sign-agree':>15}  "
          f"{'cnt pearson':>12} {'cnt sign-agree':>15}")
    print(f"  {'-'*84}")
    gt = all_df["to_chuc_trongnuoc_net"]
    for lo, hi in BANDS:
        lbl = band_label(lo, hi)
        vcol, ccol = f"val_{lbl}", f"cnt_{lbl}"
        vpear = all_df[vcol].corr(gt, method="pearson")
        cpear = all_df[ccol].corr(gt, method="pearson")
        vsign = (np.sign(all_df[vcol]) == np.sign(gt)).mean() * 100
        csign = (np.sign(all_df[ccol]) == np.sign(gt)).mean() * 100
        print(f"  {lbl:<12} {vpear:>+12.3f} {vsign:>14.1f}%  {cpear:>+12.3f} {csign:>14.1f}%")

    # ── Persistence / clustering measure: net count-imbalance across the
    # small+medium bands combined (proxy for "many same-direction small
    # trades", i.e. an order being worked) ──
    small_med_cnt_cols = [f"cnt_{band_label(lo,hi)}" for lo, hi in BANDS if lo < 1.0]
    all_df["cluster_cnt_net"] = all_df[small_med_cnt_cols].sum(axis=1)
    small_med_val_cols = [f"val_{band_label(lo,hi)}" for lo, hi in BANDS if lo < 1.0]
    all_df["cluster_val_net"] = all_df[small_med_val_cols].sum(axis=1)

    print(f"\n{'-'*86}")
    print("  Combined <1B bands (\"order-splitting\" proxy):")
    for col in ["cluster_cnt_net", "cluster_val_net"]:
        pear = all_df[col].corr(gt, method="pearson")
        sign = (np.sign(all_df[col]) == np.sign(gt)).mean() * 100
        print(f"    {col:<18}: pearson r={pear:>+.3f}  sign-agreement={sign:.1f}%")

    print(f"\n{'='*86}")
    print("  INTERPRETATION")
    print(f"{'-'*86}")
    print("  Look for the band(s) with positive pearson r and sign-agreement well")
    print("  above 50% — that's where genuine institutional footprint shows up.")
    print("  If small/medium bands beat large ones, order-splitting is the real")
    print("  pattern and whale_detector.py's 'large block buy ratio' scoring (20pts)")
    print("  and put-through deal scoring (10pts) are rewarding the wrong signal.")
    print(f"{'='*86}")


if __name__ == "__main__":
    main()
