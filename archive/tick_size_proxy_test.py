"""
tick_size_proxy_test.py — Can trade-size (from tick data) approximate the
domestic institutional net flow that nguoiquansat.vn reports directly?

Idea: individuals rarely place single trades >= ~1B VND. Tick data
(data/tick_data/*.parquet) has per-print price/volume/side (buy or sell,
active taker side). Classify each print as "large" if its value crosses a
threshold, net large-buy minus large-sell value per ticker/day, and compare
against the REAL to_chuc_trongnuoc_net (domestic institutional) from
data/investor_flow/*.parquet for the same dates — ground truth, no lookahead
concerns since this is a same-day correlation test, not a forward-return test.

Overlap window: tick_data covers ~May-Jun 2026, investor_flow covers
Sep 2024-present, so ~2 months of overlap to validate against.

Usage:  python archive/tick_size_proxy_test.py
"""

import glob, os, sys
import pandas as pd
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICK_DIR  = os.path.join(BASE, "data", "tick_data")
FLOW_DIR  = os.path.join(BASE, "data", "investor_flow")

THRESHOLDS_BN = [0.2, 0.5, 1.0, 2.0]   # candidate "institutional-size" cutoffs


def daily_large_trade_net(ticker: str) -> pd.DataFrame:
    """Per-day net (buy - sell) value of trades >= threshold, for each threshold."""
    fpath = os.path.join(TICK_DIR, f"{ticker}.parquet")
    df = pd.read_parquet(fpath)
    df["date"] = pd.to_datetime(df["td"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date"])
    df["value_bn"] = df["p"].astype(float) * df["v"].astype(float) / 1e9
    df["signed"] = np.where(df["s"] == "buy", df["value_bn"], -df["value_bn"])

    out = df.groupby("date").agg(total_value_bn=("value_bn", "sum")).reset_index()
    for thr in THRESHOLDS_BN:
        mask = df["value_bn"] >= thr
        g = df[mask].groupby("date")["signed"].sum().rename(f"large_net_{thr}")
        out = out.merge(g, on="date", how="left")
        out[f"large_net_{thr}"] = out[f"large_net_{thr}"].fillna(0.0)
    out["ticker"] = ticker
    return out


def load_flow_ground_truth(ticker: str) -> pd.DataFrame:
    fpath = os.path.join(FLOW_DIR, f"{ticker}.parquet")
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_parquet(fpath)
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "to_chuc_trongnuoc_net", "tu_doanh_net", "to_chuc_nuocngoai_net",
               "ca_nhan_trongnuoc_net"]]


def main():
    tick_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(TICK_DIR, "*.parquet"))}
    flow_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet"))}
    universe = sorted(tick_tickers & flow_tickers)
    print(f"Tick data: {len(tick_tickers)} tickers | Flow data: {len(flow_tickers)} tickers "
          f"| overlap universe: {len(universe)}")

    frames = []
    for t in universe:
        try:
            tick_df = daily_large_trade_net(t)
            flow_df = load_flow_ground_truth(t)
            if tick_df.empty or flow_df.empty:
                continue
            m = tick_df.merge(flow_df, on="date", how="inner")
            if not m.empty:
                m["ticker"] = t
                frames.append(m)
        except Exception as e:
            continue

    if not frames:
        print("No overlapping data.")
        return
    all_df = pd.concat(frames, ignore_index=True)
    print(f"Merged {len(all_df):,} ticker-day rows, {all_df['ticker'].nunique()} tickers, "
          f"date range {all_df['date'].min().date()} to {all_df['date'].max().date()}")

    print(f"\n{'='*78}")
    print("  Trade-size distribution across all tickers (single-print value, billion VND)")
    print(f"{'-'*78}")
    sample_vals = []
    for t in universe[:40]:   # sample for speed
        df = pd.read_parquet(os.path.join(TICK_DIR, f"{t}.parquet"))
        v = (df["p"].astype(float) * df["v"].astype(float) / 1e9)
        sample_vals.append(v)
    all_vals = pd.concat(sample_vals, ignore_index=True)
    print(all_vals.describe(percentiles=[.5, .75, .9, .95, .99, .995, .999]))
    for thr in THRESHOLDS_BN:
        pct = (all_vals >= thr).mean() * 100
        print(f"  >= {thr}B VND: {pct:.3f}% of all prints")

    print(f"\n{'='*78}")
    print("  Correlation: large-trade net (tick proxy) vs REAL to_chuc_trongnuoc_net")
    print(f"  (domestic institutional net, from nguoiquansat — ground truth)")
    print(f"{'-'*78}")
    for thr in THRESHOLDS_BN:
        col = f"large_net_{thr}"
        sub = all_df.dropna(subset=[col, "to_chuc_trongnuoc_net"])
        pear = sub[col].corr(sub["to_chuc_trongnuoc_net"], method="pearson")
        spear = sub[col].corr(sub["to_chuc_trongnuoc_net"], method="spearman")
        # sign agreement: does the proxy at least get buy/sell direction right?
        sign_agree = (np.sign(sub[col]) == np.sign(sub["to_chuc_trongnuoc_net"])).mean()
        print(f"  threshold >= {thr}B:  pearson r={pear:>+.3f}  spearman r={spear:>+.3f}  "
              f"sign-agreement={sign_agree*100:.1f}%  (n={len(sub):,})")

    print(f"\n{'-'*78}")
    print("  Same correlations vs tu_doanh_net (prop trading) and "
          "to_chuc_nuocngoai_net (foreign), for comparison:")
    for other in ["tu_doanh_net", "to_chuc_nuocngoai_net", "ca_nhan_trongnuoc_net"]:
        col = f"large_net_{THRESHOLDS_BN[2]}"  # 1.0B threshold
        sub = all_df.dropna(subset=[col, other])
        pear = sub[col].corr(sub[other], method="pearson")
        print(f"  large_net_1.0B vs {other}: pearson r={pear:>+.3f}  (n={len(sub):,})")

    print(f"\n{'='*78}")
    print("  INTERPRETATION")
    print(f"{'-'*78}")
    print("  If large_net_X correlates strongly (r > ~0.3-0.4) and sign-agreement is")
    print("  well above 50%, size-based tick classification is a usable same-day proxy")
    print("  for the domestic institutional flow you'd otherwise wait ~2 days for from")
    print("  nguoiquansat. If correlation is weak/near zero, size alone doesn't separate")
    print("  institutional from retail order flow in this market and the idea doesn't")
    print("  hold without a better classifier (e.g. order clustering / algo detection).")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
