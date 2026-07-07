"""
retail_pick_robustness.py — Is "foreign retail wins by picking high-growth
stocks" a real, robust pattern, or a lucky read on ~22 months of data?

Two things this adds beyond reverse_engineer_retail_picks.py:
  1. More characteristics: pre-entry price momentum (were winners already
     rallying, or laggards/turnarounds?), debt/equity (quality/leverage),
     and liquidity/size — not just np_yoy/roe/accel_score/rev_yoy.
  2. SPLIT-SAMPLE ROBUSTNESS: divide the sample period into two halves and
     re-run the winner-vs-loser characteristic comparison in EACH half
     separately. If the same characteristics separate winners from losers
     in BOTH halves independently, that's real signal, not one lucky
     stretch. If the direction flips or disappears in either half, treat
     the full-period result with much more skepticism.

Usage:  python archive/retail_pick_robustness.py
"""

import glob, os, sys
import numpy as np
import pandas as pd

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
FWD_HORIZON   = 120
SAMPLE_STRIDE = 20
MIN_STOCKS_PER_SECTOR_PERIOD = 3
TOP_QUINTILE  = 0.8


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
    if "close" not in df.columns or len(df) < ACCUM_WINDOW + FWD_HORIZON + 65:
        return pd.DataFrame()
    retail = df.get("ca_nhan_nuocngoai_net", pd.Series(0, index=df.index)).fillna(0)
    out = pd.DataFrame({"date": df["date"], "ticker": ticker, "sector": sector})
    cum   = retail.rolling(ACCUM_WINDOW).sum()
    scale = retail.abs().rolling(ACCUM_WINDOW).mean() * ACCUM_WINDOW
    out["accum_retail"] = cum / scale.replace(0, np.nan)
    out["fwd_ret"] = df["close"].shift(-FWD_HORIZON) / df["close"] - 1
    # pre-entry momentum: 60d return INTO the entry date (already known, no lookahead)
    out["mom_60d_pre"] = df["close"] / df["close"].shift(60) - 1
    out["close"] = df["close"]
    return out


def find_picks(all_df):
    winners, losers = [], []
    sample_dates = sorted(all_df["date"].unique())[::SAMPLE_STRIDE]
    for d in sample_dates:
        cross = all_df[all_df["date"] == d].dropna(subset=["accum_retail", "fwd_ret"]).copy()
        if len(cross) < 15:
            continue
        sec_counts = cross.groupby("sector")["ticker"].transform("count")
        cross = cross[sec_counts >= MIN_STOCKS_PER_SECTOR_PERIOD]
        if cross.empty:
            continue
        cross["sector_neutral_ret"] = cross["fwd_ret"] - cross.groupby("sector")["fwd_ret"].transform("mean")
        thresh = cross["accum_retail"].quantile(TOP_QUINTILE)
        heavy = cross[cross["accum_retail"] >= thresh]
        for _, row in heavy.iterrows():
            rec = {"ticker": row["ticker"], "entry_date": d, "sector": row["sector"],
                   "sector_neutral_ret": row["sector_neutral_ret"], "mom_60d_pre": row["mom_60d_pre"]}
            (winners if row["sector_neutral_ret"] > 0 else losers).append(rec)
    return pd.DataFrame(winners), pd.DataFrame(losers)


def attach_fundamentals(picks: pd.DataFrame, qfeat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, p in picks.iterrows():
        sym = p["ticker"].upper()
        sub = qfeat[qfeat["symbol"] == sym].sort_values("avail_date")
        q0 = sub[sub["avail_date"] <= p["entry_date"]].tail(1)
        if q0.empty:
            continue
        rec = dict(p)
        rec["np_yoy"] = q0["np_yoy"].iloc[0]
        rec["roe"] = q0["roe"].iloc[0]
        rec["debt_equity"] = q0["debt_equity"].iloc[0]
        rec["rev_yoy"] = q0["rev_yoy"].iloc[0]
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(label, df):
    print(f"  {label:<10} n={len(df):>4}  "
          f"np_yoy={df['np_yoy'].mean():>+7.3f}  "
          f"roe={df['roe'].mean():>6.2f}  "
          f"debt_eq={df['debt_equity'].mean():>6.2f}  "
          f"rev_yoy={df['rev_yoy'].mean():>+7.3f}  "
          f"mom60d_pre={df['mom_60d_pre'].mean()*100:>+7.2f}%")


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
    print(f"  {all_df['ticker'].nunique()} tickers, {all_df['sector'].nunique()} sectors, "
          f"date range {all_df['date'].min().date()} to {all_df['date'].max().date()}")

    winners, losers = find_picks(all_df)
    print(f"  Winners: {len(winners)}  Losers: {len(losers)}")

    print("\nLoading fundamentals...")
    qfeat = build_factor_features(symbols=universe)
    win_f = attach_fundamentals(winners, qfeat)
    los_f = attach_fundamentals(losers, qfeat)

    print(f"\n{'='*100}")
    print("  FULL PERIOD")
    print(f"{'-'*100}")
    summarize("Winners", win_f)
    summarize("Losers", los_f)

    # ── Split-sample robustness ──────────────────────────────────────────
    all_dates = pd.concat([win_f["entry_date"], los_f["entry_date"]])
    midpoint = all_dates.median()
    print(f"\n{'='*100}")
    print(f"  SPLIT-SAMPLE ROBUSTNESS  (median split at {pd.Timestamp(midpoint).date()})")
    print(f"{'-'*100}")
    for half_label, half_mask_fn in [("H1 (early)", lambda d: d <= midpoint),
                                       ("H2 (late)",  lambda d: d > midpoint)]:
        print(f"\n  -- {half_label} --")
        w_half = win_f[half_mask_fn(win_f["entry_date"])]
        l_half = los_f[half_mask_fn(los_f["entry_date"])]
        summarize("Winners", w_half)
        summarize("Losers", l_half)

    print(f"\n{'='*100}")
    print("  INTERPRETATION")
    print(f"{'-'*100}")
    print("  If Winners > Losers on np_yoy/rev_yoy in BOTH H1 and H2 (not just the full-period")
    print("  average), that's a real, repeatable characteristic — not a lucky read on one half")
    print("  of the data driving the whole result. If the gap disappears or flips sign in")
    print("  either half, the full-period number is likely noise/luck, not a stable edge.")
    print("  Also check mom_60d_pre: if winners were already rallying hard before the pick,")
    print("  this is a momentum-chasing signal (buying what's already hot), not a value/")
    print("  turnaround pick — worth knowing which one you're actually relying on.")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
