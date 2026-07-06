"""
reverse_engineer_retail_picks.py — The Fama-MacBeth test showed foreign retail
accumulation predicts a stock beating its own sector peers (t~3, survives
sector-neutralization). This digs into WHY: for the specific stocks foreign
retail piled into that then won, were they already showing stronger
fundamentals (value/quality tilt), or did their NEXT reported quarter show
an earnings acceleration/surprise the market hadn't priced in yet (informed
anticipation)? Contrasts against the "losers" — stocks retail also
accumulated heavily but which did NOT beat their sector — to see whether
fundamentals are what actually separates the winning calls from the losing
ones within the accumulated group.

Method:
  1. Same sector-neutral 60-day foreign-retail-accumulation signal and 120d
     forward return as fama_macbeth_foreign.py, sampled ~monthly.
  2. WINNERS = top-quintile signal (heaviest retail accumulation, relative
     to sector peers that period) AND positive sector-neutral forward return.
     LOSERS = top-quintile signal AND negative sector-neutral forward return.
  3. For each pick (ticker, entry_date): look up
       Q0 = latest quarterly fundamentals already PUBLISHED as of entry_date
            (point-in-time safe — avail_date <= entry_date)
       Q1 = the NEXT quarter's fundamentals, i.e. the first one published
            AFTER entry_date (avail_date > entry_date) — this is the
            earnings print that comes out only after the buying happened.
  4. Compare Q0 (already-known) fundamentals of winners vs losers vs sector
     average — tests whether winners were picked for visible quality.
     Compare Q1 (not-yet-known at entry) np_yoy of winners vs losers vs
     their own Q0 — tests whether the winning picks anticipated an
     earnings improvement the market hadn't seen yet.

Usage:  python archive/reverse_engineer_retail_picks.py
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
FWD_HORIZON   = 120     # use the horizon with the strongest FM t-stat
SAMPLE_STRIDE = 20
MIN_STOCKS_PER_SECTOR_PERIOD = 3
TOP_QUINTILE  = 0.8     # signal must be in top 20% of that period's cross-section


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
    if "close" not in df.columns or len(df) < ACCUM_WINDOW + FWD_HORIZON + 5:
        return pd.DataFrame()
    retail = df.get("ca_nhan_nuocngoai_net", pd.Series(0, index=df.index)).fillna(0)
    out = pd.DataFrame({"date": df["date"], "ticker": ticker, "sector": sector})
    cum   = retail.rolling(ACCUM_WINDOW).sum()
    scale = retail.abs().rolling(ACCUM_WINDOW).mean() * ACCUM_WINDOW
    out["accum_retail"] = cum / scale.replace(0, np.nan)
    out["fwd_ret"] = df["close"].shift(-FWD_HORIZON) / df["close"] - 1
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
                   "sector_neutral_ret": row["sector_neutral_ret"]}
            if row["sector_neutral_ret"] > 0:
                winners.append(rec)
            else:
                losers.append(rec)
    return pd.DataFrame(winners), pd.DataFrame(losers)


def lookup_fundamentals(picks: pd.DataFrame, qfeat: pd.DataFrame, universe_sector_map: dict):
    """For each pick, find Q0 (last published <= entry_date) and Q1 (first published > entry_date)."""
    rows = []
    for _, p in picks.iterrows():
        sym = p["ticker"].upper()
        sub = qfeat[qfeat["symbol"] == sym].sort_values("avail_date")
        if sub.empty:
            continue
        q0 = sub[sub["avail_date"] <= p["entry_date"]].tail(1)
        q1 = sub[sub["avail_date"] > p["entry_date"]].head(1)
        if q0.empty:
            continue
        rec = {
            "ticker": sym, "entry_date": p["entry_date"], "sector": p["sector"],
            "sector_neutral_ret": p["sector_neutral_ret"],
            "q0_np_yoy": q0["np_yoy"].iloc[0], "q0_roe": q0["roe"].iloc[0],
            "q0_accel": q0["accel_score"].iloc[0], "q0_rev_yoy": q0["rev_yoy"].iloc[0],
        }
        if not q1.empty:
            rec["q1_np_yoy"] = q1["np_yoy"].iloc[0]
            rec["q1_accel"] = q1["accel_score"].iloc[0]
        rows.append(rec)
    return pd.DataFrame(rows)


def sector_avg_q0_at(qfeat: pd.DataFrame, universe_sector_map: dict, sector: str, as_of: pd.Timestamp) -> dict:
    """Average Q0 fundamentals across ALL universe stocks in `sector`, as of `as_of` — the
    contemporaneous sector benchmark (same point-in-time rule, no lookahead)."""
    syms = [s for s, sec in universe_sector_map.items() if sec == sector]
    sub = qfeat[qfeat["symbol"].isin(syms) & (qfeat["avail_date"] <= as_of)]
    if sub.empty:
        return {}
    latest = sub.sort_values("avail_date").groupby("symbol").last()
    return {"np_yoy": latest["np_yoy"].mean(), "roe": latest["roe"].mean(),
            "accel_score": latest["accel_score"].mean()}


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
    print(f"  {all_df['ticker'].nunique()} tickers, {all_df['sector'].nunique()} sectors")

    print("Finding winner/loser picks (top-quintile retail accumulation, 120d fwd)...")
    winners, losers = find_picks(all_df)
    print(f"  Winners: {len(winners)} picks ({winners['ticker'].nunique()} unique tickers)")
    print(f"  Losers:  {len(losers)} picks ({losers['ticker'].nunique()} unique tickers)")

    print("\nLoading quarterly fundamentals...")
    qfeat = build_factor_features(symbols=universe)
    universe_sector_map = {t: ticker_to_sector.get(t) for t in universe}

    win_fund = lookup_fundamentals(winners, qfeat, universe_sector_map)
    los_fund = lookup_fundamentals(losers, qfeat, universe_sector_map)

    print(f"\n{'='*90}")
    print("  Q0 FUNDAMENTALS (already public knowledge at time of accumulation)")
    print(f"{'-'*90}")
    print(f"  {'Group':<10} {'n':>6} {'np_yoy':>9} {'roe':>8} {'accel_score':>12} {'rev_yoy':>9}")
    for label, df in [("Winners", win_fund), ("Losers", los_fund)]:
        print(f"  {label:<10} {len(df):>6} {df['q0_np_yoy'].mean():>+9.3f} "
              f"{df['q0_roe'].mean():>8.2f} {df['q0_accel'].mean():>12.2f} "
              f"{df['q0_rev_yoy'].mean():>+9.3f}")

    # sector benchmark for Q0 (pooled across each pick's own sector/date context)
    sec_bench = []
    for _, p in pd.concat([winners, losers]).drop_duplicates(["sector", "entry_date"]).iterrows():
        b = sector_avg_q0_at(qfeat, universe_sector_map, p["sector"], p["entry_date"])
        if b:
            sec_bench.append(b)
    if sec_bench:
        bdf = pd.DataFrame(sec_bench)
        print(f"  {'Sector avg':<10} {len(bdf):>6} {bdf['np_yoy'].mean():>+9.3f} "
              f"{bdf['roe'].mean():>8.2f} {bdf['accel_score'].mean():>12.2f} {'':>9}")

    print(f"\n{'='*90}")
    print("  Q1 EARNINGS SURPRISE (the NEXT quarter's np_yoy — not known at entry)")
    print(f"{'-'*90}")
    for label, df in [("Winners", win_fund), ("Losers", los_fund)]:
        sub = df.dropna(subset=["q1_np_yoy"])
        accel = (sub["q1_np_yoy"] - sub["q0_np_yoy"]).mean()
        print(f"  {label:<10} n={len(sub):>4}  Q0 np_yoy={sub['q0_np_yoy'].mean():>+.3f}  "
              f"Q1 np_yoy={sub['q1_np_yoy'].mean():>+.3f}  "
              f"change (Q1-Q0)={accel:>+.3f}")

    print(f"\n{'='*90}")
    print("  INTERPRETATION")
    print(f"{'-'*90}")
    print("  Q0 section: do winners already look fundamentally different from losers (or")
    print("  the sector) BEFORE the buying happens? If yes -> retail is following visible")
    print("  quality/value signals, not foresight.")
    print("  Q1 section: does the NEXT reported quarter show winners accelerating more than")
    print("  losers, even though that quarter wasn't public yet at entry? If yes -> retail")
    print("  buying anticipates a real earnings improvement (informed), not just noise.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
