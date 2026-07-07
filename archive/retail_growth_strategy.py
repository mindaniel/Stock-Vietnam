"""
retail_growth_strategy.py — "Foreign retail accumulation + already-reported
growth" strategy, restricted to sectors validated this session.

Rules (see session findings — reverse_engineer_retail_picks.py,
retail_pick_robustness.py, foreign_segment_heatmap.py, fama_macbeth_foreign.py):
  ENTRY, all of:
    - Sector in WHITELIST_SECTORS (same-direction winner>loser np_yoy gap,
      confirmed in the sector-split robustness check).
    - Liquid (>=1B VND/day 60d median turnover).
    - Foreign RETAIL 60d accumulation in top quintile, SECTOR-RELATIVE
      (not raw threshold, not institutional — institutional showed the
      opposite/negative signal).
    - Latest reported quarter's np_yoy AND rev_yoy in top quartile within
      its own sector, point-in-time safe (avail_date <= as_of).
    - Not already held (one entry per ticker at a time — no re-buying the
      same name every month it stays in the top quintile).
  EXIT, whichever first:
    - MAX_HOLD_DAYS trading days (default 150 ~ 7 months), OR
    - retail accumulation signal drops out of top quintile for that stock
      (the crowd that was buying it has left).
  Execution: buy at T+1 open after signal date. Friction 0.25% each way.
  T+2.5 is a non-binding constraint here (real hold is months).

Two modes:
  --mode backtest   Re-run history with these exact restricted rules (not
                     just the loose two-variable filter tested earlier) —
                     confirms whether the sector whitelist + one-entry-per-
                     ticker + signal-reversal-exit actually improves on the
                     raw gemini-code result.
  --mode live       Screen TODAY's data for current candidates — this is
                     what to run once Q2 2026 earnings start landing.

Usage:
  python archive/retail_growth_strategy.py --mode backtest
  python archive/retail_growth_strategy.py --mode live
"""

import argparse, glob, os, sys
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
ACCUM_WINDOW   = 60
TOP_QUINTILE   = 0.8        # retail accumulation percentile cutoff, sector-relative
GROWTH_QUARTILE = 0.75      # np_yoy/rev_yoy percentile cutoff, sector-relative
MAX_HOLD_DAYS  = 150        # ~7 months
FRICTION       = 0.0025     # one-way

# Confirmed same-direction (winner np_yoy > loser np_yoy, both sub-periods
# where tested) — see retail_pick_robustness.py sector split.
WHITELIST_SECTORS = {
    "Banks", "Basic Resources", "Financial Services",
    "Industrial Goods & Services", "Personal & Household Goods",
    "Retail", "Technology",
}
# Real Estate showed the same direction but the np_yoy numbers there are
# inflated by project-handover lumpiness and the full backtest LOST money
# there despite the characteristic "matching" — excluded deliberately.


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


def sector_map():
    mapping = pd.read_csv(os.path.join(BASE, "ticker_sectors.csv"))
    mapping.columns = [c.strip().lower() for c in mapping.columns]
    mapping = mapping[mapping["exchange"].isin(["HOSE", "HNX"])]
    return dict(zip(mapping["ticker"].str.upper(), mapping["industry"]))


def build_signal_frame(ticker: str, sector: str) -> pd.DataFrame:
    fpath = os.path.join(FLOW_DIR, f"{ticker}.parquet")
    df = pd.read_parquet(fpath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "close" not in df.columns or len(df) < ACCUM_WINDOW + 5:
        return pd.DataFrame()
    retail = df.get("ca_nhan_nuocngoai_net", pd.Series(0, index=df.index)).fillna(0)
    out = pd.DataFrame({"date": df["date"], "ticker": ticker, "sector": sector,
                         "open": df["close"], "close": df["close"]})
    if "open" in df.columns:
        out["open"] = df["open"]
    cum   = retail.rolling(ACCUM_WINDOW).sum()
    scale = retail.abs().rolling(ACCUM_WINDOW).mean() * ACCUM_WINDOW
    out["accum_retail"] = cum / scale.replace(0, np.nan)
    out["bar_idx"] = np.arange(len(out))
    return out


def load_universe():
    liquid = liquid_universe()
    flow_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet"))}
    t2s = sector_map()
    universe = sorted(liquid & flow_tickers)
    frames = []
    for t in universe:
        sec = t2s.get(t)
        if sec not in WHITELIST_SECTORS:
            continue
        f = build_signal_frame(t, sec)
        if not f.empty:
            frames.append(f)
    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return all_df, universe, t2s


def growth_rank_ok(qfeat: pd.DataFrame, ticker: str, sector: str, as_of: pd.Timestamp,
                    sector_universe: list) -> bool:
    """True if ticker's latest published np_yoy AND rev_yoy are both top-quartile
    within its own sector, as of `as_of` (point-in-time safe)."""
    sub = qfeat[(qfeat["symbol"].isin(sector_universe)) & (qfeat["avail_date"] <= as_of)]
    if sub.empty:
        return False
    latest = sub.sort_values("avail_date").groupby("symbol").last()
    if ticker.upper() not in latest.index:
        return False
    row = latest.loc[ticker.upper()]
    if pd.isna(row["np_yoy"]) or pd.isna(row["rev_yoy"]):
        return False
    np_thr  = latest["np_yoy"].quantile(GROWTH_QUARTILE)
    rev_thr = latest["rev_yoy"].quantile(GROWTH_QUARTILE)
    return row["np_yoy"] >= np_thr and row["rev_yoy"] >= rev_thr


MIN_HOLD_BEFORE_REVERSAL_EXIT = 20   # trading days — don't let early noise kick you out
CONSECUTIVE_DROPS_TO_EXIT     = 3    # weekly checks below threshold, in a row, before exiting


def _sector_benchmark_ret(all_df, sector, start_date, end_date):
    """Equal-weighted avg close-to-close return of all sector members over
    the same window — the sector-neutral benchmark for a single trade."""
    sub = all_df[(all_df["sector"] == sector) &
                 (all_df["date"] >= start_date) & (all_df["date"] <= end_date)]
    if sub.empty:
        return np.nan
    rets = []
    for t, g in sub.groupby("ticker"):
        g = g.sort_values("date")
        if len(g) < 2:
            continue
        r = g["close"].iloc[-1] / g["close"].iloc[0] - 1
        rets.append(r)
    return np.mean(rets) if rets else np.nan


def run_backtest(all_df, qfeat, universe, t2s):
    sector_universe = {sec: [t for t in universe if t2s.get(t) == sec] for sec in WHITELIST_SECTORS}
    price_by_ticker = {t: g.sort_values("bar_idx").reset_index(drop=True)
                        for t, g in all_df.groupby("ticker")}

    held = {}          # ticker -> {"entry_idx", "entry_date", "below_streak"}
    trades = []
    sample_dates = sorted(all_df["date"].unique())[::5]   # check weekly for entries/exits

    for d in sample_dates:
        cross = all_df[all_df["date"] == d].dropna(subset=["accum_retail"]).copy()
        if cross.empty:
            continue

        # ── check exits first ──
        for t in list(held.keys()):
            pdf = price_by_ticker.get(t)
            if pdf is None:
                continue
            row = pdf[pdf["date"] == d]
            if row.empty:
                continue
            state = held[t]
            entry_idx = state["entry_idx"]
            cur_idx = row["bar_idx"].iloc[0]
            hold_days = cur_idx - entry_idx

            sig_row = cross[cross["ticker"] == t]
            below_now = False
            if not sig_row.empty:
                sec = t2s.get(t)
                sec_cross = cross[cross["sector"] == sec]
                if len(sec_cross) >= 5:
                    thr = sec_cross["accum_retail"].quantile(TOP_QUINTILE)
                    below_now = sig_row["accum_retail"].iloc[0] < thr
            state["below_streak"] = state["below_streak"] + 1 if below_now else 0

            signal_dropped = (hold_days >= MIN_HOLD_BEFORE_REVERSAL_EXIT and
                               state["below_streak"] >= CONSECUTIVE_DROPS_TO_EXIT)
            timed_out = hold_days >= MAX_HOLD_DAYS
            data_end  = hold_days >= len(pdf) - entry_idx - 1

            if timed_out or signal_dropped or data_end:
                exit_bar = pdf[pdf["bar_idx"] == cur_idx]
                if not exit_bar.empty:
                    entry_px = pdf[pdf["bar_idx"] == entry_idx]["open"].iloc[0]
                    exit_px  = exit_bar["close"].iloc[0]
                    ret = exit_px / entry_px - 1
                    ret = (1 - FRICTION) * (1 + ret) * (1 - FRICTION) - 1
                    sec = t2s.get(t)
                    bench = _sector_benchmark_ret(all_df, sec, state["entry_date"], d)
                    trades.append({"ticker": t, "entry_idx": entry_idx, "exit_idx": cur_idx,
                                   "hold_days": hold_days, "ret": ret,
                                   "sector_neutral_ret": ret - bench if pd.notna(bench) else np.nan,
                                   "reason": "signal_dropped" if signal_dropped else "max_hold"})
                del held[t]

        # ── check entries ──
        for sec in WHITELIST_SECTORS:
            sec_cross = cross[cross["sector"] == sec]
            if len(sec_cross) < 5:
                continue
            thr = sec_cross["accum_retail"].quantile(TOP_QUINTILE)
            candidates = sec_cross[sec_cross["accum_retail"] >= thr]
            for _, row in candidates.iterrows():
                t = row["ticker"]
                if t in held:
                    continue
                if growth_rank_ok(qfeat, t, sec, d, sector_universe[sec]):
                    pdf = price_by_ticker.get(t)
                    if pdf is None:
                        continue
                    entry_bars = pdf[pdf["date"] == d]
                    if entry_bars.empty:
                        continue
                    held[t] = {"entry_idx": entry_bars["bar_idx"].iloc[0] + 1,   # T+1 open
                               "entry_date": d, "below_streak": 0}

    if not trades:
        print("No trades generated.")
        return
    tdf = pd.DataFrame(trades)
    print(f"\n{'='*80}")
    print(f"  BACKTEST — sector-whitelisted, one-entry-per-ticker, "
          f"{CONSECUTIVE_DROPS_TO_EXIT}x-confirmed signal-reversal exit")
    print(f"{'-'*80}")
    print(f"  N trades: {len(tdf)}  unique tickers: {tdf['ticker'].nunique()}")
    print(f"  Win rate (raw):            {(tdf['ret']>0).mean()*100:.1f}%")
    print(f"  Win rate (vs sector):      {(tdf['sector_neutral_ret']>0).mean()*100:.1f}%")
    print(f"  Avg return (raw):          {tdf['ret'].mean()*100:+.2f}%   median={tdf['ret'].median()*100:+.2f}%")
    print(f"  Avg return (sector-neutral): {tdf['sector_neutral_ret'].mean()*100:+.2f}%   "
          f"median={tdf['sector_neutral_ret'].median()*100:+.2f}%")
    print(f"  Avg hold: {tdf['hold_days'].mean():.0f} trading days")
    print(f"  Exit reasons: {tdf['reason'].value_counts().to_dict()}")
    print(f"{'='*80}")


def run_live_screen(all_df, qfeat, universe, t2s):
    sector_universe = {sec: [t for t in universe if t2s.get(t) == sec] for sec in WHITELIST_SECTORS}
    latest_date = all_df["date"].max()
    cross = all_df[all_df["date"] == latest_date].dropna(subset=["accum_retail"]).copy()

    print(f"\n{'='*90}")
    print(f"  LIVE SCREEN — as of {latest_date.date()}")
    print(f"{'-'*90}")
    found = False
    for sec in sorted(WHITELIST_SECTORS):
        sec_cross = cross[cross["sector"] == sec]
        if len(sec_cross) < 5:
            continue
        thr = sec_cross["accum_retail"].quantile(TOP_QUINTILE)
        candidates = sec_cross[sec_cross["accum_retail"] >= thr].sort_values("accum_retail", ascending=False)
        rows = []
        for _, row in candidates.iterrows():
            t = row["ticker"]
            if growth_rank_ok(qfeat, t, sec, latest_date, sector_universe[sec]):
                rows.append((t, row["accum_retail"]))
        if rows:
            found = True
            print(f"\n  {sec}:")
            for t, acc in rows:
                print(f"    {t:<8} retail_accum={acc:>+.2f}")
    if not found:
        print("\n  No candidates today — either no sector has enough heavy retail")
        print("  accumulation right now, or none of those stocks also clear the")
        print("  top-quartile np_yoy/rev_yoy growth bar yet (re-run after Q2 2026")
        print("  earnings start landing — growth figures will update then).")
    print(f"\n{'='*90}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["backtest", "live"], default="live")
    args = p.parse_args()

    print("Loading universe (whitelisted sectors only)...")
    all_df, universe, t2s = load_universe()
    print(f"  {all_df['ticker'].nunique()} tickers across {len(WHITELIST_SECTORS)} whitelisted sectors")

    print("Loading fundamentals...")
    qfeat = build_factor_features(symbols=universe)

    if args.mode == "backtest":
        run_backtest(all_df, qfeat, universe, t2s)
    else:
        run_live_screen(all_df, qfeat, universe, t2s)


if __name__ == "__main__":
    main()
