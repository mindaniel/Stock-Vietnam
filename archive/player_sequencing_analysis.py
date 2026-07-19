"""
player_sequencing_analysis.py — For genuine price "rides" (swing-low to
swing-high moves, identified from price action itself, not a fixed
calendar window), which investor type accumulates FIRST, which follows,
and which shows up latest (chasing the top)? And what do the winning rides
have in common (sector, growth, leverage) by who moved first?

METHOD (deliberately period-free, per the user's request):
  1. Detect swing pivots directly from price (order=SWING_ORDER bars each
     side — same technique as archive/4sectors.py's _swing_score_for_stock).
     A "ride" = a swing LOW followed by the next swing HIGH, i.e. an
     endogenously-defined trend episode, not a fixed 1M/3M/6M window.
  2. Keep only "good return" rides: gain from low to high >= MIN_RIDE_GAIN,
     duration between MIN_RIDE_DAYS and MAX_RIDE_DAYS (excludes noise blips
     and multi-year moves too long to attribute to one flow episode).
  3. For each of the 5 players, compute a flow-weighted "center of mass" day
     within the ride: sum(day_index * net_buying) / sum(net_buying), using
     only NET BUYING days (net_flow > 0) so heavy late-ride selling by one
     player doesn't drag its own timing estimate backward. Normalize to
     [0, 1] = fraction of the ride's duration (0 = right at the low,
     1 = right at the high). This is the "when did they concentrate their
     buying" signal — robust to noisy single-day flow, no arbitrary
     early/mid/late bucket boundaries.
  4. Rank players by mean/median normalized entry timing across all rides
     -> answers "who bought first/second/...last."
  5. For each ride, find whichever player had the EARLIEST center of mass
     (the "first mover") and attach start-of-ride fundamentals (sector,
     np_yoy, roe, debt_equity) -> compare fundamentals of rides grouped by
     first-mover player, to see if e.g. "foreign institutions move first"
     rides look different (growthier, different sector mix) from rides
     domestic retail moves first on.

Usage:  python archive/player_sequencing_analysis.py
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

SWING_ORDER    = 5     # bars each side to confirm a pivot (same as 4sectors.py)
MIN_RIDE_GAIN  = 0.25  # only "good return" rides: >=25% low-to-high
MIN_RIDE_DAYS  = 15    # exclude noise blips
MAX_RIDE_DAYS  = 200   # exclude multi-year moves (not one attributable flow episode)
MIN_LIQUIDITY_VND = 1_000_000_000

PLAYER_COLS = {
    "Foreign_Retail":       "ca_nhan_nuocngoai_net",
    "Foreign_Institutional":"to_chuc_nuocngoai_net",
    "Domestic_Retail":      "ca_nhan_trongnuoc_net",
    "Domestic_Institutional":"to_chuc_trongnuoc_net",
    "Proprietary":          "tu_doanh_net",
}


def find_pivots(high: np.ndarray, low: np.ndarray, order: int = SWING_ORDER):
    """Confirmed swing high/low indices (order bars needed on both sides)."""
    n = len(high)
    highs, lows = [], []
    for i in range(order, n - order):
        if high[i] == high[i - order:i + order + 1].max():
            highs.append(i)
        if low[i] == low[i - order:i + order + 1].min():
            lows.append(i)
    return highs, lows


def find_rides(df: pd.DataFrame) -> list:
    """Return list of (low_idx, high_idx, gain_pct) for every swing-low ->
    next-swing-high move meeting the gain/duration filters."""
    if "high" not in df.columns or "low" not in df.columns:
        return []
    h, l = df["high"].values, df["low"].values
    highs, lows = find_pivots(h, l)
    if not highs or not lows:
        return []

    rides = []
    for low_idx in lows:
        later_highs = [hi for hi in highs if hi > low_idx]
        if not later_highs:
            continue
        high_idx = later_highs[0]  # next confirmed high after this low
        duration = high_idx - low_idx
        if not (MIN_RIDE_DAYS <= duration <= MAX_RIDE_DAYS):
            continue
        gain = (h[high_idx] - l[low_idx]) / l[low_idx]
        if gain >= MIN_RIDE_GAIN:
            rides.append((low_idx, high_idx, gain))
    return rides


def liquid_universe():
    liquid = set()
    for fpath in glob.glob(os.path.join(PRICE_DIR, "*.parquet")):
        ticker = os.path.splitext(os.path.basename(fpath))[0].upper()
        try:
            df = pd.read_parquet(fpath)
            df.columns = [c.strip().lower() for c in df.columns]
            if "close" not in df.columns or "volume" not in df.columns:
                continue
            med_to = (df["close"] * df["volume"] * 1000).median()
            if med_to >= MIN_LIQUIDITY_VND:
                liquid.add(ticker)
        except Exception:
            pass
    return liquid


def center_of_mass_timing(flow: pd.Series) -> float:
    """Flow-weighted center-of-mass day index among NET-BUYING days only,
    normalized to [0,1] over len(flow). NaN if no net buying at all."""
    buys = flow.clip(lower=0)
    total = buys.sum()
    if total <= 0:
        return np.nan
    idx = np.arange(len(flow))
    com = (idx * buys.values).sum() / total
    return com / max(len(flow) - 1, 1)


def main():
    print("Building liquid + flow + sector universe...")
    liquid = liquid_universe()
    flow_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet"))}
    mapping = pd.read_csv(os.path.join(BASE, "ticker_sectors.csv"))
    mapping.columns = [c.strip().lower() for c in mapping.columns]
    mapping = mapping[mapping["exchange"].isin(["HOSE", "HNX"])]
    ticker_to_sector = dict(zip(mapping["ticker"].str.upper(), mapping["industry"]))
    universe = sorted(liquid & flow_tickers & set(ticker_to_sector))
    print(f"  {len(universe)} tickers")

    print("Loading fundamentals...")
    qfeat = build_factor_features(symbols=universe)
    qfeat_by_sym = {sym: g.sort_values("avail_date") for sym, g in qfeat.groupby("symbol")}

    print("Detecting price rides + player timing (this scans full history per ticker)...")
    ride_rows = []       # one row per (ride, player) -> timing
    ride_summary = []    # one row per ride -> first mover + fundamentals

    for i, ticker in enumerate(universe):
        fpath = os.path.join(FLOW_DIR, f"{ticker}.parquet")
        if not os.path.exists(fpath):
            continue
        flow = pd.read_parquet(fpath)
        flow["date"] = pd.to_datetime(flow["date"])
        flow = flow.sort_values("date").reset_index(drop=True)

        price_path = os.path.join(PRICE_DIR, f"{ticker}.parquet")
        if not os.path.exists(price_path):
            continue
        price = pd.read_parquet(price_path)
        price.columns = [c.strip().lower() for c in price.columns]
        date_col = "time" if "time" in price.columns else "date"
        price["date"] = pd.to_datetime(price[date_col])
        price = price.sort_values("date").reset_index(drop=True)
        if "high" not in price.columns or "low" not in price.columns:
            continue

        df = flow.merge(price[["date", "high", "low"]], on="date", how="inner")
        if len(df) < MIN_RIDE_DAYS + 2 * SWING_ORDER:
            continue

        rides = find_rides(df)
        if not rides:
            continue

        sector = ticker_to_sector.get(ticker)
        sub_fund = qfeat_by_sym.get(ticker)

        for low_idx, high_idx, gain in rides:
            ride_id = f"{ticker}_{df['date'].iloc[low_idx].date()}"
            timings = {}
            for label, col in PLAYER_COLS.items():
                if col not in df.columns:
                    continue
                window_flow = df[col].iloc[low_idx:high_idx + 1].fillna(0)
                t = center_of_mass_timing(window_flow)
                timings[label] = t
                ride_rows.append({
                    "ride_id": ride_id, "ticker": ticker, "sector": sector,
                    "gain_pct": gain, "duration_days": high_idx - low_idx,
                    "player": label, "timing": t,
                })

            valid_timings = {k: v for k, v in timings.items() if not np.isnan(v)}
            if not valid_timings:
                continue
            first_mover = min(valid_timings, key=valid_timings.get)

            rec = {
                "ride_id": ride_id, "ticker": ticker, "sector": sector,
                "gain_pct": gain, "duration_days": high_idx - low_idx,
                "first_mover": first_mover,
                "start_date": df["date"].iloc[low_idx],
            }
            if sub_fund is not None:
                q0 = sub_fund[sub_fund["avail_date"] <= df["date"].iloc[low_idx]].tail(1)
                if not q0.empty:
                    rec["np_yoy"] = q0["np_yoy"].iloc[0]
                    rec["roe"] = q0["roe"].iloc[0]
                    rec["debt_equity"] = q0["debt_equity"].iloc[0]
            ride_summary.append(rec)

        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(universe)} tickers, {len(ride_summary)} rides found so far")

    timing_df = pd.DataFrame(ride_rows)
    summary_df = pd.DataFrame(ride_summary)
    print(f"\nTotal rides found: {len(summary_df)} across {summary_df['ticker'].nunique()} tickers")
    print(f"Gain range: {summary_df['gain_pct'].min()*100:.0f}% - {summary_df['gain_pct'].max()*100:.0f}%  "
          f"(median {summary_df['gain_pct'].median()*100:.0f}%)")
    print(f"Duration range: {summary_df['duration_days'].min()} - {summary_df['duration_days'].max()} days  "
          f"(median {summary_df['duration_days'].median():.0f})")

    print(f"\n{'='*90}")
    print("  WHO BUYS FIRST? (mean/median normalized timing, 0=at the low, 1=at the high)")
    print(f"{'-'*90}")
    agg = timing_df.dropna(subset=["timing"]).groupby("player")["timing"].agg(
        ["mean", "median", "std", "count"]).sort_values("mean")
    print(agg.to_string(float_format=lambda x: f"{x:.3f}"))

    print(f"\n{'='*90}")
    print("  FIRST-MOVER DISTRIBUTION (which player has the earliest timing per ride)")
    print(f"{'-'*90}")
    print(summary_df["first_mover"].value_counts().to_string())

    print(f"\n{'='*90}")
    print("  FUNDAMENTALS AT RIDE START, GROUPED BY FIRST MOVER")
    print(f"{'-'*90}")
    fund_cols = [c for c in ["np_yoy", "roe", "debt_equity", "gain_pct", "duration_days"] if c in summary_df.columns]
    print(summary_df.groupby("first_mover")[fund_cols].mean().to_string(float_format=lambda x: f"{x:.3f}"))
    print()
    print("Sector distribution by first mover (top 3 sectors each):")
    for mover, g in summary_df.groupby("first_mover"):
        top_sectors = g["sector"].value_counts().head(3)
        print(f"  {mover}: {dict(top_sectors)}")

    out_path = os.path.join(BASE, "backtest_reports", "player_sequencing_rides.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved ride-level detail to {out_path}")


if __name__ == "__main__":
    main()
