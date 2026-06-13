"""
tick_whale.py
Intraday tick-level whale / liquidity zone detector for VN equities.

Per stock per day it computes:
  - Buy/sell imbalance (net direction pressure)
  - Block trade detection (single orders above per-symbol 95th pct threshold)
  - Late-session accumulation / distribution (last 45 min, ATC window)
  - Absorption signal (big flow but little price movement = hidden counter-party)
  - Volume profile: price levels with the most volume (support/resistance zones)
  - Composite whale score (0–100)

Usage:
  python tick_whale.py                        # latest available date, all symbols
  python tick_whale.py --date 2026-05-30      # specific date (YYYY-MM-DD)
  python tick_whale.py --symbol VNM HPG ACB   # one or more symbols, all dates
  python tick_whale.py --top 20               # show top 20 whale signals
  python tick_whale.py --profile HPG          # print volume profile for HPG (latest date)
  python tick_whale.py --all-dates            # process every date in the data
"""

import argparse
import os
import sys
import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TICK_DIR   = os.path.join(BASE_DIR, "data", "tick_data")
OUT_DIR    = os.path.join(BASE_DIR, "results")
SIGNAL_CSV = os.path.join(OUT_DIR, "whale_signals_daily.csv")

# HOSE session times (HH:MM)
ATC_START      = "14:15"   # Afternoon continuous close approach
LATE_SESS_START= "14:00"
MORN_END       = "11:30"
AFT_START      = "13:00"

BLOCK_PERCENTILE = 95      # per-symbol threshold for "large single order"
MIN_TRADES       = 10      # skip symbols with too few ticks on a day


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_td(td_str: str) -> pd.Timestamp | None:
    """Parse dd/mm/YYYY → pd.Timestamp."""
    try:
        return pd.to_datetime(td_str, format="%d/%m/%Y")
    except Exception:
        return None


def load_symbol(symbol: str) -> pd.DataFrame | None:
    path = os.path.join(TICK_DIR, f"{symbol.upper()}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["date"] = df["td"].apply(parse_td)
    df = df.dropna(subset=["date"])
    df["p"]  = pd.to_numeric(df["p"], errors="coerce")
    df["v"]  = pd.to_numeric(df["v"], errors="coerce")
    df["s"]  = df["s"].astype(str).str.lower().str.strip()
    df["t"]  = df["t"].astype(str).str.strip()
    df["notional"] = df["p"] * df["v"]   # VND value of this tick
    return df.dropna(subset=["p", "v"])


def available_symbols() -> list[str]:
    return [
        os.path.basename(f).replace(".parquet", "").upper()
        for f in glob.glob(os.path.join(TICK_DIR, "*.parquet"))
    ]


def latest_date(df: pd.DataFrame) -> pd.Timestamp:
    return df["date"].max()


# ── Per-symbol block threshold (computed once over full history) ──────────────

def compute_block_thresholds(symbols: list[str]) -> dict[str, float]:
    thresholds = {}
    for sym in symbols:
        df = load_symbol(sym)
        if df is None or df.empty:
            continue
        thresholds[sym] = float(df["v"].quantile(BLOCK_PERCENTILE / 100))
    return thresholds


# ── Daily signal computation ──────────────────────────────────────────────────

def daily_signals(df: pd.DataFrame, date: pd.Timestamp, block_thresh: float) -> dict | None:
    """Compute whale/liquidity signals for one symbol on one date."""
    day = df[df["date"] == date].copy()
    if len(day) < MIN_TRADES:
        return None

    # side flags
    is_buy  = day["s"] == "buy"
    is_sell = day["s"] == "sell"

    buy_vol  = float(day.loc[is_buy,  "v"].sum())
    sell_vol = float(day.loc[is_sell, "v"].sum())
    total_vol= float(day["v"].sum())
    buy_val  = float(day.loc[is_buy,  "notional"].sum())
    sell_val = float(day.loc[is_sell, "notional"].sum())
    total_val= float(day["notional"].sum())

    imbalance_vol = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0
    imbalance_val = (buy_val - sell_val) / total_val if total_val > 0 else 0.0

    # Block trades
    blocks     = day[day["v"] >= block_thresh]
    blk_buy    = blocks[blocks["s"] == "buy"]
    blk_sell   = blocks[blocks["s"] == "sell"]
    n_blk_buy  = len(blk_buy)
    n_blk_sell = len(blk_sell)
    blk_buy_vol  = float(blk_buy["v"].sum())
    blk_sell_vol = float(blk_sell["v"].sum())
    blk_buy_val  = float(blk_buy["notional"].sum())
    blk_sell_val = float(blk_sell["notional"].sum())

    # Late-session pressure (after 14:00 — distribution / accumulation tell)
    late = day[day["t"] >= LATE_SESS_START]
    late_vol   = float(late["v"].sum())
    late_pct   = late_vol / total_vol if total_vol > 0 else 0.0
    late_buy   = float(late[late["s"] == "buy"]["v"].sum())
    late_imbal = (late_buy / late_vol - 0.5) * 2 if late_vol > 0 else 0.0  # -1 to +1

    # Price info
    vwap    = float((day["p"] * day["v"]).sum() / total_vol) if total_vol > 0 else np.nan
    p_open  = float(day.sort_values("t")["p"].iloc[0])
    p_close = float(day.sort_values("t")["p"].iloc[-1])
    p_high  = float(day["p"].max())
    p_low   = float(day["p"].min())
    close_vs_vwap = (p_close - vwap) / vwap if vwap and vwap > 0 else np.nan

    # Absorption: large net flow but small price move → hidden counter-party
    price_chg_pct = abs(p_close - p_open) / p_open if p_open > 0 else np.nan
    # absorption_score: high = lot of flow, little price move
    absorption = (1 - min(price_chg_pct / 0.05, 1.0)) * abs(imbalance_vol) if not np.isnan(price_chg_pct) else 0.0

    # Max single trade
    max_trade_vol = float(day["v"].max())
    max_trade_val = float(day["notional"].max())

    # ── Composite whale score (0–100) ────────────────────────────────────────
    # Each component contributes up to ±points, then scaled to 0-100.
    # Positive = accumulation signal.  Negative = distribution signal.
    raw = 0.0

    # 1. Overall imbalance (±25 pts)
    raw += imbalance_vol * 25

    # 2. Block buy vs sell (±20 pts)
    blk_total = blk_buy_vol + blk_sell_vol
    if blk_total > 0:
        raw += ((blk_buy_vol - blk_sell_vol) / blk_total) * 20

    # 3. Late-session imbalance (±20 pts) — smarter money acts late
    raw += late_imbal * 20

    # 4. Close vs VWAP (±15 pts) — closing above VWAP = buyers in control
    if not np.isnan(close_vs_vwap):
        raw += np.clip(close_vs_vwap / 0.02, -1.0, 1.0) * 15

    # 5. Block count asymmetry (±10 pts)
    n_blk_total = n_blk_buy + n_blk_sell
    if n_blk_total > 0:
        raw += ((n_blk_buy - n_blk_sell) / n_blk_total) * 10

    # 6. Absorption bonus (up to +10 pts) — only boosts accumulation signals
    if imbalance_vol > 0.1:
        raw += absorption * 10

    # Scale: raw is in range ~[-90, +90], map to 0–100 with 50 = neutral
    whale_score = int(np.clip(raw / 90 * 50 + 50, 0, 100))

    return {
        "date":          date.strftime("%Y-%m-%d"),
        "symbol":        day["symbol"].iloc[0],
        "sector":        day["sector"].iloc[0] if "sector" in day.columns else "",
        "platform":      day["platform"].iloc[0] if "platform" in day.columns else "",
        "n_trades":      len(day),
        "total_vol":     int(total_vol),
        "buy_vol":       int(buy_vol),
        "sell_vol":      int(sell_vol),
        "imbalance_vol": round(imbalance_vol, 4),
        "imbalance_val": round(imbalance_val, 4),
        "n_blk_buy":     n_blk_buy,
        "n_blk_sell":    n_blk_sell,
        "blk_buy_vol":   int(blk_buy_vol),
        "blk_sell_vol":  int(blk_sell_vol),
        "blk_buy_val_B": round(blk_buy_val / 1e9, 2),
        "blk_sell_val_B":round(blk_sell_val / 1e9, 2),
        "block_thresh":  int(block_thresh),
        "late_pct":      round(late_pct, 3),
        "late_imbal":    round(late_imbal, 4),
        "vwap":          round(vwap, 0),
        "p_open":        p_open,
        "p_close":       p_close,
        "p_high":        p_high,
        "p_low":         p_low,
        "close_vs_vwap": round(close_vs_vwap, 4) if not np.isnan(close_vs_vwap) else None,
        "price_chg_pct": round(price_chg_pct, 4) if not np.isnan(price_chg_pct) else None,
        "max_trade_vol": int(max_trade_vol),
        "max_trade_val_B": round(max_trade_val / 1e9, 3),
        "absorption":    round(absorption, 3),
        "whale_score":   whale_score,
    }


# ── Volume profile (price–volume histogram) ───────────────────────────────────

def volume_profile(symbol: str, date: pd.Timestamp | None = None, n_bins: int = 30):
    """
    Print a text volume profile (price vs volume) for one symbol.
    If date is None, uses the latest available date.
    High-volume nodes = likely support/resistance.
    """
    df = load_symbol(symbol)
    if df is None or df.empty:
        print(f"No tick data for {symbol}")
        return

    if date is None:
        date = latest_date(df)

    day = df[df["date"] == date]
    if day.empty:
        print(f"No data for {symbol} on {date.date()}")
        return

    p_min, p_max = day["p"].min(), day["p"].max()
    if p_min == p_max:
        print(f"No price range for {symbol} on {date.date()}")
        return

    bins  = np.linspace(p_min, p_max, n_bins + 1)
    total_vol = day["v"].sum()

    print(f"\n{'='*60}")
    print(f"VOLUME PROFILE  {symbol}  {date.date()}")
    print(f"Price range: {p_min:,.0f} – {p_max:,.0f} VND   Total vol: {total_vol:,.0f} shares")
    print(f"{'='*60}")

    profile = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (day["p"] >= lo) & (day["p"] < hi)
        vol  = float(day.loc[mask, "v"].sum())
        buy_vol = float(day.loc[mask & (day["s"] == "buy"), "v"].sum())
        profile.append((lo, hi, vol, buy_vol))

    max_vol = max(v for _, _, v, _ in profile) or 1
    poc_idx = max(range(n_bins), key=lambda i: profile[i][2])

    for i, (lo, hi, vol, buy_vol) in enumerate(profile):
        bar_len = int(vol / max_vol * 40)
        buy_len = int(buy_vol / max_vol * 40) if vol > 0 else 0
        bar = "█" * buy_len + "░" * (bar_len - buy_len)
        poc = " ◄ POC" if i == poc_idx else ""
        pct = vol / total_vol * 100 if total_vol > 0 else 0
        print(f"  {(lo+hi)/2:>8,.0f}  {bar:<40}  {pct:>5.1f}%{poc}")

    print(f"\nPOC (Point of Control): {(profile[poc_idx][0]+profile[poc_idx][1])/2:,.0f} VND  "
          f"— highest volume node, key support/resistance")
    print(f"{'='*60}")

    # High-volume nodes (top 20%)
    vols   = [v for _, _, v, _ in profile]
    thresh = np.percentile(vols, 80)
    hvn    = [(lo, hi, v) for lo, hi, v, _ in profile if v >= thresh]
    print(f"\nHigh-Volume Nodes (HVN) — strong support/resistance zones:")
    for lo, hi, v in hvn:
        print(f"  {lo:>8,.0f} – {hi:,.0f}  ({v/total_vol*100:.1f}% of day's volume)")


# ── Tick-size histogram (buy vs sell) ────────────────────────────────────────

def tick_size_hist(symbol: str, date: pd.Timestamp | None = None, bar_width: int = 35):
    """
    Side-by-side buy/sell histogram of trade sizes (log-scale bins).

    Each row = one size bucket.  Bar length = share of that side's trade COUNT.
    % column = share of that side's total VOLUME in the bucket.
    """
    df = load_symbol(symbol)
    if df is None or df.empty:
        print(f"No tick data for {symbol}")
        return

    if date is None:
        date = latest_date(df)

    day = df[df["date"] == date].copy()
    if day.empty:
        print(f"No data for {symbol} on {date.date()}")
        return

    buys  = day[day["s"] == "buy"]["v"].dropna().values
    sells = day[day["s"] == "sell"]["v"].dropna().values

    if len(buys) == 0 and len(sells) == 0:
        print("No buy/sell data.")
        return

    v_max = float(day["v"].max())

    # Human-readable log-scale bin edges
    raw_edges = [100, 300, 700, 1_500, 3_000, 7_000, 15_000,
                 30_000, 70_000, 150_000, 500_000, 1_500_000, 5_000_000]
    edges = [e for e in raw_edges if e <= v_max * 1.01]
    if not edges or edges[-1] < v_max:
        edges.append(int(v_max * 1.5))
    edges = sorted(set(edges))
    n_bins = len(edges) - 1

    buy_counts  = np.zeros(n_bins)
    sell_counts = np.zeros(n_bins)
    buy_vols    = np.zeros(n_bins)
    sell_vols   = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        buy_counts[i]  = ((buys  >= lo) & (buys  < hi)).sum()
        sell_counts[i] = ((sells >= lo) & (sells < hi)).sum()
        buy_vols[i]    = buys [(buys  >= lo) & (buys  < hi)].sum()
        sell_vols[i]   = sells[(sells >= lo) & (sells < hi)].sum()

    total_buy_count  = buy_counts.sum()  or 1
    total_sell_count = sell_counts.sum() or 1
    total_buy_vol    = buy_vols.sum()    or 1
    total_sell_vol   = sell_vols.sum()   or 1

    thresh = float(df["v"].quantile(BLOCK_PERCENTILE / 100))

    print(f"\n{'='*80}")
    print(f"TICK SIZE HISTOGRAM  {symbol}  {date.date()}")
    print(f"Trades: {int(total_buy_count)} buys / {int(total_sell_count)} sells   "
          f"Block threshold (p{BLOCK_PERCENTILE}): {thresh:,.0f} shares")
    print(f"{'='*80}")
    print(f"  {'Size bucket':<18}  {'BUY':>{bar_width}}  |  {'SELL':<{bar_width}}  vol%B  vol%S")
    print(f"  {'-'*18}  {'-'*bar_width}  |  {'-'*bar_width}  -----  -----")

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        is_block = lo >= thresh

        b_len = int(buy_counts[i]  / total_buy_count  * bar_width)
        s_len = int(sell_counts[i] / total_sell_count * bar_width)

        buy_bar  = ("█" * b_len).rjust(bar_width)   # grows leftward
        sell_bar = ("█" * s_len).ljust(bar_width)    # grows rightward

        bvpct = buy_vols[i]  / total_buy_vol  * 100
        svpct = sell_vols[i] / total_sell_vol * 100

        if lo >= 1_000:
            label = f"{lo/1_000:.0f}K-{hi/1_000:.0f}K"
        else:
            label = f"{lo:,}-{hi:,}"

        block_mark = " ◄ BLOCK" if is_block else ""
        print(f"  {label:<18}  {buy_bar}  |  {sell_bar}  {bvpct:>5.1f}% {svpct:>5.1f}%{block_mark}")

    print(f"{'='*80}")
    print(f"Median trade size:  buy={np.median(buys) if len(buys) else 0:>8,.0f}  "
          f"sell={np.median(sells) if len(sells) else 0:>8,.0f}")
    print(f"Mean  trade size:   buy={buys.mean() if len(buys) else 0:>8,.0f}  "
          f"sell={sells.mean() if len(sells) else 0:>8,.0f}")
    print(f"Max   trade size:   buy={buys.max() if len(buys) else 0:>8,.0f}  "
          f"sell={sells.max() if len(sells) else 0:>8,.0f}")

    n_block_buy  = int((buys  >= thresh).sum())
    n_block_sell = int((sells >= thresh).sum())
    print(f"\nBlock trades (>={thresh:,.0f}):  {n_block_buy} buys / {n_block_sell} sells")
    if n_block_buy > n_block_sell * 1.5:
        print("  -> Block buy dominance: large players accumulating")
    elif n_block_sell > n_block_buy * 1.5:
        print("  -> Block sell dominance: large players distributing")
    else:
        print("  -> Balanced block activity")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",      default=None, help="YYYY-MM-DD (default: latest)")
    ap.add_argument("--symbol",    nargs="*",    default=None, help="Symbols to process (default: all)")
    ap.add_argument("--top",       type=int,     default=20,   help="Show top N whale signals")
    ap.add_argument("--profile",   default=None, help="Print volume profile for this symbol")
    ap.add_argument("--hist",      default=None, help="Print tick-size buy/sell histogram for this symbol")
    ap.add_argument("--all-dates", action="store_true", help="Process all dates (slow)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Volume profile mode
    if args.profile:
        target_date = pd.Timestamp(args.date) if args.date else None
        volume_profile(args.profile.upper(), target_date)
        return

    # Tick-size histogram mode
    if args.hist:
        target_date = pd.Timestamp(args.date) if args.date else None
        tick_size_hist(args.hist.upper(), target_date)
        return

    symbols = args.symbol if args.symbol else available_symbols()
    symbols = [s.upper() for s in symbols]

    if not symbols:
        sys.exit(f"No tick data found in {TICK_DIR}")

    print(f"Computing block thresholds for {len(symbols)} symbols...")
    thresholds = compute_block_thresholds(symbols)

    # Determine target dates
    if args.all_dates:
        target_dates = None   # process all dates per symbol
    elif args.date:
        target_dates = {pd.Timestamp(args.date)}
    else:
        # Use latest date available across all symbols
        sample = load_symbol(symbols[0])
        target_dates = {latest_date(sample)} if sample is not None else None

    print(f"Processing {len(symbols)} symbols...")
    all_signals = []
    for i, sym in enumerate(symbols):
        df = load_symbol(sym)
        if df is None or df.empty:
            continue
        thresh = thresholds.get(sym, df["v"].quantile(0.95))

        dates_to_run = sorted(df["date"].unique()) if target_dates is None else [
            d for d in target_dates if d in df["date"].values
        ]
        for dt in dates_to_run:
            sig = daily_signals(df, dt, thresh)
            if sig:
                all_signals.append(sig)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(symbols)} done...")

    if not all_signals:
        print("No signals computed. Check --date or data availability.")
        return

    out = pd.DataFrame(all_signals).sort_values(["date", "whale_score"], ascending=[False, False])

    # Save / merge with existing
    if os.path.exists(SIGNAL_CSV):
        existing = pd.read_csv(SIGNAL_CSV)
        out = pd.concat([existing, out], ignore_index=True)
        out = out.drop_duplicates(subset=["date", "symbol"], keep="last")
        out = out.sort_values(["date", "whale_score"], ascending=[False, False])

    out.to_csv(SIGNAL_CSV, index=False)
    print(f"\nSaved {len(out)} signal rows → {SIGNAL_CSV}")

    # ── Print top signals for the most recent date ────────────────────────────
    latest_signals = out[out["date"] == out["date"].max()].head(args.top)
    latest_date_str = out["date"].max()

    print(f"\n{'='*95}")
    print(f"TOP {args.top} WHALE SIGNALS  —  {latest_date_str}")
    print(f"{'='*95}")
    print(f"{'#':>3} {'Sym':>6} {'Sector':<28} {'Score':>5} {'Imbal':>6} {'BlkB':>5} {'BlkS':>5} "
          f"{'LateImbal':>9} {'C/VWAP':>6} {'Absorb':>6}")
    print("-" * 95)
    for rank, (_, r) in enumerate(latest_signals.iterrows(), 1):
        print(f"{rank:>3} {r['symbol']:>6} {str(r['sector']):<28} {r['whale_score']:>5} "
              f"{r['imbalance_vol']:>+6.2f} {r['n_blk_buy']:>5} {r['n_blk_sell']:>5} "
              f"{r['late_imbal']:>+9.3f} {r['close_vs_vwap'] if r['close_vs_vwap'] else 0:>+6.3f} "
              f"{r['absorption']:>6.3f}")
    print("=" * 95)
    print("Score: 80+ = strong accumulation | 60-79 = moderate | 50 = neutral | <40 = distribution")
    print("Imbal: buy-side fraction (-1 to +1) | BlkB/BlkS: block buy/sell count")
    print("LateImbal: late-session buy pressure | C/VWAP: close relative to VWAP")
    print(f"\nTo see volume profile for a symbol:")
    print(f"  python tick_whale.py --profile <SYMBOL> --date {latest_date_str}")


if __name__ == "__main__":
    main()
