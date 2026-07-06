"""
price_leads_flow_test.py — Reverse-direction test: does a price DROP predict
tu doanh (prop desk) or foreign investors becoming net BUYERS afterward?

Everything so far has tested "does flow predict price." This flips it: is
tu doanh / foreign flow reactive to price action (buying dips, selling
rallies) rather than (or in addition to) being a leading indicator?
Domestic institutional + retail are included too for comparison/contrast.

Method: for each ticker/day T, compute the trailing price return ending at
T (over h = 1/3/5/10 days), then correlate it against the FORWARD sum of
each investor type's net flow over the next h days (T+1..T+h). A negative
correlation means "price fell -> that investor type net-bought afterward"
(dip-buying / contrarian). A positive correlation means "price fell -> that
investor type net-sold afterward" (momentum-following / trend-confirming).

Universe: liquid tickers (>=1B VND/day 60d median turnover) with investor
flow data (Sep 2024-present, ~22 months — same universe/window used in
flow_predictive_test.py and accumulation_target_backtest.py).

Usage:  python archive/price_leads_flow_test.py
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

MIN_LIQUIDITY_VND = 1_000_000_000
HORIZONS = [1, 3, 5, 10]

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


def build_ticker_frame(ticker: str) -> pd.DataFrame:
    fpath = os.path.join(FLOW_DIR, f"{ticker}.parquet")
    df = pd.read_parquet(fpath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "close" not in df.columns or len(df) < max(HORIZONS) * 2 + 10:
        return pd.DataFrame()

    out = pd.DataFrame({"date": df["date"], "ticker": ticker})
    for h in HORIZONS:
        out[f"past_ret_{h}"] = df["close"] / df["close"].shift(h) - 1
    for col in FLOW_TYPES:
        if col not in df.columns:
            continue
        cs = df[col].cumsum()
        for h in HORIZONS:
            # sum of col over (t+1 .. t+h) = cumsum[t+h] - cumsum[t]
            out[f"fwd_{col}_{h}"] = cs.shift(-h) - cs
    return out


def main():
    print("Building liquid universe...")
    liquid = liquid_universe()
    flow_tickers = {os.path.splitext(os.path.basename(f))[0].upper()
                    for f in glob.glob(os.path.join(FLOW_DIR, "*.parquet"))}
    universe = sorted(liquid & flow_tickers)
    print(f"  {len(universe)} liquid tickers with flow data")

    frames = []
    for t in universe:
        f = build_ticker_frame(t)
        if not f.empty:
            frames.append(f)
    all_df = pd.concat(frames, ignore_index=True)
    print(f"  Built {len(all_df):,} ticker-day rows across {all_df['ticker'].nunique()} tickers")

    for h in HORIZONS:
        print(f"\n{'='*90}")
        print(f"  Past {h}-day return  ->  forward {h}-day net flow (Spearman IC)")
        print(f"{'-'*90}")
        print(f"  {'Investor type':<26} {'IC':>9}  {'n':>8}   interpretation")
        print(f"  {'-'*80}")
        ret_col = f"past_ret_{h}"
        for col, label in FLOW_TYPES.items():
            fwd_col = f"fwd_{col}_{h}"
            if fwd_col not in all_df.columns:
                continue
            sub = all_df.dropna(subset=[ret_col, fwd_col])
            ic = sub[ret_col].corr(sub[fwd_col], method="spearman")
            interp = ("price down -> they BUY after (dip-buyers)" if ic < -0.03 else
                      "price up -> they SELL after (profit-taking)" if ic < 0 else
                      "price up -> they BUY after (momentum-followers)" if ic > 0.03 else
                      "no clear relationship")
            print(f"  {label:<26} {ic:>+9.4f}  {len(sub):>8,}   {interp}")

    print(f"\n{'='*90}")
    print("  INTERPRETATION")
    print(f"{'-'*90}")
    print("  Negative IC: that investor type tends to BUY after price has fallen")
    print("  (contrarian/dip-buying) and SELL after price has risen. Positive IC:")
    print("  that type BUYS after price has risen (momentum-following/chasing) and")
    print("  SELLS after price has fallen (panic/stop-out selling). Compare tu doanh")
    print("  and foreign against domestic institutional/retail to see who is playing")
    print("  which role in this market.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
