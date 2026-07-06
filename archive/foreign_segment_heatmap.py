"""
foreign_segment_heatmap.py — "Foreigners" aren't one player. Splits the
aggregate foreign flow into:
  (a) foreign INSTITUTIONAL (to_chuc_nuocngoai_net) vs foreign RETAIL/
      individual (ca_nhan_nuocngoai_net) — two structurally different
      categories already distinguished in the raw data.
  (b) by SECTOR — different industries likely attract different kinds of
      foreign capital (patient long-only funds vs hot money/ETF flow vs
      short-term arb), so the same "foreign accumulation" signal could be
      informative in one sector and noise (or actively wrong) in another.

For each (investor type x sector), computes the Spearman IC of trailing
60-day foreign net accumulation vs forward return, at several horizons.
Outputs the raw numbers (also saved as CSV for the heatmap) plus prints a
text-grid preview.

Usage:  python archive/foreign_segment_heatmap.py
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
ACCUM_WINDOW = 60
FORWARD_HORIZONS = [20, 60, 120]   # keep to horizons with decent sample size
MIN_TICKERS_PER_SECTOR = 4          # skip sectors too thin to be meaningful


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
    if "close" not in df.columns or len(df) < ACCUM_WINDOW + max(FORWARD_HORIZONS) + 5:
        return pd.DataFrame()

    inst   = df.get("to_chuc_nuocngoai_net", pd.Series(0, index=df.index)).fillna(0)
    retail = df.get("ca_nhan_nuocngoai_net", pd.Series(0, index=df.index)).fillna(0)

    out = pd.DataFrame({"date": df["date"], "ticker": ticker, "sector": sector})
    for label, series in [("inst", inst), ("retail", retail)]:
        cum   = series.rolling(ACCUM_WINDOW).sum()
        scale = series.abs().rolling(ACCUM_WINDOW).mean() * ACCUM_WINDOW
        out[f"accum_{label}"] = cum / scale.replace(0, np.nan)
    for h in FORWARD_HORIZONS:
        out[f"fwd_{h}"] = df["close"].shift(-h) / df["close"] - 1
    return out


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
    print(f"  {len(universe)} liquid tickers with flow data")

    frames = []
    for t in universe:
        sec = ticker_to_sector.get(t)
        if sec is None or sec == "Unknown":
            continue
        f = build_ticker_frame(t, sec)
        if not f.empty:
            frames.append(f)
    all_df = pd.concat(frames, ignore_index=True)
    print(f"  Built {len(all_df):,} ticker-day rows, {all_df['ticker'].nunique()} tickers, "
          f"{all_df['sector'].nunique()} sectors")

    sector_counts = all_df.groupby("sector")["ticker"].nunique()
    sectors = sorted(sector_counts[sector_counts >= MIN_TICKERS_PER_SECTOR].index)
    print(f"  {len(sectors)} sectors with >= {MIN_TICKERS_PER_SECTOR} tickers: {sectors}")

    results = []  # investor_type, sector, horizon, ic, n
    for label in ["inst", "retail"]:
        for sector in sectors:
            sub_sector = all_df[all_df["sector"] == sector]
            for h in FORWARD_HORIZONS:
                sub = sub_sector.dropna(subset=[f"accum_{label}", f"fwd_{h}"])
                if len(sub) < 50:
                    ic = np.nan
                else:
                    ic = sub[f"accum_{label}"].corr(sub[f"fwd_{h}"], method="spearman")
                results.append({"investor_type": label, "sector": sector,
                                 "horizon": h, "ic": ic, "n": len(sub)})

    res_df = pd.DataFrame(results)
    out_csv = os.path.join(BASE, "results", "foreign_segment_heatmap.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    for label, title in [("inst", "FOREIGN INSTITUTIONAL"), ("retail", "FOREIGN RETAIL/INDIVIDUAL")]:
        print(f"\n{'='*90}")
        print(f"  {title} — 60d accumulation IC vs forward return, by sector x horizon")
        print(f"{'-'*90}")
        pivot = res_df[res_df["investor_type"] == label].pivot(index="sector", columns="horizon", values="ic")
        pivot = pivot.sort_values(FORWARD_HORIZONS[-1])
        print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
