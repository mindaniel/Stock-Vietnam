"""
backtest_fundamentals.py
------------------------
Backtests fundamental filters on strategy-sector stocks.

Data source : data/financials_fa/{SYMBOL}.parquet  (FireAnt annual, quarter=0)
              Ratios already computed: roe, roa, net_margin, gross_margin,
              op_margin, debt_equity, ocf_to_netprofit, free_cf

Point-in-time rule (no look-ahead):
  Annual report for year Y is available after April 30, Y+1.
  On any trade date D we only use data from year Y where April 30, Y+1 <= D.
    e.g.  Jan 2024  → use year 2022 report
          May 2024  → use year 2023 report

Method:
  For each (stock, report year) with annual data:
    1. Entry  = first trading day on/after April 30, year+1
    2. Exit   = first trading day on/after entry + FORWARD_DAYS calendar days
    3. Fwd return = (exit_close - entry_open) / entry_open
    4. Liquidity gate: skip if 60d avg daily value < MIN_LIQ
  Test each filter → compare mean/median return and win rate vs baseline.

Usage:
  python backtest_fundamentals.py                    # all 4 sectors, 60d window
  python backtest_fundamentals.py --window 30
  python backtest_fundamentals.py --window 90
  python backtest_fundamentals.py --sector Banks
  python backtest_fundamentals.py --start 2018      # use data from 2018+
"""

import argparse
import glob
import os
import sys
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "price")
FA_DIR   = os.path.join(DATA_DIR, "financials_fa")
TICK_CSV = os.path.join(BASE_DIR, "ticker_sectors.csv")

STRATEGY_SECTORS = {"Banks", "Food & Beverage", "Basic Resources", "Real Estate"}
MIN_LIQ          = 1_000_000   # avg daily value in CSV units = 1B actual VND

# Annual report publication deadline in Vietnam: April 30, Y+1
ANNUAL_LAG_MONTH = 4
ANNUAL_LAG_DAY   = 30


def annual_avail(year: int) -> date:
    return date(year + 1, ANNUAL_LAG_MONTH, ANNUAL_LAG_DAY)


# ── Load data ─────────────────────────────────────────────────────────────────

def load_financials(sector_filter=None) -> pd.DataFrame:
    tickers = pd.read_csv(TICK_CSV)
    tickers = tickers[tickers["industry"].isin(STRATEGY_SECTORS)].copy()
    if sector_filter:
        tickers = tickers[tickers["industry"] == sector_filter]
    valid   = set(tickers["ticker"])

    dfs = []
    for f in glob.glob(os.path.join(FA_DIR, "*.parquet")):
        sym = os.path.basename(f).replace(".parquet", "")
        if sym == "indicators_snapshot" or sym not in valid:
            continue
        try:
            df = pd.read_parquet(f)
            df = df[df["quarter"] == 0].copy()   # annual only
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    fin = pd.concat(dfs, ignore_index=True)
    fin["avail_date"] = fin["year"].apply(lambda y: pd.Timestamp(annual_avail(int(y))))

    # Normalise sector column — new parquets use "sector", old ones use "industry"
    if "sector" not in fin.columns:
        if "industry" in fin.columns:
            fin = fin.rename(columns={"industry": "sector"})
        else:
            # Fall back: attach from ticker_sectors.csv
            fin = fin.merge(
                tickers[["ticker", "industry"]].rename(columns={"industry": "sector"}),
                left_on="symbol", right_on="ticker", how="left"
            )
    # After the new scraper the merge is unnecessary, but apply sector_filter if requested
    if sector_filter and "sector" in fin.columns:
        fin = fin[fin["sector"] == sector_filter].copy()
    return fin


def load_price_cache(symbols: list) -> dict:
    cache = {}
    for sym in symbols:
        path = os.path.join(DATA_DIR, f"{sym}.parquet")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path)
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.rename(columns={"time": "date"}).sort_values("date").set_index("date")
            df["liq60"] = df["value"].rolling(60, min_periods=20).mean()
            cache[sym] = df
        except Exception:
            pass
    return cache


# ── Build factor dataset ──────────────────────────────────────────────────────

def build_factor_df(fin: pd.DataFrame, price_cache: dict,
                    fwd_days: int, start_year: int) -> pd.DataFrame:
    fin = fin[fin["year"] >= start_year].copy()
    records = []

    for _, row in fin.iterrows():
        sym    = row["symbol"]
        prices = price_cache.get(sym)
        if prices is None:
            continue

        avail = row["avail_date"]

        # Entry: first open on/after avail_date
        future = prices[prices.index >= avail]
        if len(future) < 5:
            continue

        entry_date  = future.index[0]
        entry_price = future["open"].iloc[0]
        if not entry_price or entry_price <= 0:
            continue

        # Liquidity gate
        liq = future["liq60"].iloc[0]
        if pd.isna(liq) or liq < MIN_LIQ:
            continue

        # Exit: first close >= entry + fwd_days calendar days
        exit_floor  = entry_date + timedelta(days=fwd_days)
        future_exit = prices[prices.index >= exit_floor]
        if future_exit.empty:
            continue
        exit_price = future_exit["close"].iloc[0]
        if not exit_price or exit_price <= 0:
            continue

        fwd_ret = (exit_price - entry_price) / entry_price

        records.append({
            "symbol":           sym,
            "sector":           row["sector"],
            "year":             int(row["year"]),
            "entry_date":       entry_date,
            # Profitability
            "roe":              row.get("roe"),
            "roa":              row.get("roa"),
            "net_margin":       row.get("net_margin"),
            "gross_margin":     row.get("gross_margin"),
            "op_margin":        row.get("op_margin"),
            # Leverage / safety
            "debt_equity":      row.get("debt_equity"),
            # Cash quality
            "ocf_quality":      row.get("ocf_to_netprofit"),
            # Growth (decimal, e.g. 0.19 = 19%)
            "revenue_growth_lfy": row.get("revenue_growth_lfy"),
            "revenue_growth_3yr": row.get("revenue_growth_3yr"),
            "profit_growth_lfy":  row.get("profit_growth_lfy"),
            "profit_growth_3yr":  row.get("profit_growth_3yr"),
            # Valuation
            "pe":               row.get("pe"),
            "pb":               row.get("pb"),
            # Return
            "fwd_ret":          fwd_ret,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Compute sector-relative metrics (vs median of same sector+year)
    for col in ["roe", "roa", "net_margin", "debt_equity"]:
        med = df.groupby(["sector", "year"])[col].transform("median")
        df[f"{col}_vs_sector"] = df[col] - med

    # Sector-relative growth (above-median revenue/profit growth)
    for col in ["revenue_growth_lfy", "profit_growth_lfy"]:
        if col in df.columns and df[col].notna().any():
            med = df.groupby(["sector", "year"])[col].transform("median")
            df[f"{col}_vs_sector"] = df[col] - med

    return df


# ── Filter definitions ────────────────────────────────────────────────────────

def define_filters():
    return [
        # Baseline
        {"name": "No filter (baseline)",          "fn": lambda d: d},

        # Profitability — absolute
        {"name": "ROE > 0%",                      "fn": lambda d: d[d["roe"] > 0]},
        {"name": "ROE > 5%",                      "fn": lambda d: d[d["roe"] > 5]},
        {"name": "ROE > 10%",                     "fn": lambda d: d[d["roe"] > 10]},
        {"name": "ROE > 15%",                     "fn": lambda d: d[d["roe"] > 15]},
        {"name": "ROE > 20%",                     "fn": lambda d: d[d["roe"] > 20]},
        {"name": "ROA > 3%",                      "fn": lambda d: d[d["roa"] > 3]},
        {"name": "ROA > 5%",                      "fn": lambda d: d[d["roa"] > 5]},
        {"name": "Net margin > 5%",               "fn": lambda d: d[d["net_margin"] > 5]},
        {"name": "Net margin > 10%",              "fn": lambda d: d[d["net_margin"] > 10]},

        # Profitability — sector-relative (beats sector median)
        {"name": "ROE > sector median",           "fn": lambda d: d[d["roe_vs_sector"] > 0]},
        {"name": "ROE > sector median + 5pp",     "fn": lambda d: d[d["roe_vs_sector"] > 5]},
        {"name": "ROA > sector median",           "fn": lambda d: d[d["roa_vs_sector"] > 0]},

        # Leverage / safety
        {"name": "D/E < 1",                       "fn": lambda d: d[d["debt_equity"] < 1]},
        {"name": "D/E < 2",                       "fn": lambda d: d[d["debt_equity"] < 2]},
        {"name": "D/E < 3",                       "fn": lambda d: d[d["debt_equity"] < 3]},

        # Cash quality: OCF > net profit (earnings are backed by real cash)
        {"name": "OCF/profit > 0  (positive OCF)","fn": lambda d: d[d["ocf_quality"] > 0]},
        {"name": "OCF/profit > 0.5",              "fn": lambda d: d[d["ocf_quality"] > 0.5]},
        {"name": "OCF/profit > 1  (OCF > profit)","fn": lambda d: d[d["ocf_quality"] > 1]},

        # Growth — revenue (LFY YoY, as decimal)
        {"name": "Revenue growth > 0%",          "fn": lambda d: d[d["revenue_growth_lfy"] > 0]},
        {"name": "Revenue growth > 10%",         "fn": lambda d: d[d["revenue_growth_lfy"] > 0.10]},
        {"name": "Revenue growth > 20%",         "fn": lambda d: d[d["revenue_growth_lfy"] > 0.20]},

        # Growth — profit (LFY YoY, as decimal)
        {"name": "Profit growth > 0%",           "fn": lambda d: d[d["profit_growth_lfy"] > 0]},
        {"name": "Profit growth > 10%",          "fn": lambda d: d[d["profit_growth_lfy"] > 0.10]},
        {"name": "Profit growth > 20%",          "fn": lambda d: d[d["profit_growth_lfy"] > 0.20]},

        # Growth — 3-year CAGR
        {"name": "Revenue CAGR 3yr > 10%",       "fn": lambda d: d[d["revenue_growth_3yr"] > 0.10]},
        {"name": "Profit CAGR 3yr > 10%",        "fn": lambda d: d[d["profit_growth_3yr"] > 0.10]},
        {"name": "Profit CAGR 3yr > 15%",        "fn": lambda d: d[d["profit_growth_3yr"] > 0.15]},

        # Valuation — PE / PB
        {"name": "PE < 15",                      "fn": lambda d: d[d["pe"] < 15]},
        {"name": "PE < 10",                      "fn": lambda d: d[d["pe"] < 10]},
        {"name": "PB < 2",                       "fn": lambda d: d[d["pb"] < 2]},
        {"name": "PB < 1",                       "fn": lambda d: d[d["pb"] < 1]},

        # Combos
        {"name": "ROE>10% & D/E<2",
         "fn": lambda d: d[(d["roe"] > 10) & (d["debt_equity"] < 2)]},
        {"name": "ROE>10% & OCF>0",
         "fn": lambda d: d[(d["roe"] > 10) & (d["ocf_quality"] > 0)]},
        {"name": "ROE>10% & OCF>0.5 & D/E<2",
         "fn": lambda d: d[(d["roe"] > 10) & (d["ocf_quality"] > 0.5) & (d["debt_equity"] < 2)]},
        {"name": "Above-sector ROE & OCF>0",
         "fn": lambda d: d[(d["roe_vs_sector"] > 0) & (d["ocf_quality"] > 0)]},
        {"name": "Above-sector ROE & D/E<2",
         "fn": lambda d: d[(d["roe_vs_sector"] > 0) & (d["debt_equity"] < 2)]},
        {"name": "ROE>10% & profit growth>10%",
         "fn": lambda d: d[(d["roe"] > 10) & (d["profit_growth_lfy"] > 0.10)]},
        {"name": "OCF>1 & profit growth>0%",
         "fn": lambda d: d[(d["ocf_quality"] > 1) & (d["profit_growth_lfy"] > 0)]},
        {"name": "PE<15 & ROE>10%",
         "fn": lambda d: d[(d["pe"] < 15) & (d["roe"] > 10)]},
        {"name": "PE<15 & profit growth>10%",
         "fn": lambda d: d[(d["pe"] < 15) & (d["profit_growth_lfy"] > 0.10)]},
    ]


# ── Statistics ────────────────────────────────────────────────────────────────

def stats(df: pd.DataFrame, base_n: int) -> dict:
    n    = len(df)
    rets = df["fwd_ret"].dropna()
    if len(rets) == 0:
        return dict(n=0, cov=0, mean=np.nan, median=np.nan,
                    win=np.nan, p25=np.nan, p75=np.nan)
    return dict(
        n      = n,
        cov    = n / base_n if base_n else 0,
        mean   = rets.mean(),
        median = rets.median(),
        win    = (rets > 0).mean(),
        p25    = rets.quantile(0.25),
        p75    = rets.quantile(0.75),
    )


# ── Print tables ──────────────────────────────────────────────────────────────

def print_filter_table(factor_df: pd.DataFrame, fwd_days: int, sector: str | None):
    scope = sector or "All 4 strategy sectors"
    print(f"\n{'='*95}")
    print(f"  BACKTEST — {scope}  |  {fwd_days}-day forward return")
    print(f"  Stocks: {factor_df['symbol'].nunique()}  |  "
          f"Observations: {len(factor_df):,}  |  "
          f"Period: {int(factor_df['year'].min())}–{int(factor_df['year'].max())}")
    print(f"{'='*95}")

    filters    = define_filters()
    base_df    = filters[0]["fn"](factor_df)
    base_n     = len(base_df)
    base_st    = stats(base_df, base_n)

    hdr = (f"{'Filter':<38} {'N':>5} {'Cov':>5} {'Mean':>7} "
           f"{'Median':>7} {'Win%':>6} {'P25':>7} {'P75':>7} {'vs Base':>9}")
    print(hdr)
    print("─" * 95)

    for f in filters:
        sub = f["fn"](factor_df)
        st  = stats(sub, base_n)
        if st["n"] < 10:
            continue
        delta = st["mean"] - base_st["mean"]
        flag  = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else " ")
        print(
            f"{f['name']:<38} "
            f"{st['n']:>5,} "
            f"{st['cov']:>5.0%} "
            f"{st['mean']:>7.1%} "
            f"{st['median']:>7.1%} "
            f"{st['win']:>6.0%} "
            f"{st['p25']:>7.1%} "
            f"{st['p75']:>7.1%} "
            f"  {flag}{abs(delta):>5.1%}"
        )

    print("─" * 95)
    print("  ▲ = beats baseline  ▼ = worse  (threshold ±0.5pp)")


def print_sector_breakdown(factor_df: pd.DataFrame, filt_fn, filt_name: str, fwd_days: int):
    sub = filt_fn(factor_df)
    print(f"\n{'─'*70}")
    print(f"  Sector breakdown — [{filt_name}]  ({fwd_days}d window)")
    print(f"{'─'*70}")
    hdr = f"{'Sector':<22} {'N':>5} {'Cov':>5} {'Mean':>7} {'Median':>7} {'Win%':>6}"
    print(hdr)
    print("─" * 55)
    for sec in sorted(factor_df["sector"].dropna().unique()):
        base_n = len(factor_df[factor_df["sector"] == sec])
        grp    = sub[sub["sector"] == sec]
        st     = stats(grp, base_n)
        if st["n"] < 5:
            continue
        print(
            f"{sec:<22} "
            f"{st['n']:>5,} "
            f"{st['cov']:>5.0%} "
            f"{st['mean']:>7.1%} "
            f"{st['median']:>7.1%} "
            f"{st['win']:>6.0%}"
        )


def print_current_snapshot(fin: pd.DataFrame):
    """Show latest-year fundamentals by sector."""
    latest = fin.sort_values(["symbol", "year"]).groupby("symbol").last().reset_index()

    print(f"\n{'='*70}")
    print("  CURRENT FUNDAMENTAL SNAPSHOT  (latest annual report per stock)")
    print(f"{'='*70}")

    for sec in sorted(STRATEGY_SECTORS):
        grp = latest[latest["sector"] == sec]
        if grp.empty:
            continue
        n = len(grp)
        print(f"\n  ── {sec}  ({n} stocks) ──")
        print(f"  {'Metric':<18} {'P25':>8} {'Median':>8} {'P75':>8} {'% valid':>8}")
        print(f"  {'─'*50}")
        for col, label in [
            ("roe",                  "ROE %"),
            ("roa",                  "ROA %"),
            ("net_margin",           "Net margin %"),
            ("debt_equity",          "D/E ratio"),
            ("ocf_to_netprofit",     "OCF/profit"),
            ("pe",                   "P/E"),
            ("pb",                   "P/B"),
            ("revenue_growth_lfy",   "Rev growth LFY"),
            ("profit_growth_lfy",    "Profit growth LFY"),
            ("profit_growth_3yr",    "Profit CAGR 3yr"),
        ]:
            if col not in grp.columns:
                continue
            s = grp[col].replace([np.inf, -np.inf], np.nan).dropna()
            # Winsorise for display
            lo, hi = s.quantile(0.02), s.quantile(0.98)
            sw = s[(s >= lo) & (s <= hi)]
            if len(sw) < 3:
                continue
            print(
                f"  {label:<18} "
                f"{sw.quantile(.25):>8.1f} "
                f"{sw.median():>8.1f} "
                f"{sw.quantile(.75):>8.1f} "
                f"{len(s)/n:>8.0%}"
            )
        n_roe_pos = (grp["roe"] > 0).sum()
        n_ocf_pos = (grp["ocf_to_netprofit"] > 0).sum() if "ocf_to_netprofit" in grp.columns else 0
        print(f"  Positive ROE: {n_roe_pos}/{n}  |  Positive OCF: {n_ocf_pos}/{n}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sector", default=None, help="Limit to one sector, e.g. 'Banks'")
    parser.add_argument("--window", type=int, default=60, help="Forward return window in calendar days")
    parser.add_argument("--start",  type=int, default=2017, help="Earliest report year to use")
    args = parser.parse_args()

    print("Loading financial data (FireAnt annual)...")
    fin = load_financials(sector_filter=args.sector)
    if fin.empty:
        print("No data in data/financials_fa/ — run fetch_financials_fireant.py first.")
        return
    print(f"  {fin['symbol'].nunique()} stocks, "
          f"{int(fin['year'].min())}–{int(fin['year'].max())}, "
          f"{len(fin):,} annual rows")

    symbols = fin["symbol"].unique().tolist()
    print(f"Loading price data for {len(symbols)} symbols...")
    price_cache = load_price_cache(symbols)

    print(f"Building factor dataset (from {args.start}, {args.window}d window)...")
    factor_df = build_factor_df(fin, price_cache, args.window, args.start)

    if factor_df.empty:
        print("No observations — check that price CSVs go back to the report years.")
        return

    # Show snapshot
    print_current_snapshot(fin)

    # Overall filter table
    print_filter_table(factor_df, args.window, args.sector)

    # Per-sector breakdown for two best candidate filters
    for fn, name in [
        (lambda d: d[d["roe"] > 10],                              "ROE > 10%"),
        (lambda d: d[(d["roe"] > 10) & (d["ocf_quality"] > 0)],  "ROE>10% & OCF>0"),
        (lambda d: d[d["roe_vs_sector"] > 0],                     "ROE above sector median"),
    ]:
        print_sector_breakdown(factor_df, fn, name, args.window)

    # Guidance
    print(f"\n{'='*70}")
    print("  HOW TO READ")
    print(f"{'='*70}")
    print("""
  Mean / Median  — average forward return for stocks passing the filter.
  Win%           — % of observations that ended positive.
  Coverage       — fraction of the universe kept.
  vs Base        — improvement over no-filter baseline.  ▲ = better.

  Point-in-time: annual report for year Y used only after April 30, Y+1.
  Liquidity gate: 60d avg daily value > 1B VND (skips tiny stocks).

  Sector-relative filters ("above sector median") are often more robust
  than absolute thresholds because they adapt to each sector's norms
  (banks have structurally lower ROA than consumer goods).

  Next: python backtest_fundamentals.py --sector Banks
        python backtest_fundamentals.py --sector "Real Estate"
        python backtest_fundamentals.py --window 30
""")


if __name__ == "__main__":
    main()
