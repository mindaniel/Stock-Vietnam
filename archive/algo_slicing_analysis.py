"""
algo_slicing_analysis.py — Are the "continuous small orders" visible in the
tape a real execution algorithm, and does their presence predict the next few
days' price move?

Two things get confused when eyeballing a tape:

  (1) BOOK SWEEP — one large order consuming many resting orders. Prints as
      many small ticks at the SAME timestamp and SAME price. Looks algorithmic,
      is not: it is a single decision, fragmented by the matching engine.
  (2) TRUE SLICING — one participant working a large parent order over time
      (VWAP/TWAP/iceberg). Prints as the SAME clip size repeated at DISTINCT
      timestamps spread across the session.

This script collapses (1) away first (grouping same-second/same-price/same-side
ticks into a single logical order), then measures (2) on what remains, and
tests whether slicing intensity predicts forward returns.

Slicing metrics per (ticker, day), computed on collapsed orders:
  slice_share    — share of the day's traded volume executed in the single most
                   repeated clip size (>=5 occurrences at distinct timestamps)
  slice_regularity — 1 - normalized std of inter-arrival gaps for that clip
                   (1.0 = perfectly evenly spaced = machine-like)
  slice_side     — net buy/sell direction of the sliced volume

Then: forward 1/3/5/10-day returns, sliced-buy days vs sliced-sell days vs
neither, with a t-test across TICKERS (one number per ticker) rather than
across days, so serial correlation within a ticker does not fake significance.

Usage:  python archive/algo_slicing_analysis.py
"""

import glob, os, sys
import numpy as np
import pandas as pd
from scipy import stats

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICK_DIR = os.path.join(BASE, "data", "tick_data")
PRICE_DIR = os.path.join(BASE, "data", "price")

MIN_CLIP_REPEATS = 5      # a clip size must repeat this often to count as sliced
FWD_HORIZONS = [1, 3, 5, 10]


def collapse_sweeps(day: pd.DataFrame) -> pd.DataFrame:
    """Collapse same-timestamp/same-price/same-side ticks into ONE logical
    order. This removes book-sweep fragmentation, which otherwise masquerades
    as algorithmic small-order flow."""
    g = day.groupby(["t", "p", "s"], as_index=False).agg(v=("v", "sum"),
                                                         n_fills=("v", "size"))
    return g


def slicing_metrics(orders: pd.DataFrame) -> dict:
    """Find the most-repeated clip size among logical orders and describe how
    machine-like its arrival pattern is."""
    out = {"slice_share": 0.0, "slice_regularity": np.nan,
           "slice_net": 0.0, "clip_size": np.nan, "clip_count": 0}
    total_vol = orders["v"].sum()
    if total_vol <= 0 or len(orders) < MIN_CLIP_REPEATS:
        return out

    counts = orders["v"].value_counts()
    counts = counts[counts >= MIN_CLIP_REPEATS]
    if counts.empty:
        return out

    # rank candidate clips by volume footprint, not raw count, so a spam of
    # 100-lot odd fills does not outrank a genuine worked order
    footprint = (counts.index.to_series().values * counts.values)
    clip = counts.index[int(np.argmax(footprint))]

    sub = orders[orders["v"] == clip].sort_values("t")
    secs = pd.to_timedelta(sub["t"]).dt.total_seconds().values
    if len(secs) < MIN_CLIP_REPEATS:
        return out

    gaps = np.diff(secs)
    if len(gaps) >= 2 and gaps.mean() > 0:
        # low relative dispersion of gaps => evenly spaced => machine-like
        out["slice_regularity"] = float(1.0 / (1.0 + gaps.std() / gaps.mean()))

    sliced_vol = sub["v"].sum()
    buy_vol = sub.loc[sub["s"] == "buy", "v"].sum()
    sell_vol = sub.loc[sub["s"] == "sell", "v"].sum()

    out["slice_share"] = float(sliced_vol / total_vol)
    out["slice_net"] = float((buy_vol - sell_vol) / max(sliced_vol, 1))
    out["clip_size"] = float(clip)
    out["clip_count"] = int(len(sub))
    return out


def load_prices(ticker: str) -> pd.DataFrame | None:
    p = os.path.join(PRICE_DIR, f"{ticker}.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "time" if "time" in df.columns else "date"
    df["date"] = pd.to_datetime(df[date_col])
    df = df.sort_values("date").reset_index(drop=True)
    if "close" not in df.columns:
        return None
    for h in FWD_HORIZONS:
        df[f"fwd{h}"] = df["close"].shift(-h) / df["close"] - 1
    return df[["date"] + [f"fwd{h}" for h in FWD_HORIZONS]]


def main():
    files = sorted(glob.glob(os.path.join(TICK_DIR, "*.parquet")))
    print(f"Scanning {len(files)} tickers of tick data...")

    rows = []
    sweep_stats = []
    for i, fp in enumerate(files):
        ticker = os.path.splitext(os.path.basename(fp))[0].upper()
        try:
            tk = pd.read_parquet(fp)
        except Exception:
            continue
        if not {"td", "t", "p", "v", "s"}.issubset(tk.columns):
            continue

        prices = load_prices(ticker)
        if prices is None:
            continue

        for td, day in tk.groupby("td"):
            if len(day) < 20:
                continue
            orders = collapse_sweeps(day)
            sweep_stats.append({
                "raw_ticks": len(day),
                "logical_orders": len(orders),
                "max_fills_one_order": int(orders["n_fills"].max()),
            })
            m = slicing_metrics(orders)
            m["ticker"] = ticker
            m["date"] = pd.to_datetime(td, format="%d/%m/%Y")
            m["day_vol"] = float(day["v"].sum())
            rows.append(m)

        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(files)}")

    df = pd.DataFrame(rows)
    sw = pd.DataFrame(sweep_stats)

    print(f"\n{'='*88}")
    print("  STEP 0 — HOW MUCH OF THE 'MANY SMALL ORDERS' IS JUST BOOK SWEEP?")
    print(f"{'-'*88}")
    print(f"  raw ticks per day (median):        {sw['raw_ticks'].median():,.0f}")
    print(f"  logical orders per day (median):   {sw['logical_orders'].median():,.0f}")
    print(f"  fragmentation ratio:               {sw['raw_ticks'].sum()/sw['logical_orders'].sum():.2f}x")
    print(f"  largest single order's fill count: {sw['max_fills_one_order'].max():,}")
    print("  (ratio >1 means the tape shows more 'orders' than actually existed)")

    # attach forward returns
    out = []
    for ticker, g in df.groupby("ticker"):
        prices = load_prices(ticker)
        if prices is None:
            continue
        out.append(g.merge(prices, on="date", how="left"))
    df = pd.concat(out, ignore_index=True)

    sliced = df[(df["slice_share"] > 0) & df["clip_size"].notna()]
    print(f"\n{'='*88}")
    print("  STEP 1 — SLICING PREVALENCE (after removing sweep fragmentation)")
    print(f"{'-'*88}")
    print(f"  ticker-days total:            {len(df):,}")
    print(f"  with a repeated clip (>={MIN_CLIP_REPEATS}):  {len(sliced):,} ({len(sliced)/max(len(df),1):.1%})")
    if len(sliced):
        print(f"  median slice_share of volume: {sliced['slice_share'].median():.1%}")
        print(f"  median clip size:             {sliced['clip_size'].median():,.0f}")
        print(f"  median regularity (1=even):   {sliced['slice_regularity'].median():.3f}")
        print(f"  most common clip sizes:       {sliced['clip_size'].value_counts().head(5).to_dict()}")

    # Direction test: does sliced BUYING predict forward returns?
    print(f"\n{'='*88}")
    print("  STEP 2 — DOES SLICED FLOW PREDICT THE NEXT FEW DAYS?")
    print("  (one mean per ticker, then t-test across tickers — no day-level pseudo-N)")
    print(f"{'-'*88}")
    strong = sliced[sliced["slice_share"] >= sliced["slice_share"].median()]
    buy_days = strong[strong["slice_net"] > 0.3]
    sell_days = strong[strong["slice_net"] < -0.3]
    print(f"  sliced-BUY ticker-days:  {len(buy_days):,}")
    print(f"  sliced-SELL ticker-days: {len(sell_days):,}")
    print()
    print(f"  {'horizon':<10} {'buy mean':>10} {'sell mean':>11} {'diff':>9} "
          f"{'t':>7} {'p':>8} {'n_tick':>7}")
    for h in FWD_HORIZONS:
        col = f"fwd{h}"
        b = buy_days.groupby("ticker")[col].mean().dropna()
        s = sell_days.groupby("ticker")[col].mean().dropna()
        common = b.index.intersection(s.index)
        if len(common) < 10:
            print(f"  fwd{h:<7} insufficient paired tickers ({len(common)})")
            continue
        diff = b[common] - s[common]
        t, p = stats.ttest_1samp(diff, 0)
        print(f"  fwd{h:<7} {b[common].mean():>+9.2%} {s[common].mean():>+10.2%} "
              f"{diff.mean():>+8.2%} {t:>7.2f} {p:>8.3f} {len(common):>7}")

    print(f"\n{'='*88}")
    print("  STEP 3 — DOES *REGULARITY* (machine-like evenness) ADD ANYTHING?")
    print(f"{'-'*88}")
    reg = sliced.dropna(subset=["slice_regularity"])
    if len(reg) > 100:
        hi = reg[reg["slice_regularity"] >= reg["slice_regularity"].quantile(0.75)]
        lo = reg[reg["slice_regularity"] <= reg["slice_regularity"].quantile(0.25)]
        print(f"  {'horizon':<10} {'high-reg':>10} {'low-reg':>10} {'diff':>9} {'t':>7} {'p':>8}")
        for h in FWD_HORIZONS:
            col = f"fwd{h}"
            a = hi.groupby("ticker")[col].mean().dropna()
            b = lo.groupby("ticker")[col].mean().dropna()
            common = a.index.intersection(b.index)
            if len(common) < 10:
                continue
            d = a[common] - b[common]
            t, p = stats.ttest_1samp(d, 0)
            print(f"  fwd{h:<7} {a[common].mean():>+9.2%} {b[common].mean():>+9.2%} "
                  f"{d.mean():>+8.2%} {t:>7.2f} {p:>8.3f}")

    outp = os.path.join(BASE, "backtest_reports", "algo_slicing_daily.csv")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    df.to_csv(outp, index=False)
    print(f"\nSaved {len(df):,} ticker-days to {outp}")


if __name__ == "__main__":
    main()
