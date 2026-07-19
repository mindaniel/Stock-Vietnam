"""
putthrough_block_test.py — Do genuine put-through BLOCK deals predict forward
returns?

Put-through file has three distinct populations (established earlier):
  vol 1-3   machine noise, prints at exact price band, no economic content
  vol 99    real odd lots (lô lẻ), priced at market, blue chips
  vol >=20k genuine negotiated block deals   <-- THIS is what we test

Block deals are the interesting case: a negotiated transfer of a large stake
at an agreed price. Two competing readings —
  (a) informed: someone with a view is taking size off-market
  (b) liquidity/exit: an insider or fund is unloading, price falls after

The discount/premium of the block price vs the day's close is the natural
discriminator. Deals struck BELOW market suggest a seller paying up for
immediacy (exit); deals ABOVE market suggest a buyer paying up for size
(accumulation).

DATA GUARDS (both found the hard way earlier in this session):
  1. The scraper stamps the previous session's rows onto NON-TRADING days
     (2026-04-30, 05-01 are holidays carrying 04-29's exact contents).
     Guard: keep only dates present in the price parquet's calendar.
  2. Tiny-volume rows are system noise, not trades. Guard: vol >= MIN_BLOCK.

INFERENCE GUARD: the sample is ~85 calendar days and every symbol shares the
same dates, so symbol-level t-tests would treat one market-wide move as N
independent observations. We therefore report BOTH:
  - a per-symbol t-test (naive, for comparison), and
  - a DATE-BLOCK bootstrap that resamples whole trading days, which is the
    honest inference given the overlapping-window / shared-calendar problem.

Usage:  python archive/putthrough_block_test.py
"""

import os, sys
import numpy as np
import pandas as pd
from scipy import stats

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PT_PATH = os.path.join(BASE, "putthrough", "putthrough_hose_all.csv")
PRICE_DIR = os.path.join(BASE, "data", "price")

MIN_BLOCK = 20_000        # HOSE put-through minimum lot
FWD = [1, 3, 5, 10]
N_BOOT = 2000
RNG = np.random.default_rng(7)


def load_price(sym: str):
    p = os.path.join(PRICE_DIR, f"{sym}.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    df.columns = [c.strip().lower() for c in df.columns]
    dc = "time" if "time" in df.columns else "date"
    df["date"] = pd.to_datetime(df[dc])
    df = df.sort_values("date").reset_index(drop=True)
    if "close" not in df.columns:
        return None
    # Bad-data guard: some parquets carry rows with close==0 / open==NaN
    # (e.g. ASG has 55). These make every return through them +/-inf, which is
    # the source of the recurring "+inf%" results seen across this project.
    df = df[df["close"] > 0].reset_index(drop=True)
    if df.empty:
        return None
    df["prev"] = df["close"].shift(1)
    for h in FWD:
        df[f"f{h}"] = df["close"].shift(-h) / df["close"] - 1
    keep = ["date", "close", "prev", "volume"] if "volume" in df.columns else ["date", "close", "prev"]
    return df[keep + [f"f{h}" for h in FWD]]


def block_bootstrap(df: pd.DataFrame, col: str) -> tuple:
    """Resample whole DATES with replacement. Preserves the cross-sectional
    correlation among all symbols sharing a trading day, which a per-symbol
    t-test wrongly discards."""
    dates = df["date"].unique()
    obs = df.groupby("date")[col].mean()
    point = df[col].mean()
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = RNG.choice(dates, size=len(dates), replace=True)
        boots[b] = obs.reindex(pick).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided bootstrap p for H0: mean == 0
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return point, lo, hi, max(p, 1.0 / N_BOOT)


def main():
    pt = pd.read_csv(PT_PATH)
    pt["date"] = pd.to_datetime(pt["date"])
    print(f"raw put-through rows: {len(pt):,}  symbols={pt['symbol'].nunique()}  days={pt['date'].nunique()}")

    # Market trading calendar: a date is a real session if MANY symbols have a
    # valid bar on it. Per-symbol date sets are not usable for this — a single
    # symbol can be halted or newly listed, which would wrongly brand a real
    # trading day as phantom.
    price_cache, day_counts = {}, {}
    for sym in pt["symbol"].unique():
        px = load_price(sym)
        if px is None:
            continue
        price_cache[sym] = px
        for dt in px["date"].values:
            day_counts[dt] = day_counts.get(dt, 0) + 1
    if not day_counts:
        print("no usable price data"); return
    busiest = max(day_counts.values())
    market_days = {d for d, c in day_counts.items() if c >= 0.5 * busiest}

    frames = []
    dropped_dates = set(pt["date"].unique()) - market_days
    for sym, g in pt.groupby("symbol"):
        px = price_cache.get(sym)
        if px is None:
            continue
        # GUARD 1 — drop phantom non-trading days (scraper repeats the prior
        # session's rows onto market holidays)
        g = g[g["date"].isin(market_days) & g["date"].isin(set(px["date"]))]
        # GUARD 2 — real blocks only
        g = g[g["volume"] >= MIN_BLOCK]
        if g.empty:
            continue
        # aggregate to symbol-day: total block volume + volume-weighted price
        agg = g.groupby("date").apply(
            lambda x: pd.Series({
                "blk_vol": x["volume"].sum(),
                "blk_vwap": np.average(x["price"], weights=x["volume"]),
                "n_deals": len(x),
            }), include_groups=False
        ).reset_index()
        agg["symbol"] = sym
        frames.append(agg.merge(px, on="date", how="left"))

    df = pd.concat(frames, ignore_index=True).dropna(subset=["close", "prev"])
    print(f"phantom non-trading dates dropped: {sorted(str(d.date()) for d in dropped_dates)}")
    print(f"block-deal symbol-days: {len(df):,}  symbols={df['symbol'].nunique()}  days={df['date'].nunique()}")

    # block price vs market close that day -> discount (neg) or premium (pos)
    df["prem"] = df["blk_vwap"] / df["close"] - 1
    # size of the block relative to that day's ordinary turnover
    if "volume" in df.columns:
        df["blk_rel"] = df["blk_vol"] / df["volume"].replace(0, np.nan)

    print(f"\n{'='*84}")
    print("  BLOCK PRICE vs MARKET CLOSE")
    print(f"{'-'*84}")
    q = df["prem"].quantile([.05, .25, .5, .75, .95])
    print("  premium quantiles: " + "  ".join(f"{k:.0%}={v:+.2%}" for k, v in q.items()))
    print(f"  struck at a DISCOUNT: {(df['prem'] < 0).mean():.1%}   at a PREMIUM: {(df['prem'] > 0).mean():.1%}")
    if "blk_rel" in df:
        print(f"  block vol as multiple of day's matched volume: median {df['blk_rel'].median():.2f}x")

    print(f"\n{'='*84}")
    print("  FORWARD RETURNS — DISCOUNT vs PREMIUM BLOCKS")
    print("  naive = per-symbol t-test | bootstrap = resample whole trading days (honest)")
    print(f"{'-'*84}")
    disc = df[df["prem"] < -0.005]
    prem = df[df["prem"] > 0.005]
    print(f"  discount blocks (< -0.5%): {len(disc):,}    premium blocks (> +0.5%): {len(prem):,}")
    print()
    print(f"  {'grp':<9} {'h':<5} {'mean':>8} {'naive t':>8} {'naive p':>8} "
          f"{'boot 95% CI':>22} {'boot p':>7}")
    for label, sub in [("discount", disc), ("premium", prem)]:
        if len(sub) < 30:
            print(f"  {label:<9} too few ({len(sub)})")
            continue
        for h in FWD:
            col = f"f{h}"
            s = sub.dropna(subset=[col])
            if len(s) < 30:
                continue
            per_sym = s.groupby("symbol")[col].mean().dropna()
            if len(per_sym) >= 8:
                t, p = stats.ttest_1samp(per_sym, 0)
            else:
                t, p = np.nan, np.nan
            pt_, lo, hi, bp = block_bootstrap(s, col)
            print(f"  {label:<9} f{h:<4} {pt_:>+7.2%} {t:>8.2f} {p:>8.3f} "
                  f"  [{lo:>+6.2%},{hi:>+6.2%}] {bp:>7.3f}")

    print(f"\n{'='*84}")
    print("  DOES BLOCK SIZE MATTER? (large blocks relative to normal turnover)")
    print(f"{'-'*84}")
    if "blk_rel" in df:
        big = df[df["blk_rel"] >= df["blk_rel"].quantile(0.75)].dropna(subset=["f5"])
        small = df[df["blk_rel"] <= df["blk_rel"].quantile(0.25)].dropna(subset=["f5"])
        for label, sub in [("large blk", big), ("small blk", small)]:
            if len(sub) < 30:
                continue
            for h in [5, 10]:
                s = sub.dropna(subset=[f"f{h}"])
                pt_, lo, hi, bp = block_bootstrap(s, f"f{h}")
                print(f"  {label:<10} f{h:<3} mean={pt_:+.2%}  boot95=[{lo:+.2%},{hi:+.2%}]  p={bp:.3f}  n={len(s)}")

    out = os.path.join(BASE, "backtest_reports", "putthrough_blocks.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nsaved {len(df):,} rows to {out}")


if __name__ == "__main__":
    main()
