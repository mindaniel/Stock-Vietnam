"""
ablation_trendfollow.py — Is DeepSeek's surviving "Trend Following" signal
actually about FOREIGN INSTITUTIONAL FLOW, or is it just momentum beta?

Their Strategy A entry = FI_strong_buy AND uptrend. It reports +2.96% net,
p=0.0003. But the entry has TWO conditions and only one of them is about flow.
The decisive test is an ablation:

    ARM 1  uptrend only              (no flow condition at all)
    ARM 2  uptrend AND FI strong     (DeepSeek's actual signal)
    ARM 3  uptrend AND FI WEAK       (the opposite flow condition)
    ARM 4  random entry, same dates  (pure benchmark)

If ARM 2 ~= ARM 1 ~= ARM 3, the flow term contributes nothing and the result is
momentum, not flow alpha. If ARM 2 > ARM 1 > ARM 3, the flow term is real.

Three additional corrections to their harness, each of which independently
matters:

 A) LIQUIDITY LOOK-AHEAD. They filter on the LAST 2 YEARS of turnover, then
    backtest all history. Stocks liquid in 2026 are survivors; trading them
    from 2015 uses knowledge from the future. Here liquidity is evaluated
    POINT-IN-TIME on a trailing window at each candidate entry.

 B) OVERLAPPING TRADES. They emit a signal every qualifying day and hold up to
    60 days, so trades overlap heavily; weekly aggregation cannot fix a 12-week
    holding period. Here we (i) enforce ONE OPEN POSITION PER TICKER, and
    (ii) bootstrap by resampling whole ENTRY DATES.

 C) BENCHMARK. Report the return of every stock over the same horizon on the
    same dates, so "edge" means edge over the market, not over zero.

Price source: data/price close (zero-price guarded). Note the flow parquet also
carries a close column; they used that one. Both are checked here for agreement.

Usage:  python archive/ablation_trendfollow.py
"""

import glob, os, sys
import numpy as np
import pandas as pd
from scipy import stats

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOW_DIR = os.path.join(BASE, "data", "investor_flow")
PRICE_DIR = os.path.join(BASE, "data", "price")

COST = 0.005                 # 0.5% round trip, same as DeepSeek
HOLD = 60                    # their time stop
STOP = -0.127                # their effective stop (0.90*0.97-1)
MIN_LIQ_BN = 50              # their 50B VND/day bar
LIQ_WINDOW = 250             # but measured POINT-IN-TIME, not on the last 2y
WARMUP = 180
N_BOOT = 2000
RNG = np.random.default_rng(11)


def load(t):
    fp = os.path.join(FLOW_DIR, f"{t}.parquet")
    pp = os.path.join(PRICE_DIR, f"{t}.parquet")
    if not (os.path.exists(fp) and os.path.exists(pp)):
        return None
    try:
        f = pd.read_parquet(fp); p = pd.read_parquet(pp)
    except Exception:
        return None
    if "date" not in f.columns or len(f) < 500:
        return None
    f["date"] = pd.to_datetime(f["date"])
    p.columns = [c.strip().lower() for c in p.columns]
    dc = "time" if "time" in p.columns else "date"
    p["date"] = pd.to_datetime(p[dc])
    if not {"close", "volume"}.issubset(p.columns):
        return None
    p = p[p["close"] > 0]
    need = ["date", "close", "to_chuc_nuocngoai_net"]
    if not set(need).issubset(f.columns):
        return None
    f = f.rename(columns={"close": "flow_close"})
    df = f[["date", "flow_close", "to_chuc_nuocngoai_net"]].merge(
        p[["date", "close", "volume"]], on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    df = df[df["close"] > 0].reset_index(drop=True)
    if len(df) < 500:
        return None

    df["value_bn"] = df["close"] * 1000 * df["volume"] / 1e9
    # POINT-IN-TIME liquidity: trailing window only, shifted so today is unknown
    df["liq_pit"] = df["value_bn"].rolling(LIQ_WINDOW, min_periods=100).mean().shift(1)

    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["uptrend"] = (df["ma20"] > df["ma60"]) & (df["close"] > df["ma20"])

    fi = df["to_chuc_nuocngoai_net"].astype(float).fillna(0)
    df["FI_5d"] = fi.rolling(5, min_periods=3).sum()
    df["fi_mu"] = df["FI_5d"].expanding(min_periods=60).mean().shift(1)
    df["fi_sd"] = df["FI_5d"].expanding(min_periods=60).std().shift(1)
    df["FI_strong"] = (df["FI_5d"] > df["fi_mu"] + 0.5 * df["fi_sd"]) & (fi > 0)
    df["FI_weak"] = (df["FI_5d"] < df["fi_mu"] - 0.5 * df["fi_sd"])
    return df


def simulate(df, entry_mask):
    """One open position per ticker. Fixed HOLD with stop. Returns list of
    (entry_date, net_return)."""
    out = []
    close = df["close"].values
    dates = df["date"].values
    n = len(df)
    mask = entry_mask.values
    i = WARMUP
    while i < n - 1:
        if not mask[i]:
            i += 1
            continue
        entry = close[i]
        if not np.isfinite(entry) or entry <= 0:
            i += 1
            continue
        j_end = min(i + HOLD, n - 1)
        exit_i, gross = j_end, None
        for j in range(i + 1, j_end + 1):
            r = close[j] / entry - 1
            if r <= STOP:
                exit_i, gross = j, STOP
                break
        if gross is None:
            gross = close[exit_i] / entry - 1
        out.append((dates[i], gross - COST))
        i = exit_i + 1          # <-- no overlapping positions
    return out


def boot(trades_df, n=N_BOOT):
    if trades_df.empty:
        return (np.nan,) * 4
    dts = trades_df["date"].unique()
    obs = trades_df.groupby("date")["net"].mean()
    b = np.empty(n)
    for i in range(n):
        b[i] = obs.reindex(RNG.choice(dts, len(dts), replace=True)).mean()
    b = b[np.isfinite(b)]
    if not len(b):
        return (np.nan,) * 4
    return obs.mean(), *np.percentile(b, [2.5, 97.5]), 2 * min((b <= 0).mean(), (b >= 0).mean())


def main():
    tickers = [os.path.splitext(os.path.basename(f))[0].upper()
               for f in sorted(glob.glob(os.path.join(FLOW_DIR, "*.parquet")))]
    print(f"scanning {len(tickers)} tickers (point-in-time liquidity >= {MIN_LIQ_BN}B VND/day)...")

    arms = {"1_uptrend_only": [], "2_uptrend_FIstrong": [],
            "3_uptrend_FIweak": [], "4_benchmark_all": []}
    kept = 0
    px_gap = []
    for i, t in enumerate(tickers):
        d = load(t)
        if d is None:
            continue
        liq_ok = d["liq_pit"] >= MIN_LIQ_BN
        if liq_ok.sum() < 100:
            continue
        kept += 1
        # sanity: does the flow parquet's close agree with the price parquet's?
        both = d[["flow_close", "close"]].dropna()
        if len(both) > 100:
            px_gap.append(float((both["flow_close"] / both["close"] - 1).abs().median()))

        up = d["uptrend"].fillna(False) & liq_ok
        for name, m in [
            ("1_uptrend_only", up),
            ("2_uptrend_FIstrong", up & d["FI_strong"].fillna(False)),
            ("3_uptrend_FIweak", up & d["FI_weak"].fillna(False)),
            ("4_benchmark_all", liq_ok),
        ]:
            for dt, r in simulate(d, m):
                arms[name].append({"date": dt, "net": r, "ticker": t})

        if (i + 1) % 400 == 0:
            print(f"  ...{i+1}  kept={kept}")

    print(f"usable tickers: {kept}")
    if px_gap:
        print(f"flow_close vs price close, median |gap|: {np.median(px_gap):.4%} "
              f"(large => split/dividend adjustment differs)")

    print(f"\n{'='*96}")
    print(f"  ABLATION — hold {HOLD}d, stop {STOP:.1%}, cost {COST:.1%}, ONE position per ticker")
    print(f"  bootstrap resamples ENTRY DATES (whole-market days), not trades")
    print(f"{'-'*96}")
    print(f"  {'arm':<22}{'trades':>8}{'dates':>7}{'win%':>7}{'net mean':>10}"
          f"{'boot 95% CI':>22}{'boot p':>9}")
    results = {}
    for name, rows in arms.items():
        df = pd.DataFrame(rows)
        if df.empty or len(df) < 50:
            print(f"  {name:<22} too few"); continue
        m, lo, hi, p = boot(df)
        results[name] = m
        print(f"  {name:<22}{len(df):>8,}{df['date'].nunique():>7,}"
              f"{100*(df['net'] > 0).mean():>6.1f}%{m:>+10.2%}"
              f"   [{lo:>+6.2%},{hi:>+6.2%}]{p:>9.3f}")

    print(f"\n{'='*96}")
    print("  VERDICT")
    print(f"{'-'*96}")
    if "2_uptrend_FIstrong" in results and "1_uptrend_only" in results:
        edge = results["2_uptrend_FIstrong"] - results["1_uptrend_only"]
        print(f"  FI-strong minus uptrend-only : {edge:+.2%}   <-- what the FLOW term adds")
    if "2_uptrend_FIstrong" in results and "3_uptrend_FIweak" in results:
        print(f"  FI-strong minus FI-weak      : "
              f"{results['2_uptrend_FIstrong'] - results['3_uptrend_FIweak']:+.2%}")
    if "1_uptrend_only" in results and "4_benchmark_all" in results:
        print(f"  uptrend-only minus benchmark : "
              f"{results['1_uptrend_only'] - results['4_benchmark_all']:+.2%}   <-- momentum beta")
    print("\n  If the first line is ~0, the surviving signal is MOMENTUM, not foreign flow.")

    out = os.path.join(BASE, "backtest_reports", "ablation_trendfollow.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.concat([pd.DataFrame(v).assign(arm=k) for k, v in arms.items() if v]).to_csv(out, index=False)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
