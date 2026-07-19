"""
momentum_and_reversal.py — Two simple, classical questions, tested with the
harness this project converged on.

Q1 MOMENTUM. Fama-French / Jegadeesh-Titman momentum is the 12-1 formation:
   return over the past 12 months SKIPPING the most recent month (the skip
   avoids short-term reversal, which is a separate and opposite effect).
   We test detection rules and holding periods:
     mom12_1   top-tercile 12-1 return
     mom6_1    top-tercile 6-1 return
     ma_cross  price > MA20 > MA60  (what "uptrend" meant earlier)
     mom12_1 + ma_cross
   Held 20 / 60 / 120 days.

Q2 REVERSAL AFTER A CRASH. The user's hypothesis: after a stock runs up
   +80/100/200% and then gives back 30-50%, is the expected forward return
   higher? Event study:
     runup   >= RUNUP_MIN over RUNUP_WIN days (peak vs trough)
     crash   >= CRASH_MIN drawdown from that peak
   then measure forward returns from the crash point vs benchmark.

*** SURVIVORSHIP WARNING ***
The price dataset contains ~1 delisted ticker in 1564 — i.e. it is essentially
all CURRENTLY-LISTED names. Vietnamese delistings 2014-2026 are absent. A
"bought the crash" test is therefore biased UPWARD: the crashes that went to
zero were removed from the sample. Q2 results are an UPPER BOUND, not an
estimate. Q1 is affected too but far less (momentum entries are not selected
on having crashed).

Guards: point-in-time liquidity, back-adjusted zero-price-guarded prices,
one position per ticker, costs, date-block bootstrap, Bonferroni across the
whole grid.

Usage:  python archive/momentum_and_reversal.py
"""

import glob, os, sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICE_DIR = os.path.join(BASE, "data", "price")

COST = 0.005
MIN_LIQ_BN = 20
LIQ_WINDOW = 250
WARMUP = 260
N_BOOT = 2000
RNG = np.random.default_rng(17)

HOLDS = [20, 60, 120]
RUNUP_WIN = 250
RUNUP_MINS = [0.80, 2.00]
CRASH_MINS = [0.30, 0.50]


def load(t):
    p = os.path.join(PRICE_DIR, f"{t}.parquet")
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    df.columns = [c.strip().lower() for c in df.columns]
    dc = "time" if "time" in df.columns else "date"
    df["date"] = pd.to_datetime(df[dc])
    if not {"close", "volume"}.issubset(df.columns):
        return None
    df = df[df["close"] > 0].sort_values("date").reset_index(drop=True)
    if len(df) < 600:
        return None
    c = df["close"]
    df["value_bn"] = c * 1000 * df["volume"] / 1e9
    df["liq_pit"] = df["value_bn"].rolling(LIQ_WINDOW, min_periods=100).mean().shift(1)
    df["ma20"] = c.rolling(20).mean()
    df["ma60"] = c.rolling(60).mean()
    df["ma_cross"] = (c > df["ma20"]) & (df["ma20"] > df["ma60"])
    # 12-1 and 6-1 momentum: skip the most recent 21 sessions
    df["mom12_1"] = c.shift(21) / c.shift(252) - 1
    df["mom6_1"] = c.shift(21) / c.shift(126) - 1
    # rolling peak/trough for the reversal event
    df["peak250"] = c.rolling(RUNUP_WIN, min_periods=200).max()
    df["trough250"] = c.rolling(RUNUP_WIN, min_periods=200).min()
    df["dd_from_peak"] = c / df["peak250"] - 1
    df["runup"] = df["peak250"] / df["trough250"] - 1
    return df


def simulate(df, mask, hold):
    out = []
    c = df["close"].values
    d = df["date"].values
    m = mask.values
    n = len(df)
    i = WARMUP
    while i < n - 1:
        if not m[i]:
            i += 1
            continue
        j = min(i + hold, n - 1)
        out.append((d[i], c[j] / c[i] - 1 - COST))
        i = j + 1
    return out


def boot(rows, n=N_BOOT):
    """Trade-weighted mean, winsorized, with a date-resampling bootstrap.

    Two bugs are deliberately avoided here:
      1. Reporting the mean of DATE-MEANS instead of the mean across TRADES.
         That equal-weights a date holding one freak trade (e.g. CFV +1327% in
         20 sessions) against a date holding fifty ordinary ones, and inflated
         a true ~+1.5% 20-day benchmark to a nonsensical +32%.
      2. Letting untruncated tails dominate. Vietnamese microcaps produce
         genuine 10x prints; winsorizing at 1/99 keeps them directional
         without letting a handful set the mean.
    The bootstrap still resamples whole DATES (shared-calendar correlation),
    but recomputes a TRADE-WEIGHTED mean inside each resample.
    """
    df = pd.DataFrame(rows, columns=["date", "net"])
    if len(df) < 30:
        return None
    lo_c, hi_c = df["net"].quantile([0.01, 0.99])
    df["w"] = df["net"].clip(lo_c, hi_c)

    groups = {d: g["w"].values for d, g in df.groupby("date")}
    dts = np.array(list(groups))
    b = np.empty(n)
    for i in range(n):
        pick = RNG.choice(dts, len(dts), replace=True)
        vals = np.concatenate([groups[d] for d in pick])
        b[i] = vals.mean()
    b = b[np.isfinite(b)]
    if not len(b):
        return None
    return dict(n=len(df), dates=df["date"].nunique(), mean=df["w"].mean(),
                raw=df["net"].mean(),
                lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5),
                p=2 * min((b <= 0).mean(), (b >= 0).mean()),
                win=100 * (df["net"] > 0).mean())


def main():
    # data/price contains non-ticker artefacts (e.g. VPS_PANEL_ALL, a multi-ticker
    # panel whose consecutive rows jump between different stocks). Globbing it as
    # a ticker produced 55% of all trades with a +225% mean. Validate against the
    # official ticker list instead of trusting the directory listing.
    sect = pd.read_csv(os.path.join(BASE, "ticker_sectors.csv"))
    sect.columns = [c.strip().lower() for c in sect.columns]
    valid = set(sect["ticker"].astype(str).str.upper())
    tick = sorted(t for t in
                  (os.path.splitext(os.path.basename(f))[0].upper()
                   for f in glob.glob(os.path.join(PRICE_DIR, "*.parquet")))
                  if t in valid)
    print(f"loading {len(tick)} validated tickers...")
    data = {}
    for t in tick:
        d = load(t)
        if d is not None and (d["liq_pit"] >= MIN_LIQ_BN).sum() >= 200:
            data[t] = d
    print(f"  usable: {len(data)}")

    # cross-sectional momentum ranks per date
    print("ranking momentum cross-sectionally...")
    panel = pd.concat([d[["date", "mom12_1", "mom6_1"]].assign(ticker=t)
                       for t, d in data.items()], ignore_index=True).dropna()
    panel["r12"] = panel.groupby("date")["mom12_1"].rank(pct=True)
    panel["r6"] = panel.groupby("date")["mom6_1"].rank(pct=True)
    # Merge on BOTH keys in one pass. A per-ticker MultiIndex .loc + index merge
    # blows up to a cartesian join here (39.5 GiB) — keep it keyed and explicit.
    ranks = panel[["ticker", "date", "r12", "r6"]].drop_duplicates(["ticker", "date"])

    arms = {}
    for t, d in data.items():
        rk = ranks[ranks["ticker"] == t][["date", "r12", "r6"]]
        if rk.empty:
            continue
        d = d.merge(rk, on="date", how="left")
        liq = d["liq_pit"] >= MIN_LIQ_BN
        defs = {
            "benchmark":        liq,
            "mom12_1":          liq & (d["r12"] >= 2/3).fillna(False),
            "mom6_1":           liq & (d["r6"] >= 2/3).fillna(False),
            "ma_cross":         liq & d["ma_cross"].fillna(False),
            "mom12_1+ma_cross": liq & (d["r12"] >= 2/3).fillna(False) & d["ma_cross"].fillna(False),
        }
        for ru in RUNUP_MINS:
            for cr in CRASH_MINS:
                defs[f"crash_ru{int(ru*100)}_dd{int(cr*100)}"] = (
                    liq & (d["runup"] >= ru).fillna(False)
                        & (d["dd_from_peak"] <= -cr).fillna(False))
        for name, mask in defs.items():
            for h in HOLDS:
                arms.setdefault((name, h), []).extend(simulate(d, mask, h))

    results = {k: boot(v) for k, v in arms.items()}
    results = {k: v for k, v in results.items() if v}
    bonf = 0.05 / max(len(results), 1)

    def show(title, keys):
        print(f"\n{'='*100}")
        print(f"  {title}")
        print(f"{'-'*100}")
        print(f"  {'arm':<26}{'hold':>5}{'trades':>8}{'dates':>7}{'win%':>7}"
              f"{'net':>9}{'boot 95% CI':>21}{'p':>8}  sig")
        for k in keys:
            if k not in results:
                continue
            r = results[k]
            bench = results.get(("benchmark", k[1]))
            edge = f"{r['mean']-bench['mean']:+.2%}" if bench else "  --"
            sig = "***" if r["p"] < bonf else "*" if r["p"] < 0.05 else ""
            print(f"  {k[0]:<26}{k[1]:>5}{r['n']:>8,}{r['dates']:>7,}{r['win']:>6.1f}%"
                  f"{r['mean']:>+9.2%}   [{r['lo']:>+6.2%},{r['hi']:>+6.2%}]{r['p']:>8.3f}  {sig}"
                  f"   edge {edge}")

    print(f"\n{len(results)} arms tested. Bonferroni threshold p < {bonf:.2e}")
    mom_names = ["benchmark", "mom12_1", "mom6_1", "ma_cross", "mom12_1+ma_cross"]
    show("Q1 — MOMENTUM", [(n, h) for h in HOLDS for n in mom_names])

    cr_names = ["benchmark"] + [f"crash_ru{int(ru*100)}_dd{int(cr*100)}"
                                for ru in RUNUP_MINS for cr in CRASH_MINS]
    show("Q2 — REVERSAL AFTER CRASH  (*** SURVIVORSHIP-INFLATED: UPPER BOUND ***)",
         [(n, h) for h in HOLDS for n in cr_names])

    print(f"\n{'='*100}")
    print("  Q2 CAVEAT: dataset holds ~1 delisted ticker in 1564. Crashes that went to")
    print("  zero are ABSENT. Treat every Q2 number as a ceiling on the true return.")
    print(f"{'='*100}")

    out = os.path.join(BASE, "backtest_reports", "momentum_reversal.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame([{"arm": k[0], "hold": k[1], **v} for k, v in results.items()]).to_csv(out, index=False)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
