"""
momentum_quality_test.py — Combine the only two things that survived this
project's testing: MOMENTUM (uptrend, +0.95% over benchmark, boot p=0.000 in
ablation_trendfollow.py) and QUALITY (ROE, p=0.001 in the panel regressions).

Flow is deliberately absent. Every flow specification tested — levels,
thresholds, liquidity terciles, concentration, cycles, ticks, put-through,
lead-lag, and DeepSeek's trend/contrarian rules — either died to costs,
clustering, zero-price rows, the sum-to-zero identity, or (in the last case)
turned out to be momentum wearing a flow costume.

ARMS (all long-only, same universe, same dates, one position per ticker):
  1 benchmark        every liquid stock
  2 momentum         uptrend only
  3 quality          top-tercile ROE only
  4 momentum+quality uptrend AND top-tercile ROE
  5 mom+qual, sector-neutral   as 4, but ROE ranked WITHIN sector

Arm 5 is the real question: Vietnamese sectors have very different ROE levels
(banks vs real estate vs utilities), so a raw cross-sectional ROE screen is
partly a disguised sector bet. Ranking within sector removes that.

INFERENCE GUARDS (all learned the hard way in this project):
  - point-in-time liquidity (trailing window, shifted) — no survivorship
  - fundamentals lagged via avail_date — no look-ahead into unreleased reports
  - back-adjusted prices from data/price, zero-price guarded
  - one open position per ticker — no overlapping pseudo-trades
  - DATE-BLOCK bootstrap — resamples whole entry dates, since all tickers
    share a calendar and one market-wide move is not N independent bets

Usage:  python archive/momentum_quality_test.py
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

PRICE_DIR = os.path.join(BASE, "data", "price")

COST = 0.005
HOLD = 60
STOP = -0.127
MIN_LIQ_BN = 50
LIQ_WINDOW = 250
WARMUP = 180
N_BOOT = 2000
RNG = np.random.default_rng(13)


def load_price(t):
    p = os.path.join(PRICE_DIR, f"{t}.parquet")
    if not os.path.exists(p):
        return None
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
    if len(df) < 500:
        return None
    df["value_bn"] = df["close"] * 1000 * df["volume"] / 1e9
    df["liq_pit"] = df["value_bn"].rolling(LIQ_WINDOW, min_periods=100).mean().shift(1)
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["uptrend"] = (df["ma20"] > df["ma60"]) & (df["close"] > df["ma20"])
    return df[["date", "close", "liq_pit", "uptrend"]]


def simulate(df, mask):
    out = []
    close = df["close"].values
    dates = df["date"].values
    m = mask.values
    n = len(df)
    i = WARMUP
    while i < n - 1:
        if not m[i]:
            i += 1
            continue
        entry = close[i]
        j_end = min(i + HOLD, n - 1)
        exit_i, gross = j_end, None
        for j in range(i + 1, j_end + 1):
            if close[j] / entry - 1 <= STOP:
                exit_i, gross = j, STOP
                break
        if gross is None:
            gross = close[exit_i] / entry - 1
        out.append((dates[i], gross - COST))
        i = exit_i + 1
    return out


def boot(df, n=N_BOOT):
    if df.empty:
        return (np.nan,) * 4
    dts = df["date"].unique()
    obs = df.groupby("date")["net"].mean()
    b = np.empty(n)
    for i in range(n):
        b[i] = obs.reindex(RNG.choice(dts, len(dts), replace=True)).mean()
    b = b[np.isfinite(b)]
    if not len(b):
        return (np.nan,) * 4
    return obs.mean(), *np.percentile(b, [2.5, 97.5]), 2 * min((b <= 0).mean(), (b >= 0).mean())


def main():
    tickers = sorted(os.path.splitext(os.path.basename(f))[0].upper()
                     for f in glob.glob(os.path.join(PRICE_DIR, "*.parquet")))
    sect = pd.read_csv(os.path.join(BASE, "ticker_sectors.csv"))
    sect.columns = [c.strip().lower() for c in sect.columns]
    sect = sect[sect["exchange"].isin(["HOSE", "HNX"])]
    t2s = dict(zip(sect["ticker"].str.upper(), sect["industry"]))

    print("loading prices...")
    px = {}
    for t in tickers:
        d = load_price(t)
        if d is not None and (d["liq_pit"] >= MIN_LIQ_BN).sum() >= 100 and t in t2s:
            px[t] = d
    print(f"  liquid tickers with sector: {len(px)}")

    print("loading fundamentals...")
    qf = build_factor_features(symbols=list(px))
    qf["avail_date"] = pd.to_datetime(qf["avail_date"])
    qf = qf.dropna(subset=["roe"]).sort_values("avail_date")

    # Point-in-time ROE panel: as-of merge so only released reports are used.
    roe_rows = []
    for t, d in px.items():
        sub = qf[qf["symbol"] == t][["avail_date", "roe"]]
        if sub.empty:
            continue
        m = pd.merge_asof(d[["date"]].sort_values("date"), sub,
                          left_on="date", right_on="avail_date", direction="backward")
        m["ticker"] = t
        m["sector"] = t2s[t]
        roe_rows.append(m[["date", "ticker", "sector", "roe"]])
    roe = pd.concat(roe_rows, ignore_index=True).dropna(subset=["roe"])
    print(f"  ROE panel rows: {len(roe):,}  tickers: {roe['ticker'].nunique()}")

    # Cross-sectional ROE ranks per DATE (uses only same-day released data)
    roe["roe_rank"] = roe.groupby("date")["roe"].rank(pct=True)
    roe["roe_rank_sec"] = roe.groupby(["date", "sector"])["roe"].rank(pct=True)
    roe_idx = roe.set_index(["ticker", "date"])

    arms = {k: [] for k in ["1_benchmark", "2_momentum", "3_quality",
                            "4_mom_qual", "5_mom_qual_sectorneutral"]}
    for t, d in px.items():
        sub = roe_idx.loc[t] if t in roe_idx.index.get_level_values(0) else None
        if sub is None or sub.empty:
            continue
        d = d.merge(sub[["roe_rank", "roe_rank_sec"]], left_on="date",
                    right_index=True, how="left")
        liq = d["liq_pit"] >= MIN_LIQ_BN
        up = d["uptrend"].fillna(False) & liq
        hi = (d["roe_rank"] >= 2 / 3).fillna(False) & liq
        hi_s = (d["roe_rank_sec"] >= 2 / 3).fillna(False) & liq
        for name, mask in [
            ("1_benchmark", liq),
            ("2_momentum", up),
            ("3_quality", hi),
            ("4_mom_qual", up & hi),
            ("5_mom_qual_sectorneutral", up & hi_s),
        ]:
            for dt, r in simulate(d, mask):
                arms[name].append({"date": dt, "net": r, "ticker": t})

    print(f"\n{'='*96}")
    print(f"  MOMENTUM x QUALITY — hold {HOLD}d, stop {STOP:.1%}, cost {COST:.1%}")
    print(f"  one position per ticker | point-in-time liquidity + lagged fundamentals")
    print(f"  bootstrap resamples whole ENTRY DATES")
    print(f"{'-'*96}")
    print(f"  {'arm':<28}{'trades':>8}{'dates':>7}{'win%':>7}{'net mean':>10}"
          f"{'boot 95% CI':>22}{'boot p':>9}")
    res = {}
    for k, v in arms.items():
        df = pd.DataFrame(v)
        if len(df) < 50:
            print(f"  {k:<28} too few ({len(df)})"); continue
        m, lo, hi_, p = boot(df)
        res[k] = m
        print(f"  {k:<28}{len(df):>8,}{df['date'].nunique():>7,}"
              f"{100*(df['net'] > 0).mean():>6.1f}%{m:>+10.2%}"
              f"   [{lo:>+6.2%},{hi_:>+6.2%}]{p:>9.3f}")

    print(f"\n{'='*96}")
    print("  EDGE DECOMPOSITION (vs benchmark)")
    print(f"{'-'*96}")
    b = res.get("1_benchmark", np.nan)
    for k in ["2_momentum", "3_quality", "4_mom_qual", "5_mom_qual_sectorneutral"]:
        if k in res:
            print(f"  {k:<28} {res[k] - b:+.2%}")
    if "4_mom_qual" in res and "2_momentum" in res:
        print(f"\n  quality added to momentum   : {res['4_mom_qual'] - res['2_momentum']:+.2%}")
    if "5_mom_qual_sectorneutral" in res and "4_mom_qual" in res:
        print(f"  sector-neutral vs raw ROE   : "
              f"{res['5_mom_qual_sectorneutral'] - res['4_mom_qual']:+.2%}")
        print("  (if raw >> sector-neutral, the ROE screen was a disguised SECTOR bet)")

    out = os.path.join(BASE, "backtest_reports", "momentum_quality.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.concat([pd.DataFrame(v).assign(arm=k) for k, v in arms.items() if v]).to_csv(out, index=False)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
