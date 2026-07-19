"""
player_role_test.py — Test the user's ROLE hypothesis about who each investor
bucket actually is, using behaviour rather than returns.

Hypothesis (user's, restated):
  - Foreign institutional (funds) : actively investing and managing
  - Proprietary (tu doanh)        : actively investing
  - Domestic institutional        : partly PASSIVE absorption (banks/corporates
                                    with contractual or balance-sheet motives,
                                    not a view on the stock)
  - Retail (both)                 : reactive

Why this matters more than another return test: if `to_chuc_trongnuoc` mixes
passive absorbers with active funds, it is a BADLY-DEFINED variable, and every
null result we got on it is uninformative rather than evidence of no effect.

We do NOT test returns here. We test behavioural fingerprints that separate an
active manager from a passive absorber:

  persistence   AR(1) of the player's own normalised daily flow. A passive
                absorber taking delivery on a schedule is highly persistent;
                an active manager starting and stopping is less so.
  burstiness    share of the player's total absolute flow occurring on its
                top-5% most active days. Active = lumpy, passive = smooth.
  price_beta    regression of today's flow on the past 5d return. Measures
                whether the player REACTS to price at all. A passive absorber
                should be near zero (indifferent to price); a momentum trader
                positive; a contrarian negative.
  vol_beta      regression of |flow| on 20d volatility. Active managers should
                trade more when there is more to trade on.
  news_beta     regression of |flow| on abnormal volume (a crude event proxy,
                since we have no news dates). Active should respond to events.

INFERENCE: one number per ticker per player, then a cross-ticker t-test, plus
paired comparisons between players on the SAME tickers (which removes any
ticker-level confound entirely). No forward returns, so no look-ahead concern
and no overlapping-window problem.

Usage:  python archive/player_role_test.py
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

PLAYERS = {
    "FgRetail":  "ca_nhan_nuocngoai_net",
    "FgInst":    "to_chuc_nuocngoai_net",
    "DomRetail": "ca_nhan_trongnuoc_net",
    "DomInst":   "to_chuc_trongnuoc_net",
    "Prop":      "tu_doanh_net",
}
MIN_DAYS = 400
MIN_LIQ_VND = 1_000_000_000


def safe_beta(y: np.ndarray, x: np.ndarray) -> float:
    """OLS slope of y on x, standardised so magnitudes are comparable across
    tickers with different flow scales."""
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    if len(y) < 100 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.polyfit((x - x.mean()) / x.std(), (y - y.mean()) / y.std(), 1)[0])


def main():
    files = sorted(glob.glob(os.path.join(FLOW_DIR, "*.parquet")))
    print(f"scanning {len(files)} flow files...")
    rows = []

    for i, fp in enumerate(files):
        t = os.path.splitext(os.path.basename(fp))[0].upper()
        try:
            f = pd.read_parquet(fp)
        except Exception:
            continue
        if "date" not in f.columns:
            continue
        f["date"] = pd.to_datetime(f["date"])
        f = f.sort_values("date").reset_index(drop=True)
        if len(f) < MIN_DAYS:
            continue

        pp = os.path.join(PRICE_DIR, f"{t}.parquet")
        if not os.path.exists(pp):
            continue
        p = pd.read_parquet(pp)
        p.columns = [c.strip().lower() for c in p.columns]
        dc = "time" if "time" in p.columns else "date"
        p["date"] = pd.to_datetime(p[dc])
        if "close" not in p.columns or "volume" not in p.columns:
            continue
        p = p[p["close"] > 0].sort_values("date").reset_index(drop=True)  # zero-price guard
        if (p["close"] * p["volume"] * 1000).tail(60).median() < MIN_LIQ_VND:
            continue

        if "close" in f.columns:
            f = f.drop(columns=["close"])
        df = f.merge(p[["date", "close", "volume"]], on="date", how="inner")
        if len(df) < MIN_DAYS:
            continue

        df["ret5"] = df["close"].pct_change(5)
        df["vol20"] = df["close"].pct_change().rolling(20).std()
        df["abnvol"] = df["volume"] / df["volume"].rolling(60).mean()

        for name, col in PLAYERS.items():
            if col not in df.columns:
                continue
            raw = df[col].astype(float).fillna(0)
            scale = raw.abs().rolling(60, min_periods=20).mean().clip(lower=1e-6)
            x = (raw / scale).replace([np.inf, -np.inf], np.nan)
            if x.notna().sum() < MIN_DAYS * 0.5 or x.std() == 0:
                continue

            a = x.dropna()
            persistence = float(a.autocorr(lag=1)) if len(a) > 100 else np.nan

            absf = raw.abs()
            tot = absf.sum()
            if tot > 0:
                thr = absf.quantile(0.95)
                burst = float(absf[absf >= thr].sum() / tot)
            else:
                burst = np.nan

            rows.append({
                "ticker": t, "player": name,
                "persistence": persistence,
                "burstiness": burst,
                "price_beta": safe_beta(x.values, df["ret5"].values),
                "vol_beta":   safe_beta(x.abs().values, df["vol20"].values),
                "news_beta":  safe_beta(x.abs().values, df["abnvol"].values),
            })

        if (i + 1) % 300 == 0:
            print(f"  ...{i+1}/{len(files)}  ({len({r['ticker'] for r in rows})} tickers kept)")

    d = pd.DataFrame(rows)
    print(f"\n{d['ticker'].nunique()} tickers x {d['player'].nunique()} players = {len(d):,} rows")

    metrics = ["persistence", "burstiness", "price_beta", "vol_beta", "news_beta"]

    print(f"\n{'='*96}")
    print("  BEHAVIOURAL FINGERPRINT BY PLAYER  (mean across tickers)")
    print(f"{'-'*96}")
    print(f"  {'player':<11}" + "".join(f"{m:>16}" for m in metrics))
    for name in PLAYERS:
        g = d[d.player == name]
        if g.empty:
            continue
        print(f"  {name:<11}" + "".join(f"{g[m].mean():>16.3f}" for m in metrics))

    print(f"\n{'='*96}")
    print("  PAIRED TESTS vs DomInst  (same tickers only — removes ticker confounds)")
    print("  Hypothesis: DomInst is partly PASSIVE, so it should show HIGHER")
    print("  persistence, LOWER burstiness, and price/vol/news betas nearer ZERO.")
    print(f"{'-'*96}")
    base = d[d.player == "DomInst"].set_index("ticker")
    for name in ["FgInst", "Prop", "DomRetail", "FgRetail"]:
        other = d[d.player == name].set_index("ticker")
        common = base.index.intersection(other.index)
        if len(common) < 30:
            continue
        print(f"\n  DomInst vs {name}   (n={len(common)} tickers)")
        for m in metrics:
            diff = (base.loc[common, m] - other.loc[common, m]).dropna()
            if len(diff) < 30:
                continue
            t, p = stats.ttest_1samp(diff, 0)
            star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
            print(f"    {m:<13} DomInst-{name:<10} {diff.mean():>+8.3f}  t={t:>7.2f}  p={p:<8.4f}{star}")

    print(f"\n{'='*96}")
    print("  IS DomInst BIMODAL?  (a mixed passive+active bucket should show two")
    print("  populations of tickers, not one; pure buckets should look unimodal)")
    print(f"{'-'*96}")
    for name in PLAYERS:
        g = d[d.player == name]["persistence"].dropna()
        if len(g) < 50:
            continue
        # dip-style crude check: gap between tercile means relative to spread
        q = g.quantile([.1, .5, .9])
        print(f"  {name:<11} persistence  p10={q[.1]:+.3f}  med={q[.5]:+.3f}  "
              f"p90={q[.9]:+.3f}  sd={g.std():.3f}  skew={g.skew():+.2f}")

    out = os.path.join(BASE, "backtest_reports", "player_roles.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
