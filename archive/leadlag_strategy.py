"""
leadlag_strategy.py — Two parts.

PART 1 — LEAD-LAG. Who moves first?
Level tests are crippled by the sum-to-zero identity (the five net flows must
cancel, so "FgInst buys" IS "someone else sells" — same fact twice). Timing is
not: cross-correlation of player A's flow at t with player B's flow at t+k
asks who ACTS first, which survives the identity.

For each ticker, for each ordered pair (A,B), correlate A_flow[t] with
B_flow[t+k] for k in LAGS. Aggregate one number per ticker, then test across
tickers. Also lead-lag of each player's flow vs FUTURE RETURN.

PART 2 — STRATEGY from the role fingerprints (player_role_test.py):
    FgInst    price_beta +0.060  -> momentum / trend follower
    DomRetail price_beta -0.047  -> contrarian, buys dips
    Prop      price_beta ~0, burstiness 0.79, vol_beta high -> market maker
    DomInst   price_beta ~0      -> passive absorber

Trade thesis: if FgInst is the momentum player and DomRetail the contrarian,
then FgInst buying WHILE DomRetail sells is the configuration where the
informed-momentum side is being supplied by the reactive side. Prop and DomInst
are excluded from signals entirely — the role test says neither carries
direction.

HONEST INFERENCE (every earlier version of this test in this project failed for
lack of it):
  - point-in-time thresholds only (expanding, shifted), no full-sample quantile
  - transaction costs charged
  - DATE-BLOCK bootstrap: resample whole trading days, because all tickers share
    a calendar and one market-wide move is not N independent bets
  - zero-price guard on every price load

Usage:  python archive/leadlag_strategy.py
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
LAGS = [1, 2, 3, 5, 10]
MIN_DAYS = 400
MIN_LIQ_VND = 1_000_000_000
COST = 0.005          # 0.5% round-trip, project convention
HOLD = 10
N_BOOT = 2000
RNG = np.random.default_rng(7)


def load(t):
    fp = os.path.join(FLOW_DIR, f"{t}.parquet")
    pp = os.path.join(PRICE_DIR, f"{t}.parquet")
    if not (os.path.exists(fp) and os.path.exists(pp)):
        return None
    f = pd.read_parquet(fp)
    if "date" not in f.columns or len(f) < MIN_DAYS:
        return None
    f["date"] = pd.to_datetime(f["date"])
    if "close" in f.columns:
        f = f.drop(columns=["close"])
    p = pd.read_parquet(pp)
    p.columns = [c.strip().lower() for c in p.columns]
    dc = "time" if "time" in p.columns else "date"
    p["date"] = pd.to_datetime(p[dc])
    if "close" not in p.columns or "volume" not in p.columns:
        return None
    p = p[p["close"] > 0]                      # zero-price guard
    if p.empty or (p["close"] * p["volume"] * 1000).tail(60).median() < MIN_LIQ_VND:
        return None
    df = f.merge(p[["date", "close", "volume"]], on="date", how="inner").sort_values("date")
    df = df.reset_index(drop=True)
    if len(df) < MIN_DAYS:
        return None
    for n, c in PLAYERS.items():
        if c not in df.columns:
            return None
        raw = df[c].astype(float).fillna(0)
        sc = raw.abs().rolling(60, min_periods=20).mean().clip(lower=1e-6)
        df[f"{n}_n"] = (raw / sc).replace([np.inf, -np.inf], np.nan)
    df["ret1"] = df["close"].pct_change()
    df[f"fwd{HOLD}"] = df["close"].shift(-HOLD) / df["close"] - 1
    return df


def boot_dates(df, col, n=N_BOOT):
    dts = df["date"].unique()
    obs = df.groupby("date")[col].mean()
    b = np.empty(n)
    for i in range(n):
        b[i] = obs.reindex(RNG.choice(dts, len(dts), replace=True)).mean()
    b = b[np.isfinite(b)]
    if not len(b):
        return np.nan, np.nan, np.nan, np.nan
    return df[col].mean(), *np.percentile(b, [2.5, 97.5]), 2 * min((b <= 0).mean(), (b >= 0).mean())


def main():
    tickers = [os.path.splitext(os.path.basename(f))[0].upper()
               for f in sorted(glob.glob(os.path.join(FLOW_DIR, "*.parquet")))]
    print(f"loading {len(tickers)} tickers...")
    data = {}
    for i, t in enumerate(tickers):
        d = load(t)
        if d is not None:
            data[t] = d
        if (i + 1) % 400 == 0:
            print(f"  ...{i+1}  kept={len(data)}")
    print(f"usable tickers: {len(data)}")

    names = list(PLAYERS)

    # ---------- PART 1a: player -> player lead-lag ----------
    print(f"\n{'='*92}")
    print("  LEAD-LAG: corr( A_flow[t] , B_flow[t+k] )   mean across tickers")
    print("  positive at k>0 => A's move today is followed by B doing the same later")
    print(f"{'-'*92}")
    res = {}
    for a in names:
        for b in names:
            if a == b:
                continue
            for k in LAGS:
                vals = []
                for d in data.values():
                    x, y = d[f"{a}_n"], d[f"{b}_n"].shift(-k)
                    m = x.notna() & y.notna()
                    if m.sum() > 200:
                        c = x[m].corr(y[m])
                        if np.isfinite(c):
                            vals.append(c)
                if len(vals) > 30:
                    v = np.array(vals)
                    t_, p_ = stats.ttest_1samp(v, 0)
                    res[(a, b, k)] = (v.mean(), t_, p_, len(v))

    # strongest lead relationships, Bonferroni-aware
    nt = len(res)
    bonf = 0.05 / max(nt, 1)
    print(f"  {nt} pair-lag tests, Bonferroni threshold p<{bonf:.2e}")
    top = sorted(res.items(), key=lambda kv: -abs(kv[1][0]))[:14]
    print(f"  {'A leads':<11}{'B follows':<11}{'k':>3}{'corr':>9}{'t':>9}{'p':>11}  sig")
    for (a, b, k), (m, t_, p_, n_) in top:
        print(f"  {a:<11}{b:<11}{k:>3}{m:>+9.3f}{t_:>9.1f}{p_:>11.2e}  {'YES' if p_ < bonf else ''}")

    # asymmetry: does A->B differ from B->A? that is the real "who is faster"
    print(f"\n{'-'*92}")
    print("  ASYMMETRY  corr(A[t],B[t+k]) - corr(B[t],A[t+k])   >0 means A leads B")
    print(f"{'-'*92}")
    for k in [1, 3, 5]:
        print(f"  k={k}")
        for i, a in enumerate(names):
            for b in names[i+1:]:
                if (a, b, k) in res and (b, a, k) in res:
                    d1, d2 = res[(a, b, k)][0], res[(b, a, k)][0]
                    print(f"    {a:<10} vs {b:<10} {d1-d2:>+7.3f}  "
                          f"{'(' + a + ' leads)' if d1 > d2 else '(' + b + ' leads)'}")

    # ---------- PART 1b: flow -> future return ----------
    print(f"\n{'='*92}")
    print("  FLOW -> FUTURE RETURN  corr( flow[t] , ret[t+k] )")
    print(f"{'-'*92}")
    print(f"  {'player':<11}" + "".join(f"k={k:<8}" for k in LAGS))
    for a in names:
        cells = []
        for k in LAGS:
            vals = []
            for d in data.values():
                x, y = d[f"{a}_n"], d["ret1"].shift(-k)
                m = x.notna() & y.notna()
                if m.sum() > 200:
                    c = x[m].corr(y[m])
                    if np.isfinite(c):
                        vals.append(c)
            if len(vals) > 30:
                v = np.array(vals)
                t_, p_ = stats.ttest_1samp(v, 0)
                cells.append(f"{v.mean():+.3f}{'*' if p_ < bonf else ' '}  ")
            else:
                cells.append("  --     ")
        print(f"  {a:<11}" + "".join(cells))
    print("  (* = survives Bonferroni across all pair-lag tests)")

    # ---------- PART 2: strategy ----------
    print(f"\n{'='*92}")
    print("  STRATEGY — FgInst (momentum) buying WHILE DomRetail (contrarian) sells")
    print(f"  entry: FgInst_n > expanding mean+1sd  AND  DomRetail_n < 0   |  hold {HOLD}d")
    print(f"  point-in-time thresholds, {COST:.1%} round-trip cost, date-block bootstrap")
    print(f"{'-'*92}")
    sig_rows = []
    for t, d in data.items():
        fi = d["FgInst_n"]
        mu = fi.expanding(min_periods=120).mean().shift(1)
        sd = fi.expanding(min_periods=120).std().shift(1)
        entry = (fi > mu + sd) & (d["DomRetail_n"] < 0)
        sub = d.loc[entry & d[f"fwd{HOLD}"].notna(), ["date", f"fwd{HOLD}"]].copy()
        if sub.empty:
            continue
        sub["ticker"] = t
        sig_rows.append(sub)
    if not sig_rows:
        print("  no signals"); return
    sig = pd.concat(sig_rows, ignore_index=True)
    sig["net"] = sig[f"fwd{HOLD}"] - COST

    print(f"  trades={len(sig):,}  tickers={sig['ticker'].nunique()}  dates={sig['date'].nunique()}")
    print(f"  gross mean={sig[f'fwd{HOLD}'].mean():+.2%}   net mean={sig['net'].mean():+.2%}   "
          f"win rate={100*(sig['net'] > 0).mean():.1f}%")
    per_t = sig.groupby("ticker")["net"].mean().dropna()
    t_, p_ = stats.ttest_1samp(per_t, 0)
    print(f"  naive per-ticker t-test:  t={t_:.2f}  p={p_:.4f}   (INFLATED - shared calendar)")
    m_, lo, hi, bp = boot_dates(sig, "net")
    print(f"  date-block bootstrap:     mean={m_:+.2%}  95%CI=[{lo:+.2%},{hi:+.2%}]  p={bp:.3f}  <-- honest")

    # benchmark: unconditional forward return over same dates
    base_rows = []
    for t, d in data.items():
        s = d.loc[d[f"fwd{HOLD}"].notna(), ["date", f"fwd{HOLD}"]].copy()
        s["ticker"] = t
        base_rows.append(s)
    base = pd.concat(base_rows, ignore_index=True)
    base = base[base["date"].isin(set(sig["date"]))]
    print(f"  benchmark (all stocks, same dates): {base[f'fwd{HOLD}'].mean():+.2%}")
    print(f"  EDGE over benchmark (gross): {sig[f'fwd{HOLD}'].mean() - base[f'fwd{HOLD}'].mean():+.2%}")

    out = os.path.join(BASE, "backtest_reports", "leadlag_strategy_trades.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sig.to_csv(out, index=False)
    print(f"\nsaved {len(sig):,} signals to {out}")


if __name__ == "__main__":
    main()
