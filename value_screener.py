"""
value_screener.py
Concentrated value + earnings-momentum screener for VN equities.

Logic:
  1. Price drawn down from 52-week high  → beaten-down / cheap
  2. Cheap valuation (low PE, PB, PS)    → margin of safety
  3. Earnings/revenue accelerating YoY  → fundamental catalyst
  4. Quality guardrail (profitable, not zombie, not over-leveraged)
  5. Composite score → top 15 names

Usage:
  python value_screener.py               # top 15, default settings
  python value_screener.py --top 20
  python value_screener.py --min-drawdown 0.15   # at least 15% off peak
  python value_screener.py --sector "Banks"
  python value_screener.py --detail HPG          # full quarterly history for one stock
"""

import argparse, os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data", "price")
FA_DIR      = os.path.join(DATA_DIR, "financials_fa")
OUT_DIR     = os.path.join(BASE_DIR, "results")
TICKER_CSV  = os.path.join(BASE_DIR, "ticker_sectors.csv")
os.makedirs(OUT_DIR, exist_ok=True)


def load_exchange_map() -> dict:
    """Return {TICKER: exchange} from ticker_sectors.csv. Empty dict if file missing."""
    if not os.path.exists(TICKER_CSV):
        return {}
    df = pd.read_csv(TICKER_CSV, usecols=lambda c: c in ("ticker", "exchange"))
    if "exchange" not in df.columns:
        return {}
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["exchange"] = df["exchange"].fillna("").astype(str).str.strip().str.upper()
    return dict(zip(df["ticker"], df["exchange"]))


# ── Load price data (daily) ───────────────────────────────────────────────────

# Daily turnover rate assumption for market cap estimation.
# Liquid VN stocks trade ~0.3–0.7% of market cap per day.
# close in CSV is in units of 1,000 VND; value = close * volume.
# So: market_cap_vnd ≈ (avg_val_20d * 1000) / DAILY_TURNOVER
# Verified: HPG avg_val_20d=650M → est. mktcap = 650M*1000/0.005 = 130T VND ≈ actual 117T.
DAILY_TURNOVER = 0.005   # 0.5% of market cap traded per day


def load_price_features() -> pd.DataFrame:
    """
    For each stock: latest close, 52W high/low, drawdown, 6M & 12M return,
    and an estimated market cap (used to compute PE/PB).
    """
    rows = []
    for f in glob.glob(os.path.join(DATA_DIR, "*.parquet")):
        sym = os.path.basename(f).replace(".parquet", "").upper()
        try:
            df = pd.read_parquet(f, columns=["time", "close", "value"])
            df["date"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.dropna(subset=["date", "close"]).sort_values("date")
            df = df[df["close"] > 0]
            if len(df) < 30:
                continue
            latest      = df.iloc[-1]
            close       = latest["close"]            # in 1,000 VND units
            avg_val_20d = df["value"].iloc[-20:].mean()

            # Estimated market cap in actual VND
            # close * 1000 = actual price per share in VND
            # avg_val_20d * 1000 = daily traded VND
            mktcap_est_vnd = (avg_val_20d * 1000) / DAILY_TURNOVER

            # 52-week window
            w52 = df[df["date"] >= df["date"].iloc[-1] - pd.Timedelta(days=365)]
            hi52 = w52["close"].max()
            lo52 = w52["close"].min()
            drawdown_from_hi = (close - hi52) / hi52   # negative = below peak

            # Returns
            def ret_n(days):
                cutoff = df["date"].iloc[-1] - pd.Timedelta(days=days)
                past = df[df["date"] <= cutoff]
                if past.empty: return np.nan
                return (close - past["close"].iloc[-1]) / past["close"].iloc[-1]

            rows.append({
                "symbol":        sym,
                "close":         close,
                "hi52":          hi52,
                "lo52":          lo52,
                "drawdown":      drawdown_from_hi,
                "pct_above_lo":  (close - lo52) / lo52,
                "ret_3m":        ret_n(90),
                "ret_6m":        ret_n(180),
                "ret_12m":       ret_n(365),
                "avg_val_20d":   avg_val_20d,
                "mktcap_est_vnd": mktcap_est_vnd,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


# ── Load quarterly financials ─────────────────────────────────────────────────

def load_quarterly() -> pd.DataFrame:
    """
    Load all quarterly (quarter in 1-4) rows.
    Keep last 8 quarters per symbol.
    """
    rows = []
    for f in glob.glob(os.path.join(FA_DIR, "*.parquet")):
        try:
            df = pd.read_parquet(f)
            df = df[df["quarter"].isin([1, 2, 3, 4])].copy()
            if df.empty:
                continue
            df["symbol"] = df["symbol"].astype(str).str.upper()
            for c in ["revenue", "net_profit", "gross_profit", "operating_profit",
                      "roe", "roa", "net_margin", "gross_margin", "op_margin",
                      "pe", "pb", "ps", "debt_equity", "ocf", "equity",
                      "eps_basic", "current_ratio", "interest_coverage"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                else:
                    df[c] = np.nan
            df["period"] = df["year"].astype(str) + "Q" + df["quarter"].astype(str)
            df = df.sort_values(["symbol", "year", "quarter"])
            rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ── Fundamental feature engineering ──────────────────────────────────────────

def fundamental_features(qdf: pd.DataFrame,
                          price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per symbol: compute earnings momentum, valuation, quality.
    PE and PB are estimated from market cap proxy (avg daily value / turnover rate).
    """
    # Build price lookup: symbol → (close_1000vnd, mktcap_est_vnd, avg_val_20d)
    px = price_df.set_index("symbol")[["close", "mktcap_est_vnd", "avg_val_20d"]].to_dict("index")

    out = []
    for sym, g in qdf.groupby("symbol"):
        g = g.sort_values(["year", "quarter"]).tail(8)
        if len(g) < 2:
            continue

        latest  = g.iloc[-1]
        prev_q  = g.iloc[-2]
        sector  = str(latest.get("sector") or latest.get("industry") or "")

        rev_q  = latest["revenue"]
        np_q   = latest["net_profit"]
        roe    = latest["roe"]
        equity = latest["equity"]
        ocf    = latest["ocf"]
        de     = latest["debt_equity"]
        net_m  = latest["net_margin"]
        gross_m= latest["gross_margin"]
        cr     = latest["current_ratio"]

        # ── PE and PB via estimated market cap ───────────────────────────────
        pe_est = pb_est = ps_est = np.nan
        if sym in px:
            mktcap = px[sym]["mktcap_est_vnd"]
            # TTM net profit = sum last 4 quarters
            ttm_np = g["net_profit"].tail(4).sum()
            ttm_rev= g["revenue"].tail(4).sum()
            if pd.notna(ttm_np) and ttm_np > 0 and mktcap > 0:
                pe_est = mktcap / ttm_np
            if pd.notna(equity) and equity > 0 and mktcap > 0:
                pb_est = mktcap / equity
            if pd.notna(ttm_rev) and ttm_rev > 0 and mktcap > 0:
                ps_est = mktcap / ttm_rev

        # ── YoY growth (same quarter last year) ──────────────────────────────
        same_q_ly = g[(g["year"] == latest["year"] - 1) &
                      (g["quarter"] == latest["quarter"])]
        rev_yoy = np_yoy = np.nan
        if not same_q_ly.empty:
            base_rev = same_q_ly["revenue"].iloc[0]
            base_np  = same_q_ly["net_profit"].iloc[0]
            if pd.notna(base_rev) and base_rev != 0:
                rev_yoy = (rev_q - base_rev) / abs(base_rev)
            if pd.notna(base_np) and base_np != 0:
                np_yoy  = (np_q  - base_np)  / abs(base_np)

        # ── QoQ growth ────────────────────────────────────────────────────────
        rev_qoq = np_qoq = np.nan
        if pd.notna(prev_q["revenue"]) and prev_q["revenue"] != 0:
            rev_qoq = (rev_q - prev_q["revenue"]) / abs(prev_q["revenue"])
        if pd.notna(prev_q["net_profit"]) and prev_q["net_profit"] != 0:
            np_qoq  = (np_q  - prev_q["net_profit"]) / abs(prev_q["net_profit"])

        # ── Earnings acceleration: YoY improvement across last 3 quarters ────
        # Use 3 recent quarters → 3 YoY comparisons → 2 consecutive diffs → max score = 2
        accel_score = 0
        if len(g) >= 4:
            recent3 = g.tail(3)   # Q-2, Q-1, Q0
            yoy_vals = []
            for _, row in recent3.iterrows():
                ly = g[(g["year"] == row["year"] - 1) & (g["quarter"] == row["quarter"])]
                if not ly.empty and pd.notna(ly["net_profit"].iloc[0]) and ly["net_profit"].iloc[0] != 0:
                    yoy_vals.append((row["net_profit"] - ly["net_profit"].iloc[0])
                                    / abs(ly["net_profit"].iloc[0]))
            if len(yoy_vals) >= 2:
                diffs = [yoy_vals[i+1] - yoy_vals[i] for i in range(len(yoy_vals)-1)]
                accel_score = sum(1 for d in diffs if d > 0)  # max = 2

        # ── Margin trend (vs 4 quarters ago) ─────────────────────────────────
        margin_trend = np.nan
        if len(g) >= 5:
            old_gm = g.iloc[-5]["gross_margin"]
            if pd.notna(old_gm) and pd.notna(gross_m):
                margin_trend = gross_m - old_gm

        # ── PE upside: current vs own historical median PE (estimated) ────────
        # Build historical PE estimates if we have enough price history embedded
        pe_upside = np.nan
        if pd.notna(pe_est) and pe_est > 0 and pe_est < 150:
            # Use the 3-year trailing average ROE as proxy for "normal" earnings power
            old_roe = g["roe"].dropna()
            if len(old_roe) >= 4:
                roe_median = old_roe.median()
                if pd.notna(roe_median) and roe_median > 0 and pd.notna(roe):
                    # If current ROE is recovering toward historical median,
                    # earnings will grow by this factor → PE will compress by same
                    earnings_recovery = roe_median / roe if roe > 0 else np.nan
                    if pd.notna(earnings_recovery) and earnings_recovery > 1:
                        pe_upside = earnings_recovery - 1  # % upside from earnings normalisation

        out.append({
            "symbol":       sym,
            "sector":       sector,
            "latest_q":     latest["period"],
            "pe_est":       round(pe_est,  1) if pd.notna(pe_est)  else np.nan,
            "pb_est":       round(pb_est,  2) if pd.notna(pb_est)  else np.nan,
            "ps_est":       round(ps_est,  2) if pd.notna(ps_est)  else np.nan,
            "roe":          roe,
            "net_margin":   net_m,
            "gross_margin": gross_m,
            "debt_equity":  de,
            "current_ratio":cr,
            "ocf":          ocf,
            "rev_yoy":      rev_yoy,
            "np_yoy":       np_yoy,
            "rev_qoq":      rev_qoq,
            "np_qoq":       np_qoq,
            "accel_score":  accel_score,
            "margin_trend": margin_trend,
            "pe_upside":    pe_upside,
            "rev_latest_B": rev_q / 1e9 if pd.notna(rev_q) else np.nan,
            "np_latest_B":  np_q  / 1e9 if pd.notna(np_q)  else np.nan,
            "ttm_np_B":     g["net_profit"].tail(4).sum() / 1e9,
        })

    return pd.DataFrame(out)


# ── Composite score ───────────────────────────────────────────────────────────

def build_score(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()

    def norm(s, invert=False):
        """Min-max normalise a series to [0,1], optionally invert (lower=better)."""
        s = pd.to_numeric(s, errors="coerce")
        mn, mx = s.quantile(0.05), s.quantile(0.95)
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        n = (s - mn) / (mx - mn)
        n = n.clip(0, 1)
        return 1 - n if invert else n

    # ── Value component (40%) ─────────────────────────────────────────────────
    df["s_drawdown"] = norm(-df["drawdown"])               # more beaten-down = higher score
    df["s_pb"]       = norm(df["pb_est"], invert=True)     # lower PB = better
    df["s_pe"]       = norm(df["pe_est"], invert=True)
    df["s_pe"]       = df["s_pe"].where(df["pe_est"] > 0)  # ignore loss-making
    df["s_pe_upside"]= norm(df["pe_upside"])               # earnings recovery upside

    value_cols = ["s_drawdown", "s_pb", "s_pe", "s_pe_upside"]
    df["value_score"] = df[value_cols].mean(axis=1, skipna=True)

    # ── Earnings momentum component (40%) ─────────────────────────────────────
    df["s_rev_yoy"]   = norm(df["rev_yoy"])
    df["s_np_yoy"]    = norm(df["np_yoy"])
    df["s_accel"]     = norm(df["accel_score"])
    df["s_margin"]    = norm(df["margin_trend"])

    earn_cols = ["s_rev_yoy", "s_np_yoy", "s_accel", "s_margin"]
    df["earn_score"] = df[earn_cols].mean(axis=1, skipna=True)

    # ── Quality guardrail (20%) ───────────────────────────────────────────────
    df["s_roe"]  = norm(df["roe"])
    df["s_ocf"]  = norm(df["ocf"])
    df["quality_score"] = df[["s_roe", "s_ocf"]].mean(axis=1, skipna=True)

    # ── Hard filters (remove value traps) ────────────────────────────────────
    # Must be profitable in latest quarter
    df = df[df["np_latest_B"].fillna(-999) > 0]
    # Debt/equity not extreme
    df = df[df["debt_equity"].fillna(999) < 3.0]
    # Revenue positive
    df = df[df["rev_latest_B"].fillna(0) > 0]

    # ── Final score ───────────────────────────────────────────────────────────
    df["score"] = (df["value_score"]   * 0.40 +
                   df["earn_score"]    * 0.40 +
                   df["quality_score"] * 0.20)

    return df.sort_values("score", ascending=False)


# ── Detail view: quarterly history for one stock ──────────────────────────────

def show_detail(symbol: str, qdf: pd.DataFrame, price_df: pd.DataFrame):
    sym = symbol.upper()
    q   = qdf[qdf["symbol"] == sym].sort_values(["year", "quarter"]).tail(12)
    if q.empty:
        print(f"No quarterly data for {sym}")
        return

    px = price_df[price_df["symbol"] == sym]

    print(f"\n{'='*80}")
    print(f"FUNDAMENTAL DETAIL: {sym}")
    if not px.empty:
        r = px.iloc[0]
        print(f"Price: {r['close']:,.0f}  |  52W high: {r['hi52']:,.0f}  "
              f"({r['drawdown']:+.1%} from peak)  |  "
              f"6M: {r['ret_6m']:+.1%}  12M: {r['ret_12m']:+.1%}")
    print(f"{'='*80}")
    print(f"{'Period':<8} {'Rev (B)':>9} {'Rev YoY':>8} {'Profit (B)':>11} {'NP YoY':>8} "
          f"{'Margin%':>8} {'ROE%':>7} {'PE':>6} {'PB':>6}")
    print("-" * 80)

    prev_rev = prev_np = None
    for _, row in q.iterrows():
        period  = row["period"]
        rev_b   = row["revenue"]  / 1e9 if pd.notna(row["revenue"])    else np.nan
        np_b    = row["net_profit"]/ 1e9 if pd.notna(row["net_profit"]) else np.nan

        # YoY vs same quarter prior year
        ly = q[(q["year"] == row["year"]-1) & (q["quarter"] == row["quarter"])]
        rev_yoy = np_yoy = np.nan
        if not ly.empty:
            base_r = ly["revenue"].iloc[0]
            base_n = ly["net_profit"].iloc[0]
            if pd.notna(base_r) and base_r != 0:
                rev_yoy = (row["revenue"] - base_r) / abs(base_r)
            if pd.notna(base_n) and base_n != 0:
                np_yoy  = (row["net_profit"] - base_n) / abs(base_n)

        gm  = row["gross_margin"] if pd.notna(row["gross_margin"]) else np.nan
        roe = row["roe"]          if pd.notna(row["roe"])          else np.nan
        pe  = row["pe"]           if pd.notna(row["pe"])           else np.nan
        pb  = row["pb"]           if pd.notna(row["pb"])           else np.nan

        rev_str = f"{rev_b:>9.1f}" if pd.notna(rev_b) else f"{'N/A':>9}"
        np_str  = f"{np_b:>11.2f}" if pd.notna(np_b)  else f"{'N/A':>11}"
        ry_str  = f"{rev_yoy:>+8.1%}" if pd.notna(rev_yoy) else f"{'N/A':>8}"
        ny_str  = f"{np_yoy:>+8.1%}"  if pd.notna(np_yoy)  else f"{'N/A':>8}"
        gm_str  = f"{gm:>8.1f}"    if pd.notna(gm)  else f"{'N/A':>8}"
        roe_str = f"{roe:>7.2f}"   if pd.notna(roe) else f"{'N/A':>7}"
        pe_str  = f"{pe:>6.1f}"    if pd.notna(pe)  else f"{'N/A':>6}"
        pb_str  = f"{pb:>6.2f}"    if pd.notna(pb)  else f"{'N/A':>6}"

        print(f"{period:<8} {rev_str} {ry_str} {np_str} {ny_str} "
              f"{gm_str} {roe_str} {pe_str} {pb_str}")

    print(f"{'='*80}")

    # ROE recovery upside (since PE not available)
    roe_hist = q["roe"].dropna()
    if len(roe_hist) >= 4:
        roe_med = roe_hist.median()
        roe_now = q.iloc[-1]["roe"]
        if pd.notna(roe_now) and roe_now > 0 and roe_med > roe_now:
            recovery = roe_med / roe_now - 1
            print(f"\nROE now: {roe_now:.2f}%  |  Historical median ROE: {roe_med:.2f}%  "
                  f"|  Earnings recovery upside: {recovery:+.1%}")

    # Earnings trend narrative
    recent = q.tail(4)
    np_vals = recent["net_profit"].values
    if all(pd.notna(v) for v in np_vals[-3:]):
        trend = "IMPROVING" if np_vals[-1] > np_vals[-2] > np_vals[-3] else \
                "MIXED" if np_vals[-1] > np_vals[-3] else "DECLINING"
        print(f"Profit trend (last 3Q): {trend}  "
              f"({np_vals[-3]/1e9:.2f}B → {np_vals[-2]/1e9:.2f}B → {np_vals[-1]/1e9:.2f}B)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top",          type=int,   default=15)
    ap.add_argument("--min-drawdown", type=float, default=0.05,
                    help="Minimum decline from 52W high, e.g. 0.20 = at least 20%% below peak")
    ap.add_argument("--min-liq",      type=float, default=50_000_000,
                    help="Min 20-day avg daily value (VND). Default 50M.")
    ap.add_argument("--sector",       default=None)
    ap.add_argument("--detail",       default=None, help="Show full quarterly history for SYMBOL")
    ap.add_argument("--no-upcom",     action="store_true",
                    help="Exclude UPCOM-listed stocks (keep HOSE + HNX only, recommended)")
    args = ap.parse_args()

    # Exchange universe filter
    exch_map = load_exchange_map()
    upcom_set = {t for t, e in exch_map.items() if e == "UPCOM"} if args.no_upcom else set()

    print("Loading price data...")
    price_df = load_price_features()
    if upcom_set:
        before = len(price_df)
        price_df = price_df[~price_df["symbol"].isin(upcom_set)].copy()
        print(f"  Exchange filter: {before} → {len(price_df)} stocks (dropped {before - len(price_df)} UPCOM)")

    print("Loading quarterly financials...")
    qdf = load_quarterly()
    if upcom_set:
        qdf = qdf[~qdf["symbol"].isin(upcom_set)].copy()

    if args.detail:
        show_detail(args.detail, qdf, price_df)
        return

    print("Computing fundamental features...")
    fund_df = fundamental_features(qdf, price_df)

    print("Merging and scoring...")
    merged = price_df.merge(fund_df, on="symbol", how="inner")

    # Apply filters
    merged = merged[merged["drawdown"] <= -args.min_drawdown]     # beaten down
    merged = merged[merged["avg_val_20d"] >= args.min_liq]        # liquid enough
    if args.sector:
        merged = merged[merged["sector"].str.contains(args.sector, case=False, na=False)]

    scored = build_score(merged)
    top    = scored.head(args.top)

    if top.empty:
        print("No stocks match the filters. Try lowering --min-drawdown or --min-liq.")
        return

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"VALUE + EARNINGS MOMENTUM SCREENER  —  top {args.top} stocks")
    print(f"Filter: >= {args.min_drawdown:.0%} off 52W high | liq >= {args.min_liq/1e6:.0f}M VND/day")
    print(f"{'='*100}")
    print(f"{'#':>3} {'Sym':>5} {'Sector':<25} {'Score':>6} "
          f"{'ΔPeak':>7} {'PE~':>6} {'PB~':>5} "
          f"{'Rev YoY':>8} {'NP YoY':>8} {'Accel':>6} "
          f"{'TTM NP':>8} {'Recov↑':>7} {'Liq/d':>7}")
    print("-" * 103)

    for rank, (_, r) in enumerate(top.iterrows(), 1):
        pe_str    = f"{r['pe_est']:.1f}x"    if pd.notna(r['pe_est'])    else "N/A"
        pb_str    = f"{r['pb_est']:.2f}x"    if pd.notna(r['pb_est'])    else "N/A"
        ryr_str   = f"{r['rev_yoy']:+.0%}"   if pd.notna(r['rev_yoy'])   else "N/A"
        nyoy_str  = f"{r['np_yoy']:+.0%}"    if pd.notna(r['np_yoy'])    else "N/A"
        accel_str = f"{int(r['accel_score'])}/2" if pd.notna(r['accel_score']) else "N/A"
        up_str    = f"{r['pe_upside']:+.0%}" if pd.notna(r['pe_upside']) else "N/A"
        liq_str   = f"{r['avg_val_20d']/1e6:.0f}M"
        ttm_str   = f"{r['ttm_np_B']:.1f}B"  if pd.notna(r.get('ttm_np_B')) else "N/A"

        print(f"{rank:>3} {r['symbol']:>5} {str(r.get('sector','')):<25} "
              f"{r['score']:>6.3f} "
              f"{r['drawdown']:>+7.1%} {pe_str:>6} {pb_str:>5} "
              f"{ryr_str:>8} {nyoy_str:>8} {accel_str:>6} "
              f"{ttm_str:>8} {up_str:>7} {liq_str:>7}")

    print(f"{'='*103}")
    print("Score : value(40%) + earnings momentum(40%) + quality(20%)")
    print("PE~/PB~: estimated from avg daily value / 0.5% turnover (±30% accuracy)")
    print("Accel : 0/2-2/2 = quarters with accelerating YoY profit growth")
    print("Recov↑: % earnings upside if ROE reverts to own 3Y historical median")
    print(f"\nFor full quarterly history: python value_screener.py --detail SYMBOL")

    # Save
    out_path = os.path.join(OUT_DIR, "value_screener_latest.csv")
    save_cols = ["symbol", "sector", "latest_q", "score",
                 "close", "drawdown", "ret_6m", "ret_12m",
                 "pe_est", "pb_est", "ps_est", "pe_upside",
                 "rev_yoy", "np_yoy", "rev_qoq", "np_qoq",
                 "accel_score", "margin_trend",
                 "roe", "net_margin", "gross_margin", "debt_equity",
                 "ttm_np_B", "np_latest_B", "rev_latest_B", "avg_val_20d"]
    top[[c for c in save_cols if c in top.columns]].to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
