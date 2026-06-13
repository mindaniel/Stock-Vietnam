"""
value_backtest.py
Backtest for the value + earnings-momentum screener strategy.

Key design choices to avoid look-ahead bias:
  1. Filing lag: quarterly results assumed available 2 months after quarter end
       Q1 (Jan-Mar) → usable from June 1
       Q2 (Apr-Jun) → usable from September 1
       Q3 (Jul-Sep) → usable from December 1
       Q4 (Oct-Dec) → usable from April 1 (next year)
  2. Price features computed with data up to rebalance date only
  3. 52W high computed from rolling past 12 months of CLOSING prices
  4. Market cap estimated from avg daily value over past 20 trading days
     (converted to monthly avg daily value from the monthly panel)

Note on data density:
  Quarterly financials are sparse before 2023 (30-82 stocks).
  The strategy becomes meaningful from ~April 2024 (1000+ stocks).
  Results before that are shown but flagged.

Usage:
  python value_backtest.py
  python value_backtest.py --start 2023-01-01 --top 15 --cost-bps 30
  python value_backtest.py --no-regime
"""

import argparse, os, sys, glob, warnings
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data", "price")
FA_DIR      = os.path.join(DATA_DIR, "financials_fa")
OUT_DIR     = os.path.join(BASE_DIR, "results")
TICKER_CSV  = os.path.join(BASE_DIR, "ticker_sectors.csv")
os.makedirs(OUT_DIR, exist_ok=True)

DAILY_TURNOVER = 0.005   # 0.5% of market cap traded per day


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


def load_sector_map() -> dict:
    """Return {TICKER: industry} from ticker_sectors.csv. Empty dict if file missing."""
    if not os.path.exists(TICKER_CSV):
        return {}
    df = pd.read_csv(TICKER_CSV, usecols=lambda c: c in ("ticker", "industry"))
    if "industry" not in df.columns:
        return {}
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["industry"] = df["industry"].fillna("Unknown").astype(str).str.strip()
    return dict(zip(df["ticker"], df["industry"]))


# ── Filing lag ────────────────────────────────────────────────────────────────

def avail_date(year: int, quarter: int) -> pd.Timestamp:
    """When this quarter's results are safely available (2-month lag)."""
    if quarter == 1:   return pd.Timestamp(year,   6,  1)
    elif quarter == 2: return pd.Timestamp(year,   9,  1)
    elif quarter == 3: return pd.Timestamp(year,  12,  1)
    else:              return pd.Timestamp(year+1,  4,  1)


# ── Load quarterly fundamentals with availability dates ───────────────────────

def load_quarterly() -> pd.DataFrame:
    rows = []
    for f in glob.glob(os.path.join(FA_DIR, "*.parquet")):
        try:
            df = pd.read_parquet(f)
            if "quarter" not in df.columns:
                continue
            df = df[df["quarter"].isin([1, 2, 3, 4])].copy()
            if df.empty:
                continue
            df["symbol"] = df["symbol"].astype(str).str.upper()
            for c in ["revenue", "net_profit", "gross_profit", "equity",
                      "roe", "roa", "net_margin", "gross_margin",
                      "ocf", "debt_equity", "current_ratio"]:
                df[c] = pd.to_numeric(df.get(c), errors="coerce")
            rows.append(df[["symbol", "year", "quarter",
                            "revenue", "net_profit", "gross_profit", "equity",
                            "roe", "roa", "net_margin", "gross_margin",
                            "ocf", "debt_equity", "current_ratio"]])
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    q = pd.concat(rows, ignore_index=True)
    q["avail_date"] = q.apply(lambda r: avail_date(int(r["year"]), int(r["quarter"])), axis=1)
    return q.sort_values(["symbol", "avail_date"]).reset_index(drop=True)


# ── Pre-compute fundamental features per quarterly observation ────────────────

def build_quarterly_features(qdf: pd.DataFrame) -> pd.DataFrame:
    """
    For every (symbol, year, quarter) row compute:
      rev_yoy, np_yoy, accel_score, margin_trend, ttm_np, ttm_rev
    Returns one row per (symbol, avail_date) — the last available observation.
    """
    out = []
    for sym, g in qdf.groupby("symbol"):
        g = g.sort_values(["year", "quarter"]).reset_index(drop=True)
        for i in range(len(g)):
            row = g.iloc[i]
            yr, qt = int(row["year"]), int(row["quarter"])

            # YoY vs same quarter last year
            ly = g[(g["year"] == yr - 1) & (g["quarter"] == qt)]
            rev_yoy = np_yoy = np.nan
            if not ly.empty:
                base_r = ly["revenue"].iloc[0]
                base_n = ly["net_profit"].iloc[0]
                if pd.notna(base_r) and base_r != 0:
                    rev_yoy = (row["revenue"] - base_r) / abs(base_r)
                if pd.notna(base_n) and base_n != 0:
                    np_yoy  = (row["net_profit"] - base_n) / abs(base_n)

            # TTM (trailing 12M) = sum of this + prior 3 quarters
            past4 = g.iloc[max(0, i-3): i+1]
            ttm_np  = past4["net_profit"].sum() if past4["net_profit"].notna().any() else np.nan
            ttm_rev = past4["revenue"].sum()     if past4["revenue"].notna().any()    else np.nan

            # Earnings acceleration: YoY improvement across last 2 consecutive quarters
            accel_score = 0
            if i >= 2:
                yoy_vals = []
                for j in [i-2, i-1, i]:
                    r2 = g.iloc[j]
                    ly2 = g[(g["year"] == int(r2["year"])-1) & (g["quarter"] == int(r2["quarter"]))]
                    if not ly2.empty and pd.notna(ly2["net_profit"].iloc[0]) and ly2["net_profit"].iloc[0] != 0:
                        yoy_vals.append((r2["net_profit"] - ly2["net_profit"].iloc[0])
                                        / abs(ly2["net_profit"].iloc[0]))
                if len(yoy_vals) >= 2:
                    diffs = [yoy_vals[k+1] - yoy_vals[k] for k in range(len(yoy_vals)-1)]
                    accel_score = sum(1 for d in diffs if d > 0)  # 0-2

            # Margin trend vs 4 quarters ago
            margin_trend = np.nan
            if i >= 4:
                old_gm = g.iloc[i-4]["gross_margin"]
                new_gm = row["gross_margin"]
                if pd.notna(old_gm) and pd.notna(new_gm):
                    margin_trend = new_gm - old_gm

            out.append({
                "symbol":       sym,
                "year":         yr,
                "quarter":      qt,
                "avail_date":   row["avail_date"],
                "rev_yoy":      rev_yoy,
                "np_yoy":       np_yoy,
                "accel_score":  accel_score,
                "margin_trend": margin_trend,
                "ttm_np":       ttm_np,
                "ttm_rev":      ttm_rev,
                "equity":       row["equity"],
                "roe":          row["roe"],
                "ocf":          row["ocf"],
                "net_margin":   row["net_margin"],
                "gross_margin": row["gross_margin"],
                "debt_equity":  row["debt_equity"],
            })

    return pd.DataFrame(out).sort_values(["symbol", "avail_date"])


# ── Load monthly price panel ──────────────────────────────────────────────────

def load_monthly_prices() -> pd.DataFrame:
    rows = []
    for f in glob.glob(os.path.join(DATA_DIR, "*.parquet")):
        sym = os.path.basename(f).replace(".parquet", "").upper()
        try:
            df = pd.read_parquet(f, columns=["time", "close", "value"])
            df["date"]  = pd.to_datetime(df["time"], errors="coerce")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["date", "close"]).sort_values("date")
            df = df[df["close"] > 0]
            if len(df) < 30:
                continue
            # Monthly: last close of month, mean daily value
            m = (df.groupby(pd.Grouper(key="date", freq="ME"))
                   .agg(close=("close", "last"), avg_val=("value", "mean"))
                   .reset_index())
            m["symbol"] = sym
            rows.append(m)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    # Monthly return
    out["ret"] = out.groupby("symbol")["close"].pct_change(fill_method=None)
    return out.dropna(subset=["ret"])


# ── Rolling 52W high from monthly panel ──────────────────────────────────────

def add_rolling_features(monthly: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly.sort_values(["symbol", "date"])
    # 52W high = max close over past 12 months (shift 1 to exclude current)
    monthly["hi52"] = (monthly.groupby("symbol")["close"]
                              .transform(lambda s: s.shift(1).rolling(12, min_periods=3).max()))
    monthly["drawdown"] = (monthly["close"] - monthly["hi52"]) / monthly["hi52"]
    # Estimated market cap (in actual VND) = avg_val * 1000 / DAILY_TURNOVER
    monthly["mktcap_est"] = monthly["avg_val"] * 1000 / DAILY_TURNOVER
    # 3-month trailing return (shift 1 so no look-ahead: uses close 3M ago vs 1M ago)
    monthly["ret_3m"] = (monthly.groupby("symbol")["close"]
                                .transform(lambda s: s.shift(1).pct_change(3, fill_method=None)))
    return monthly


# ── Regime filter ─────────────────────────────────────────────────────────────

def load_regime() -> pd.DataFrame:
    vn = pd.read_csv(os.path.join(BASE_DIR, "VNINDEX.csv"))
    vn["date"]  = pd.to_datetime(vn["date"])
    vn["close"] = pd.to_numeric(vn["close"], errors="coerce")
    vn = vn.dropna(subset=["date", "close"]).sort_values("date")
    m  = vn.set_index("date")["close"].resample("ME").last().to_frame("close")
    m["ma10"]    = m["close"].rolling(10, min_periods=10).mean()
    m["risk_on"] = (m["close"] >= m["ma10"]).astype(int)
    m["mkt_ret"] = m["close"].pct_change(fill_method=None)
    return m.reset_index()


# ── Composite score ───────────────────────────────────────────────────────────

def zscore_col(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    m, sd = s.mean(), s.std()
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - m) / sd


def composite_score(g: pd.DataFrame) -> pd.Series:
    """
    Cross-sectional composite score for a single date's cross-section.
    Value(40%) + Earnings momentum(40%) + Quality(20%).
    """
    # ── Value ────────────────────────────────────────────────────────────────
    s_draw = zscore_col(-g["drawdown"])    # more beaten-down = higher

    pe = g["mktcap_est"] / g["ttm_np"].replace(0, np.nan)
    pe = pe.where((pe > 0) & (pe < 200))
    s_pe = zscore_col(-pe)                 # lower PE = higher score

    pb = g["mktcap_est"] / g["equity"].replace(0, np.nan)
    pb = pb.where((pb > 0) & (pb < 20))
    s_pb = zscore_col(-pb)                 # lower PB = higher score

    value = pd.concat([s_draw, s_pe, s_pb], axis=1).mean(axis=1, skipna=True)

    # ── Earnings momentum ────────────────────────────────────────────────────
    s_rev_yoy = zscore_col(g["rev_yoy"].clip(-2, 10))
    s_np_yoy  = zscore_col(g["np_yoy"].clip(-2, 10))
    s_accel   = zscore_col(g["accel_score"])
    s_margin  = zscore_col(g["margin_trend"])

    earn = pd.concat([s_rev_yoy, s_np_yoy, s_accel, s_margin],
                     axis=1).mean(axis=1, skipna=True)

    # ── Quality ──────────────────────────────────────────────────────────────
    s_roe = zscore_col(g["roe"].clip(-50, 100))
    s_ocf = zscore_col(g["ocf"])

    qual = pd.concat([s_roe, s_ocf], axis=1).mean(axis=1, skipna=True)

    return 0.40 * value + 0.40 * earn + 0.20 * qual


# ── Main backtest ─────────────────────────────────────────────────────────────

def _pick_portfolio(monthly_dt, qfeat, dt, top_n, min_liq,
                    min_np_yoy, min_mom):
    """
    Score the cross-section at date dt and return (picks_df, universe_size).
    Returns (None, n) if not enough candidates.
    """
    px = monthly_dt.copy()
    px = px[px["avg_val"] >= min_liq]
    px = px.dropna(subset=["ret", "drawdown", "mktcap_est"])
    if len(px) < 10:
        return None, 0

    avail = qfeat[qfeat["avail_date"] <= dt]
    if avail.empty:
        return None, 0
    latest_fund = (avail.sort_values("avail_date")
                        .groupby("symbol").last().reset_index())

    g = px.merge(latest_fund, on="symbol", how="inner")
    g = g.dropna(subset=["ttm_np", "equity"])

    # Hard filters
    g = g[g["ttm_np"] > 0]
    if min_np_yoy is not None:
        g = g[g["np_yoy"].isna() | (g["np_yoy"] >= min_np_yoy)]
    g = g[g["debt_equity"].fillna(999) < 3.0]
    if min_mom > 0 and "ret_3m" in g.columns:
        g = g[g["ret_3m"].fillna(-1) >= min_mom]

    universe_n = len(g)
    if universe_n < 5:
        return None, universe_n

    g = g.copy()
    g["score"] = composite_score(g)
    g = g.dropna(subset=["score"])
    if len(g) < 5:
        return None, universe_n

    picks = g.nlargest(top_n, "score")
    return picks, universe_n


def run_backtest(monthly: pd.DataFrame,
                 qfeat:   pd.DataFrame,
                 regime:  pd.DataFrame | None,
                 top_n:   int,
                 min_liq: float,
                 cost_bps: float,
                 start:   str,
                 min_mom:    float = 0.0,
                 min_np_yoy: float | None = None,
                 rebal_months: int = 1) -> pd.DataFrame:
    """
    rebal_months=1 → rebalance every month (original behaviour)
    rebal_months=3 → rebalance quarterly, hold positions in between
      Natural VN quarterly rebalance dates align with filing lags:
        April  (Q4 data fresh), June (Q1), September (Q2), December (Q3)
    """
    # Months that trigger a rebalance when rebal_months=3
    QUARTERLY_MONTHS = {4, 6, 9, 12}   # Apr=Q4 fresh, Jun=Q1 fresh, Sep=Q2 fresh, Dec=Q3 fresh

    dates = sorted(monthly["date"].unique())
    dates = [d for d in dates if d >= pd.Timestamp(start)]

    prev_holds  = {}   # {symbol: weight}
    next_rebal  = None # will rebalance on first eligible date
    rets = []

    # Pre-index monthly by date for fast lookup
    monthly_by_date = {d: grp for d, grp in monthly.groupby("date")}

    for dt in dates:
        # ── Regime gate ──────────────────────────────────────────────────────
        if regime is not None:
            ro = regime.loc[regime["date"] == dt, "risk_on"]
            if len(ro) == 0 or int(ro.iloc[0]) == 0:
                prev_holds = {}
                next_rebal = None   # force rebalance on next regime-on month
                rets.append({"date": dt, "ret": 0.0, "n": 0, "universe": 0,
                             "turnover": 0.0, "regime": "OFF", "rebal": False})
                continue

        px_dt = monthly_by_date.get(dt, pd.DataFrame())

        # ── Decide whether to rebalance ──────────────────────────────────────
        is_rebal_month = (
            rebal_months == 1
            or next_rebal is None          # first active month after cash
            or dt >= next_rebal            # scheduled rebalance date reached
            or (rebal_months == 3 and dt.month in QUARTERLY_MONTHS)
        )

        if is_rebal_month:
            picks, uni = _pick_portfolio(px_dt, qfeat, dt, top_n, min_liq,
                                         min_np_yoy, min_mom)
            if picks is None or len(picks) == 0:
                # Not enough candidates — stay in cash this month
                prev_holds = {}
                rets.append({"date": dt, "ret": 0.0, "n": 0,
                             "universe": uni, "turnover": 0.0,
                             "regime": "ON", "rebal": True})
                continue

            w_new    = {s: 1/len(picks) for s in picks["symbol"]}
            gross    = picks.set_index("symbol")["ret"].mean()
            all_s    = set(prev_holds) | set(w_new)
            turnover = sum(abs(w_new.get(s, 0) - prev_holds.get(s, 0)) for s in all_s)
            net      = gross - turnover * cost_bps / 10_000

            prev_holds = w_new
            # Schedule next rebalance
            next_rebal = dt + pd.DateOffset(months=rebal_months)

            rets.append({
                "date": dt, "ret": net, "gross": gross,
                "n": len(picks), "universe": uni,
                "turnover": round(turnover, 3),
                "regime": "ON", "rebal": True,
                "top5": ", ".join(picks.nlargest(5, "score")["symbol"].tolist()),
            })

        else:
            # ── Hold month: apply existing weights to current returns ─────────
            if not prev_holds:
                rets.append({"date": dt, "ret": 0.0, "n": 0, "universe": 0,
                             "turnover": 0.0, "regime": "ON", "rebal": False})
                continue

            px_ret = px_dt.set_index("symbol")["ret"] if not px_dt.empty else pd.Series(dtype=float)
            gross  = sum(prev_holds.get(s, 0) * px_ret.get(s, 0.0)
                         for s in prev_holds)
            net    = gross   # no turnover cost on hold months

            rets.append({
                "date": dt, "ret": net, "gross": gross,
                "n": len(prev_holds), "universe": 0,
                "turnover": 0.0, "regime": "ON", "rebal": False,
                "top5": ", ".join(list(prev_holds.keys())[:5]),
            })

    return pd.DataFrame(rets).sort_values("date")


# ── Perf stats ────────────────────────────────────────────────────────────────

def perf(r: pd.Series, label: str) -> dict:
    r = r.dropna()
    if len(r) == 0: return {}
    eq = (1 + r).cumprod()
    yrs  = len(r) / 12
    cagr = eq.iloc[-1] ** (1/yrs) - 1 if yrs > 0 else np.nan
    vol  = r.std() * np.sqrt(12)
    sh   = (r.mean() * 12) / vol if vol > 0 else np.nan
    dd   = eq / eq.cummax() - 1
    cal  = cagr / abs(dd.min()) if dd.min() != 0 else np.nan
    return dict(label=label, months=len(r), cagr=cagr, vol=vol,
                sharpe=sh, maxdd=dd.min(), calmar=cal)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",    default="2023-01-01")
    ap.add_argument("--top",      type=int,   default=15)
    ap.add_argument("--min-liq",  type=float, default=50_000_000)
    ap.add_argument("--cost-bps", type=float, default=30.0,
                    help="One-way transaction cost bps (default 30 — wider for small/mid caps)")
    ap.add_argument("--no-regime",      action="store_true")
    ap.add_argument("--rebal-months",   type=int, default=1,
                    help="Rebalance every N months (1=monthly, 3=quarterly). "
                         "Quarterly aligns with VN filing lag: Apr/Jun/Sep/Dec.")
    ap.add_argument("--no-upcom",       action="store_true",
                    help="Exclude UPCOM-listed stocks (keep HOSE + HNX only, recommended)")
    ap.add_argument("--no-realestate",  action="store_true",
                    help="Exclude Real Estate sector (structural falling knives, recommended)")
    ap.add_argument("--min-mom",        type=float, default=0.0,
                    help="Min 3-month price return required (e.g. 0.0 = any, 0.05 = +5%%). Filters falling knives.")
    ap.add_argument("--min-np-yoy",     type=float, default=None,
                    help="Min YoY net profit growth required (e.g. -0.20 = allow up to -20%%, 0.0 = must grow). "
                         "Prevents buying companies in structural decline. Default: no filter.")
    args = ap.parse_args()

    # Exchange universe filter
    exch_map = load_exchange_map()
    upcom_set = {t for t, e in exch_map.items() if e == "UPCOM"} if args.no_upcom else set()
    if upcom_set:
        print(f"  Exchange filter active: excluding {len(upcom_set)} UPCOM stocks (HOSE+HNX only)")

    # Sector exclusion
    sector_map = load_sector_map()
    RE_KEYWORDS = {"real estate", "realestate", "bất động sản"}
    realestate_set = (
        {t for t, s in sector_map.items() if any(k in s.lower() for k in RE_KEYWORDS)}
        if args.no_realestate else set()
    )
    if realestate_set:
        print(f"  Sector filter: excluding {len(realestate_set)} Real Estate stocks")

    print("Loading quarterly fundamentals...")
    qdf   = load_quarterly()
    if upcom_set:
        qdf = qdf[~qdf["symbol"].isin(upcom_set)].copy()
    if realestate_set:
        qdf = qdf[~qdf["symbol"].isin(realestate_set)].copy()
    print(f"  {len(qdf):,} quarterly rows, "
          f"{qdf['symbol'].nunique()} symbols, "
          f"years {qdf['year'].min()}–{qdf['year'].max()}")

    print("Building fundamental features (YoY, TTM, accel)...")
    qfeat = build_quarterly_features(qdf)
    print(f"  {len(qfeat):,} feature rows")

    print("Loading monthly price panel...")
    monthly = load_monthly_prices()
    if upcom_set:
        monthly = monthly[~monthly["symbol"].isin(upcom_set)].copy()
    if realestate_set:
        monthly = monthly[~monthly["symbol"].isin(realestate_set)].copy()
    print(f"  {len(monthly):,} rows, {monthly['symbol'].nunique()} symbols")

    monthly = add_rolling_features(monthly)

    regime = None if args.no_regime else load_regime()

    # Data density warning
    density = (qfeat[qfeat["avail_date"] >= pd.Timestamp(args.start)]
               .groupby("avail_date")["symbol"].nunique())
    print(f"\nUniverse density (stocks with fresh fundamental data):")
    for dt, n in density.resample("QE").max().items():
        flag = "  << sparse" if n < 100 else ""
        print(f"  {dt.date()}: {n} stocks{flag}")

    print(f"\nRunning backtest (top={args.top}, cost={args.cost_bps}bps, "
          f"regime={'OFF' if args.no_regime else 'ON'})...")

    result = run_backtest(monthly, qfeat, regime,
                          args.top, args.min_liq, args.cost_bps, args.start,
                          min_mom=args.min_mom, min_np_yoy=args.min_np_yoy,
                          rebal_months=args.rebal_months)

    if result.empty or result["ret"].abs().sum() == 0:
        print("No trades generated. Try --start 2024-01-01 or lower --min-liq.")
        return

    vni = load_regime()
    vni = vni[vni["date"] >= pd.Timestamp(args.start)].dropna(subset=["mkt_ret"])

    # ── Stats ─────────────────────────────────────────────────────────────────
    active = result[result["n"] > 0]
    print("\n" + "=" * 65)
    print(f"VALUE + EARNINGS BACKTEST  {args.start} – 2026")
    rebal_label = f"every {args.rebal_months}m" if args.rebal_months > 1 else "monthly"
    rebal_months_data = result[result.get("rebal", True) == True] if "rebal" in result.columns else active
    print(f"Universe size: {result['universe'].replace(0, np.nan).mean():.0f} avg stocks scored")
    print(f"Portfolio:     {active['n'].mean():.0f} avg holdings (top {args.top})")
    print(f"Rebalance:     {rebal_label}  ({(result['rebal']==True).sum()} rebal events)")
    print(f"Avg turnover:  {active[active['turnover']>0]['turnover'].mean():.1%} on rebal months")
    print("=" * 65)
    print(f"{'Strategy':<30} {'Mo':>4} {'CAGR':>7} {'Vol':>6} "
          f"{'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7}")
    print("-" * 65)

    strats = [("Value+Earn Strategy", result["ret"]),
              ("VNINDEX",             vni.set_index("date")["mkt_ret"])]
    all_stats = []
    for name, r in strats:
        s = perf(r.replace(0, np.nan).dropna(), name)
        if not s: continue
        all_stats.append(s)
        print(f"{name:<30} {s['months']:>4} {s['cagr']:>7.1%} {s['vol']:>6.1%} "
              f"{s['sharpe']:>7.2f} {s['maxdd']:>8.1%} {s['calmar']:>7.2f}")
    print("=" * 65)

    # ── Monthly detail ────────────────────────────────────────────────────────
    print(f"\nMonthly detail (active months only):")
    print(f"{'Date':<12} {'Ret':>7} {'N':>4} {'Universe':>9} {'Top holdings'}")
    print("-" * 75)
    for _, r in active.tail(20).iterrows():
        top5 = r.get("top5", "")
        print(f"{str(r['date'].date()):<12} {r['ret']:>+7.2%} "
              f"{int(r['n']):>4} {int(r['universe']):>9}  {top5}")

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    fig.suptitle(f"Value + Earnings Momentum Backtest  {args.start}–2026\n"
                 f"top={args.top} stocks | cost={args.cost_bps}bps | "
                 f"regime={'ON' if not args.no_regime else 'OFF'}",
                 fontsize=12, fontweight="bold")

    ax1 = axes[0]
    eq_strat = (1 + result.set_index("date")["ret"]).cumprod()
    eq_vni   = (1 + vni.set_index("date")["mkt_ret"]).cumprod()
    ax1.plot(eq_strat.index, eq_strat.values, label="Value+Earn Strategy",
             color="#e63946", linewidth=2.2)
    ax1.plot(eq_vni.index,   eq_vni.values,   label="VNINDEX",
             color="#aaaaaa", linewidth=1.3, linestyle="--")
    if regime is not None:
        for _, row in regime[regime["date"] >= pd.Timestamp(args.start)].iterrows():
            if row["risk_on"] == 0:
                ax1.axvspan(row["date"] - pd.offsets.MonthBegin(1), row["date"],
                            alpha=0.12, color="red", linewidth=0)
    ax1.set_ylabel("Value (start=1)")
    ax1.set_title("Cumulative returns  |  red = regime OFF / cash")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    active_bar = result[result["n"] > 0].copy()
    ax2.bar(active_bar["date"], active_bar["ret"],
            color=["#2a9d8f" if r >= 0 else "#e63946" for r in active_bar["ret"]],
            alpha=0.8, width=25)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Monthly return")
    ax2.set_title("Monthly returns (active months)")
    ax2.grid(True, alpha=0.3, axis="y")

    ax3 = axes[2]
    ax3.bar(active_bar["date"], active_bar["universe"],
            color="#457b9d", alpha=0.7, width=25, label="Scored universe")
    ax3.bar(active_bar["date"], active_bar["n"],
            color="#e63946", alpha=0.9, width=25, label=f"Holdings (top {args.top})")
    ax3.set_ylabel("Number of stocks")
    ax3.set_title("Universe size vs portfolio size")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    tag = f"value_bt_top{args.top}_cost{args.cost_bps}_reg{int(not args.no_regime)}"
    fig_path = os.path.join(OUT_DIR, f"{tag}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Save returns
    out_path = os.path.join(OUT_DIR, f"{tag}_returns.csv")
    result.to_csv(out_path, index=False)

    print(f"\nChart:   {fig_path}")
    print(f"Returns: {out_path}")


if __name__ == "__main__":
    main()
