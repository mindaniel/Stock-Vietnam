"""
factor_stock_ranker.py
======================
Ranks stocks within a sector signal using quarterly fundamental quality.

Used as a plug-in for 4sectors.py stock selection.
Answers the question: within a sector that is already confirmed to be
recovering, which individual stocks are fundamentally improving vs which
are just falling knives?

Default scoring (within-sector, cross-sectional z-score):
  50%  np_yoy          — earnings growth YoY (earnings direction)
  30%  accel_score     — consecutive quarters of accelerating YoY growth
  20%  roe             — return on equity (quality of recovery)

Real Estate scoring (sector-specific):
  30%  np_yoy          — earnings growth (lumpy for RE, lower weight)
  25%  accel_score     — consecutive quarters of accelerating profit
  20%  roe             — return on equity
  25%  assets_yoy      — total assets growth (proxy for land-bank / pipeline growth)
  D/E cap relaxed to 5.0 (RE companies naturally carry more debt)

Hard filters (applied before scoring, soft fallback if too aggressive):
  ttm_np > 0           — must be profitable on trailing 12M basis
  np_yoy >= MIN_NP_YOY — profit not in structural collapse
  debt_equity < max_de — not over-leveraged (sector-specific cap)

Usage (from 4sectors.py):
  from factor_stock_ranker import build_factor_features, rank_by_factor

  # At startup (once):
  _QFEAT = build_factor_features()

  # At each buy signal:
  top_tickers = rank_by_factor(candidate_tickers, exec_date, _QFEAT, top_pct=0.5)
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
FA_DIR   = os.path.join(BASE_DIR, "data", "financials_fa")

# Hard filter defaults — can be overridden by caller
DEFAULT_MIN_NP_YOY  = -0.30   # allow up to -30% YoY decline
DEFAULT_MAX_DE      =  3.0    # max debt/equity ratio
# Fallback fraction: if hard filters remove > (1 - FALLBACK_KEEP) of candidates,
# relax to ttm_np > 0 only to avoid leaving you with no stocks
FALLBACK_KEEP       =  0.30   # keep at least 30% of candidates


# ── Filing lag ────────────────────────────────────────────────────────────────

def _avail_date(year: int, quarter: int) -> pd.Timestamp:
    """Quarter results are available 2 months after quarter end (VN filing lag)."""
    if quarter == 1:   return pd.Timestamp(year,   6,  1)
    elif quarter == 2: return pd.Timestamp(year,   9,  1)
    elif quarter == 3: return pd.Timestamp(year,  12,  1)
    else:              return pd.Timestamp(year+1,  4,  1)


# ── Feature builder ───────────────────────────────────────────────────────────

def build_factor_features(fa_dir: str = None,
                          symbols: list = None) -> pd.DataFrame:
    """
    Load quarterly fundamentals and compute factor features.

    Parameters
    ----------
    fa_dir  : directory containing per-symbol .parquet files
    symbols : optional list of tickers to load (e.g. strategy universe).
              If None, loads all ~1500 symbols (slow ~30s).
              Pass the strategy-sector ticker list to cut loading to ~3s.

    Returns DataFrame with columns:
      symbol, year, quarter, avail_date,
      np_yoy, rev_yoy, assets_yoy, accel_score, ttm_np, ttm_rev,
      total_assets, debt_equity, roe, ocf, net_margin

    One row per (symbol, year, quarter). Caller filters by avail_date <= exec_date.
    """
    if fa_dir is None:
        fa_dir = FA_DIR

    # Build lookup set for fast membership check (uppercase)
    sym_filter = {s.upper() for s in symbols} if symbols else None

    raw_rows = []
    n_loaded = 0
    for fpath in glob.glob(os.path.join(fa_dir, "*.parquet")):
        sym = os.path.basename(fpath).replace(".parquet", "").upper()
        if sym == "INDICATORS_SNAPSHOT":
            continue
        # Skip if caller provided a symbol filter and this ticker isn't in it
        if sym_filter is not None and sym not in sym_filter:
            continue
        try:
            df = pd.read_parquet(fpath)
            if "quarter" not in df.columns:
                continue
            df = df[df["quarter"].isin([1, 2, 3, 4])].copy()
            if df.empty:
                continue
            df["symbol"] = sym
            for c in ["revenue", "net_profit", "equity", "net_margin",
                      "gross_margin", "debt_equity", "ocf", "roe",
                      "total_assets"]:
                df[c] = pd.to_numeric(df.get(c), errors="coerce")
            raw_rows.append(df[["symbol", "year", "quarter",
                                 "revenue", "net_profit", "equity",
                                 "net_margin", "gross_margin",
                                 "debt_equity", "ocf", "roe",
                                 "total_assets"]])
            n_loaded += 1
        except Exception:
            continue

    if not raw_rows:
        print("  [FACTOR] No quarterly parquets found — factor ranking disabled")
        return pd.DataFrame()

    qdf = pd.concat(raw_rows, ignore_index=True)
    qdf = qdf.sort_values(["symbol", "year", "quarter"]).reset_index(drop=True)
    qdf["avail_date"] = qdf.apply(
        lambda r: _avail_date(int(r["year"]), int(r["quarter"])), axis=1)

    # ── Compute features per symbol ───────────────────────────────────────────
    out = []
    for sym, g in qdf.groupby("symbol"):
        g = g.sort_values(["year", "quarter"]).reset_index(drop=True)
        for i in range(len(g)):
            row   = g.iloc[i]
            yr    = int(row["year"])
            qt    = int(row["quarter"])

            # YoY vs same quarter last year
            ly        = g[(g["year"] == yr - 1) & (g["quarter"] == qt)]
            np_yoy    = np.nan
            rev_yoy   = np.nan
            assets_yoy = np.nan
            if not ly.empty:
                base_n = ly["net_profit"].iloc[0]
                base_r = ly["revenue"].iloc[0]
                base_a = ly["total_assets"].iloc[0]
                if pd.notna(base_n) and base_n != 0:
                    np_yoy  = (row["net_profit"] - base_n) / abs(base_n)
                if pd.notna(base_r) and base_r != 0:
                    rev_yoy = (row["revenue"] - base_r) / abs(base_r)
                # Assets YoY — proxy for land-bank / inventory pipeline growth (RE)
                if pd.notna(base_a) and base_a > 0:
                    assets_yoy = (row["total_assets"] - base_a) / base_a

            # TTM (trailing 4 quarters)
            past4   = g.iloc[max(0, i - 3): i + 1]
            ttm_np  = past4["net_profit"].sum() if past4["net_profit"].notna().any() else np.nan
            ttm_rev = past4["revenue"].sum()     if past4["revenue"].notna().any()    else np.nan

            # Acceleration score (0–2): how many consecutive YoY improvements
            accel_score = 0
            if i >= 2:
                yoy_vals = []
                for j in [i - 2, i - 1, i]:
                    r2  = g.iloc[j]
                    ly2 = g[(g["year"] == int(r2["year"]) - 1) &
                             (g["quarter"] == int(r2["quarter"]))]
                    if (not ly2.empty
                            and pd.notna(ly2["net_profit"].iloc[0])
                            and ly2["net_profit"].iloc[0] != 0):
                        yoy_vals.append(
                            (r2["net_profit"] - ly2["net_profit"].iloc[0])
                            / abs(ly2["net_profit"].iloc[0])
                        )
                if len(yoy_vals) >= 2:
                    diffs = [yoy_vals[k + 1] - yoy_vals[k]
                             for k in range(len(yoy_vals) - 1)]
                    accel_score = sum(1 for d in diffs if d > 0)   # 0, 1, or 2

            out.append({
                "symbol":      sym,
                "year":        yr,
                "quarter":     qt,
                "avail_date":  row["avail_date"],
                "np_yoy":      np_yoy,
                "rev_yoy":     rev_yoy,
                "assets_yoy":  assets_yoy,
                "ttm_np":      ttm_np,
                "ttm_rev":     ttm_rev,
                "total_assets": row["total_assets"],
                "accel_score": accel_score,
                "debt_equity": row["debt_equity"],
                "roe":         row["roe"],
                "ocf":         row["ocf"],
                "net_margin":  row["net_margin"],
            })

    qfeat = pd.DataFrame(out)
    qfeat = qfeat.sort_values(["symbol", "avail_date"]).reset_index(drop=True)
    scope = f"{n_loaded} symbols" if sym_filter is None else f"{n_loaded}/{len(sym_filter)} symbols (filtered)"
    print(f"  [FACTOR] Loaded {scope}, "
          f"{len(qfeat):,} quarterly rows, "
          f"years {qfeat['year'].min():.0f}–{qfeat['year'].max():.0f}")
    return qfeat


# ── Ranker ────────────────────────────────────────────────────────────────────

def rank_by_factor(tickers:    list,
                   as_of_date,
                   qfeat:      pd.DataFrame,
                   top_pct:    float = 0.5,
                   min_np_yoy: float = DEFAULT_MIN_NP_YOY,
                   max_de:     float = DEFAULT_MAX_DE,
                   sector:     str   = None) -> list:
    """
    Given a list of candidate tickers and a date, return them sorted by
    fundamental quality score, filtered to the top `top_pct` fraction.

    Parameters
    ----------
    tickers    : candidate tickers (e.g. the liquid stocks in the sector)
    as_of_date : execution date — only uses data with avail_date <= this date
    qfeat      : output of build_factor_features()
    top_pct    : keep the best this fraction (0.5 = top 50%)
    min_np_yoy : hard filter — reject if YoY profit growth below this
    max_de     : hard filter — reject if debt/equity above this
    sector     : sector name for sector-specific scoring/filters
                 "Real Estate" → relaxed D/E (5.0), adds assets_yoy to score

    Returns
    -------
    List of tickers sorted best-first. If data is insufficient, returns
    original list unchanged (safe fallback — never leaves you with nothing).
    """
    if qfeat is None or qfeat.empty or not tickers:
        return tickers

    as_of    = pd.Timestamp(as_of_date)
    tick_up  = [t.upper() for t in tickers]

    # Sector-specific parameter overrides
    is_re = (sector is not None and "real estate" in sector.lower())
    if is_re and max_de == DEFAULT_MAX_DE:
        # RE companies carry naturally higher leverage (land financing)
        max_de = 5.0

    # Latest available quarter per symbol as of exec_date
    avail = qfeat[(qfeat["avail_date"] <= as_of) &
                  (qfeat["symbol"].isin(tick_up))]
    if avail.empty:
        return tickers

    latest = (avail.sort_values("avail_date")
                   .groupby("symbol").last()
                   .reset_index())

    # ── Hard filters ─────────────────────────────────────────────────────────
    n_before = len(latest)
    filtered = latest[latest["ttm_np"].fillna(-1) > 0].copy()
    filtered = filtered[filtered["np_yoy"].isna() | (filtered["np_yoy"] >= min_np_yoy)]
    filtered = filtered[filtered["debt_equity"].fillna(999) < max_de]

    # Soft fallback: if filters are too aggressive, relax to ttm_np > 0 only
    if len(filtered) < max(2, int(n_before * FALLBACK_KEEP)):
        filtered = latest[latest["ttm_np"].fillna(-1) > 0].copy()

    # If still nothing scored — return original order (safe fallback)
    if filtered.empty:
        return tickers

    # ── Scoring ───────────────────────────────────────────────────────────────
    def _zscore(s: pd.Series) -> pd.Series:
        m, sd = s.mean(), s.std()
        if pd.isna(sd) or sd < 1e-9:
            return pd.Series(0.0, index=s.index)
        return (s - m) / sd

    filtered = filtered.copy()
    np_yoy_z  = _zscore(filtered["np_yoy"].clip(-2, 10).fillna(0))
    accel_z   = _zscore(filtered["accel_score"].fillna(0))
    roe_z     = _zscore(filtered["roe"].fillna(0))

    if is_re and "assets_yoy" in filtered.columns:
        # RE: revenue recognition is lumpy (project handover-driven)
        # Use total_assets YoY as proxy for land-bank / pipeline growth
        # Clip: ignore >50% asset growth (leverage buildup risk) and <-20%
        assets_yoy_z = _zscore(filtered["assets_yoy"].clip(-0.20, 0.50).fillna(0))
        filtered["factor_score"] = (0.30 * np_yoy_z
                                  + 0.25 * accel_z
                                  + 0.20 * roe_z
                                  + 0.25 * assets_yoy_z)
    else:
        filtered["factor_score"] = 0.50 * np_yoy_z + 0.30 * accel_z + 0.20 * roe_z

    filtered = filtered.sort_values("factor_score", ascending=False)

    # Keep top_pct of the scored stocks
    n_keep  = max(int(len(filtered) * top_pct), min(3, len(filtered)))
    top_set = set(filtered.head(n_keep)["symbol"].tolist())

    # Preserve original tickers that had no fundamental data (don't block new listings)
    scored_set = set(latest["symbol"].tolist())
    no_data    = [t for t in tickers if t.upper() not in scored_set]

    # Return: factor-ranked top picks + unscored tickers at end
    ordered = [t for t in tickers if t.upper() in top_set]
    return ordered + no_data


def quality_flags(tickers: list,
                  as_of_date,
                  qfeat: pd.DataFrame,
                  min_np_yoy: float = DEFAULT_MIN_NP_YOY,
                  max_de:     float = DEFAULT_MAX_DE,
                  sector:     str   = None) -> dict:
    """
    Return a dict {ticker: True/False} indicating whether each ticker passes
    fundamental quality filters. Used by the demand scanner to show ✓/✗ flags.

    True  = passes quality (profitable, growing, reasonable D/E)
    False = fails (loss-making, in collapse, or over-leveraged)
    None  = no data available (neutral)
    """
    if qfeat is None or qfeat.empty or not tickers:
        return {t: None for t in tickers}

    as_of   = pd.Timestamp(as_of_date)
    tick_up = [t.upper() for t in tickers]

    is_re = (sector is not None and "real estate" in sector.lower())
    de_cap = 5.0 if is_re else max_de

    avail = qfeat[(qfeat["avail_date"] <= as_of) &
                  (qfeat["symbol"].isin(tick_up))]
    if avail.empty:
        return {t: None for t in tickers}

    latest = (avail.sort_values("avail_date")
                   .groupby("symbol").last()
                   .reset_index())

    result = {}
    for t in tickers:
        row = latest[latest["symbol"] == t.upper()]
        if row.empty:
            result[t] = None
            continue
        r = row.iloc[0]
        passes = bool(
            (pd.notna(r["ttm_np"]) and r["ttm_np"] > 0)
            and (pd.isna(r["np_yoy"]) or r["np_yoy"] >= min_np_yoy)
            and (pd.isna(r["debt_equity"]) or r["debt_equity"] < de_cap)
        )
        result[t] = passes   # always Python bool (not numpy.bool_)
    return result


# ── Quick diagnostic ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("Building factor features...")
    qf = build_factor_features()
    if qf.empty:
        print("No data found.")
    else:
        print(f"Shape: {qf.shape}")
        print(f"Symbols: {qf['symbol'].nunique()}")
        print(f"\nSample — HPG latest:")
        hpg = qf[qf["symbol"] == "HPG"].tail(4)
        print(hpg[["year","quarter","avail_date","np_yoy","accel_score","ttm_np","roe"]].to_string())
        print(f"\nSample ranking — Banks as of 2024-06-30:")
        banks = ["VCB","BID","CTG","TCB","MBB","VPB","STB","HDB","ACB","TPB",
                 "LPB","SHB","MSB","VIB","OCB","ABB","SSB","BAB","NVB","KLB"]
        ranked = rank_by_factor(banks, "2024-06-30", qf, top_pct=0.5, sector="Banks")
        print("Top half by factor score:", ranked[:len(ranked)//2])

        print(f"\nSample ranking — Real Estate as of 2026-04-01 (with assets_yoy):")
        re_stocks = ["VHM","VIC","NLG","KDH","DXG","PDR","BCM","VRE","HDG","NVL",
                     "FIR","IDJ","SJS","NDN","VPI","NRC","TAL"]
        ranked_re = rank_by_factor(re_stocks, "2026-04-01", qf, top_pct=0.5, sector="Real Estate")
        print("Top half:", ranked_re[:len(ranked_re)//2])
        flags = quality_flags(re_stocks, "2026-04-01", qf, sector="Real Estate")
        print("\nQuality flags:")
        for t, ok in flags.items():
            mark = "✓" if ok else ("✗" if ok is False else "?")
            print(f"  {t:6s} {mark}")
