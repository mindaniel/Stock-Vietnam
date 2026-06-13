"""
=============================================================================
LIVE SIGNAL GENERATOR  —  TRANCHE2 + MOM_BOT50 edition
=============================================================================
Tells you exactly what to do TODAY across five areas:

  1. SECTOR DASHBOARD   — state of all 4 sectors right now
  2. EXIT CHECK         — should you sell your current position?
  3. TRANCHE 2 STATUS   — is T2 due today? which stocks? how many?
  4. TAKE-PROFIT ALERTS — which holdings hit +25%? sell at tomorrow's open
  5. ENTRY SIGNAL       — which sector to enter, T1 stock list, T2 plan

HOW TO USE DAILY:
  1. Update the YOUR STATE section below (PORTFOLIO, HELD_SECTOR, etc.)
  2. Run: python live_signal_generator.py
  3. Read the output top-to-bottom — follow the actions in order
  4. Output also saved to signals_YYYYMMDD.txt

=============================================================================
"""

import pandas as pd
import numpy as np
import glob, os, warnings
import sys
from datetime import datetime, date
warnings.filterwarnings("ignore")

try:
    from factor_stock_ranker import build_factor_features, rank_by_factor, quality_flags
    _FACTOR_RANKER_AVAILABLE = True
except ImportError:
    _FACTOR_RANKER_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    # Avoid Windows cp1252 console crashes when printing Unicode box characters
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ═════════════════════════════════════════════════════════════════
# YOUR STATE — UPDATE THESE EVERY DAY
# ═════════════════════════════════════════════════════════════════

# Which sector are you currently holding? None if in cash.
HELD_SECTOR = ""           # e.g. "Real Estate" or "Banks"

# Date you executed Tranche 1 (YYYY-MM-DD). None if in cash.
ENTRY_DATE  = ""           # e.g. "2025-01-15"

# Has Tranche 2 been deployed yet?
TRANCHE2_DONE = False        # set to True once you execute T2

# Your current holdings (update daily with actual shares and prices)
# ticker → {shares, entry_price (in thousands VND), entry_date, tranche}
# entry_price: e.g. 25.5 means 25,500 VND per share
PORTFOLIO = {
    # "HPG": {"shares": 10000, "entry_price": 25.50,
    #         "entry_date": "2025-01-15", "tranche": "T1"},
    # "HSG": {"shares": 5000,  "entry_price": 18.20,
    #         "entry_date": "2025-01-15", "tranche": "T1"},
}

# Your CURRENT total portfolio value in VND
# (settled cash + unrealised value of holdings)
TOTAL_CAPITAL_VND = 100_000_000

# Chart options
SAVE_CYCLE_CHART = True
SHOW_CYCLE_CHART = False


# ═════════════════════════════════════════════════════════════════
# PATHS — update to your machine once, then leave alone
# ═════════════════════════════════════════════════════════════════
TICKER_SECTORS = os.path.join(BASE_DIR, "ticker_sectors.csv")
VNINDEX_PATH   = os.path.join(BASE_DIR, "VNINDEX.csv")
INDIVIDUAL_DIR = os.path.join(BASE_DIR, "data", "price")

# Keep generated outputs tidy in one place
RESULTS_ROOT = os.path.join(BASE_DIR, "results", "4sector_live_signals")
REPORTS_DIR  = os.path.join(RESULTS_ROOT, "reports")
CHARTS_DIR   = os.path.join(RESULTS_ROOT, "charts")


# ═════════════════════════════════════════════════════════════════
# STRATEGY CONFIG — must match backtest exactly
# ═════════════════════════════════════════════════════════════════
SECTOR_GROUPS = {
    "Banks":           "Banks",
    "Food & Beverage": "Food & Beverage",
    "Basic Resources": "Basic Resources",
    "Real Estate":     "Real Estate",
}
Z_WINDOW            = 252
SMOOTH_WINDOW       = 20
VELOCITY_WINDOW     = 20
VOL_WINDOW          = 60
RECOV_WINDOW_LONG   = 504
RECOVERY_WINDOW     = 40
MIN_LIQUIDITY_VND   = 1_000_000_000
MAX_PARTICIPATION   = 0.20
STOCK_TP_PCT        = 0.25
# ── Ichimoku Kijun-sen entry guidance ────────────────────────────
# Displayed in stock tables as a per-stock entry recommendation.
# Does NOT automate anything — tells you which stocks to buy today
# vs which to watch for a pullback.
KIJUN_PERIOD        = 26      # 26-period midline (Ichimoku standard)
KIJUN_BUY_THRESHOLD = 0.03    # "ready" if price <= kijun * (1 + 3%)
ENTRY_MAX_WAIT_DAYS = 3       # after 3 days, buy at market regardless
MIN_ENTRY_SCORE     = 3.0
SPREAD_EXIT         = -2.0
SPREAD_PEAK         =  5.0
SPREAD_CRASH        = -15.0
SPREAD_HIGH_EXIT    = 50.0
MOMENTUM_FLOOR      = -0.05
MOMENTUM_LOOKBACK   = 20
RECOVERY_WINDOW     = 40
EARLY_ENTRY_SECTORS = {"Banks"}
EARLY_ENTRY_K       = 0.25
FRICTION            = 0.0025

# ── EARLY_V3 parameters ───────────────────────────────────────────
# Enter while sector is still DROWNING (spread still negative) when:
#   1. Spread > EARLY_V3_THRESHOLD  (not too deep, above -10)
#   2. Velocity has been positive for EARLY_V3_VEL_DAYS in a row
#      (spread is rising consistently — not a 1-day blip)
# This fires BEFORE the baseline RECOVERY signal to get a better price.
# Validated OOS: 4/5 walk-forward windows better Sharpe than baseline,
# same combined MDD (-41.3%).  Weakness: choppy markets (2018-2020).
EARLY_V3_THRESHOLD  = -10.0   # spread floor for early entry
EARLY_V3_VEL_DAYS   =  3      # consecutive days velocity > 0 required

# ── Demand-early entry (stock-flow based pre-signal) ─────────────
# Fires when sector heat (avg full-candle score of liquid stocks)
# exceeds threshold while sector is still DROWNING.
# WFO validated: 5/5 windows beat baseline Sharpe.
DEMAND_EARLY_ENABLED     = True
DEMAND_HEAT_WINDOW       = 20
DEMAND_HEAT_THRESHOLD    = 0.012   # fallback when history too short
DEMAND_HEAT_SPREAD_FLOOR = -15.0
# Adaptive threshold: fire when heat > Nth-pct of own 252-day history.
# Self-calibrates to market regime — low-vol periods lower the bar,
# high-vol periods raise it.
DEMAND_HEAT_USE_ADAPTIVE = True
DEMAND_HEAT_PERCENTILE   = 75   # top 25% of own history = genuine anomaly
DEMAND_HEAT_MIN_HISTORY  = 60   # min points before adaptive threshold kicks in
DEMAND_HEAT_HISTORY_DAYS = 252  # rolling look-back for percentile computation
# Late-entry momentum filter for DEMAND_EARLY stock selection:
# When DEMAND fires, some stocks have already moved a lot (they drove the heat).
# This cap skips stocks with >15% 20d momentum, leaving only laggards that
# haven't run yet — better risk/reward since full TP upside is still available.
# Set to None to disable and show all stocks.
DEMAND_EARLY_MAX_MOM_PCT = 0.15

# ── Per-sector stock selection (volume experiment results) ────────
# Banks:           MOM_BOT50  — vol metrics hurt (macro/news driven, stocks move together)
# Basic Resources: VOL_LEADERS — top 5 by vol_score; win rate 39.8% → 50.8%
# Food & Beverage: VOL_RANK   — top 50% by vol_score; win rate 41.3% → 50.0%
# Real Estate:     VOL_LEADERS N=12 — top 12 by full-candle score; +21% total return
#                                      vs static vol_score, win rate 67%, avg +14.2%
SECTOR_STOCK_SELECTION = {
    "Banks":           "MOM_BOT50",
    "Basic Resources": "VOL_LEADERS",
    "Food & Beverage": "VOL_RANK",
    "Real Estate":     "VOL_LEADERS",
}
VOL_LEADERS_N = 12
# Rank VOL_LEADERS by recent full-body candle score (body_pct × body_ratio,
# recency-weighted) — picks stocks with a strong recent conviction buying
# session rather than accumulated historical flow.
FULL_CANDLE_WINDOW          = 10
VOL_LEADERS_USE_FULL_CANDLE = True

TRANCHE1_FRAC = 0.50
TRANCHE2_FRAC = 0.50

# ── T2 timing ────────────────────────────────────────────────────────────────
# Default: fixed 3 trading days after T1 (TRANCHE2 mode — best over 20yr backtest)
# Alternative: DIP mode (deploy on next red day, max 5d wait) — better post-2018
#   but misses fast 2013-style rallies; ~45% less compounded wealth over full period
# Change T2_MODE to "DIP" only if you believe the current regime is choppy/volatile
T2_MODE         = "TRANCHE2"  # "TRANCHE2" (fixed 3d) or "DIP" (next red day)
T2_FIXED_DAYS   = 3           # trading days for TRANCHE2 mode
DIP_THRESHOLD   = -0.003      # sector return <= -0.3% triggers T2 in DIP mode
DIP_MAX_WAIT    = 5           # max trading days before deploying at market in DIP mode

# ── Fundamental quality filter ────────────────────────────────────
# Set to False to show all liquidity-passing stocks (no fundamental screen).
FUNDAMENTAL_FILTER_ENABLED = True

SECTOR_FUND_CRITERIA = {
    "Banks":           {"roe_min": 10.0},
    "Real Estate":     {"revenue_or_asset_growth_min": -0.10},
    "Food & Beverage": {"revenue_growth_min": 0.0, "ocf_min": 0.5, "roe_min": 8.0},
    "Basic Resources": {"revenue_growth_min": 0.0, "ocf_min": 0.0},
}
_FUND_LAG_MONTH = 4
_FUND_LAG_DAY   = 30
_FUND_DATA      = {}

# ── Factor-enhanced stock selection (quarterly fundamentals) ─────────────────
# Ranks candidates within each sector by earnings quality before buying.
# Removes falling knives (declining profits) that passed the price/volume screen.
# Scores: 50% np_yoy + 30% accel_score + 20% ROE — keeps top FACTOR_TOP_PCT.
# Set FACTOR_SELECTION_ENABLED = False to disable entirely.
FACTOR_SELECTION_ENABLED = True
FACTOR_TOP_PCT           = 0.50    # keep top 50% by factor score
FACTOR_MIN_NP_YOY        = -0.30   # allow up to -30% YoY profit decline
FACTOR_SECTORS = {                 # per-sector toggle
    "Basic Resources": True,       # helps most — volatile earnings, quality matters
    "Banks":           True,       # removes banks with NPL/credit cycle issues
    "Real Estate":     True,       # removes structural zombies
    "Food & Beverage": False,      # stable sector — factor filter too strict
}
_QFEAT = None   # populated at startup


# ═════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════

def load_data():
    print("Loading data...")
    mapping = pd.read_csv(TICKER_SECTORS)
    mapping.columns = [c.strip().lower() for c in mapping.columns]
    sector_col = next((c for c in ["industry","sector","super_sector"]
                       if c in mapping.columns), None)
    mapping["super_sector"] = mapping[sector_col].map(SECTOR_GROUPS)
    mapping = mapping[mapping["super_sector"].notna()].copy()
    if "exchange" in mapping.columns:
        mapping = mapping[mapping["exchange"].isin(["HOSE","HNX"])].copy()
    ticker_to_sector = dict(zip(mapping["ticker"].str.upper(),
                                mapping["super_sector"]))

    files = glob.glob(os.path.join(INDIVIDUAL_DIR, "*.parquet"))
    stock_data, sector_data = {}, {}
    total_in_sectors = 0
    for fpath in files:
        ticker = os.path.splitext(os.path.basename(fpath))[0].upper()
        if ticker not in ticker_to_sector: continue
        total_in_sectors += 1
        try:
            df = pd.read_parquet(fpath)
            df.columns = [c.strip().lower() for c in df.columns]
            date_col = "time" if "time" in df.columns else "date"
            df["date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
            for col in ["open","close","volume"]:
                if col not in df.columns: raise ValueError()
            # Keep high/low for Kijun-sen if available
            hl_cols = ["high","low"] if ("high" in df.columns and "low" in df.columns) else []
            df = df[["open"] + hl_cols + ["close","volume"]].astype(float)
            df = df[(df["close"] > 0) & (df["open"] > 0)]
            df = df[df.index <= pd.Timestamp.today()]
            if len(df) < 60: continue
            # Liquidity filter — match backtest universe (≥1B VND median daily turnover)
            med_to = (df["close"] * df["volume"] * 1000).tail(60).median()
            if med_to < MIN_LIQUIDITY_VND: continue
            stock_data[ticker] = df
            sec = ticker_to_sector[ticker]
            sector_data.setdefault(sec, {})[ticker] = df
        except Exception:
            pass

    latest_date = max(df.index.max() for df in stock_data.values())
    print(f"  {len(stock_data)} stocks loaded (liquidity ≥{MIN_LIQUIDITY_VND/1e9:.0f}B VND/day, "
          f"from {total_in_sectors} strategy-sector tickers) | latest data: {latest_date.date()}")
    return stock_data, sector_data, latest_date


# ═════════════════════════════════════════════════════════════════
# FUNDAMENTAL FILTER HELPERS
# ═════════════════════════════════════════════════════════════════

def load_fundamental_data(symbols=None):
    import datetime as _dt
    fa_dir = os.path.join(BASE_DIR, "data", "financials_fa")
    if not os.path.isdir(fa_dir):
        print(f"  [FUND] {fa_dir} not found — filter disabled")
        return {}
    sym_filter = {s.upper() for s in symbols} if symbols else None
    result, loaded = {}, 0
    for fpath in glob.glob(os.path.join(fa_dir, "*.parquet")):
        sym = os.path.basename(fpath).replace(".parquet", "")
        if sym == "indicators_snapshot":
            continue
        if sym_filter is not None and sym.upper() not in sym_filter:
            continue
        try:
            df = pd.read_parquet(fpath)
            if "quarter" in df.columns:
                df = df[df["quarter"] == 0].copy()
            elif "type" in df.columns:
                df = df[df["type"] == "yearly"].copy()
            if df.empty:
                continue
            df = df.sort_values("year").reset_index(drop=True)
            if "sector" not in df.columns or df["sector"].isna().all():
                if "industry" in df.columns:
                    df["sector"] = df["industry"]
            elif "industry" in df.columns:
                df["sector"] = df["sector"].fillna(df["industry"])
            for raw_col, growth_col in [
                ("revenue",     "revenue_growth_lfy"),
                ("net_profit",  "profit_growth_lfy"),
                ("total_assets","asset_growth_lfy"),
            ]:
                if raw_col in df.columns:
                    if growth_col not in df.columns or df[growth_col].isna().all():
                        df[growth_col] = df[raw_col].pct_change()
            df["avail_date"] = df["year"].apply(
                lambda y: _dt.date(int(y) + 1, _FUND_LAG_MONTH, _FUND_LAG_DAY))
            df = df.sort_values("avail_date").reset_index(drop=True)
            result[sym] = df
            loaded += 1
        except Exception as _e:
            if loaded == 0:   # print first error only to diagnose
                print(f"  [FUND] ERR {sym}: {_e}")
    print(f"  [FUND] Loaded {loaded} symbols")
    return result


def get_fund_row(symbol, as_of_date, fund_data):
    df = fund_data.get(symbol)
    if df is None or df.empty:
        return None
    as_of = as_of_date.date() if hasattr(as_of_date, "date") else as_of_date
    available = df[df["avail_date"] <= as_of]
    return available.iloc[-1] if not available.empty else None


def passes_fundamental_filter(symbol, exec_date, fund_data=None):
    import math as _math
    if fund_data is None:
        fund_data = _FUND_DATA
    if not FUNDAMENTAL_FILTER_ENABLED or not fund_data:
        return True
    row = get_fund_row(symbol, exec_date, fund_data)
    if row is None:
        return True
    sector   = row.get("sector", "")
    criteria = SECTOR_FUND_CRITERIA.get(sector)
    if criteria is None:
        return True

    def _ok(val, thr):
        if thr is None or val is None:
            return True
        try:
            v = float(val)
        except (TypeError, ValueError):
            return True
        return True if (_math.isnan(v) or _math.isinf(v)) else v >= thr

    if sector == "Banks":
        if not _ok(row.get("roe"), criteria.get("roe_min")):
            return False
    elif sector == "Real Estate":
        thr = criteria.get("revenue_or_asset_growth_min", -0.10)
        rev = row.get("revenue_growth_lfy")
        ast = row.get("asset_growth_lfy")
        has_rev = rev is not None and not (isinstance(rev, float) and (_math.isnan(rev) or _math.isinf(rev)))
        has_ast = ast is not None and not (isinstance(ast, float) and (_math.isnan(ast) or _math.isinf(ast)))
        if has_rev or has_ast:
            if not ((_ok(rev, thr) if has_rev else False) or (_ok(ast, thr) if has_ast else False)):
                return False
    elif sector == "Food & Beverage":
        if not _ok(row.get("revenue_growth_lfy"), criteria.get("revenue_growth_min")): return False
        if not _ok(row.get("ocf_to_netprofit"),   criteria.get("ocf_min")):            return False
        if not _ok(row.get("roe"),                 criteria.get("roe_min")):            return False
    elif sector == "Basic Resources":
        if not _ok(row.get("revenue_growth_lfy"), criteria.get("revenue_growth_min")): return False
        if not _ok(row.get("ocf_to_netprofit"),   criteria.get("ocf_min")):            return False
    return True


# ═════════════════════════════════════════════════════════════════
# ICHIMOKU KIJUN-SEN ENTRY HELPER
# ═════════════════════════════════════════════════════════════════

def kijun_status(ticker, current_price, stock_data, today):
    """
    Compute Kijun-sen (26-period baseline) for a stock and return
    an entry guidance dict:

      ready       : bool   — True if price is within KIJUN_BUY_THRESHOLD above Kijun
      kijun       : float  — Kijun-sen value (or None if insufficient history)
      pct_above   : float  — how far price is above Kijun (%)
      limit_price : float  — price at KIJUN_BUY_THRESHOLD above Kijun (limit target)
      label       : str    — one-line summary for printing

    Uses high/low if available (proper Ichimoku), else close proxy.
    """
    df = stock_data.get(ticker)
    if df is None:
        return {"ready": True, "kijun": None, "pct_above": 0.0,
                "limit_price": current_price, "label": "—"}

    hist = df[df.index < today].tail(KIJUN_PERIOD)
    if len(hist) < max(KIJUN_PERIOD // 2, 5):
        return {"ready": True, "kijun": None, "pct_above": 0.0,
                "limit_price": current_price, "label": "no hist"}

    if "high" in hist.columns and "low" in hist.columns:
        kijun = (hist["high"].max() + hist["low"].min()) / 2
    else:
        kijun = (hist["close"].max() + hist["close"].min()) / 2

    pct_above   = (current_price / kijun - 1) * 100
    limit_price = round(kijun * (1 + KIJUN_BUY_THRESHOLD), 2)
    ready       = pct_above <= KIJUN_BUY_THRESHOLD * 100

    if ready:
        label = f"BUY  (Kijun {kijun:.2f}, {pct_above:+.1f}%)"
    else:
        label = f"WAIT (Kijun {kijun:.2f}, {pct_above:+.1f}% above → lim {limit_price:.2f})"

    return {"ready": ready, "kijun": round(kijun, 2),
            "pct_above": round(pct_above, 1),
            "limit_price": limit_price, "label": label}


# ═════════════════════════════════════════════════════════════════
# SIGNAL ENGINE
# ═════════════════════════════════════════════════════════════════

def rolling_slope(series, window):
    result = np.full(len(series), np.nan)
    vals = series.values
    for i in range(window - 1, len(vals)):
        y = vals[i - window + 1:i + 1]
        if np.any(np.isnan(y)): continue
        result[i] = np.polyfit(np.arange(window, dtype=float), y, 1)[0]
    return pd.Series(result, index=series.index)


def compute_rolling_recovery(spread, window):
    vals, n = spread.values, len(spread)
    out = np.full(n, np.nan)
    for i in range(window, n):
        w = vals[max(0, i-window):i]
        cyc, in_neg, tp = [], False, None
        for j, v in enumerate(w):
            if v < 0 and not in_neg:  in_neg=True; tp=j
            elif v >= 0 and in_neg:   in_neg=False; cyc.append(j-tp); tp=None
        if cyc: out[i] = np.mean(cyc)
    return pd.Series(out, index=spread.index).ffill().bfill().fillna(90)


def build_sector_signal(sector_stocks):
    records = []
    for ticker, df in sector_stocks.items():
        records.append(df[["close"]].assign(ticker=ticker))
    if not records: return None
    all_prices = pd.concat(records).pivot_table(
        index="date", columns="ticker", values="close", aggfunc="last")

    sma = all_prices.rolling(Z_WINDOW, min_periods=max(int(Z_WINDOW*0.8),60)).mean()
    std = all_prices.rolling(Z_WINDOW, min_periods=max(int(Z_WINDOW*0.8),60)).std()
    z   = (all_prices - sma) / std

    drown_sm = (z <= -1.5).astype(float).mean(axis=1).ewm(span=SMOOTH_WINDOW).mean() * 100
    peak_sm  = (z >=  1.5).astype(float).mean(axis=1).ewm(span=SMOOTH_WINDOW).mean() * 100
    spread   = peak_sm - drown_sm
    velocity = rolling_slope(spread, VELOCITY_WINDOW)

    trough_depth = spread.rolling(252, min_periods=30).min().abs()
    was_pos      = (spread >= 0).astype(int)
    cumsum_pos   = was_pos.cumsum()
    last_pos     = cumsum_pos.where(was_pos == 1).ffill().fillna(0)
    trough_dur   = (cumsum_pos - last_pos).clip(0, 252)
    cross_up     = ((spread >= 0) & (spread.shift(1) < 0)).astype(int)
    recent_cross = cross_up.rolling(RECOVERY_WINDOW, min_periods=1).sum()

    states = []
    for i in range(len(spread)):
        sp  = spread.iloc[i]
        vel = velocity.iloc[i] if not np.isnan(velocity.iloc[i]) else 0
        if   sp < SPREAD_CRASH:            states.append("CRASH")
        elif sp < 0:                        states.append("DROWNING")
        elif sp >= SPREAD_PEAK and vel < 0: states.append("PEAKING")
        elif sp >= 0 and recent_cross.iloc[i] >= 1: states.append("RECOVERY")
        else:                               states.append("LEADING")
    state_s = pd.Series(states, index=spread.index)

    vel_raw  = velocity.fillna(0).clip(lower=0)
    score    = (trough_depth.clip(lower=0)**0.5 *
                trough_dur.clip(lower=1).apply(np.log1p) *
                vel_raw**0.3)
    score    = score.where(state_s.isin(["RECOVERY","LEADING"]), 0.0)
    roll_std    = spread.rolling(VOL_WINDOW, min_periods=20).std().fillna(spread.std())
    rolling_K   = (0.75 - (compute_rolling_recovery(spread, RECOV_WINDOW_LONG) - 40) / 200).clip(0.25, 0.75)
    mom_20d     = spread.rolling(20).sum()

    # Consecutive days velocity > 0  (used by EARLY_V3)
    streak, streaks = 0, []
    for v in velocity.fillna(0):
        streak = streak + 1 if v > 0 else 0
        streaks.append(streak)
    rising_streak = pd.Series(streaks, index=spread.index)

    return pd.DataFrame({
        "spread": spread, "velocity": velocity, "state": state_s,
        "score": score, "roll_std": roll_std,
        "rolling_K": rolling_K, "mom_20d": mom_20d,
        "rising_streak": rising_streak,
    })


def get_threshold(row, sector):
    k   = EARLY_ENTRY_K if sector in EARLY_ENTRY_SECTORS else \
          float(row["rolling_K"]) if not np.isnan(row["rolling_K"]) else 0.50
    std = float(row["roll_std"]) if not np.isnan(row["roll_std"]) else 5.0
    return k * std


def entry_ok(row, sector):
    if row is None: return False
    if float(row["spread"]) < get_threshold(row, sector): return False
    state = row["state"]
    if state == "RECOVERY": return True
    if state == "LEADING":  return float(row["score"]) >= MIN_ENTRY_SCORE
    return False


def _demand_heat(sector_stocks, as_of_date, window=None):
    """Avg full-candle score across liquid stocks — sector heat for DEMAND_EARLY."""
    if window is None:
        window = DEMAND_HEAT_WINDOW
    scores = []
    for ticker, df in sector_stocks.items():
        hist = df[df.index <= as_of_date]
        med_to = (hist["close"] * hist["volume"] * 1000).tail(60).median()
        if med_to < MIN_LIQUIDITY_VND:
            continue
        recent = hist.tail(window)
        if (recent["volume"] > 0).sum() < int(window * 0.75):
            continue
        fc = _full_candle_score_from_df(df, as_of_date, window=window)
        if not np.isnan(fc):
            scores.append(fc)
    if not scores:
        return 0.0
    scores.sort(reverse=True)
    return float(np.mean(scores[:max(len(scores) // 2, 1)]))


def _adaptive_heat_threshold(sector_stocks, as_of_date, window=None,
                              history_days=None, percentile=None, sample_every=5):
    """
    Compute the adaptive demand threshold for a sector as of as_of_date.

    Samples heat scores at `sample_every`-day intervals over the past
    `history_days` trading days, then returns the Nth percentile.

    Falls back to DEMAND_HEAT_THRESHOLD when insufficient history.
    """
    if not DEMAND_HEAT_USE_ADAPTIVE:
        return DEMAND_HEAT_THRESHOLD
    if window is None:
        window = DEMAND_HEAT_WINDOW
    if history_days is None:
        history_days = DEMAND_HEAT_HISTORY_DAYS
    if percentile is None:
        percentile = DEMAND_HEAT_PERCENTILE

    # Build a date index from available data in this sector
    all_dates_in_sec = sorted(set().union(
        *[set(df.index[df.index <= as_of_date]) for df in sector_stocks.values()]
    ))
    # Take rolling window and sample every N days
    hist_window = [d for d in all_dates_in_sec
                   if d <= as_of_date][-history_days:]
    sample_dates = hist_window[::sample_every]

    heat_hist = []
    for dt in sample_dates:
        h = _demand_heat(sector_stocks, dt, window=window)
        heat_hist.append(h)

    if len(heat_hist) < DEMAND_HEAT_MIN_HISTORY // sample_every:
        return DEMAND_HEAT_THRESHOLD

    return float(np.percentile(heat_hist, percentile))


def entry_ok_early(row, sector):
    """
    EARLY_V3: fires while sector is still DROWNING.
    Conditions (all must be true):
      1. State is DROWNING (spread negative, not yet recovered)
      2. Spread > EARLY_V3_THRESHOLD (-10.0) — not in deep crash
      3. Velocity has been positive for EARLY_V3_VEL_DAYS in a row
         (spread is consistently rising, not a 1-day blip)
    Does NOT fire if baseline entry_ok() already fires — no duplication.
    """
    if row is None: return False
    if entry_ok(row, sector): return False          # baseline already covers this
    if row["state"] != "DROWNING": return False
    if float(row["spread"]) < EARLY_V3_THRESHOLD: return False
    if int(row.get("rising_streak", 0)) < EARLY_V3_VEL_DAYS: return False
    return True


def _full_candle_score_from_df(df, as_of_date, window=None):
    """
    Recent full-body candle score (no lookahead).
    body_pct × body_ratio per bullish bar, recency-weighted average.
    Requires high/low columns; returns np.nan if unavailable.
    """
    if window is None:
        window = FULL_CANDLE_WINDOW
    hist = df[df.index < as_of_date].tail(window)
    if len(hist) < 3:
        return np.nan
    if "high" not in hist.columns or "low" not in hist.columns:
        return np.nan
    scores = []
    for _, row in hist.iterrows():
        body = row["close"] - row["open"]
        rng  = row["high"]  - row["low"]
        if body <= 0 or rng <= 0 or row["open"] <= 0:
            scores.append(0.0)
            continue
        scores.append((body / row["open"]) * (body / rng))
    if not scores:
        return np.nan
    weights = np.arange(1, len(scores) + 1, dtype=float)
    return float(np.average(scores, weights=weights))


def _vol_score_from_df(df, as_of_date):
    """
    Volume accumulation score for one stock given its OHLCV DataFrame.
    Uses only data strictly before as_of_date (no lookahead).
    Higher = stronger buyer accumulation.  Returns np.nan if insufficient data.
    """
    hist = df[df.index < as_of_date]
    if len(hist) < 60:
        return np.nan

    # 1. Relative volume: recent 10d vs 60d baseline
    base_v  = hist["volume"].tail(60).mean()
    short_v = hist["volume"].tail(10).mean()
    if base_v == 0:
        return np.nan
    rv_norm = float(np.clip((short_v / base_v) - 1.0, -1.0, 2.0) / 2.0)

    # 2. Vol pressure: net buying (close > open = buying day)
    rec    = hist.tail(10)
    signs  = np.sign(rec["close"].values - rec["open"].values)
    tot_v  = rec["volume"].sum()
    vp     = float((rec["volume"].values * signs).sum() / tot_v) if tot_v > 0 else 0.0
    vp_norm = float(np.clip(vp, -1.0, 1.0))

    # 3. Price-volume correlation (inverted: neg corr = price down + vol up = accumulation)
    ch   = hist.tail(20)
    rets = ch["close"].pct_change().fillna(0).values
    vols = ch["volume"].values.astype(float)
    if vols.std() == 0 or rets.std() == 0:
        pvc_norm = 0.0
    else:
        pvc      = float(np.corrcoef(rets, vols)[0, 1])
        pvc_norm = float(np.clip(-pvc if not np.isnan(pvc) else 0.0, -1.0, 1.0))

    # 4. Dip vol slope: successive troughs attracting more volume?
    dh  = hist.tail(40)
    cl  = dh["close"].values
    vd  = dh["volume"].values.astype(float)
    trs = [vd[i] for i in range(1, len(cl) - 1)
           if cl[i] < cl[i - 1] and cl[i] < cl[i + 1]]
    if len(trs) >= 3:
        x        = np.arange(len(trs), dtype=float)
        mean_v   = np.mean(trs) if np.mean(trs) > 0 else 1.0
        slope    = float(np.polyfit(x, trs, 1)[0])
        dvs_norm = float(np.clip(slope / mean_v, -1.0, 1.0))
    else:
        dvs_norm = 0.0

    return 0.25 * rv_norm + 0.40 * vp_norm + 0.20 * pvc_norm + 0.15 * dvs_norm


def exit_needed(row):
    sp  = float(row["spread"])
    vel = float(row["velocity"]) if not np.isnan(row["velocity"]) else 0
    if sp < SPREAD_EXIT:                                  return True, f"spread = {sp:.1f} (< {SPREAD_EXIT})"
    if row["state"] == "PEAKING":                         return True, "sector state = PEAKING"
    if SPREAD_HIGH_EXIT and sp >= SPREAD_HIGH_EXIT and vel < 0:
                                                          return True, f"spread too high ({sp:.0f}) and falling"
    return False, ""


# ═════════════════════════════════════════════════════════════════
# STOCK SELECTION — MOM_BOT50
# ═════════════════════════════════════════════════════════════════

def select_stocks(sector_stocks, available_cash, latest_date, sector=None,
                  stock_data_all=None, demand_mode=False):
    # stock_data_all used for Kijun lookup (has high/low); fall back to sector_stocks
    stock_data_ref = stock_data_all if stock_data_all is not None else sector_stocks
    today_ref      = latest_date
    candidates = []
    for ticker, df in sector_stocks.items():
        hist = df[df.index < latest_date].tail(20)
        if len(hist) < 5: continue
        median_val = (hist["close"] * hist["volume"] * 1000).median()
        if median_val < MIN_LIQUIDITY_VND: continue

        row_today = df[df.index <= latest_date].tail(1)
        if row_today.empty: continue
        price = float(row_today["close"].iloc[0])
        vol   = float(row_today["volume"].iloc[0])
        if price <= 0 or vol == 0: continue

        hist30 = df[df.index <= latest_date].tail(22)
        mom = np.nan
        if len(hist30) >= 21:
            p20 = hist30["close"].iloc[-21]
            if p20 > 0: mom = (price - p20) / p20

        # Volume accumulation score (used by VOL_RANK / VOL_LEADERS fallback)
        vs = _vol_score_from_df(df, latest_date)
        # Full-candle score (used by VOL_LEADERS when USE_FULL_CANDLE=True)
        fc = _full_candle_score_from_df(df, latest_date)

        candidates.append({"ticker": ticker, "price": price,
                            "median_val": median_val, "mom_20d": mom,
                            "vol_score": vs, "full_candle_score": fc})

    if not candidates: return []

    df_c = pd.DataFrame(candidates)

    # ── Per-sector stock selection ─────────────────────────────────
    sec_method = SECTOR_STOCK_SELECTION.get(sector, "MOM_BOT50")

    if sec_method in ("VOL_LEADERS", "VOL_RANK"):
        # VOL_LEADERS: rank by full-candle score if enabled, else vol_score
        # VOL_RANK: always rank by vol_score (top 50%)
        if VOL_LEADERS_USE_FULL_CANDLE and sec_method == "VOL_LEADERS":
            rank_col = "full_candle_score"
        else:
            rank_col = "vol_score"
        df_vs = df_c.dropna(subset=[rank_col])
        if len(df_vs) >= 2:
            df_vs = df_vs.sort_values(rank_col, ascending=False)
            if sec_method == "VOL_LEADERS":
                n_keep = min(VOL_LEADERS_N, max(len(df_vs) // 2, 2))
            else:   # VOL_RANK: top 50%
                n_keep = max(len(df_vs) // 2, 3)
            df_c = df_vs.head(n_keep).copy()
        # fallback: if too few scores, use all candidates
    else:
        # MOM_BOT50: keep bottom half by 20d momentum (most oversold)
        df_v = df_c.dropna(subset=["mom_20d"])
        if len(df_v) >= 4:
            df_c = df_v[df_v["mom_20d"] <= df_v["mom_20d"].median()].copy()

    # ── DEMAND_EARLY late-entry filter ────────────────────────────
    # Skip stocks that already ran >DEMAND_EARLY_MAX_MOM_PCT in 20 days.
    # These are the "leaders" that created the heat signal — they have
    # limited remaining upside to the 25% TP.  Buy the laggards instead.
    if demand_mode and DEMAND_EARLY_MAX_MOM_PCT is not None:
        mom_ok = df_c["mom_20d"].isna() | (df_c["mom_20d"] <= DEMAND_EARLY_MAX_MOM_PCT)
        filtered = df_c[mom_ok]
        if not filtered.empty:
            df_c = filtered   # only apply if at least one stock passes

    # ── Fundamental quality filter ────────────────────────────────
    if FUNDAMENTAL_FILTER_ENABLED and _FUND_DATA:
        today_for_fund = latest_date.date() if hasattr(latest_date, "date") else latest_date
        pre_fund = len(df_c)
        fund_pass = [s for s in df_c.to_dict("records")
                     if passes_fundamental_filter(s["ticker"], today_for_fund, _FUND_DATA)]
        if fund_pass:
            df_c = pd.DataFrame(fund_pass)
        post_fund = len(df_c)
        if sector and pre_fund != post_fund:
            print(f"  [FUND] {sector}: {pre_fund} → {post_fund} stocks "
                  f"({pre_fund - post_fund} filtered by fundamentals)")

    # ── Factor-enhanced ranking (quarterly fundamentals) ─────────
    # Second-pass: rank remaining candidates by earnings quality
    # (np_yoy, accel_score, ROE) and keep top FACTOR_TOP_PCT.
    # Removes falling knives that passed price/volume/annual screens
    # but have deteriorating quarterly earnings.
    if (FACTOR_SELECTION_ENABLED
            and _FACTOR_RANKER_AVAILABLE
            and _QFEAT is not None
            and FACTOR_SECTORS.get(sector, True)
            and len(df_c) >= 3):
        tickers_in  = df_c["ticker"].tolist()
        ranked      = rank_by_factor(tickers_in, latest_date, _QFEAT,
                                     top_pct=FACTOR_TOP_PCT,
                                     min_np_yoy=FACTOR_MIN_NP_YOY,
                                     sector=sector)
        keep_set    = set(ranked)
        df_filtered = df_c[df_c["ticker"].isin(keep_set)]
        if not df_filtered.empty:
            pre_factor = len(df_c)
            df_c = df_filtered
            if sector and pre_factor != len(df_c):
                print(f"  [FACTOR] {sector}: {pre_factor} → {len(df_c)} stocks "
                      f"(factor quality filter: kept top {FACTOR_TOP_PCT*100:.0f}%)")

    # Participation cap iteration
    included = df_c.to_dict("records")
    for _ in range(3):
        if not included: break
        alloc = available_cash / len(included)
        still = [s for s in included if alloc <= s["median_val"] * MAX_PARTICIPATION]
        if len(still) == len(included): break
        included = still if still else included; break

    if not included: return []

    # Sort for display: VOL_LEADERS → full_candle or vol_score; VOL_RANK → vol_score; MOM → momentum
    if sec_method == "VOL_LEADERS" and VOL_LEADERS_USE_FULL_CANDLE:
        included.sort(key=lambda x: -(x.get("full_candle_score") or -99))
    elif sec_method in ("VOL_LEADERS", "VOL_RANK"):
        included.sort(key=lambda x: -(x.get("vol_score") or -99))
    else:
        included.sort(key=lambda x: x.get("mom_20d", 0))

    alloc = available_cash / len(included)
    result = []
    for s in included:
        price_vnd = s["price"] * 1000
        shares    = int(alloc / (price_vnd * (1 + FRICTION)))
        cost      = shares * price_vnd * (1 + FRICTION)
        if shares <= 0: continue
        tp_price  = s["price"] * (1 + STOCK_TP_PCT)
        kij       = kijun_status(s["ticker"], s["price"], stock_data_ref, today_ref)
        result.append({**s, "shares": shares, "cost_vnd": cost,
                       "tp_price": tp_price, "kijun_info": kij})
    return result


# ═════════════════════════════════════════════════════════════════
# PORTFOLIO CHECK
# ═════════════════════════════════════════════════════════════════

def check_portfolio(stock_data, latest_date):
    results = []
    for ticker, pos in PORTFOLIO.items():
        ep    = pos["entry_price"]
        entry = pd.Timestamp(pos["entry_date"])
        shares= pos["shares"]
        df    = stock_data.get(ticker)
        if df is None:
            results.append({"ticker": ticker, "status": "⚠️  NO DATA"}); continue

        row = df[df.index <= latest_date].tail(1)
        if row.empty:
            results.append({"ticker": ticker, "status": "⚠️  NO DATA"}); continue

        cp        = float(row["close"].iloc[0])
        gain      = (cp - ep) / ep
        hold_days = (latest_date - entry).days
        pnl_vnd   = shares * (cp - ep) * 1000
        max_gain  = (1.07 ** max(hold_days * 0.71, 1)) - 1

        if gain > max_gain:
            status = "⚠️  DATA GAP — skip"
        elif gain >= STOCK_TP_PCT:
            status = f"🔴 SELL at open — TP hit (+{gain*100:.1f}%)"
        elif gain >= STOCK_TP_PCT * 0.80:
            status = f"🟡 Near TP ({gain*100:+.1f}%) — watch closely"
        elif gain <= -0.15:
            status = f"🔴 Large loss ({gain*100:.1f}%) — review"
        else:
            status = f"✅ Hold ({gain*100:+.1f}%)"

        results.append({
            "ticker": ticker, "tranche": pos.get("tranche","?"),
            "shares": shares, "entry_price": ep, "current_price": cp,
            "gain_pct": gain*100, "pnl_vnd": pnl_vnd,
            "hold_days": hold_days, "tp_price": ep*(1+STOCK_TP_PCT),
            "status": status,
        })
    return results


# ═════════════════════════════════════════════════════════════════
# VISUALISATION — CYCLE LINE (SPREAD)
# ═════════════════════════════════════════════════════════════════

def plot_cycle_line_chart(sector_signals, output_path, focus_sector=None, show_chart=False):
    """
    Plot sector cycle lines (spread = peak% - drown%) so you can visually see
    when each sector is in DROWNING vs PEAKING phases.

    - Red ▼ markers: transition into DROWNING
    - Orange ▲ markers: transition into PEAKING
    - Dashed horizontal lines: key spread thresholds
    """
    if not sector_signals:
        return "No sector signals available for charting"

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return "Matplotlib not available, skipped chart"

    colors = {
        "Banks": "#1f77b4",
        "Food & Beverage": "#2ca02c",
        "Basic Resources": "#9467bd",
        "Real Estate": "#17becf",
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    # Trim to latest 2 years so the chart stays readable
    cutoff_2y = pd.Timestamp.today() - pd.DateOffset(years=2)

    for sec in sorted(sector_signals.keys()):
        sig = sector_signals[sec]
        if sig is None or sig.empty:
            continue

        sig    = sig[sig.index >= cutoff_2y]
        if sig.empty:
            continue

        spread = sig["spread"]
        state = sig["state"]
        transitions = state.ne(state.shift(1))

        line_w = 2.4 if focus_sector and sec == focus_sector else 1.4
        alpha = 1.0 if focus_sector and sec == focus_sector else 0.80
        col = colors.get(sec, None)

        ax.plot(spread.index, spread.values, lw=line_w, alpha=alpha,
                color=col, label=sec)

        drowning_points = spread[(state == "DROWNING") & transitions]
        peaking_points = spread[(state == "PEAKING") & transitions]

        if not drowning_points.empty:
            ax.scatter(drowning_points.index, drowning_points.values,
                       marker="v", color="#d62728", s=26, zorder=4)
        if not peaking_points.empty:
            ax.scatter(peaking_points.index, peaking_points.values,
                       marker="^", color="#ff7f0e", s=26, zorder=4)

    ax.axhline(0, color="#555", ls="--", lw=1.0, alpha=0.7)
    ax.axhline(SPREAD_EXIT, color="#d62728", ls=":", lw=0.9, alpha=0.75)
    ax.axhline(SPREAD_PEAK, color="#ff7f0e", ls=":", lw=0.9, alpha=0.75)
    if SPREAD_HIGH_EXIT:
        ax.axhline(SPREAD_HIGH_EXIT, color="#8c564b", ls=":", lw=0.8, alpha=0.6)

    title_focus = f" (focus: {focus_sector})" if focus_sector else ""
    ax.set_title(f"Sector cycle line (spread) — DROWNING vs PEAKING timing{title_focus}")
    ax.set_ylabel("Spread = Peak% - Drown%")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.20)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show_chart:
        plt.show()
    else:
        plt.close(fig)

    return f"Cycle chart saved: {output_path}"


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def demand_scanner(stock_data, sector_data, latest_date, window=20, top_n=8):
    """
    Scan all stocks for recent demand signals over the last `window` trading days.

    Per stock: full-candle score (body_pct × body_ratio, recency-weighted),
    momentum over the window, and volume ratio (recent vs baseline).

    Returns:
      sector_heat : dict  sector → avg full-candle score across all stocks
      top_stocks  : list of dicts sorted by full-candle score (all sectors)
    """
    ticker_to_sector = {}
    for sec, stocks in sector_data.items():
        for t in stocks:
            ticker_to_sector[t] = sec

    MIN_DAYS_TRADED   = int(window * 0.75)   # must trade ≥75% of window days
    MIN_CANDLE_TURNOVER = MIN_LIQUIDITY_VND  # best candle day turnover ≥ 1B VND

    rows = []
    for ticker, df in stock_data.items():
        sec = ticker_to_sector.get(ticker)
        if sec is None:
            continue
        hist = df[df.index <= latest_date]
        if len(hist) < window + 5:
            continue
        if "high" not in hist.columns or "low" not in hist.columns:
            continue

        recent = hist.tail(window)

        # ── Liquidity filters ──────────────────────────────────────
        # 1. Median daily turnover over last 60d must meet threshold
        med_turnover = (hist["close"] * hist["volume"] * 1000).tail(60).median()
        if med_turnover < MIN_LIQUIDITY_VND:
            continue

        # 2. Stock must have traded on most days in the window (no ghost stocks)
        days_with_vol = (recent["volume"] > 0).sum()
        if days_with_vol < MIN_DAYS_TRADED:
            continue

        # Full-candle score: body_pct × body_ratio per bullish bar, recency-weighted
        scores = []
        best_body_pct   = 0.0
        best_turnover   = 0.0
        for _, row in recent.iterrows():
            body     = row["close"] - row["open"]
            rng      = row["high"]  - row["low"]
            turnover = row["close"] * row["volume"] * 1000
            if body > 0 and rng > 0 and row["open"] > 0:
                s = (body / row["open"]) * (body / rng)
                scores.append(s)
                if s > best_body_pct:
                    best_body_pct = s
                    best_turnover = turnover
            else:
                scores.append(0.0)

        # 3. Best candle day must itself have meaningful turnover
        if best_turnover < MIN_CANDLE_TURNOVER:
            continue

        weights  = np.arange(1, len(scores) + 1, dtype=float)
        fc_score = float(np.average(scores, weights=weights))

        # Price momentum over the window
        p_start = hist.iloc[-(window+1)]["close"] if len(hist) > window else hist.iloc[0]["close"]
        p_end   = hist.iloc[-1]["close"]
        mom     = (p_end / p_start - 1) * 100 if p_start > 0 else 0.0

        # Volume ratio: avg volume in window vs prior 60-day baseline
        baseline_vol = hist.iloc[-(window+60):-(window)]["volume"].mean() if len(hist) >= window + 60 else hist["volume"].mean()
        recent_vol   = recent["volume"].mean()
        vol_ratio    = recent_vol / baseline_vol if baseline_vol > 0 else 1.0

        rows.append({
            "ticker":    ticker,
            "sector":    sec,
            "fc_score":  fc_score,
            "best_candle": best_body_pct,
            "mom_pct":   mom,
            "vol_ratio": vol_ratio,
            "price":     p_end,
        })

    if not rows:
        return {}, []

    rows.sort(key=lambda x: -x["fc_score"])

    # Sector heat: average fc_score weighted by top half of stocks per sector
    sector_heat = {}
    for sec in sector_data:
        sec_rows = [r for r in rows if r["sector"] == sec]
        if not sec_rows:
            continue
        top_half = sec_rows[:max(len(sec_rows)//2, 1)]
        sector_heat[sec] = np.mean([r["fc_score"] for r in top_half])

    return sector_heat, rows[:top_n * len(sector_data)]   # top N per sector worth


def main():
    today_str = date.today().strftime("%Y-%m-%d")
    lines = []

    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    def p(s=""):
        try:
            print(s)
        except UnicodeEncodeError:
            print(str(s).encode("ascii", errors="replace").decode("ascii"))
        lines.append(s)

    stock_data, sector_data, latest_date = load_data()
    today_date = latest_date  # use latest data date as "today"

    # Extract strategy-sector universe once — used to filter both fundamental
    # and factor data loads (192 tickers instead of all ~1566)
    _strategy_tickers = [t for stocks in sector_data.values() for t in stocks.keys()]

    global _FUND_DATA, _QFEAT
    if FUNDAMENTAL_FILTER_ENABLED:
        _FUND_DATA = load_fundamental_data(symbols=_strategy_tickers)

    if FACTOR_SELECTION_ENABLED and _FACTOR_RANKER_AVAILABLE:
        print("Loading quarterly factor features...")
        _QFEAT = build_factor_features(symbols=_strategy_tickers)
    elif FACTOR_SELECTION_ENABLED and not _FACTOR_RANKER_AVAILABLE:
        print("  [FACTOR] factor_stock_ranker.py not found — factor selection disabled")

    # Build signals
    sector_signals = {}
    for sec, stocks in sector_data.items():
        sig = build_sector_signal(stocks)
        if sig is not None and not sig.empty:
            sector_signals[sec] = sig

    cycle_chart_msg = None
    if SAVE_CYCLE_CHART:
        cycle_chart_file = os.path.join(CHARTS_DIR, f"sector_cycle_line_{today_str}.png")
        cycle_chart_msg = plot_cycle_line_chart(
            sector_signals,
            output_path=cycle_chart_file,
            focus_sector=HELD_SECTOR,
            show_chart=SHOW_CYCLE_CHART,
        )

    p()
    p("═" * 70)
    p(f"  DAILY SIGNAL REPORT  —  {today_str}")
    p(f"  Latest data: {latest_date.date()}  |  Capital: {TOTAL_CAPITAL_VND/1e6:.1f}M VND")
    p(f"  Strategy: MOM_BOT50 + TRANCHE2 (50%/50%, 3 trading days apart)")
    if FUNDAMENTAL_FILTER_ENABLED:
        def _fmt_crit(crit):
            parts = []
            if "roe_min"                    in crit: parts.append(f"ROE>{crit['roe_min']:.0f}%")
            if "revenue_growth_min"         in crit: parts.append(f"RevGrowth>{crit['revenue_growth_min']*100:.0f}%")
            if "revenue_or_asset_growth_min"in crit: parts.append(f"RevOrAssetGrowth>{crit['revenue_or_asset_growth_min']*100:.0f}%")
            if "ocf_min"                    in crit: parts.append(f"OCF/NP>{crit['ocf_min']:.1f}x")
            return " & ".join(parts) if parts else "—"
        crit_str = " | ".join(f"{s}: {_fmt_crit(c)}" for s, c in SECTOR_FUND_CRITERIA.items())
        fund_syms = len(_FUND_DATA) if _FUND_DATA else 0
        p(f"  Fundamental filter: ON ({fund_syms} symbols) — {crit_str}")
    else:
        p(f"  Fundamental filter: OFF")
    p("═" * 70)

    # ─── 0. DEMAND SCANNER (last 20 trading days ≈ 1 month) ───────
    SCAN_WINDOW = 20
    sector_heat, hot_stocks = demand_scanner(
        stock_data, sector_data, latest_date, window=SCAN_WINDOW, top_n=6)

    p()
    p("━" * 70)
    p(f"  [0] DEMAND SCANNER  —  last {SCAN_WINDOW} trading days")
    p("━" * 70)
    if HELD_SECTOR:
        p(f"  (skipped — currently holding {HELD_SECTOR}. Run in cash to see full scanner.)")
    else:
        p(f"  Ranks stocks by full-candle score (conviction buying: large body, no wicks).")
        p(f"  Higher score = stronger institutional demand recently.")
    if not HELD_SECTOR:
        p()

        # Sector heat bar
        p(f"  SECTOR HEAT  (avg full-candle score, top-50% of stocks per sector)")
        max_heat = max(sector_heat.values()) if sector_heat else 1.0
        for sec in sorted(sector_heat, key=lambda s: -sector_heat[s]):
            h     = sector_heat[sec]
            bar   = "█" * max(1, int(h / max(max_heat, 0.001) * 20))
            arrow = "🔥" if h == max(sector_heat.values()) else "  "
            p(f"  {arrow} {sec:<22}  {h:.4f}  {bar}")
        p()

        # Pre-compute quality flags for demand scanner (✓ = buyable quality)
        _scan_quality = {}
        if _FACTOR_RANKER_AVAILABLE and _QFEAT is not None:
            # Group hot_stocks by sector to call quality_flags once per sector
            _scan_tickers_by_sec = {}
            for r in hot_stocks:
                _scan_tickers_by_sec.setdefault(r["sector"], []).append(r["ticker"])
            for _sec, _tks in _scan_tickers_by_sec.items():
                _flags = quality_flags(_tks, latest_date, _QFEAT, sector=_sec)
                _scan_quality.update(_flags)

        # Top stocks table
        p(f"  TOP STOCKS BY DEMAND  (sorted by full-candle score)")
        p(f"  {'':2} {'Ticker':<7} {'Sector':<16} {'FC Score':>9} {'Best Candle':>11} {'1M Mom%':>8} {'Vol Ratio':>10}  Q")
        p(f"  {'-'*72}")
        shown = 0
        last_sec = None
        for r in hot_stocks:
            if shown >= 30:
                break
            if r["sector"] != last_sec:
                if last_sec is not None:
                    p()
                last_sec = r["sector"]
            tag  = "🔥" if r["fc_score"] >= 0.01 else "  "
            ok   = _scan_quality.get(r["ticker"])
            qflag = " ✓" if ok else (" ✗" if ok is not None else " ?")
            p(f"  {tag}{r['ticker']:<7} {r['sector']:<16} {r['fc_score']:>9.4f} "
              f"{r['best_candle']*100:>10.1f}%  {r['mom_pct']:>+7.1f}%  {r['vol_ratio']:>8.2f}x {qflag}")
            shown += 1
        p()
        p(f"  Q flag: ✓ = passes quality filter (profitable+D/E ok)  ✗ = zombie/loss-maker  ? = no data")
        p(f"  Best candle: largest single-day body_pct × body_ratio in the window.")
        p(f"  Vol ratio  : recent {SCAN_WINDOW}d avg volume vs prior 60d baseline.")

    # ─── 1. SECTOR DASHBOARD ──────────────────────────────────────
    p()
    p("━" * 70)
    p("  [1] SECTOR DASHBOARD")
    p("━" * 70)
    p(f"  {'Sector':<22} {'State':<11} {'Spread':>8} {'Vel':>5} {'Rising':>7} {'Score':>7}  {'Baseline':>10}  {'Early V3':>10}  {'Demand':>10}")
    p(f"  {'-'*103}")
    sector_rows = {}
    # Pre-compute sector heat and adaptive threshold for all sectors
    sector_heat_now   = {}
    sector_heat_thresh = {}
    if DEMAND_EARLY_ENABLED:
        for sec in sector_data:
            sector_heat_now[sec] = _demand_heat(
                sector_data.get(sec, {}), latest_date,
                window=DEMAND_HEAT_WINDOW)
            sector_heat_thresh[sec] = _adaptive_heat_threshold(
                sector_data.get(sec, {}), latest_date)
    for sec in sorted(sector_signals):
        sig  = sector_signals[sec]
        row  = sig.iloc[-1]
        sp   = float(row["spread"])
        st   = row["state"]
        sc   = float(row["score"]) if not np.isnan(row["score"]) else 0.0
        thr  = get_threshold(row, sec)
        vel  = float(row["velocity"]) if not np.isnan(row["velocity"]) else 0.0
        rs   = int(row.get("rising_streak", 0))
        flag = {"CRASH":"🔴","DROWNING":"🔴","RECOVERY":"🟢",
                "LEADING":"🟢","PEAKING":"🟡"}.get(st,"⚪")
        base_str  = f"✅ ENTRY"  if entry_ok(row, sec)       else f"needs >{thr:.1f}"
        early_str = f"⚡ EARLY"  if entry_ok_early(row, sec) else \
                    (f"{EARLY_V3_VEL_DAYS-rs}d more" if st=="DROWNING" and sp>EARLY_V3_THRESHOLD and rs>0
                     else "—")
        heat       = sector_heat_now.get(sec, 0.0)
        heat_thr   = sector_heat_thresh.get(sec, DEMAND_HEAT_THRESHOLD)
        demand_ok  = (DEMAND_EARLY_ENABLED and st == "DROWNING"
                      and sp >= DEMAND_HEAT_SPREAD_FLOOR
                      and heat >= heat_thr)
        demand_str = (f"🔥 {heat:.4f}" if demand_ok
                      else f"{heat:.4f}" if heat > 0
                      else "—")
        p(f"  {sec:<22} {flag} {st:<9}  {sp:>7.1f}  {vel:>+4.1f}  {rs:>5}d  {sc:>7.2f}  {base_str:>10}  {early_str:>10}  {demand_str:>10}")
        sector_rows[sec] = row
    p()
    p(f"  Current position: {HELD_SECTOR or 'CASH'}"
      + (f"  (T1 entered {ENTRY_DATE}"
         + (", T2 done ✓" if TRANCHE2_DONE else f", T2 pending")
         + ")" if HELD_SECTOR else ""))
    if cycle_chart_msg:
        p(f"  📈 {cycle_chart_msg}")

    # ─── 2. EXIT CHECK ────────────────────────────────────────────
    if HELD_SECTOR:
        p()
        p("━" * 70)
        p("  [2] EXIT CHECK")
        p("━" * 70)
        if HELD_SECTOR not in sector_signals:
            p(f"  ⚠️  No signal data for {HELD_SECTOR}")
        else:
            row = sector_signals[HELD_SECTOR].iloc[-1]
            should_exit, reason = exit_needed(row)
            if should_exit:
                p(f"  🔴 SELL SIGNAL — exit {HELD_SECTOR}")
                p(f"     Reason: {reason}")
                p(f"     Action: SELL ALL holdings at tomorrow's open (T+1)")
                p(f"     Cancel any pending T2 if not yet executed")
            else:
                sp = float(row["spread"]); st = row["state"]
                sc = float(row["score"]) if not np.isnan(row["score"]) else 0.0
                p(f"  ✅ HOLD — {HELD_SECTOR} signal still intact")
                p(f"     Spread: {sp:.1f}  |  State: {st}  |  Score: {sc:.2f}")
                p(f"     Exit triggers: spread < {SPREAD_EXIT} | PEAKING | mom < {MOMENTUM_FLOOR*100:.0f}%")

    # ─── 3. TRANCHE 2 STATUS ─────────────────────────────────────
    if HELD_SECTOR and not TRANCHE2_DONE:
        p()
        p("━" * 70)
        p("  [3] TRANCHE 2 STATUS")
        p("━" * 70)

        entry_ts  = pd.Timestamp(ENTRY_DATE) if ENTRY_DATE else None
        days_held = (today_date - entry_ts).days if entry_ts else 0

        # Get today's sector spread and return
        row_t2    = sector_signals.get(HELD_SECTOR, pd.DataFrame())
        row_t2    = row_t2.iloc[-1] if not row_t2.empty else None
        spread_now= float(row_t2["spread"]) if row_t2 is not None else 0
        s_ret_now = float(row_t2["s_ret"])  if row_t2 is not None else 0

        t2_cash = None

        if row_t2 is not None and spread_now < 0:
            p(f"  🔴 SKIP T2 — spread turned negative ({spread_now:.1f})")
            p(f"     Sector reversed. Do NOT deploy T2 — signal failed.")
            p(f"     Set TRANCHE2_DONE = True to suppress this alert.")

        elif T2_MODE == "DIP":
            t2_deadline_cal = int(DIP_MAX_WAIT * 7 / 5) + 1
            dip_today    = s_ret_now <= DIP_THRESHOLD
            deadline_hit = days_held >= t2_deadline_cal
            if dip_today:
                t2_cash = TOTAL_CAPITAL_VND * TRANCHE2_FRAC
                p(f"  🟢 DEPLOY T2 NOW — DIP TRIGGER")
                p(f"     Sector return today: {s_ret_now*100:+.2f}% (≤ {DIP_THRESHOLD*100:.1f}%)")
                p(f"     Days since T1: {days_held}  |  Spread: {spread_now:.1f} ✓")
                p(f"     Buy at tomorrow's open")
            elif deadline_hit:
                t2_cash = TOTAL_CAPITAL_VND * TRANCHE2_FRAC
                p(f"  🟡 DEPLOY T2 — DEADLINE ({DIP_MAX_WAIT} trading days, no dip came)")
                p(f"     Sector ran straight up — buy at market tomorrow")
            else:
                days_remaining = max(0, t2_deadline_cal - days_held)
                p(f"  ⏳ WAITING FOR DIP — T2 not yet triggered")
                p(f"     Days since T1: {days_held}  |  Spread: {spread_now:.1f}  |  Sector ret: {s_ret_now*100:+.2f}%")
                p(f"     Trigger: sector down ≥{abs(DIP_THRESHOLD)*100:.1f}% → buy next open")
                p(f"     Deadline: {days_remaining} calendar day(s) left then buy at market")

        else:
            # TRANCHE2 (default): fixed 3 trading days ≈ 4-5 calendar days
            t2_cal_days = int(T2_FIXED_DAYS * 7 / 5) + 1
            if days_held < t2_cal_days:
                p(f"  ⏳ T2 not yet due — {days_held} calendar day(s) since T1")
                p(f"     T2 fires after ~{t2_cal_days} calendar days ({T2_FIXED_DAYS} trading days)")
                p(f"     Check spread on T2 day: if spread < 0, skip T2")
            else:
                t2_cash = TOTAL_CAPITAL_VND * TRANCHE2_FRAC
                p(f"  🟢 DEPLOY TRANCHE 2 — buy at tomorrow's open")
                p(f"     Days since T1: {days_held}  |  Spread: {spread_now:.1f} ✓")

        # Show stock list only when T2 should fire today
        if t2_cash is not None and spread_now >= 0:
            p()
            stocks_t2 = select_stocks(
                sector_data.get(HELD_SECTOR, {}), t2_cash, today_date,
                sector=HELD_SECTOR, stock_data_all=stock_data)
            if stocks_t2:
                trigger_lbl = "DIP trigger" if dip_today else f"deadline day {DIP_MAX_WAIT}"
                p(f"  Stocks to buy for T2 tomorrow ({trigger_lbl}):")
                p(f"  {'#':<4} {'Ticker':<8} {'Price':>7} {'Shares':>8} "
                  f"{'Cost (M)':>9} {'Mom20d':>8}  TP @    Kijun entry guidance")
                p(f"  {'-'*90}")
                for i, s in enumerate(stocks_t2, 1):
                    kij   = s.get("kijun_info", {})
                    ready = "✅ BUY " if kij.get("ready", True) else "⏳ WAIT"
                    klbl  = kij.get("label", "—")
                    p(f"  {i:<4} {s['ticker']:<8} {s['price']:>7.2f}  {s['shares']:>8,}"
                      f"  {s['cost_vnd']/1e6:>8.2f}M  {s['mom_20d']*100:>+7.1f}%"
                      f"  {s['tp_price']:>6.2f}  {ready} {klbl}")
                total = sum(s["cost_vnd"] for s in stocks_t2)
                p(f"  {'':4} {'TOTAL':<8} {'':>7}  {sum(s['shares'] for s in stocks_t2):>8,}"
                  f"  {total/1e6:>8.2f}M")
                p()
                p(f"  After executing: set TRANCHE2_DONE = True and add stocks to PORTFOLIO")
            else:
                p(f"  ⚠️  No eligible stocks found for T2 — check liquidity")

    # ─── 4. TP ALERTS ────────────────────────────────────────────
    if PORTFOLIO:
        p()
        p("━" * 70)
        p("  [4] TAKE-PROFIT ALERTS  (sell individual stocks at +25%)")
        p("━" * 70)
        holdings = check_portfolio(stock_data, today_date)
        has_tp   = any("SELL" in h.get("status","") for h in holdings if "gain_pct" in h)

        if has_tp:
            p(f"  ⚠️  STOCKS TO SELL AT TOMORROW'S OPEN:")
        p()
        p(f"  {'Ticker':<8} {'T':<3} {'Shares':>7} {'Entry':>7} "
          f"{'Now':>7} {'Gain':>7} {'P&L':>8}  TP @   Status")
        p(f"  {'-'*78}")
        total_pnl = 0
        for h in holdings:
            if "gain_pct" not in h:
                p(f"  {h['ticker']:<8}  —  {h['status']}"); continue
            total_pnl += h["pnl_vnd"]
            p(f"  {h['ticker']:<8} {h['tranche']:<3} {h['shares']:>7,}"
              f"  {h['entry_price']:>7.2f}  {h['current_price']:>7.2f}"
              f"  {h['gain_pct']:>+6.1f}%  {h['pnl_vnd']/1e6:>7.2f}M"
              f"  {h['tp_price']:>5.2f}  {h['status']}")
        p(f"  {'─'*78}")
        p(f"  {'Total unrealised P&L':>48}  {total_pnl/1e6:>7.2f}M")

    # ─── 5. ENTRY SIGNAL ─────────────────────────────────────────
    p()
    p("━" * 70)
    p("  [5] ENTRY SIGNAL")
    p("━" * 70)

    if HELD_SECTOR:
        p(f"  ⏸  Currently holding {HELD_SECTOR} — no new entry until exit")
    else:
        # ── 5A: BASELINE signal ───────────────────────────────────
        candidates = {sec: float(row["score"])
                      for sec, row in sector_rows.items()
                      if entry_ok(row, sec)}

        p("  ── BASELINE (confirmed recovery) " + "─" * 37)
        if not candidates:
            p("  ⏳ NO BASELINE SIGNAL — stay in cash")
            p("     Spread has not yet crossed the vol-adjusted threshold")
        else:
            best_sec   = max(candidates, key=candidates.get)
            best_row   = sector_rows[best_sec]
            best_score = candidates[best_sec]
            t1_cash    = TOTAL_CAPITAL_VND * TRANCHE1_FRAC
            t2_cash    = TOTAL_CAPITAL_VND * TRANCHE2_FRAC

            p(f"  🟢 BUY SIGNAL — {best_sec}")
            p(f"     State: {best_row['state']}  |  Score: {best_score:.2f}"
              f"  |  Spread: {float(best_row['spread']):.1f}")
            p()
            p(f"  EXECUTION PLAN (TRANCHE2):")
            p(f"  ┌─ T1: deploy {TRANCHE1_FRAC*100:.0f}% = {t1_cash/1e6:.1f}M VND → buy at TOMORROW's open")
            p(f"  └─ T2: deploy {TRANCHE2_FRAC*100:.0f}% = {t2_cash/1e6:.1f}M VND → {"next dip (≥"+str(abs(DIP_THRESHOLD)*100)+"%) or day "+str(DIP_MAX_WAIT) if T2_MODE=="DIP" else str(T2_FIXED_DAYS)+" trading days later"}")
            p(f"         (check spread on T2 day — if spread < 0, skip T2)")
            p()

            stocks = select_stocks(sector_data.get(best_sec, {}), t1_cash, today_date,
                                   sector=best_sec, stock_data_all=stock_data)

            if stocks:
                _bm = SECTOR_STOCK_SELECTION.get(best_sec, "MOM_BOT50")
                _sort_lbl = ("full-candle score ↓" if (_bm == "VOL_LEADERS" and VOL_LEADERS_USE_FULL_CANDLE)
                             else "vol_score ↓ (leaders)" if _bm in ("VOL_LEADERS","VOL_RANK")
                             else "20d momentum (most oversold)")
                p(f"  STOCKS TO BUY TOMORROW (T1 — {t1_cash/1e6:.1f}M VND):")
                p(f"  Method: {_bm}  |  Sorted by {_sort_lbl}")
                p()
                p(f"  {'#':<4} {'Ticker':<8} {'Price':>7} {'Shares':>8} "
                  f"{'Cost (M)':>9} {'Mom20d':>8}  TP @    Kijun entry guidance")
                p(f"  {'-'*90}")
                for i, s in enumerate(stocks, 1):
                    kij   = s.get("kijun_info", {})
                    ready = "✅ BUY " if kij.get("ready", True) else "⏳ WAIT"
                    klbl  = kij.get("label", "—")
                    note  = "  ← most oversold" if i == 1 else ""
                    p(f"  {i:<4} {s['ticker']:<8} {s['price']:>7.2f}  {s['shares']:>8,}"
                      f"  {s['cost_vnd']/1e6:>8.2f}M  {s['mom_20d']*100:>+7.1f}%"
                      f"  {s['tp_price']:>6.2f}  {ready} {klbl}{note}")
                total = sum(s["cost_vnd"] for s in stocks)
                p(f"  {'':4} {'TOTAL':<8} {'':>7}  {sum(s['shares'] for s in stocks):>8,}"
                  f"  {total/1e6:>8.2f}M")
                p()
                p(f"  After buying:")
                p(f"    • Set HELD_SECTOR = \"{best_sec}\"")
                p(f"    • Set ENTRY_DATE  = \"{today_str}\" (or actual execution date)")
                p(f"    • Set TRANCHE2_DONE = False")
                p(f"    • Add all stocks to PORTFOLIO with tranche: \"T1\"")
                p(f"    • T2 will appear in [3] above ~3 trading days from now")
            else:
                p(f"  ⚠️  No eligible stocks found — check liquidity or data")

            if len(candidates) > 1:
                others = {s:v for s,v in candidates.items() if s != best_sec}
                p()
                p(f"  Other sectors with valid baseline signals:")
                for sec, score in sorted(others.items(), key=lambda x: -x[1]):
                    row = sector_rows[sec]
                    p(f"    {sec:<22}  score={score:.2f}  spread={float(row['spread']):.1f}"
                      f"  state={row['state']}")

        # ── 5B: EARLY_V3 signal ───────────────────────────────────
        p()
        p("  ── EARLY V3 (pre-signal, spread > -10, 3d rising) " + "─" * 19)
        p("  Logic: sector still DROWNING but spread rising consistently.")
        p("  OOS validated: better Sharpe 4/5 windows vs baseline.")
        p("  Risk: lower win rate (~42%) — more false starts in choppy markets.")
        p()

        early_candidates = {sec: row
                            for sec, row in sector_rows.items()
                            if entry_ok_early(row, sec)}

        if not early_candidates:
            # Show progress toward early signal for DROWNING sectors
            watching = []
            for sec, row in sector_rows.items():
                if row["state"] != "DROWNING": continue
                sp = float(row["spread"])
                rs = int(row.get("rising_streak", 0))
                if sp <= EARLY_V3_THRESHOLD: continue   # too deep
                days_needed = max(0, EARLY_V3_VEL_DAYS - rs)
                watching.append((sec, sp, rs, days_needed,
                                 float(row["velocity"]) if not np.isnan(row["velocity"]) else 0))

            if watching:
                p("  ⏳ NO EARLY SIGNAL YET — sectors approaching:")
                p(f"  {'Sector':<22} {'Spread':>8} {'Rising':>8} {'Need':>8}  Velocity")
                p(f"  {'-'*60}")
                for sec, sp, rs, nd, vel in sorted(watching, key=lambda x: x[3]):
                    bar   = "█" * rs + "░" * max(0, EARLY_V3_VEL_DAYS - rs)
                    ready = "→ READY NEXT RUN" if nd == 0 else f"→ {nd} more day(s)"
                    p(f"  {sec:<22} {sp:>7.1f}  {rs:>6}d  {bar}  vel={vel:>+5.2f}  {ready}")
            else:
                p("  ⏳ NO EARLY SIGNAL — no DROWNING sector near threshold")
        else:
            best_early = max(early_candidates,
                             key=lambda s: float(early_candidates[s]["spread"]))
            t1_cash = TOTAL_CAPITAL_VND * TRANCHE1_FRAC
            t2_cash = TOTAL_CAPITAL_VND * TRANCHE2_FRAC

            for sec, row in sorted(early_candidates.items(),
                                   key=lambda x: -float(x[1]["spread"])):
                sp = float(row["spread"])
                rs = int(row.get("rising_streak", 0))
                vel= float(row["velocity"]) if not np.isnan(row["velocity"]) else 0
                star = "  ← BEST" if sec == best_early else ""
                p(f"  ⚡ EARLY SIGNAL — {sec}{star}")
                p(f"     State: DROWNING  |  Spread: {sp:.1f}  |  Threshold: {EARLY_V3_THRESHOLD}"
                  f"  |  Rising: {rs} consecutive days  |  Velocity: {vel:+.2f}")

            p()
            p(f"  Best early candidate: {best_early}  (highest spread = closest to recovery)")
            p(f"  EXECUTION PLAN (TRANCHE2 — same as baseline):")
            p(f"  ┌─ T1: {TRANCHE1_FRAC*100:.0f}% = {t1_cash/1e6:.1f}M VND → buy at TOMORROW's open")
            p(f"  └─ T2: {TRANCHE2_FRAC*100:.0f}% = {t2_cash/1e6:.1f}M VND → {"next dip or day "+str(DIP_MAX_WAIT) if T2_MODE=="DIP" else str(T2_FIXED_DAYS)+" trading days later"}")
            p(f"         (skip T2 if spread < 0 on T2 day — more likely with early entry)")
            p()
            p(f"  ⚠️  NOTE: You are entering BEFORE confirmed recovery.")
            p(f"     The spread is still negative — baseline has NOT fired yet.")
            p(f"     Expected: lower entry price, lower win rate (~42%), higher reward if right.")

            # Show stock lists for ALL firing sectors so you can compare
            for sec, row in sorted(early_candidates.items(),
                                   key=lambda x: -float(x[1]["spread"])):
                star = "  ← BEST (execute this one)" if sec == best_early else ""
                p()
                p(f"  STOCKS — {sec} (T1 — {t1_cash/1e6:.1f}M VND){star}")
                p(f"  {'#':<4} {'Ticker':<8} {'Price':>7} {'Shares':>8} "
                  f"{'Cost (M)':>9} {'Mom20d':>8}  TP @    Kijun entry guidance")
                p(f"  {'-'*90}")
                stocks = select_stocks(sector_data.get(sec, {}), t1_cash, today_date,
                                       sector=sec, stock_data_all=stock_data)
                if stocks:
                    for i, s in enumerate(stocks, 1):
                        kij   = s.get("kijun_info", {})
                        ready = "✅ BUY " if kij.get("ready", True) else "⏳ WAIT"
                        klbl  = kij.get("label", "—")
                        note  = "  ← most oversold" if i == 1 else ""
                        p(f"  {i:<4} {s['ticker']:<8} {s['price']:>7.2f}  {s['shares']:>8,}"
                          f"  {s['cost_vnd']/1e6:>8.2f}M  {s['mom_20d']*100:>+7.1f}%"
                          f"  {s['tp_price']:>6.2f}  {ready} {klbl}{note}")
                    total = sum(s["cost_vnd"] for s in stocks)
                    p(f"  {'':4} {'TOTAL':<8} {'':>7}  {sum(s['shares'] for s in stocks):>8,}"
                      f"  {total/1e6:>8.2f}M")
                else:
                    p(f"  ⚠️  No eligible stocks found — check liquidity or data")

        # ── 5C: DEMAND_EARLY signal ───────────────────────────────
        if DEMAND_EARLY_ENABLED:
            p()
            mode_str = (f"adaptive {DEMAND_HEAT_PERCENTILE}th-pct of 252d history"
                        if DEMAND_HEAT_USE_ADAPTIVE
                        else f"fixed ≥ {DEMAND_HEAT_THRESHOLD:.3f}")
            p(f"  ── DEMAND EARLY (stock flow — heat threshold: {mode_str}) " + "─" * 5)
            p("  Logic: liquid stocks show conviction buying (big full candles)")
            p("  while sector is still DROWNING. Enters earlier than Early V3.")
            p("  WFO validated: 5/5 windows beat baseline Sharpe.")
            p()

            demand_candidates = {}
            for sec, row in sector_rows.items():
                if row is None: continue
                sp = float(row["spread"])
                if row["state"] != "DROWNING": continue
                if sp < DEMAND_HEAT_SPREAD_FLOOR: continue
                if entry_ok(row, sec) or entry_ok_early(row, sec): continue
                heat     = sector_heat_now.get(sec, 0.0)
                heat_thr = sector_heat_thresh.get(sec, DEMAND_HEAT_THRESHOLD)
                if heat >= heat_thr:
                    demand_candidates[sec] = heat

            if not demand_candidates:
                # Show current heat for all DROWNING sectors
                watching = [(sec, sector_heat_now.get(sec, 0.0),
                             sector_heat_thresh.get(sec, DEMAND_HEAT_THRESHOLD))
                            for sec, row in sector_rows.items()
                            if row is not None and row["state"] == "DROWNING"]
                if watching:
                    p("  ⏳ NO DEMAND SIGNAL — heat below adaptive threshold:")
                    p(f"  {'Sector':<22} {'Heat':>8} {'Thresh':>8}  Progress")
                    p(f"  {'-'*60}")
                    for sec, h, thr in sorted(watching, key=lambda x: -x[1]):
                        bar  = "█" * int(min(h / max(thr, 1e-6), 1.0) * 15)
                        gap  = max(0.0, thr - h)
                        p(f"  {sec:<22} {h:>8.4f} {thr:>8.4f}  {bar}  need +{gap:.4f}")
                else:
                    p("  ⏳ NO DROWNING SECTORS — demand signal not applicable")
            else:
                best_demand = max(demand_candidates, key=demand_candidates.get)
                t1_cash = TOTAL_CAPITAL_VND * TRANCHE1_FRAC
                t2_cash = TOTAL_CAPITAL_VND * TRANCHE2_FRAC
                for sec, heat in sorted(demand_candidates.items(),
                                        key=lambda x: -x[1]):
                    star = "  ← BEST" if sec == best_demand else ""
                    row  = sector_rows[sec]
                    p(f"  🔥 DEMAND SIGNAL — {sec}{star}")
                    thr_disp = sector_heat_thresh.get(sec, DEMAND_HEAT_THRESHOLD)
                    p(f"     Heat: {heat:.4f} (threshold {thr_disp:.4f} adaptive)  |  "
                      f"Spread: {float(row['spread']):.1f}  |  State: DROWNING")
                p()
                p(f"  Best candidate: {best_demand}  (highest heat score)")
                p(f"  EXECUTION PLAN: same as baseline — T1 tomorrow, T2 in 3 days")
                p(f"  ⚠️  Entering BEFORE breadth confirmation — higher risk, earlier price.")

                # Show stock lists for ALL firing demand sectors
                for sec, heat in sorted(demand_candidates.items(),
                                        key=lambda x: -x[1]):
                    star = "  ← BEST (execute this one)" if sec == best_demand else ""
                    thr_disp = sector_heat_thresh.get(sec, DEMAND_HEAT_THRESHOLD)
                    p()
                    p(f"  STOCKS — {sec}  heat={heat:.4f} thr={thr_disp:.4f}"
                      f"  (T1 — {t1_cash/1e6:.1f}M VND){star}")
                    p(f"  {'#':<4} {'Ticker':<8} {'Price':>7} {'Shares':>8} "
                      f"{'Cost (M)':>9} {'Mom20d':>8}  TP @    Kijun entry guidance")
                    p(f"  {'-'*90}")
                    stocks = select_stocks(sector_data.get(sec, {}), t1_cash,
                                           today_date, sector=sec,
                                           stock_data_all=stock_data,
                                           demand_mode=True)
                    if stocks:
                        for i, s in enumerate(stocks, 1):
                            kij   = s.get("kijun_info", {})
                            ready = "✅ BUY " if kij.get("ready", True) else "⏳ WAIT"
                            klbl  = kij.get("label", "—")
                            p(f"  {i:<4} {s['ticker']:<8} {s['price']:>7.2f}  {s['shares']:>8,}"
                              f"  {s['cost_vnd']/1e6:>8.2f}M  {s['mom_20d']*100:>+7.1f}%"
                              f"  {s['tp_price']:>6.2f}  {ready} {klbl}")
                        total = sum(s["cost_vnd"] for s in stocks)
                        p(f"  {'':4} {'TOTAL':<8} {'':>7}  {sum(s['shares'] for s in stocks):>8,}"
                          f"  {total/1e6:>8.2f}M")
                    else:
                        p(f"  ⚠️  No eligible stocks found")

    # ─── FINAL DECISION (single action summary) ──────────────────
    p()
    p("=" * 70)
    p("  [6] FINAL DECISION — WHAT TO DO TODAY")
    p("=" * 70)
    # Determine which signal (if any) is the actual action
    # Priority: BASELINE > EARLY_V3 > DEMAND_EARLY (same as backtest)
    _action_sec   = None
    _action_type  = None
    _action_stocks = []

    if HELD_SECTOR:
        _action_type = "HOLD"
        p(f"  HOLD {HELD_SECTOR} — you are already in a position.")
        p(f"  Check [2] EXIT CHECK and [4] TAKE-PROFIT above for any action needed.")
    else:
        # Check what fired (re-use variables from above sections)
        if 'best_sec' in dir() and candidates:
            _action_sec  = best_sec
            _action_type = "BASELINE"
        elif 'best_early' in dir() and early_candidates:
            _action_sec  = best_early
            _action_type = "EARLY_V3"
        elif 'best_demand' in dir() and demand_candidates:
            _action_sec  = best_demand
            _action_type = "DEMAND_EARLY"

        if _action_type is None or _action_sec is None:
            p("  ACTION : STAY IN CASH")
            p("  REASON : No sector signal has fired (baseline, early, or demand).")
            p("  WATCH  : Check again tomorrow.")
        else:
            icons = {"BASELINE": "🟢", "EARLY_V3": "⚡", "DEMAND_EARLY": "🔥"}
            risks = {
                "BASELINE":     "Confirmed recovery — standard risk.",
                "EARLY_V3":     "Pre-signal — sector still DROWNING. Lower win rate (~42%).",
                "DEMAND_EARLY": "Earliest entry — sector DROWNING but heat is high. Highest risk, best price.",
            }
            t1c = TOTAL_CAPITAL_VND * TRANCHE1_FRAC
            t2c = TOTAL_CAPITAL_VND * TRANCHE2_FRAC
            p(f"  ACTION : {icons[_action_type]} BUY {_action_sec}  [{_action_type}]")
            p(f"  RISK   : {risks[_action_type]}")
            p(f"  CAPITAL: T1 = {t1c/1e6:.1f}M VND tomorrow open")
            p(f"           T2 = {t2c/1e6:.1f}M VND in 3 trading days")
            p(f"           (skip T2 if sector spread < 0 on T2 day)")
            p()
            p(f"  WHY NOT the others?")
            if _action_type == "BASELINE":
                p(f"    EARLY_V3 / DEMAND_EARLY: baseline already confirmed — no need to go earlier.")
            elif _action_type == "EARLY_V3":
                p(f"    BASELINE: not confirmed yet (spread below threshold).")
                if 'best_demand' in dir() and demand_candidates:
                    p(f"    DEMAND_EARLY ({best_demand}): EARLY_V3 fired first — backtest priority goes to EARLY_V3.")
                    p(f"    Different sector shown in [5C] — ignore it, only one sector at a time.")
            elif _action_type == "DEMAND_EARLY":
                p(f"    BASELINE / EARLY_V3: neither has fired yet.")
            p()
            p(f"  SELL when (check daily):")
            p(f"    • Any stock hits +25% gain  → sell that stock next open")
            p(f"    • Sector spread drops < {SPREAD_EXIT}  → exit ALL stocks next open")
            p(f"    • Sector state = PEAKING    → exit ALL stocks next open")
            p(f"    • 20d momentum < -5%        → exit ALL stocks next open")
            p(f"    DEMAND_EARLY uses identical exit rules as baseline — no difference.")
    p("=" * 70)

    # ─── FOOTER ──────────────────────────────────────────────────
    p()
    p("━" * 70)
    p("  QUICK REFERENCE")
    p("━" * 70)
    p(f"  EXIT ALL stocks  → when sector spread < {SPREAD_EXIT} | PEAKING | mom < {MOMENTUM_FLOOR*100:.0f}%")
    p(f"  SELL one stock   → when gain ≥ +25% (check [4] above daily)")
    p(f"  T2 RULE          → {"DIP: next red day (≥"+str(abs(DIP_THRESHOLD)*100)+"%) or day "+str(DIP_MAX_WAIT)+" at market" if T2_MODE=="DIP" else "TRANCHE2: fixed "+str(T2_FIXED_DAYS)+" trading days after T1"}; skip if spread < 0")
    p(f"  TP target        → entry_price × 1.25")
    p()
    p("═" * 70)
    p(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p("═" * 70)

    fname = os.path.join(REPORTS_DIR, f"signals_{today_str}.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  → Saved to {fname}")


def _parse_capital(arg):
    """
    Parse a capital string into an integer VND amount.
    Accepts:  10m  100m  1b  1.5b  5000000  50_000_000
    Examples:
      10m  -> 10_000_000
      100m -> 100_000_000
      1b   -> 1_000_000_000
      1.5b -> 1_500_000_000
    """
    s = arg.strip().lower().replace(",", "").replace("_", "")
    if s.endswith("b"):
        return int(float(s[:-1]) * 1_000_000_000)
    elif s.endswith("m"):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith("k"):
        return int(float(s[:-1]) * 1_000)
    else:
        return int(float(s))


if __name__ == "__main__":
    # Optional CLI args:
    #   python .4Sectorlivesignals.py [capital]
    #   python .4Sectorlivesignals.py 10m
    #   python .4Sectorlivesignals.py 500m
    #   python .4Sectorlivesignals.py 1b
    import sys as _sys
    if len(_sys.argv) >= 2:
        try:
            TOTAL_CAPITAL_VND = _parse_capital(_sys.argv[1])
            print(f"  Capital set from CLI: {TOTAL_CAPITAL_VND:,.0f} VND "
                  f"({_sys.argv[1].upper()})")
        except ValueError:
            print(f"  WARNING: could not parse '{_sys.argv[1]}' as capital — "
                  f"using default {TOTAL_CAPITAL_VND:,.0f} VND")
    main()
    input("\nPress Enter to exit...")
