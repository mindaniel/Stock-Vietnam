"""
=============================================================================
REALISTIC 3-SECTOR BACKTEST  —  Banks + Food & Beverage + Oil & Gas
=============================================================================

KEY REALISM RULES:
  1. Liquidity pre-filter — signal universe = execution universe (no mismatch)
  2. T+1 open execution — orders fill at NEXT day's open price
  3. T+3 cash settlement — sale proceeds only available 3 TRADING DAYS
     after the sell date. You cannot buy the next sector immediately after
     selling; you must wait for cash to settle.
  4. HOSE / HNX only — no UPCOM
  5. Whole shares only — no fractional
  6. 0.25% friction per leg (broker 0.15% + slippage 0.10%)

T+3 SETTLEMENT MECHANICS:
  - Day 0: sell signal fires
  - Day 1: sell executes at open (T+1)
  - Day 1+3 trading days: cash arrives in account
  - Only then can a new buy be executed

  While waiting for settlement, a new buy signal CAN fire — it goes into a
  pending queue and executes as soon as settled cash is available.

3-SECTOR ROTATION LOGIC (same as 2-sector but with 3 candidates):
  - Each day evaluate all 3 sector signals independently
  - If currently held sector wants to exit → queue sell
  - If not invested (and cash settled) → buy best scoring sector
  - Sectors ranked by score; challenger must score MIN_SCORE_GAP higher
    to displace current holding

=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats
import os, glob, warnings
from collections import deque
from pathlib import Path
try:
    from factor_stock_ranker import build_factor_features, rank_by_factor
    _FACTOR_RANKER_AVAILABLE = True
except ImportError:
    _FACTOR_RANKER_AVAILABLE = False
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
STOCK_DATA_PATH     = "all_stocks_with_industries.parquet"
VNINDEX_PATH        = "VNINDEX.csv"
INDIVIDUAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "price")

# ═══════════════════════════════════════════════════════════════════
# BEST-OF-4 — single pool, original 2-sector logic, 4 candidates
# ═══════════════════════════════════════════════════════════════════
#
# Identical engine to the original 2-sector backtest, extended to
# pick the best qualifying sector from 4 candidates instead of 2.
#
# Entry: RECOVERY  OR  LEADING (score ≥ MIN_ENTRY_SCORE)
# Capital: 100M deployed 100% into one sector at a time
# When multiple sectors qualify: pick highest score
# Cash when nothing qualifies: protection, not waste
#
# Sectors (each unique SECTOR_GROUPS value = one candidate):
#   Banks (20t)             — credit cycle, early mover
#   Financial Services (22t)— market sentiment, complements Banks
#   Food & Beverage (17t)   — defensive consumer, different timing
#   Real Estate (51t)       — late cycle, rate-sensitive

# Sector grouping: maps Level-2 industry → slot name
# Each unique slot name = one independent capital slot
SECTOR_GROUPS = {
    "Banks":                "Banks",
    "Food & Beverage":      "Food & Beverage",
    "Basic Resources":      "Basic Resources",
    "Real Estate":          "Real Estate",
    # 4-sector pool: each fills different cycle phase
    # Banks (66d):          credit cycle anchor
    # Food & Bev (46d):     fills 2009-2013 gaps, defensive
    # Basic Resources(145d): commodity cycle, fires 2021-2023
    # Real Estate (83d):    rate-sensitive, late cycle
}

SECTORS_OVERRIDE    = None

CAPITAL_VND         = 100_000_000
SETTLEMENT_DAYS     = 3
BACKTEST_START      = "2005-01-01"

# ── Regime engine (adaptive position sizing + sector filter) ──────
# Set True to enable.  Requires regime_engine.py in the same directory.
# Regime is recomputed monthly during the backtest; each month's regime:
#   - gates which sectors are eligible for new entries
#   - scales the capital deployed per trade by regime_multiplier
#     BULL=1.0  NEUTRAL=0.70  ROTATE=0.40  DEFENSIVE=0.20  PANIC=0.05
REGIME_ENGINE_ENABLED = False
COMPARE_REGIME        = False  # True -> run twice, print filter-on vs filter-off table

# ── Foreign-flow confirmation overlay ────────────────────────────
# When enabled, a sector entry is only allowed when the sector's
# constituent stocks have been net-foreign-bought for >= FG_CONFIRM_STREAK
# consecutive days.  Coverage starts 2018 (earlier dates pass through).
#
# Proxy test results (sector momentum < -3%, fwd 10d):
#   Banks:       +1.53% lift   (confirmed 16% of entries)
#   Real Estate: +2.89% lift   (confirmed 20% of entries)
#   Food & Bev:  -1.07% drag   (contrarian — excluded)
#   Basic Res:   -0.27% drag   (minor — excluded)
FG_CONFIRM_ENABLED = False  # TESTED: -21,927% total drag — do not enable
FG_CONFIRM_SECTORS = {"Banks", "Real Estate"}
FG_CONFIRM_STREAK  = 3     # min consecutive days of sector-avg net fg buying
COMPARE_FG         = False  # True -> run twice, print with/without fg table

# ── Investor-flow signal overlay (flow_signals.py) ───────────────────────────
# Uses data from data/investor_flow/*.parquet  (coverage: Sep 2024 onwards).
# Tickers without flow data fall back to normal selection — never penalised.
#
# FLOW_SIGNAL_ENABLED: master toggle.  Set True to activate.
#   When on, adds a FLOW_RANK pass AFTER all other stock selection methods:
#   within the candidate list, re-rank by smart_score (institutional flow balance)
#   and keep the top FLOW_RANK_TOP_N.  Tickers with no data score 0.0 and sit
#   last (they are NOT excluded from the final pick — if the FLOW_RANK results
#   in fewer than 3 stocks, the full unranked list is used as a fallback).
#
# FLOW_SECTORS: only apply the flow ranking for these sectors (set to None for all)
#
# FLOW_DIST_EXIT: also use distribution_alert as an additional exit trigger.
#   Fires when institutions are actively handing off positions to retail.
#   Conservative recommendation: leave False until live-tested.
#
# FLOW_RANK_WINDOW: lookback days for the smart_score normalisation.
#   20d = default, captures medium-term flow trend.
#   10d = more sensitive, noisier.
#
# Note on FG_CONFIRM vs FLOW_SIGNAL:
#   FG_CONFIRM used raw CSV foreign data (1/10th of real flows, single exchange).
#   FLOW_SIGNAL uses the full NDT classification API — all investor types,
#   all exchanges.  Direction of the signal is also different:
#   FLOW_SIGNAL is NOT "foreigners buying" (which hurt): it uses DOMESTIC
#   institutional flow as the primary signal (to_chuc_trongnuoc = 0.50 weight).
FLOW_SIGNAL_ENABLED  = False          # set True to activate
FLOW_SECTORS         = None           # None = all sectors; or {"Banks","Real Estate"}
FLOW_RANK_TOP_N      = 10             # keep top N by smart_score; None = no cap
FLOW_RANK_WINDOW     = 20             # lookback window for smart_score (trading days)
FLOW_DIST_EXIT       = False          # simple distribution-alert exit (legacy)
FLOW_PEAK_EXIT       = False          # composite peak-exit: distribution + heavy_sell + flip_down

MIN_LIQUIDITY_VND   = 1_000_000_000

Z_WINDOW            = 252
SMOOTH_WINDOW       = 20
CRASH_THRESHOLD     = 35.0
TROUGH_WINDOW       = 252
VELOCITY_WINDOW     = 20
RECOVERY_WINDOW     = 40

MIN_ENTRY_SCORE     = 3.0
MOMENTUM_LOOKBACK   = 20
MOMENTUM_FLOOR      = -0.05
SPREAD_EXIT         = -2.0
SPREAD_PEAK         =  5.0
SPREAD_CRASH        = -15.0
MIN_HOLD_DAYS       = 15    # match original 2-sector
MIN_SCORE_GAP       = 0.25  # match original 2-sector
SPREAD_HIGH_EXIT    = 50.0

# ── Post-peak cooldown ────────────────────────────────────────────
# After any PEAKING exit, block RECOVERY entries for this many days.
# LEADING entries are still allowed (spread already confirmed positive).
#
# Why RECOVERY only, not LEADING:
#   RECOVERY = spread just crossed zero, unconfirmed — risky right after a peak
#   LEADING  = spread already sustainably positive — genuine sector rotation
#
# From trade log analysis:
#   Nov 2020 (6d gap, LEADING):  +40.9% — must NOT block this
#   Sep 2023 (7d gap, RECOVERY): -19.5% — must block this
#   Any value 8-30d achieves both. Using 20d as a round number.
#
# Theory: after a sector peaks, the broad market cycle is extended.
# Rushing into a new RECOVERY signal within days is chasing a potential
# market top. Wait for the dust to settle.
COOLDOWN_AFTER_PEAK = 20    # days to block RECOVERY entries after any peaking exit

# ── Stock-level trailing stop ────────────────────────────────────
# Each individual stock is sold when its price falls more than
# STOCK_TRAILING_STOP_PCT below its *peak price since entry*.
# Peak price is updated daily — the stop only ever moves up, never down.
#
# This is a PARTIAL exit — only triggered stocks are sold, the rest stay.
# Proceeds settle T+3 normally.
#
# Advantage over a fixed TP: lets winners run as far as momentum carries
# them, then captures most of the gain on the way down.
#
# HOSE daily limit = 7%, HNX = 10%.  Because moves are capped daily,
# a 10% trailing stop on HOSE gives very predictable behaviour:
#   Worst case: stock peaks → limit-down next day → exits ~7-10% below peak.
#   In practice the stop triggers at exactly STOCK_TRAILING_STOP_PCT below peak.
#
# In real trading this must be monitored and set manually each day —
# Vietnamese brokers have no native trailing-stop order type.
#
# WHIPSAW PROTECTION — TRAIL_ACTIVATE_PCT:
#   The trailing stop only arms once the stock has gained at least this much
#   from entry. Before that threshold, the stop is dormant (only SL applies).
#   This prevents small early corrections from shaking you out of a position
#   before momentum has had a chance to develop.
#
#   Example: activate=0.15, trail=0.12
#     Stock must first reach +15% gain, THEN if it pulls back 12% from its
#     peak it exits. Without this, a stock that goes +5% then -12% would
#     trigger the trailing stop at a loss.
#
#   Set TRAIL_ACTIVATE_PCT to 0.0 to arm immediately from entry (no guard).
#
# Suggested values:
#   STOCK_TRAILING_STOP_PCT: 0.10 (tight) / 0.12 (balanced) / 0.15 (loose)
#   TRAIL_ACTIVATE_PCT:      0.10–0.20 (arm only after stock has proven itself)
#
# Set STOCK_TRAILING_STOP_PCT to None to disable entirely.
STOCK_TRAILING_STOP_PCT = None   # None to disable — TP=25% outperforms trailing stop
TRAIL_ACTIVATE_PCT      = 0.15   # trailing stop arms only after +15% gain from entry (if enabled)

# ── Stock-level fixed take-profit (optional override) ────────────
# If set, sells immediately when gain from entry reaches this level,
# regardless of trailing stop. Use this if you want a hard ceiling.
# Set to None to rely solely on the trailing stop (recommended).
STOCK_TP_PCT        = 0.25   # 25% fixed TP — outperforms trailing stop in backtest

# ── Stock-level stop-loss ─────────────────────────────────────────
# Sells if price falls more than STOCK_SL_PCT below *entry price*.
# This is the falling-knife guard for momentum picks that immediately
# reverse — the trailing stop alone won't protect you if peak = entry.
#
# Set to None to disable.
# Tested values: 0.10 (-10%), 0.12 (-12%), 0.15 (-15%)
STOCK_SL_PCT        = None   # disabled — SL fights against recovery in volatile markets

# ── Laggard exit (fund-style rule) ───────────────────────────────
# Inspired by fund practice: "if a stock isn't running after 2 weeks, sell it"
#
# Two independent checks — either can fire:
#
# 1. FLAT laggard:  after LAGGARD_FLAT_DAYS, sell if return < LAGGARD_FLAT_THRESH
#    Idea: a stock that's been held N days and hasn't moved is dead weight — free
#    the capital.  Recovery stocks should show early momentum within ~3 weeks.
#    Default: after 20 trading days, exit if still < +3% (not running at all).
#
# 2. LOSS laggard:  after LAGGARD_LOSS_DAYS, sell if return < -LAGGARD_LOSS_THRESH
#    Idea: a stock that's down more than X% after N days is going the wrong way.
#    Separate from STOCK_SL_PCT (which fires immediately) — this gives a grace
#    period for volatile low-spread stocks to settle before cutting.
#    Default: after 10 trading days, exit if down more than -10%.
#
# Set either threshold to None to disable that check individually.
# Both checks respect the T+3 rule (won't fire before entry_date + 3 days).
#
LAGGARD_FLAT_DAYS   = 20     # days held before flat-laggard check kicks in
LAGGARD_FLAT_THRESH = None   # OFF — flat laggard fights mean-reversion (recoveries take time)
LAGGARD_LOSS_DAYS   = 10     # days held before loss-laggard check kicks in
LAGGARD_LOSS_THRESH = None   # OFF — loss laggard cuts recovery stocks before bounce

# ── Ichimoku Kijun-sen entry timing ──────────────────────────────
# After a sector signal fires, each individual stock is evaluated against
# its Kijun-sen (26-period midline = (period_high + period_low) / 2).
# Stocks that are too far above the Kijun are deferred — we wait for
# a small pullback before buying.  This typically saves 2-5% on entry.
#
# A stock is "ready to buy" when:
#   (open_price / kijun_sen) - 1  <=  KIJUN_BUY_THRESHOLD
#
# Deferred stocks are re-checked daily.  After ENTRY_MAX_WAIT_DAYS trading
# days they are force-bought at market — we cannot miss the sector cycle.
#
# Cash for deferred stocks is reserved from the T1 tranche allocation so
# equal weighting is preserved across all stocks.  T2 (DCA) always buys
# at market regardless, topping up all positions.
#
# In real trading: check each stock against its Kijun on your broker chart
# the morning of the signal.  Set limit orders near the Kijun for extended
# stocks; cancel and buy market after ENTRY_MAX_WAIT_DAYS if not filled.
#
# Set ENTRY_TIMING_KIJUN = False to disable (all stocks bought immediately).
ENTRY_TIMING_KIJUN   = False  # WFO: 0/5 OOS windows beat baseline — do not use
KIJUN_PERIOD         = 26    # Kijun-sen lookback (26 = Ichimoku standard)
KIJUN_BUY_THRESHOLD  = 0.03  # buy if price is within 3% above Kijun
ENTRY_MAX_WAIT_DAYS  = 3     # force-buy after this many trading days

# ── DCA Entry (dollar-cost averaging into positions) ──────────────
# Instead of deploying 100% on entry day, split into tranches.
#
# DCA_MODE options:
#   "NONE"        — deploy 100% on day 1 (current behaviour)
#   "TRANCHE2"    — 50% day 1, 50% day 4 (simple 2-split)
#   "CONDITIONAL" — 50% day 1, 50% day 4 only if spread still positive
#                   If signal fades in 3-day window → skip tranche 2
#                   This is the "confirmation window" approach
#   "SCORE"       — size by signal score:
#                   score ≥ DCA_SCORE_HIGH → 100% day 1 (high conviction)
#                   score ≥ DCA_SCORE_MID  → 67% day 1, 33% day 4
#                   score <  DCA_SCORE_MID → 33% day 1, 33% day 4, 33% day 7
#
DCA_MODE            = "TRANCHE2"      # "NONE" / "TRANCHE2" / "CONDITIONAL" / "SCORE" / "DIP"
DCA_SCORE_HIGH      = 8.0         # score threshold for full deployment
DCA_SCORE_MID       = 4.0         # score threshold for 2-tranche

# ── DIP mode parameters ───────────────────────────────────────────
# DCA_MODE = "DIP": deploy T1 immediately, wait for the next pullback
# before deploying T2. Avoids buying T2 into a short-term rally that
# immediately reverses — instead waits for the next red day to add.
#
# DIP_THRESHOLD : sector must fall at least this much on the day
#                 -0.005 = any day the sector is down >= 0.5%
#                 -0.010 = only deploy on days down >= 1%
# DIP_MAX_WAIT  : after this many trading days without a dip, deploy
#                 T2 at market anyway (don't miss the whole move)
# Guard: if sector spread goes negative before T2 fires → skip T2
#        (sector reversed — original signal was wrong)
DIP_THRESHOLD       = -0.003  # deploy T2 when sector down >= 0.3% on that day
DIP_MAX_WAIT        = 5       # max trading days to wait; then deploy at market
# When a sector signal fires, which stocks within the sector to buy.
# All methods apply AFTER liquidity/volume filters.
#
# "ALL"       — buy all liquid stocks equally (original behaviour)
# "MOM_BOT50" — buy bottom 50% by 20-day momentum (most oversold)
#               Stocks that fell most recently → strongest mean reversion
# "MA_BOT50"  — buy bottom 50% by distance from 200-day MA
#               Stocks furthest below long-term trend → deepest value
# "MOM_BOT30" — bottom 30% by momentum (more concentrated)
# "MA_BOT30"  — bottom 30% by MA distance (more concentrated)
#
# Theory: within a sector recovery signal, the most recently beaten-down
# stocks have the highest mean-reversion potential. The sector signal
# provides timing; the stock filter concentrates in the best opportunities.
#
# Tested in stock_selection_tester.py:
#   MOM_BOT50 → Sharpe +0.118 vs baseline (strongest signal)
#   MA_BOT50  → Sharpe +0.107 vs baseline
#   Z_BOT50   → Sharpe -0.036 vs baseline (Z redundant with sector signal)
STOCK_SELECTION     = "MOM_BOT50"  # "ALL" to disable / revert to original

# ── Per-sector stock selection (from volume experiment results) ───
# Banks:           MOM_BOT50  — vol metrics HURT (macro-driven, high intra-sector
#                               correlation; MOM_BOT50 win rate 49.6% vs vol 45-46%)
# Basic Resources: VOL_LEADERS — strongest accumulation signal; win rate +11pp
#                               (39.8% MOM → 50.8% VOL_LEADERS)
# Food & Beverage: VOL_RANK   — top 50% by vol_score; win rate +8.7pp (41.3% → 50.0%)
# Real Estate:     VOL_LEADERS N=12 — top 12 by vol_score; win rate +6pp vs VOL_RANK,
#                                      +25% total return vs VOL_RANK (~25 stocks)
SECTOR_STOCK_SELECTION = {
    "Banks":           "MOM_BOT50",
    "Basic Resources": "VOL_LEADERS",
    "Food & Beverage": "VOL_RANK",
    "Real Estate":     "VOL_LEADERS",
}

# ── Per-sector breadth universe cap ──────────────────────────────────────────
# Limits which stocks are used to COMPUTE the Z-score breadth signal.
# Real Estate: 51 stocks in the sector but strategy only buys top 12.
# Zombie/distressed small RE companies dilute the signal — VIC/VHM can be
# in full recovery while the breadth reads DROWNING because 30 small caps
# are still underwater. Cap to top 20 by rolling liquidity so the signal
# reflects the stocks you'd actually trade.
#
# Note: tcks[sec] (the buy universe) is still ALL liquid stocks — the cap
# only affects the breadth SIGNAL, not which stocks are eligible to buy.
SECTOR_BREADTH_CAP = {
    # "Real Estate": 20,  # tested — HURTS (-23B PnL). Original 48-stock breadth is better calibrated.
    # The DEMAND_EARLY signal already handles the "large caps leading, rest lagging" case.
}

# ── VN-Index market breadth gate ─────────────────────────────────────────────
# ~80% of VN stock returns are correlated with VN-Index (market beta).
# When VN-Index has been falling hard and SUSTAINING (not just a short dip),
# even valid sector RECOVERY signals face strong headwinds — most stocks
# will still drift down with the market.
#
# Strategy: buy beaten-down stocks ready to bounce BACK.
# Gate role: avoid entries during SUSTAINED bear markets (e.g. 2022),
#            but NOT block entries during sharp crashes that recover fast (e.g. 2020).
# Signal: 63-day (3-month) VN-Index compounded return — captures sustained moves.
#
# Modes:
#   "OFF"  — no gate, all entries at full size (baseline)
#   "SOFT" — half size if VN 63d < SOFT threshold; skip if < HARD threshold
#   "HARD" — skip entry entirely if VN 63d < HARD threshold
#
VNINDEX_GATE             = "OFF"    # "OFF", "SOFT", "HARD"  ← override via --vn-gate
VNINDEX_GATE_SOFT_THRESH = -0.05   # VN-Index down >5%  in 3M → half size
VNINDEX_GATE_HARD_THRESH = -0.10   # VN-Index down >10% in 3M → skip entry
VNINDEX_GATE_WINDOW      =  63     # trading days (~3 months)

# ── VN-Index exit accelerator ────────────────────────────────────────────────
# When HOLDING a position, accelerate exit if VN-Index drops sharply.
# Two triggers (both independent — either can fire):
#   FLASH     (5-day):  VN drops > X% in a week    → crash unfolding, exit now
#   SUSTAINED (20-day): VN drops > X% in a month   → prolonged bear, cut losses
#
# Backtest findings (compare_vn_exit.py):
#   Thresholds of 5d/-7% and 20d/-12% HURT overall (CAGR -1.77pp):
#   - Flash (-7% / 5d) fires during bull corrections (Feb 2021 → -68.9% that year)
#   - Sustained helped in 2015 Aug (Chinese crash) but hurt 2011/2012
#   - Neither threshold caught the 2022 problem (strategy was in cash or bear-bounce)
#
# Kept here for experimentation — default OFF until better thresholds are found.
# To re-test: raise flash to -12% or add sector-momentum confirmation.
#
VNINDEX_EXIT_ENABLED          = False   # OFF by default — current thresholds hurt more than help
VNINDEX_EXIT_FLASH_DAYS       =  5      # short window for flash crash
VNINDEX_EXIT_FLASH_THRESH     = -0.07   # -7% in 5 days  → flash exit  (too sensitive)
VNINDEX_EXIT_SUSTAINED_DAYS   = 20      # medium window for sustained bear
VNINDEX_EXIT_SUSTAINED_THRESH = -0.12   # -12% in 20 days → sustained exit

# ── Factor-enhanced stock selection ──────────────────────────────────────────
# When FACTOR_SELECTION_ENABLED = True, after the sector method produces its
# candidate list, a second pass ranks and filters by quarterly fundamental
# quality (np_yoy, accel_score, ROE) — keeping only the top FACTOR_TOP_PCT.
#
# This catches the "falling knife" problem: within a sector recovery signal,
# some stocks are falling because their earnings are genuinely deteriorating
# (not just market overreaction). The factor filter removes those.
#
# FACTOR_TOP_PCT = 0.5 → keep top 50% by fundamental score (recommended)
# FACTOR_TOP_PCT = 1.0 → disable filtering, use factor ranking only for ordering
#
# MIN_NP_YOY: hard reject if YoY profit growth < this value
#   -0.30 = allow up to 30% YoY decline (lenient, good for early recovery)
#   -0.10 = allow up to 10% decline (stricter)
#    0.00 = require profit growth (strictest — may cut too many in sector troughs)
FACTOR_SELECTION_ENABLED = True
FACTOR_TOP_PCT           = 0.50
FACTOR_MIN_NP_YOY        = -0.30   # lenient: sector is already bottoming

# Per-sector overrides — set to False to disable factor ranking for that sector.
# Basic Resources: factor helps most (volatile earnings, quality is discriminating)
# Banks:           factor helps (ROE filter removes weak banks)
# Real Estate:     factor helps (removes structural zombies)
# Food & Beverage: factor hurts (stable sector, YoY filter too strict on recovery)
FACTOR_SECTORS = {
    "Basic Resources": True,
    "Banks":           True,
    "Real Estate":     True,
    "Food & Beverage": False,   # disable — F&B recovery doesn't need quality filter
}

_QFEAT = None        # populated at startup by main()
_FLOW_ENGINE = None  # FlowSignalEngine, populated at startup when FLOW_SIGNAL_ENABLED
VOL_LEADERS_N = 12  # number of stocks kept by VOL_LEADERS method

# ── Order-flow ranking (OB_RANK) ──────────────────────────────────
# Rank stocks by rolling ob_ratio = buy_volume/sell_volume from
# data/order_history/.  ob_ratio > 1 = buyers more aggressive than sellers.
# Falls back to full-candle score when order history unavailable (pre-2018).
OB_ORDER_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data", "order_history")
OB_RANK_WINDOW = 5    # trading days to average ob_ratio for stock ranking
OB_RANK_TOP_N  = 12   # keep top N stocks by ob_ratio (same as VOL_LEADERS_N)
# ── Full-candle ranking for VOL_LEADERS ───────────────────────────
# Instead of ranking by accumulated vol_score, rank by a recent
# "big full-body candle" signal: body_pct × body_ratio, recency-weighted.
# Captures stocks with a strong conviction buying session (marubozu-style)
# rather than stocks that accumulated weeks ago.
FULL_CANDLE_WINDOW   = 10    # look-back days for best full candle
VOL_LEADERS_USE_FULL_CANDLE = True

# ── Demand-early entry (stock-flow based pre-signal) ─────────────
# Enter a DROWNING sector when enough liquid stocks show conviction
# buying (high full-candle scores), before breadth Z-score confirms.
# Fires earlier than EARLY_V3 (which requires 3d positive velocity).
DEMAND_EARLY_ENABLED     = True
DEMAND_HEAT_WINDOW       = 20    # days to compute sector heat
DEMAND_HEAT_THRESHOLD    = 0.012 # fallback fixed threshold (used when history too short)
DEMAND_HEAT_SPREAD_FLOOR = -15.0 # don't enter if spread below this (crash)
# Adaptive threshold: instead of fixed 0.012, fire when heat is in the
# top DEMAND_HEAT_PERCENTILE % of the sector's own rolling 252-day history.
# This adapts to market regimes — calm periods lower the bar, volatile
# periods raise it.  Falls back to DEMAND_HEAT_THRESHOLD when < MIN_HISTORY days.
DEMAND_HEAT_USE_ADAPTIVE = False  # validated: fixed 0.012 beats adaptive in WFO
DEMAND_HEAT_PERCENTILE   = 75    # percentile used when adaptive is enabled
DEMAND_HEAT_MIN_HISTORY  = 60    # min history points before adaptive kicks in
# Late-entry momentum filter for DEMAND_EARLY stock selection:
# When the sector heat fires, some stocks have already moved a lot
# (they were the "leaders" that generated the heat).  Buying them
# gives worse risk/reward — limited upside to TP, they already ran.
# This cap filters out stocks whose 20d price momentum exceeds the
# threshold, leaving only stocks that haven't moved yet (laggards).
# Set to None to disable (buy all stocks regardless of how far they moved).
DEMAND_EARLY_MAX_MOM_PCT  = 0.15   # skip stocks with >15% 20d momentum at entry

# When no sector signal fires and capital is idle, deploy a fraction
# into a broad market ETF (FUEVN100) to capture trending bull markets.
# A cash reserve is kept at all times so sector signals can fire
# immediately without waiting for ETF settlement (T+3).
#
# ETF_TICKER:       which ETF to buy (must have a CSV in INDIVIDUAL_DIR)
# ETF_ALLOC:        fraction of idle settled_cash to deploy into ETF
# ETF_CASH_RESERVE: fraction always kept as cash (for instant sector entry)
#
# Capital split when idle:
#   settled_cash × ETF_ALLOC        → ETF position
#   settled_cash × ETF_CASH_RESERVE → stays cash
#
# When sector signal fires:
#   Buy sector stocks from cash reserve immediately (T+1 exec)
#   Sell ETF simultaneously (proceeds arrive T+3 to replenish reserve)
#
# Set ETF_TICKER = None to disable the overlay entirely.
ETF_TICKER       = None      # disabled — idle cash IS the strategy's protection
ETF_ALLOC        = 0.65
ETF_CASH_RESERVE = 0.35

# ── Vol-adjusted entry threshold ─────────────────────────────────
# threshold(t) = K × rolling_std(spread, VOL_WINDOW)
#
# K is derived PURELY from each sector's average recovery speed
# (days from trough to zero crossing), measured from the spread series.
# No backtest feedback — K is set before we see any returns.
#
# Theory: faster recovery → higher K (spread recovers quickly so we
# can afford to wait for confirmation without missing the move).
# Slower recovery → lower K (must enter earlier to catch the full run).
#
# Monotonic formula:
#   K = clip(0.75 - (recovery_days - 40) / 200, 0.25, 0.75)
#   recovery_days=40  → K=0.75  (fastest possible — wait for confirmation)
#   recovery_days=90  → K=0.50  (medium)
#   recovery_days=140 → K=0.25  (slow — enter early)
#
# Special case: Banks (speed_asym=2.11) recovers TWICE as fast as it
# drops — unique among all sectors. Even though recovery_days=66 would
# give K≈0.64, we use K=0.25 because the signal fires BEFORE the trough
# (market prices bank recovery early). This is a theory-driven override,
# not a return-driven one.
#
# Measured recovery_days (from volatility analysis):
#   Food & Beverage:  46d → K = clip(0.75-(46-40)/200, 0.25, 0.75) = 0.72 → rounds to 0.75
#   Banks:            66d → K = 0.25 (theory override: early-entry sector)
#   Real Estate:      83d → K = clip(0.75-(83-40)/200, 0.25, 0.75) = 0.54 → 0.50
#   Construction:     84d → K = 0.53 → 0.50
#   Retail:          124d → K = 0.33 → 0.25–0.50
#   Financial Serv:  142d → K = 0.25 (floor)
#   Basic Resources: 145d → K = 0.25 (floor)

VOL_WINDOW          = 60    # rolling window for spread std (days)
RECOV_WINDOW_LONG   = 504   # ~2 years of history to compute rolling recovery speed
                             # long enough to cover 2-3 full cycles but not the whole period

# Banks special case: speed_asym=2.11 means market prices recovery BEFORE the
# spread crosses zero. So K must be low (0.25) regardless of computed recovery_days.
# This is a structural property of the banking credit cycle, not a backtest observation.
EARLY_ENTRY_SECTORS = {"Banks"}
EARLY_ENTRY_K       = 0.25


BROKER_FEE          = 0.0015
SLIPPAGE            = 0.0010
FRICTION            = BROKER_FEE + SLIPPAGE

# ─────────────────────────────────────────────────────────────────
# FUNDAMENTAL QUALITY FILTER  (uses data/financials_fa/ parquets)
# ─────────────────────────────────────────────────────────────────
# Toggle: set FUNDAMENTAL_FILTER_ENABLED = True to activate.
#
# Criteria are sector-specific — each sector's fundamentals work
# differently (banks use ROE/profit growth; real estate uses asset
# or revenue growth; F&B uses OCF quality + ROE; basic resources
# uses revenue growth + cash generation).
#
# Point-in-time: report for year Y only available after April 30, Y+1.
# Stocks with no fundamental data available yet pass by default.
#
# All growth/ratio thresholds use the same units as the parquet:
#   - growth fields (revenue_growth_lfy, profit_growth_lfy, asset_growth_lfy):
#     decimal fraction  (0.10 = +10%,  -0.05 = -5%)
#   - roe:              percentage     (12.5 = 12.5%)
#   - ocf_to_netprofit: ratio          (0.5 = OCF covers 50% of net profit)
FUNDAMENTAL_FILTER_ENABLED = True   # flip to True to activate

SECTOR_FUND_CRITERIA = {
    # Banks — ROE only; profit growth excluded (provisioning cycles make it noisy).
    # PB/OCF not used: banks trade at various book multiples structurally.
    "Banks": {
        "roe_min": 10.0,
    },
    # Real Estate — OR logic: revenue growth OR asset growth > -10%.
    # Threshold -0.10 allows small dips without cutting pipeline builders.
    # OCF and PB excluded by design (construction phases are cash-negative).
    "Real Estate": {
        "revenue_or_asset_growth_min": -0.10,
    },
    # Food & Beverage — most "normal" sector; all three criteria apply.
    "Food & Beverage": {
        "revenue_growth_min": 0.0,   # market share / volume growing
        "ocf_min":            0.5,   # earnings backed by real cash
        "roe_min":            8.0,   # profitable on brand/distribution assets
    },
    # Basic Resources — commodity cycles dominate; just need business not shrinking
    # and generating positive cash (capital-intensive, so cash matters a lot).
    "Basic Resources": {
        "revenue_growth_min": 0.0,   # revenue not contracting
        "ocf_min":            0.0,   # positive OCF
    },
}

# Annual report publication lag: Vietnamese companies file by April 30 of Y+1.
_FUND_LAG_MONTH = 4
_FUND_LAG_DAY   = 30
_FUND_DATA      = {}   # populated at startup by load_fundamental_data()

# ─────────────────────────────────────────────────────────────────
# LOAD INDIVIDUAL STOCK CSVs
# ─────────────────────────────────────────────────────────────────

def load_individual_stocks(data_dir):
    print(f"Loading individual stock files...")
    files      = glob.glob(os.path.join(data_dir, "*.parquet"))
    stock_data = {}
    failed     = 0
    for fpath in files:
        ticker = os.path.splitext(os.path.basename(fpath))[0].upper()
        try:
            df = pd.read_parquet(fpath)
            df.columns = [c.strip().lower() for c in df.columns]
            date_col = "time" if "time" in df.columns else "date"
            df["date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
            for col in ["open", "close", "volume"]:
                if col not in df.columns:
                    raise ValueError(f"missing {col}")
            # Keep high/low for Ichimoku Kijun-sen if available
            cols = ["open", "close", "volume"]
            if "high" in df.columns and "low" in df.columns:
                cols = ["open", "high", "low", "close", "volume"]
            df = df[cols].astype(float)
            df = df[(df["close"] > 0) & (df["open"] > 0)]
            stock_data[ticker] = df
        except Exception:
            failed += 1
    print(f"  Loaded {len(stock_data)} stocks  ({failed} failed)")
    return stock_data


# ─────────────────────────────────────────────────────────────────
# FUNDAMENTAL DATA LOADER
# ─────────────────────────────────────────────────────────────────

def load_fundamental_data(fa_dir=None):
    """
    Load annual fundamental data for all tickers.
    Returns dict: {symbol -> pd.DataFrame sorted by avail_date}.
    avail_date = April 30, year+1  (point-in-time publication lag).
    """
    if fa_dir is None:
        fa_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "financials_fa")
    if not os.path.isdir(fa_dir):
        print(f"  [FUND] {fa_dir} not found — fundamental filter disabled")
        return {}

    import datetime as _dt
    result = {}
    loaded = 0
    for fpath in glob.glob(os.path.join(fa_dir, "*.parquet")):
        sym = os.path.basename(fpath).replace(".parquet", "")
        if sym == "indicators_snapshot":
            continue
        try:
            df = pd.read_parquet(fpath)
            # Annual rows only
            if "quarter" in df.columns:
                df = df[df["quarter"] == 0].copy()
            elif "type" in df.columns:
                df = df[df["type"] == "yearly"].copy()
            if df.empty:
                continue
            # Sort by year once for all pct_change computations below
            df = df.sort_values("year").reset_index(drop=True)
            # Normalise sector: 'industry' column is used by the old REST API
            # fetcher; 'sector' by the HTML scraper. Merge them.
            if "sector" not in df.columns or df["sector"].isna().all():
                if "industry" in df.columns:
                    df["sector"] = df["industry"]
            else:
                # Fill any remaining None/NaN cells from 'industry'
                if "industry" in df.columns:
                    df["sector"] = df["sector"].fillna(df["industry"])
            # Compute growth rates from raw values when pre-computed columns
            # are missing or all-NaN (happens when REST-API data overwrites
            # the HTML-scraper rows that had pre-computed growth fields).
            for raw_col, growth_col in [
                ("revenue",    "revenue_growth_lfy"),
                ("net_profit", "profit_growth_lfy"),
                ("total_assets", "asset_growth_lfy"),
            ]:
                if raw_col in df.columns:
                    if growth_col not in df.columns or df[growth_col].isna().all():
                        df[growth_col] = df[raw_col].pct_change()
            # Compute availability date
            df["avail_date"] = df["year"].apply(
                lambda y: _dt.date(int(y) + 1, _FUND_LAG_MONTH, _FUND_LAG_DAY)
            )
            df = df.sort_values("avail_date").reset_index(drop=True)
            result[sym] = df
            loaded += 1
        except Exception:
            pass
    print(f"  [FUND] Loaded fundamental data for {loaded} symbols")
    return result


def get_fund_row(symbol: str, as_of_date, fund_data: dict):
    """
    Return the most recent annual fundamental row available as_of_date
    (point-in-time: uses avail_date = April 30, year+1).
    Returns None if no data available yet.
    """
    df = fund_data.get(symbol)
    if df is None or df.empty:
        return None
    if hasattr(as_of_date, "date"):
        as_of = as_of_date.date()
    else:
        as_of = as_of_date
    available = df[df["avail_date"] <= as_of]
    if available.empty:
        return None
    return available.iloc[-1]


def passes_fundamental_filter(symbol: str, exec_date, fund_data: dict = None) -> bool:
    """
    True if the stock passes sector-specific fundamental criteria as of exec_date.
    Passes by default when no data is available (don't block pre-data history).
    """
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
        return True   # unknown sector → pass

    def _check(val, threshold):
        """Return True if val >= threshold, or if either is missing/invalid."""
        if threshold is None or val is None:
            return True
        try:
            v = float(val)
        except (TypeError, ValueError):
            return True
        if _math.isnan(v) or _math.isinf(v):
            return True
        return v >= threshold

    if sector == "Banks":
        if not _check(row.get("roe"),              criteria.get("roe_min")):
            return False
        if not _check(row.get("profit_growth_lfy"), criteria.get("profit_growth_min")):
            return False

    elif sector == "Real Estate":
        # OR logic: revenue growth OR asset growth must be positive.
        # Only block if we have data for both and both are negative.
        threshold = criteria.get("revenue_or_asset_growth_min", 0.0)
        rev = row.get("revenue_growth_lfy")
        ast = row.get("asset_growth_lfy")
        has_rev = rev is not None and not (isinstance(rev, float) and (_math.isnan(rev) or _math.isinf(rev)))
        has_ast = ast is not None and not (isinstance(ast, float) and (_math.isnan(ast) or _math.isinf(ast)))
        if has_rev or has_ast:
            rev_ok = _check(rev, threshold) if has_rev else False
            ast_ok = _check(ast, threshold) if has_ast else False
            if not rev_ok and not ast_ok:
                return False

    elif sector == "Food & Beverage":
        if not _check(row.get("revenue_growth_lfy"), criteria.get("revenue_growth_min")):
            return False
        if not _check(row.get("ocf_to_netprofit"),   criteria.get("ocf_min")):
            return False
        if not _check(row.get("roe"),                criteria.get("roe_min")):
            return False

    elif sector == "Basic Resources":
        if not _check(row.get("revenue_growth_lfy"), criteria.get("revenue_growth_min")):
            return False
        if not _check(row.get("ocf_to_netprofit"),   criteria.get("ocf_min")):
            return False

    return True


# ─────────────────────────────────────────────────────────────────
# LIQUIDITY FILTER
# ─────────────────────────────────────────────────────────────────

def get_liquid_tickers(stock_data):
    print(f"Computing liquidity for {len(stock_data)} tickers...")
    liquid = set()
    for ticker, sd in stock_data.items():
        if len(sd) < 20:
            continue
        daily_val   = sd["close"] * sd["volume"] * 1000
        rolling_avg = daily_val.rolling(20, min_periods=10).mean()
        if rolling_avg.median() >= MIN_LIQUIDITY_VND:
            liquid.add(ticker)
    print(f"  Liquid (≥{MIN_LIQUIDITY_VND/1e6:.0f}M VND median): {len(liquid)} tickers")
    return liquid


# ─────────────────────────────────────────────────────────────────
# LOAD SECTOR DATA  (liquidity-filtered BEFORE Z-score)
# ─────────────────────────────────────────────────────────────────

def load_sector_data(stock_path, vn_path, liquid_tickers):
    print("Loading sector data...")
    if stock_path.endswith(".parquet"):
        df = pd.read_parquet(stock_path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df = pd.read_csv(stock_path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["ticker","date"])
    df = df[df["close"] >= 5.0]

    if "exchange" in df.columns:
        df = df[df["exchange"].isin(["HOSE","HNX"])]

    hose_hnx = set(df["ticker"].unique())

    # Liquidity filter BEFORE Z-score — signal universe = execution universe
    df = df[df["ticker"].isin(liquid_tickers)]

    # ── Apply super-sector grouping ───────────────────────────────
    # Map Level-2 industry → super-sector name for breadth calculation.
    # Tickers not in the mapping are excluded.
    if SECTOR_GROUPS:
        df["super_sector"] = df["industry"].map(SECTOR_GROUPS)
        excluded = df["super_sector"].isna().sum()
        df = df[df["super_sector"].notna()].copy()
        print(f"  Super-sector grouping: {df['super_sector'].nunique()} groups "
              f"({excluded} rows excluded — unmapped sectors)")
        for grp in sorted(df["super_sector"].unique()):
            n   = df[df["super_sector"] == grp]["ticker"].nunique()
            raw = sorted(df[df["super_sector"] == grp]["industry"].unique())
            print(f"    {grp:<22}: {n:>3} tickers  ← {', '.join(raw)}")
    else:
        df["super_sector"] = df["industry"]

    df["ret"] = df.groupby("ticker")["close"].pct_change().replace([np.inf,-np.inf], np.nan)
    df["val"] = df["close"] * df["volume"] * 1000   # daily traded VND (for breadth cap)
    min_p = max(int(Z_WINDOW * 0.8), 60)
    grp   = df.groupby("ticker")["close"]
    df["sma"]   = grp.transform(lambda x: x.rolling(Z_WINDOW, min_periods=min_p).mean())
    df["std"]   = grp.transform(lambda x: x.rolling(Z_WINDOW, min_periods=min_p).std())
    df["z"]     = (df["close"] - df["sma"]) / df["std"]
    df["drown"] = (df["z"] <= -1.5).astype(int)
    df["peak"]  = (df["z"] >=  1.5).astype(int)

    gb = df.groupby("date")[["drown","peak"]].mean().reset_index()
    gb["g_drown"] = gb["drown"].ewm(span=SMOOTH_WINDOW).mean() * 100
    gb = gb.set_index("date")

    vn = pd.read_csv(vn_path)
    vn.columns = [c.strip().lower() for c in vn.columns]
    vn["date"] = pd.to_datetime(vn["date"], errors="coerce")
    vn = vn.dropna(subset=["date"]).sort_values("date").set_index("date")
    vn["vn_ret"] = vn["close"].pct_change().replace([np.inf,-np.inf], np.nan).fillna(0)

    # Show sector breakdown
    for sec in sorted(df["super_sector"].unique()):
        n = df[df["super_sector"] == sec]["ticker"].nunique()
        print(f"  {sec:<24}: {n} tickers")

    return df, gb, vn[["vn_ret"]], hose_hnx


# ─────────────────────────────────────────────────────────────────
# SECTOR SIGNAL BUILDER
# ─────────────────────────────────────────────────────────────────

def rolling_slope(series, window):
    def _slope(y):
        x    = np.arange(len(y), dtype=float)
        mask = ~np.isnan(y.values)
        if mask.sum() < 3:
            return np.nan
        s, *_ = scipy_stats.linregress(x[mask], y.values[mask])
        return s
    return series.rolling(window, min_periods=max(3, window//2)).apply(_slope, raw=False)


def _vol_score_for_stock(ticker, as_of_date, stock_data):
    """
    Volume accumulation score for one stock as of as_of_date (no lookahead).
    Higher = stronger buyer accumulation at the dip.

    Components:
      rel_vol        (0.25) — recent 10d volume vs 60d baseline; >1 = elevated interest
      vol_pressure   (0.40) — net buying: Σ(vol × sign(close−open)) / Σ(vol)
      price_vol_corr (0.20) — inverted: negative corr (price down, vol up) = accumulation
      dip_vol_slope  (0.15) — slope of volume at successive troughs; + = each dip more contested

    Returns float score or np.nan if insufficient data.
    """
    sd = stock_data.get(ticker)
    if sd is None:
        return np.nan
    hist = sd[sd.index < as_of_date]
    if len(hist) < 60:
        return np.nan

    # 1. Relative volume
    base_v  = hist["volume"].tail(60).mean()
    short_v = hist["volume"].tail(10).mean()
    if base_v == 0:
        return np.nan
    rv_norm = float(np.clip((short_v / base_v) - 1.0, -1.0, 2.0) / 2.0)

    # 2. Vol pressure (net buying dominance)
    rec    = hist.tail(10)
    signs  = np.sign(rec["close"].values - rec["open"].values)
    tot_v  = rec["volume"].sum()
    vp     = float((rec["volume"].values * signs).sum() / tot_v) if tot_v > 0 else 0.0
    vp_norm = float(np.clip(vp, -1.0, 1.0))

    # 3. Price-volume correlation (inverted: accumulation = negative corr)
    ch   = hist.tail(20)
    rets = ch["close"].pct_change().fillna(0).values
    vols = ch["volume"].values.astype(float)
    if vols.std() == 0 or rets.std() == 0:
        pvc_norm = 0.0
    else:
        pvc      = float(np.corrcoef(rets, vols)[0, 1])
        pvc_norm = float(np.clip(-pvc if not np.isnan(pvc) else 0.0, -1.0, 1.0))

    # 4. Dip volume slope (successive troughs attracting more volume?)
    dh  = hist.tail(40)
    cl  = dh["close"].values
    vd  = dh["volume"].values.astype(float)
    trs = [vd[i] for i in range(1, len(cl) - 1)
           if cl[i] < cl[i - 1] and cl[i] < cl[i + 1]]
    if len(trs) >= 3:
        x      = np.arange(len(trs), dtype=float)
        mean_v = np.mean(trs) if np.mean(trs) > 0 else 1.0
        slope  = float(np.polyfit(x, trs, 1)[0])
        dvs_norm = float(np.clip(slope / mean_v, -1.0, 1.0))
    else:
        dvs_norm = 0.0

    return 0.25 * rv_norm + 0.40 * vp_norm + 0.20 * pvc_norm + 0.15 * dvs_norm


def _full_candle_score_for_stock(ticker, as_of_date, stock_data, window=None):
    """
    Recent full-body candle score (no lookahead).
    Measures conviction buying: large bullish body with minimal wicks.

    score per bar = body_pct × body_ratio
      body_pct   = (close - open) / open   (only positive / bullish bars)
      body_ratio = (close - open) / (high - low)  (1.0 = pure marubozu)

    Returns recency-weighted average over the last `window` bars.
    Requires high/low columns; returns np.nan if unavailable.
    """
    if window is None:
        window = FULL_CANDLE_WINDOW
    sd = stock_data.get(ticker)
    if sd is None:
        return np.nan
    hist = sd[sd.index < as_of_date].tail(window)
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
        body_pct   = body / row["open"]
        body_ratio = body / rng
        scores.append(body_pct * body_ratio)

    if not scores:
        return np.nan
    # Recency weight: most recent bar has highest weight
    weights = np.arange(1, len(scores) + 1, dtype=float)
    return float(np.average(scores, weights=weights))


def _ob_score_for_stock(ticker, as_of_date, order_data, window=None):
    """
    Rolling average ob_ratio (buy_volume / sell_volume) over the last `window`
    trading days before as_of_date.  ob_ratio > 1 = buyers more aggressive.

    Returns NaN when order history is missing or has too few rows
    (pre-2018, or stock not in order_history folder) — caller falls back
    to full-candle score in that case.
    """
    if window is None:
        window = OB_RANK_WINDOW
    od = order_data.get(ticker)
    if od is None:
        return np.nan
    hist = od[od["date"] < as_of_date].tail(window)
    if len(hist) < max(2, window // 2):
        return np.nan
    valid = hist["ob_ratio"].dropna()
    return float(valid.mean()) if len(valid) >= 2 else np.nan


def _build_fg_sector_streaks(tickers_df):
    """
    Pre-build per-sector daily fg streak series.
    Returns dict: {sector -> pd.Series(date -> consecutive_fg_buying_days)}
    Coverage: 2018-present (dates before 2018 have no fg data and return 0).
    """
    sector_fg_avg = {}
    for sector in tickers_df["industry"].unique():
        syms = tickers_df[tickers_df["industry"] == sector]["ticker"].tolist()
        per_stock = {}
        for sym in syms:
            fp = os.path.join(INDIVIDUAL_DATA_DIR, f"{sym}.parquet")
            if not os.path.exists(fp):
                continue
            try:
                df = pd.read_parquet(fp).rename(columns={"time": "date"})
                df["date"] = pd.to_datetime(df["date"])
                if "foreign_buy_vol" not in df.columns:
                    continue
                vol = df["volume"].replace(0, np.nan)
                fg_net = df["foreign_buy_vol"].fillna(0) - df["foreign_sell_vol"].fillna(0)
                pct = fg_net / vol * 100
                pct.index = df["date"]
                if pct.notna().sum() > 30:
                    per_stock[sym] = pct
            except Exception:
                continue
        if not per_stock:
            continue
        avg = pd.DataFrame(per_stock).mean(axis=1)
        avg = avg[avg.notna()]
        # Compute streak: count consecutive positive days
        streak = pd.Series(0, index=avg.index, dtype=int)
        for i in range(1, len(avg)):
            streak.iloc[i] = streak.iloc[i - 1] + 1 if avg.iloc[i] > 0 else 0
        sector_fg_avg[sector] = streak
    return sector_fg_avg


def _fg_streak_on_date(sector, date, fg_streaks):
    """Return fg streak for sector on or before date. Returns 0 before 2018."""
    if pd.Timestamp(date) < pd.Timestamp("2018-01-01"):
        return 999  # pass-through — no filter before fg data exists
    s = fg_streaks.get(sector)
    if s is None or s.empty:
        return 999  # sector not covered — pass through
    past = s[s.index <= pd.Timestamp(date)]
    return int(past.iloc[-1]) if len(past) > 0 else 0


def _vn_momentum(vn, as_of_date, window=None):
    """
    Compound VN-Index return over the last `window` trading days ending at as_of_date.
    Returns float: e.g. +0.12 = +12%, -0.15 = -15% over 3 months.
    Returns 0.0 if insufficient history (neutral — don't block entry).
    """
    if window is None:
        window = VNINDEX_GATE_WINDOW
    hist = vn[vn.index <= as_of_date].tail(window + 1)
    if len(hist) < window // 3:   # need at least 1/3 of window
        return 0.0
    return float((1 + hist["vn_ret"]).prod() - 1)


def _vn_gate_multiplier(vn, as_of_date):
    """
    Returns capital size multiplier based on VN-Index momentum gate.
      1.0 = full size  (market healthy or gate OFF)
      0.5 = half size  (SOFT mode, VN down > SOFT threshold)
      0.0 = skip entry (HARD threshold breached)
    """
    if VNINDEX_GATE == "OFF":
        return 1.0
    mom = _vn_momentum(vn, as_of_date)
    if mom <= VNINDEX_GATE_HARD_THRESH:
        return 0.0   # skip
    if VNINDEX_GATE == "SOFT" and mom <= VNINDEX_GATE_SOFT_THRESH:
        return 0.5   # half size
    return 1.0       # full size


def _sector_heat(sector, ref_date, stock_data, ticker_sector_map, window=None):
    """
    Average full-candle score across liquid stocks in the sector as of ref_date.
    Uses top-50% of stocks by score (avoids noise from weak movers).
    Returns float heat score, or 0.0 if insufficient data.
    """
    if window is None:
        window = DEMAND_HEAT_WINDOW
    tickers = [t for t, s in ticker_sector_map.items() if s == sector]
    scores = []
    for ticker in tickers:
        sd = stock_data.get(ticker)
        if sd is None:
            continue
        hist = sd[sd.index <= ref_date]
        # Liquidity gate: median daily turnover >= MIN_LIQUIDITY_VND
        med_to = (hist["close"] * hist["volume"] * 1000).tail(60).median()
        if med_to < MIN_LIQUIDITY_VND:
            continue
        recent = hist.tail(window)
        # Must trade most days in window
        if (recent["volume"] > 0).sum() < int(window * 0.75):
            continue
        fc = _full_candle_score_for_stock(ticker, ref_date, stock_data, window=window)
        if not np.isnan(fc):
            scores.append(fc)
    if not scores:
        return 0.0
    scores.sort(reverse=True)
    top_half = scores[:max(len(scores) // 2, 1)]
    return float(np.mean(top_half))


def _build_heat_series(sectors, all_dates, stock_data, ticker_sector_map,
                       window=None, sample_every=5):
    """
    Pre-compute a time series of sector heat (avg full-candle score) for each
    sector by sampling every `sample_every` trading days, then forward-filling
    to daily frequency.

    Returns: dict[sector -> pd.Series indexed by date]
    """
    if window is None:
        window = DEMAND_HEAT_WINDOW
    result = {sec: {} for sec in sectors}
    sample_dates = all_dates[::sample_every]
    for dt in sample_dates:
        for sec in sectors:
            result[sec][dt] = _sector_heat(sec, dt, stock_data, ticker_sector_map,
                                           window=window)
    heat_ts = {}
    for sec in sectors:
        s = pd.Series(result[sec]).sort_index()
        heat_ts[sec] = s.reindex(all_dates).ffill().bfill().fillna(0.0)
    return heat_ts


def build_sector_signal(df, sector, top_n: int = None):
    """
    Build breadth signal for sector.

    top_n: if set, only use the top N stocks by rolling 60-day median daily
           traded value at each date. Prevents zombie/distressed small caps
           from diluting the signal in large sectors (e.g. Real Estate).
           The buy universe (tcks[sec]) is NOT affected — only the breadth calc.
    """
    # Filter by super_sector
    sec = df[df["super_sector"] == sector].copy()

    # ── Optional breadth-universe cap ────────────────────────────────────────
    if top_n is not None and top_n > 0 and "val" in sec.columns:
        sec = sec.sort_values(["ticker", "date"])
        # Rolling 60-day median value per ticker (point-in-time, no look-ahead)
        sec["_roll_val"] = (sec.groupby("ticker")["val"]
                               .transform(lambda s: s.rolling(60, min_periods=10).median()))
        # At each date keep only top_n tickers by rolling liquidity
        sec["_liq_rank"] = (sec.groupby("date")["_roll_val"]
                               .rank(ascending=False, method="first", na_option="bottom"))
        sec = sec[sec["_liq_rank"] <= top_n].drop(columns=["_roll_val", "_liq_rank"])

    sb  = (sec.groupby("date")
              .agg(s_drown=("drown","mean"), s_peak=("peak","mean"),
                   s_ret=("ret","mean"))
              .reset_index())
    sb["s_ret"]      = sb["s_ret"].fillna(0)
    sb["s_drown_sm"] = sb["s_drown"].ewm(span=SMOOTH_WINDOW).mean() * 100
    sb["s_peak_sm"]  = sb["s_peak"].ewm(span=SMOOTH_WINDOW).mean()  * 100
    sb["s_spread"]   = sb["s_peak_sm"] - sb["s_drown_sm"]
    sb = sb.set_index("date").sort_index()

    spread   = sb["s_spread"]
    velocity = rolling_slope(spread, VELOCITY_WINDOW)

    trough_depth = spread.rolling(TROUGH_WINDOW, min_periods=30).min().abs()
    was_pos      = (spread >= 0).astype(int)
    cumsum       = was_pos.cumsum()
    last_pos     = cumsum.where(was_pos == 1).ffill().fillna(0)
    trough_dur   = (cumsum - last_pos).clip(0, TROUGH_WINDOW)
    cross_up     = ((spread >= 0) & (spread.shift(1) < 0)).astype(int)
    recent_cross = cross_up.rolling(RECOVERY_WINDOW, min_periods=1).sum()

    states = []
    for i in range(len(spread)):
        sp  = spread.iloc[i]
        vel = velocity.iloc[i] if not np.isnan(velocity.iloc[i]) else 0
        rc  = recent_cross.iloc[i]
        if   sp < SPREAD_CRASH:               states.append("CRASH")
        elif sp < 0:                           states.append("DROWNING")
        elif sp >= SPREAD_PEAK and vel < 0:    states.append("PEAKING")
        elif sp >= 0 and rc >= 1:              states.append("RECOVERY")
        else:                                  states.append("LEADING")

    state_s = pd.Series(states, index=spread.index)

    # Track what state preceded each RECOVERY crossing.
    # We look back up to TROUGH_WINDOW days before the zero-cross to find
    # the deepest state reached during the most recent negative period.
    # This lets wants_entry filter: only enter RECOVERY that followed DROWNING or CRASH.
    prev_trough_state = pd.Series("NONE", index=spread.index)
    for i in range(1, len(spread)):
        if states[i] == "RECOVERY" and states[i-1] != "RECOVERY":
            # Just crossed into RECOVERY — look back to find deepest prior state
            lookback = max(0, i - TROUGH_WINDOW)
            prior_states = states[lookback:i]
            if "CRASH" in prior_states:
                prev_trough_state.iloc[i] = "CRASH"
            elif "DROWNING" in prior_states:
                prev_trough_state.iloc[i] = "DROWNING"
            else:
                prev_trough_state.iloc[i] = "SHALLOW"  # barely went negative
        elif states[i] == "RECOVERY":
            # Still in RECOVERY window — carry forward the label
            prev_trough_state.iloc[i] = prev_trough_state.iloc[i-1]

    vel_raw  = velocity.fillna(0).clip(lower=0)
    score    = (
        trough_depth.clip(lower=0) ** 0.5 *
        trough_dur.clip(lower=1).apply(np.log1p) *
        vel_raw ** 0.3
    )
    score    = score.where(state_s.isin(["RECOVERY","LEADING"]), 0.0)
    mom_20d  = sb["s_ret"].rolling(MOMENTUM_LOOKBACK).sum()
    roll_std = spread.rolling(VOL_WINDOW, min_periods=20).std().fillna(spread.std())

    # ── Rolling recovery speed ────────────────────────────────────
    # For each day, look back RECOV_WINDOW_LONG days and compute the
    # average number of days it took the spread to go from trough to
    # zero crossing in each completed negative cycle.
    # This is a pure time-series computation — no backtest feedback.
    #
    # Result: rolling_recov_days[t] = avg recovery speed over recent history
    # Then K[t] = 0.75 - (rolling_recov_days[t] - 40) / 200, clipped [0.25, 0.75]
    # Threshold[t] = K[t] × roll_std[t]

    def compute_rolling_recovery(spread_series, window):
        """
        Rolling average recovery days (trough to zero crossing).
        For each date, looks back `window` days and measures completed cycles.
        Returns a Series of avg_recovery_days, forward-filled between cycles.
        """
        vals  = spread_series.values
        idx   = spread_series.index
        n     = len(vals)
        out   = np.full(n, np.nan)

        for i in range(window, n):
            # Window of spread values ending at i
            w_vals = vals[max(0, i - window):i]
            w_idx  = list(range(max(0, i - window), i))

            # Find completed negative cycles within window
            cycle_lengths = []
            in_neg     = False
            trough_pos = None

            for j, v in enumerate(w_vals):
                if v < 0 and not in_neg:
                    in_neg     = True
                    trough_pos = j
                elif v < 0 and in_neg:
                    pass   # still descending/in trough
                elif v >= 0 and in_neg:
                    # crossed back above zero — completed cycle
                    in_neg = False
                    if trough_pos is not None:
                        cycle_lengths.append(j - trough_pos)
                    trough_pos = None

            if cycle_lengths:
                out[i] = np.mean(cycle_lengths)
            # else: stays NaN until enough cycles observed

        result = pd.Series(out, index=idx)
        # Forward-fill: carry last known recovery estimate forward
        result = result.ffill()
        # Backfill first window with the first valid estimate
        result = result.bfill()
        # Fallback: if still NaN, use 90d (neutral K=0.5)
        result = result.fillna(90)
        return result

    rolling_recov = compute_rolling_recovery(spread, RECOV_WINDOW_LONG)

    # K series: monotonic formula from rolling_recov
    # K = clip(0.75 - (recov_days - 40) / 200, 0.25, 0.75)
    rolling_K = (0.75 - (rolling_recov - 40) / 200).clip(0.25, 0.75)

    return pd.DataFrame({
        "spread":        spread,
        "velocity":      velocity,
        "state":         state_s,
        "prev_trough":   prev_trough_state,
        "score":         score,
        "s_ret":         sb["s_ret"],
        "mom_20d":       mom_20d,
        "roll_std":      roll_std,
        "rolling_recov": rolling_recov,  # avg recovery days over recent history
        "rolling_K":     rolling_K,      # dynamic K derived from rolling_recov
    })


def get_entry_threshold(row, sector):
    """
    threshold(t) = K(t) × rolling_std(spread, VOL_WINDOW)

    K(t) is computed from the rolling average recovery speed over
    the past RECOV_WINDOW_LONG days — no hardcoding, no backtest feedback.
    Formula: K = clip(0.75 - (rolling_recov_days - 40) / 200, 0.25, 0.75)

    Exception: EARLY_ENTRY_SECTORS (Banks) always use EARLY_ENTRY_K=0.25
    because the credit cycle is priced before the spread bottoms.
    """
    if sector in EARLY_ENTRY_SECTORS:
        k = EARLY_ENTRY_K
    else:
        k = float(row["rolling_K"]) if not np.isnan(row["rolling_K"]) else 0.50
    roll_std = float(row["roll_std"]) if not np.isnan(row["roll_std"]) else 5.0
    return k * roll_std


def wants_entry(row, sector="", in_cooldown=False):
    """
    Entry: RECOVERY or LEADING(score≥MIN_ENTRY_SCORE).
    Gated by:
      1. Vol-adjusted spread threshold: spread > K(t) × rolling_std(60d)
      2. Post-peak cooldown: RECOVERY entries blocked for COOLDOWN_AFTER_PEAK
         days after any peaking exit. LEADING entries still allowed.
    """
    if row is None:
        return False
    threshold = get_entry_threshold(row, sector)
    if float(row["spread"]) < threshold:
        return False
    state = row["state"]
    # Block RECOVERY (but not LEADING) during post-peak cooldown
    if in_cooldown and state == "RECOVERY":
        return False
    return (state == "RECOVERY") or \
           (state == "LEADING" and float(row["score"]) >= MIN_ENTRY_SCORE)


def entry_score(row):
    """Use raw trough opportunity score — same as original 2-sector."""
    if row is None:
        return 0.0
    return float(row["score"]) if not np.isnan(row["score"]) else 0.0


def momentum_score(row):
    """Alias kept for compatibility."""
    return entry_score(row)


def wants_exit(row):
    sp  = row["spread"]
    vel = row["velocity"] if not np.isnan(row["velocity"]) else 0
    if sp < SPREAD_EXIT:
        return True, f"spread_exit({sp:.1f})"
    if row["state"] == "PEAKING":
        return True, "peaking"
    if SPREAD_HIGH_EXIT and sp >= SPREAD_HIGH_EXIT and vel < 0:
        return True, f"high_spread({sp:.0f})"
    if not np.isnan(row["mom_20d"]) and row["mom_20d"] < MOMENTUM_FLOOR:
        return True, f"momentum({row['mom_20d']*100:.1f}%)"
    return False, None


# ─────────────────────────────────────────────────────────────────
# EXECUTION HELPERS
# ─────────────────────────────────────────────────────────────────

def get_open_price(ticker, date, stock_data):
    if ticker not in stock_data:
        return None
    sd     = stock_data[ticker]
    future = sd[sd.index >= date]
    if future.empty:
        return None
    return future.iloc[0]["open"], future.index[0]


def compute_kijun(ticker, ref_date, stock_data):
    """
    Compute Kijun-sen (26-period baseline) for ticker as of ref_date.
    Uses high/low if available (proper Ichimoku), otherwise close proxy.
    Returns kijun value, or None if insufficient history (caller buys immediately).
    """
    sd = stock_data.get(ticker)
    if sd is None:
        return None
    hist = sd[sd.index < ref_date].tail(KIJUN_PERIOD)
    if len(hist) < max(KIJUN_PERIOD // 2, 5):
        return None   # not enough history — treat as "ready to buy"
    if "high" in sd.columns and "low" in sd.columns:
        return (hist["high"].max() + hist["low"].min()) / 2
    return (hist["close"].max() + hist["close"].min()) / 2


def buy_sector(sector, signal_date, exec_date, sector_tickers,
               stock_data, available_cash, kijun_filter=False,
               max_mom_pct=None, order_data=None):
    """
    Buy all eligible stocks in a sector equally weighted.

    Execution-time filters (applied at the moment of buying):
      1. Stock must have a valid open price on/after exec_date
      2. Volume > 0 on the execution date (not suspended/halted)
      3. Dynamic liquidity: our allocation ≤ MAX_PARTICIPATION_PCT of the
         stock's own 20-day median daily traded value.
         As portfolio grows, illiquid stocks automatically drop out.
         This prevents price impact and ensures fills are realistic.
      4. max_mom_pct: if set, skip stocks whose 20d momentum exceeds this
         value at entry time.  Used for DEMAND_EARLY to avoid buying stocks
         that have already run before the signal fired.

    MAX_PARTICIPATION_PCT = 0.20 (our order ≤ 20% of typical daily volume)
    """
    eligible = []
    # First pass: find all stocks with valid prices and non-zero volume
    for ticker in sorted(sector_tickers):
        result = get_open_price(ticker, exec_date, stock_data)
        if result is None:
            continue
        op, actual_date = result
        if op <= 0:
            continue

        sd = stock_data.get(ticker)
        if sd is None:
            continue

        # Check 1: volume > 0 on execution date (not suspended)
        exec_data = sd[sd.index == actual_date]
        if exec_data.empty:
            # Try within 2 days
            exec_data = sd[(sd.index >= actual_date) &
                          (sd.index <= actual_date + pd.Timedelta(days=2))]
        if exec_data.empty or exec_data.iloc[0]["volume"] == 0:
            continue   # suspended or no data — skip

        # Compute 20-day median daily traded value before exec_date
        hist = sd[sd.index < actual_date].tail(20)
        if len(hist) >= 5:
            median_val = (hist["close"] * hist["volume"] * 1000).median()
        else:
            # Not enough history — use static minimum as fallback
            median_val = MIN_LIQUIDITY_VND

        eligible.append((ticker, op, actual_date, median_val))

    if not eligible:
        return [], 0.0

    # Second pass: apply dynamic liquidity filter
    # Tentative allocation = available_cash / n_eligible
    # But allocation must be ≤ MAX_PARTICIPATION_PCT × median_daily_value
    MAX_PARTICIPATION_PCT = 0.20

    # Iterate: some stocks may be too illiquid at current capital level,
    # which raises allocation for remaining stocks, which may exclude more.
    # Converge in max 3 iterations.
    included = eligible[:]
    for _ in range(3):
        if not included:
            break
        alloc_per = available_cash / len(included)
        still_in  = []
        for ticker, op, actual_date, median_val in included:
            max_alloc = median_val * MAX_PARTICIPATION_PCT
            if alloc_per <= max_alloc:
                still_in.append((ticker, op, actual_date, median_val))
        if len(still_in) == len(included):
            break   # converged
        included = still_in

    if not included:
        return [], 0.0

    # ── Per-sector stock selection ─────────────────────────────────
    # Sector-specific method from experiment (volume_confirm backtest):
    #   VOL_LEADERS / VOL_RANK — rank by volume accumulation score
    #   MOM_BOT50 / MOM_BOT30  — rank by 20d momentum (most oversold)
    #   ALL                    — no filtering
    sec_method = SECTOR_STOCK_SELECTION.get(sector, STOCK_SELECTION)

    if sec_method in ("OB_RANK", "OB_BOT") and len(included) >= 2:
        # Rank by rolling ob_ratio (buy_volume/sell_volume) from order_history.
        # OB_RANK: highest ob_ratio first (accumulation leaders)
        # OB_BOT:  lowest ob_ratio first (contrarian — most selling pressure,
        #          hasn't bounced yet, more mean-reversion upside remaining)
        # Falls back to full-candle score when order history unavailable (pre-2018).
        ob_scored = []
        has_ob_data = False
        for ticker, op, actual_date, median_val in included:
            sc = _ob_score_for_stock(ticker, actual_date, order_data or {})
            if not np.isnan(sc):
                has_ob_data = True
            else:
                # Pre-2018 fallback: full-candle for OB_RANK, inverted for OB_BOT
                sc = _full_candle_score_for_stock(ticker, exec_date, stock_data)
                sc = sc if not np.isnan(sc) else 0.0
                if sec_method == "OB_BOT":
                    sc = -sc   # invert so lowest candle score = most oversold
            ob_scored.append((ticker, op, actual_date, median_val, sc))
        # OB_RANK: descending (high ob = accumulation); OB_BOT: ascending (low ob = oversold)
        ob_scored.sort(key=lambda x: x[4], reverse=(sec_method == "OB_RANK"))
        n_keep = min(OB_RANK_TOP_N, max(len(ob_scored) // 2, 2))
        included = [(t, o, d, m) for t, o, d, m, _ in ob_scored[:n_keep]]

    elif sec_method in ("VOL_LEADERS", "VOL_RANK") and len(included) >= 2:
        # Score each stock by volume accumulation signal (no lookahead).
        # VOL_LEADERS_USE_FULL_CANDLE=True: rank by recent full-body candle
        # score (body_pct × body_ratio, recency-weighted) instead of the
        # accumulated vol_score — favours stocks with a recent conviction
        # buying session rather than stocks that accumulated weeks ago.
        vol_scored = []
        for ticker, op, actual_date, median_val in included:
            if VOL_LEADERS_USE_FULL_CANDLE and sec_method == "VOL_LEADERS":
                rank_score = _full_candle_score_for_stock(
                    ticker, exec_date, stock_data)
                if np.isnan(rank_score):
                    rank_score = -99.0
            else:
                vs = _vol_score_for_stock(ticker, exec_date, stock_data)
                rank_score = float(vs) if not np.isnan(vs) else -99.0
            vol_scored.append((ticker, op, actual_date, median_val, rank_score))
        vol_scored.sort(key=lambda x: -x[4])   # leaders first (highest score)
        if sec_method == "VOL_LEADERS":
            n_keep = min(VOL_LEADERS_N, max(len(vol_scored) // 2, 2))
        else:   # VOL_RANK: top 50%
            n_keep = max(len(vol_scored) // 2, 3)
        included = [(t, o, d, m) for t, o, d, m, _ in vol_scored[:n_keep]]

    elif sec_method != "ALL" and len(included) >= 4:
        # Momentum / MA ranking (MOM_BOT50, MA_BOT50, etc.)
        scored = []
        for ticker, op, actual_date, median_val in included:
            sd = stock_data.get(ticker)
            score = 0.0
            if sd is not None:
                hist = sd[sd.index < actual_date]
                if sec_method in ("MOM_BOT50", "MOM_BOT30"):
                    if len(hist) >= 22:
                        p20 = hist["close"].iloc[-20]
                        if p20 > 0:
                            score = (op - p20) / p20
                elif sec_method in ("MA_BOT50", "MA_BOT30"):
                    if len(hist) >= 60:
                        ma = hist["close"].tail(200).mean()
                        if ma > 0:
                            score = (op - ma) / ma
            scored.append((ticker, op, actual_date, median_val, score))
        scored.sort(key=lambda x: x[4])   # ascending: most oversold first
        if sec_method in ("MOM_BOT50", "MA_BOT50"):
            cutoff = max(len(scored) // 2, 3)
        else:
            cutoff = max(len(scored) * 3 // 10, 3)
        included = [(t, o, d, m) for t, o, d, m, _ in scored[:cutoff]]

    # ── Factor-enhanced ranking (quarterly fundamentals) ─────────────
    # Second-pass filter: within the candidate list produced above,
    # rank by earnings quality (np_yoy, accel_score, ROE) and keep top %.
    # Catches falling knives that passed the price/volume screen but have
    # deteriorating fundamentals — the key risk in sector recovery plays.
    if (FACTOR_SELECTION_ENABLED
            and _FACTOR_RANKER_AVAILABLE
            and _QFEAT is not None
            and FACTOR_SECTORS.get(sector, True)   # per-sector toggle
            and len(included) >= 3):
        tickers_in   = [t for t, o, d, m in included]
        ranked       = rank_by_factor(tickers_in, exec_date, _QFEAT,
                                      top_pct=FACTOR_TOP_PCT,
                                      min_np_yoy=FACTOR_MIN_NP_YOY,
                                      sector=sector)
        # ranked is sorted best-first; slice to top_pct (rank_by_factor handles this)
        keep_set     = set(ranked)
        factor_filtered = [(t, o, d, m) for t, o, d, m in included
                           if t in keep_set]
        if factor_filtered:   # safety: never leave empty
            included = factor_filtered

    # ── Fundamental quality filter ────────────────────────────────
    # Drops stocks failing PB/PE/ROE/OCF thresholds (point-in-time).
    # Toggle via FUNDAMENTAL_FILTER_ENABLED; data loaded into _FUND_DATA at startup.
    if FUNDAMENTAL_FILTER_ENABLED:
        filtered = [
            (t, o, d, m) for t, o, d, m in included
            if passes_fundamental_filter(t, exec_date, _FUND_DATA)
        ]
        if filtered:
            included = filtered
        # If filter removes everyone, fall back to unfiltered (safety net)

    # ── Investor-flow ranking (FLOW_SIGNAL_ENABLED) ───────────────────────────
    # Uses FlowSignalEngine.smart_score() — higher = more domestic institutional
    # accumulation.  Tickers with no flow data score 0.0 and sort last.
    # Only active from Sep 2024 onwards (when flow data starts).
    # Falls back to full included list if ranking produces fewer than 3 stocks.
    if (FLOW_SIGNAL_ENABLED
            and _FLOW_ENGINE is not None
            and (FLOW_SECTORS is None or sector in FLOW_SECTORS)
            and len(included) >= 3
            and exec_date >= pd.Timestamp("2024-09-16")):
        flow_scored = []
        for item in included:
            ticker, op, actual_date, median_val = item
            sc = _FLOW_ENGINE.smart_score(ticker, actual_date, window=FLOW_RANK_WINDOW)
            # Skip stocks with active distribution alerts (moved to back)
            dist = _FLOW_ENGINE.distribution_alert(ticker, actual_date)
            flow_scored.append((ticker, op, actual_date, median_val, sc, dist))
        # Sort: non-distributing first, then by smart_score descending
        flow_scored.sort(key=lambda x: (x[5], -x[4]))
        n_keep = min(FLOW_RANK_TOP_N, len(flow_scored)) if FLOW_RANK_TOP_N else len(flow_scored)
        flow_filtered = [(t, o, d, m) for t, o, d, m, _s, _dist in flow_scored[:n_keep]]
        if len(flow_filtered) >= 3:
            included = flow_filtered

    # ── Late-entry momentum cap (DEMAND_EARLY only) ────────────────
    # Skip stocks that have already moved more than max_mom_pct in the
    # past 20 days.  These are "leaders" — they generated the heat signal
    # but are now expensive relative to where the rally started.
    # We prefer "laggards" — same sector, same heat, haven't moved yet.
    if max_mom_pct is not None and included:
        filtered = []
        for ticker, op, actual_date, median_val in included:
            sd = stock_data.get(ticker)
            if sd is None:
                filtered.append((ticker, op, actual_date, median_val))
                continue
            hist = sd[sd.index < actual_date]
            if len(hist) >= 21:
                p20 = hist["close"].iloc[-20]
                mom = (op / p20 - 1) if p20 > 0 else 0.0
            else:
                mom = 0.0
            if mom <= max_mom_pct:
                filtered.append((ticker, op, actual_date, median_val))
        # Only apply filter if it doesn't remove ALL stocks
        if filtered:
            included = filtered

    # ── Ichimoku Kijun-sen entry filter (T1 only, kijun_filter=True) ──
    # Stocks extended above their Kijun are deferred — we wait for a
    # pullback.  Equal alloc_per is computed over ALL stocks so weighting
    # is preserved whether a stock buys today or is deferred.
    n_total   = len(included)
    alloc_per = available_cash / n_total

    deferred  = []   # stocks to watch for a better entry
    buy_now   = []

    if kijun_filter and ENTRY_TIMING_KIJUN:
        for item in included:
            ticker, op, actual_date, median_val = item
            kijun = compute_kijun(ticker, exec_date, stock_data)
            if kijun is None or (op / kijun - 1) <= KIJUN_BUY_THRESHOLD:
                buy_now.append(item)
            else:
                deferred.append({
                    "ticker":      ticker,
                    "sector":      sector,
                    "signal_date": signal_date,
                    "entry_state": "",   # filled by caller
                    "alloc":       alloc_per,
                    "kijun":       kijun,
                    "deferred_on": exec_date,
                })
    else:
        buy_now = included

    positions   = []
    total_spent = 0.0

    for ticker, op, actual_date, median_val in buy_now:
        price_vnd = op * 1000
        shares    = int(alloc_per / (price_vnd * (1 + FRICTION)))
        if shares <= 0:
            continue
        spent        = shares * price_vnd * (1 + FRICTION)
        total_spent += spent
        positions.append({
            "ticker":      ticker,
            "sector":      sector,
            "signal_date": signal_date,
            "entry_date":  actual_date,
            "entry_price": op,
            "peak_price":  op,
            "shares":      shares,
            "cost":        spent,
        })

    return positions, total_spent, deferred


def sell_all(positions, exec_date, stock_data, exit_reason, signal_state):
    """
    Sell all open positions at exec_date open price.

    T+3 enforcement:
      - Stocks with no forward data (suspended/delisted) exit at entry price:
        capital is returned at cost minus friction. Never skipped.
      - Stocks where only price found is same-day as entry exit at entry price.

    Price sanity:
      - Exit price capped at entry × (1.07 ^ trading_days) to block
        data gaps or unadjusted splits from producing impossible gains.
        Uses HOSE daily limit (±7%) × trading days (calendar × 0.71).
    """
    trades         = []
    total_proceeds = 0.0

    for pos in positions:
        ticker = pos["ticker"]
        result = get_open_price(ticker, exec_date, stock_data)

        if result is None:
            # No forward data — use last known close, or entry price as fallback
            if ticker in stock_data:
                last = stock_data[ticker][stock_data[ticker].index <= exec_date]
                if last.empty:
                    exit_p           = pos["entry_price"]
                    actual_exit_date = pos["entry_date"]
                else:
                    exit_p           = last.iloc[-1]["close"]
                    actual_exit_date = last.index[-1]
            else:
                exit_p           = pos["entry_price"]
                actual_exit_date = pos["entry_date"]
        else:
            exit_p, actual_exit_date = result

        # T+3 / data availability:
        # If no price exists after entry (stock suspended/delisted immediately),
        # exit at entry price — recover cost minus friction rather than losing it.
        if actual_exit_date <= pos["entry_date"]:
            exit_p           = pos["entry_price"]
            actual_exit_date = pos["entry_date"]

        hold_days    = max((actual_exit_date - pos["entry_date"]).days, 1)
        trading_days = max(hold_days * 0.71, 1)
        max_gain     = (1.07 ** trading_days) - 1
        gain         = (exit_p - pos["entry_price"]) / pos["entry_price"]
        if gain > max_gain:
            exit_p = pos["entry_price"] * (1 + max_gain)

        proceeds        = pos["shares"] * exit_p * 1000 * (1 - FRICTION)
        pnl_vnd         = proceeds - pos["cost"]
        total_proceeds += proceeds

        trades.append({
            "ticker":       ticker,
            "sector":       pos["sector"],
            "signal_date":  pos["signal_date"],
            "entry_date":   pos["entry_date"],
            "exit_date":    actual_exit_date,
            "hold_days":    hold_days,
            "entry_price":  pos["entry_price"],
            "exit_price":   round(exit_p, 4),
            "shares":       pos["shares"],
            "cost_vnd":     round(pos["cost"]),
            "proceeds_vnd": round(proceeds),
            "pnl_vnd":      round(pnl_vnd),
            "pnl_pct":      round(pnl_vnd / pos["cost"] * 100, 2),
            "exit_reason":  exit_reason,
            "entry_state":  signal_state,
        })

    return trades, total_proceeds


def sell_tp_stocks(positions, today, stock_data):
    """
    Check each open position daily for individual-stock exits:

    1. Trailing stop (STOCK_TRAILING_STOP_PCT):
         Update peak_price if today's price is a new high.
         Sell when price falls more than STOCK_TRAILING_STOP_PCT below peak.
         This lets winners run and captures most of the gain on the way down.
         On HOSE (7%/day limit) worst-case overshoot is ~7% past the stop.

    2. Fixed take-profit (STOCK_TP_PCT):
         Sell immediately when gain from entry reaches this level.
         Optional hard ceiling — set to None to rely on trailing stop only.

    3. Entry stop-loss (STOCK_SL_PCT):
         Sell when loss from *entry price* exceeds this level.
         Protects against momentum picks that immediately reverse before
         the trailing stop has had any chance to move up.

    Returns (exit_trades, remaining_positions, total_proceeds).
    Peak prices are updated in-place on remaining positions.
    """
    nothing_active = (STOCK_TRAILING_STOP_PCT is None
                      and STOCK_TP_PCT is None
                      and STOCK_SL_PCT is None
                      and LAGGARD_FLAT_THRESH is None
                      and LAGGARD_LOSS_THRESH is None
                      and not FLOW_DIST_EXIT
                      and not FLOW_PEAK_EXIT)
    if nothing_active or not positions:
        return [], positions, 0.0

    remaining      = []
    exit_trades    = []
    total_proceeds = 0.0

    for pos in positions:
        ticker = pos["ticker"]
        result = get_open_price(ticker, today, stock_data)
        if result is None:
            remaining.append(pos)
            continue

        current_price, actual_date = result

        # T+3: cannot exit on same day as purchase
        if actual_date <= pos["entry_date"]:
            remaining.append(pos)
            continue

        hold_days = max((actual_date - pos["entry_date"]).days, 1)
        gain      = (current_price - pos["entry_price"]) / pos["entry_price"]

        # Sanity check: physically impossible gain → skip (data gap / split)
        trading_days = max(hold_days * 0.71, 1)
        max_possible = (1.07 ** trading_days) - 1
        if gain > max_possible:
            remaining.append(pos)
            continue

        # Update peak price (trailing stop ratchet — only moves up)
        if current_price > pos.get("peak_price", pos["entry_price"]):
            pos["peak_price"] = current_price

        peak  = pos.get("peak_price", pos["entry_price"])
        drawdown_from_peak = (peak - current_price) / peak

        # Determine exit reason (checked in priority order)
        # Trailing stop only arms once gain >= TRAIL_ACTIVATE_PCT (whipsaw guard)
        trail_armed = (STOCK_TRAILING_STOP_PCT is not None
                       and gain >= TRAIL_ACTIVATE_PCT)
        exit_reason = None
        if STOCK_TP_PCT is not None and gain >= STOCK_TP_PCT:
            exit_reason = f"stock_tp({STOCK_TP_PCT*100:.0f}%)"
        elif trail_armed and drawdown_from_peak >= STOCK_TRAILING_STOP_PCT:
            peak_gain = (peak - pos["entry_price"]) / pos["entry_price"]
            exit_reason = f"trail_stop({STOCK_TRAILING_STOP_PCT*100:.0f}%,peak+{peak_gain*100:.0f}%)"
        elif STOCK_SL_PCT is not None and gain <= -STOCK_SL_PCT:
            exit_reason = f"stock_sl({STOCK_SL_PCT*100:.0f}%)"
        # ── Laggard exits ────────────────────────────────────────────────────
        # Loss laggard: still down >X% after LAGGARD_LOSS_DAYS — going wrong way
        elif (LAGGARD_LOSS_THRESH is not None
              and hold_days >= LAGGARD_LOSS_DAYS
              and gain <= -LAGGARD_LOSS_THRESH):
            exit_reason = f"laggard_loss({gain*100:.1f}%,{hold_days}d)"
        # Flat laggard: still not running after LAGGARD_FLAT_DAYS — dead weight
        elif (LAGGARD_FLAT_THRESH is not None
              and hold_days >= LAGGARD_FLAT_DAYS
              and gain < LAGGARD_FLAT_THRESH):
            exit_reason = f"laggard_flat({gain*100:.1f}%,{hold_days}d)"

        # ── Investor-flow peak exit ───────────────────────────────────
        # FLOW_PEAK_EXIT: composite signal — distribution alert, heavy-sell
        # regime, or momentum flip turning negative. Identifies the price
        # peak by watching when smart money starts exiting.
        if (exit_reason is None
                and FLOW_PEAK_EXIT
                and _FLOW_ENGINE is not None
                and actual_date >= pd.Timestamp("2024-09-16")):
            triggered, reason = _FLOW_ENGINE.peak_exit_signal(ticker, actual_date)
            if triggered:
                exit_reason = reason

        # ── Legacy: simple distribution-alert exit ────────────────────
        # FLOW_DIST_EXIT: original single-signal version (backward compat).
        # Use FLOW_PEAK_EXIT instead for the composite peak signal.
        if (exit_reason is None
                and FLOW_DIST_EXIT
                and _FLOW_ENGINE is not None
                and actual_date >= pd.Timestamp("2024-09-16")
                and _FLOW_ENGINE.distribution_alert(ticker, actual_date)):
            exit_reason = "flow_distribution"

        if exit_reason is not None:
            proceeds        = pos["shares"] * current_price * 1000 * (1 - FRICTION)
            pnl_vnd         = proceeds - pos["cost"]
            total_proceeds += proceeds
            exit_trades.append({
                "ticker":       ticker,
                "sector":       pos["sector"],
                "signal_date":  pos["signal_date"],
                "entry_date":   pos["entry_date"],
                "exit_date":    actual_date,
                "hold_days":    hold_days,
                "entry_price":  pos["entry_price"],
                "exit_price":   current_price,
                "shares":       pos["shares"],
                "cost_vnd":     round(pos["cost"]),
                "proceeds_vnd": round(proceeds),
                "pnl_vnd":      round(pnl_vnd),
                "pnl_pct":      round(pnl_vnd / pos["cost"] * 100, 2),
                "exit_reason":  exit_reason,
                "entry_state":  pos.get("entry_state", ""),
            })
        else:
            remaining.append(pos)

    return exit_trades, remaining, total_proceeds


def etf_get_price(date, stock_data):
    """Get ETF open price on or after date. Returns (price, actual_date) or None."""
    if ETF_TICKER is None or ETF_TICKER not in stock_data:
        return None
    sd     = stock_data[ETF_TICKER]
    future = sd[sd.index >= date]
    if future.empty:
        return None
    return future.iloc[0]["open"], future.index[0]


def etf_buy(available_cash, date, stock_data):
    """
    Buy ETF with ETF_ALLOC fraction of available cash.
    Returns (etf_position dict, spent) or (None, 0) if not available.
    """
    if ETF_TICKER is None:
        return None, 0.0
    result = etf_get_price(date, stock_data)
    if result is None:
        return None, 0.0
    price, actual_date = result
    deploy    = available_cash * ETF_ALLOC
    price_vnd = price * 1000
    shares    = int(deploy / (price_vnd * (1 + FRICTION)))
    if shares <= 0:
        return None, 0.0
    spent = shares * price_vnd * (1 + FRICTION)
    return {
        "ticker":      ETF_TICKER,
        "entry_date":  actual_date,
        "entry_price": price,
        "shares":      shares,
        "cost":        spent,
    }, spent


def etf_sell(etf_pos, date, stock_data, all_dates):
    """
    Sell ETF position. Returns (proceeds, trade_record) with T+3 settlement.
    """
    if etf_pos is None:
        return 0.0, None
    result = etf_get_price(date, stock_data)
    if result is None:
        # No price — use last known close
        sd   = stock_data.get(ETF_TICKER)
        last = sd[sd.index <= date] if sd is not None else None
        if last is None or last.empty:
            price      = etf_pos["entry_price"]
            actual_date= etf_pos["entry_date"]
        else:
            price       = last.iloc[-1]["close"]
            actual_date = last.index[-1]
    else:
        price, actual_date = result

    # Sanity cap
    hold     = max((actual_date - etf_pos["entry_date"]).days, 1)
    trading  = max(hold * 0.71, 1)
    max_gain = (1.07 ** trading) - 1
    gain     = (price - etf_pos["entry_price"]) / etf_pos["entry_price"]
    if gain > max_gain:
        price = etf_pos["entry_price"] * (1 + max_gain)

    proceeds = etf_pos["shares"] * price * 1000 * (1 - FRICTION)
    pnl      = proceeds - etf_pos["cost"]
    trade    = {
        "ticker":       ETF_TICKER,
        "sector":       "ETF",
        "signal_date":  etf_pos["entry_date"],
        "entry_date":   etf_pos["entry_date"],
        "exit_date":    actual_date,
        "hold_days":    hold,
        "entry_price":  etf_pos["entry_price"],
        "exit_price":   round(price, 4),
        "shares":       etf_pos["shares"],
        "cost_vnd":     round(etf_pos["cost"]),
        "proceeds_vnd": round(proceeds),
        "pnl_vnd":      round(pnl),
        "pnl_pct":      round(pnl / etf_pos["cost"] * 100, 2),
        "exit_reason":  "etf_sector_entry",
        "entry_state":  "ETF",
    }
    return proceeds, trade


# ─────────────────────────────────────────────────────────────────
# T+3 SETTLEMENT QUEUE
# ─────────────────────────────────────────────────────────────────

class SettlementQueue:
    """
    Tracks pending cash from stock sales.
    Vietnamese market T+3: proceeds from selling on trading day N
    are available at the START of trading day N+3 (trading days only).

    Usage:
        sq = SettlementQueue()
        sq.add(proceeds, sell_date, all_trading_dates)
        available = sq.release(today, all_trading_dates)
    """
    def __init__(self):
        self.queue = deque()   # each item: (settlement_date, amount)

    def add(self, amount, sell_date, trading_dates):
        """Schedule cash to arrive SETTLEMENT_DAYS trading days after sell_date."""
        td_list = sorted(trading_dates)
        try:
            idx = td_list.index(sell_date)
        except ValueError:
            # sell_date not in list — find next
            idx = next((i for i, d in enumerate(td_list) if d >= sell_date), len(td_list)-1)

        settle_idx = min(idx + SETTLEMENT_DAYS, len(td_list) - 1)
        settle_date = td_list[settle_idx]
        self.queue.append((settle_date, amount))

    def release(self, today):
        """Release all cash whose settlement date <= today. Returns total released."""
        released = 0.0
        while self.queue and self.queue[0][0] <= today:
            _, amt = self.queue.popleft()
            released += amt
        return released

    def pending_total(self):
        return sum(amt for _, amt in self.queue)

    def next_settlement_date(self):
        return self.queue[0][0] if self.queue else None


# ─────────────────────────────────────────────────────────────────
# 3-SECTOR BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────

def run_backtest(df_all, gb, vn, stock_data, cutoff, regime_enabled=None, fg_confirm_enabled=None):
    # ── Build signals for ALL qualifying super-sectors ─────────────
    all_sector_names = sorted(df_all["super_sector"].dropna().unique())
    if SECTORS_OVERRIDE:
        all_sector_names = [s for s in SECTORS_OVERRIDE if s in all_sector_names]

    print(f"  Building signals for {len(all_sector_names)} sectors...")
    sigs = {}
    tcks = {}
    for sec in all_sector_names:
        n     = df_all[df_all["super_sector"] == sec]["ticker"].nunique()
        cap   = SECTOR_BREADTH_CAP.get(sec)
        sigs[sec] = build_sector_signal(df_all, sec, top_n=cap)
        tcks[sec] = sorted(df_all[df_all["super_sector"] == sec]["ticker"].unique())
        cap_str = f" (breadth cap: top {cap})" if cap else ""
        print(f"    {sec:<28}: {n} tickers{cap_str}")

    SECTORS = list(sigs.keys())
    # Map ticker → super_sector (used by demand-early heat check)
    ticker_sector_map = dict(zip(df_all["ticker"], df_all["super_sector"]))
    print(f"  Candidates: {' | '.join(SECTORS)}")
    print(f"  Single pool — 100M deployed into best sector at a time")

    # ── Regime engine init ────────────────────────────────────────
    _use_regime       = REGIME_ENGINE_ENABLED if regime_enabled is None else regime_enabled
    _regime_is_panic  = False
    _regime_series    = None   # daily DataFrame from build_regime_series()
    if _use_regime:
        try:
            from regime_engine import build_regime_series as _build_regime
            from regime_engine import _load_vnindex, _load_all_stocks
            _regime_vn = _load_vnindex()
            _regime_sd = _load_all_stocks(sample_cap=150)
            _regime_series = _build_regime(_regime_vn, _regime_sd)
            print("  Regime engine ready — daily PANIC detection active")
        except ImportError:
            print("  [WARN] regime_engine.py not found — running without regime filter")

    all_dates = sorted(set().union(*[set(sigs[s].index) for s in SECTORS]))

    # ── Load order-flow data for OB_RANK stock selection ─────────
    _order_data = {}
    if any(m == "OB_RANK" for m in SECTOR_STOCK_SELECTION.values()):
        import glob as _glob
        for _f in _glob.glob(os.path.join(OB_ORDER_DIR, "*.parquet")):
            _sym = os.path.basename(_f).replace(".parquet", "").upper()
            try:
                _d = pd.read_parquet(_f)[["date", "ob_ratio"]].dropna(subset=["ob_ratio"])
                _d["date"] = pd.to_datetime(_d["date"])
                _order_data[_sym] = _d.sort_values("date").reset_index(drop=True)
            except Exception:
                pass
        print(f"  Order-flow data loaded: {len(_order_data)} stocks (OB_RANK)")

    # ── Load fg streak data for confirmation filter ───────────────
    _use_fg_confirm = FG_CONFIRM_ENABLED if fg_confirm_enabled is None else fg_confirm_enabled
    _fg_streaks = {}
    if _use_fg_confirm:
        try:
            _tickers_sectors = pd.read_csv(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_sectors.csv")
            )
            _tickers_sectors = _tickers_sectors[
                _tickers_sectors["industry"].isin(FG_CONFIRM_SECTORS)
            ]
            _fg_streaks = _build_fg_sector_streaks(_tickers_sectors)
            print(f"  FG confirm: loaded streaks for {list(_fg_streaks.keys())}")
        except Exception as _e:
            print(f"  [WARN] FG confirm load failed: {_e}")

    # ── Pre-compute heat series for adaptive demand threshold ──────
    heat_ts = {}
    if DEMAND_EARLY_ENABLED and DEMAND_HEAT_USE_ADAPTIVE:
        print("  Pre-computing sector heat series (adaptive threshold)...")
        heat_ts = _build_heat_series(SECTORS, all_dates, stock_data,
                                     ticker_sector_map,
                                     window=DEMAND_HEAT_WINDOW,
                                     sample_every=5)
        print(f"    Done — {len(all_dates)} days, sampled every 5")

    # ── State — identical to original 2-sector engine ─────────────
    settled_cash        = float(CAPITAL_VND)
    sq                  = SettlementQueue()
    open_positions      = []
    held_sector         = None
    held_state          = None
    days_held           = 0
    pending_exit        = (False, None, None)
    pending_entry       = (False, None, None)
    peak_cooldown_days  = 0   # counts down after a peaking exit
    etf_position        = None  # current ETF holding (or None)
    # DCA tranche state: list of (deploy_date, fraction_of_total_cash)
    dca_tranches        = []    # remaining tranches to deploy
    dca_total_cash      = 0.0   # cash reserved for all tranches at signal time
    # Ichimoku deferred entries: stocks waiting for Kijun pullback
    ichi_deferred       = []    # list of deferred stock dicts
    ichi_reserved_cash  = 0.0   # cash held for deferred buys (out of settled_cash)
    capital_reset_done  = False

    all_trades    = []
    daily_records = []

    for dt in all_dates:
        # ── Capital reset ──────────────────────────────────────────
        if not capital_reset_done and dt >= cutoff:
            open_positions     = []
            settled_cash       = float(CAPITAL_VND)
            sq                 = SettlementQueue()
            held_sector        = None
            held_state         = None
            days_held          = 0
            pending_exit       = (False, None, None)
            pending_entry      = (False, None, None)
            peak_cooldown_days = 0
            etf_position       = None
            dca_tranches       = []
            dca_total_cash     = 0.0
            ichi_deferred      = []
            ichi_reserved_cash = 0.0
            capital_reset_done = True
            print(f"  Capital reset to {CAPITAL_VND/1e6:.0f}M VND at {dt.date()}")

        released      = sq.release(dt)
        settled_cash += released

        g_drown  = gb.loc[dt, "g_drown"] if dt in gb.index else 0
        is_crash = g_drown >= CRASH_THRESHOLD
        vn_ret   = float(vn.loc[dt, "vn_ret"] if dt in vn.index else 0)
        rows     = {s: sigs[s].loc[dt] if dt in sigs[s].index else None
                    for s in SECTORS}

        # ── Execute pending SELL ───────────────────────────────────
        if pending_exit[0] and open_positions:
            reason = pending_exit[1]
            trades, proceeds = sell_all(
                open_positions, dt, stock_data, reason, held_state)
            all_trades.extend(trades)
            sq.add(proceeds, dt, all_dates)
            if dt >= cutoff:
                print(f"  SELL {held_sector:<22} | {dt.date()} "
                      f"| {proceeds/1e6:.1f}M → T+3 | {reason}")
            open_positions = []
            held_sector    = None
            held_state     = None
            days_held      = 0
            pending_exit   = (False, None, None)
            # Cancel remaining DCA tranches — position is exited
            if dca_tranches:
                refund = sum(dca_total_cash * t["frac"] for t in dca_tranches)
                settled_cash  += refund
                dca_tranches   = []
                dca_total_cash = 0.0
            # Return cash reserved for Ichimoku deferred entries
            if ichi_deferred:
                settled_cash      += ichi_reserved_cash
                ichi_deferred      = []
                ichi_reserved_cash = 0.0

        # ── Execute pending BUY (tranche 1) ───────────────────────
        if pending_entry[0] and not open_positions and not is_crash:
            if settled_cash > 0:
                sec         = pending_entry[1]
                signal_date = pending_entry[2]
                row         = rows.get(sec)
                sig_state   = row["state"] if row is not None else "?"
                # Tag demand-early entries so they show in trade log
                if (DEMAND_EARLY_ENABLED and sig_state == "DROWNING"
                        and row is not None
                        and float(row["spread"]) > DEMAND_HEAT_SPREAD_FLOOR):
                    sig_state = "DEMAND_EARLY"
                score       = float(row["score"]) if row is not None and \
                              not np.isnan(row["score"]) else 0.0

                # Determine tranche schedule based on DCA_MODE
                if DCA_MODE == "NONE":
                    # Deploy 100% immediately
                    fracs   = [1.0]
                    offsets = [0]          # trading days after today
                elif DCA_MODE == "TRANCHE2":
                    fracs   = [0.50, 0.50]
                    offsets = [0, 3]
                elif DCA_MODE == "CONDITIONAL":
                    fracs   = [0.50, 0.50]
                    offsets = [0, 3]       # tranche 2 conditional on spread check
                elif DCA_MODE == "DIP":
                    # T1 immediate, T2 on next dip (deadline = T1 + DIP_MAX_WAIT days)
                    fracs   = [0.50, 0.50]
                    offsets = [0, DIP_MAX_WAIT]  # deadline for T2, not a fixed trigger
                elif DCA_MODE == "SCORE":
                    if score >= DCA_SCORE_HIGH:
                        fracs, offsets = [1.0],                [0]
                    elif score >= DCA_SCORE_MID:
                        fracs, offsets = [0.67, 0.33],         [0, 3]
                    else:
                        fracs, offsets = [0.33, 0.33, 0.34],   [0, 3, 6]
                else:
                    fracs, offsets = [1.0], [0]

                # ── VN-Index market breadth gate ─────────────────────
                # Reduce or skip entry when VN-Index 63d momentum is deeply negative.
                # "80% of stock returns follow VN-Index" — sustained bear = headwind.
                vn_mult = _vn_gate_multiplier(vn, dt)
                if vn_mult == 0.0:
                    # Skip this entry entirely — return reserved cash
                    pending_entry = (False, None, None)
                    if dt >= cutoff:
                        vn_mom_now = _vn_momentum(vn, dt)
                        print(f"  SKIP {sec:<22} | {dt.date()} "
                              f"| VN-gate HARD (VN 63d={vn_mom_now:+.1%})")
                    continue   # back to main date loop

                # Reserve total cash for all tranches at signal time
                total_reserved = round(settled_cash * vn_mult)
                settled_cash  -= total_reserved
                if vn_mult < 1.0 and dt >= cutoff:
                    vn_mom_now = _vn_momentum(vn, dt)
                    print(f"  SOFT {sec:<22} | {dt.date()} "
                          f"| VN-gate SOFT (VN 63d={vn_mom_now:+.1%}) → {vn_mult:.0%} size")

                # Build tranche schedule: list of (trading_day_offset, fraction)
                # Convert trading-day offsets to calendar dates
                td_list    = sorted(all_dates)
                try:
                    today_idx = td_list.index(dt)
                except ValueError:
                    today_idx = next(i for i,d in enumerate(td_list) if d>=dt)

                dca_tranches   = []
                dca_total_cash = total_reserved
                for frac, off in zip(fracs, offsets):
                    dep_idx  = min(today_idx + off, len(td_list)-1)
                    dep_date = td_list[dep_idx]
                    dca_tranches.append({
                        "date":     dep_date,
                        "frac":     frac,
                        "sector":   sec,
                        "sig_date": signal_date,
                        "state":    sig_state,
                        "score":    score,
                    })

                pending_entry = (False, None, None)

                # Deploy tranche 1 immediately (offset=0)
                t1 = dca_tranches.pop(0)
                cash1     = round(dca_total_cash * t1["frac"])
                # DEMAND_EARLY: apply momentum cap to skip stocks already run
                demand_mom_cap = (DEMAND_EARLY_MAX_MOM_PCT
                                  if sig_state == "DEMAND_EARLY"
                                     and DEMAND_EARLY_MAX_MOM_PCT is not None
                                  else None)
                positions, spent, deferred = buy_sector(
                    sec, signal_date, dt, tcks[sec], stock_data, cash1,
                    kijun_filter=True, max_mom_pct=demand_mom_cap,
                    order_data=_order_data)
                if positions or deferred:
                    open_positions     = positions
                    held_sector        = sec
                    held_state         = sig_state
                    days_held          = 0
                    # Reserve cash for deferred (Ichimoku) stocks
                    deferred_alloc     = sum(d["alloc"] for d in deferred)
                    ichi_deferred      = deferred
                    ichi_reserved_cash = deferred_alloc
                    for d in ichi_deferred:
                        d["entry_state"] = sig_state
                    # Return truly unspent cash (not reserved for deferred)
                    settled_cash += (cash1 - spent - deferred_alloc)
                    if dt >= cutoff:
                        pct = t1["frac"]*100
                        defer_str = (f" | {len(deferred)} deferred(Kijun)"
                                     if deferred else "")
                        print(f"  BUY  {sec:<22} | {dt.date()} "
                              f"| T1({pct:.0f}%) state={sig_state} "
                              f"| {len(positions)} stocks "
                              f"| {spent/1e6:.1f}M"
                              + (f" | {len(dca_tranches)} more tranches"
                                 if dca_tranches else "")
                              + defer_str)
                else:
                    # Buy failed — return all reserved cash
                    settled_cash += total_reserved
                    dca_tranches   = []
                    dca_total_cash = 0.0

        # ── Execute pending DCA tranches ───────────────────────────
        if dca_tranches and open_positions and not pending_exit[0] and not is_crash:
            next_t  = dca_tranches[0]
            sec     = next_t["sector"]
            t_frac  = next_t["frac"]
            cash_t  = round(dca_total_cash * t_frac)
            row_now = rows.get(sec)

            # ── Determine whether to fire T2 today ────────────────
            fire_tranche = False
            skip_tranche = False

            if DCA_MODE == "DIP":
                spread_now  = float(row_now["spread"]) if row_now is not None else -99
                s_ret_today = float(row_now["s_ret"])  if row_now is not None else 0.0

                if spread_now < 0:
                    # Sector reversed — abort T2
                    skip_tranche = True
                    if dt >= cutoff:
                        print(f"  DCA  {sec:<22} | {dt.date()} "
                              f"| T2 SKIPPED (spread={spread_now:.1f} < 0 — signal faded)")
                elif s_ret_today <= DIP_THRESHOLD:
                    # Sector had a down day — this is our dip, buy now
                    fire_tranche = True
                    if dt >= cutoff:
                        print(f"  DCA  {sec:<22} | {dt.date()} "
                              f"| T2 DIP  (sector ret={s_ret_today*100:+.2f}%"
                              f" <= {DIP_THRESHOLD*100:.1f}% threshold)")
                elif dt >= next_t["date"]:
                    # Deadline reached — no dip came, buy at market
                    fire_tranche = True
                    if dt >= cutoff:
                        print(f"  DCA  {sec:<22} | {dt.date()} "
                              f"| T2 MARKET (deadline reached — no dip in {DIP_MAX_WAIT}d)")

            elif DCA_MODE == "CONDITIONAL":
                if dt >= next_t["date"]:
                    if row_now is None or float(row_now["spread"]) < 0:
                        skip_tranche = True
                        if dt >= cutoff:
                            sv = float(row_now["spread"]) if row_now is not None else 0
                            print(f"  DCA  {sec:<22} | {dt.date()} "
                                  f"| T{len(dca_tranches)+1} SKIPPED "
                                  f"(spread={sv:.1f} < 0)")
                    else:
                        fire_tranche = True

            else:
                # TRANCHE2 / SCORE: fire on scheduled date
                if dt >= next_t["date"]:
                    fire_tranche = True

            if fire_tranche or skip_tranche:
                dca_tranches.pop(0)

            if fire_tranche and cash_t > 0:
                # Add to existing positions (top-up) — no Kijun filter on T2
                new_pos, spent, _ = buy_sector(
                    sec, next_t["sig_date"], dt,
                    tcks[sec], stock_data, cash_t,
                    kijun_filter=False, order_data=_order_data)
                if new_pos:
                    open_positions.extend(new_pos)
                    settled_cash += (cash_t - spent)
                    if dt >= cutoff:
                        tnum = len(dca_tranches) + 2
                        print(f"  DCA  {sec:<22} | {dt.date()} "
                              f"| T{tnum}({t_frac*100:.0f}%) "
                              f"| {len(new_pos)} stocks "
                              f"| {spent/1e6:.1f}M")
                else:
                    settled_cash += cash_t  # return if no stocks found
            elif skip_tranche:
                settled_cash += cash_t  # return skipped cash

            # Clear DCA if sector exited or no more tranches
            if not dca_tranches:
                dca_total_cash = 0.0

        # ── Ichimoku deferred entry check ──────────────────────────
        # Each day, re-check deferred stocks.  Buy when:
        #   (a) price has pulled back within KIJUN_BUY_THRESHOLD of Kijun, OR
        #   (b) ENTRY_MAX_WAIT_DAYS trading days have passed (force-buy)
        if ichi_deferred and open_positions and not pending_exit[0] and not is_crash:
            td_list    = sorted(all_dates)
            still_wait = []
            for item in ichi_deferred:
                ticker = item["ticker"]
                result = get_open_price(ticker, dt, stock_data)
                if result is None:
                    still_wait.append(item)
                    continue
                price, actual_date = result
                # Count trading days since deferral
                defer_idx   = next((i for i,d in enumerate(td_list)
                                    if d >= item["deferred_on"]), 0)
                today_idx   = next((i for i,d in enumerate(td_list)
                                    if d >= dt), 0)
                days_waited = today_idx - defer_idx

                # Re-compute Kijun with today's data for freshness
                kijun       = compute_kijun(ticker, dt, stock_data)
                kijun_ready = (kijun is None or
                               (price / kijun - 1) <= KIJUN_BUY_THRESHOLD)
                force_buy   = days_waited >= ENTRY_MAX_WAIT_DAYS

                if kijun_ready or force_buy:
                    alloc     = item["alloc"]
                    price_vnd = price * 1000
                    shares    = int(alloc / (price_vnd * (1 + FRICTION)))
                    if shares > 0:
                        spent = shares * price_vnd * (1 + FRICTION)
                        open_positions.append({
                            "ticker":      ticker,
                            "sector":      item["sector"],
                            "signal_date": item["signal_date"],
                            "entry_date":  actual_date,
                            "entry_price": price,
                            "peak_price":  price,
                            "shares":      shares,
                            "cost":        spent,
                        })
                        ichi_reserved_cash -= alloc
                        settled_cash       += (alloc - spent)
                        if dt >= cutoff:
                            reason = ("kijun_ready" if kijun_ready
                                      else f"max_wait({days_waited}d)")
                            kstr   = f"{kijun:.2f}" if kijun else "n/a"
                            print(f"  ICHI {ticker:<8} {item['sector']:<20}"
                                  f"| {dt.date()} | {reason}"
                                  f" | px={price:.2f} kijun={kstr}")
                    else:
                        # Can't buy minimum lot — return the cash
                        ichi_reserved_cash -= alloc
                        settled_cash       += alloc
                else:
                    still_wait.append(item)
            ichi_deferred = still_wait

        # ── Mark to market ─────────────────────────────────────────
        pos_val = 0.0
        for pos in open_positions:
            tk = pos["ticker"]
            if tk in stock_data and dt in stock_data[tk].index:
                pos_val += pos["shares"] * stock_data[tk].loc[dt, "close"] * 1000
            else:
                pos_val += pos["cost"]

        # ETF mark-to-market
        etf_val = 0.0
        if etf_position is not None:
            tk = ETF_TICKER
            if tk in stock_data and dt in stock_data[tk].index:
                etf_val = etf_position["shares"] * stock_data[tk].loc[dt, "close"] * 1000
            else:
                etf_val = etf_position["cost"]

        # DCA reserved cash: money set aside for future tranches
        # counts as portfolio value — it's not lost, just not yet deployed
        dca_reserved = sum(dca_total_cash * t["frac"] for t in dca_tranches) \
                       if dca_tranches else 0.0

        port_value = settled_cash + sq.pending_total() + pos_val + etf_val + dca_reserved

        # ── Stock-level trailing stop / TP / SL ─────────────────────
        # Check each open position daily. Updates peak prices and exits
        # any stock that has triggered its trailing stop, fixed TP, or SL.
        # This is a partial exit: only triggered stocks are sold, rest stays.
        any_exit_active = (STOCK_TRAILING_STOP_PCT is not None
                           or STOCK_TP_PCT is not None
                           or STOCK_SL_PCT is not None
                           or LAGGARD_FLAT_THRESH is not None
                           or LAGGARD_LOSS_THRESH is not None)
        if open_positions and any_exit_active and not pending_exit[0]:
            tp_trades, open_positions, tp_proceeds = sell_tp_stocks(
                open_positions, dt, stock_data)
            if tp_trades:
                settle_date = dt + pd.tseries.offsets.BDay(SETTLEMENT_DAYS)
                sq.add(tp_proceeds, dt, all_dates)
                all_trades.extend(tp_trades)
                n_tp = len(tp_trades)
                avg_gain = np.mean([t["pnl_pct"] for t in tp_trades])
                if dt >= cutoff:
                    print(f"  EXIT {held_sector:<22} | {dt.date()} "
                          f"| {n_tp} stocks exited "
                          f"| avg {avg_gain:+.1f}% "
                          f"| {tp_proceeds/1e6:.1f}M → T+3")
                # If ALL positions taken profit → no more held sector
                if not open_positions:
                    held_sector = None
                    held_state  = None
                    days_held   = 0

        # ── ETF overlay — buy when idle, sell when sector fires ─────
        if ETF_TICKER is not None and dt >= cutoff:
            sector_active = bool(open_positions) or pending_entry[0]

            if not sector_active and not is_crash:
                # No sector position → buy ETF with idle cash if not already held
                if etf_position is None and settled_cash > 0:
                    etf_pos, spent = etf_buy(settled_cash, dt, stock_data)
                    if etf_pos is not None:
                        settled_cash -= spent
                        etf_position  = etf_pos
                        if dt >= cutoff:
                            print(f"  ETF BUY  {ETF_TICKER:<12} | {dt.date()} "
                                  f"| {spent/1e6:.1f}M ({ETF_ALLOC*100:.0f}% of cash)")
            else:
                # Sector is active or about to be — sell ETF if held
                if etf_position is not None:
                    proceeds, etf_trade = etf_sell(etf_position, dt, stock_data, all_dates)
                    sq.add(proceeds, dt, all_dates)
                    if etf_trade:
                        all_trades.append(etf_trade)
                    if dt >= cutoff:
                        gain = (proceeds - etf_position["cost"]) / etf_position["cost"] * 100
                        print(f"  ETF SELL {ETF_TICKER:<12} | {dt.date()} "
                              f"| {proceeds/1e6:.1f}M → T+3 | {gain:+.1f}%")
                    etf_position = None

        # ── Daily regime lookup ───────────────────────────────────
        # Pre-computed series → O(1) per day.  Only PANIC blocks entries.
        if _regime_series is not None:
            _row = (_regime_series.loc[dt]
                    if dt in _regime_series.index else None)
            prev_panic = _regime_is_panic
            _regime_is_panic = bool(_row["is_panic"]) if _row is not None else False
            if dt >= cutoff and _regime_is_panic != prev_panic:
                lbl = _row["label"] if _row is not None else "?"
                scr = _row["score"] if _row is not None else 0
                print(f"  REGIME {dt.date()} {lbl:<10} score={scr:.0f}/100"
                      + ("  ⚫ PANIC — entries blocked" if _regime_is_panic
                         else "  ✓ entries resumed"))

        # ── Signal logic ───────────────────────────────────────────
        # Tick down cooldown each day
        if peak_cooldown_days > 0:
            peak_cooldown_days -= 1

        in_cooldown = (peak_cooldown_days > 0)

        if is_crash:
            if open_positions and not pending_exit[0]:
                pending_exit  = (True, "crash", dt)
                pending_entry = (False, None, None)
            # Also liquidate ETF on crash
            if etf_position is not None:
                proceeds, etf_trade = etf_sell(etf_position, dt, stock_data, all_dates)
                sq.add(proceeds, dt, all_dates)
                if etf_trade:
                    etf_trade["exit_reason"] = "crash"
                    all_trades.append(etf_trade)
                etf_position = None

        elif held_sector:
            row = rows.get(held_sector)
            if row is not None and days_held >= MIN_HOLD_DAYS:
                should_exit, reason = wants_exit(row)
                # ── VN-Index exit accelerator ─────────────────────────────────
                # When holding a position and VN-Index drops sharply, exit faster
                # than sector spread/momentum signals alone would trigger.
                #   Flash     (5d):  -7%  → market crash unfolding, exit now
                #   Sustained (20d): -12% → prolonged bear, cut losses
                if not should_exit and VNINDEX_EXIT_ENABLED:
                    _vn5  = _vn_momentum(vn, dt, window=VNINDEX_EXIT_FLASH_DAYS)
                    _vn20 = _vn_momentum(vn, dt, window=VNINDEX_EXIT_SUSTAINED_DAYS)
                    if _vn5 < VNINDEX_EXIT_FLASH_THRESH:
                        should_exit = True
                        reason      = f"vn_flash({_vn5:+.1%})"
                    elif _vn20 < VNINDEX_EXIT_SUSTAINED_THRESH:
                        should_exit = True
                        reason      = f"vn_sustained({_vn20:+.1%})"
                if should_exit:
                    pending_exit  = (True, reason, dt)
                    pending_entry = (False, None, None)
                    # Start cooldown if this was a peaking exit
                    if reason == "peaking":
                        peak_cooldown_days = COOLDOWN_AFTER_PEAK
                        if dt >= cutoff:
                            print(f"  PEAK-COOLDOWN {COOLDOWN_AFTER_PEAK}d starts "
                                  f"{dt.date()} — RECOVERY entries blocked")
                else:
                    days_held += 1
            else:
                days_held += 1

        elif not open_positions and not _regime_is_panic:
            candidates = {
                sec: entry_score(rows[sec])
                for sec in SECTORS
                if rows.get(sec) is not None
                and wants_entry(rows[sec], sec, in_cooldown)
            }
            # ── Demand-early entry (stock-flow based) ──────────────
            # If no baseline/leading signal, check sector heat:
            # heavy full-candle activity in a DROWNING sector fires
            # earlier than EARLY_V3 (no velocity requirement needed).
            if not candidates and DEMAND_EARLY_ENABLED and not in_cooldown:
                for sec in SECTORS:
                    row = rows.get(sec)
                    if row is None:
                        continue
                    sp  = float(row["spread"])
                    if row["state"] != "DROWNING":
                        continue
                    if sp < DEMAND_HEAT_SPREAD_FLOOR:
                        continue
                    heat = _sector_heat(sec, dt, stock_data, ticker_sector_map,
                                        window=DEMAND_HEAT_WINDOW)
                    # Adaptive threshold: rolling Nth-percentile of sector's
                    # own heat history.  Falls back to fixed when insufficient data.
                    if DEMAND_HEAT_USE_ADAPTIVE and sec in heat_ts:
                        past = heat_ts[sec].loc[:dt].tail(252).dropna()
                        if len(past) >= DEMAND_HEAT_MIN_HISTORY:
                            thresh = float(past.quantile(DEMAND_HEAT_PERCENTILE / 100))
                        else:
                            thresh = DEMAND_HEAT_THRESHOLD
                    else:
                        thresh = DEMAND_HEAT_THRESHOLD
                    if heat >= thresh:
                        # Score: use heat directly (vs spread score for baseline)
                        candidates[sec] = heat
            # ── FG confirmation filter ─────────────────────────────
            if _use_fg_confirm and _fg_streaks and candidates:
                filtered = {}
                for sec, score in candidates.items():
                    if sec not in FG_CONFIRM_SECTORS:
                        filtered[sec] = score
                        continue
                    streak = _fg_streak_on_date(sec, dt, _fg_streaks)
                    if streak >= FG_CONFIRM_STREAK:
                        filtered[sec] = score
                    elif dt >= cutoff:
                        print(f"  FG-SKIP {sec:<22} | {dt.date()} | streak={streak} < {FG_CONFIRM_STREAK}")
                candidates = filtered

            if candidates:
                ranked   = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
                best_sec = ranked[0][0]
                pending_entry = (True, best_sec, dt)
            else:
                pending_entry = (False, None, None)
        # ── Record ─────────────────────────────────────────────────
        if dt >= cutoff:
            rec = {
                "date":         dt,
                "port_value":   port_value,
                "settled_cash": settled_cash,
                "pending_cash": sq.pending_total(),
                "pos_value":    pos_val,
                "active":       held_sector or "CASH",
                "is_crash":     is_crash,
                "vn_ret":       vn_ret,
                "n_pos":        len(open_positions),
            }
            for sec in SECTORS:
                key = f"in_{sec.replace(' ','_').replace('&','and')[:12]}"
                rec[key] = (held_sector == sec)
            daily_records.append(rec)

    # Close any open positions at end of data
    last_dt = all_dates[-1]
    if open_positions:
        trades, _ = sell_all(
            open_positions, last_dt, stock_data, "end_of_data", held_state)
        all_trades.extend(trades)

    df_d = pd.DataFrame(daily_records).set_index("date")
    df_d["ret"]    = df_d["port_value"].pct_change().fillna(0)
    df_d["cum"]    = df_d["port_value"] / df_d["port_value"].iloc[0]
    df_d["cum_vn"] = (1 + df_d["vn_ret"]).cumprod()

    df_t = pd.DataFrame(all_trades)
    if not df_t.empty:
        df_t = df_t[df_t["entry_date"] >= cutoff]

    return df_d, df_t, sigs, SECTORS


# ─────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────

def metrics(ret):
    r    = ret.replace([np.inf,-np.inf], 0).dropna()
    cum  = (1 + r).cumprod()
    tot  = cum.iloc[-1] - 1
    n    = len(r) / 252
    ann  = (1 + tot) ** (1 / max(n, 0.1)) - 1
    sh   = (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0
    mdd  = ((cum / cum.cummax()) - 1).min()
    return tot*100, ann*100, sh, mdd*100


def print_report(df_d, df_t, sectors):
    tot, ann, sh, mdd = metrics(df_d["ret"])
    inv = (df_d["active"] != "CASH").mean() * 100
    start = df_d["port_value"].iloc[0]
    end   = df_d["port_value"].iloc[-1]

    print(f"\n{'='*72}")
    print(f"  BEST-OF-{len(sectors)} — vol-adjusted threshold  "
          f"(K×rolling_std per sector)")
    print(f"  Sectors: {' | '.join(sectors)}")
    print(f"  Entry: RECOVERY or LEADING(score≥{MIN_ENTRY_SCORE})  |  "
          f"T+3  |  T+1 exec  |  {MIN_LIQUIDITY_VND/1e6:.0f}M VND liquidity")
    print(f"{'='*72}")
    print(f"  Start capital    : {start/1e6:>10.2f}M VND")
    print(f"  End value        : {end/1e6:>10.2f}M VND  ({end/1e6 - start/1e6:+.1f}M)")
    print(f"  Total return     : {tot:>10.2f}%")
    print(f"  Annualised       : {ann:>10.2f}%")
    print(f"  Sharpe           : {sh:>10.3f}")
    print(f"  Max drawdown     : {mdd:>10.2f}%")
    print(f"  Time invested    : {inv:>10.1f}%")
    print(f"{'='*72}")

    if df_t.empty:
        print("  No trades recorded.")
        return

    wins = (df_t["pnl_pct"] > 0).sum()
    n    = len(df_t)
    print(f"\n  TRADE SUMMARY  ({n} stock-trades)")
    print(f"  Win rate   : {wins/n*100:.1f}%  ({wins}W / {n-wins}L)")
    print(f"  Avg winner : +{df_t[df_t['pnl_pct']>0]['pnl_pct'].mean():.2f}%")
    print(f"  Avg loser  :  {df_t[df_t['pnl_pct']<=0]['pnl_pct'].mean():.2f}%")
    print(f"  Total PnL  : {df_t['pnl_vnd'].sum()/1e6:+.2f}M VND")

    # Rounds — group by sector + signal_date, use FIRST exit date
    print(f"\n  SIGNAL ROUNDS")
    rounds = (df_t.groupby(["sector","signal_date"])
                  .agg(n_stocks=("ticker","count"),
                       entry_date=("entry_date","min"),
                       exit_date=("exit_date","first"),   # first stock exit = round exit
                       pnl_vnd=("pnl_vnd","sum"),
                       pnl_pct=("pnl_pct","mean"),
                       entry_state=("entry_state","first"),
                       exit_reason=("exit_reason","first"))
                  .reset_index().sort_values("entry_date"))

    print(f"  {'Sector':<22} {'Entry':10} {'Exit':10} {'Days':>5} "
          f"{'#':>3} {'PnL VND':>10} {'Avg%':>7}  State")
    print(f"  {'-'*80}")
    for _, r in rounds.iterrows():
        days = (r["exit_date"] - r["entry_date"]).days
        flag = "✓" if r["pnl_vnd"] > 0 else "✗"
        print(f"  {r['sector']:<22} "
              f"{str(r['entry_date'].date()):10} "
              f"{str(r['exit_date'].date()):10} "
              f"{days:>5} {int(r['n_stocks']):>3} "
              f"{r['pnl_vnd']/1e6:>8.2f}M "
              f"{flag}{r['pnl_pct']:>+6.2f}%  "
              f"{r['entry_state']}")

    # Per-sector
    print(f"\n  PER-SECTOR")
    for sec in sectors:
        st = df_t[df_t["sector"] == sec]
        if st.empty:
            print(f"  {sec:<22}: no trades")
            continue
        sw = (st["pnl_pct"] > 0).sum()
        print(f"  {sec:<22}: {len(st):>3} trades  "
              f"Win {sw/len(st)*100:.0f}%  "
              f"PnL {st['pnl_vnd'].sum()/1e6:+.1f}M  "
              f"Avg {st['pnl_pct'].mean():+.2f}%")

    # Annual breakdown
    print(f"\n  ANNUAL BREAKDOWN")
    print(f"  {'Year':<6} {'Return':>8} {'Sharpe':>7} {'MDD':>8}  "
          f"{'vs VN':>8}  Invested  Active sectors")
    print(f"  {'-'*72}")
    for yr in sorted(df_d.index.year.unique()):
        yr_d = df_d[df_d.index.year == yr]
        if len(yr_d) < 5:
            continue
        r    = yr_d["ret"].replace([np.inf,-np.inf], 0)
        cum  = (1 + r).cumprod()
        tot  = (cum.iloc[-1] - 1) * 100
        sh   = (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0
        mdd  = ((cum / cum.cummax()) - 1).min() * 100
        vn_r = (((1 + yr_d["vn_ret"]).cumprod().iloc[-1]) - 1) * 100
        inv  = (yr_d["active"] != "CASH").mean() * 100
        flag = "✓" if tot > vn_r else "✗"
        # Which sectors were active this year
        active_secs = [s for s in sectors
                       if yr_d["active"].str.contains(s.split()[0], na=False).any()]
        secs_str = "+".join(s.split()[0] for s in active_secs) if active_secs else "—"
        print(f"  {yr:<6} {tot:>+7.1f}%  {sh:>6.2f}  {mdd:>7.1f}%  "
              f"  {flag}{vn_r:>+7.1f}%  {inv:>5.0f}%   {secs_str}")
    print()


# ─────────────────────────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────────────────────────

# Colour palette — cycles through for any number of sectors
import matplotlib.cm as _cm
def make_colors(sectors):
    fixed = {
        "Banks":           "#2980b9",
        "Food & Beverage": "#27ae60",
        "Real Estate":     "#e67e22",
        "Oil & Gas":       "#8e44ad",
        "Chemicals":       "#c0392b",
        "Construction & Materials": "#16a085",
        "Financial Services":       "#d35400",
        "Basic Resources":          "#7f8c8d",
        "Industrial Goods & Services": "#2c3e50",
        "Utilities":       "#27ae60",
    }
    palette = ["#2980b9","#27ae60","#e67e22","#8e44ad","#c0392b",
               "#16a085","#d35400","#7f8c8d","#2c3e50","#e74c3c",
               "#f39c12","#1abc9c","#34495e","#e91e63"]
    colors = {"CASH": "#cccccc"}
    for i, sec in enumerate(sectors):
        colors[sec] = fixed.get(sec, palette[i % len(palette)])
    return colors


def plot(df_d, df_t, sigs, sectors):
    COLORS = make_colors(sectors)
    fig = plt.figure(figsize=(20, 24), facecolor="white")
    gs  = gridspec.GridSpec(5, 2, figure=fig,
                            height_ratios=[2.2, 1.4, 1.2, 1.0, 0.7],
                            hspace=0.45, wspace=0.3)
    fig.subplots_adjust(top=0.93, bottom=0.04, left=0.08, right=0.96)

    ax_eq     = fig.add_subplot(gs[0, :])
    ax_spread = fig.add_subplot(gs[1, :])   # NEW: sector spread lines
    ax_rnd    = fig.add_subplot(gs[2, 0])
    ax_sec    = fig.add_subplot(gs[2, 1])
    ax_cash   = fig.add_subplot(gs[3, :])
    ax_pos    = fig.add_subplot(gs[4, :])

    TC = "#111111"
    for ax in [ax_eq, ax_spread, ax_rnd, ax_sec, ax_cash, ax_pos]:
        ax.set_facecolor("#f9f9f9")
        ax.tick_params(colors=TC, labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#aaaaaa")
        ax.grid(True, linestyle=":", alpha=0.4, color="#cccccc")

    # ── Year bands ────────────────────────────────────────────────
    yr_min = df_d.index.year.min()
    yr_max = df_d.index.year.max() + 1
    for ax in [ax_eq, ax_spread, ax_pos]:
        for yr in range(yr_min, yr_max + 1):
            if yr % 2 == 0:
                ax.axvspan(pd.Timestamp(f"{yr}-01-01"),
                           pd.Timestamp(f"{yr+1}-01-01"),
                           color="#ebebeb", alpha=0.6, zorder=0)

    # ── Equity curve ─────────────────────────────────────────────
    vals_m = df_d["port_value"] / 1e6
    vn_m   = df_d["cum_vn"] * (df_d["port_value"].iloc[0] / 1e6)

    ax_eq.plot(df_d.index, vals_m, color="#1a1a2e", lw=2.2,
               label=f"Strategy (start {CAPITAL_VND/1e6:.0f}M VND)", zorder=3)
    ax_eq.plot(df_d.index, vn_m, color="#e74c3c", lw=1.5,
               linestyle=":", alpha=0.85, label="VN-Index", zorder=3)

    base = vals_m.min()
    for sec in sectors:
        flag_col = f"in_{sec.replace(' ','_').replace('&','and')[:12]}"
        where = df_d[flag_col] if flag_col in df_d.columns else \
                df_d["active"].str.contains(sec.split()[0], na=False)
        ax_eq.fill_between(df_d.index, base, vals_m,
                           where=where, color=COLORS.get(sec,"#888"),
                           alpha=0.10, label=sec)

    # Per-year return labels
    for yr in range(yr_min, yr_max):
        yr_d = df_d[df_d.index.year == yr]
        if len(yr_d) < 20:
            continue
        ret  = (yr_d["port_value"].iloc[-1] / yr_d["port_value"].iloc[0] - 1) * 100
        mid  = yr_d.index[len(yr_d)//2]
        yval = yr_d["port_value"].iloc[-1] / 1e6
        ax_eq.text(mid, yval * 1.015, f"{ret:+.0f}%",
                   ha="center", fontsize=7,
                   color="#006400" if ret >= 0 else "#cc0000",
                   fontweight="bold", zorder=4)

    ax_eq.set_ylabel("Portfolio Value (M VND)", fontsize=9, color=TC)
    ax_eq.set_title(
        f"Best-of-{len(sectors)} — vol-adjusted threshold  ({yr_min}–{yr_max-1})\n"
        f"K×rolling_std entry filter  |  T+1 open  |  T+3 settlement  |  "
        f"Fee: {FRICTION*100:.2f}% per leg  |  Liq ≥ {MIN_LIQUIDITY_VND/1e6:.0f}M VND/day",
        fontsize=10, fontweight="bold", color=TC, pad=12)
    ax_eq.legend(loc="upper left", fontsize=6, facecolor="white",
                 labelcolor=TC, edgecolor="#aaa", framealpha=0.9, ncol=3)

    # ── Sector spread lines with recovery-type colouring ─────────
    # Background shading shows what TYPE of recovery just started:
    #   Green  = DROWNING recovery  (genuine trough — this is what we buy)
    #   Red    = CRASH recovery     (post-crash turn — also buyable)
    #   Yellow = SHALLOW recovery   (noise blip — we skip these)
    # The spread line is drawn on top in each sector's colour.
    for sec in sectors:
        sig = sigs.get(sec)
        if sig is None:
            continue
        sig_w = sig[sig.index >= df_d.index[0]]
        if sig_w.empty:
            continue
        spread = sig_w["spread"]
        col    = COLORS.get(sec, "#888")

        ax_spread.plot(spread.index, spread.values, color=col,
                       lw=1.2, alpha=0.75, label=sec)

        prev_t        = sig_w["prev_trough"]
        recovery_mask = sig_w["state"] == "RECOVERY"
        ax_spread.fill_between(spread.index, spread.min()*1.05, spread.max()*1.05,
                               where=(recovery_mask & (prev_t == "DROWNING")),
                               color="#27ae60", alpha=0.10, zorder=0)
        ax_spread.fill_between(spread.index, spread.min()*1.05, spread.max()*1.05,
                               where=(recovery_mask & (prev_t == "CRASH")),
                               color="#e74c3c", alpha=0.10, zorder=0)
        ax_spread.fill_between(spread.index, spread.min()*1.05, spread.max()*1.05,
                               where=(recovery_mask & (prev_t == "SHALLOW")),
                               color="#f1c40f", alpha=0.12, zorder=0)

    ax_spread.axhline(0,           color="#555",    lw=0.8, ls="--", alpha=0.6)
    ax_spread.axhline(SPREAD_EXIT, color="#e74c3c", lw=0.8, ls=":", alpha=0.7)
    if SPREAD_HIGH_EXIT:
        ax_spread.axhline(SPREAD_HIGH_EXIT, color="#e67e22", lw=0.7, ls=":", alpha=0.6)

    if not df_t.empty:
        rounds = (df_t.groupby(["sector","signal_date"])
                      .agg(entry_date=("entry_date","min"),
                           exit_date=("exit_date","max"))
                      .reset_index())
        for _, r in rounds.iterrows():
            col = COLORS.get(r["sector"], "#888")
            ax_spread.axvline(r["entry_date"], color=col, lw=0.6, alpha=0.4)
            ax_spread.axvline(r["exit_date"],  color=col, lw=0.6, ls="--", alpha=0.3)

    ax_spread.set_ylabel("Spread %", fontsize=9, color=TC)
    ax_spread.set_title("Sector breadth spreads — solid=entry  dashed=exit",
                        fontsize=8, color=TC)
    import matplotlib.patches as mpatches
    sec_handles  = [plt.Line2D([0],[0], color=COLORS.get(s,"#888"), lw=1.5, label=s)
                    for s in sectors]
    type_handles = [
        mpatches.Rectangle((0,0),1,1, fc="#27ae60", alpha=0.5, label="Drowning rec"),
        mpatches.Rectangle((0,0),1,1, fc="#e74c3c", alpha=0.5, label="Crash rec"),
        mpatches.Rectangle((0,0),1,1, fc="#f1c40f", alpha=0.6, label="Shallow (skip)"),
    ]
    ax_spread.legend(handles=sec_handles + type_handles,
                     loc="upper right", fontsize=6, facecolor="white",
                     labelcolor=TC, edgecolor="#aaa", framealpha=0.9, ncol=4)

    # ── Round PnL ────────────────────────────────────────────────
    if not df_t.empty:
        rounds = (df_t.groupby(["sector","signal_date"])
                      .agg(pnl_pct=("pnl_pct","mean"))
                      .reset_index().sort_values("signal_date"))
        bar_colors = [COLORS.get(s, "#888") for s in rounds["sector"]]
        edge_c     = ["#006400" if p > 0 else "#cc0000" for p in rounds["pnl_pct"]]
        ax_rnd.bar(range(len(rounds)), rounds["pnl_pct"].values,
                   color=bar_colors, edgecolor=edge_c, alpha=0.8, lw=0.8)
        ax_rnd.axhline(0, color="#555", lw=0.8, linestyle="--")
        ax_rnd.set_xticks([])
        ax_rnd.set_ylabel("Avg PnL % per round", fontsize=8, color=TC)
        ax_rnd.set_title("Round PnL % by sector", fontsize=9, color=TC)

        for sec in sectors:
            st = df_t[df_t["sector"] == sec].sort_values("entry_date")
            if not st.empty:
                ax_sec.plot(range(len(st)), st["pnl_vnd"].cumsum().values / 1e6,
                            color=COLORS.get(sec,"#888"), lw=1.5,
                            marker="o", markersize=2, label=sec.split()[0])
        ax_sec.axhline(0, color="#555", lw=0.8, linestyle="--")
        ax_sec.set_ylabel("Cumulative PnL (M VND)", fontsize=8, color=TC)
        ax_sec.set_title("Cumulative PnL by sector", fontsize=9, color=TC)
        ax_sec.legend(fontsize=7, facecolor="white", labelcolor=TC,
                      edgecolor="#aaa", ncol=2)

    # ── Cash breakdown ────────────────────────────────────────────
    ax_cash.stackplot(
        df_d.index,
        df_d["settled_cash"] / 1e6,
        df_d["pending_cash"] / 1e6,
        labels=["Settled cash (spendable)", "Pending T+3"],
        colors=["#2ecc71", "#f39c12"], alpha=0.7)
    ax_cash.set_ylabel("Cash (M VND)", fontsize=8, color=TC)
    ax_cash.set_title("Cash — settled vs pending T+3", fontsize=9, color=TC)
    ax_cash.legend(fontsize=8, facecolor="white", labelcolor=TC,
                   edgecolor="#aaa", loc="upper right")

    # ── Position timeline — one row per slot ──────────────────────
    n = len(sectors)
    for i, sec in enumerate(sectors):
        y0       = i / n
        y1       = (i + 1) / n
        flag_col = f"in_{sec.replace(' ','_').replace('&','and')[:12]}"
        where    = df_d[flag_col] if flag_col in df_d.columns else \
                   df_d["active"].str.contains(sec.split()[0], na=False)
        ax_pos.fill_between(df_d.index, y0, y1, where=where,
                            color=COLORS.get(sec,"#888"), alpha=0.8, label=sec.split()[0])
        ax_pos.fill_between(df_d.index, y0, y1, where=~where,
                            color="#dddddd", alpha=0.25)
        ax_pos.axhline(y1, color="#aaa", lw=0.4)
        ax_pos.text(df_d.index[0], (y0+y1)/2, f" {sec.split()[0]}",
                    va="center", fontsize=6, color=TC)

    ax_pos.set_ylim(0, 1); ax_pos.set_yticks([])
    ax_pos.set_ylabel("Slots", fontsize=8, color=TC)
    ax_pos.set_title(f"Holdings — single position across {n} sectors",
                     fontsize=8, color=TC)

    fname = "option_c_momentum_rank.png"
    for path in [fname, f"/mnt/user-data/outputs/{fname}"]:
        try:
            plt.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
        except Exception: pass
    print(f"  Chart saved → {fname}")
    plt.show()


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def safe(s): return s.replace(" ","_").replace("&","and")

def run_walk_forward(df_all, gb, vn, stock_data):
    """
    Expanding-window walk-forward validation.

    Structure:
      IS (in-sample):  2005-01-01 → window_start  (grows each window)
      OOS (test):      window_start → window_end   (2-year windows)

    Parameters are NEVER re-fitted between windows.
    K(t) is computed from rolling data — automatically uses only past data.
    Capital resets to 100M at each OOS window start for fair comparison.

    This tests whether the signal edge is real (holds OOS) or overfitted
    (degrades on unseen data).
    """
    # OOS windows: [start, end]
    oos_windows = [
        ("2016-01-01", "2018-01-01"),
        ("2018-01-01", "2020-01-01"),
        ("2020-01-01", "2022-01-01"),
        ("2022-01-01", "2024-01-01"),
        ("2024-01-01", "2026-01-01"),
    ]

    print(f"\n{'='*72}")
    print(f"  WALK-FORWARD VALIDATION — expanding window")
    print(f"  OOS windows: {len(oos_windows)} × 2-year periods (2016–2026)")
    print(f"  Parameters: FIXED — no re-fitting between windows")
    print(f"  K(t): auto from rolling data — only uses past observations")
    print(f"{'='*72}\n")

    oos_results   = []
    full_oos_rets = []   # concatenated daily returns across all OOS windows

    for i, (start_str, end_str) in enumerate(oos_windows):
        oos_start = pd.Timestamp(start_str)
        oos_end   = pd.Timestamp(end_str)

        print(f"  Window {i+1}/{len(oos_windows)}: OOS {oos_start.date()} → {oos_end.date()}")
        print(f"    IS: 2005-01-01 → {oos_start.date()}  "
              f"({(oos_start - pd.Timestamp('2005-01-01')).days // 365}yr)")

        # Run backtest — signals use full history but records only OOS period
        # Capital resets to 100M at oos_start
        df_d, df_t, sigs, sectors = run_backtest(
            df_all, gb, vn, stock_data, cutoff=oos_start)

        # Trim to OOS window only
        df_oos = df_d[df_d.index < oos_end].copy()
        if len(df_oos) < 20:
            print(f"    ← insufficient data, skipping")
            continue

        df_t_oos = df_t[
            (df_t["entry_date"] >= oos_start) &
            (df_t["entry_date"] < oos_end)
        ] if not df_t.empty else pd.DataFrame()

        tot, ann, sh, mdd = metrics(df_oos["ret"])
        inv = (df_oos["active"] != "CASH").mean() * 100
        n_tr= len(df_t_oos)
        wr  = (df_t_oos["pnl_pct"] > 0).mean() * 100 if n_tr > 0 else 0

        # VN-Index return over same period
        vn_ret_oos = ((1 + df_oos["vn_ret"]).cumprod().iloc[-1] - 1) * 100

        print(f"    Return: {tot:>+7.1f}%  Ann: {ann:>+5.1f}%  "
              f"Sharpe: {sh:>5.2f}  MDD: {mdd:>6.1f}%  "
              f"Invested: {inv:>4.0f}%  Trades: {n_tr}  "
              f"Win: {wr:.0f}%  vs VN: {vn_ret_oos:>+6.1f}%")

        oos_results.append({
            "window":       f"{start_str[:4]}-{end_str[:4]}",
            "oos_start":    oos_start,
            "oos_end":      oos_end,
            "total_ret":    round(tot, 2),
            "ann_ret":      round(ann, 2),
            "sharpe":       round(sh, 3),
            "mdd":          round(mdd, 2),
            "invested_pct": round(inv, 1),
            "n_trades":     n_tr,
            "win_rate":     round(wr, 1),
            "vn_ret":       round(vn_ret_oos, 2),
            "beat_vn":      tot > vn_ret_oos,
            "df_oos":       df_oos,
        })
        full_oos_rets.append(df_oos["ret"])

    if not oos_results:
        print("  No OOS results generated.")
        return

    # Concatenated OOS performance
    if full_oos_rets:
        combined = pd.concat(full_oos_rets).sort_index()
        tot_c, ann_c, sh_c, mdd_c = metrics(combined)
        beat_count = sum(r["beat_vn"] for r in oos_results)

        print(f"\n{'='*72}")
        print(f"  WALK-FORWARD SUMMARY")
        print(f"{'='*72}")
        print(f"  {'Window':<12} {'Return':>8} {'Ann':>6} {'Sharpe':>7} "
              f"{'MDD':>7} {'Win%':>6} {'vs VN':>7}  Verdict")
        print(f"  {'-'*70}")
        for r in oos_results:
            verdict = "✓ BEAT" if r["beat_vn"] else "✗ MISS"
            print(f"  {r['window']:<12} {r['total_ret']:>+7.1f}%  "
                  f"{r['ann_ret']:>+5.1f}%  {r['sharpe']:>6.2f}  "
                  f"{r['mdd']:>6.1f}%  {r['win_rate']:>5.0f}%  "
                  f"{r['vn_ret']:>+6.1f}%  {verdict}")
        print(f"  {'-'*70}")
        print(f"  {'COMBINED OOS':<12} {tot_c:>+7.1f}%  {ann_c:>+5.1f}%  "
              f"{sh_c:>6.2f}  {mdd_c:>6.1f}%")
        print(f"  Beat VN-Index: {beat_count}/{len(oos_results)} windows")
        print(f"{'='*72}")

    # Save OOS results CSV
    out_rows = [{k: v for k, v in r.items() if k != "df_oos"}
                for r in oos_results]
    out = pd.DataFrame(out_rows)
    for path in ["walkforward_results.csv",
                 "/mnt/user-data/outputs/walkforward_results.csv"]:
        try:
            out.to_csv(path, index=False)
            print(f"  OOS results → {path}")
            break
        except Exception:
            pass

    # Plot
    plot_walk_forward(oos_results)


def plot_walk_forward(oos_results):
    """Chart showing equity curves for each OOS window side by side."""
    if not oos_results:
        return

    TC  = "#111111"
    n   = len(oos_results)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 10), facecolor="white")
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.06,
                        right=0.97, hspace=0.35, wspace=0.3)

    colors = ["#2980b9","#27ae60","#e67e22","#8e44ad","#e74c3c"]

    for i, r in enumerate(oos_results):
        df   = r["df_oos"]
        ax_e = axes[0, i]
        ax_d = axes[1, i]

        for ax in [ax_e, ax_d]:
            ax.set_facecolor("#f9f9f9")
            ax.tick_params(colors=TC, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#aaaaaa")
            ax.grid(True, ls=":", alpha=0.35)

        # Equity curve
        cum     = (1 + df["ret"]).cumprod()
        cum_vn  = (1 + df["vn_ret"]).cumprod()
        ax_e.plot(df.index, cum.values,    color=colors[i], lw=2.0,
                  label=f"Strategy")
        ax_e.plot(df.index, cum_vn.values, color="#e74c3c", lw=1.2,
                  ls=":", alpha=0.8, label="VN-Index")
        ax_e.axhline(1.0, color="#555", lw=0.5, ls="--")
        ax_e.fill_between(df.index, 1.0, cum.values,
                          where=(cum.values >= 1.0),
                          color=colors[i], alpha=0.08)
        ax_e.fill_between(df.index, 1.0, cum.values,
                          where=(cum.values < 1.0),
                          color="#e74c3c", alpha=0.10)
        verdict = "BEAT VN ✓" if r["beat_vn"] else "MISS VN ✗"
        ax_e.set_title(
            f"OOS {r['window']}\n"
            f"{r['total_ret']:+.1f}%  Sh={r['sharpe']:.2f}  {verdict}",
            fontsize=8, color=TC, fontweight="bold")
        ax_e.legend(fontsize=6, facecolor="white", loc="upper left")
        ax_e.set_ylabel("Cumulative", fontsize=7, color=TC)

        # Drawdown
        mdd_ser = (cum / cum.cummax()) - 1
        ax_d.fill_between(df.index, 0, mdd_ser.values * 100,
                          color="#e74c3c", alpha=0.5)
        ax_d.axhline(0, color="#555", lw=0.5)
        ax_d.set_ylabel("Drawdown %", fontsize=7, color=TC)
        ax_d.set_title(
            f"MDD: {r['mdd']:.1f}%  "
            f"Invested: {r['invested_pct']:.0f}%  "
            f"Trades: {r['n_trades']}",
            fontsize=7, color=TC)

    fig.suptitle(
        f"Walk-Forward Validation — 5 OOS windows (2016–2026)\n"
        f"Parameters fixed — K(t) auto-computed from rolling data only",
        fontsize=10, color=TC, y=0.96)

    fname = "walkforward_chart.png"
    for path in [fname, f"/mnt/user-data/outputs/{fname}"]:
        try:
            plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
            print(f"  Chart → {path}")
        except Exception:
            pass
    plt.show()
    plt.close(fig)


def main():
    global _FUND_DATA, _QFEAT, _FLOW_ENGINE
    stock_data     = load_individual_stocks(INDIVIDUAL_DATA_DIR)
    liquid_tickers = get_liquid_tickers(stock_data)

    if FUNDAMENTAL_FILTER_ENABLED:
        _FUND_DATA = load_fundamental_data()

    if FACTOR_SELECTION_ENABLED and not _FACTOR_RANKER_AVAILABLE:
        print("  [FACTOR] factor_stock_ranker.py not found — factor selection disabled")
        for _sec, _crit in SECTOR_FUND_CRITERIA.items():
            print(f"  [FUND] {_sec}: {_crit}")

    df_all, gb, vn, approved = load_sector_data(
        STOCK_DATA_PATH, VNINDEX_PATH, liquid_tickers)

    stock_data = {t: d for t, d in stock_data.items()
                  if t in approved and t in liquid_tickers}

    if FACTOR_SELECTION_ENABLED and _FACTOR_RANKER_AVAILABLE:
        print("Loading quarterly factor features...")
        # Only load parquets for the strategy-sector universe (~200 tickers)
        # instead of all ~1566 symbols — cuts load time from ~30s to ~3s
        _QFEAT = build_factor_features(symbols=list(approved))

    # ── Investor-flow signal engine ──────────────────────────────────────────
    if FLOW_SIGNAL_ENABLED:
        try:
            from flow_signals import FlowSignalEngine
            _flow_dir = Path(__file__).parent / "data" / "investor_flow"
            if _flow_dir.exists():
                _FLOW_ENGINE = FlowSignalEngine(_flow_dir).load()
                print(f"  [FLOW] Loaded investor flow for {len(_FLOW_ENGINE.tickers)} tickers: "
                      f"{', '.join(_FLOW_ENGINE.tickers)}")
                print(f"  [FLOW] Flow ranking: sectors={FLOW_SECTORS or 'all'}, "
                      f"top_n={FLOW_RANK_TOP_N}, window={FLOW_RANK_WINDOW}d, "
                      f"dist_exit={FLOW_DIST_EXIT}")
            else:
                print(f"  [FLOW] data/investor_flow/ not found — "
                      f"run: python fetch_investor_flow.py --all")
        except ImportError:
            print("  [FLOW] flow_signals.py not found — flow overlay disabled")
        except Exception as _e:
            print(f"  [FLOW] Load failed: {_e} — flow overlay disabled")

    # Re-inject ETF ticker into stock_data — it's not a sector stock
    # so it gets filtered out above, but we need it for the overlay
    if ETF_TICKER is not None:
        etf_all = load_individual_stocks(INDIVIDUAL_DATA_DIR)
        if ETF_TICKER in etf_all:
            stock_data[ETF_TICKER] = etf_all[ETF_TICKER]
            print(f"  ETF overlay: {ETF_TICKER} loaded "
                  f"({len(etf_all[ETF_TICKER])} rows, "
                  f"last date: {etf_all[ETF_TICKER].index.max().date()})")
        else:
            print(f"  ETF overlay: {ETF_TICKER}.csv not found in {INDIVIDUAL_DATA_DIR}")
            print(f"               — ETF overlay disabled for this run")

    # ── Mode: set WALK_FORWARD=True to run OOS validation ─────────
    WALK_FORWARD = False

    if WALK_FORWARD:
        run_walk_forward(df_all, gb, vn, stock_data)
        return

    cutoff = pd.Timestamp(BACKTEST_START)

    # ── Pre-run: check Banks vs FinServ correlation ────────────────
    if "Financial Services" in (SECTOR_GROUPS or {}):
        print("\n  PRE-CHECK: Banks vs Financial Services correlation")
        from scipy import stats as scipy_stats
        banks_sig = build_sector_signal(df_all, "Banks")
        fs_sig    = build_sector_signal(df_all, "Financial Services")
        common    = banks_sig.index.intersection(fs_sig.index)
        b_sp = banks_sig.loc[common, "spread"].dropna()
        f_sp = fs_sig.loc[common, "spread"].dropna()
        common2   = b_sp.index.intersection(f_sp.index)
        b_sp, f_sp = b_sp.loc[common2], f_sp.loc[common2]
        corr_0  = b_sp.corr(f_sp)
        # Lead-lag: does FinServ lag Banks?
        corrs = {}
        for lag in range(-30, 31, 5):
            if lag < 0:
                corrs[lag] = b_sp.iloc[:lag].corr(f_sp.iloc[-lag:])
            elif lag > 0:
                corrs[lag] = b_sp.iloc[lag:].corr(f_sp.iloc[:-lag])
            else:
                corrs[lag] = corr_0
        best_lag  = max(corrs, key=corrs.get)
        best_corr = corrs[best_lag]
        print(f"    Contemporaneous correlation:  {corr_0:+.3f}")
        print(f"    Best correlation lag:         {best_lag:+d} days → {best_corr:+.3f}")
        if best_lag > 0:
            print(f"    → FinServ LAGS Banks by {best_lag}d")
        elif best_lag < 0:
            print(f"    → FinServ LEADS Banks by {-best_lag}d")
        else:
            print(f"    → Synchronous (no meaningful lead/lag)")
        if corr_0 > 0.75:
            print(f"    ⚠ HIGH correlation ({corr_0:.2f}) — likely to split capital")
            print(f"      without adding diversification")
        elif corr_0 > 0.5:
            print(f"    ⚡ MODERATE correlation ({corr_0:.2f}) — may add value")
        else:
            print(f"    ✓ LOW correlation ({corr_0:.2f}) — good diversifier")
        print()

    print(f"\n{'='*64}")
    print(f"  BEST-OF-N — fully dynamic vol-adjusted entry threshold")
    print(f"  Sectors : {' | '.join(SECTOR_GROUPS.values() if SECTOR_GROUPS else ['all'])}")
    print(f"  Capital : {CAPITAL_VND/1e6:.0f}M VND  |  100% concentrated")
    print(f"  Entry   : RECOVERY or LEADING(score≥{MIN_ENTRY_SCORE})")
    print(f"  Threshold: K(t) × rolling_std(spread,{VOL_WINDOW}d)")
    print(f"  K(t)    : 0.75-(rolling_recov_days(t)-40)/200  clip[0.25,0.75]")
    print(f"  Recov   : rolling avg over last {RECOV_WINDOW_LONG}d  (no hardcoded values)")
    print(f"  Cooldown: {COOLDOWN_AFTER_PEAK}d RECOVERY block after any peaking exit")
    trail_str = (f"{STOCK_TRAILING_STOP_PCT*100:.0f}% (arms after +{TRAIL_ACTIVATE_PCT*100:.0f}%)"
                 if STOCK_TRAILING_STOP_PCT else "off")
    tp_str    = f"{STOCK_TP_PCT*100:.0f}%" if STOCK_TP_PCT else "off"
    sl_str    = f"{STOCK_SL_PCT*100:.0f}%" if STOCK_SL_PCT else "off"
    etf_str   = f"{ETF_TICKER} ({ETF_ALLOC*100:.0f}% idle)" if ETF_TICKER else "disabled"
    dca_str   = DCA_MODE if DCA_MODE != "NONE" else "off (100% day 1)"
    if DCA_MODE == "SCORE":
        dca_str += f" (high≥{DCA_SCORE_HIGH} mid≥{DCA_SCORE_MID})"
    print(f"  Trail stop:{trail_str}  |  Fixed TP: {tp_str}  |  SL: {sl_str}")
    kijun_str = (f"on (within {KIJUN_BUY_THRESHOLD*100:.0f}% of Kijun-{KIJUN_PERIOD}, "
                 f"max wait {ENTRY_MAX_WAIT_DAYS}d)"
                 if ENTRY_TIMING_KIJUN else "off")
    print(f"  Kijun entry:{kijun_str}")
    print(f"  Selection: {STOCK_SELECTION} — stock filter within sector")
    print(f"  DCA mode:  {dca_str}")
    print(f"  ETF idle:  {etf_str}")
    print(f"  Liq filter: {MIN_LIQUIDITY_VND/1e6:.0f}M VND/day static  "
          f"+ dynamic ≤20% of stock's own 20d median volume")
    print(f"  Window  : {cutoff.date()} → {vn.index.max().date()}")
    print(f"{'='*64}\n")

    if COMPARE_REGIME:
        _compare_regime(df_all, gb, vn, stock_data, cutoff)
        return

    if COMPARE_FG:
        _compare_fg(df_all, gb, vn, stock_data, cutoff)
        return

    df_d, df_t, sigs, sectors = run_backtest(df_all, gb, vn, stock_data, cutoff)

    print_report(df_d, df_t, sectors)
    plot(df_d, df_t, sigs, sectors)
    save_results(df_d, df_t, sectors)


def _compare_regime(df_all, gb, vn, stock_data, cutoff):
    """Run backtest twice (regime on vs off) and print side-by-side annual comparison."""
    print("\n" + "="*64)
    print("  REGIME FILTER COMPARISON")
    print("="*64)

    print("\n  Running WITHOUT regime filter...")
    df_no, dt_no, _, _ = run_backtest(df_all, gb, vn, stock_data, cutoff,
                                       regime_enabled=False)
    print("\n  Running WITH regime filter...")
    df_on, dt_on, _, _ = run_backtest(df_all, gb, vn, stock_data, cutoff,
                                       regime_enabled=True)

    def _annual_ret(df_d):
        out = {}
        for yr in sorted(df_d.index.year.unique()):
            r = df_d[df_d.index.year == yr]["ret"].replace([np.inf, -np.inf], 0)
            if len(r) < 5:
                continue
            out[yr] = round(((1 + r).cumprod().iloc[-1] - 1) * 100, 1)
        return out

    no_ann  = _annual_ret(df_no)
    on_ann  = _annual_ret(df_on)
    tot_no, ann_no, sh_no, mdd_no = metrics(df_no["ret"])
    tot_on, ann_on, sh_on, mdd_on = metrics(df_on["ret"])

    years = sorted(set(no_ann) | set(on_ann))
    print(f"\n  {'Year':<6} {'No Filter':>10} {'With Filter':>12} {'Delta':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*8}")
    for yr in years:
        n = no_ann.get(yr, float("nan"))
        w = on_ann.get(yr, float("nan"))
        d = w - n if not (np.isnan(n) or np.isnan(w)) else float("nan")
        flag = "  +" if d > 0 else ("  -" if d < 0 else "")
        print(f"  {yr:<6} {n:>+9.1f}%  {w:>+10.1f}%  {d:>+7.1f}%{flag}")

    d_tot = tot_on - tot_no
    d_ann = ann_on - ann_no
    d_sh  = sh_on  - sh_no
    print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*8}")
    print(f"  {'TOTAL':<6} {tot_no:>+9.1f}%  {tot_on:>+10.1f}%  {d_tot:>+7.1f}%")
    print(f"  {'Ann':<6} {ann_no:>+9.1f}%  {ann_on:>+10.1f}%  {d_ann:>+7.1f}%")
    print(f"  {'Sharpe':<6} {sh_no:>10.2f}  {sh_on:>11.2f}  {d_sh:>+7.2f}")
    print(f"  {'MDD':<6} {mdd_no:>+9.1f}%  {mdd_on:>+10.1f}%  {mdd_on-mdd_no:>+7.1f}%")
    print(f"  {'Trades':<6} {len(dt_no):>10}  {len(dt_on):>12}")
    print()


def _compare_fg(df_all, gb, vn, stock_data, cutoff):
    """Run backtest twice (fg confirm on vs off) and print side-by-side annual table."""
    print("\n" + "="*64)
    print("  FG CONFIRMATION FILTER COMPARISON")
    print(f"  Sectors filtered: {sorted(FG_CONFIRM_SECTORS)}")
    print(f"  Min streak: {FG_CONFIRM_STREAK} consecutive fg-buying days")
    print("="*64)

    print("\n  Running WITHOUT fg confirmation...")
    df_no, dt_no, _, _ = run_backtest(df_all, gb, vn, stock_data, cutoff,
                                       fg_confirm_enabled=False)
    print("\n  Running WITH fg confirmation...")
    df_on, dt_on, _, _ = run_backtest(df_all, gb, vn, stock_data, cutoff,
                                       fg_confirm_enabled=True)

    def _annual_ret(df_d):
        out = {}
        for yr in sorted(df_d.index.year.unique()):
            r = df_d[df_d.index.year == yr]["ret"].replace([np.inf, -np.inf], 0)
            if len(r) < 5:
                continue
            out[yr] = round(((1 + r).cumprod().iloc[-1] - 1) * 100, 1)
        return out

    no_ann  = _annual_ret(df_no)
    on_ann  = _annual_ret(df_on)
    tot_no, ann_no, sh_no, mdd_no = metrics(df_no["ret"])
    tot_on, ann_on, sh_on, mdd_on = metrics(df_on["ret"])

    years = sorted(set(no_ann) | set(on_ann))
    print(f"\n  {'Year':<6} {'No Filter':>10} {'With Filter':>12} {'Delta':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*8}")
    for yr in years:
        n = no_ann.get(yr, float("nan"))
        w = on_ann.get(yr, float("nan"))
        d = w - n if not (np.isnan(n) or np.isnan(w)) else float("nan")
        flag = "  +" if d > 0 else ("  -" if d < 0 else "")
        print(f"  {yr:<6} {n:>+9.1f}%  {w:>+10.1f}%  {d:>+7.1f}%{flag}")

    d_tot = tot_on - tot_no
    d_ann = ann_on - ann_no
    d_sh  = sh_on  - sh_no
    print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*8}")
    print(f"  {'TOTAL':<6} {tot_no:>+9.1f}%  {tot_on:>+10.1f}%  {d_tot:>+7.1f}%")
    print(f"  {'Ann':<6} {ann_no:>+9.1f}%  {ann_on:>+10.1f}%  {d_ann:>+7.1f}%")
    print(f"  {'Sharpe':<6} {sh_no:>10.2f}  {sh_on:>11.2f}  {d_sh:>+7.2f}")
    print(f"  {'MDD':<6} {mdd_no:>+9.1f}%  {mdd_on:>+10.1f}%  {mdd_on-mdd_no:>+7.1f}%")
    print(f"  {'Trades':<6} {len(dt_no):>10}  {len(dt_on):>12}")
    print()


def save_results(df_d, df_t, sectors):
    """Save a compact results CSV so you can upload instead of pasting output."""
    tot, ann, sh, mdd = metrics(df_d["ret"])
    inv  = (df_d["active"] != "CASH").mean() * 100
    rows = []

    # Overall summary row
    rows.append({
        "type": "summary",
        "label": "TOTAL",
        "total_return_pct": round(tot, 2),
        "ann_return_pct": round(ann, 2),
        "sharpe": round(sh, 3),
        "max_dd_pct": round(mdd, 2),
        "time_invested_pct": round(inv, 1),
        "start_capital_M": round(df_d["port_value"].iloc[0]/1e6, 2),
        "end_capital_M": round(df_d["port_value"].iloc[-1]/1e6, 2),
        "n_trades": len(df_t) if not df_t.empty else 0,
        "win_rate_pct": round((df_t["pnl_pct"]>0).mean()*100, 1) if not df_t.empty else 0,
        "avg_winner_pct": round(df_t[df_t["pnl_pct"]>0]["pnl_pct"].mean(), 2) if not df_t.empty else 0,
        "avg_loser_pct": round(df_t[df_t["pnl_pct"]<=0]["pnl_pct"].mean(), 2) if not df_t.empty else 0,
    })

    # Per-sector rows
    for sec in sectors:
        if df_t.empty:
            continue
        st = df_t[df_t["sector"] == sec]
        if st.empty:
            continue
        rows.append({
            "type": "sector",
            "label": sec,
            "total_return_pct": None,
            "ann_return_pct": None,
            "sharpe": None,
            "max_dd_pct": None,
            "time_invested_pct": None,
            "start_capital_M": None,
            "end_capital_M": round(st["pnl_vnd"].sum()/1e6, 2),
            "n_trades": len(st),
            "win_rate_pct": round((st["pnl_pct"]>0).mean()*100, 1),
            "avg_winner_pct": round(st[st["pnl_pct"]>0]["pnl_pct"].mean(), 2) if (st["pnl_pct"]>0).any() else 0,
            "avg_loser_pct": round(st[st["pnl_pct"]<=0]["pnl_pct"].mean(), 2) if (st["pnl_pct"]<=0).any() else 0,
        })

    # Annual breakdown rows
    for yr in sorted(df_d.index.year.unique()):
        yr_d = df_d[df_d.index.year == yr]
        if len(yr_d) < 5:
            continue
        r   = yr_d["ret"].replace([np.inf,-np.inf], 0)
        cum = (1+r).cumprod()
        t   = (cum.iloc[-1]-1)*100
        s   = (r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else 0
        m   = ((cum/cum.cummax())-1).min()*100
        vn  = (((1+yr_d["vn_ret"]).cumprod().iloc[-1])-1)*100
        inv_yr = (yr_d["active"]!="CASH").mean()*100
        rows.append({
            "type": "annual",
            "label": str(yr),
            "total_return_pct": round(t, 2),
            "ann_return_pct": round(t, 2),
            "sharpe": round(s, 3),
            "max_dd_pct": round(m, 2),
            "time_invested_pct": round(inv_yr, 1),
            "start_capital_M": None,
            "end_capital_M": None,
            "n_trades": None,
            "win_rate_pct": None,
            "avg_winner_pct": round(vn, 2),   # repurposed as vn_return
            "avg_loser_pct": None,
        })

    out = pd.DataFrame(rows)
    for path in ["backtest_results.csv", "/mnt/user-data/outputs/backtest_results.csv"]:
        try:
            out.to_csv(path, index=False)
            print(f"  Results saved → {path}")
            break
        except Exception:
            pass

    if not df_t.empty:
        for path in ["/mnt/user-data/outputs/trades_option_c.csv",
                     "trades_option_c.csv"]:
            try:
                df_t.to_csv(path, index=False)
                print(f"  Trade log → {path}")
                break
            except Exception:
                pass


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Simple CLI overrides
    _args = sys.argv[1:]

    # --vn-gate off/soft/hard
    if "--vn-gate" in _args:
        _idx = _args.index("--vn-gate")
        if _idx + 1 < len(_args):
            _gate_val = _args[_idx + 1].upper()
            if _gate_val in ("OFF", "SOFT", "HARD"):
                VNINDEX_GATE = _gate_val
                print(f"[CLI] VNINDEX_GATE = {VNINDEX_GATE}")

    # --vn-exit   → enable  VN exit accelerator
    # --no-vn-exit → disable VN exit accelerator
    if "--vn-exit" in _args:
        VNINDEX_EXIT_ENABLED = True
        print("[CLI] VNINDEX_EXIT_ENABLED = True")
    elif "--no-vn-exit" in _args:
        VNINDEX_EXIT_ENABLED = False
        print("[CLI] VNINDEX_EXIT_ENABLED = False")

    main()