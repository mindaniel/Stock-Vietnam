"""
flow_signals.py  —  Investor-flow trading signals for VN stocks
================================================================
Loads data from data/investor_flow/*.parquet  (built by fetch_investor_flow.py)
and turns it into actionable signals you can use to:
  • Know WHEN TO SELL (distribution alert)
  • Find OPPORTUNITIES (institutional accumulation)
  • AVOID RISK (heavy-sell regime)

STANDALONE USE:
  python flow_signals.py ACB VCB TCB          # signals for specific tickers
  python flow_signals.py --screen             # rank all available tickers
  python flow_signals.py --sector Banks       # all bank stocks (needs parquet map)
  python flow_signals.py --date 2025-10-01    # signals as of a past date
  python flow_signals.py --sell-scan          # only show stocks with active sell alerts

IMPORT INTO BACKTEST:
  from flow_signals import FlowSignalEngine
  fse = FlowSignalEngine().load()
  score = fse.smart_score("ACB", pd.Timestamp("2025-06-01"))  # -1..+1
  alert = fse.distribution_alert("ACB", pd.Timestamp("2025-06-01"))  # bool
  regime = fse.flow_regime("ACB", pd.Timestamp("2025-06-01"))  # string

CORE SIGNALS
────────────────────────────────────────────────────────────────────────────
smart_score(ticker, date, window=20):  float in [-1, +1]
  Weighted z-score of recent net flows across all 5 investor types.
  + → institutional accumulation (bullish)
  - → distribution / heavy selling (bearish)
  Weights: Tổ chức TN=0.50, Cá nhân TN=0.20, Tự doanh=0.15,
           Tổ chức NN=0.10, Cá nhân NN=0.05

distribution_alert(ticker, date, window=10):  bool
  True when "hand-off to retail" pattern is active:
    ✗ Tổ chức NN + Tự doanh both net-SELLING
    ✓ Cá nhân TN net-BUYING  (retail absorbing institutional exits)
  Historically precedes price peaks.  Suggested action: tighten stop / reduce.

flow_regime(ticker, date):  str
  "SQUEEZE"     all types buying together   → momentum confirmation
  "ACCUMULATE"  domestic institutions buying, foreigners selling  → value entry
  "DISTRIBUTE"  institutions + foreigners selling, retail buying  → exit warning
  "HEAVY_SELL"  4+ types net selling   → avoid / cut losses
  "NEUTRAL"     no clear pattern
  "NO_DATA"     insufficient history

momentum_flip(ticker, date):  dict
  Compares 5d smart_score vs 20d smart_score.
  If 5d > 20d + threshold → turning bullish (fresh accumulation)
  If 5d < 20d - threshold → turning bearish (distribution starting)
  Returns {"direction": "up"/"down"/"flat", "delta": float}
────────────────────────────────────────────────────────────────────────────
DATA:  Sep 2024 onwards.  Returns 0.0 / "NO_DATA" before that — never penalises
       stocks with no coverage so backtest fallback stays unbiased.
"""

import sys, argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8")

BASE     = Path(__file__).parent.parent  # repo root (lib/ is one level down)
FLOW_DIR = BASE / "data" / "investor_flow"

DATA_START = pd.Timestamp("2024-09-16")

FLOW_COLS = [
    "tu_doanh_net",
    "ca_nhan_trongnuoc_net",
    "to_chuc_trongnuoc_net",
    "ca_nhan_nuocngoai_net",
    "to_chuc_nuocngoai_net",
]

TYPE_LABELS = {
    "tu_doanh_net":           "Tự doanh  ",
    "ca_nhan_trongnuoc_net":  "Cá nhân TN",
    "to_chuc_trongnuoc_net":  "Tổ chức TN",
    "ca_nhan_nuocngoai_net":  "Cá nhân NN",
    "to_chuc_nuocngoai_net":  "Tổ chức NN",
}

# Weights based on empirical analysis:
#   Tổ chức TN: patient accumulator, best predictor in VN banks/real estate
#   Cá nhân TN: skilled retail, profitable across windows
#   Tự doanh:   short-term, noisy but real-time signal
#   Tổ chức NN: mixed (positive signal in VHM/TCB, negative in ACB/BID/MBB/HPG)
#   Cá nhân NN: tiny, noisy
SMART_WEIGHTS = {
    "to_chuc_trongnuoc_net":  0.50,
    "ca_nhan_trongnuoc_net":  0.20,
    "tu_doanh_net":           0.15,
    "to_chuc_nuocngoai_net":  0.10,
    "ca_nhan_nuocngoai_net":  0.05,
}

# Distribution alert: all of these must be net selling
DIST_SELLERS = ["to_chuc_nuocngoai_net", "tu_doanh_net"]
# Distribution alert: this must be net buying (retail absorbing)
DIST_BUYER   = "ca_nhan_trongnuoc_net"

# Regime thresholds
ACCUMULATE_SCORE_THRESHOLD  =  0.25
DISTRIBUTE_SCORE_THRESHOLD  = -0.25
FLIP_DELTA_THRESHOLD        =  0.15   # 5d vs 20d gap to declare a flip


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class FlowSignalEngine:
    """
    Load investor flow parquets and produce point-in-time trading signals.

    Usage:
        fse = FlowSignalEngine().load()            # load all available tickers
        fse = FlowSignalEngine().load(["ACB","VCB"]) # specific tickers only
        score = fse.smart_score("ACB", pd.Timestamp("2025-06-01"))
    """

    def __init__(self, flow_dir: Path = None):
        self.flow_dir = Path(flow_dir) if flow_dir else FLOW_DIR
        self._data: dict = {}
        self._loaded = False

    # ── loading ────────────────────────────────────────────────────────────────
    def load(self, tickers=None) -> "FlowSignalEngine":
        """Load parquet files.  Call once before using any signal method."""
        if tickers:
            paths = [self.flow_dir / f"{t.upper()}.parquet" for t in tickers]
        else:
            paths = sorted(self.flow_dir.glob("*.parquet"))
        for p in paths:
            if not p.exists():
                continue
            try:
                df = pd.read_parquet(p)
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                self._data[p.stem.upper()] = df
            except Exception:
                pass
        self._loaded = True
        return self

    @property
    def tickers(self) -> list:
        return sorted(self._data.keys())

    def has_data(self, ticker: str) -> bool:
        return ticker.upper() in self._data

    # ── internal helpers ───────────────────────────────────────────────────────
    def _get(self, ticker: str, as_of: pd.Timestamp) -> pd.DataFrame:
        """Rows on or before as_of (point-in-time safe)."""
        df = self._data.get(ticker.upper())
        if df is None or df.empty:
            return pd.DataFrame()
        return df[df["date"] <= as_of].copy()

    def _z(self, series: pd.Series, window: int) -> float:
        """
        Z-score of the most recent value vs the prior `window` values.
        Clamped to [-3, +3] then normalised to [-1, +1].
        Returns 0.0 if insufficient data.
        """
        hist = series.dropna()
        if len(hist) < max(5, window // 4):
            return 0.0
        recent = float(hist.iloc[-1])
        prior  = hist.iloc[-window - 1:-1] if len(hist) > window else hist.iloc[:-1]
        if prior.empty:
            return 0.0
        mu  = float(prior.mean())
        std = float(prior.std())
        if std < 1e-9:
            # No variation → check direction of absolute flow
            return float(np.clip(np.sign(recent), -1, 1)) * 0.3
        z = (recent - mu) / std
        return float(np.clip(z / 3.0, -1.0, 1.0))

    def _raw_mean(self, df: pd.DataFrame, col: str, window: int) -> float:
        """Simple mean of last `window` rows for `col`."""
        if col not in df.columns or df.empty:
            return 0.0
        return float(df[col].tail(window).mean())

    # ── core signals ──────────────────────────────────────────────────────────
    def smart_score(
        self,
        ticker: str,
        as_of: pd.Timestamp,
        window: int = 20,
    ) -> float:
        """
        Weighted z-score signal.  Range: [-1, +1].
        0.0 = no data (not a bearish signal — do not penalise uncovered stocks).
        + = institutional accumulation (bullish).
        - = distribution / selling pressure (bearish).
        """
        df = self._get(ticker, as_of)
        if df.empty or len(df) < max(5, window // 4):
            return 0.0
        score   = 0.0
        total_w = 0.0
        for col, w in SMART_WEIGHTS.items():
            if col not in df.columns:
                continue
            score   += w * self._z(df[col], window)
            total_w += w
        return float(score / total_w) if total_w > 0 else 0.0

    def retail_accum_score(
        self,
        ticker: str,
        as_of: pd.Timestamp,
        window: int = 60,
    ) -> float:
        """
        Foreign-RETAIL-only accumulation score (ca_nhan_nuocngoai_net), NOT
        the blended smart_score (which weights domestic institutional at
        0.50 vs retail-foreign at only 0.05 — the opposite signal).

        Validated (2024-09 to 2026-07 sample, Fama-MacBeth sector-neutral):
        foreign retail accumulation predicts a stock beating its OWN sector
        peers (t~3 at 60-120d horizons) — foreign institutional accumulation
        does the opposite. Use this, not smart_score, for a "ride retail
        into a growth story" signal. See archive/retail_growth_strategy.py.

        Returns rolling `window`-day net retail-foreign flow, normalized by
        its own rolling mean absolute flow (so it's comparable across
        stocks of different size) — NOT bounded to [-1, 1] like smart_score.
        0.0 = no data or insufficient history.
        """
        df = self._get(ticker, as_of)
        col = "ca_nhan_nuocngoai_net"
        if df.empty or col not in df.columns or len(df) < window:
            return 0.0
        recent = df[col].tail(window)
        cum   = recent.sum()
        scale = recent.abs().mean() * window
        if not scale or pd.isna(scale) or scale < 1e-9:
            return 0.0
        return float(cum / scale)

    def distribution_alert(
        self,
        ticker: str,
        as_of: pd.Timestamp,
        window: int = 10,
    ) -> bool:
        """
        True when institutions are distributing TO retail:
          - Foreign + Tự doanh both average net-negative (selling)
          - Cá nhân TN average net-positive (retail absorbing)
        Action: tighten stop or reduce position.
        """
        df = self._get(ticker, as_of)
        if df.empty or len(df) < 5:
            return False
        for col in DIST_SELLERS:
            if self._raw_mean(df, col, window) >= 0:
                return False
        return self._raw_mean(df, DIST_BUYER, window) > 0

    def accumulation_signal(
        self,
        ticker: str,
        as_of: pd.Timestamp,
        window: int = 20,
    ) -> bool:
        """
        True when domestic institutions are actively accumulating:
          - Tổ chức TN net-positive AND above its recent average
          - NOT simultaneously in DISTRIBUTE mode
        Action: consider entry / hold existing position.
        """
        df = self._get(ticker, as_of)
        if df.empty or len(df) < 10:
            return False
        col = "to_chuc_trongnuoc_net"
        if col not in df.columns:
            return False
        dom_inst_z = self._z(df[col], window)
        dist = self.distribution_alert(ticker, as_of)
        return dom_inst_z > 0.2 and not dist

    def flow_regime(
        self,
        ticker: str,
        as_of: pd.Timestamp,
        window: int = 10,
    ) -> str:
        """Classify the current investor flow regime. See module docstring."""
        df = self._get(ticker, as_of)
        if df.empty or len(df) < 5:
            return "NO_DATA"
        recent = df.tail(window)
        avgs = {col: float(recent[col].mean())
                for col in FLOW_COLS if col in recent.columns}
        if not avgs:
            return "NO_DATA"

        dom_inst = avgs.get("to_chuc_trongnuoc_net", 0)
        fg_inst  = avgs.get("to_chuc_nuocngoai_net", 0)
        retail   = avgs.get("ca_nhan_trongnuoc_net", 0)

        n_pos = sum(1 for v in avgs.values() if v > 0)
        n_neg = sum(1 for v in avgs.values() if v < 0)

        if n_pos >= 4:
            return "SQUEEZE"       # everyone buying = momentum confirmation
        if n_neg >= 4:
            return "HEAVY_SELL"    # everyone selling = danger
        if dom_inst > 0 and fg_inst < 0:
            return "ACCUMULATE"    # domestic smart money buying, foreigners exiting
        if dom_inst < 0 and fg_inst < 0 and retail > 0:
            return "DISTRIBUTE"    # institutions handing off to retail
        return "NEUTRAL"

    def momentum_flip(
        self,
        ticker: str,
        as_of: pd.Timestamp,
    ) -> dict:
        """
        Detect when smart money flow is changing direction.
        Compares 5-day vs 20-day smart_score.
        Returns: {"direction": "up"/"down"/"flat", "delta": float,
                  "s5": float, "s20": float}
        """
        s5  = self.smart_score(ticker, as_of, window=5)
        s20 = self.smart_score(ticker, as_of, window=20)
        delta = s5 - s20
        if delta > FLIP_DELTA_THRESHOLD:
            direction = "up"
        elif delta < -FLIP_DELTA_THRESHOLD:
            direction = "down"
        else:
            direction = "flat"
        return {"direction": direction, "delta": delta, "s5": s5, "s20": s20}

    def peak_exit_signal(
        self,
        ticker: str,
        as_of: pd.Timestamp,
        window: int = 10,
    ) -> tuple:
        """
        Composite exit signal: fires when the stock is likely at or near its
        highest price level and smart money is leaving.

        Signal hierarchy (checked in priority order):
          1. distribution_alert  — institutions distributing TO retail.
                                   Classic topping pattern: smart money sells
                                   into retail demand at elevated prices.
          2. HEAVY_SELL regime   — 4+ investor types net-selling simultaneously.
                                   Broad institutional exit; do not hold.
          3. momentum_flip down  — short-term flow score has turned negative
                                   relative to the 20-day average, AND the
                                   overall smart_score is already negative.
                                   Conviction signal: the reversal has landed.

        Returns (should_exit: bool, reason: str).
        Returns (False, "") before DATA_START (Sep 2024) — no data, no signal.
        """
        if as_of < DATA_START:
            return False, ""

        # 1. Institutions distributing to retail → classic price peak
        if self.distribution_alert(ticker, as_of, window):
            return True, "flow_distribution"

        # 2. Everyone selling → get out
        if self.flow_regime(ticker, as_of, window) == "HEAVY_SELL":
            return True, "flow_heavy_sell"

        # 3. Flow momentum turned negative with score confirmation
        flip  = self.momentum_flip(ticker, as_of)
        score = self.smart_score(ticker, as_of, window=20)
        if flip["direction"] == "down" and score < -0.10:
            return True, f"flow_flip_down(s={score:.2f})"

        return False, ""

    def get_signals(self, ticker: str, as_of: pd.Timestamp) -> dict:
        """All signals for a ticker as of a date (for display / logging)."""
        df = self._get(ticker, as_of)
        price = float(df["close"].iloc[-1]) if not df.empty else 0.0
        raw_5d = {col: self._raw_mean(df, col, 5)  for col in FLOW_COLS}
        raw_20d = {col: self._raw_mean(df, col, 20) for col in FLOW_COLS}
        flip = self.momentum_flip(ticker, as_of)
        return dict(
            ticker       = ticker.upper(),
            date         = as_of,
            price        = price,
            smart_score  = self.smart_score(ticker, as_of, 20),
            smart_5d     = self.smart_score(ticker, as_of, 5),
            distribution = self.distribution_alert(ticker, as_of),
            accumulation = self.accumulation_signal(ticker, as_of),
            regime       = self.flow_regime(ticker, as_of),
            flip         = flip,
            raw_5d       = raw_5d,
            raw_20d      = raw_20d,
        )

    def sector_smart_score(
        self,
        tickers: list,
        as_of: pd.Timestamp,
        window: int = 20,
    ) -> float:
        """Mean smart_score across covered tickers (ignores uncovered ones)."""
        scores = [self.smart_score(t, as_of, window) for t in tickers]
        valid  = [s for s in scores if s != 0.0]
        return float(np.mean(valid)) if valid else 0.0

    def rank_for_entry(
        self,
        tickers: list,
        as_of: pd.Timestamp,
        window: int = 20,
        top_n: int = None,
        exclude_distribution: bool = True,
    ) -> list:
        """
        Rank tickers by smart_score for entry selection.
        Used by 4sectors.py FLOW_RANK selection method.

        Returns list of (ticker, score) sorted best-first.
        Tickers with no data return score 0.0 and are sorted last.
        If exclude_distribution=True, tickers with active distribution
        alerts are moved to the end regardless of score.
        """
        scored = []
        for t in tickers:
            s = self.smart_score(t, as_of, window)
            d = self.distribution_alert(t, as_of) if exclude_distribution else False
            scored.append((t, s, d))

        # Sort: non-distributing first, then by score descending
        scored.sort(key=lambda x: (x[2], -x[1]))
        result = [(t, s) for t, s, _ in scored]
        if top_n:
            result = result[:top_n]
        return result


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE CLI  —  display / screening
# ══════════════════════════════════════════════════════════════════════════════

REGIME_SYMBOLS = {
    "SQUEEZE":    ("🟢 SQUEEZE",    "#22c55e"),
    "ACCUMULATE": ("📥 ACCUM",      "#3b82f6"),
    "NEUTRAL":    ("⬜ NEUTRAL",    "#9ca3af"),
    "DISTRIBUTE": ("📤 DISTRIB",    "#f59e0b"),
    "HEAVY_SELL": ("🔴 SELL",       "#ef4444"),
    "NO_DATA":    ("⬛ NO DATA",    "#6b7280"),
}

SCORE_GRADES = [
    ( 0.40, "★★★ STRONG BUY",  "📗"),
    ( 0.20, "★★☆ BUY",         "🟢"),
    ( 0.00, "★☆☆ MILD BUY",    "💚"),
    (-0.20, "☆☆☆ NEUTRAL",     "⬜"),
    (-0.40, "▽   MILD SELL",   "🟡"),
    (-9.99, "▼▼  SELL",        "🔴"),
]


def grade(score: float) -> str:
    for threshold, label, _ in SCORE_GRADES:
        if score > threshold:
            return label
    return "▼▼  SELL"


def emoji(score: float) -> str:
    for threshold, _, em in SCORE_GRADES:
        if score > threshold:
            return em
    return "🔴"


def print_signal(sig: dict, verbose: bool = False):
    t  = sig["ticker"]
    s  = sig["smart_score"]
    s5 = sig["smart_5d"]
    rg = sig["regime"]
    ds = sig["distribution"]
    ac = sig["accumulation"]
    fl = sig["flip"]
    px = sig["price"]

    regime_label = REGIME_SYMBOLS.get(rg, ("? " + rg, "#fff"))[0]
    flip_arrow   = {"up": " ↑ TURNING BULLISH", "down": " ↓ TURNING BEARISH", "flat": ""}

    dist_tag = "  ⚠  DISTRIBUTION ALERT — consider sell/reduce" if ds else ""
    accu_tag = "  ✓  ACCUMULATION — domestic institutions buying" if ac else ""

    print(f"\n  {'─'*60}")
    print(f"  {t}  |  price: {px:.2f}k  |  {regime_label}")
    print(f"  smart_score (20d): {s:>+.3f}  {emoji(s)}  {grade(s)}")
    print(f"  smart_score  (5d): {s5:>+.3f}  {flip_arrow[fl['direction']]}")
    if dist_tag:
        print(f"  {dist_tag}")
    if accu_tag:
        print(f"  {accu_tag}")

    if verbose:
        print(f"\n  {'Type':<12}  {'5d avg':>9}  {'20d avg':>9}")
        for col, label in TYPE_LABELS.items():
            v5  = sig["raw_5d"].get(col, 0)
            v20 = sig["raw_20d"].get(col, 0)
            bar5  = "▲" if v5  > 0 else "▼"
            bar20 = "▲" if v20 > 0 else "▼"
            print(f"  {label}  {bar5}{v5:>8.1f}  {bar20}{v20:>8.1f}  tỷ/day")


def print_screen_table(results: list):
    """Print ranked table for --screen or --sector."""
    print(f"\n  {'Ticker':>6}  {'Score':>6}  {'5d':>6}  {'Regime':<14}  {'Dist?':>5}  {'Action'}")
    print(f"  {'─'*68}")
    for sig in results:
        t   = sig["ticker"]
        s   = sig["smart_score"]
        s5  = sig["smart_5d"]
        rg  = sig["regime"]
        ds  = "⚠ YES" if sig["distribution"] else "  no"
        gr  = grade(s)
        em  = emoji(s)
        regime_label = REGIME_SYMBOLS.get(rg, ("? " + rg, ""))[0]
        print(f"  {t:>6}  {s:>+6.3f}  {s5:>+6.3f}  {regime_label:<14}  {ds:>5}  {em} {gr}")
    print()


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Investor flow signals for VN stocks")
    p.add_argument("tickers",    nargs="*", default=[])
    p.add_argument("--screen",   action="store_true",
                   help="Screen all available tickers")
    p.add_argument("--sell-scan", action="store_true",
                   help="Show only tickers with active distribution/sell alerts")
    p.add_argument("--sector",   default=None,
                   help="Filter by sector name (e.g. Banks, Real Estate)")
    p.add_argument("--date",     default=None,
                   help="Signal as of this date (YYYY-MM-DD), default=today")
    p.add_argument("--verbose",  action="store_true",
                   help="Show per-type flow breakdown")
    p.add_argument("--top",      type=int, default=None,
                   help="Show only top N tickers by score")
    args = p.parse_args()

    # Determine date
    if args.date:
        as_of = pd.Timestamp(args.date)
    else:
        as_of = pd.Timestamp.now().normalize()

    print(f"\n  Flow Signal Engine  |  as of {as_of.date()}")
    print(f"  Data dir: {FLOW_DIR}")

    fse = FlowSignalEngine().load()
    if not fse.tickers:
        print("  No data found. Run: python fetch_investor_flow.py --all")
        return

    # Determine ticker list
    tickers_to_show = []
    if args.sector:
        # Try to filter by sector using the stock parquet
        try:
            sp_path = BASE / "data" / "all_stocks_with_industries.parquet"
            if sp_path.exists():
                sp = pd.read_parquet(sp_path)
                sector_tickers = set(
                    sp[sp["industry"].str.contains(args.sector, case=False, na=False)]["ticker"]
                    .str.upper().unique()
                )
                tickers_to_show = [t for t in fse.tickers if t in sector_tickers]
                print(f"  Sector '{args.sector}': {len(tickers_to_show)} covered tickers")
            else:
                print(f"  WARNING: all_stocks_with_industries.parquet not found, "
                      f"showing all {len(fse.tickers)} tickers")
                tickers_to_show = fse.tickers
        except Exception as e:
            print(f"  Sector filter failed: {e}. Showing all.")
            tickers_to_show = fse.tickers
    elif args.tickers:
        tickers_to_show = [t.upper() for t in args.tickers]
    elif args.screen or args.sell_scan:
        tickers_to_show = fse.tickers
    else:
        tickers_to_show = fse.tickers

    # Compute signals
    results = []
    for t in tickers_to_show:
        if not fse.has_data(t):
            print(f"  {t}: no data — run fetch_investor_flow.py {t}")
            continue
        sig = fse.get_signals(t, as_of)
        results.append(sig)

    if not results:
        print("  No results.")
        return

    # Filter
    if args.sell_scan:
        results = [s for s in results
                   if s["distribution"] or s["smart_score"] < DISTRIBUTE_SCORE_THRESHOLD]
        print(f"  SELL SCAN: {len(results)} ticker(s) with active sell/distribution signals")

    # Sort by smart_score descending
    results.sort(key=lambda x: -x["smart_score"])
    if args.top:
        results = results[:args.top]

    # Display
    if len(results) == 1 or args.verbose:
        for sig in results:
            print_signal(sig, verbose=args.verbose)
    else:
        print_screen_table(results)

    # Summary
    n_buy   = sum(1 for r in results if r["smart_score"] >  ACCUMULATE_SCORE_THRESHOLD)
    n_sell  = sum(1 for r in results if r["smart_score"] <  DISTRIBUTE_SCORE_THRESHOLD)
    n_dist  = sum(1 for r in results if r["distribution"])
    n_accum = sum(1 for r in results if r["accumulation"])
    print(f"  Summary: {n_buy} strong-buy  |  {n_sell} sell/reduce  |  "
          f"{n_dist} distribution alerts  |  {n_accum} accumulation signals")

    # HOW TO USE
    print(f"""
  HOW TO USE THESE SIGNALS
  ─────────────────────────────────────────────────────────────────────────
  SELL POINT (when to exit a position):
    1. ⚠  Distribution alert fires  → start reducing, tighten stop
    2.    Smart score crosses below -0.25  → exit or hedge
    3. 🔴 Regime = HEAVY_SELL  → exit immediately
    4.    5d score < 20d score by >0.15  → momentum fading, reduce risk

  BUY OPPORTUNITY (when to enter):
    1. 📥 Regime = ACCUMULATE  → domestic institutions buying into weakness
    2. 🟢 Smart score > +0.25  → institutional accumulation confirmed
    3.    5d score > 20d score by >0.15  → fresh buying emerging
    4. ⬛ NO_DATA available yet → use price/vol signals only (no edge here)

  RISK TO AVOID:
    1. ⚠  Distribution alert = active → do NOT add to this position
    2. 🔴 HEAVY_SELL regime = avoid sector entirely
    3.    Foreign (Tổ chức NN) consistently net-selling = headwind
         (they have large positions to unwind, takes months)
  ─────────────────────────────────────────────────────────────────────────""")


if __name__ == "__main__":
    main()
