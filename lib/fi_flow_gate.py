"""
fi_flow_gate.py — Foreign-institutional flow gate for 4sectors.py.

WHAT THIS IS, AND WHY IT IS NOT THE SAME AS THE FLOW SIGNALS ALREADY TRIED

4sectors.py already contains two foreign-flow experiments, BOTH disabled after
losing money:

    FG_CONFIRM_ENABLED   "TESTED: -21,927% total drag, do not enable"
    FLOW_SIGNAL_ENABLED  institutional flow RE-RANKING; the file's own comment
                         records that foreign RETAIL accumulation predicts
                         outperformance while INSTITUTIONAL "does the OPPOSITE"

Independent testing in this project reached the same place: using FI flow to
RANK candidates scored -0.80% versus not using it at all, and FI-weak beat
FI-strong. So ranking by FI flow is a dead end and this module does not do it.

What did work — narrowly — was a different mechanism: a BINARY GATE requiring
BOTH high FI flow AND price confirmation, with the exit driven by PRICE, never
by flow. Measured against an identical price-only rule:

    liquidity bar    FI+price     price-only     edge      PF
        20bn          +2.45%        +2.16%      +0.29%   1.49 vs 1.43
         5bn          +3.45%        +2.96%      +0.49%   1.69 vs 1.58
         1bn          +4.29%        +3.61%      +0.68%   1.86 vs 1.71

The edge is small and grows as the universe widens. Treat it as a modest
filter that removes weak setups, NOT as an alpha source. The critical detail
is that flow is used ONLY to admit a candidate; it never ranks and never exits.

POINT-IN-TIME SAFETY
Ranks use expanding().rank(pct=True), which sees only data up to and including
that row, then are shifted by FLOW_LAG_DAYS to reflect that nguoiquansat flow
for day T is not scraped until later. Set the lag to match live conditions.

Usage from 4sectors.py:

    from fi_flow_gate import FIFlowGate
    _FI_GATE = FIFlowGate(flow_dir, lag_days=1)
    _FI_GATE.load(tickers)                      # once at startup
    keep = _FI_GATE.filter(candidates, exec_date)
"""

import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

FI_COL = "to_chuc_nuocngoai_net"


class FIFlowGate:
    """Binary admit/reject gate on foreign-institutional flow.

    A ticker passes on `date` when BOTH hold, using data available by then:
      - its 20-day FI net flow ranks at or above `entry_pct` within its own
        history (self-relative, so a small cap is not compared to a mega cap)
      - its 5-day FI net flow is positive (money arriving now, not merely
        having arrived at some point in the past)
    """

    def __init__(self, flow_dir: str, entry_pct: float = 0.80,
                 lag_days: int = 1, min_history: int = 120):
        self.flow_dir = flow_dir
        self.entry_pct = entry_pct
        self.lag_days = int(lag_days)
        self.min_history = int(min_history)
        self._rank: dict[str, pd.Series] = {}
        self._f5: dict[str, pd.Series] = {}
        self.loaded = False

    def load(self, tickers: Iterable[str]) -> int:
        """Precompute per-ticker FI rank and 5d flow series. Returns count."""
        n = 0
        for t in {str(x).upper() for x in tickers}:
            path = os.path.join(self.flow_dir, f"{t}.parquet")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            if "date" not in df.columns or FI_COL not in df.columns:
                continue
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
            if len(df) < self.min_history:
                continue

            raw = df[FI_COL].astype(float).fillna(0.0)
            f20 = raw.rolling(20, min_periods=10).sum()
            rank = f20.expanding(min_periods=self.min_history).rank(pct=True)
            f5 = raw.rolling(5, min_periods=3).sum()

            # Publication lag: flow for day T is not usable until T+lag.
            if self.lag_days > 0:
                rank = rank.shift(self.lag_days)
                f5 = f5.shift(self.lag_days)

            idx = pd.DatetimeIndex(df["date"].values)
            self._rank[t] = pd.Series(rank.values, index=idx)
            self._f5[t] = pd.Series(f5.values, index=idx)
            n += 1
        self.loaded = True
        return n

    def _as_of(self, series: pd.Series, date) -> float:
        """Latest value at or before `date`. Never looks forward."""
        try:
            pos = series.index.searchsorted(pd.Timestamp(date), side="right") - 1
        except Exception:
            return np.nan
        if pos < 0:
            return np.nan
        v = series.iloc[pos]
        return float(v) if np.isfinite(v) else np.nan

    def passes(self, ticker: str, date) -> bool:
        """True if the gate admits this ticker. Tickers with NO flow coverage
        pass by default — the gate must never silently shrink the universe to
        only those stocks that happen to have flow data, which would introduce
        a coverage bias masquerading as a signal."""
        t = str(ticker).upper()
        if t not in self._rank:
            return True
        rk = self._as_of(self._rank[t], date)
        f5 = self._as_of(self._f5[t], date)
        if not np.isfinite(rk) or not np.isfinite(f5):
            return True          # insufficient history -> do not penalise
        return bool(rk >= self.entry_pct and f5 > 0)

    def filter(self, candidates: Sequence, date, key=lambda c: c[0],
               min_keep: int = 2) -> list:
        """Filter a candidate list, keeping order.

        If the gate would leave fewer than `min_keep`, the ORIGINAL list is
        returned untouched. A sector rotation strategy must still deploy
        capital; an empty sector is a much larger error than a slightly weaker
        candidate, and silently emptying the sector would also make results
        impossible to compare against the ungated baseline.
        """
        if not self.loaded or not candidates:
            return list(candidates)
        kept = [c for c in candidates if self.passes(key(c), date)]
        return kept if len(kept) >= min_keep else list(candidates)

    def coverage(self, tickers: Iterable[str]) -> dict:
        ts = {str(t).upper() for t in tickers}
        have = ts & set(self._rank)
        return {"requested": len(ts), "with_flow": len(have),
                "missing": len(ts - have)}
