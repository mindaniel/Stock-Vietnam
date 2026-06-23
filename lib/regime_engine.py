"""
=============================================================================
REGIME ENGINE  —  Adaptive market regime scoring for Vietnamese stocks
=============================================================================

Produces a composite regime score (0–100) from four PRICE-BASED sources.
No sector blocking — when deposit rates rise all sectors lose retail buyers,
so the right response is to scale position size uniformly, not pick winners.

  Component                  Weight   Data source
  ─────────────────────────  ──────   ──────────────────────────────────────
  Bank relative strength       35 pts  data/{BANK}.csv vs VNINDEX.csv  (60d)
  VNINDEX trend                30 pts  VNINDEX.csv (MA crossover + ROC)
  Market breadth               25 pts  data/*.csv (% stocks above 20d MA)
  Foreign flow                 10 pts  data/*.csv (foreign_buy - foreign_sell)

  Bank ROA trend removed — quarterly data is too slow; price already
  reflects it.  All signals are now same-day price data.

Regime thresholds  (multiplier scales ALL sector capital equally):
  75–100 → BULL       multiplier 1.00
  50– 74 → NEUTRAL    multiplier 0.70
  30– 49 → CAUTION    multiplier 0.40
  15– 29 → DEFENSIVE  multiplier 0.20
   0– 14 → PANIC      multiplier 0.05

Integration with 4sectors.py:
  from regime_engine import compute_regime

  regime = compute_regime(as_of="2025-10-01")
  # regime["score"]      → 55
  # regime["label"]      → "NEUTRAL"
  # regime["multiplier"] → 0.70
  # regime["details"]    → sub-scores dict

  Use regime["multiplier"] to scale capital deployed per trade.

Standalone usage:
  python regime_engine.py                      # today
  python regime_engine.py --date 2025-10-01   # historical snapshot
  python regime_engine.py --history 2018-01-01 2026-01-01  # full series
=============================================================================
"""

import os
import sys
import glob
import argparse
import datetime as dt
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
DATA_DIR         = os.path.join(_HERE, "data", "price")
VNINDEX_PATH     = os.path.join(_HERE, "VNINDEX.csv")
FINANCIALS_FA    = os.path.join(_HERE, "data", "financials_fa")

# Bank basket — HOSE large-caps; used for bank relative strength signal
BANK_TICKERS = [
    "VCB", "BID", "CTG", "MBB", "ACB", "TCB", "VPB",
    "HDB", "STB", "TPB", "LPB", "VIB", "MSB", "OCB", "SHB",
]

# ─────────────────────────────────────────────────────────────────
# REGIME MAP
# ─────────────────────────────────────────────────────────────────
REGIME_MAP = [
    (75, "BULL",      1.00),
    (50, "NEUTRAL",   0.70),
    (30, "CAUTION",   0.40),
    (15, "DEFENSIVE", 0.20),
    ( 0, "PANIC",     0.05),
]

def _label_from_score(score: float) -> tuple:
    """Return (label, multiplier) for a given score."""
    for threshold, label, mult in REGIME_MAP:
        if score >= threshold:
            return label, mult
    return "PANIC", 0.05


# ─────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────

def _load_vnindex() -> pd.DataFrame:
    vn = pd.read_csv(VNINDEX_PATH)
    vn.columns = [c.strip().lower() for c in vn.columns]
    vn["date"] = pd.to_datetime(vn["date"], errors="coerce")
    vn = vn.dropna(subset=["date"]).sort_values("date").set_index("date")
    return vn[["close"]].astype(float)


def _load_stock(ticker: str) -> pd.DataFrame | None:
    fpath = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_parquet(fpath)
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = "time" if "time" in df.columns else "date"
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
        return df
    except Exception:
        return None


def _load_all_stocks(sample_cap: int = 200) -> dict[str, pd.DataFrame]:
    """
    Load all stock CSVs from DATA_DIR.  Bank tickers are always included;
    remaining slots filled with a random sample up to sample_cap.
    Returns dict: ticker → DataFrame with close + foreign cols.
    """
    files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    bank_files = {os.path.splitext(os.path.basename(f))[0].upper(): f
                  for f in files if os.path.splitext(os.path.basename(f))[0].upper() in BANK_TICKERS}
    other_files = [f for f in files
                   if os.path.splitext(os.path.basename(f))[0].upper() not in BANK_TICKERS]
    np.random.seed(42)
    n_other = max(0, sample_cap - len(bank_files))
    if len(other_files) > n_other:
        other_files = list(np.random.choice(other_files, n_other, replace=False))
    selected = list(bank_files.values()) + other_files

    result = {}
    for fpath in selected:
        ticker = os.path.splitext(os.path.basename(fpath))[0].upper()
        try:
            df = pd.read_parquet(fpath)
            df.columns = [c.strip().lower() for c in df.columns]
            date_col = "time" if "time" in df.columns else "date"
            df["date"] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
            if "close" in df.columns:
                result[ticker] = df
        except Exception:
            pass
    return result


# ─────────────────────────────────────────────────────────────────
# RAW COMPONENT SERIES  (vectorised, computed once over full history)
# ─────────────────────────────────────────────────────────────────
# Each function returns a daily pd.Series of raw values (not 0-100).
# Values are later converted to percentile ranks — no fixed thresholds.

def _raw_bank_rs(vn: pd.DataFrame, stock_data: dict, window: int = 60) -> pd.Series:
    """Bank basket median `window`-day return minus VNINDEX `window`-day return."""
    bank_rets = []
    for tkr in BANK_TICKERS:
        sd = stock_data.get(tkr)
        if sd is None or "close" not in sd.columns:
            continue
        r = sd["close"].pct_change(window)
        bank_rets.append(r)
    if not bank_rets:
        return pd.Series(dtype=float)
    basket = pd.concat(bank_rets, axis=1).median(axis=1)
    vn_ret = vn["close"].pct_change(window)
    return (basket - vn_ret).rename("bank_rs")


def _raw_vni_momentum(vn: pd.DataFrame, window: int = 60) -> pd.Series:
    """VNINDEX `window`-day rate of change."""
    return vn["close"].pct_change(window).rename("vni_mom")


def _raw_breadth(stock_data: dict, ma_window: int = 20) -> pd.Series:
    """Fraction of stocks with close > `ma_window`-day MA, daily."""
    above_series = []
    for tkr, df in stock_data.items():
        if "close" not in df.columns:
            continue
        ma = df["close"].rolling(ma_window, min_periods=int(ma_window * 0.7)).mean()
        above_series.append((df["close"] > ma).astype(float))
    if not above_series:
        return pd.Series(dtype=float)
    combined = pd.concat(above_series, axis=1)
    return combined.mean(axis=1).rename("breadth")


def _raw_foreign_flow(stock_data: dict, window: int = 20) -> pd.Series:
    """Median of per-stock net-foreign-buy / avg-turnover over `window` days."""
    net_series = []
    for tkr, df in stock_data.items():
        buy_col  = ("foreignbuy"  if "foreignbuy"  in df.columns else
                    "foreign_buy" if "foreign_buy" in df.columns else None)
        sell_col = ("foreignsell" if "foreignsell" in df.columns else
                    "foreign_sell" if "foreign_sell" in df.columns else None)
        if buy_col is None or sell_col is None:
            continue
        net      = (df[buy_col] - df[sell_col]).rolling(window).sum()
        turnover = (df[buy_col] + df[sell_col]).rolling(60, min_periods=20).mean().abs()
        normalised = net / turnover.replace(0, np.nan)
        net_series.append(normalised)
    if not net_series:
        return pd.Series(dtype=float)
    combined = pd.concat(net_series, axis=1)
    return combined.median(axis=1).rename("ff")


# ─────────────────────────────────────────────────────────────────
# PERCENTILE RANK  (the only "scoring" function)
# ─────────────────────────────────────────────────────────────────

def _pct_rank(series: pd.Series, lookback: int = 252, min_periods: int = 60) -> pd.Series:
    """
    For each day, score the value as its percentile within the trailing
    `lookback` window of its own history.  Score 70 = better than 70%
    of the past year.  No fixed thresholds, fully self-normalising.
    """
    def _rank(x):
        if len(x) < 2:
            return 50.0
        return float((x[:-1] < x[-1]).sum() / (len(x) - 1) * 100)
    return series.rolling(lookback, min_periods=min_periods).apply(_rank, raw=True)


# ─────────────────────────────────────────────────────────────────
# BUILD REGIME SERIES  (primary API for 4sectors.py backtest)
# ─────────────────────────────────────────────────────────────────

def build_regime_series(
    vn:         pd.DataFrame,
    stock_data: dict,
    lookback:   int = 252,
) -> pd.DataFrame:
    """
    Pre-compute daily regime scores for the full history.
    Call once at backtest startup; look up score per day in O(1).

    Returns DataFrame indexed by date with columns:
      bank_rs | vni_mom | breadth | ff          ← raw values
      bank_rs_rank | vni_rank | breadth_rank | ff_rank  ← percentile ranks 0-100
      score    ← equal-weight average of the four ranks
      label    ← BULL / NEUTRAL / CAUTION / DEFENSIVE / PANIC
      is_panic ← bool

    Equal weights (25% each) — no parameters to tune.
    """
    print("  Building regime series (vectorised)...")
    raw_bank = _raw_bank_rs(vn, stock_data)
    raw_vni  = _raw_vni_momentum(vn)
    raw_br   = _raw_breadth(stock_data)
    raw_ff   = _raw_foreign_flow(stock_data)

    rank_bank = _pct_rank(raw_bank, lookback)
    rank_vni  = _pct_rank(raw_vni,  lookback)
    rank_br   = _pct_rank(raw_br,   lookback)
    rank_ff   = _pct_rank(raw_ff,   lookback)

    score = pd.concat([rank_bank, rank_vni, rank_br, rank_ff], axis=1).mean(axis=1)

    labels = score.apply(lambda s: _label_from_score(s if not np.isnan(s) else 50.0)[0])
    is_panic = labels == "PANIC"

    df = pd.DataFrame({
        "bank_rs":      raw_bank,
        "vni_mom":      raw_vni,
        "breadth":      raw_br,
        "ff":           raw_ff,
        "bank_rs_rank": rank_bank,
        "vni_rank":     rank_vni,
        "breadth_rank": rank_br,
        "ff_rank":      rank_ff,
        "score":        score.round(1),
        "label":        labels,
        "is_panic":     is_panic,
    })
    n_panic = is_panic.sum()
    print(f"  Regime series built — {len(df)} days, {n_panic} PANIC days")
    return df


# ─────────────────────────────────────────────────────────────────
# SINGLE-DATE SNAPSHOT  (for standalone CLI / daily monitoring)
# ─────────────────────────────────────────────────────────────────

def compute_regime(
    as_of:      str | dt.date | None = None,
    stock_data: dict | None = None,
    bank_fa:    dict | None = None,   # unused, kept for API compatibility
    vn:         pd.DataFrame | None = None,
    lookback:   int = 252,
) -> dict:
    """
    Compute regime for a single date.  Builds the full series internally
    and extracts the requested date — suitable for CLI and daily alerts.
    For backtesting use build_regime_series() instead.
    """
    if as_of is None:
        as_of_ts = pd.Timestamp.today().normalize()
    elif isinstance(as_of, str):
        as_of_ts = pd.Timestamp(as_of)
    elif isinstance(as_of, (dt.date, dt.datetime)):
        as_of_ts = pd.Timestamp(as_of)
    else:
        as_of_ts = as_of

    if vn is None:
        vn = _load_vnindex()
    if stock_data is None:
        stock_data = _load_all_stocks()

    series = build_regime_series(vn, stock_data, lookback=lookback)

    # Find closest available date ≤ as_of
    available = series.index[series.index <= as_of_ts]
    if available.empty:
        row = None
    else:
        row = series.loc[available[-1]]

    if row is None or pd.isna(row["score"]):
        score, label, mult = 50.0, "NEUTRAL", 0.70
        details = {"bank_rs_rank": 50.0, "vni_rank": 50.0,
                   "breadth_rank": 50.0, "ff_rank": 50.0}
    else:
        score = float(row["score"])
        label, mult = _label_from_score(score)
        details = {
            "bank_rs_rank":  round(float(row["bank_rs_rank"]), 1),
            "vni_rank":      round(float(row["vni_rank"]),     1),
            "breadth_rank":  round(float(row["breadth_rank"]), 1),
            "ff_rank":       round(float(row["ff_rank"]),      1),
        }

    return {
        "as_of":           as_of_ts.strftime("%Y-%m-%d"),
        "score":           round(score, 1),
        "label":           label,
        "multiplier":      mult,
        "allowed_sectors": None,
        "details":         details,
    }


# ─────────────────────────────────────────────────────────────────
# TELEGRAM ALERT
# ─────────────────────────────────────────────────────────────────

def _send_telegram(message: str):
    import requests
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


def alert_regime_change(prev_label: str, new_regime: dict):
    """Send Telegram alert when regime changes."""
    emoji_map = {
        "BULL": "🟢", "NEUTRAL": "🟡", "CAUTION": "🟠",
        "DEFENSIVE": "🔴", "PANIC": "⚫",
    }
    new_label = new_regime["label"]
    if prev_label == new_label:
        return
    icon = emoji_map.get(new_label, "❓")
    prev = emoji_map.get(prev_label, "❓")
    d    = new_regime["details"]
    msg = (
        f"{icon} <b>REGIME CHANGE</b>: {prev} {prev_label} → {icon} {new_label}\n"
        f"Score: {new_regime['score']:.1f}/100\n\n"
        f"<i>Percentile ranks (vs past year):</i>\n"
        f"  Bank RS   : {d.get('bank_rs_rank', '?'):.0f}th pct\n"
        f"  VNI trend : {d.get('vni_rank', '?'):.0f}th pct\n"
        f"  Breadth   : {d.get('breadth_rank', '?'):.0f}th pct\n"
        f"  Foreign $ : {d.get('ff_rank', '?'):.0f}th pct"
    )
    _send_telegram(msg)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def _print_regime(r: dict):
    emoji_map = {
        "BULL": "🟢", "NEUTRAL": "🟡", "CAUTION": "🟠",
        "DEFENSIVE": "🔴", "PANIC": "⚫",
    }
    icon = emoji_map.get(r["label"], "❓")
    d    = r["details"]
    print(f"\n{'═'*54}")
    print(f"  REGIME  as of {r['as_of']}")
    print(f"{'═'*54}")
    print(f"  {icon}  {r['label']:<12}  score {r['score']:.1f}/100")
    print(f"  (score = equal-weight avg of 4 percentile ranks)")
    print(f"{'─'*54}")
    print(f"  Bank RS vs VNI  (60d spread, 252d rank)  {d.get('bank_rs_rank', 50.0):5.1f}th pct")
    print(f"  VNINDEX momentum (60d ROC, 252d rank)    {d.get('vni_rank',     50.0):5.1f}th pct")
    print(f"  Market breadth  (% >20dMA, 252d rank)   {d.get('breadth_rank', 50.0):5.1f}th pct")
    print(f"  Foreign flow    (20d net,  252d rank)    {d.get('ff_rank',      50.0):5.1f}th pct")
    print(f"{'═'*54}\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Market regime engine")
    parser.add_argument("--date",     default=None, help="Snapshot date YYYY-MM-DD")
    parser.add_argument("--history",  action="store_true",
                        help="Print annual regime distribution")
    parser.add_argument("--telegram", action="store_true",
                        help="Send result to Telegram")
    args = parser.parse_args()

    vn         = _load_vnindex()
    stock_data = _load_all_stocks()
    series     = build_regime_series(vn, stock_data)

    if args.history:
        monthly = series["label"].resample("ME").last()
        print("\nRegime distribution (daily):")
        counts = series["label"].value_counts()
        total  = len(series.dropna(subset=["score"]))
        for lbl, n in counts.items():
            print(f"  {lbl:<12} {n:>5} days  {n/total*100:5.1f}%")
        print("\nYear-by-year dominant regime:")
        for yr, grp in series.groupby(series.index.year):
            valid = grp.dropna(subset=["score"])
            if valid.empty:
                continue
            dom = valid["label"].value_counts().idxmax()
            counts_yr = valid["label"].value_counts()
            parts = "  ".join(f"{l}:{n}" for l, n in counts_yr.items())
            print(f"  {yr}  {dom:<12}  {parts}")
        out_path = os.path.join(_HERE, "regime_history.csv")
        series.to_csv(out_path)
        print(f"\nSaved → {out_path}")
        return

    r = compute_regime(as_of=args.date, vn=vn, stock_data=stock_data)
    _print_regime(r)

    if args.telegram:
        d    = r["details"]
        emoji_map = {"BULL": "🟢", "NEUTRAL": "🟡", "CAUTION": "🟠",
                     "DEFENSIVE": "🔴", "PANIC": "⚫"}
        icon = emoji_map.get(r["label"], "❓")
        msg = (
            f"{icon} <b>REGIME</b> {r['label']}  ({r['as_of']})\n"
            f"Score: {r['score']:.1f}/100\n"
            f"BankRS={d.get('bank_rs_rank',50):.0f}  VNI={d.get('vni_rank',50):.0f}  "
            f"Breadth={d.get('breadth_rank',50):.0f}  FF={d.get('ff_rank',50):.0f}"
        )
        _send_telegram(msg)


if __name__ == "__main__":
    main()
