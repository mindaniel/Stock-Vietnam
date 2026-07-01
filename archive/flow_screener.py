"""
flow_screener.py — screen all stocks for smart money accumulation signals.

Detects the institutional flow cycle for each stock:

  Phase 1  FRESH     score just crossed positive  ← best entry point
  Phase 2  ACCUM     score positive, price still flat / building
  Phase 3  MARKUP    score positive, price already running
  Phase 4  DIST      score turned negative, smart money selling
  Phase 5  MARKDOWN  score deeply negative

For each current accumulation candidate, shows historical stats:
how much price typically rises per cycle and how long it lasts.

Usage:
    python flow_screener.py              # yesterday's data, top 25
    python flow_screener.py 2026-06-15  # specific date
    python flow_screener.py --top 40    # more results
    python flow_screener.py --all       # show all phases including DIST/MARKDOWN
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

FLOW_DIR  = os.path.join(BASE_DIR, "data", "investor_flow")
PRICE_DIR = os.path.join(BASE_DIR, "data", "price")
SECTORS_F = os.path.join(BASE_DIR, "ticker_sectors.csv")

# ── Parameters ────────────────────────────────────────────────────────────────
SMOOTH_DAYS    = 3      # rolling average to reduce noise
ACCUM_THRESH   = 0.04   # score above this = accumulation
DIST_THRESH    = -0.04  # score below this = distribution
FRESH_DAYS     = 5      # "fresh" if score just crossed positive within N days
MIN_CYCLE_DAYS = 4      # ignore cycles shorter than this (noise)
MIN_MARKUP_PCT = 1.0    # ignore cycles with less than this % gain (noise)


# ── Score calculation ─────────────────────────────────────────────────────────

def add_scores(df):
    """
    Daily smart_score = (dom_institutional + foreign_institutional) / total_abs_flow.
    Range: -1.0 (all retail buying) to +1.0 (all institutional buying).
    Then apply rolling smoothing.
    """
    smart = df["to_chuc_trongnuoc_net"] + df["to_chuc_nuocngoai_net"]
    total = (
        df["tu_doanh_net"].abs()
        + df["ca_nhan_trongnuoc_net"].abs()
        + df["to_chuc_trongnuoc_net"].abs()
        + df["ca_nhan_nuocngoai_net"].abs()
        + df["to_chuc_nuocngoai_net"].abs()
    )
    df = df.copy()
    df["score_raw"]    = (smart / total.replace(0, np.nan)).fillna(0)
    df["smart_score"]  = df["score_raw"].rolling(SMOOTH_DAYS, min_periods=1).mean()
    return df


# ── Historical cycle extraction ───────────────────────────────────────────────

def extract_cycles(df):
    """
    Find past accumulation→markup→distribution episodes.
    Episode starts when smoothed score crosses ACCUM_THRESH from below,
    ends when it crosses DIST_THRESH from above.

    Returns list of dicts: start_date, end_date, gain_pct, days
    """
    df = df.sort_values("date").reset_index(drop=True)
    score  = df["smart_score"].values
    prices = df["close"].values

    cycles  = []
    in_ep   = False
    ep_start = None

    for i in range(1, len(df)):
        if not in_ep and score[i-1] < ACCUM_THRESH and score[i] >= ACCUM_THRESH:
            in_ep    = True
            ep_start = i
        elif in_ep and score[i-1] >= ACCUM_THRESH and score[i] < DIST_THRESH:
            days = i - ep_start
            p0   = prices[ep_start]
            p1   = prices[i]
            if days >= MIN_CYCLE_DAYS and p0 > 0:
                gain = (p1 - p0) / p0 * 100
                if gain >= MIN_MARKUP_PCT:
                    cycles.append({
                        "start":    df["date"].iloc[ep_start],
                        "end":      df["date"].iloc[i],
                        "days":     days,
                        "gain_pct": gain,
                        "p_start":  p0,
                        "p_end":    p1,
                    })
            in_ep = False

    return cycles


# ── Current phase detection ───────────────────────────────────────────────────

def current_phase(df, as_of):
    """
    Determine phase and meta-stats as of a given date.
    Returns dict: phase, score, days_in_phase, days_since_flip, price_chg_5d
    """
    sub = df[df["date"] <= as_of].tail(40)
    if len(sub) < 5:
        return None

    scores  = sub["smart_score"].values
    prices  = sub["close"].values
    score   = float(scores[-1])

    # How many consecutive days in the current region?
    days_in = 1
    for i in range(len(scores) - 2, -1, -1):
        same = (
            (score >= ACCUM_THRESH  and scores[i] >= ACCUM_THRESH) or
            (DIST_THRESH < score < ACCUM_THRESH and DIST_THRESH < scores[i] < ACCUM_THRESH) or
            (score <= DIST_THRESH   and scores[i] <= DIST_THRESH)
        )
        if same:
            days_in += 1
        else:
            break

    # Days since most recent flip INTO positive territory
    days_since_flip = None
    if score >= ACCUM_THRESH:
        for i in range(len(scores) - 2, -1, -1):
            if scores[i] < ACCUM_THRESH:
                days_since_flip = len(scores) - 1 - i
                break
        if days_since_flip is None:
            days_since_flip = days_in

    # 5-day price change (using flow parquet close as proxy)
    price_5d = None
    if len(sub) >= 6 and prices[-6] > 0:
        price_5d = (prices[-1] - prices[-6]) / prices[-6] * 100

    # Phase label
    if score >= ACCUM_THRESH:
        if days_since_flip is not None and days_since_flip <= FRESH_DAYS:
            phase = "FRESH"
        elif price_5d is not None and price_5d >= 3.0:
            phase = "MARKUP"
        else:
            phase = "ACCUM"
    elif score <= DIST_THRESH:
        phase = "DIST"
    else:
        phase = "NEUTRAL"

    return {
        "phase":          phase,
        "score":          score,
        "days_in":        days_in,
        "days_since_flip": days_since_flip,
        "price_5d":       price_5d,
    }


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_flow(ticker):
    fpath = os.path.join(FLOW_DIR, f"{ticker}.parquet")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_parquet(fpath)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return None


def load_sectors():
    if not os.path.exists(SECTORS_F):
        return {}
    try:
        df = pd.read_csv(SECTORS_F, encoding="utf-8-sig")
        col = next((c for c in df.columns if "sector" in c.lower() or "industry" in c.lower()), None)
        if col is None:
            return {}
        return dict(zip(df["ticker"].str.upper(), df[col]))
    except Exception:
        return {}


def get_current_price(ticker, as_of):
    """Price from price parquet (more accurate than flow parquet close)."""
    fpath = os.path.join(PRICE_DIR, f"{ticker}.parquet")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_parquet(fpath)
        col = "time" if "time" in df.columns else "date"
        sub = df[df[col].astype(str) <= as_of]
        return float(sub.iloc[-1]["close"]) if not sub.empty else None
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args  = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = sys.argv[1:]

    top_n    = 25
    for i, f in enumerate(flags):
        if f == "--top" and i + 1 < len(flags):
            try: top_n = int(flags[i + 1])
            except: pass

    show_all = "--all" in flags

    if args:
        as_of = args[0]
    else:
        d = datetime.date.today()
        # Use yesterday if market likely closed, else day before
        as_of = (d - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    sectors  = load_sectors()
    tickers  = sorted(f.replace(".parquet", "") for f in os.listdir(FLOW_DIR) if f.endswith(".parquet"))

    print(f"\nFlow Screener  |  as of {as_of}  |  {len(tickers)} tickers")
    print("Scanning...\n")

    rows = []
    for ticker in tickers:
        df = load_flow(ticker)
        if df is None or len(df) < 30:
            continue

        df   = add_scores(df)
        info = current_phase(df, as_of)
        if info is None:
            continue

        phase = info["phase"]
        if not show_all and phase not in ("FRESH", "ACCUM", "MARKUP"):
            continue

        # Historical cycles (only from data BEFORE current open cycle)
        flip_date = None
        if phase in ("FRESH", "ACCUM", "MARKUP") and info["days_since_flip"] is not None:
            sub = df[df["date"] <= as_of]
            if len(sub) >= info["days_since_flip"] + 1:
                flip_date = sub["date"].iloc[-(info["days_since_flip"] + 1)]
        hist = df[df["date"] < (flip_date or as_of)] if flip_date else df[df["date"] < as_of]
        cycles = extract_cycles(hist)

        price    = get_current_price(ticker, as_of)
        sector   = sectors.get(ticker, "")

        rows.append({
            "ticker":   ticker,
            "sector":   sector[:18] if sector else "—",
            "phase":    phase,
            "score":    info["score"],
            "days_in":  info["days_in"],
            "d_flip":   info["days_since_flip"] or 0,
            "price":    price,
            "p5d":      info["price_5d"],
            "n_cyc":    len(cycles),
            "avg_gain": np.mean([c["gain_pct"] for c in cycles]) if cycles else None,
            "avg_days": np.mean([c["days"]     for c in cycles]) if cycles else None,
            "best":     max((c["gain_pct"] for c in cycles), default=None),
        })

    if not rows:
        print("  No signals found.")
        return

    phase_order = {"FRESH": 0, "ACCUM": 1, "MARKUP": 2, "NEUTRAL": 3, "DIST": 4}
    out = (
        pd.DataFrame(rows)
        .assign(_ord=lambda d: d["phase"].map(phase_order).fillna(5))
        .sort_values(["_ord", "score"], ascending=[True, False])
        .head(top_n)
    )

    # ── Table header ──────────────────────────────────────────────────────────
    print(f"  {'#':<3} {'Ticker':<6}  {'Phase':<7}  {'Score':>7}  {'DaysIn':>6}  "
          f"{'Price':>6}  {'5d%':>6}  {'AvgGain':>8}  {'Best':>6}  {'Cyc':>4}  Sector")
    print("  " + "-" * 100)

    for rank, (_, r) in enumerate(out.iterrows(), 1):
        star      = "* " if r["phase"] == "FRESH" else "  "
        flip_tag  = f" (+{r['d_flip']}d)" if r["phase"] == "FRESH" else ""
        price_s   = f"{r['price']:>6.2f}"          if r["price"]    is not None else "     —"
        p5d_s     = f"{r['p5d']:>+6.1f}%"          if r["p5d"]      is not None else "      —"
        avgG_s    = f"{r['avg_gain']:>+7.1f}%" if (r["avg_gain"] is not None and not np.isnan(r["avg_gain"])) else "       -"
        best_s    = f"{r['best']:>+5.1f}%"  if (r["best"]     is not None and not np.isnan(r["best"]))     else "     -"
        cyc_s     = f"{r['n_cyc']:>4}"      if r["n_cyc"] > 0 else "   -"

        print(f"{star}{rank:<3} {r['ticker']:<6}  {r['phase']:<7}  {r['score']:>+7.3f}  "
              f"{r['days_in']:>6}  {price_s}  {p5d_s}  {avgG_s}  {best_s}  {cyc_s}  {r['sector']}{flip_tag}")

    print()
    print("  * FRESH  = score just turned positive (best entry window)")
    print("  ACCUM    = institutional buying, price not yet moved much")
    print("  MARKUP   = price already running (entry risk higher)")
    print()
    print("  Score    = (dom.institution + foreign.institution) / total_flow  (-1 to +1)")
    print("  AvgGain  = avg price rise across past completed accumulation cycles")
    print("  Best     = best single cycle gain in history")
    print("  Cyc      = number of historical cycles found (higher = more reliable)")
    print()


if __name__ == "__main__":
    main()
