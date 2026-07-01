"""
Position management check — price + flow + tick
================================================
For each held stock, combines three independent signals:

  [price]  Daily OHLCV swing structure — HL/LL pivot + correction depth
  [flow]   NDT institutional net VND during the pullback phase
           (to_chuc_trongnuoc + tu_doanh; data from Sep 2024)
  [tick]   Buy vs sell volume from tick data (recent ~3 months only)

Final verdict per stock:
  STRONG ADD  — HL + institutions accumulating + net buy ticks
  ADD         — HL + at least one confirming signal
  HOLD        — mixed signals, no clear direction
  CAUTION     — HL price but institutions distributing, OR deep correction
  REDUCE      — LL confirmed + institutional selling
  STRONG REDUCE — LL + institutional selling + net sell ticks

Usage:
  python swing_4sector.py                 # all holdings in portfolio_state.json
  python swing_4sector.py FPT ACB VPB    # specific tickers
  python swing_4sector.py --detail        # show full flow + tick breakdown
"""

import pandas as pd
import numpy as np
import sys, argparse, json, glob
from pathlib import Path
from datetime import date, timedelta

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE_DIR   = Path(__file__).parent
PRICE_DIR  = BASE_DIR / "data" / "price"
FLOW_DIR   = BASE_DIR / "data" / "investor_flow"
TICK_DIR   = BASE_DIR / "data" / "tick_data"
STATE_FILE = BASE_DIR / "portfolio_state.json"

ORDER      = 5    # bars each side to confirm pivot
TICK_DAYS  = 5    # how many recent trading days to average for tick signal


# ── Data loaders ──────────────────────────────────────────────────────

def load_daily(ticker):
    path = PRICE_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["time","open","high","low","close","volume"])
    df["date"] = pd.to_datetime(df["time"], errors="coerce")
    df = (df.dropna(subset=["date"]).set_index("date")
            .rename(columns={"open":"Open","high":"High","low":"Low",
                              "close":"Close","volume":"Volume"})
            [["Open","High","Low","Close","Volume"]].sort_index().dropna())
    return df


def load_flow(ticker):
    """Load investor flow parquet. Returns df indexed by date, or None."""
    path = FLOW_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df


def load_tick(ticker):
    """
    Load tick data and aggregate to daily net buy/sell.
    tvb/tvs are cumulative within each day — take the last row per day.

    Returns df with date index and columns: net_vol (tvb-tvs), tvb, tvs.
    """
    path = TICK_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    t = pd.read_parquet(path)
    if "td" not in t.columns or "tvb" not in t.columns or "tvs" not in t.columns:
        return None
    t["date"] = pd.to_datetime(t["td"], dayfirst=True, errors="coerce")
    t = t.dropna(subset=["date"])
    # Last row per day = end-of-day cumulative totals
    daily = (t.sort_values("date").groupby(t["date"].dt.date)
               .last()[["tvb","tvs"]])
    daily.index = pd.to_datetime(daily.index)
    daily["net_vol"] = daily["tvb"] - daily["tvs"]
    return daily


# ── Pivot detection ────────────────────────────────────────────────────

def find_pivots(df, order=ORDER):
    h, l = df["High"].values, df["Low"].values
    highs, lows = [], []
    for i in range(order, len(df) - order):
        if h[i] == h[i-order:i+order+1].max(): highs.append(i)
        if l[i] == l[i-order:i+order+1].min(): lows.append(i)
    return highs, lows


def label_pivots(df, highs, lows):
    pts = ([(i, df["High"].iloc[i], "H") for i in highs]
         + [(i, df["Low"].iloc[i],  "L") for i in lows])
    pts.sort(key=lambda x: x[0])
    last_h = last_l = None
    labeled = []
    for idx, price, ptype in pts:
        if ptype == "H":
            lbl = "HH" if last_h and price > last_h else ("LH" if last_h else "H")
            last_h = price
        else:
            lbl = "HL" if last_l and price > last_l else ("LL" if last_l else "L")
            last_l = price
        labeled.append({"idx": idx, "price": price, "type": ptype,
                         "label": lbl, "date": df.index[idx]})
    return labeled


# ── Signal 1: price structure ──────────────────────────────────────────

def price_signal(ticker, df=None):
    """
    Returns dict with keys:
      setup       : STRONG / OK / DEEP / BROKEN  (HL quality)
      correction  : float %
      bars_down   : int
      trend       : str
      high_date   : pd.Timestamp of last swing high
      low_date    : pd.Timestamp of last swing low
      low_label   : HL / LL / H / L
      current     : current close price
    """
    if df is None:
        df = load_daily(ticker)
    if df is None or len(df) < ORDER * 2 + 10:
        return {"setup": "NO_DATA"}

    confirmed = df.iloc[:len(df) - ORDER]
    highs, lows = find_pivots(confirmed, ORDER)
    labeled = label_pivots(confirmed, highs, lows)

    H_pts = [p for p in labeled if p["type"] == "H"]
    L_pts = [p for p in labeled if p["type"] == "L"]
    if not H_pts or not L_pts:
        return {"setup": "NO_DATA"}

    last_H = H_pts[-1]
    last_L = L_pts[-1]

    hh = len(H_pts) >= 2 and H_pts[-1]["price"] > H_pts[-2]["price"]
    hl = len(L_pts) >= 2 and L_pts[-1]["price"] > L_pts[-2]["price"]
    lh = len(H_pts) >= 2 and H_pts[-1]["price"] < H_pts[-2]["price"]
    ll = len(L_pts) >= 2 and L_pts[-1]["price"] < L_pts[-2]["price"]

    if hh and hl:   trend = "HH+HL"
    elif lh and ll: trend = "LH+LL"
    elif hh or hl:  trend = "HH/HL"
    elif lh or ll:  trend = "LH/LL"
    else:           trend = "mixed"

    ref_H_list = [p for p in H_pts if p["idx"] < last_L["idx"]]
    if not ref_H_list:
        return {"setup": "NO_DATA", "trend": trend}
    ref_H = ref_H_list[-1]

    correction = (ref_H["price"] - last_L["price"]) / ref_H["price"] * 100
    bars_down  = last_L["idx"] - ref_H["idx"]
    is_hl      = last_L["label"] in ("HL", "L")
    is_ll      = last_L["label"] == "LL"

    if is_ll or correction > 15:
        setup = "BROKEN"
    elif correction <= 5 and is_hl:
        setup = "STRONG"
    elif correction <= 10 and is_hl:
        setup = "OK"
    elif correction <= 15 and is_hl:
        setup = "DEEP"
    else:
        setup = "BROKEN"

    return {
        "setup":      setup,
        "correction": round(correction, 1),
        "bars_down":  bars_down,
        "trend":      trend,
        "low_label":  last_L["label"],
        "high_date":  ref_H["date"],
        "low_date":   last_L["date"],
        "high_price": round(ref_H["price"], 2),
        "low_price":  round(last_L["price"], 2),
        "current":    round(df["Close"].iloc[-1], 2),
    }


# ── Signal 2: NDT institutional flow during pullback ──────────────────

def flow_signal(ticker, high_date, low_date):
    """
    Cumulative institutional net VND during the pullback (high → low).
    Uses: to_chuc_trongnuoc_net + tu_doanh_net (domestic smart money).
    Also shows foreign institutional for context.

    Returns dict:
      dom_net     : domestic institutional cumulative net (VND bn)
      dom_days    : number of days with flow data in the window
      foreign_net : foreign institutional net
      direction   : BUYING / SELLING / NEUTRAL
      note        : '' or 'partial data' if window predates Sep-2024
    """
    df = load_flow(ticker)
    if df is None:
        return {"direction": "NO_DATA", "dom_net": None, "dom_days": 0}

    # Clamp to flow data range (starts Sep 2024)
    start = max(pd.Timestamp(high_date), df.index.min())
    end   = pd.Timestamp(low_date)

    if start > end:
        # Pullback predates all flow data — use whatever recent data exists
        recent = df.tail(20)
        start  = recent.index.min()
        end    = df.index.max()
        note   = "no flow in pullback window — showing recent 20d"
    else:
        note = ("partial: pullback started before flow data"
                if pd.Timestamp(high_date) < df.index.min() else "")

    window = df[(df.index >= start) & (df.index <= end)]
    if window.empty:
        return {"direction": "NO_DATA", "dom_net": None, "dom_days": 0, "note": note}

    dom_cols = []
    if "to_chuc_trongnuoc_net" in window.columns:
        dom_cols.append("to_chuc_trongnuoc_net")
    if "tu_doanh_net" in window.columns:
        dom_cols.append("tu_doanh_net")

    dom_net = float(window[dom_cols].sum().sum()) if dom_cols else 0.0
    foreign_net = 0.0
    if "to_chuc_nuocngoai_net" in window.columns:
        foreign_net = float(window["to_chuc_nuocngoai_net"].sum())

    dom_days = len(window)
    threshold = 5.0  # VND billion — treat ±5B as neutral noise
    direction = ("BUYING" if dom_net > threshold
                 else "SELLING" if dom_net < -threshold
                 else "NEUTRAL")

    return {
        "dom_net":     round(dom_net, 1),
        "foreign_net": round(foreign_net, 1),
        "dom_days":    dom_days,
        "direction":   direction,
        "note":        note,
        "window_start": start.date(),
        "window_end":   end.date(),
    }


# ── Signal 3: tick buy/sell balance ───────────────────────────────────

def tick_signal(ticker, n_days=TICK_DAYS):
    """
    Net buy/sell volume from tick data over the last n_days trading days.
    net_vol = tvb - tvs per day (cumulative end-of-day).

    Returns dict:
      net_vol_avg : average daily net volume (positive = net buy)
      net_vol_sum : total net over n_days
      days        : actual days of data
      direction   : BUYING / SELLING / NEUTRAL
    """
    df = load_tick(ticker)
    if df is None or df.empty:
        return {"direction": "NO_DATA", "net_vol_avg": None, "days": 0}

    recent = df.tail(n_days)
    if recent.empty:
        return {"direction": "NO_DATA", "net_vol_avg": None, "days": 0}

    net_sum = float(recent["net_vol"].sum())
    net_avg = float(recent["net_vol"].mean())
    days    = len(recent)

    # Threshold: 100k shares net per day is meaningful
    threshold = 100_000
    direction = ("BUYING"  if net_avg > threshold
                 else "SELLING" if net_avg < -threshold
                 else "NEUTRAL")

    return {
        "net_vol_avg": round(net_avg),
        "net_vol_sum": round(net_sum),
        "days":        days,
        "direction":   direction,
        "last_date":   recent.index[-1].date(),
    }


# ── Combined verdict ───────────────────────────────────────────────────

def combined_verdict(price, flow, tick):
    """
    Combine three signals into one action.

    Price:  STRONG / OK / DEEP / BROKEN / NO_DATA
    Flow:   BUYING / SELLING / NEUTRAL / NO_DATA
    Tick:   BUYING / SELLING / NEUTRAL / NO_DATA
    """
    p = price.get("setup", "NO_DATA")
    f = flow.get("direction",  "NO_DATA")
    t = tick.get("direction",  "NO_DATA")

    # Score each signal: +1 bullish, 0 neutral/missing, -1 bearish
    def score(sig):
        return 1 if sig == "BUYING" else (-1 if sig == "SELLING" else 0)

    pscore = {"STRONG": 2, "OK": 1, "DEEP": 0, "BROKEN": -2, "NO_DATA": 0}[p]
    fscore = score(f)
    tscore = score(t)
    total  = pscore + fscore + tscore

    if p == "NO_DATA":
        return "NO_DATA", "insufficient price history"

    if p in ("STRONG", "OK") and f == "BUYING" and t != "SELLING":
        return "STRONG ADD", f"HL ({price.get('correction',0):.0f}% corr) + institutions buying"
    if p in ("STRONG", "OK") and t == "BUYING" and f != "SELLING":
        return "ADD", f"HL ({price.get('correction',0):.0f}% corr) + tick net buy"
    if p in ("STRONG", "OK") and f == "SELLING":
        return "CAUTION", f"HL structure but institutions distributing"
    if p in ("STRONG", "OK"):
        return "HOLD", f"HL ({price.get('correction',0):.0f}% corr), flow/tick neutral"
    if p == "DEEP":
        if f == "BUYING":
            return "HOLD", f"Deep correction ({price.get('correction',0):.0f}%) but institutions buying"
        return "CAUTION", f"HL but deep correction ({price.get('correction',0):.0f}%)"
    if p == "BROKEN":
        if f == "SELLING" or t == "SELLING":
            return "STRONG REDUCE", f"LL ({price.get('low_label','?')}) + institutional selling"
        if f == "BUYING":
            return "HOLD", f"LL structure but institutions buying — mixed signal"
        return "REDUCE", f"LL ({price.get('low_label','?')}) confirmed — trend broken"

    return "HOLD", "mixed signals"


# ── Main portfolio check ───────────────────────────────────────────────

VERDICT_ICON = {
    "STRONG ADD":    "🟢🟢",
    "ADD":           "🟢  ",
    "HOLD":          "🟡  ",
    "CAUTION":       "🟠  ",
    "REDUCE":        "🔴  ",
    "STRONG REDUCE": "🔴🔴",
    "NO_DATA":       "⬜  ",
}

VERDICT_ORDER = {
    "STRONG REDUCE": 0,
    "REDUCE":        1,
    "CAUTION":       2,
    "HOLD":          3,
    "ADD":           4,
    "STRONG ADD":    5,
    "NO_DATA":       6,
}


def portfolio_check(tickers_override=None, detail=False):
    portfolio = {}
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding="utf-8") as f:
            state = json.load(f)
        portfolio = state.get("portfolio", {})
    if tickers_override:
        portfolio = {t: {} for t in tickers_override}
    if not portfolio:
        print("No portfolio found."); return

    print(f"\n{'═'*90}")
    print(f"  POSITION CHECK  —  {date.today()}")
    print(f"  Signals: [price] daily OHLCV pivots | [flow] NDT institutional (Sep24+) | [tick] buy-sell vol (Apr26+)")
    print(f"{'─'*90}")

    rows = []
    for tkr, pos in portfolio.items():
        df_price = load_daily(tkr)
        price = price_signal(tkr, df_price)
        ep    = pos.get("entry_price", 0)
        cur   = price.get("current", 0)
        pnl   = (cur - ep) / ep * 100 if ep else 0

        # Flow: use pullback window if available
        if price.get("setup") != "NO_DATA" and price.get("high_date") is not None:
            flow = flow_signal(tkr, price["high_date"], price["low_date"])
        else:
            flow = flow_signal(tkr,
                               date.today() - timedelta(days=30),
                               date.today())

        tick   = tick_signal(tkr)
        verdict, reason = combined_verdict(price, flow, tick)

        rows.append({
            "ticker":  tkr,
            "pos":     pos,
            "price":   price,
            "flow":    flow,
            "tick":    tick,
            "verdict": verdict,
            "reason":  reason,
            "pnl":     pnl,
            "ep":      ep,
        })

    rows.sort(key=lambda r: (VERDICT_ORDER.get(r["verdict"], 9), r["pnl"]))

    # ── Table ──────────────────────────────────────────────────────
    print(f"  {'':2} {'Tick':<5} {'Verdict':<13} {'P&L':>6} │ "
          f"{'[price]':<22} {'[flow]':<20} {'[tick]'}")
    print(f"  {'─'*88}")

    for r in rows:
        tkr     = r["ticker"]
        price   = r["price"]
        flow    = r["flow"]
        tick    = r["tick"]
        pnl_s   = f"{r['pnl']:>+5.1f}%" if r["ep"] else "    —"
        icon    = VERDICT_ICON.get(r["verdict"], "   ")

        # [price] column
        if price.get("setup") == "NO_DATA":
            p_col = "no data"
        else:
            p_col = (f"{price['low_label']} {price['correction']:.0f}% "
                     f"{price['trend'][:7]}")

        # [flow] column
        if flow.get("direction") == "NO_DATA":
            f_col = "no flow data"
        else:
            dn  = flow["dom_net"]
            sig = "▲" if flow["direction"]=="BUYING" else ("▼" if flow["direction"]=="SELLING" else "─")
            f_col = f"{sig} {dn:>+.0f}B dom ({flow['dom_days']}d)"

        # [tick] column
        if tick.get("direction") == "NO_DATA":
            t_col = "no tick data"
        else:
            avg = tick["net_vol_avg"]
            sig = "▲" if tick["direction"]=="BUYING" else ("▼" if tick["direction"]=="SELLING" else "─")
            t_col = f"{sig} {avg/1e6:>+.2f}M/d ({tick['days']}d)"

        print(f"  {icon} {tkr:<5} {r['verdict']:<13} {pnl_s} │ "
              f"{p_col:<22} {f_col:<20} {t_col}")

        if detail:
            print(f"       reason: {r['reason']}")
            if flow.get("foreign_net") is not None:
                print(f"       foreign inst: {flow['foreign_net']:>+.1f}B  "
                      f"window: {flow.get('window_start','')} → {flow.get('window_end','')}")
            if flow.get("note"):
                print(f"       note: {flow['note']}")

    # ── Summary by verdict ─────────────────────────────────────────
    print(f"\n  {'─'*88}")
    for v in ["STRONG REDUCE","REDUCE","CAUTION","HOLD","ADD","STRONG ADD"]:
        grp = [r for r in rows if r["verdict"] == v]
        if not grp: continue
        avg_pnl = np.mean([r["pnl"] for r in grp if r["ep"]])
        tickers = " ".join(r["ticker"] for r in grp)
        print(f"  {VERDICT_ICON[v]} {v:<13} ({len(grp):>2})  avg P&L {avg_pnl:>+5.1f}%  →  {tickers}")

    print(f"\n  Action guide:")
    print(f"    🔴🔴 STRONG REDUCE — LL confirmed + institutions distributing. Exit.")
    print(f"    🔴   REDUCE        — LL confirmed. Trend broken. Trim position.")
    print(f"    🟠   CAUTION       — HL price but flow warns of distribution. Hold, don't add.")
    print(f"    🟡   HOLD          — mixed or neutral signals. Wait.")
    print(f"    🟢   ADD           — HL + one confirming signal. Can add on dips.")
    print(f"    🟢🟢 STRONG ADD    — HL + institutions buying. Best add zone.")
    print(f"{'═'*90}")


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("tickers", nargs="*", help="Specific tickers to check")
    p.add_argument("--detail", action="store_true", help="Show full breakdown per stock")
    args = p.parse_args()
    tickers = [t.upper() for t in args.tickers] if args.tickers else None
    portfolio_check(tickers, detail=args.detail)


if __name__ == "__main__":
    main()
