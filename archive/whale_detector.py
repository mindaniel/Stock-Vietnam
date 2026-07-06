#!/usr/bin/env python3
"""
whale_detector.py
Scores each stock by real-time institutional-flow signals from today's data.
Run after daily_update.py. Outputs a ranked watchlist for the next trading day.

REVISED (see archive/tick_size_proxy_test.py and tick_size_band_scan.py):
Cross-validated against real domestic-institutional net flow (nguoiquansat,
May-Jun 2026 overlap, 149 tickers, ~7.9k ticker-days). Findings:
  - Tick-tape aggressor imbalance (buy-dominance, late-session ratio, large
    blocks) is NEGATIVELY correlated with real domestic institutional net
    flow at EVERY trade-size band tested (-0.08 to -0.23 pearson r,
    sign-agreement 13-44%, i.e. worse than a coin flip at the "large block"
    end). Domestic institutions in VN behave like patient/contrarian value
    buyers on passive limit orders — their accumulation shows up on days
    the tick tape looks like net SELLING (retail panic being absorbed).
  - Aggregate tick buy-dominance instead correlates with FOREIGN flow
    (r=+0.24) — already captured directly and better below via real
    foreign_buy_vol/sell_vol data, no tick proxy needed.
  - Put-through (block) deals are unvalidated and structurally the same
    trap as tick blocks: negotiated crossings are often ownership
    transfers / foreign-room arrangements, not organic conviction.
So: tick-tape "buy dominance"/"large blocks" and put-through scoring have
been removed. Weight is concentrated on the two directly-reported, real,
low-lag flows that DO have ground truth behind them: foreign net and tu
doanh (prop) net. Domestic institutional accumulation — the highest-weight,
best-predicting flow type elsewhere in this repo (lib/flow_signals.py) —
has no usable same-day tick proxy; it only shows up ~1-2 days later via
nguoiquansat (see .4Sectorlivesignals.py's distribution-alert tag).

Score breakdown (max 100):
  Foreign net           : 30 pts  (net vol relative to 20d avg volume)
  Foreign streak        : 10 pts  (consecutive net-positive days)
  Tu doanh net          : 30 pts  (broker proprietary net buy value)
  Volume surge          : 10 pts  (today vol vs 20d avg)
  Close position        : 10 pts  (close near high = strong demand)
  Tick buy-dominance    : 10 pts  (weak foreign-flow proxy only — NOT a
                                   domestic-institutional/"whale" signal)
"""

import os
import sys
import datetime as dt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file lives in archive/)
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR  = os.path.join(BASE_DIR, "data")
PRICE_DIR = os.path.join(DATA_DIR, "price")
TICK_DIR  = os.path.join(DATA_DIR, "tick_data")
TD_FILE  = os.path.join(BASE_DIR, "tudoanh", "tudoanh_all.csv")
VN_TZ    = dt.timezone(dt.timedelta(hours=7))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TOP_N    = 20   # candidates to display
WORKERS  = 12   # parallel threads for file I/O

# ─── Date helpers ─────────────────────────────────────────────────────────────

def _today_iso():
    """YYYY-MM-DD — used by price CSVs."""
    return dt.datetime.now(VN_TZ).strftime("%Y-%m-%d")

def _today_vn():
    """dd/mm/YYYY — used by tick parquets and tudoanh."""
    return dt.datetime.now(VN_TZ).strftime("%d/%m/%Y")

def _is_weekend():
    return dt.datetime.now(VN_TZ).weekday() >= 5

def _latest_tick_date() -> str:
    """Return the most recent date present in tick parquet files (dd/mm/YYYY)."""
    try:
        files = [f for f in os.listdir(TICK_DIR) if f.endswith(".parquet")][:20]
        dates = set()
        for f in files:
            df = pd.read_parquet(os.path.join(TICK_DIR, f))
            if "td" in df.columns and not df.empty:
                dates.add(df["td"].max())
        if not dates:
            return _today_vn()
        # dates are dd/mm/YYYY — parse and find max
        parsed = sorted(dates, key=lambda d: dt.datetime.strptime(d, "%d/%m/%Y"))
        return parsed[-1]
    except Exception:
        return _today_vn()


def _latest_price_date() -> str:
    """Return most recent date in price CSVs (YYYY-MM-DD)."""
    try:
        sample = [f for f in os.listdir(PRICE_DIR) if f.endswith(".parquet")][:10]
        dates = set()
        for f in sample:
            df = pd.read_parquet(os.path.join(PRICE_DIR, f), columns=["time"])
            if not df.empty:
                dates.add(df["time"].max())
        if not dates:
            return _today_iso()
        return max(dates)
    except Exception:
        return _today_iso()

# ─── Shared data loaders (called once) ────────────────────────────────────────

def _load_tudoanh(today_iso: str) -> pd.DataFrame:
    """today_iso: YYYY-MM-DD. tudoanh_all.csv stores dates in this format
    (not dd/mm/YYYY, despite the tick/tudoanh grouping elsewhere in this file)."""
    if not os.path.exists(TD_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(TD_FILE)
        today_rows = df[df["date"] == today_iso]
        return today_rows.set_index("symbol") if not today_rows.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ─── Per-symbol scorers ────────────────────────────────────────────────────────

def _score_tick(symbol: str, today_vn: str) -> tuple:
    """Max 10 pts from tick data. Weak FOREIGN-flow proxy only (r=+0.24 vs
    real foreign net) — NOT a domestic-institutional/"whale" signal (tick
    aggressor imbalance anti-correlates with real institutional net flow at
    every size band tested, see module docstring). "Large block" scoring
    removed entirely: sign-agreement with real institutional flow was
    13.9% at >=2B VND, i.e. actively wrong, not just noisy."""
    path = os.path.join(TICK_DIR, f"{symbol}.parquet")
    if not os.path.exists(path):
        return 0.0, {}
    try:
        df = pd.read_parquet(path)
        t = df[df["td"] == today_vn].copy()
        if len(t) < 5:
            return 0.0, {}

        # Data is sorted descending; row 0 = end-of-session cumulative totals
        tvb   = float(t.iloc[0]["tvb"])
        tvs   = float(t.iloc[0]["tvs"])
        total = tvb + tvs
        if total == 0:
            return 0.0, {}

        overall_ratio = tvb / total

        s = max(0.0, (overall_ratio - 0.5) / 0.5) * 10

        return round(s, 2), {
            "buy%": round(overall_ratio * 100, 1),
        }
    except Exception:
        return 0.0, {}


def _score_price_foreign(symbol: str, today_iso: str) -> tuple:
    """Max 60 pts from daily price CSV (foreign + volume + candle shape).
    Foreign net is a directly-reported, real, low-lag flow — not a tick
    proxy — and the one signal in this script with validated correlation
    to genuine institutional-side activity."""
    path = os.path.join(PRICE_DIR, f"{symbol}.parquet")
    if not os.path.exists(path):
        return 0.0, {}
    try:
        pdf = pd.read_parquet(path).sort_values("time").reset_index(drop=True)
        rows = pdf[pdf["time"] == today_iso]
        if rows.empty:
            return 0.0, {}
        row = rows.iloc[-1]

        fb    = float(row.get("foreign_buy_vol",  0) or 0)
        fs    = float(row.get("foreign_sell_vol", 0) or 0)
        f_net = fb - fs

        # Streak: consecutive net-positive foreign sessions before today
        past   = pdf[pdf["time"] < today_iso].tail(10)
        streak = 0
        for _, r in past.iloc[::-1].iterrows():
            rb = float(r.get("foreign_buy_vol",  0) or 0)
            rs = float(r.get("foreign_sell_vol", 0) or 0)
            if rb - rs > 0:
                streak += 1
            else:
                break

        avg_vol      = pdf["volume"].tail(20).mean() if "volume" in pdf.columns else 0
        f_net_ratio  = max(0.0, f_net / avg_vol) if avg_vol > 0 else 0.0
        vol          = float(row.get("volume", 0) or 0)
        vol_ratio    = vol / avg_vol if avg_vol > 0 else 1.0

        close = float(row.get("close", 0) or 0)
        high  = float(row.get("high",  0) or 0)
        low   = float(row.get("low",   0) or 0)
        rng   = high - low
        close_pos = (close - low) / rng if rng > 0 else 0.5

        s  = min(30.0, f_net_ratio * 30)
        s += min(10.0, float(streak))
        s += min(10.0, max(0.0, (vol_ratio - 1) / 4) * 10)
        s += close_pos * 10

        return round(s, 2), {
            "f_net_K": int(f_net / 1000),
            "f_streak": streak,
            "vol_x":   round(vol_ratio, 1),
            "close_pos": round(close_pos, 2),
        }
    except Exception:
        return 0.0, {}


def _score_tudoanh(symbol: str, td_df: pd.DataFrame) -> tuple:
    """Max 30 pts from proprietary trader (tu doanh) net buy — directly-
    reported, real, low-lag flow (not a tick proxy)."""
    if td_df.empty or symbol not in td_df.index:
        return 0.0, {}
    try:
        row     = td_df.loc[symbol]
        net_val = float(row.get("net_value",  0) or 0)
        net_vol = float(row.get("net_volume", 0) or 0)
        if net_val <= 0:
            return 0.0, {}
        score = min(30.0, (net_val / 1e9) * 6)  # 6 pts per 1 B VND net
        return round(score, 2), {
            "td_net_B": round(net_val / 1e9, 2),
            "td_net_K": int(net_vol / 1000),
        }
    except Exception:
        return 0.0, {}


# NOTE: put-through (block deal) scoring removed. Unvalidated, and
# structurally the same trap as tick large-blocks (see module docstring):
# negotiated crossings are often ownership transfers or foreign-room
# arrangements, not organic buying conviction.

# ─── Combined per-symbol analysis ─────────────────────────────────────────────

def _analyze(args):
    symbol, today_iso, today_vn, td_df = args
    s1, d1 = _score_tick(symbol, today_vn)
    s2, d2 = _score_price_foreign(symbol, today_iso)
    s3, d3 = _score_tudoanh(symbol, td_df)
    total  = round(s1 + s2 + s3, 1)
    return symbol, total, {**d1, **d2, **d3}

# ─── Output formatting ─────────────────────────────────────────────────────────

def _fmt(rank: int, symbol: str, score: float, d: dict) -> str:
    tags = []
    if d.get("f_net_K"):
        tags.append(f"foreign={d['f_net_K']:+,}K")
    if d.get("f_streak"):
        tags.append(f"streak={d['f_streak']}d")
    if d.get("td_net_B") is not None:
        tags.append(f"td={d['td_net_B']:+.1f}B")
    if d.get("vol_x"):
        tags.append(f"vol={d['vol_x']}x")
    if d.get("buy%"):
        tags.append(f"tick={d['buy%']}%buy (weak foreign proxy)")
    signal_str = "  ".join(tags) if tags else "—"
    return f"#{rank:<2} {symbol:<6} score={score:<6} {signal_str}"

# ─── Telegram ─────────────────────────────────────────────────────────────────

def _send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": int(TELEGRAM_CHAT_ID), "text": msg, "parse_mode": "HTML"},
            timeout=15,
        )
    except Exception as e:
        print(f"Telegram error: {e}")

# ─── Programmatic API (called by daily_update.py) ─────────────────────────────

def run_analysis(today_iso: str, today_vn: str, top_n: int = TOP_N) -> list:
    """
    Run whale analysis for the given dates. Returns sorted list of
    (symbol, score, details_dict) — highest score first.

    today_iso : YYYY-MM-DD  (price CSVs)
    today_vn  : dd/mm/YYYY  (tick parquets, tudoanh)
    """
    if not os.path.exists(TICK_DIR):
        print("   No tick_data dir — skipping whale analysis.")
        return []

    td_df = _load_tudoanh(today_iso)

    symbols = [f.replace(".parquet", "") for f in os.listdir(TICK_DIR) if f.endswith(".parquet")]
    print(f"   Whale scan: {len(symbols)} symbols | tudoanh={len(td_df)}")

    tasks   = [(sym, today_iso, today_vn, td_df) for sym in symbols]
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_analyze, t): t[0] for t in tasks}
        for future in as_completed(futures):
            sym, score, details = future.result()
            if score > 5:
                results.append((sym, score, details))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]

# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Foreign + prop-trading real-time flow detector "
                    "(NOT domestic-institutional — see module docstring)")
    parser.add_argument("--date", help="Override analysis date (YYYY-MM-DD).")
    args = parser.parse_args()

    if _is_weekend() and not args.date:
        print("Weekend — no trading data. Exiting.")
        return

    if args.date:
        try:
            d = dt.datetime.strptime(args.date, "%Y-%m-%d")
            today_iso = args.date
            today_vn  = d.strftime("%d/%m/%Y")
        except ValueError:
            print(f"Invalid date format: {args.date}. Use YYYY-MM-DD.")
            return
    else:
        today_vn  = _latest_tick_date()
        today_iso = _latest_price_date()
        try:
            gap = abs((dt.datetime.strptime(today_iso, "%Y-%m-%d") -
                       dt.datetime.strptime(today_vn,  "%d/%m/%Y")).days)
            if gap > 5:
                print(f"⚠️  Tick data ({today_vn}) is {gap} days behind price data ({today_iso}).")
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"  FOREIGN/PROP FLOW DETECTOR  |  tick={today_vn}  price={today_iso}")
    print(f"{'='*60}\n")

    top = run_analysis(today_iso, today_vn)

    if not top:
        print("No signals detected today.")
        return

    print(f"\nTOP {len(top)} WATCHLIST — foreign/prop buying, next session\n")
    print(f"{'#':<4} {'Symbol':<8} {'Score':<8} Signals")
    print("-" * 70)
    for i, (sym, score, d) in enumerate(top, 1):
        print(_fmt(i, sym, score, d))
    print(f"\nScore guide: 70+ strong | 50-69 moderate | 30-49 watch")
    print("This tracks FOREIGN + PROP (tu doanh) flow only — NOT domestic")
    print("institutional accumulation, which has no same-day proxy (see")
    print(".4Sectorlivesignals.py's flow distribution-alert tag instead).")
    print("Always verify with chart + market context before buying.\n")

    tg_lines = [f"📊 FOREIGN/PROP FLOW WATCHLIST {today_iso}"]
    for i, (sym, score, d) in enumerate(top[:10], 1):
        tg_lines.append(_fmt(i, sym, score, d))
    tg_lines.append("\n⚠️ Screener only — confirm with chart before buying.")
    _send_telegram("\n".join(tg_lines))
    print("Done.")


if __name__ == "__main__":
    main()
