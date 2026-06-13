#!/usr/bin/env python3
"""
whale_detector.py
Scores each stock by institutional/whale buying signals from today's data.
Run after daily_update.py. Outputs a ranked watchlist for the next trading day.

Score breakdown (max 100):
  Tick buy dominance    : 15 pts  (overall tvb/tvs ratio)
  Tick late session     : 15 pts  (14:00+ buy ratio — smart money time)
  Tick large blocks     : 20 pts  (v > 3x avg tick, buy side)
  Foreign net           : 15 pts  (net vol relative to 20d avg volume)
  Foreign streak        :  5 pts  (consecutive net-positive days)
  Volume surge          :  3 pts  (today vol vs 20d avg)
  Close position        :  2 pts  (close near high = strong demand)
  Tu doanh net          : 15 pts  (broker proprietary net buy value)
  Put-through deals     : 10 pts  (block deal count + value)
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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR  = os.path.join(BASE_DIR, "data")
PRICE_DIR = os.path.join(DATA_DIR, "price")
TICK_DIR  = os.path.join(DATA_DIR, "tick_data")
TD_FILE  = os.path.join(BASE_DIR, "tudoanh", "tudoanh_all.csv")
PT_FILE  = os.path.join(BASE_DIR, "putthrough", "putthrough_hose_all.csv")
VN_TZ    = dt.timezone(dt.timedelta(hours=7))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TOP_N    = 20   # candidates to display
WORKERS  = 12   # parallel threads for file I/O

# ─── Date helpers ─────────────────────────────────────────────────────────────

def _today_iso():
    """YYYY-MM-DD — used by price CSVs and putthrough."""
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

def _load_tudoanh(today_vn: str) -> pd.DataFrame:
    if not os.path.exists(TD_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(TD_FILE)
        today_rows = df[df["date"] == today_vn]
        return today_rows.set_index("symbol") if not today_rows.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _load_putthrough(today_iso: str) -> pd.DataFrame:
    if not os.path.exists(PT_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(PT_FILE)
        return df[df["date"] == today_iso].copy()
    except Exception:
        return pd.DataFrame()

# ─── Per-symbol scorers ────────────────────────────────────────────────────────

def _score_tick(symbol: str, today_vn: str) -> tuple:
    """Max 50 pts from tick data."""
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

        # Late session: 14:00 onwards (smart money moves last)
        late = t[t["t"] >= "14:00:00"]
        if not late.empty:
            lb = late[late["s"] == "buy"]["v"].sum()
            ls = late[late["s"] == "sell"]["v"].sum()
            late_ratio = lb / (lb + ls) if (lb + ls) > 0 else overall_ratio
        else:
            late_ratio = overall_ratio

        # Large block buys: single trade v > 3× avg tick, buy side
        avg_v     = t["v"].mean()
        threshold = max(avg_v * 3, 50_000)
        big_buys  = t[(t["v"] >= threshold) & (t["s"] == "buy")]
        day_vol   = t["v"].sum()
        big_ratio = big_buys["v"].sum() / day_vol if day_vol > 0 else 0.0

        s  = max(0.0, (overall_ratio - 0.5) / 0.5) * 15
        s += max(0.0, (late_ratio    - 0.5) / 0.5) * 15
        s += min(1.0, big_ratio * 3) * 20  # 33%+ large buys = max score

        return round(s, 2), {
            "buy%":      round(overall_ratio * 100, 1),
            "late_buy%": round(late_ratio    * 100, 1),
            "block_buy%":round(big_ratio     * 100, 1),
            "n_blocks":  len(big_buys),
        }
    except Exception:
        return 0.0, {}


def _score_price_foreign(symbol: str, today_iso: str) -> tuple:
    """Max 25 pts from daily price CSV (foreign + volume + candle shape)."""
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

        s  = min(15.0, f_net_ratio * 15)
        s += min(5.0,  float(streak))
        s += min(3.0,  max(0.0, (vol_ratio - 1) / 4) * 3)
        s += close_pos * 2

        return round(s, 2), {
            "f_net_K": int(f_net / 1000),
            "f_streak": streak,
            "vol_x":   round(vol_ratio, 1),
            "close_pos": round(close_pos, 2),
        }
    except Exception:
        return 0.0, {}


def _score_tudoanh(symbol: str, td_df: pd.DataFrame) -> tuple:
    """Max 15 pts from proprietary trader (tu doanh) net buy."""
    if td_df.empty or symbol not in td_df.index:
        return 0.0, {}
    try:
        row     = td_df.loc[symbol]
        net_val = float(row.get("net_value",  0) or 0)
        net_vol = float(row.get("net_volume", 0) or 0)
        if net_val <= 0:
            return 0.0, {}
        score = min(15.0, (net_val / 1e9) * 3)  # 3 pts per 1 B VND net
        return round(score, 2), {
            "td_net_B": round(net_val / 1e9, 2),
            "td_net_K": int(net_vol / 1000),
        }
    except Exception:
        return 0.0, {}


def _score_putthrough(symbol: str, pt_df: pd.DataFrame) -> tuple:
    """Max 10 pts from block (put-through) deals."""
    if pt_df.empty:
        return 0.0, {}
    try:
        deals = pt_df[pt_df["symbol"] == symbol]
        if deals.empty:
            return 0.0, {}
        val   = float(deals["value"].sum())
        n     = len(deals)
        score = min(10.0, (val / 1e9) * 2 + n * 0.5)
        return round(score, 2), {
            "pt_deals": n,
            "pt_val_B": round(val / 1e9, 2),
        }
    except Exception:
        return 0.0, {}

# ─── Combined per-symbol analysis ─────────────────────────────────────────────

def _analyze(args):
    symbol, today_iso, today_vn, td_df, pt_df = args
    s1, d1 = _score_tick(symbol, today_vn)
    s2, d2 = _score_price_foreign(symbol, today_iso)
    s3, d3 = _score_tudoanh(symbol, td_df)
    s4, d4 = _score_putthrough(symbol, pt_df)
    total  = round(s1 + s2 + s3 + s4, 1)
    return symbol, total, {**d1, **d2, **d3, **d4}

# ─── Output formatting ─────────────────────────────────────────────────────────

def _fmt(rank: int, symbol: str, score: float, d: dict) -> str:
    tags = []
    if d.get("buy%"):
        tags.append(f"tick={d['buy%']}%buy")
    if d.get("late_buy%"):
        tags.append(f"late={d['late_buy%']}%")
    if d.get("n_blocks"):
        tags.append(f"blocks={d['n_blocks']}({d.get('block_buy%',0)}%)")
    if d.get("f_net_K"):
        tags.append(f"foreign={d['f_net_K']:+,}K")
    if d.get("f_streak"):
        tags.append(f"streak={d['f_streak']}d")
    if d.get("td_net_B") is not None:
        tags.append(f"td={d['td_net_B']:+.1f}B")
    if d.get("pt_deals"):
        tags.append(f"block={d['pt_deals']}x/{d['pt_val_B']:.1f}B")
    if d.get("vol_x"):
        tags.append(f"vol={d['vol_x']}x")
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

    today_iso : YYYY-MM-DD  (price CSVs, putthrough)
    today_vn  : dd/mm/YYYY  (tick parquets, tudoanh)
    """
    if not os.path.exists(TICK_DIR):
        print("   No tick_data dir — skipping whale analysis.")
        return []

    td_vn = dt.datetime.strptime(today_iso, "%Y-%m-%d").strftime("%d/%m/%Y")
    td_df = _load_tudoanh(td_vn)
    pt_df = _load_putthrough(today_iso)

    symbols = [f.replace(".parquet", "") for f in os.listdir(TICK_DIR) if f.endswith(".parquet")]
    print(f"   Whale scan: {len(symbols)} symbols | tudoanh={len(td_df)} | putthrough={len(pt_df)}")

    tasks   = [(sym, today_iso, today_vn, td_df, pt_df) for sym in symbols]
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
    parser = argparse.ArgumentParser(description="Whale/institutional buying detector")
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
    print(f"  WHALE DETECTOR  |  tick={today_vn}  price={today_iso}")
    print(f"{'='*60}\n")

    top = run_analysis(today_iso, today_vn)

    if not top:
        print("No whale signals detected today.")
        return

    print(f"\nTOP {len(top)} WHALE WATCHLIST — buy candidates for next session\n")
    print(f"{'#':<4} {'Symbol':<8} {'Score':<8} Signals")
    print("-" * 70)
    for i, (sym, score, d) in enumerate(top, 1):
        print(_fmt(i, sym, score, d))
    print(f"\nScore guide: 70+ strong | 50-69 moderate | 30-49 watch")
    print("Always verify with chart + market context before buying.\n")

    tg_lines = [f"🐋 WHALE WATCHLIST {today_iso}"]
    for i, (sym, score, d) in enumerate(top[:10], 1):
        tg_lines.append(_fmt(i, sym, score, d))
    tg_lines.append("\n⚠️ Screener only — confirm with chart before buying.")
    _send_telegram("\n".join(tg_lines))
    print("Done.")


if __name__ == "__main__":
    main()
