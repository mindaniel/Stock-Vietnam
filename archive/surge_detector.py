#!/usr/bin/env python3
"""
surge_detector.py
-----------------
Screens for stocks likely to have a sudden NEXT-DAY surge.
Run after market close (after daily_update.py).

Score breakdown (max 100):
  Volume surge         : 20 pts  — today vol vs 20d avg
  Price breakout       : 15 pts  — close above N-day resistance
  Late accumulation    : 20 pts  — 14:00+ buy ratio (smart money window)
  Intraday accel       : 10 pts  — last-hour vol vs first-hour vol
  Consolidation break  : 10 pts  — tight N-day range exploding today
  Block clustering     : 10 pts  — large trades in last 2 hours
  Foreign momentum     :  8 pts  — net buy + consecutive-day streak
  Tu doanh net buy     :  7 pts  — proprietary desk positioning

Output: ranked table + "why" reason string + "when" (time of peak accumulation).
"""

import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "data", "price")
TICK_DIR = os.path.join(DATA_DIR, "tick_data")
TD_FILE  = os.path.join(BASE_DIR, "tudoanh", "tudoanh_all.csv")
PT_FILE  = os.path.join(BASE_DIR, "putthrough", "putthrough_hose_all.csv")
SM_FILE  = os.path.join(BASE_DIR, "sector_master.csv")
VN_TZ    = dt.timezone(dt.timedelta(hours=7))

TOP_N   = 25
WORKERS = 12

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# ─── Date helpers ──────────────────────────────────────────────────────────────

def _latest_tick_date() -> str:
    """Most recent date found across all tick files (dd/mm/YYYY)."""
    try:
        files = [f for f in os.listdir(TICK_DIR) if f.endswith(".parquet")]
        best_dt = None
        best_str = ""
        for f in files:
            df = pd.read_parquet(os.path.join(TICK_DIR, f), columns=["td"])
            if df.empty:
                continue
            for raw in df["td"].dropna().unique():
                try:
                    parsed = dt.datetime.strptime(raw, "%d/%m/%Y")
                    if best_dt is None or parsed > best_dt:
                        best_dt = parsed
                        best_str = raw
                except ValueError:
                    continue
        return best_str if best_str else dt.datetime.now(VN_TZ).strftime("%d/%m/%Y")
    except Exception:
        return dt.datetime.now(VN_TZ).strftime("%d/%m/%Y")


def _latest_price_date() -> str:
    """Most recent date in daily CSVs (YYYY-MM-DD)."""
    try:
        sample = [f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")][:15]
        dates = set()
        for f in sample:
            df = pd.read_parquet(os.path.join(DATA_DIR, f), columns=["time"])
            if not df.empty:
                dates.add(df["time"].max())
        return max(dates) if dates else dt.datetime.now(VN_TZ).strftime("%Y-%m-%d")
    except Exception:
        return dt.datetime.now(VN_TZ).strftime("%Y-%m-%d")


def _vn_to_iso(vn: str) -> str:
    """dd/mm/YYYY → YYYY-MM-DD"""
    return dt.datetime.strptime(vn, "%d/%m/%Y").strftime("%Y-%m-%d")


def _iso_to_vn(iso: str) -> str:
    """YYYY-MM-DD → dd/mm/YYYY"""
    return dt.datetime.strptime(iso, "%Y-%m-%d").strftime("%d/%m/%Y")

# ─── Shared loaders (called once for all symbols) ─────────────────────────────

def _load_tudoanh(today_vn: str) -> pd.DataFrame:
    if not os.path.exists(TD_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(TD_FILE, encoding="utf-8-sig")
        if "date" not in df.columns:
            return pd.DataFrame()
        # Normalise all dates to YYYY-MM-DD (handles both DD/MM/YYYY and YYYY-MM-DD)
        parsed = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
        mask   = parsed.isna()
        if mask.any():
            parsed[mask] = pd.to_datetime(df["date"][mask], format="%Y-%m-%d", errors="coerce")
        df["date"] = parsed.dt.strftime("%Y-%m-%d")
        today_iso = _vn_to_iso(today_vn)
        rows = df[df["date"] == today_iso]
        return rows.set_index("symbol") if not rows.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _load_sector_map() -> dict:
    try:
        df = pd.read_csv(SM_FILE, encoding="utf-8-sig")
        return df.set_index("symbol")["sector"].to_dict()
    except Exception:
        return {}

# ─── Signal 1 + 3 + 5: Price CSV signals ─────────────────────────────────────

def _score_price(symbol: str, today_iso: str) -> tuple:
    """
    Returns (score 0-53, detail_dict).
    Signals: volume surge, price breakout, consolidation-then-breakout,
             foreign momentum.
    """
    path = os.path.join(DATA_DIR, f"{symbol}.parquet")
    if not os.path.exists(path):
        return 0.0, {}
    try:
        pdf = pd.read_parquet(path).sort_values("time").reset_index(drop=True)
        if "time" not in pdf.columns or pdf.empty:
            return 0.0, {}

        # Use the last available row (handles lag between tick date and price date)
        row = pdf.iloc[-1]
        row_date = str(row.get("time", ""))

        close = float(row.get("close", 0) or 0)
        high  = float(row.get("high",  0) or 0)
        low   = float(row.get("low",   0) or 0)
        vol   = float(row.get("volume", 0) or 0)
        if close <= 0 or vol <= 0:
            return 0.0, {}

        past = pdf.iloc[:-1].tail(20)
        if len(past) < 5:
            return 0.0, {}

        avg_vol  = past["volume"].mean()
        vol_x    = vol / avg_vol if avg_vol > 0 else 1.0

        # ── Signal: Volume surge (0-20) ──────────────────────────────────────
        if vol_x >= 3.0:
            vol_score = 20.0
        elif vol_x >= 2.0:
            vol_score = 14.0
        elif vol_x >= 1.5:
            vol_score = 8.0
        elif vol_x >= 1.2:
            vol_score = 4.0
        else:
            vol_score = 0.0

        # ── Signal: Price breakout (0-15) ────────────────────────────────────
        high_20 = past["high"].max() if "high" in past.columns else close
        high_10 = past.tail(10)["high"].max() if "high" in past.columns else close
        high_5  = past.tail(5)["high"].max()  if "high" in past.columns else close

        if close > high_20:
            breakout_score = 15.0
            breakout_label = "20d-high breakout"
        elif close > high_10:
            breakout_score = 10.0
            breakout_label = "10d-high breakout"
        elif close > high_5:
            breakout_score = 5.0
            breakout_label = "5d-high breakout"
        else:
            breakout_score = 0.0
            breakout_label = ""

        # ── Signal: Tight consolidation → breakout (0-10) ────────────────────
        # Low daily range over last 5 days means coiling, explosion = breakout
        if "high" in past.columns and "low" in past.columns:
            last5       = past.tail(5)
            avg_range_5 = ((last5["high"] - last5["low"]) / last5["close"]).mean()
            today_range = (high - low) / close if close > 0 else 0
            consol_score = min(10.0, max(0.0, (today_range - avg_range_5) / avg_range_5 * 10))
            coiling = avg_range_5 < 0.015  # < 1.5% daily range = tight coil
        else:
            consol_score = 0.0
            coiling      = False

        # ── Signal: Foreign momentum (0-8) ────────────────────────────────────
        fb    = float(row.get("foreign_buy_vol",  0) or 0)
        fs    = float(row.get("foreign_sell_vol", 0) or 0)
        f_net = fb - fs

        streak = 0
        for _, r in past.iloc[::-1].iterrows():
            rb = float(r.get("foreign_buy_vol",  0) or 0)
            rs = float(r.get("foreign_sell_vol", 0) or 0)
            if rb - rs > 0:
                streak += 1
            else:
                break

        f_ratio     = max(0.0, f_net / avg_vol) if avg_vol > 0 else 0
        foreign_score = min(6.0, f_ratio * 6) + min(2.0, streak * 0.5)

        # ── Close quality ─────────────────────────────────────────────────────
        rng       = high - low
        close_pos = (close - low) / rng if rng > 0 else 0.5

        total = round(vol_score + breakout_score + consol_score + foreign_score, 2)
        return total, {
            "vol_x":         round(vol_x, 1),
            "vol_score":     vol_score,
            "breakout":      breakout_label,
            "breakout_score":breakout_score,
            "coiling":       coiling,
            "consol_score":  round(consol_score, 1),
            "f_net_K":       int(f_net / 1000),
            "f_streak":      streak,
            "close_pos":     round(close_pos, 2),
        }
    except Exception:
        return 0.0, {}

# ─── Signal 2 + 4 + 6: Tick data signals ──────────────────────────────────────

def _score_tick(symbol: str, today_vn: str) -> tuple:
    """
    Returns (score 0-40, detail_dict).
    Signals: late-session buy pressure, intraday acceleration, block clustering.
    Also returns peak_window (time of max accumulation).
    """
    path = os.path.join(TICK_DIR, f"{symbol}.parquet")
    if not os.path.exists(path):
        return 0.0, {}
    try:
        df = pd.read_parquet(path)
        t  = df[df["td"] == today_vn].copy()
        if len(t) < 10:
            return 0.0, {}

        t["t"]  = t["t"].astype(str)
        t["v"]  = pd.to_numeric(t["v"], errors="coerce").fillna(0)
        t["s"]  = t["s"].astype(str).str.strip().str.lower()

        day_vol = t["v"].sum()
        if day_vol == 0:
            return 0.0, {}

        # ── Late session buy pressure 14:00-14:45 (0-20) ─────────────────────
        late = t[t["t"] >= "14:00:00"]
        late_buy  = late[late["s"] == "buy"]["v"].sum()
        late_sell = late[late["s"] == "sell"]["v"].sum()
        late_tot  = late_buy + late_sell
        late_ratio = late_buy / late_tot if late_tot > 0 else 0.5
        late_score = max(0.0, (late_ratio - 0.5) / 0.5) * 20

        # ── Intraday volume acceleration (0-10) ───────────────────────────────
        # First 60 min vs last 60 min of session
        first_hr = t[(t["t"] >= "09:00:00") & (t["t"] < "10:00:00")]["v"].sum()
        last_hr  = t[t["t"] >= "13:45:00"]["v"].sum()
        if first_hr > 0:
            accel_ratio = last_hr / first_hr
            # >2× = strong late accumulation
            accel_score = min(10.0, max(0.0, (accel_ratio - 1) / 3) * 10)
        else:
            accel_ratio = 1.0
            accel_score = 0.0

        # ── Block clustering in last 2 hours (0-10) ───────────────────────────
        avg_v       = t["v"].mean()
        block_thresh = max(avg_v * 5, 100_000)
        last2h_buys = t[(t["t"] >= "13:00:00") & (t["s"] == "buy") & (t["v"] >= block_thresh)]
        last2h_vol  = last2h_buys["v"].sum()
        block_score = min(10.0, (last2h_vol / day_vol) * 30)  # 33% of vol as big buys = max
        n_blocks    = len(last2h_buys)

        # ── Peak accumulation window ──────────────────────────────────────────
        peak_window = _find_peak_window(t)

        total = round(late_score + accel_score + block_score, 2)
        return total, {
            "late_buy_pct":  round(late_ratio * 100, 1),
            "late_score":    round(late_score, 1),
            "accel_x":       round(accel_ratio, 1),
            "accel_score":   round(accel_score, 1),
            "n_blocks_2h":   n_blocks,
            "block_score":   round(block_score, 1),
            "peak_window":   peak_window,
        }
    except Exception:
        return 0.0, {}


def _find_peak_window(t: pd.DataFrame) -> str:
    """Return 30-min bucket with highest buy dominance (buy_vol / total_vol)."""
    try:
        t = t.copy()
        t["bucket"] = t["t"].str[:4] + "0"  # floor to 10-min bucket: "14:2" → "14:20"

        buckets = []
        for bucket, grp in t.groupby("bucket"):
            bv = grp[grp["s"] == "buy"]["v"].sum()
            sv = grp[grp["s"] == "sell"]["v"].sum()
            tot = bv + sv
            if tot > 0:
                buckets.append((bucket, bv / tot, tot))

        if not buckets:
            return ""

        # Weight by volume to prefer meaningful windows
        best = max(buckets, key=lambda x: x[1] * (x[2] ** 0.5))
        return f"{best[0]}  ({round(best[1]*100)}% buy  {int(best[2]/1000)}K shares)"
    except Exception:
        return ""

# ─── Signal 7: Tu doanh ───────────────────────────────────────────────────────

def _score_tudoanh(symbol: str, td_df: pd.DataFrame) -> tuple:
    """Max 7 pts."""
    if td_df.empty or symbol not in td_df.index:
        return 0.0, {}
    try:
        row     = td_df.loc[symbol]
        net_val = float(row.get("net_value", 0) or 0)
        if net_val <= 0:
            return 0.0, {}
        score = min(7.0, (net_val / 1e9) * 1.5)
        return round(score, 2), {"td_net_B": round(net_val / 1e9, 2)}
    except Exception:
        return 0.0, {}

# ─── Combine ──────────────────────────────────────────────────────────────────

def _analyze(args):
    symbol, today_iso, today_vn, td_df = args

    s_price,  d_price  = _score_price(symbol, today_iso)
    s_tick,   d_tick   = _score_tick(symbol, today_vn)
    s_td,     d_td     = _score_tudoanh(symbol, td_df)

    total = round(s_price + s_tick + s_td, 1)
    details = {**d_price, **d_tick, **d_td}
    return symbol, total, details

# ─── Reason builder ───────────────────────────────────────────────────────────

def _build_reason(d: dict) -> list:
    """Return ordered list of plain-English reason strings."""
    reasons = []

    vol_x = d.get("vol_x", 1.0)
    if vol_x >= 1.2:
        reasons.append(f"Volume {vol_x}x normal")

    bk = d.get("breakout", "")
    if bk:
        reasons.append(bk)

    if d.get("coiling"):
        consol = d.get("consol_score", 0)
        if consol > 3:
            reasons.append(f"Tight coil breakout (range expand {d['consol_score']})")

    lp = d.get("late_buy_pct", 0)
    if lp >= 55:
        reasons.append(f"Late-session {lp}% buy (14:00+)")

    ax = d.get("accel_x", 1.0)
    if ax >= 1.5:
        reasons.append(f"Vol accel {ax}x (last-hr vs first-hr)")

    nb = d.get("n_blocks_2h", 0)
    if nb >= 2:
        reasons.append(f"{nb} block trades in last 2h")

    fk = d.get("f_net_K", 0)
    fs = d.get("f_streak", 0)
    if fk > 0:
        reasons.append(f"Foreign net +{fk:,}K" + (f" ({fs}d streak)" if fs >= 2 else ""))
    elif fk < 0 and fs >= 3:
        reasons.append(f"Foreign streak {fs}d sell (watch contra)")

    td = d.get("td_net_B")
    if td and td > 0:
        reasons.append(f"Tu doanh net +{td:.1f}B VND")

    return reasons

# ─── Output ───────────────────────────────────────────────────────────────────

def _print_results(results: list, target_date: str, sector_map: dict):
    print(f"\n{'='*70}")
    print(f"  SURGE CANDIDATES → {target_date}  (top {len(results)})")
    print(f"{'='*70}")

    strength_labels = {
        (70, 200): "STRONG",
        (50,  70): "WATCH",
        (30,  50): "SPECULATIVE",
        (0,   30): "WEAK",
    }

    for rank, (sym, score, d) in enumerate(results, 1):
        sector  = sector_map.get(sym, "")
        reasons = _build_reason(d)
        peak    = d.get("peak_window", "")
        cp      = d.get("close_pos", 0)

        label = next((v for (lo, hi), v in strength_labels.items() if lo <= score < hi), "")

        print(f"\n#{rank:<3} {sym:<6}  score={score:<6}  [{label}]  {sector}")
        if reasons:
            print(f"     Why : {' | '.join(reasons)}")
        if peak:
            print(f"     When: peak buy window → {peak}")
        print(f"     Close position in day range: {int(cp*100)}%"
              + ("  (strong close)" if cp >= 0.8 else ""))

    print(f"\n{'='*70}")
    print(f"Score guide: 70+ strong signal | 50-69 watchlist | 30-49 speculative")
    print(f"{'='*70}\n")


def _telegram_message(results: list, target_date: str) -> str:
    lines = [f"<b>SURGE WATCHLIST → {target_date}</b>", ""]
    for rank, (sym, score, d) in enumerate(results[:15], 1):
        reasons = _build_reason(d)
        tag_str = " | ".join(reasons[:3]) if reasons else "—"
        icon    = "🟢" if score >= 70 else "🟡" if score >= 50 else "⚪"
        lines.append(f"{icon} <b>#{rank} {sym}</b>  [{score}]  {tag_str}")
        peak = d.get("peak_window", "")
        if peak:
            lines.append(f"     ⏱ {peak}")

    lines += [
        "",
        "70+ strong | 50-69 watch | 30-49 speculative",
        "⚠️ Verify on chart before entering.",
    ]
    return "\n".join(lines)


def _send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import requests
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": int(TELEGRAM_CHAT_ID), "text": msg, "parse_mode": "HTML"},
            timeout=15,
        )
        r.raise_for_status()
        print("Telegram sent.")
    except Exception as e:
        print(f"Telegram error: {e}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def run(top_n: int = TOP_N, today_iso: str = None, today_vn: str = None,
        send_telegram: bool = True) -> list:
    """
    Run surge detection.

    Returns sorted list of (symbol, score, details_dict).
    today_iso : YYYY-MM-DD  (price CSVs). Auto-detected if None.
    today_vn  : dd/mm/YYYY  (tick files). Auto-detected if None.
    """
    if not os.path.exists(TICK_DIR):
        print("No tick_data directory found. Run daily_update.py first.")
        return []

    if today_iso is None:
        today_iso = _latest_price_date()
    if today_vn is None:
        today_vn  = _latest_tick_date()

    # If tick date is newer, use it as reference
    tick_as_iso = _vn_to_iso(today_vn)
    if tick_as_iso > today_iso:
        today_iso = tick_as_iso

    # Next trading day (for display — naive +1 weekday)
    d = dt.datetime.strptime(today_iso, "%Y-%m-%d")
    offset = 3 if d.weekday() == 4 else 1   # Friday → Monday
    next_day = (d + dt.timedelta(days=offset)).strftime("%Y-%m-%d")

    print(f"\nSURGE DETECTOR")
    print(f"  Analysing : {today_iso} (tick: {today_vn})")
    print(f"  Target day: {next_day}")

    td_df      = _load_tudoanh(today_vn)
    sector_map = _load_sector_map()

    symbols = [f.replace(".parquet", "") for f in os.listdir(TICK_DIR)
               if f.endswith(".parquet")]
    print(f"  Symbols   : {len(symbols)} | tu_doanh rows: {len(td_df)}")

    tasks   = [(sym, today_iso, today_vn, td_df) for sym in symbols]
    results = []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_analyze, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            sym, score, details = fut.result()
            if score > 0:
                results.append((sym, score, details))

    results.sort(key=lambda x: x[1], reverse=True)
    top = results[:top_n]

    _print_results(top, next_day, sector_map)

    if send_telegram and top:
        msg = _telegram_message(top, next_day)
        _send_telegram(msg)

    return top


if __name__ == "__main__":
    run()
