#!/usr/bin/env python3
"""
fetch_foreign_flow_long.py
Long-history TOTAL foreign flow (institutional + retail combined — NOT split,
that split only exists in data/investor_flow from Sep 2024) from FireAnt's
historical-quotes endpoint. Goes back to each stock's listing date, so this
extends well before nguoiquansat's Sep 2024 start — built specifically to
stress-test the aggregate foreign-flow-vs-returns finding across market
regimes nguoiquansat can't reach (2018 correction, 2020 COVID crash, 2022
bear market).

Output: data/foreign_flow_long/{SYMBOL}.parquet — one row per trading day:
  date, close, total_volume, buy_foreign_value, sell_foreign_value,
  foreign_net_value, buy_foreign_qty, sell_foreign_qty,
  prop_trading_net_value (null before ~2013, disclosure started later),
  current_foreign_room, putthrough_value

Usage:
  python download/fetch_foreign_flow_long.py                  # all liquid tickers
  python download/fetch_foreign_flow_long.py --symbol VIC      # single ticker
  python download/fetch_foreign_flow_long.py --force           # re-fetch fresh files too
"""

import argparse
import glob
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICE_DIR = os.path.join(BASE_DIR, "data", "price")
OUT_DIR   = os.path.join(BASE_DIR, "data", "foreign_flow_long")

MIN_LIQUIDITY_VND = 1_000_000_000
START_DATE = "2000-01-01"   # FireAnt clips to each symbol's actual listing date
FRESHNESS_DAYS = 3
WORKERS = 6

_FA_URL = "https://restv2.fireant.vn/symbols/{symbol}/historical-quotes"

# Same long-lived public token used by download/fetch_financials.py's Job C
FA_TOKEN = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6IkdYdExONzViZlZQakdvNERWdjV4"
    "QkRITHpnSSIsImtpZCI6IkdYdExONzViZlZQakdvNERWdjV4QkRITHpnSSJ9.eyJpc3MiOiJo"
    "dHRwczovL2FjY291bnRzLmZpcmVhbnQudm4iLCJhdWQiOiJodHRwczovL2FjY291bnRzLmZp"
    "cmVhbnQudm4vcmVzb3VyY2VzIiwiZXhwIjoxODg5NjIyNTMwLCJuYmYiOjE1ODk2MjI1MzAs"
    "ImNsaWVudF9pZCI6ImZpcmVhbnQudHJhZGVzdGF0aW9uIiwic2NvcGUiOlsiYWNhZGVteS1y"
    "ZWFkIiwiYWNhZGVteS13cml0ZSIsImFjY291bnRzLXJlYWQiLCJhY2NvdW50cy13cml0ZSIs"
    "ImJsb2ctcmVhZCIsImNvbXBhbmllcy1yZWFkIiwiZmluYW5jZS1yZWFkIiwiaW5kaXZpZHVh"
    "bHMtcmVhZCIsImludmVzdG9wZWRpYS1yZWFkIiwib3JkZXJzLXJlYWQiLCJvcmRlcnMtd3Jp"
    "dGUiLCJwb3N0cy1yZWFkIiwicG9zdHMtd3JpdGUiLCJzZWFyY2giLCJzeW1ib2xzLXJlYWQi"
    "LCJ1c2VyLWRhdGEtcmVhZCIsInVzZXItZGF0YS13cml0ZSIsInVzZXJzLXJlYWQiXSwianRp"
    "IjoiMjYxYTZhYWQ2MTQ5Njk1ZmJiYzcwODM5MjM0Njc1NWQifQ.dA5-HVzWv-BRfEiAd24u"
    "NBiBxASO-PAyWeWESovZm_hj4aXMAZA1-bWNZeXt88dqogo18AwpDQ-h6gefLPdZSFrG5umC1"
    "dVWaeYvUnGm62g4XS29fj6p01dhKNNqrsu5KrhnhdnKYVv9VdmbmqDfWR8wDgglk5cJFqalzq"
    "6dJWJInFQEPmUs9BW_Zs8tQDn-i5r4tYq2U8vCdqptXoM7YgPllXaPVDeccC9QNu2Xlp9WUv"
    "oROzoQXg25lFub1IYkTrM66gJ6t9fJRZToewCt495WNEOQFa_rwLCZ1QwzvL0iYkONHS_jZ0B"
    "OhBCdW9dWSawD6iF1SIQaFROvMDH1rg"
)

_thread_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_thread_local, "s"):
        s = requests.Session()
        s.headers.update({
            "accept": "application/json, text/plain, */*",
            "authorization": f"Bearer {FA_TOKEN}",
            "origin": "https://fireant.vn",
            "referer": "https://fireant.vn/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/147.0.0.0 Safari/537.36",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
        })
        _thread_local.s = s
    return _thread_local.s


def liquid_universe() -> list:
    liquid = []
    for fpath in glob.glob(os.path.join(PRICE_DIR, "*.parquet")):
        ticker = os.path.splitext(os.path.basename(fpath))[0].upper()
        try:
            df = pd.read_parquet(fpath)
            df.columns = [c.strip().lower() for c in df.columns]
            if "close" not in df.columns or "volume" not in df.columns:
                continue
            med_to = (df["close"] * df["volume"] * 1000).tail(60).median()
            if med_to >= MIN_LIQUIDITY_VND:
                liquid.append(ticker)
        except Exception:
            pass
    return sorted(liquid)


def is_fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    age_days = (time.time() - os.path.getmtime(path)) / 86400
    return age_days < FRESHNESS_DAYS


def fetch_symbol(symbol: str, end_date: str) -> pd.DataFrame:
    session = _session()
    all_rows = []
    offset = 0
    limit = 1000
    for _ in range(50):   # safety cap ~50k rows per symbol, far more than needed
        for attempt in range(3):
            try:
                r = session.get(_FA_URL.format(symbol=symbol),
                                 params={"startDate": START_DATE, "endDate": end_date,
                                         "offset": offset, "limit": limit},
                                 timeout=30)
                if r.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                time.sleep(2)
                data = None
        if not data:
            break
        all_rows.extend(data)
        if len(data) < limit:
            break
        offset += limit
        time.sleep(0.2)
    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    out = pd.DataFrame({
        "date": df["date"],
        "close": df.get("priceClose"),
        "total_volume": df.get("totalVolume"),
        "buy_foreign_value": df.get("buyForeignValue"),
        "sell_foreign_value": df.get("sellForeignValue"),
        "buy_foreign_qty": df.get("buyForeignQuantity"),
        "sell_foreign_qty": df.get("sellForeignQuantity"),
        "prop_trading_net_value": df.get("propTradingNetValue"),
        "current_foreign_room": df.get("currentForeignRoom"),
        "putthrough_value": df.get("putthroughValue"),
    })
    out["foreign_net_value"] = out["buy_foreign_value"] - out["sell_foreign_value"]
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])
    return out.reset_index(drop=True)


def worker(args):
    symbol, end_date, force, idx, total = args
    out_path = os.path.join(OUT_DIR, f"{symbol}.parquet")
    if not force and is_fresh(out_path):
        print(f"[{idx:>4}/{total}] {symbol:<6} SKIP (fresh)")
        return symbol, "skipped"
    try:
        df = fetch_symbol(symbol, end_date)
        if df.empty:
            print(f"[{idx:>4}/{total}] {symbol:<6} NO DATA")
            return symbol, "failed"
        df.to_parquet(out_path, index=False)
        print(f"[{idx:>4}/{total}] {symbol:<6} OK {len(df):>5} rows "
              f"({df['date'].min().date()} to {df['date'].max().date()})")
        return symbol, "ok"
    except Exception as e:
        print(f"[{idx:>4}/{total}] {symbol:<6} ERROR: {e}")
        return symbol, "error"


def main():
    p = argparse.ArgumentParser(description="Fetch long-history total foreign flow from FireAnt")
    p.add_argument("--symbol", default=None, help="Single symbol (skips liquidity screen)")
    p.add_argument("--force", action="store_true", help="Re-fetch even if file is fresh")
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    symbols = [args.symbol.upper()] if args.symbol else liquid_universe()
    print(f"Fetching {len(symbols)} symbols, {START_DATE} -> {args.end_date}, "
          f"workers={args.workers}, force={args.force}")
    print(f"Output: {OUT_DIR}\n")

    tasks = [(sym, args.end_date, args.force, i + 1, len(symbols)) for i, sym in enumerate(symbols)]
    ok = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            _, status = fut.result()
            if status == "ok": ok += 1
            elif status == "skipped": skipped += 1
            else: failed += 1

    print(f"\nDone. ok={ok} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
