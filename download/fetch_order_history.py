"""
fetch_order_history.py
----------------------
Downloads daily buy/sell order flow data from dulieu.nguoiquansat.vn
for strategy-sector stocks.

Data per row (per trading day per stock):
  date, symbol, close, volume_matched, value_bn,
  buy_orders, sell_orders, net_orders,
  buy_volume, sell_volume, net_volume

Key derived metrics:
  net_volume  = buy_volume - sell_volume  (+ = net buying pressure)
  net_orders  = buy_orders - sell_orders
  ob_ratio    = buy_volume / sell_volume  (> 1 = more buy than sell)

Usage:
  python fetch_order_history.py                         # strategy sectors, last 365d
  python fetch_order_history.py --all                   # all tickers
  python fetch_order_history.py --days 3650 --force     # full ~10yr history
  python fetch_order_history.py --workers 4
  python fetch_order_history.py --max-pages 200         # cap per symbol (default 200)
"""

import argparse
import datetime as dt
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TICKER_CSV = os.path.join(BASE_DIR, "ticker_sectors.csv")
SAVE_DIR   = os.path.join(BASE_DIR, "data", "order_history")

STRATEGY_SECTORS = {"Banks", "Food & Beverage", "Basic Resources", "Real Estate"}
DEDUP_COLS       = ["symbol", "date"]

BASE_URL = "https://dulieu.nguoiquansat.vn/History/OrderHistory"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Referer": "https://dulieu.nguoiquansat.vn/du-lieu-giao-dich",
    "X-Requested-With": "XMLHttpRequest",
    "sec-ch-ua": '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

_print_lock = threading.Lock()


def log(msg: str):
    with _print_lock:
        print(msg, flush=True)


def _parse_num(s: str):
    """Parse Vietnamese number strings: '1,965,234' → 1965234, '5.98' → 5.98"""
    s = s.strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def fetch_page(session: requests.Session, symbol: str,
               from_date: str, to_date: str, page: int) -> list[dict]:
    """Fetch one page of order history. Returns list of row dicts."""
    params = {
        "page":     page,
        "fromDate": from_date,
        "toDate":   to_date,
        "exId":     "",
        "code":     symbol,
        "idNganh":  "",
        "_":        str(int(time.time() * 1000)),
    }
    for attempt in range(3):
        try:
            r = session.get(BASE_URL, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(20 * (attempt + 1))
                continue
            if r.status_code != 200:
                return []
            break
        except Exception:
            if attempt == 2:
                return []
            time.sleep(5)
    else:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 10:
            continue
        vals = [td.get_text(strip=True) for td in tds]
        try:
            rows.append({
                "date":          vals[1],        # dd/MM/yyyy
                "symbol":        vals[2],
                "close":         _parse_num(vals[3]),
                "volume":        _parse_num(vals[4]),   # total volume matched
                "value_bn":      _parse_num(vals[5]),   # total value (billion VND)
                "buy_orders":    _parse_num(vals[6]),
                "sell_orders":   _parse_num(vals[7]),
                "net_orders":    _parse_num(vals[8]),
                "buy_volume":    _parse_num(vals[9]),
                "sell_volume":   _parse_num(vals[10]),
                "net_volume":    _parse_num(vals[11]) if len(vals) > 11 else None,
            })
        except IndexError:
            continue

    return rows


def fetch_symbol(symbol: str, from_date: str, to_date: str,
                 max_pages: int = 200) -> pd.DataFrame:
    """Fetch all pages for a symbol across the date range."""
    session = requests.Session()
    session.headers.update(HEADERS)

    all_rows = []
    for page in range(1, max_pages + 1):
        rows = fetch_page(session, symbol, from_date, to_date, page)
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < 15:      # last page (partial)
            break
        time.sleep(0.15)        # gentle pacing

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Derived metrics
    df["ob_ratio"] = df.apply(
        lambda r: round(r["buy_volume"] / r["sell_volume"], 4)
        if r["sell_volume"] and r["sell_volume"] != 0 else None,
        axis=1,
    )
    return df


def is_fresh(path: str, max_age_days: int) -> bool:
    if not os.path.exists(path):
        return False
    return (time.time() - os.path.getmtime(path)) / 86_400 < max_age_days


def merge_and_save(path: str, new_df: pd.DataFrame) -> int:
    if os.path.exists(path):
        try:
            old = pd.read_parquet(path)
            combined = (
                pd.concat([new_df, old], ignore_index=True)
                  .drop_duplicates(subset=DEDUP_COLS, keep="first")
                  .sort_values("date")
                  .reset_index(drop=True)
            )
        except Exception:
            combined = new_df
    else:
        combined = new_df
    combined.to_parquet(path, index=False, engine="pyarrow")
    return len(combined)


def process_symbol(args) -> tuple[str, str]:
    symbol, industry, total, position, from_date, to_date, force, fresh_days, max_pages = args
    out_path = os.path.join(SAVE_DIR, f"{symbol}.parquet")

    if not force and is_fresh(out_path, fresh_days):
        log(f"[{position:>4}/{total}] {symbol:>6}  SKIP")
        return symbol, "skipped"

    action = "UPDATE" if os.path.exists(out_path) else "NEW"
    log(f"[{position:>4}/{total}] {symbol:>6}  {action} ...")

    df = fetch_symbol(symbol, from_date, to_date, max_pages=max_pages)
    if df.empty:
        log(f"[{position:>4}/{total}] {symbol:>6}  EMPTY")
        return symbol, "failed"

    df["sector"] = industry
    n = merge_and_save(out_path, df)
    log(f"[{position:>4}/{total}] {symbol:>6}  OK  {n:>4} rows")
    return symbol, "ok"


def run():
    parser = argparse.ArgumentParser(description="Download order-flow data from nguoiquansat.")
    parser.add_argument("--all",       action="store_true",
                        help="All tickers, not just strategy sectors")
    parser.add_argument("--force",     action="store_true",
                        help="Re-download even if fresh")
    parser.add_argument("--days",      type=int, default=365,
                        help="How many calendar days of history to fetch (default 365)")
    parser.add_argument("--fresh",     type=int, default=7,
                        help="Skip file if modified within N days (default 7)")
    parser.add_argument("--workers",   type=int, default=4,
                        help="Parallel threads (default 4)")
    parser.add_argument("--max-pages", type=int, default=200,
                        help="Max pages per symbol (15 rows/page; default 200 ≈ 12yr)")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    today     = dt.date.today()
    from_dt   = today - dt.timedelta(days=args.days)
    from_date = from_dt.strftime("%d/%m/%Y")
    to_date   = today.strftime("%d/%m/%Y")

    tickers = pd.read_csv(TICKER_CSV)
    if not args.all:
        tickers = tickers[tickers["industry"].isin(STRATEGY_SECTORS)].copy()
    tickers = tickers.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    scope = "ALL" if args.all else "strategy sectors"
    print(f"[{today}]  Scope: {scope}")
    print(f"  Tickers : {len(tickers)}")
    print(f"  Period  : {from_date} → {to_date}  ({args.days} days)")
    print(f"  Fresh threshold : {args.fresh} days  (--force={args.force})")
    print(f"  Workers : {args.workers}\n")

    tasks = [
        (row["ticker"], row["industry"], len(tickers), i + 1,
         from_date, to_date, args.force, args.fresh, args.max_pages)
        for i, row in tickers.iterrows()
    ]

    ok, skipped, failed = 0, 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_symbol, t): t[0] for t in tasks}
        for future in as_completed(futures):
            sym, status = future.result()
            if status == "ok":       ok      += 1
            elif status == "skipped": skipped += 1
            else:                    failed.append(sym)

    # Retry failed (sequential)
    wait = 20
    for attempt in range(1, 4):
        if not failed:
            break
        print(f"\nRetry {attempt}/3 — {len(failed)} symbols (waiting {wait}s...)")
        time.sleep(wait)
        wait *= 2
        still_failed = []
        for i, sym in enumerate(failed, 1):
            row = tickers[tickers["ticker"] == sym].iloc[0]
            task = (sym, row["industry"], len(failed), i,
                    from_date, to_date, True, args.fresh, args.max_pages)
            _, status = process_symbol(task)
            if status == "ok": ok += 1
            else:              still_failed.append(sym)
            time.sleep(2)
        failed = still_failed

    print(f"\n{'─'*50}")
    print(f"  Downloaded : {ok}")
    print(f"  Skipped    : {skipped}")
    print(f"  Failed     : {len(failed)}")
    if failed:
        print(f"  Failed     : {failed}")
    print(f"  Output     : {SAVE_DIR}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    run()
