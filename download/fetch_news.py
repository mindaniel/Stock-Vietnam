"""
fetch_news.py
-------------
Downloads company news from dulieu.nguoiquansat.vn for strategy-sector stocks.

Two types of news items are captured:
  "article"      — links to nguoiquansat.vn editorial articles
  "announcement" — internal company disclosures (earnings, board decisions, etc.)
                   often includes a PDF URL and brief content snippet

Output : data/news/{SYMBOL}.parquet
         Columns: symbol, datetime, title, url, pdf_url, source_type,
                  content_snippet, sector

Usage:
  python fetch_news.py                   # strategy sectors (~388 stocks)
  python fetch_news.py --all             # all 1565 tickers
  python fetch_news.py --force           # re-download even if fresh
  python fetch_news.py --fresh 3         # skip files updated within 3 days
  python fetch_news.py --workers 4
"""

import argparse
import datetime as dt
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TICKER_CSV = os.path.join(BASE_DIR, "ticker_sectors.csv")
SAVE_DIR   = os.path.join(BASE_DIR, "data", "news")

STRATEGY_SECTORS = {"Banks", "Food & Beverage", "Basic Resources", "Real Estate"}
DEDUP_COLS       = ["symbol", "title", "datetime"]  # no duplicate (sym+title+date)

BASE_URL = "https://dulieu.nguoiquansat.vn/company/newspartial"
MAX_PAGES = 50   # safety cap; most stocks have <10 pages

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "X-Requested-With": "XMLHttpRequest",
    "sec-ch-ua": '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

_DATE_RE = re.compile(r'\((\d{2})/(\d{2})\s+(\d{2}):(\d{2})\)')   # (DD/MM HH:MM)

_print_lock = threading.Lock()


def log(msg: str):
    with _print_lock:
        print(msg, flush=True)


def _infer_datetime(day: int, month: int, hour: int, minute: int,
                    ref_date: dt.date) -> dt.datetime:
    """
    Infer year for a date that only has DD/MM HH:MM.
    Rule: if (month, day) is strictly after ref_date's (month, day)
    in the calendar, the item belongs to the previous year.
    """
    year = ref_date.year
    try:
        candidate = dt.datetime(year, month, day, hour, minute)
    except ValueError:
        return None
    if candidate.date() > ref_date:
        candidate = candidate.replace(year=year - 1)
    return candidate


def parse_items(html: str, symbol: str, ref_date: dt.date) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for li in soup.select("ul.ul-news li"):
        a    = li.find("a")
        span = li.find("span")
        if not a or not span:
            continue

        title = a.get_text(strip=True)
        if not title:
            continue

        # Parse date
        m = _DATE_RE.search(span.get_text())
        if not m:
            continue
        d_day, d_mon, d_hr, d_min = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        item_dt = _infer_datetime(d_day, d_mon, d_hr, d_min, ref_date)
        if item_dt is None:
            continue

        href = a.get("href", "")
        is_announcement = "javascript" in href or href == ""

        url         = href if not is_announcement else None
        pdf_url     = None
        content_snip = None

        if is_announcement:
            # Look for hidden spans with content and PDF URL
            data_id = a.get("data-id", "")
            if data_id:
                url_span     = li.find("span", id=f"url-{data_id}")
                content_span = li.find("span", id=f"content-{data_id}")
                if url_span:
                    pdf_url = url_span.get_text(strip=True)
                if content_span:
                    # Strip HTML tags inside content
                    inner = BeautifulSoup(content_span.get_text(strip=True), "html.parser")
                    content_snip = inner.get_text(strip=True)[:500]

        rows.append({
            "symbol":          symbol,
            "datetime":        item_dt,
            "title":           title,
            "url":             url,
            "pdf_url":         pdf_url,
            "source_type":     "announcement" if is_announcement else "article",
            "content_snippet": content_snip,
        })

    has_next = bool(soup.select("a.news-a-btn-paging-next:not(.no-active)"))
    return rows, has_next


def fetch_all_news(symbol: str, sector: str,
                   session: requests.Session) -> pd.DataFrame:
    ref_date = dt.date.today()
    all_rows = []

    for page in range(1, MAX_PAGES + 1):
        try:
            r = session.get(BASE_URL, params={
                "page": page, "code": symbol,
                "_": str(int(time.time() * 1000)),
            }, timeout=20)
            if r.status_code != 200:
                break
        except Exception:
            break

        rows, has_next = parse_items(r.text, symbol, ref_date)
        all_rows.extend(rows)

        if not has_next or not rows:
            break
        time.sleep(0.2)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["sector"] = sector
    df = df.sort_values("datetime", ascending=False).reset_index(drop=True)
    return df


# ── Freshness / merge / save ──────────────────────────────────────────────────

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
                  .sort_values("datetime", ascending=False)
                  .reset_index(drop=True)
            )
        except Exception:
            combined = new_df
    else:
        combined = new_df
    combined.to_parquet(path, index=False, engine="pyarrow")
    return len(combined)


# ── Worker ────────────────────────────────────────────────────────────────────

def process_symbol(args) -> tuple[str, str]:
    symbol, sector, total, position, force, fresh_days = args
    out_path = os.path.join(SAVE_DIR, f"{symbol}.parquet")

    if not force and is_fresh(out_path, fresh_days):
        log(f"[{position:>4}/{total}] {symbol:>6}  SKIP")
        return symbol, "skipped"

    action = "UPDATE" if os.path.exists(out_path) else "NEW"
    log(f"[{position:>4}/{total}] {symbol:>6}  {action} ...")

    session = requests.Session()
    session.headers.update(HEADERS)
    session.headers["Referer"] = f"https://dulieu.nguoiquansat.vn/doanh-nghiep/{symbol.lower()}"

    df = fetch_all_news(symbol, sector, session)
    if df.empty:
        log(f"[{position:>4}/{total}] {symbol:>6}  EMPTY")
        return symbol, "failed"

    n = merge_and_save(out_path, df)
    log(f"[{position:>4}/{total}] {symbol:>6}  OK  {n:>4} items")
    return symbol, "ok"


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    parser = argparse.ArgumentParser(description="Download company news from nguoiquansat.")
    parser.add_argument("--all",     action="store_true",
                        help="All tickers, not just strategy sectors")
    parser.add_argument("--force",   action="store_true",
                        help="Re-download even if fresh")
    parser.add_argument("--fresh",   type=int, default=3,
                        help="Skip files updated within N days (default 3)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel threads (default 4)")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    tickers = pd.read_csv(TICKER_CSV)
    if not args.all:
        tickers = tickers[tickers["industry"].isin(STRATEGY_SECTORS)].copy()
    tickers = tickers.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    scope = "ALL" if args.all else "strategy sectors"
    print(f"[{dt.date.today()}]  Scope: {scope}")
    print(f"  Tickers : {len(tickers)}")
    print(f"  Fresh   : skip if updated within {args.fresh} days (--force={args.force})")
    print(f"  Workers : {args.workers}")
    print(f"  Output  : {SAVE_DIR}\n")

    tasks = [
        (row["ticker"], row["industry"], len(tickers), i + 1, args.force, args.fresh)
        for i, row in tickers.iterrows()
    ]

    ok, skipped, failed = 0, 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_symbol, t): t[0] for t in tasks}
        for future in as_completed(futures):
            sym, status = future.result()
            if status == "ok":        ok      += 1
            elif status == "skipped": skipped += 1
            else:                     failed.append(sym)

    # Retry failed
    wait = 15
    for attempt in range(1, 4):
        if not failed:
            break
        print(f"\nRetry {attempt}/3 — {len(failed)} symbols (waiting {wait}s...)")
        time.sleep(wait)
        wait *= 2
        still_failed = []
        for i, sym in enumerate(failed, 1):
            row = tickers[tickers["ticker"] == sym].iloc[0]
            task = (sym, row["industry"], len(failed), i, True, args.fresh)
            _, status = process_symbol(task)
            if status == "ok": ok += 1
            else:              still_failed.append(sym)
            time.sleep(1)
        failed = still_failed

    print(f"\n{'─'*50}")
    print(f"  Downloaded : {ok}")
    print(f"  Skipped    : {skipped}")
    print(f"  Failed     : {len(failed)}")
    if failed:
        print(f"  Failed list: {failed[:20]}")
    print(f"  Output     : {SAVE_DIR}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    run()
