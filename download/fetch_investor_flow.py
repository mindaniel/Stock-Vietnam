"""
fetch_investor_flow.py
======================
Scrapes dulieu.nguoiquansat.vn/History/PhanLoaiNDTHistory for all 5 investor
type net flows and saves per-ticker parquet files.

Columns in output parquet:
  date                     – trading date (datetime64)
  ticker                   – stock code
  close                    – closing price (VND thousands)
  tu_doanh_net             – Tự doanh ròng (tỷ VND, matched order)
  ca_nhan_trongnuoc_net    – Cá nhân trong nước ròng
  to_chuc_trongnuoc_net    – Tổ chức trong nước ròng
  ca_nhan_nuocngoai_net    – Cá nhân nước ngoài ròng
  to_chuc_nuocngoai_net    – Tổ chức nước ngoài ròng
  [total columns]          – Tổng GTGD versions (same 5 types, suffix _total)

Usage:
  # Backfill one or more tickers (auto-incremental):
  python fetch_investor_flow.py ACB VCB TCB

  # Full date range (override):
  python fetch_investor_flow.py ACB --from 2024-09-01 --to 2026-06-12

  # All major tickers (predefined list):
  python fetch_investor_flow.py --all

  # All HOSE+HNX tickers, 4 parallel workers, skip already-fetched:
  python fetch_investor_flow.py --hose-hnx --skip-existing --workers 4

  # Daily update (fetch last 14 days for all stored tickers):
  python fetch_investor_flow.py --update
"""

import sys, os, time, argparse, threading, io
import requests
import pandas as pd
import pyarrow  # pre-load before threads start to avoid double-registration race
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# ── Config ────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent.parent  # repo root
OUTDIR = BASE / "data" / "investor_flow"
OUTDIR.mkdir(parents=True, exist_ok=True)

URL        = "https://dulieu.nguoiquansat.vn/History/PhanLoaiNDTHistory"
DATA_START = datetime(2024, 9, 16)
CHUNK_DAYS = 60
PAGE_DELAY = 0.5    # seconds between requests per worker
RETRY_WAIT = 5
MAX_RETRIES = 4

# Different User-Agents simulate separate browser profiles
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

_print_lock = threading.Lock()
_save_lock  = threading.Lock()
_counter_lock = threading.Lock()

def _tprint(*args, **kwargs):
    """Thread-safe print."""
    with _print_lock:
        print(*args, **kwargs)

ALL_TICKERS = [
    # ── Banks (20 stocks — HOSE/HNX liquid) ──────────────────────────────────
    "ACB","VCB","BID","CTG","TCB","MBB","VPB","STB","HDB","LPB",
    "OCB","MSB","SSB","VIB","EIB","TPB","SHB","BAB","KLB","NAB",
    # ── Real Estate (21 stocks) ───────────────────────────────────────────────
    "VHM","VIC","NVL","PDR","KDH","DXG","DIG","NLG","VRE","BCM",
    "KBC","HDG","CEO","SCR","CRE","DPG","GEX","SJS","QCG","NTL","DXS",
    # ── Basic Resources (8 stocks — steel, rubber, materials) ────────────────
    "HPG","HSG","NKG","POM","GVR","PHR","CSM","TLG",
    # ── Food & Beverage (12 stocks) ───────────────────────────────────────────
    "MSN","SAB","VNM","MCH","QNS","SBT","ANV","VCF","KDC","CAN","HHC","TAC",
    # ── Oil & Gas / Energy ────────────────────────────────────────────────────
    "GAS","PLX","PVT","PVD","BSR",
    # ── Tech / Retail / Diversified ───────────────────────────────────────────
    "FPT","MWG","VGI","REE","BVH","VNM",
]
ALL_TICKERS = list(dict.fromkeys(ALL_TICKERS))


def _make_session(worker_id: int = 0) -> requests.Session:
    """Create a session with a unique User-Agent for this worker."""
    s = requests.Session()
    ua = _USER_AGENTS[worker_id % len(_USER_AGENTS)]
    s.headers.update({
        "accept":           "*/*",
        "accept-language":  "en-US,en;q=0.9",
        "x-requested-with": "XMLHttpRequest",
        "referer":          "https://dulieu.nguoiquansat.vn/du-lieu-giao-dich",
        "user-agent":       ua,
    })
    return s


# ── Parser ────────────────────────────────────────────────────────────────────
def _parse_page(html: str) -> tuple[list[dict], int]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="plndt-history-table")
    rows = []
    if table:
        for tr in table.find("tbody").find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) < 14:
                continue
            def f(v):
                try:
                    return float(v.replace(",", "")) if v not in ("", "-", "N/A") else 0.0
                except Exception:
                    return 0.0
            rows.append({
                "date":                   cells[1],
                "ticker":                 cells[2],
                "close":                  f(cells[3]),
                "tu_doanh_net":           f(cells[4]),
                "tu_doanh_net_total":     f(cells[5]),
                "ca_nhan_trongnuoc_net":  f(cells[6]),
                "ca_nhan_trongnuoc_total":f(cells[7]),
                "to_chuc_trongnuoc_net":  f(cells[8]),
                "to_chuc_trongnuoc_total":f(cells[9]),
                "ca_nhan_nuocngoai_net":  f(cells[10]),
                "ca_nhan_nuocngoai_total":f(cells[11]),
                "to_chuc_nuocngoai_net":  f(cells[12]),
                "to_chuc_nuocngoai_total":f(cells[13]),
            })

    pagination = soup.find("ul", class_="common-paging")
    max_page = 1
    if pagination:
        page_links = pagination.find_all("a", attrs={"data-page": True})
        if page_links:
            max_page = max(int(a["data-page"]) for a in page_links)
    return rows, max_page


def _fetch_chunk(ticker: str, from_dt: datetime, to_dt: datetime,
                 session: requests.Session) -> list[dict]:
    all_rows = []
    fd_str = from_dt.strftime("%d/%m/%Y")
    td_str = to_dt.strftime("%d/%m/%Y")
    page = 1
    while True:
        params = {
            "page":      page,
            "fromDate":  fd_str,
            "toDate":    td_str,
            "exId":      "",
            "code":      ticker,
            "idNganh":   "",
            "_":         str(int(time.time() * 1000)),
        }
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                r = session.get(URL, params=params, timeout=25)
                r.raise_for_status()
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                time.sleep(RETRY_WAIT * (attempt + 1))  # exponential back-off
        if last_exc:
            _tprint(f"    ERROR {ticker} page {page}: {last_exc}")
            return all_rows

        rows, max_page = _parse_page(r.text)
        all_rows.extend(rows)
        time.sleep(PAGE_DELAY)
        if page >= max_page:
            break
        page += 1
    return all_rows


# ── Per-ticker fetch (called from worker thread) ───────────────────────────────
def fetch_ticker(ticker: str, from_dt: datetime, to_dt: datetime,
                 existing: pd.DataFrame | None = None,
                 session: requests.Session | None = None) -> pd.DataFrame:
    ticker  = ticker.upper()
    session = session or _make_session()

    if existing is not None and not existing.empty:
        last_stored = existing["date"].max()
        new_from = last_stored + timedelta(days=1)
        if new_from > to_dt:
            _tprint(f"  {ticker}: already up to date ({last_stored.date()})")
            return existing
        from_dt = new_from
        _tprint(f"  {ticker}: incremental from {from_dt.date()} → {to_dt.date()}")
    else:
        _tprint(f"  {ticker}: full fetch {from_dt.date()} → {to_dt.date()}")

    all_rows = []
    chunk_start = from_dt
    while chunk_start <= to_dt:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), to_dt)
        rows = _fetch_chunk(ticker, chunk_start, chunk_end, session)
        all_rows.extend(rows)
        chunk_start = chunk_end + timedelta(days=1)

    if not all_rows:
        _tprint(f"  {ticker}: no new rows found")
        return existing if existing is not None else pd.DataFrame()

    new_df = pd.DataFrame(all_rows)
    new_df["date"] = pd.to_datetime(new_df["date"], format="%d/%m/%Y")
    new_df = new_df.sort_values("date").drop_duplicates(subset=["date", "ticker"])

    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "ticker"]).sort_values("date")
        return combined

    return new_df


def load_existing(ticker: str) -> pd.DataFrame | None:
    path = OUTDIR / f"{ticker}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return None


def fetch_snapshot_all(from_dt: datetime, to_dt: datetime,
                       session: requests.Session = None) -> dict:
    """Fetch ALL tickers at once (code= empty) for a short date range.

    Returns {ticker: DataFrame} for every ticker that appears in the response.
    Much faster than per-ticker requests for 1-2 day windows:
      ~600 tickers × 1 day = ~40 pages × 0.5s = ~20s total.
    """
    session = session or _make_session()
    fd_str = from_dt.strftime("%d/%m/%Y")
    td_str = to_dt.strftime("%d/%m/%Y")

    all_rows = []
    page = 1
    while True:
        params = {
            "page":     page,
            "fromDate": fd_str,
            "toDate":   td_str,
            "exId":     "",
            "code":     "",   # empty = all tickers
            "idNganh":  "",
            "_":        str(int(time.time() * 1000)),
        }
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                r = session.get(URL, params=params, timeout=25)
                r.raise_for_status()
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                time.sleep(RETRY_WAIT)
        if last_exc:
            print(f"    ERROR snapshot page {page}: {last_exc}")
            break

        rows, max_page = _parse_page(r.text)
        if not rows:
            break
        all_rows.extend(rows)
        print(f"  Snapshot page {page}/{max_page} ({len(rows)} rows, {len(all_rows)} total)")
        time.sleep(PAGE_DELAY)
        if page >= max_page:
            break
        page += 1

    if not all_rows:
        return {}

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date"])
    df["ticker"] = df["ticker"].str.upper().str.strip()

    by_ticker = {}
    for tkr, grp in df.groupby("ticker"):
        by_ticker[str(tkr)] = grp.reset_index(drop=True)
    return by_ticker


def save_ticker(ticker: str, df: pd.DataFrame):
    path = OUTDIR / f"{ticker}.parquet"
    with _save_lock:
        df.to_parquet(path, index=False)
    _tprint(f"  Saved {len(df)} rows → {path.name}")


# ── listing loader ────────────────────────────────────────────────────────────
def load_hose_hnx_tickers(xlsx_path: Path | None = None) -> list[str]:
    if xlsx_path is None:
        xlsx_path = BASE / "data" / "vndirect_listing.xlsx"
    if not xlsx_path.exists():
        print(f"  [WARN] {xlsx_path.name} not found — falling back to ALL_TICKERS")
        return ALL_TICKERS
    try:
        df = pd.read_excel(xlsx_path)
        col_floor = next((c for c in df.columns if c.lower() in ("floor","exchange","san")), None)
        col_code  = next((c for c in df.columns if c.lower() in ("code","ticker","symbol","ma")), None)
        if col_code is None:
            print("  [WARN] Cannot find code column in xlsx — using ALL_TICKERS")
            return ALL_TICKERS
        if col_floor:
            df = df[df[col_floor].isin(["HOSE", "HNX"])]
        tickers = sorted(df[col_code].dropna().astype(str).str.upper().unique().tolist())
        print(f"  Loaded {len(tickers)} HOSE+HNX tickers from {xlsx_path.name}")
        return tickers
    except Exception as e:
        print(f"  [WARN] Could not read xlsx: {e} — using ALL_TICKERS")
        return ALL_TICKERS


# ── Worker task ───────────────────────────────────────────────────────────────
def _worker_task(ticker: str, from_dt: datetime, to_dt: datetime,
                 skip_existing: bool, update_mode: bool,
                 worker_id: int) -> tuple[str, str]:
    """
    Returns (ticker, status) where status is 'saved', 'skipped', or 'error'.
    """
    try:
        session  = _make_session(worker_id)
        existing = None
        if update_mode:
            existing = load_existing(ticker)
        elif not skip_existing:
            existing = load_existing(ticker)

        df = fetch_ticker(ticker, from_dt, to_dt, existing=existing, session=session)
        if df is not None and not df.empty:
            save_ticker(ticker, df)
            return ticker, "saved"
        else:
            return ticker, "skipped"
    except Exception as e:
        _tprint(f"  ERROR {ticker}: {e}")
        return ticker, "error"


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fetch investor flow data from nguoiquansat.vn",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_investor_flow.py ACB VCB        # specific tickers
  python fetch_investor_flow.py --all          # predefined ~70 key stocks
  python fetch_investor_flow.py --hose-hnx     # ALL HOSE+HNX from vndirect_listing.xlsx
  python fetch_investor_flow.py --hose-hnx --skip-existing --workers 4
  python fetch_investor_flow.py --update       # incremental update for stored tickers
        """)
    parser.add_argument("tickers",         nargs="*",
                        help="Stock tickers to fetch")
    parser.add_argument("--from",          dest="from_date", default=None,
                        help="Start date YYYY-MM-DD (default: 2024-09-16)")
    parser.add_argument("--to",            dest="to_date",   default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--all",           action="store_true",
                        help="Fetch predefined ALL_TICKERS list (~70 key stocks)")
    parser.add_argument("--hose-hnx",      action="store_true",
                        help="Fetch all HOSE+HNX stocks from vndirect_listing.xlsx")
    parser.add_argument("--update",        action="store_true",
                        help="Incremental update: last 14 days for all stored parquets")
    parser.add_argument("--snapshot",      action="store_true",
                        help="Fast daily update: fetch all tickers at once for yesterday+today")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tickers that already have a parquet file")
    parser.add_argument("--workers",       type=int, default=1,
                        help="Number of parallel workers / browser sessions (default: 1)")
    parser.add_argument("--xlsx",          default=None,
                        help="Path to xlsx listing file")
    args = parser.parse_args()

    to_dt   = datetime.strptime(args.to_date,   "%Y-%m-%d") if args.to_date   else datetime.today()
    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d") if args.from_date else DATA_START
    workers = max(1, args.workers)

    # ── build ticker list ──────────────────────────────────────────────────────
    if args.snapshot:
        # Fast path: one multi-ticker request for yesterday + today
        from_dt = datetime.today() - timedelta(days=1)
        to_dt   = datetime.today()
        existing_tickers = {p.stem for p in OUTDIR.glob("*.parquet")}
        if not existing_tickers:
            print("No stored parquets found. Run a backfill first.")
            return
        print(f"Snapshot: all tickers {from_dt.date()} -> {to_dt.date()} "
              f"({len(existing_tickers)} parquets exist)")

        session    = _make_session()
        by_ticker  = fetch_snapshot_all(from_dt, to_dt, session)
        updated = skipped = 0
        for tkr, new_df in by_ticker.items():
            if tkr not in existing_tickers:
                skipped += 1
                continue
            path     = OUTDIR / f"{tkr}.parquet"
            existing = pd.read_parquet(path)
            existing["date"] = pd.to_datetime(existing["date"])
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = (combined
                        .drop_duplicates(subset=["date", "ticker"])
                        .sort_values("date")
                        .reset_index(drop=True))
            combined.to_parquet(path, index=False)
            updated += 1
        print(f"Snapshot done: {len(by_ticker)} tickers in response, "
              f"{updated} parquets updated, {skipped} new tickers skipped")
        return

    if args.update:
        tickers = [p.stem for p in sorted(OUTDIR.glob("*.parquet"))]
        if not tickers:
            print("No stored tickers found. Run a backfill first.")
            return
        from_dt = datetime.today() - timedelta(days=14)
        print(f"Updating {len(tickers)} stored tickers (last 14 days)...")

    elif args.hose_hnx:
        xlsx_path = Path(args.xlsx) if args.xlsx else None
        tickers   = load_hose_hnx_tickers(xlsx_path)

    elif args.all:
        tickers = ALL_TICKERS

    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]

    else:
        parser.print_help()
        return

    # ── skip-existing filter ───────────────────────────────────────────────────
    if args.skip_existing and not args.update:
        already = {p.stem for p in OUTDIR.glob("*.parquet")}
        before  = len(tickers)
        tickers = [t for t in tickers if t not in already]
        print(f"  --skip-existing: {before - len(tickers)} already done, "
              f"{len(tickers)} remaining")

    if not tickers:
        print("Nothing to fetch.")
        return

    total = len(tickers)
    print(f"\nFetching {total} tickers with {workers} worker(s): "
          f"{', '.join(tickers[:8])}{'...' if total > 8 else ''}")
    print(f"Date range: {from_dt.date()} → {to_dt.date()}\n")

    # ── warm up pyarrow+pandas in main thread before workers start ────────────
    # pd.to_datetime() and to_parquet() both trigger pyarrow extension type
    # registration; if two threads race on first registration it raises
    # "type extension already defined". One call here prevents all races.
    _w = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"], format="%Y-%m-%d"), "v": [0.0]})
    _w.to_parquet(io.BytesIO())
    del _w

    # ── stagger worker starts slightly to avoid burst ──────────────────────────
    # Each worker pulls from a shared queue via futures
    counters = {"ok": 0, "skipped": 0, "error": 0, "done": 0}
    failed_tickers: list[str] = []

    def _run_batch(ticker_list: list[str], label: str = ""):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for idx, ticker in enumerate(ticker_list):
                # Stagger submissions so workers don't all start simultaneously
                wid = idx % workers
                fut = pool.submit(
                    _worker_task, ticker, from_dt, to_dt,
                    args.skip_existing, args.update, wid
                )
                futures[fut] = ticker
                if idx < workers:
                    time.sleep(1.5)  # small ramp-up delay between first N workers

            for fut in as_completed(futures):
                ticker, status = fut.result()
                with _counter_lock:
                    counters["done"] += 1
                    if status == "saved":
                        counters["ok"] += 1
                    elif status == "error":
                        counters["error"] += 1
                        failed_tickers.append(ticker)
                    else:
                        counters["skipped"] += 1
                    n = counters["done"]
                if n % 50 == 0:
                    _tprint(f"\n  ── {label}Progress: {n}/{total} done "
                            f"({counters['ok']} saved, {counters['skipped']} empty, "
                            f"{counters['error']} errors) ──\n")

    # ── first pass ────────────────────────────────────────────────────────────
    _run_batch(tickers)

    # ── retry pass: re-run tickers that errored ────────────────────────────────
    if failed_tickers:
        print(f"\n  ── Retrying {len(failed_tickers)} failed tickers ──\n")
        retry_list = list(failed_tickers)
        failed_tickers.clear()
        counters["error"] = 0
        # Reset done counter for retry pass progress
        counters["done"] = 0
        total_retry = len(retry_list)
        original_total = total
        total = total_retry  # so progress % is relative to retry batch
        _run_batch(retry_list, label="RETRY ")
        total = original_total

    print(f"\nDone.  {counters['ok']} saved  |  "
          f"{counters['skipped']} empty/no-data  |  "
          f"{counters['error']} failed")


if __name__ == "__main__":
    main()
