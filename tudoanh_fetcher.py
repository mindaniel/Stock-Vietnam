import datetime as dt
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

import pandas as pd
import requests

BASE_URL = "https://dulieu.nguoiquansat.vn/History/TuDoanhHistory"
_THREAD_LOCAL = threading.local()

USER_PROFILES = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
    },
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept-Language": "en-GB,en;q=0.9,vi;q=0.7",
    },
    {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.8,vi;q=0.6",
    },
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Accept-Language": "en-US,en;q=0.7,vi;q=0.6",
    },
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Accept-Language": "en-US,en;q=0.8,vi;q=0.7",
    },
]


def get_session(profile_idx: int):
    if not hasattr(_THREAD_LOCAL, "session"):
        _THREAD_LOCAL.session = requests.Session()
    s = _THREAD_LOCAL.session
    prof = USER_PROFILES[profile_idx % len(USER_PROFILES)]
    s.headers.update({
        **prof,
        "Accept": "*/*",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://dulieu.nguoiquansat.vn/du-lieu-giao-dich",
        "Connection": "keep-alive",
    })
    return s


def vn_to_number(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "-"}:
        return None
    s = s.replace(",", "")
    try:
        n = float(s)
        return int(n) if n == int(n) else n
    except Exception:
        return None


def fetch_page(symbol: str, from_date: str, to_date: str, page: int = 1,
               ex_id: str = "", id_nganh: str = "", profile_idx: int = 0):
    ts = int(dt.datetime.now().timestamp() * 1000)
    params = {
        "page": page,
        "fromDate": from_date,
        "toDate": to_date,
        "exId": ex_id,
        "code": symbol,
        "idNganh": id_nganh,
        "_": ts,
    }
    session = get_session(profile_idx)
    html = ""
    last_err = None
    for attempt in range(3):
        try:
            r = session.get(BASE_URL, params=params, timeout=30)
            r.raise_for_status()
            html = r.text
            break
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1) + random.uniform(0.2, 0.8))
    if not html:
        if last_err:
            raise last_err
        return pd.DataFrame()
    if "no matching records found" in html.lower():
        return pd.DataFrame()

    tables = pd.read_html(StringIO(html))
    if not tables:
        return pd.DataFrame()

    df = tables[0].copy()

    # Flatten multi-index columns (Vietnamese chars may be garbled — use positional mapping)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x).strip() for x in col if str(x).strip() and str(x).strip().lower() != "nan"])
            for col in df.columns
        ]

    # Positional mapping — table layout is fixed:
    # 0:STT, 1:Date, 2:Symbol, 3:KL_buy, 4:KL_sell, 5:KL_net, 6:GT_buy, 7:GT_sell, 8:GT_net
    if df.shape[1] < 9:
        return pd.DataFrame()

    df.columns = ["_stt", "date", "symbol",
                  "buy_volume", "sell_volume", "net_volume",
                  "buy_value_bn", "sell_value_bn", "net_value_bn"]
    df = df.drop(columns=["_stt"])

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()

    for c in ["buy_volume", "sell_volume", "net_volume", "buy_value_bn", "sell_value_bn", "net_value_bn"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sell columns come through as negative from the website — store as absolute
    df["sell_volume"] = df["sell_volume"].abs()
    df["sell_value_bn"] = df["sell_value_bn"].abs()

    df = df[df["date"].notna()].copy()
    return df


def fetch_all_pages(symbol: str, from_date: str, to_date: str,
                    max_pages: int = 500, profile_idx: int = 0):
    all_pages = []
    seen_first_date = None
    for page in range(1, max_pages + 1):
        df = fetch_page(symbol, from_date, to_date, page=page, profile_idx=profile_idx)
        if df.empty:
            break
        first_date = df.iloc[0]["date"] if "date" in df.columns else None
        if page > 1 and first_date == seen_first_date:
            break
        seen_first_date = first_date
        all_pages.append(df)
        time.sleep(random.uniform(0.2, 0.6))
        if len(df) < 15:
            break

    if not all_pages:
        return pd.DataFrame()
    out = pd.concat(all_pages, ignore_index=True).drop_duplicates(subset=["date", "symbol"])
    out = out.sort_values(["symbol", "date"])
    return out


def fetch_backward_until_unavailable(symbol: str, end_date_ddmmyyyy: str,
                                     profile_idx: int = 0, window_days: int = 180):
    all_chunks = []
    end_dt = dt.datetime.strptime(end_date_ddmmyyyy, "%d/%m/%Y")
    empty_streak = 0
    for _ in range(80):
        start_dt = end_dt - dt.timedelta(days=window_days - 1)
        from_date = start_dt.strftime("%d/%m/%Y")
        to_date = end_dt.strftime("%d/%m/%Y")
        probe = fetch_page(symbol, from_date, to_date, page=1, profile_idx=profile_idx)
        if probe.empty:
            empty_streak += 1
            end_dt = start_dt - dt.timedelta(days=1)
            if empty_streak >= 3:
                break
            time.sleep(random.uniform(0.4, 1.0))
            continue
        empty_streak = 0
        chunk = fetch_all_pages(symbol, from_date, to_date, profile_idx=profile_idx)
        if chunk.empty:
            break
        all_chunks.append(chunk)
        end_dt = start_dt - dt.timedelta(days=1)
        time.sleep(random.uniform(0.4, 1.0))

    if not all_chunks:
        return pd.DataFrame()
    out = pd.concat(all_chunks, ignore_index=True)
    out = out.drop_duplicates(subset=["date", "symbol"])
    out = out.sort_values(["symbol", "date"])
    return out


def upsert_into_master(new_df: pd.DataFrame, master_path: str):
    """Merge new rows into the master tudoanh CSV, deduplicating on (symbol, date)."""
    # Normalise new data: convert bn values → raw VND to match existing schema
    out = new_df.copy()
    out["buy_value"] = (out["buy_value_bn"] * 1_000_000_000).round(0)
    out["sell_value"] = (out["sell_value_bn"] * 1_000_000_000).round(0)
    out["net_value"] = (out["net_value_bn"] * 1_000_000_000).round(0)
    out = out[["symbol", "buy_volume", "sell_volume", "buy_value",
               "sell_value", "net_volume", "net_value", "date"]]

    if os.path.exists(master_path):
        try:
            existing = pd.read_csv(master_path)
            # Normalise existing dates to YYYY-MM-DD for dedup
            existing["date"] = pd.to_datetime(existing["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            existing = existing[existing["date"].notna()]
            combined = pd.concat([existing, out], ignore_index=True)
            combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
            combined = combined.sort_values(["symbol", "date"])
        except Exception:
            combined = out
    else:
        combined = out

    os.makedirs(os.path.dirname(master_path), exist_ok=True)
    combined.to_csv(master_path, index=False, encoding="utf-8-sig")
    return combined


def get_symbols_from_data(data_dir: str, limit: int = None):
    syms = [f[:-8] for f in os.listdir(data_dir) if f.endswith(".parquet")]
    syms = [s for s in syms if re.fullmatch(r"[A-Z0-9]{2,6}", s)]
    syms = sorted(set(syms))
    return syms[:limit] if limit else syms


def main():
    import argparse
    p = argparse.ArgumentParser(description="Fetch tu doanh history from dulieu.nguoiquansat.vn")
    today = dt.datetime.now().strftime("%d/%m/%Y")
    p.add_argument("--from-date", default="01/01/2012", help="dd/mm/yyyy (default: 01/01/2012)")
    p.add_argument("--to-date", default=today, help=f"dd/mm/yyyy (default: {today})")
    p.add_argument("--symbol", default=None, help="Single symbol to fetch")
    p.add_argument("--limit", type=int, default=None, help="Max symbols in all-symbols mode")
    p.add_argument("--data-dir", default=os.path.join("data", "price"), help="Folder with per-symbol price parquets (for symbol list)")
    p.add_argument("--master-out", default="tudoanh/tudoanh_all.csv", help="Master CSV to upsert into")
    p.add_argument("--raw-out", default="results/tudoanh_history_raw.csv", help="Raw output CSV")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else get_symbols_from_data(args.data_dir, args.limit)
    os.makedirs("results", exist_ok=True)

    print(f"Symbols: {len(symbols)} | Date range: {args.from_date} -> {args.to_date} | Workers: {args.workers}")

    all_rows = []

    def process(task):
        i, sym = task
        try:
            profile_idx = i % len(USER_PROFILES)
            df = fetch_all_pages(sym, args.from_date, args.to_date, profile_idx=profile_idx)
            if df.empty:
                return sym, "no_data", None
            return sym, f"ok ({len(df)} rows)", df
        except Exception as e:
            return sym, f"error: {e}", None

    tasks = list(enumerate(symbols, 1))
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(process, t): t for t in tasks}
        for fut in as_completed(futures):
            i, sym = futures[fut]
            sym, status, df = fut.result()
            print(f"[{i}/{len(symbols)}] {sym}: {status}")
            if df is not None and not df.empty:
                all_rows.append(df)

    if not all_rows:
        print("No data fetched.")
        return

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(args.raw_out, index=False, encoding="utf-8-sig")
    print(f"Raw saved: {args.raw_out} ({len(full):,} rows)")

    combined = upsert_into_master(full, args.master_out)
    print(f"Master updated: {args.master_out} ({len(combined):,} rows, {combined['symbol'].nunique()} symbols)")


if __name__ == "__main__":
    main()
