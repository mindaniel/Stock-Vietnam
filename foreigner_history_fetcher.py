import argparse
import datetime as dt
import os
import random
import re
import threading
import time
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests


BASE_URL = "https://dulieu.nguoiquansat.vn/History/KLNDTNNHistory"
_THREAD_LOCAL = threading.local()

USER_PROFILES = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    },
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
        "Accept-Language": "en-GB,en;q=0.9",
    },
    {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.8",
    },
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
        "Accept-Language": "en-US,en;q=0.7",
    },
]


def get_session(profile_idx: int):
    if not hasattr(_THREAD_LOCAL, "session"):
        _THREAD_LOCAL.session = requests.Session()
    s = _THREAD_LOCAL.session
    prof = USER_PROFILES[profile_idx % len(USER_PROFILES)]
    s.headers.update(
        {
            **prof,
            "Accept": "*/*",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://dulieu.nguoiquansat.vn/du-lieu-giao-dich",
            "Connection": "keep-alive",
        }
    )
    return s


def coalesce_columns(df: pd.DataFrame, base_name: str) -> pd.DataFrame:
    candidates = [c for c in df.columns if c == base_name or c.startswith(base_name + "_")]
    if not candidates:
        return df
    if len(candidates) == 1:
        if candidates[0] != base_name:
            df = df.rename(columns={candidates[0]: base_name})
        return df
    df[base_name] = df[candidates].bfill(axis=1).iloc[:, 0]
    drop_cols = [c for c in candidates if c != base_name]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def vn_to_number(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "-"}:
        return None
    s = s.replace(",", "")
    try:
        n = float(s)
        return int(n) if n.is_integer() else n
    except Exception:
        return None


def normalize_name(name: str) -> str:
    s = str(name).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def map_columns(columns):
    mapped = {}
    seen = {}
    for c in columns:
        n = normalize_name(c)
        target = None
        if "ng" in n and "y" in n:
            target = "date"
        elif "m ck" in n or "ma ck" in n:
            target = "symbol"
        elif "nn s h" in n or "nn so huu" in n or "so huu" in n:
            target = "foreign_ownership_pct"
        elif "tong khoi luong mua" in n or ("mua" in n and "khoi" in n and "luong" in n):
            target = "foreign_buy_volume"
        elif "tong khoi luong ban" in n or ("ban" in n and "khoi" in n and "luong" in n):
            target = "foreign_sell_volume"
        elif "tong gia tri mua" in n or ("mua" in n and "gia" in n and "tri" in n):
            target = "foreign_buy_value_bn"
        elif "tong gia tri ban" in n or ("ban" in n and "gia" in n and "tri" in n):
            target = "foreign_sell_value_bn"
        elif "chenh lech kl" in n or ("chenh" in n and "kl" in n):
            target = "net_volume"
        elif "chenh lech gia tri" in n or ("chenh" in n and "gia" in n and "tri" in n):
            target = "net_value_bn"

        if target:
            seen[target] = seen.get(target, 0) + 1
            if seen[target] > 1:
                target = f"{target}_{seen[target]}"
            mapped[c] = target
    return mapped


def fetch_page(symbol: str, from_date: str, to_date: str, page: int = 1, ex_id: str = "", id_nganh: str = "", profile_idx: int = 0):
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
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x).strip() for x in col if str(x).strip() and str(x).strip().lower() != "nan"])
            for col in df.columns
        ]

    df = df.rename(columns=map_columns(df.columns))

    # Positional fallback for this endpoint's fixed table layout
    # Expected columns: STT, Ngay, Ma CK, %NN so huu, KL Mua, KL Ban, GT Mua, GT Ban, Chenh KL, Chenh GT
    if "date" not in df.columns and df.shape[1] >= 2:
        df = df.rename(columns={df.columns[1]: "date"})
    if "symbol" not in df.columns and df.shape[1] >= 3:
        df = df.rename(columns={df.columns[2]: "symbol"})
    if "foreign_ownership_pct" not in df.columns and df.shape[1] >= 4:
        df = df.rename(columns={df.columns[3]: "foreign_ownership_pct"})
    if "foreign_buy_volume" not in df.columns and df.shape[1] >= 5:
        df = df.rename(columns={df.columns[4]: "foreign_buy_volume"})
    if "foreign_sell_volume" not in df.columns and df.shape[1] >= 6:
        df = df.rename(columns={df.columns[5]: "foreign_sell_volume"})
    if "foreign_buy_value_bn" not in df.columns and df.shape[1] >= 7:
        df = df.rename(columns={df.columns[6]: "foreign_buy_value_bn"})
    if "foreign_sell_value_bn" not in df.columns and df.shape[1] >= 8:
        df = df.rename(columns={df.columns[7]: "foreign_sell_value_bn"})
    if "net_volume" not in df.columns and df.shape[1] >= 9:
        df = df.rename(columns={df.columns[8]: "net_volume"})
    if "net_value_bn" not in df.columns and df.shape[1] >= 10:
        df = df.rename(columns={df.columns[9]: "net_value_bn"})
    if "symbol" not in df.columns:
        matches = re.findall(r">\s*([A-Z]{2,5})\s*<", html)
        if matches:
            df["symbol"] = matches[: len(df)]

    # de-duplicate mapped fallbacks
    for base in ["foreign_buy_volume", "foreign_sell_volume", "foreign_buy_value_bn", "foreign_sell_value_bn", "net_volume", "net_value_bn"]:
        dupes = [c for c in df.columns if c == base or c.startswith(base + "_")]
        if len(dupes) > 1 and base not in df.columns:
            df[base] = df[dupes[0]]

    keep = [
        "date", "symbol", "foreign_ownership_pct",
        "foreign_buy_volume", "foreign_sell_volume",
        "foreign_buy_value_bn", "foreign_sell_value_bn",
        "net_volume", "net_value_bn",
    ]
    exists = [c for c in keep if c in df.columns]
    if exists:
        df = df[exists]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()

    for c in [
        "foreign_ownership_pct", "foreign_buy_volume", "foreign_sell_volume",
        "foreign_buy_value_bn", "foreign_sell_value_bn", "net_volume", "net_value_bn",
    ]:
        if c in df.columns:
            df[c] = df[c].apply(vn_to_number)

    if "date" in df.columns:
        df = df[df["date"].notna()].copy()
    return df


def fetch_all_pages(symbol: str, from_date: str, to_date: str, max_pages: int = 500, profile_idx: int = 0):
    all_pages = []
    seen_first_date = None
    for page in range(1, max_pages + 1):
        df = fetch_page(symbol, from_date, to_date, page=page, profile_idx=profile_idx)
        if df.empty:
            break
        first_date = df.iloc[0]["date"] if "date" in df.columns and not df.empty else None
        if page > 1 and first_date == seen_first_date:
            break
        seen_first_date = first_date
        all_pages.append(df)
        time.sleep(random.uniform(0.2, 0.6))
        if len(df) < 15:
            break

    if not all_pages:
        return pd.DataFrame()
    out = pd.concat(all_pages, ignore_index=True).drop_duplicates(subset=[c for c in ["date", "symbol"] if c in all_pages[0].columns])
    out = out.sort_values([c for c in ["symbol", "date"] if c in out.columns])
    return out


def fetch_backward_until_unavailable(symbol: str, end_date_ddmmyyyy: str, profile_idx: int = 0, window_days: int = 180):
    """
    Keep moving date windows backward until site returns "No matching records found".
    """
    all_chunks = []
    end_dt = dt.datetime.strptime(end_date_ddmmyyyy, "%d/%m/%Y")

    empty_streak = 0
    for _ in range(80):  # safety cap
        start_dt = end_dt - dt.timedelta(days=window_days - 1)
        from_date = start_dt.strftime("%d/%m/%Y")
        to_date = end_dt.strftime("%d/%m/%Y")

        # probe first page to detect website limit
        probe = fetch_page(symbol, from_date, to_date, page=1, profile_idx=profile_idx)
        if probe.empty:
            empty_streak += 1
            # step further back to find first available range (avoid stopping too early)
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

        # move older
        end_dt = start_dt - dt.timedelta(days=1)
        time.sleep(random.uniform(0.4, 1.0))

    if not all_chunks:
        return pd.DataFrame()
    out = pd.concat(all_chunks, ignore_index=True)
    out = out.drop_duplicates(subset=[c for c in ["date", "symbol"] if c in out.columns])
    out = out.sort_values([c for c in ["symbol", "date"] if c in out.columns])
    return out


def merge_ownership_into_price_file(symbol: str, foreign_df: pd.DataFrame, data_dir: str):
    fp = os.path.join(data_dir, f"{symbol}.parquet")
    if not os.path.exists(fp):
        return None
    px = pd.read_parquet(fp)
    for b in ["foreign_ownership_pct", "foreign_buy_value_bn", "foreign_sell_value_bn"]:
        px = coalesce_columns(px, b)
    if "time" not in px.columns:
        return None
    px["time"] = pd.to_datetime(px["time"], errors="coerce").dt.strftime("%Y-%m-%d")

    f = foreign_df.copy()
    f = f[f["symbol"] == symbol]
    if f.empty or "date" not in f.columns or "foreign_ownership_pct" not in f.columns:
        return None
    wanted = ["date", "foreign_ownership_pct",
              "foreign_buy_volume", "foreign_sell_volume",
              "foreign_buy_value_bn", "foreign_sell_value_bn"]
    existing = [c for c in wanted if c in f.columns]
    if "date" not in existing or "foreign_ownership_pct" not in existing:
        return None
    f = f[existing].drop_duplicates(subset=["date"], keep="last")

    # Rename API volume cols to avoid collision; we'll fill-in-place below
    f = f.rename(columns={
        "foreign_buy_volume": "_api_buy_vol",
        "foreign_sell_volume": "_api_sell_vol",
    })

    # Drop cols that come wholesale from the API (ownership + bn values)
    for c in ["foreign_ownership_pct", "foreign_buy_value_bn", "foreign_sell_value_bn"]:
        if c in px.columns:
            px = px.drop(columns=[c])

    merged = px.merge(f, left_on="time", right_on="date", how="left")
    merged = merged.drop(columns=["date"])

    # Fill foreign_buy_vol / foreign_sell_vol only where currently missing
    for old_col, api_col in [("foreign_buy_vol", "_api_buy_vol"), ("foreign_sell_vol", "_api_sell_vol")]:
        if api_col not in merged.columns:
            continue
        if old_col in merged.columns:
            mask = merged[old_col].isna() & merged[api_col].notna()
            merged.loc[mask, old_col] = merged.loc[mask, api_col]
        else:
            merged = merged.rename(columns={api_col: old_col})
            api_col = None
        if api_col and api_col in merged.columns:
            merged = merged.drop(columns=[api_col])

    # Derive foreign_buy_val / foreign_sell_val from bn cols where missing
    # existing data uses thousands-VND; foreign_buy_value_bn is billions VND → × 1,000,000
    for val_col, bn_col in [("foreign_buy_val", "foreign_buy_value_bn"), ("foreign_sell_val", "foreign_sell_value_bn")]:
        if bn_col not in merged.columns:
            continue
        if val_col in merged.columns:
            mask = merged[val_col].isna() & merged[bn_col].notna()
            merged.loc[mask, val_col] = (merged.loc[mask, bn_col] * 1_000_000).round(2)
        else:
            merged[val_col] = (merged[bn_col] * 1_000_000).round(2)

    merged.to_parquet(fp, index=False, engine="pyarrow")
    return merged


def scale_diagnostics(merged_df: pd.DataFrame, symbol: str):
    if merged_df is None:
        return []
    req_base = ["time"]
    if any(c not in merged_df.columns for c in req_base):
        return []

    has_buy = all(c in merged_df.columns for c in ["foreign_buy_val", "foreign_buy_value_bn"])
    has_sell = all(c in merged_df.columns for c in ["foreign_sell_val", "foreign_sell_value_bn"])
    if not has_buy and not has_sell:
        return []

    d = merged_df.copy()
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d = d.sort_values("time", ascending=False).head(20)
    subset = []
    if has_buy:
        subset += ["foreign_buy_val", "foreign_buy_value_bn"]
    if has_sell:
        subset += ["foreign_sell_val", "foreign_sell_value_bn"]
    d = d.dropna(subset=subset)
    rows = []
    for _, r in d.iterrows():
        b_ratio = None
        s_ratio = None
        if has_buy and r["foreign_buy_value_bn"] not in [0, None]:
            b_ratio = r["foreign_buy_val"] / r["foreign_buy_value_bn"]
        if has_sell and r["foreign_sell_value_bn"] not in [0, None]:
            s_ratio = r["foreign_sell_val"] / r["foreign_sell_value_bn"]
        rows.append({
            "symbol": symbol,
            "date": r["time"].strftime("%Y-%m-%d") if pd.notna(r["time"]) else None,
            "buy_ratio_old_over_new": b_ratio,
            "sell_ratio_old_over_new": s_ratio,
            "old_buy_val": r.get("foreign_buy_val"),
            "new_buy_val_bn": r.get("foreign_buy_value_bn"),
            "old_sell_val": r.get("foreign_sell_val"),
            "new_sell_val_bn": r.get("foreign_sell_value_bn"),
        })
    return rows


def get_symbols_from_data(data_dir: str, limit: int = None):
    syms = [f[:-8] for f in os.listdir(data_dir) if f.endswith(".parquet")]
    syms = [s for s in syms if re.fullmatch(r"[A-Z0-9]{2,6}", s)]
    syms = sorted(set(syms))
    return syms[:limit] if limit else syms


def detect_backfill_end_before_existing(symbol: str, data_dir: str):
    """
    Find the end date for backfill (day before first existing foreign data date).
    Priority:
      1) foreign_buy_val / foreign_sell_val (legacy foreign data)
      2) foreign_ownership_pct
    Returns (to_date_ddmmyyyy, source_col) or None.
    """
    fp = os.path.join(data_dir, f"{symbol}.parquet")
    if not os.path.exists(fp):
        return None
    df = pd.read_parquet(fp)
    df = coalesce_columns(df, "foreign_ownership_pct")
    if "time" not in df.columns:
        return None

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df[df["time"].notna()].copy()
    if df.empty:
        return None

    candidate = None
    source_col = None

    trade_cols = [c for c in ["foreign_buy_val", "foreign_sell_val"] if c in df.columns]
    if trade_cols:
        tmp = pd.DataFrame({"time": df["time"]})
        ok = pd.Series(False, index=df.index)
        for c in trade_cols:
            v = pd.to_numeric(df[c], errors="coerce")
            ok = ok | v.notna()
        d = df[ok]
        if not d.empty:
            candidate = d["time"].min()
            source_col = "/".join(trade_cols)

    if candidate is None and "foreign_ownership_pct" in df.columns:
        v = pd.to_numeric(df["foreign_ownership_pct"], errors="coerce")
        d = df[v.notna()]
        if not d.empty:
            candidate = d["time"].min()
            source_col = "foreign_ownership_pct"

    if candidate is None:
        return None

    to_dt = candidate - dt.timedelta(days=1)
    return to_dt.strftime("%d/%m/%Y"), source_col


def main():
    p = argparse.ArgumentParser(description="Backfill foreign ownership and merge into existing data/*.csv")
    today = dt.datetime.now().strftime("%d/%m/%Y")
    p.add_argument("--from-date", default="01/01/2012", help="dd/mm/yyyy (default: 01/01/2012)")
    p.add_argument("--to-date", default=today, help=f"dd/mm/yyyy (default: {today})")
    p.add_argument("--symbol", default=None, help="Single symbol")
    p.add_argument("--all-symbols", action="store_true", help="Process all symbols from data/*.csv")
    p.add_argument("--limit", type=int, default=None, help="Limit symbols in all-symbols mode")
    p.add_argument("--data-dir", default=os.path.join("data", "price"))
    p.add_argument("--raw-out", default="results/foreigner_history_raw.csv")
    p.add_argument("--audit-out", default="results/foreigner_unit_audit.csv")
    p.add_argument("--workers", type=int, default=6, help="Parallel symbol workers")
    p.add_argument("--before-existing-only", action="store_true", help="Fetch only dates before first existing foreign data in each file")
    args = p.parse_args()

    if not args.symbol and not args.all_symbols:
        # no args mode: process all symbols in data folder
        args.all_symbols = True

    symbols = [args.symbol.upper()] if args.symbol else get_symbols_from_data(args.data_dir, args.limit)
    os.makedirs("results", exist_ok=True)

    all_foreign = []
    audit_rows = []
    updated = 0

    print(f"Mode: {'ALL SYMBOLS' if args.all_symbols else 'SINGLE SYMBOL'}")
    print(f"Date range: {args.from_date} -> {args.to_date}")
    print(f"Workers: {args.workers}")
    print(f"Before-existing-only: {args.before_existing_only}")

    def process_symbol(task):
        i, sym = task
        try:
            profile_idx = i % len(USER_PROFILES)
            from_date = args.from_date
            to_date = args.to_date
            if args.before_existing_only:
                backfill_info = detect_backfill_end_before_existing(sym, args.data_dir)
                if not backfill_info:
                    return sym, "skip_no_backfill_window", None, []
                backfill_to, source_col = backfill_info
                fdf = fetch_backward_until_unavailable(sym, backfill_to, profile_idx=profile_idx)
                status_suffix = f"(before {backfill_to} by {source_col})"
            else:
                fdf = fetch_all_pages(sym, from_date, to_date, profile_idx=profile_idx)
                status_suffix = ""
            if fdf.empty:
                return sym, f"no_data {status_suffix}".strip(), None, []
            merged = merge_ownership_into_price_file(sym, fdf, args.data_dir)
            if merged is None:
                return sym, f"fetched_not_merged {status_suffix}".strip(), fdf, []
            return sym, f"merged {status_suffix}".strip(), fdf, scale_diagnostics(merged, sym)
        except Exception as e:
            return sym, f"error: {e}", None, []

    tasks = list(enumerate(symbols, 1))
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(process_symbol, t): t for t in tasks}
        for fut in as_completed(futures):
            i, sym = futures[fut]
            sym, status, fdf, aud = fut.result()
            print(f"[{i}/{len(symbols)}] {sym}: {status}")
            if fdf is not None and not fdf.empty:
                all_foreign.append(fdf)
            if status == "merged":
                updated += 1
            if aud:
                audit_rows.extend(aud)

    if all_foreign:
        full = pd.concat(all_foreign, ignore_index=True)
        full.to_csv(args.raw_out, index=False, encoding="utf-8-sig")
        print(f"Raw saved: {args.raw_out} ({len(full):,} rows)")
    if audit_rows:
        adf = pd.DataFrame(audit_rows)
        adf.to_csv(args.audit_out, index=False, encoding="utf-8-sig")
        print(f"Audit saved: {args.audit_out} ({len(adf):,} rows)")
        if not adf.empty:
            print("Median buy ratio old/new:", adf["buy_ratio_old_over_new"].dropna().median())
            print("Median sell ratio old/new:", adf["sell_ratio_old_over_new"].dropna().median())

    print(f"Done. Updated files: {updated}/{len(symbols)}")


if __name__ == "__main__":
    main()
