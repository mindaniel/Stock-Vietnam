"""
backfill_prices.py — fill missing OHLCV rows in data/price/*.parquet
                     using VNDirect dchart API (historical data).

Usage:
    python backfill_prices.py                        # fill yesterday for all tickers
    python backfill_prices.py 2026-06-13             # fill a specific date
    python backfill_prices.py 2026-06-10 2026-06-15  # fill a date range
    python backfill_prices.py --overwrite            # overwrite existing rows too (fix stale data)
    python backfill_prices.py --vnindex              # also update VNINDEX.csv

The VPS realtime API only returns current session data.
This script uses VNDirect dchart API for confirmed historical OHLCV.

NOTE: prices from dchart are in thousands of VND (e.g. 22.832 = 22,832 VND),
which should match the VPS API scale. If the existing parquet shows a flat
close for several consecutive days, those rows are likely stale VPS snapshots
— run with --overwrite to replace them with correct dchart values.
"""
import os
import sys
import requests
import pandas as pd
import datetime
import time

try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.getcwd()

PRICE_DIR  = os.path.join(BASE_DIR, "data", "price")
VNINDEX_CSV = os.path.join(BASE_DIR, "VNINDEX.csv")
HEADERS = {"User-Agent": "Mozilla/5.0"}
SLEEP_BETWEEN = 0.15   # seconds between API calls


def fetch_dchart(symbol, start_date, end_date):
    """
    Fetch daily OHLCV from VNDirect dchart API for symbol between two dates.
    Returns a DataFrame with columns: time(str YYYY-MM-DD), open, high, low,
    close, volume, value  — or None on error.
    """
    start_ts = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts   = int(datetime.datetime.combine(
        datetime.datetime.strptime(end_date, "%Y-%m-%d").date(),
        datetime.time(23, 59, 59)
    ).timestamp())
    url = (
        "https://dchart-api.vndirect.com.vn/dchart/history"
        f"?resolution=1D&symbol={symbol}&from={start_ts}&to={end_ts}"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        js = r.json()
        if js.get("s") != "ok" or "t" not in js:
            return None
        df = pd.DataFrame({
            "time":   [datetime.datetime.fromtimestamp(x).strftime("%Y-%m-%d") for x in js["t"]],
            "open":   js["o"],
            "high":   js["h"],
            "low":    js["l"],
            "close":  js["c"],
            "volume": js["v"],
        })
        df["value"] = df["close"] * df["volume"]
        return df
    except Exception as e:
        print(f"  ERROR {symbol}: {e}")
        return None


def last_trading_day():
    """Most recent weekday before today (Vietnam calendar, no holiday check)."""
    d = datetime.date.today() - datetime.timedelta(days=1)
    while d.weekday() >= 5:
        d -= datetime.timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def weekdays_in_range(start, end):
    """All weekday dates between start and end (inclusive), as YYYY-MM-DD strings."""
    d   = datetime.datetime.strptime(start, "%Y-%m-%d").date()
    end = datetime.datetime.strptime(end,   "%Y-%m-%d").date()
    out = []
    while d <= end:
        if d.weekday() < 5:
            out.append(d.strftime("%Y-%m-%d"))
        d += datetime.timedelta(days=1)
    return out


def backfill_price_parquets(start_date, end_date, overwrite=False):
    parquets = sorted(f for f in os.listdir(PRICE_DIR) if f.endswith(".parquet"))
    print(f"  {len(parquets)} tickers in data/price/")

    target_weekdays = set(weekdays_in_range(start_date, end_date))
    updated = skipped = failed = 0

    for fname in parquets:
        symbol = fname.replace(".parquet", "")
        fpath  = os.path.join(PRICE_DIR, fname)

        try:
            old_df = pd.read_parquet(fpath)
        except Exception:
            failed += 1
            continue

        # Normalize: some old parquets use "date" instead of "time"
        date_col = "time" if "time" in old_df.columns else "date" if "date" in old_df.columns else None
        if date_col is None:
            failed += 1
            continue
        if date_col == "date":
            old_df = old_df.rename(columns={"date": "time"})

        existing_dates = set(old_df["time"].astype(str))

        if overwrite:
            # Overwrite any date in range (remove existing rows first)
            dates_to_fetch = target_weekdays
        else:
            # Only add dates that are not yet present
            dates_to_fetch = target_weekdays - existing_dates

        if not dates_to_fetch:
            skipped += 1
            continue

        new_df = fetch_dchart(symbol, start_date, end_date)
        if new_df is None or new_df.empty:
            print(f"  WARN {symbol}: API returned no data")
            failed += 1
            time.sleep(SLEEP_BETWEEN)
            continue

        new_rows = new_df[new_df["time"].isin(dates_to_fetch)].copy()
        if new_rows.empty:
            skipped += 1
            time.sleep(SLEEP_BETWEEN)
            continue

        # Add foreign-flow columns as 0 (dchart doesn't provide them)
        for col in ["foreign_buy_vol", "foreign_sell_vol",
                    "foreign_buy_val", "foreign_sell_val", "foreign_room"]:
            if col not in new_rows.columns:
                new_rows[col] = 0

        # Reorder to match existing parquet columns
        existing_cols = list(old_df.columns)
        for col in new_rows.columns:
            if col not in existing_cols:
                existing_cols.append(col)
        new_rows = new_rows.reindex(columns=existing_cols)

        if overwrite:
            # Drop existing rows for dates we're about to replace
            old_df = old_df[~old_df["time"].astype(str).isin(dates_to_fetch)]

        combined = (pd.concat([old_df, new_rows], ignore_index=True)
                      .drop_duplicates(subset=["time"])
                      .sort_values("time")
                      .reset_index(drop=True))
        # Write via temp file + rename to avoid OneDrive lock conflicts
        tmp_path = fpath + ".tmp"
        write_ok = False
        for attempt in range(3):
            try:
                combined.to_parquet(tmp_path, index=False, engine="pyarrow")
                os.replace(tmp_path, fpath)
                write_ok = True
                break
            except Exception:
                time.sleep(1.0)
        if write_ok:
            updated += 1
        else:
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass
            print(f"  SKIP {symbol}: file locked after 3 retries, will update next run")
            failed += 1
        time.sleep(SLEEP_BETWEEN)

    return updated, skipped, failed


def backfill_vnindex(start_date, end_date):
    """Append missing VNINDEX rows to VNINDEX.csv via dchart API."""
    if not os.path.exists(VNINDEX_CSV):
        print("  WARN VNINDEX.csv not found, skipping.")
        return

    old = pd.read_csv(VNINDEX_CSV)
    existing = set(old["date"].astype(str))
    target   = set(weekdays_in_range(start_date, end_date))
    missing  = target - existing
    if not missing:
        print("  VNINDEX: already up to date.")
        return

    df = fetch_dchart("VNINDEX", start_date, end_date)
    if df is None or df.empty:
        print("  WARN VNINDEX: API returned no data.")
        return

    df = df.rename(columns={"time": "date"})
    new_rows = df[df["date"].isin(missing)]
    combined = (pd.concat([old, new_rows], ignore_index=True)
                  .drop_duplicates(subset=["date"])
                  .sort_values("date")
                  .reset_index(drop=True))
    combined.to_csv(VNINDEX_CSV, index=False)
    print(f"  VNINDEX: added {len(new_rows)} row(s).")


def main():
    flags     = [a for a in sys.argv[1:] if a.startswith("--")]
    args      = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_vnindex = "--vnindex" in flags
    overwrite  = "--overwrite" in flags

    if len(args) >= 2:
        start_date, end_date = args[0], args[1]
    elif len(args) == 1:
        start_date = end_date = args[0]
    else:
        start_date = end_date = last_trading_day()

    mode = "overwrite + fill" if overwrite else "fill missing only"
    print(f"\nBackfill prices [{mode}]: {start_date} to {end_date}")
    print(f"Source: VNDirect dchart API (historical)\n")

    updated, skipped, failed = backfill_price_parquets(start_date, end_date, overwrite=overwrite)
    print(f"\n  Price parquets: OK {updated} updated | -- {skipped} already current | ERR {failed} failed")

    if do_vnindex:
        print()
        backfill_vnindex(start_date, end_date)

    print()


if __name__ == "__main__":
    main()
