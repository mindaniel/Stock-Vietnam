import os
import sys
import glob
import requests
import pandas as pd
import datetime as dt
import json
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import RequestException, Timeout

# Ensure Windows terminal can print Unicode logs safely
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ==============================================================================
# CẤU HÌNH CHUNG & ĐƯỜNG DẪN
# ==============================================================================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

sys.path.insert(0, os.path.join(BASE_DIR, "lib"))

DATA_DIR  = os.path.join(BASE_DIR, "data")
PRICE_DIR = os.path.join(DATA_DIR, "price")
PUT_DIR = os.path.join(BASE_DIR, "putthrough")
TD_DIR = os.path.join(BASE_DIR, "tudoanh")

os.makedirs(PRICE_DIR, exist_ok=True)
os.makedirs(PUT_DIR, exist_ok=True)
os.makedirs(TD_DIR, exist_ok=True)

VN_TZ = dt.timezone(dt.timedelta(hours=7))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Origin": "https://banggia.vps.com.vn",
    "Referer": "https://banggia.vps.com.vn/",
    "Accept": "application/json, text/plain, */*",
    "Connection": "keep-alive"
}

# ==============================================================================
# TELEGRAM NOTIFICATION CONFIG
# ==============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


def send_telegram_message(message):
    """Send a Telegram message when bot token/chat id are configured."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  Telegram chưa cấu hình. Bỏ qua gửi thông báo.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": int(TELEGRAM_CHAT_ID), "text": message, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        print("✅ Đã gửi thông báo Telegram.")
    except Exception as e:
        print(f"❌ Gửi Telegram thất bại: {e}")


# ==============================================================================
# SHARED HELPERS
# ==============================================================================
def get_logical_now():
    """Adjust current time for late-night/early-morning runs."""
    now_vn = dt.datetime.now(VN_TZ)
    return now_vn - dt.timedelta(days=1) if now_vn.hour < 8 else now_vn


def get_today_str():
    return get_logical_now().strftime("%Y-%m-%d")


def is_weekend():
    return get_logical_now().weekday() >= 5


def fetch_with_retry(url, max_retries=3, timeout=30):
    """Fetch URL with retry logic and exponential backoff."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"   ⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    return None


def fetch_json_with_fallback(urls, max_retries=3, connect_timeout=10, read_timeout=45):
    """Fetch JSON from fallback URLs with retry/backoff."""
    last_error = None
    for attempt in range(max_retries):
        for url in urls:
            try:
                r = requests.get(url, headers=HEADERS, timeout=(connect_timeout, read_timeout))
                r.raise_for_status()
                try:
                    return r.json()
                except Exception:
                    return json.loads(r.text)
            except Timeout as e:
                last_error = e
                print(f"   ⚠️ Timeout ({url}) on attempt {attempt + 1}: {e}")
            except RequestException as e:
                last_error = e
                print(f"   ⚠️ Request error ({url}) on attempt {attempt + 1}: {e}")
            except Exception as e:
                last_error = e
                print(f"   ⚠️ Unexpected error ({url}) on attempt {attempt + 1}: {e}")
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 5
            print(f"   ↻ Retrying all endpoints in {wait_time}s...")
            time.sleep(wait_time)
    raise RuntimeError(f"Không thể lấy dữ liệu sau {max_retries} lần thử. Lỗi cuối: {last_error}")


def _update_parquet_safe(filepath, new_row_dict, symbol, existing_files, date_col="time"):
    """Thread-safe parquet update helper."""
    new_row_df = pd.DataFrame([new_row_dict])
    try:
        if symbol in existing_files:
            old_df = pd.read_parquet(filepath)
            if not old_df.empty:
                last_row = old_df.iloc[-1]
                date_val = str(new_row_dict.get(date_col, ""))
                if (str(last_row.get(date_col, "")) == date_val and
                    float(new_row_dict.get("volume", 0)) == float(last_row.get("volume", 0)) and
                    float(new_row_dict.get("close", 0)) == float(last_row.get("close", 0))):
                    return False
                if date_val in old_df[date_col].values:
                    old_df = old_df[old_df[date_col] != date_val]
            updated_df = pd.concat([old_df, new_row_df], ignore_index=True)
        else:
            updated_df = new_row_df
        updated_df.to_parquet(filepath, index=False, engine="pyarrow")
        return True
    except Exception:
        return False


# ==============================================================================
# JOB 1: CẬP NHẬT GIÁ & NƯỚC NGOÀI (SNAPSHOT) — PARALLELIZED
# ==============================================================================
def _fetch_vps_chunk(chunk_symbols):
    """Fetch one chunk of symbols from VPS API."""
    url = f"https://bgapidatafeed.vps.com.vn/getliststockdata/{','.join(chunk_symbols)}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except Exception:
            return json.loads(r.text)
    except Exception:
        return None


def _fetch_vps_symbols(exchange):
    """Fetch symbol list for one exchange."""
    url = f"https://bgapidatafeed.vps.com.vn/getlistckindex/{exchange}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        data = json.loads(r.text)
        return [s for s in data if isinstance(s, str)]
    except Exception:
        return []


def _fetch_ssi_exchange(exchange):
    """Fetch SSI data for one exchange."""
    ssi_headers = {
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/149.0.0.0 Safari/537.36",
        "Referer": "https://iboard.ssi.com.vn/"
    }
    url = f"https://iboard-query.ssi.com.vn/stock/exchange/{exchange}"
    try:
        r = requests.get(url, headers=ssi_headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            return data.get("data", data) if isinstance(data, dict) else data
    except Exception:
        pass
    return []


def _process_ssi_data(all_ssi_data):
    """Convert SSI data to standard format."""
    df_ssi = pd.DataFrame(all_ssi_data)
    df = pd.DataFrame()
    df["symbol"] = df_ssi["stockSymbol"]
    df["date"] = get_today_str()
    
    for col, ssi_col in [("open", "openPrice"), ("high", "highest"), ("low", "lowest"), ("close", "matchedPrice")]:
        df[col] = pd.to_numeric(df_ssi.get(ssi_col, 0), errors="coerce").fillna(0) / 1000.0
    
    df["volume"] = pd.to_numeric(df_ssi.get("nmTotalTradedQty", 0), errors="coerce").fillna(0)
    df["value"] = df["close"] * df["volume"]
    df["foreign_buy_vol"] = pd.to_numeric(df_ssi.get("buyForeignQtty", 0), errors="coerce").fillna(0)
    df["foreign_sell_vol"] = pd.to_numeric(df_ssi.get("sellForeignQtty", 0), errors="coerce").fillna(0)
    df["foreign_buy_val"] = pd.to_numeric(df_ssi.get("buyForeignValue", 0), errors="coerce").fillna(0) / 1000.0
    df["foreign_sell_val"] = pd.to_numeric(df_ssi.get("sellForeignValue", 0), errors="coerce").fillna(0) / 1000.0
    df["foreign_room"] = pd.to_numeric(df_ssi.get("remainForeignQtty", 0), errors="coerce").fillna(0)
    return df


def job_update_prices():
    print("\n--- [1/5] CAP NHAT GIA & NUOC NGOAI ---")
    if is_weekend():
        print("⛔ Hôm nay là cuối tuần. Bỏ qua.")
        return False

    # Fetch symbols in parallel
    with ThreadPoolExecutor(max_workers=3) as pool:
        symbol_lists = list(pool.map(_fetch_vps_symbols, ["hose", "hnx", "upcom"]))
    
    symbols = list(set(s for sublist in symbol_lists for s in sublist))
    
    if symbols:
        chunk_size = 100
        chunks = [symbols[i:i+chunk_size] for i in range(0, len(symbols), chunk_size)]
        print(f"⏳ Đang tải dữ liệu cho {len(symbols)} mã qua {len(chunks)} chunks (song song)...")
        
        all_vps_data = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(_fetch_vps_chunk, chunks))
        
        vps_failed = any(r is None for r in results)
        if not vps_failed:
            for r in results:
                all_vps_data.extend(r)
    else:
        vps_failed = True

    df = pd.DataFrame()
    if not vps_failed and all_vps_data:
        raw_df = pd.DataFrame(all_vps_data)
        rename_map = {
            "sym": "symbol", "lastPrice": "close", "openPrice": "open",
            "highPrice": "high", "lowPrice": "low", "avePrice": "average",
            "lot": "lot", "fBVol": "foreign_buy_vol", "fSVolume": "foreign_sell_vol",
            "fBValue": "foreign_buy_val", "fSValue": "foreign_sell_val", "fRoom": "foreign_room"
        }
        df = raw_df.rename(columns={k: v for k, v in rename_map.items() if k in raw_df.columns})
        df["date"] = get_today_str()
        df["volume"] = pd.to_numeric(df.get("lot", 0), errors="coerce").fillna(0) * 10
        
        numeric_cols = ["open", "high", "low", "close", "foreign_buy_vol", 
                       "foreign_sell_vol", "foreign_buy_val", "foreign_sell_val", "foreign_room"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
        df["value"] = df["close"] * df["volume"]
    else:
        print("❌ VPS API thất bại. Chuyển sang SSI fallback...")
        with ThreadPoolExecutor(max_workers=3) as pool:
            ssi_results = list(pool.map(_fetch_ssi_exchange, ["hose", "hnx", "upcom"]))
        all_ssi = [item for sublist in ssi_results for item in sublist]
        df = _process_ssi_data(all_ssi) if all_ssi else pd.DataFrame()

    if df.empty:
        print("❌ Cả VPS và SSI API đều thất bại. Bỏ qua cập nhật giá.")
        return False

    # Save data in parallel
    existing_files = {f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')}
    
    def _save_symbol(symbol_group):
        symbol, group = symbol_group
        filepath = os.path.join(PRICE_DIR, f"{symbol}.parquet")
        row = group.iloc[-1].to_dict()
        new_row = {
            "time": row["date"], "open": row["open"], "high": row["high"],
            "low": row["low"], "close": row["close"], "volume": row["volume"],
            "value": row["value"], "foreign_buy_vol": row["foreign_buy_vol"],
            "foreign_sell_vol": row["foreign_sell_vol"],
            "foreign_buy_val": row["foreign_buy_val"],
            "foreign_sell_val": row["foreign_sell_val"], "foreign_room": row["foreign_room"]
        }
        return _update_parquet_safe(filepath, new_row, symbol, existing_files)

    groups = list(df.groupby("symbol"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_save_symbol, groups))
    
    count_updated = sum(1 for r in results if r)
    if count_updated == 0:
        print("⚠️ Giá: 0 mã được cập nhật — có thể đã cập nhật trước đó.")
        return False
    
    print(f"✅ Đã cập nhật giá: {count_updated} mã.")
    return True


# ==============================================================================
# JOB 2: CẬP NHẬT THỎA THUẬN
# ==============================================================================
def job_update_putthrough():
    print("\n--- [2/5] CAP NHAT THOA THUAN ---")
    if is_weekend():
        print("⛔ Hôm nay là cuối tuần. Bỏ qua.")
        return
    
    MASTER_FILE = os.path.join(PUT_DIR, "putthrough_hose_all.csv")
    today_str = get_today_str()
    
    try:
        r = fetch_with_retry("https://bgapidatafeed.vps.com.vn/getlistpt", max_retries=3, timeout=30)
        data = r.json()
        if not data:
            print("⚠️ API Thỏa thuận không trả dữ liệu.")
            return
        
        df = pd.DataFrame(data)
        df = df.rename(columns={"sym": "symbol", "marketID": "floor_code"})
        df = df[df["floor_code"].astype(str) == "10"].copy()
        if df.empty:
            print("⚠️ Không có thỏa thuận HOSE hôm nay.")
            return
        
        df["date"] = today_str
        df["floor"] = "HOSE"
        df["cum_volume"] = df.groupby("symbol")["volume"].cumsum()
        df["cum_value"] = df.groupby("symbol")["value"].cumsum()
        final_df = df[["date", "time", "symbol", "price", "volume", "value", "cum_volume", "cum_value", "floor"]]
        
        if os.path.exists(MASTER_FILE):
            old = pd.read_csv(MASTER_FILE)
            if today_str in old["date"].values:
                print(f"⚠️ Thỏa thuận ngày {today_str} đã tồn tại, bỏ qua.")
                return
            final_df = pd.concat([old, final_df], ignore_index=True)
        
        final_df.to_csv(MASTER_FILE, index=False, encoding="utf-8-sig")
        print(f"✅ Đã lưu thỏa thuận ngày {today_str} vào {MASTER_FILE}")
    except Exception as e:
        print(f"❌ Lỗi cập nhật Thỏa Thuận: {e}")


# ==============================================================================
# JOB 3: CẬP NHẬT TỰ DOANH — FIXED: Always update today if available
# ==============================================================================
def job_update_tudoanh():
    print("\n--- [3/5] CAP NHAT TU DOANH ---")
    if is_weekend():
        print("⛔ Hôm nay là cuối tuần. Bỏ qua.")
        return
    
    MASTER_FILE = os.path.join(TD_DIR, "tudoanh_all.csv")
    base_urls = [
        "https://histdatafeed.vps.com.vn/proprietary/snapshot/TOTAL",
        "https://histdatafeed.vps.com.vn/proprietary/snapshot/total",
        "https://bgapidatafeed.vps.com.vn/proprietary/snapshot/TOTAL",
    ]
    proxy_prefixes = [
        "https://api.codetabs.com/v1/proxy?quest=",
        "https://api.allorigins.win/raw?url="
    ]
    
    urls = base_urls + [f"{p}{b}" for p in proxy_prefixes for b in base_urls]
    today_str = get_today_str()
    
    try:
        data = fetch_json_with_fallback(urls, max_retries=4, connect_timeout=10, read_timeout=45)
        data = data.get("data", []) if isinstance(data, dict) else data
        if not data:
            print("⚠️ API Tự Doanh không trả dữ liệu.")
            return
        
        df = pd.DataFrame(data)
        print(f"  API trả về {len(df)} dòng | columns: {list(df.columns)}")
        
        if "symbol" not in df.columns:
            for c in ["Symbol", "sym", "stockCode", "code"]:
                if c in df.columns:
                    df = df.rename(columns={c: "symbol"})
                    break
        
        if "symbol" not in df.columns:
            print(f"❌ Không tìm thấy cột mã cổ phiếu. Columns: {list(df.columns)}")
            return
        
        def pick_col(candidates):
            return next((c for c in candidates if c in df.columns), None)
        
        c_buy_vol = pick_col(["TBuyVol", "buyVolume", "buy_vol"])
        c_sell_vol = pick_col(["TSellVol", "sellVolume", "sell_vol"])
        c_buy_val = pick_col(["TBuyVal", "buyValue", "buy_val"])
        c_sell_val = pick_col(["TSellVal", "sellValue", "sell_val"])
        
        for col, src in [("buy_volume", c_buy_vol), ("sell_volume", c_sell_vol),
                         ("buy_value", c_buy_val), ("sell_value", c_sell_val)]:
            df[col] = pd.to_numeric(df[src], errors="coerce").fillna(0) if src else 0
        
        df["net_volume"] = df["buy_volume"] - df["sell_volume"]
        df["net_value"] = df["buy_value"] - df["sell_value"]
        
        trading_date_col = pick_col(["TradingDate", "tradingDate", "trading_date"])
        now_vn = get_logical_now()
        
        if trading_date_col:
            try:
                df["date"] = pd.to_datetime(df[trading_date_col], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
            except Exception:
                df["date"] = today_str
        else:
            df["date"] = (now_vn - dt.timedelta(days=1)).strftime("%Y-%m-%d") if now_vn.hour < 16 else today_str
        
        api_date = df["date"].dropna().max()
        print(f"  Ngày từ API: {api_date} | Hôm nay: {today_str}")
        
        final_cols = ["date", "symbol", "buy_volume", "sell_volume", "buy_value", "sell_value", "net_volume", "net_value"]
        df = df[[c for c in final_cols if c in df.columns]]
        
        if os.path.exists(MASTER_FILE):
            old = pd.read_csv(MASTER_FILE)
            old["date"] = pd.to_datetime(old["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            old = old[old["date"].notna()]
            
            # FIXED: Only skip if api_date is STRICTLY older than today AND already exists
            # If api_date == today, always update (overwrites today's data)
            if api_date and api_date < today_str:
                if api_date in old["date"].values:
                    print(f"⚠️ Tự doanh ngày {api_date} (<{today_str}) đã tồn tại, bỏ qua.")
                    return
                else:
                    print(f"   Thêm dữ liệu ngày {api_date} (chưa có trong CSV)")
            
            # Remove existing rows for the same date(s) before concatenating
            old = old[~old["date"].isin(df["date"].unique())]
            df = pd.concat([old, df], ignore_index=True)
            df = df.drop_duplicates(subset=["symbol", "date"], keep="last")
            df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        
        df.to_csv(MASTER_FILE, index=False, encoding="utf-8-sig")
        print(f"✅ Đã lưu tự doanh ngày {api_date} vào {MASTER_FILE} ({len(df):,} dòng | max date = {df['date'].max()})")
    except Exception as e:
        print(f"❌ Lỗi cập nhật Tự Doanh: {e}")


# ==============================================================================
# JOB 4: CẬP NHẬT CHỈ SỐ (VNINDEX) — FIXED: numpy fillna issue
# ==============================================================================
def job_update_index():
    print("\n--- [4/5] CAP NHAT CHI SO (VNINDEX) ---")
    if is_weekend():
        print("⛔ Hôm nay là cuối tuần. Bỏ qua.")
        return
    
    df = None
    today = get_logical_now()
    
    # Try VPS API
    try:
        to_ts = int(today.timestamp())
        from_ts = int((today - dt.timedelta(days=7)).timestamp())
        
        r = requests.get(
            "https://web7.vps.com.vn/trading-view/api/public/history",
            params={"symbol": "VNINDEX", "resolution": "1D", "from": from_ts, "to": to_ts, "countback": 10},
            headers={"accept": "*/*", "origin": "https://web5.vps.com.vn",
                    "referer": "https://web5.vps.com.vn/", "user-agent": HEADERS["User-Agent"]},
            timeout=15
        )
        
        if r.status_code == 200:
            data = r.json()
            if data.get("s") == "ok":
                dates = [dt.datetime.fromtimestamp(ts, tz=VN_TZ).strftime('%Y-%m-%d') for ts in data.get("t", [])]
                
                # FIXED: Convert to pandas Series first to avoid numpy fillna issue
                volume_raw = pd.Series(data.get("v", []), dtype=float)
                volume_raw = volume_raw.fillna(0) * 10
                
                df = pd.DataFrame({
                    "date": dates,
                    "open": pd.Series(data.get("o", []), dtype=float).fillna(0),
                    "high": pd.Series(data.get("h", []), dtype=float).fillna(0),
                    "low": pd.Series(data.get("l", []), dtype=float).fillna(0),
                    "close": pd.Series(data.get("c", []), dtype=float).fillna(0),
                    "volume": volume_raw.astype(int)
                })
                print(f"   ✅ Lấy {len(df)} dòng từ VPS API.")
            else:
                print(f"   ⚠️ VPS API lỗi: {data.get('s')}")
        else:
            print(f"   ⚠️ HTTP {r.status_code} từ VPS API")
    except Exception as e:
        print(f"   ⚠️ Lỗi VPS API: {e}")
    
    # Fallback to vnstock
    if df is None or df.empty:
        print("   🔄 Chuyển sang vnstock fallback...")
        try:
            from vnstock import Quote
            start_date = (today - dt.timedelta(days=7)).strftime("%Y-%m-%d")
            end_date = today.strftime("%Y-%m-%d")
            
            raw_df = Quote(symbol='VNINDEX', source='VCI').history(start=start_date, end=end_date)
            if raw_df is not None and len(raw_df) > 0:
                df = pd.DataFrame(raw_df) if not isinstance(raw_df, pd.DataFrame) else raw_df
                if 'time' not in df.columns and 'dt' not in df.columns and 'date' not in df.columns:
                    df = df.reset_index()
                df = df.rename(columns={'time': 'date', 'dt': 'date'})
                df['date'] = pd.to_datetime(df.get('date'), errors='coerce').dt.strftime('%Y-%m-%d')
                df['volume'] = pd.to_numeric(df.get('volume', 0), errors='coerce').fillna(0)
                for c in ['open', 'high', 'low', 'close']:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors='coerce')
                df = df[[c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]]
                print(f"   ✅ Lấy dữ liệu từ vnstock.")
        except ImportError:
            print("   ❌ vnstock chưa cài đặt. Cài: pip install vnstock")
            return
        except Exception as e:
            print(f"❌ Lỗi vnstock: {e}")
            return
    
    if df is None or df.empty:
        print("❌ Không lấy được dữ liệu VNINDEX.")
        return
    
    try:
        filepath = os.path.join(BASE_DIR, "VNINDEX.csv")
        if os.path.exists(filepath):
            old_df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                d_str = row['date']
                if pd.isna(d_str):
                    continue
                if d_str in old_df['date'].values:
                    old_vol = old_df.loc[old_df['date'] == d_str, 'volume'].iloc[0]
                    if float(row['volume']) != float(old_vol):
                        old_df = old_df[old_df['date'] != d_str]
                        old_df = pd.concat([old_df, pd.DataFrame([row])], ignore_index=True)
                else:
                    old_df = pd.concat([old_df, pd.DataFrame([row])], ignore_index=True)
            old_df = old_df.sort_values(by="date").reset_index(drop=True)
            old_df.to_csv(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)
        print("✅ Đã hoàn tất cập nhật VNINDEX.")
    except Exception as e:
        print(f"❌ Lỗi lưu VNINDEX.csv: {e}")


# ==============================================================================
# JOB 5: TICK DATA — FIXED: Minimal console output
# ==============================================================================
_TICK_WORKERS = 5
_TICK_PROFILES = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0"},
    {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0"},
]
_tick_local = threading.local()
_tick_lock = threading.Lock()
_tick_count_lock = threading.Lock()
_tick_counter = {"ok": 0, "skip": 0, "fail": 0}


def _tick_log(msg, force=False):
    """Only print force messages or final summary to reduce spam."""
    if force:
        with _tick_lock:
            print(msg, flush=True)


def _tick_session(profile):
    if not hasattr(_tick_local, "session"):
        s = requests.Session()
        s.headers.update({
            "Accept": "application/json", "Origin": "https://finpath.vn",
            "Referer": "https://finpath.vn/", "Client-Type": "web",
            **{k: v for k, v in profile.items() if v},
        })
        _tick_local.session = s
    return _tick_local.session


def _fetch_symbol_trades(symbol, profile):
    session = _tick_session(profile)
    page, all_trades = 1, []
    while True:
        session.headers.update({"timestamp": str(int(time.time() * 1000)), "uuid": str(uuid.uuid4())})
        url = f"https://api.finpath.vn/api/stocks/v2/trades/{symbol}?page={page}&pageSize=1000"
        try:
            r = session.get(url, timeout=10)
            if r.status_code == 429:
                time.sleep(10)
                continue
            if r.status_code != 200:
                break
            trades = r.json().get("data", {}).get("trades", [])
            if not trades:
                break
            all_trades.extend(trades)
            page += 1
            time.sleep(0.5)
        except Exception:
            break
    return pd.DataFrame(all_trades)


def _fetch_and_save_tick(args):
    symbol, out_path, sector, platform, profile_idx, total, position = args
    
    if os.path.exists(out_path):
        mtime = dt.datetime.fromtimestamp(os.path.getmtime(out_path), tz=VN_TZ)
        now = dt.datetime.now(VN_TZ)
        market_close = now.replace(hour=15, minute=15, second=0, microsecond=0)
        
        if now >= market_close and mtime >= market_close and mtime.date() == now.date():
            with _tick_count_lock:
                _tick_counter["skip"] += 1
            return symbol, "skip"
        
        mins_since_update = (now - mtime).total_seconds() / 60.0
        if mins_since_update < 15:
            with _tick_count_lock:
                _tick_counter["skip"] += 1
            return symbol, "skip"
    
    new_df = _fetch_symbol_trades(symbol, _TICK_PROFILES[profile_idx % len(_TICK_PROFILES)])
    if new_df.empty:
        with _tick_count_lock:
            _tick_counter["fail"] += 1
        return symbol, "failed"
    
    new_df.insert(0, "symbol", symbol)
    new_df.insert(1, "sector", sector)
    new_df.insert(2, "platform", platform)
    
    try:
        if os.path.exists(out_path):
            old_df = pd.read_parquet(out_path)
            combined_df = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates()
            if 'time' in combined_df.columns:
                combined_df = combined_df.sort_values(by=['time']).reset_index(drop=True)
            combined_df.to_parquet(out_path, index=False, engine="pyarrow")
        else:
            new_df.to_parquet(out_path, index=False, engine="pyarrow")
        
        with _tick_count_lock:
            _tick_counter["ok"] += 1
        return symbol, "ok"
    except Exception:
        with _tick_count_lock:
            _tick_counter["fail"] += 1
        return symbol, "failed"


def _print_tick_progress():
    """Print tick progress every 30 seconds."""
    last_print = [0]
    def _maybe_print():
        now = time.time()
        if now - last_print[0] >= 30:
            with _tick_count_lock:
                total = _tick_counter["ok"] + _tick_counter["skip"] + _tick_counter["fail"]
            _tick_log(f"  Tick progress: {_tick_counter['ok']} ok | {_tick_counter['skip']} skip | {_tick_counter['fail']} fail | {total} processed", force=True)
            last_print[0] = now
    return _maybe_print


def job_update_tick_data():
    print("\n--- [5/5] CAP NHAT TICK DATA ---")
    if is_weekend():
        print("Hôm nay là cuối tuần. Bỏ qua.")
        return
    
    sector_csv = os.path.join(BASE_DIR, "data", "sector_master.csv")
    if not os.path.exists(sector_csv):
        print(f"Không tìm thấy sector_master.csv tại {sector_csv}")
        return
    
    sectors = pd.read_csv(sector_csv, encoding="utf-8-sig")
    symbols = sectors["symbol"].dropna().unique().tolist()
    sector_lookup = sectors.set_index("symbol")["sector"].to_dict()
    platform_lookup = sectors.set_index("symbol")["platform"].to_dict()
    
    master_dir = os.path.join(DATA_DIR, "tick_data")
    os.makedirs(master_dir, exist_ok=True)
    
    # Reset counters
    with _tick_count_lock:
        _tick_counter["ok"] = 0
        _tick_counter["skip"] = 0
        _tick_counter["fail"] = 0
    
    print(f"{len(symbols)} symbols to update. Running {_TICK_WORKERS} parallel sessions...")
    
    tasks = [
        (sym, os.path.join(master_dir, f"{sym}.parquet"),
         sector_lookup.get(sym, ""), platform_lookup.get(sym, ""),
         i % _TICK_WORKERS, len(symbols), i + 1)
        for i, sym in enumerate(symbols)
    ]
    
    failed_syms = []
    print_progress = _print_tick_progress()
    
    with ThreadPoolExecutor(max_workers=_TICK_WORKERS) as pool:
        futures = {pool.submit(_fetch_and_save_tick, t): t[0] for t in tasks}
        for future in as_completed(futures):
            sym, status = future.result()
            if status == "failed":
                failed_syms.append(sym)
            print_progress()
    
    # Retry failed symbols (silent)
    if failed_syms:
        retry_wait = 15
        for attempt in range(1, 4):
            if not failed_syms:
                break
            time.sleep(retry_wait)
            retry_wait *= 2
            still_failed = []
            for i, sym in enumerate(failed_syms):
                out_path = os.path.join(master_dir, f"{sym}.parquet")
                task = (sym, out_path, sector_lookup.get(sym, ""), platform_lookup.get(sym, ""), i, len(failed_syms), i)
                _, status = _fetch_and_save_tick(task)
                if status == "ok":
                    with _tick_count_lock:
                        _tick_counter["ok"] += 1
                else:
                    still_failed.append(sym)
                time.sleep(2)
            failed_syms = still_failed
    
    # Final summary (always printed)
    total = _tick_counter["ok"] + _tick_counter["skip"] + _tick_counter["fail"]
    print(f"✅ Tick data: {_tick_counter['ok']} updated | {_tick_counter['skip']} skipped | {_tick_counter['fail']} failed | {total} total")
    if failed_syms:
        print(f"   Failed: {', '.join(failed_syms[:10])}{'...' if len(failed_syms) > 10 else ''}")


def job_update_investor_flow():
    print("\n--- [7/7] CAP NHAT NDT FLOW ---")
    if is_weekend():
        print("⛔ Cuối tuần. Bỏ qua.")
        return False
    
    flow_dir = os.path.join(BASE_DIR, "data", "investor_flow")
    script = os.path.join(BASE_DIR, "download", "fetch_investor_flow.py")
    
    if not os.path.exists(script):
        print(f"❌ fetch_investor_flow.py not found at {script}")
        return False
    
    files = [f for f in os.listdir(flow_dir) if f.endswith(".parquet")] if os.path.isdir(flow_dir) else []
    if not files:
        print(f"⚠️  No parquet files found. Run first-time backfill: python fetch_investor_flow.py --all")
        return False
    
    # Calculate days behind
    try:
        max_date = None
        for f in files[:5]:
            df = pd.read_parquet(os.path.join(flow_dir, f))
            date_col = "date" if "date" in df.columns else df.columns[0]
            latest = pd.to_datetime(df[date_col]).max().date()
            if max_date is None or latest > max_date:
                max_date = latest
        
        today_date = get_logical_now().date()
        
        if max_date:
            days_gap = (today_date - max_date).days
            print(f"   📅 Dữ liệu NDT mới nhất: {max_date} | Hôm nay: {today_date} | Gap: {days_gap} ngày")
            
            # NDT flow is always 1 day behind (T+1)
            # If gap is 1 day, that's normal - skip download
            if days_gap <= 1:
                print(f"⚠️ NDT flow đã cập nhật đến {max_date} (lag 1 ngày là bình thường). Bỏ qua tải thêm.")
                return False
            
            # Only download if more than 1 day behind (weekend/holiday catch-up)
            days_to_fetch = min(days_gap, 14)
            print(f"   🔄 Dữ liệu trễ {days_gap} ngày. Đang tải thêm {days_to_fetch} ngày...")
        else:
            print("   ⚠️ Không xác định được ngày mới nhất. Dùng mặc định 3 ngày.")
            days_to_fetch = 3
    except Exception as e:
        print(f"   ⚠️ Không tính được ngày gần nhất ({e}). Dùng mặc định 3 ngày.")
        days_to_fetch = 3
        max_date = None
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, script, "--update", "--days", str(days_to_fetch), "--workers", "4"],
            capture_output=True, text=True, encoding="utf-8", timeout=300, cwd=BASE_DIR
        )
        if result.returncode != 0:
            print(f"❌ fetch_investor_flow exit {result.returncode}: {(result.stderr or '').strip()}")
            return False
        
        # Verify what was downloaded
        if max_date:
            # Check new max date after download
            try:
                df = pd.read_parquet(os.path.join(flow_dir, files[0]))
                date_col = "date" if "date" in df.columns else df.columns[0]
                new_max = pd.to_datetime(df[date_col]).max().date()
                print(f"✅ NDT flow đã cập nhật: {max_date} → {new_max} (thêm {(new_max - max_date).days} ngày).")
            except Exception:
                print(f"✅ NDT flow đã cập nhật {days_to_fetch} ngày.")
        else:
            print(f"✅ NDT flow đã cập nhật {days_to_fetch} ngày.")
    except subprocess.TimeoutExpired:
        print("❌ fetch_investor_flow timeout (>5 min).")
        return False
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False

# ==============================================================================
# JOB 8: FOREIGN FLOW LONG — FIXED: Print date downloaded
# ==============================================================================
def job_update_foreign_flow_long():
    print("\n--- [8/8] CAP NHAT FOREIGN FLOW LONG ---")
    if is_weekend():
        print("⛔ Cuối tuần. Bỏ qua.")
        return False
    
    out_dir = os.path.join(DATA_DIR, "foreign_flow_long")
    script = os.path.join(BASE_DIR, "download", "fetch_foreign_flow_long.py")
    
    if not os.path.exists(script):
        print(f"❌ fetch_foreign_flow_long.py not found at {script}")
        return False
    
    if not os.path.isdir(out_dir) or not os.listdir(out_dir):
        print(f"⚠️  {out_dir} trống. Chạy backfill: python download/fetch_foreign_flow_long.py")
        return False
    
    # Check current max date before update
    try:
        files = [f for f in os.listdir(out_dir) if f.endswith(".parquet")]
        max_date_before = None
        if files:
            df = pd.read_parquet(os.path.join(out_dir, files[0]))
            max_date_before = pd.to_datetime(df["date"]).max().date() if "date" in df.columns else None
    except Exception:
        max_date_before = None
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, script, "--update", "--lookback-days", "10", "--workers", "6"],
            capture_output=True, text=True, encoding="utf-8", timeout=600, cwd=BASE_DIR
        )
        for line in (result.stdout or "").splitlines():
            if "ERROR" in line or "Done." in line:
                print(f"   {line}")
        
        if result.returncode != 0:
            print(f"❌ fetch_foreign_flow_long exit {result.returncode}")
            return False
        
        # Check max date after update
        try:
            df = pd.read_parquet(os.path.join(out_dir, files[0]))
            max_date_after = pd.to_datetime(df["date"]).max().date() if "date" in df.columns and files else None
            if max_date_before and max_date_after:
                if max_date_after > max_date_before:
                    print(f"✅ Foreign flow long đã cập nhật (ngày mới nhất: {max_date_after}, trước đó: {max_date_before}).")
                else:
                    print(f"✅ Foreign flow long đã chạy (không có ngày mới, mới nhất: {max_date_after}).")
            elif max_date_after:
                print(f"✅ Foreign flow long đã cập nhật (ngày mới nhất: {max_date_after}).")
            else:
                print("✅ Foreign flow long đã chạy.")
        except Exception:
            print("✅ Foreign flow long đã chạy.")
    except subprocess.TimeoutExpired:
        print("❌ fetch_foreign_flow_long timeout (>10 min).")
        return False
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


def job_update_merged_dashboard_parquet():
    """Rebuild data/all_stocks_with_industries.parquet from data/price/*.parquet
    + ticker_sectors.csv — the single-file source the webapp dashboard reads
    (webapp/api/main.py). Was a one-off manual build that silently went stale
    (last date drifted to April while data/price/ itself stayed current) —
    rebuilding it here each run keeps it in lockstep with the price pipeline."""
    print("\n--- CAP NHAT all_stocks_with_industries.parquet (dashboard) ---")
    sectors_path = os.path.join(BASE_DIR, "ticker_sectors.csv")
    out_path = os.path.join(DATA_DIR, "all_stocks_with_industries.parquet")
    if not os.path.exists(sectors_path):
        print(f"❌ {sectors_path} not found")
        return False
    try:
        sec = pd.read_csv(sectors_path)
        sec.columns = [c.strip().lower() for c in sec.columns]
        sec["ticker"] = sec["ticker"].str.upper()
        sec_map = sec.set_index("ticker")[["industry", "exchange"]].to_dict("index")

        frames = []
        for fp in glob.glob(os.path.join(PRICE_DIR, "*.parquet")):
            ticker = os.path.splitext(os.path.basename(fp))[0].upper()
            info = sec_map.get(ticker)
            if info is None:
                continue
            df = pd.read_parquet(fp, columns=["time", "open", "high", "low", "close", "volume"])
            df = df.rename(columns={"time": "date"})
            df["date"] = pd.to_datetime(df["date"])
            df["ticker"] = ticker
            df["industry"] = info["industry"]
            df["exchange"] = info["exchange"]
            frames.append(df)

        if not frames:
            print("❌ No price files matched ticker_sectors.csv")
            return False

        out = pd.concat(frames, ignore_index=True)
        out = out[["ticker", "date", "open", "high", "low", "close", "volume", "industry", "exchange"]]
        out = out.sort_values(["ticker", "date"])
        out.to_parquet(out_path, index=False)
        print(f"✅ Rebuilt ({len(out):,} rows, {out['ticker'].nunique()} tickers, "
              f"latest date {out['date'].max().date()})")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


# ==============================================================================
# DATE VERIFICATION
# ==============================================================================
def _latest_dates_summary():
    """Return dict of label -> latest date string for each data source."""
    out = {}
    
    # Prices
    try:
        files = [f for f in os.listdir(PRICE_DIR) if f.endswith(".parquet")]
        if files:
            target = os.path.join(PRICE_DIR, "ACB.parquet")
            df = pd.read_parquet(target if "ACB.parquet" in files else os.path.join(PRICE_DIR, files[0]))
            out["Giá"] = str(df["time"].max()) if "time" in df.columns else "?"
    except Exception:
        out["Giá"] = "ERR"
    
    # Putthrough
    try:
        pt = os.path.join(PUT_DIR, "putthrough_hose_all.csv")
        out["Thỏa thuận"] = str(pd.read_csv(pt)["date"].max()) if os.path.exists(pt) else "ERR"
    except Exception:
        out["Thỏa thuận"] = "ERR"
    
    # Tu doanh
    try:
        td = os.path.join(TD_DIR, "tudoanh_all.csv")
        out["Tự doanh"] = str(pd.read_csv(td)["date"].max()) if os.path.exists(td) else "ERR"
    except Exception:
        out["Tự doanh"] = "ERR"
    
    # VNINDEX
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "VNINDEX.csv"))
        date_col = "time" if "time" in df.columns else df.columns[0]
        out["VNINDEX"] = str(df[date_col].max())
    except Exception:
        out["VNINDEX"] = "ERR"
    
    # Tick data
    try:
        tick_dir = os.path.join(DATA_DIR, "tick_data")
        if os.path.isdir(tick_dir):
            tick_files = [f for f in os.listdir(tick_dir) if f.endswith(".parquet")]
            today_str = dt.datetime.now(VN_TZ).strftime("%Y-%m-%d")
            updated = sum(1 for f in tick_files if dt.datetime.fromtimestamp(
                os.path.getmtime(os.path.join(tick_dir, f)), tz=VN_TZ).strftime("%Y-%m-%d") == today_str)
            out["Tick data"] = f"{updated}/{len(tick_files)} mã cập nhật hôm nay"
        else:
            out["Tick data"] = "chưa có dữ liệu"
    except Exception:
        out["Tick data"] = "ERR"
    
    # Investor flow
    try:
        flow_dir = os.path.join(BASE_DIR, "data", "investor_flow")
        files = [f for f in os.listdir(flow_dir) if f.endswith(".parquet")] if os.path.isdir(flow_dir) else []
        if files:
            df = pd.read_parquet(os.path.join(flow_dir, files[0]))
            date_col = "date" if "date" in df.columns else df.columns[0]
            out["NDT Flow"] = str(pd.to_datetime(df[date_col]).max().date())
    except Exception:
        out["NDT Flow"] = "ERR"
    
    # Foreign flow long
    try:
        ffl_dir = os.path.join(BASE_DIR, "data", "foreign_flow_long")
        files = [f for f in os.listdir(ffl_dir) if f.endswith(".parquet")] if os.path.isdir(ffl_dir) else []
        if files:
            df = pd.read_parquet(os.path.join(ffl_dir, files[0]))
            out["Foreign Flow Long"] = str(pd.to_datetime(df["date"]).max().date())
    except Exception:
        out["Foreign Flow Long"] = "ERR"
    
    # Dashboard merged parquet
    try:
        p = os.path.join(DATA_DIR, "all_stocks_with_industries.parquet")
        if os.path.exists(p):
            df = pd.read_parquet(p, columns=["date"])
            out["Dashboard Parquet"] = str(df["date"].max().date())
    except Exception:
        out["Dashboard Parquet"] = "ERR"

    return out


# ==============================================================================
# MAIN EXECUTION — PARALLEL JOB EXECUTION
# ==============================================================================
def _run_job(job_label, job_func):
    """Run a single job and return (label, status, detail)."""
    try:
        result = job_func()
        if result is False:
            return (job_label, "FAILED", "lỗi nội bộ (xem log)")
        return (job_label, "OK", "")
    except Exception as e:
        err = str(e)[:120]
        print(f"ERROR {job_label}: {err}")
        return (job_label, "FAILED", err)


if __name__ == "__main__":
    # Jobs that can run in parallel (independent of each other)
    independent_jobs = [
        ("Giá & Nước ngoài", job_update_prices),
        ("Thỏa thuận", job_update_putthrough),
        ("Tự doanh", job_update_tudoanh),
        ("VNINDEX", job_update_index),
        ("Tick data", job_update_tick_data),
    ]
    
    # Jobs that depend on price data being done first
    dependent_jobs = [
        ("NDT Flow", job_update_investor_flow),
        ("Foreign Flow Long", job_update_foreign_flow_long),
        ("Dashboard Parquet", job_update_merged_dashboard_parquet),
    ]
    
    print(f"BẮT ĐẦU CHẠY UPDATE NGÀY: {get_today_str()}")
    print(f"Thư mục lưu dữ liệu: {DATA_DIR}")
    print(f"Chạy {len(independent_jobs)} jobs song song...")
    
    # Phase 1: Run independent jobs in parallel
    job_results = []
    with ThreadPoolExecutor(max_workers=len(independent_jobs)) as pool:
        futures = {pool.submit(_run_job, label, func): label for label, func in independent_jobs}
        for future in as_completed(futures):
            job_results.append(future.result())
    
    # Phase 2: Run dependent jobs sequentially after price data is ready
    for label, func in dependent_jobs:
        job_results.append(_run_job(label, func))
    
    print("\nHOÀN TẤT TOÀN BỘ QUÁ TRÌNH UPDATE!")
    
    # Build Telegram summary
    failed_jobs = [j for j in job_results if j[1] == "FAILED"]
    done_time = dt.datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")
    dates = _latest_dates_summary()
    
    job_lines = []
    for label, status, detail in job_results:
        icon = "✅" if status == "OK" else "❌"
        suffix = f" — {detail}" if status == "FAILED" and detail else ""
        job_lines.append(f"  {icon} {label}{suffix}")
    
    date_lines = [f"  📅 {label}: {val}" for label, val in dates.items()]
    
    header = "⚠️ DAILY UPDATE (CÓ LỖI)" if failed_jobs else "✅ DAILY UPDATE HOÀN TẤT"
    fail_summary = f"  {len(failed_jobs)} job lỗi / {len(job_results)} tổng\n" if failed_jobs else ""
    
    telegram_msg = (
        f"{header}\n"
        f"🕐 {done_time} (VN)\n"
        f"{fail_summary}\n"
        f"<b>Trạng thái job:</b>\n" + "\n".join(job_lines) +
        f"\n\n<b>Dữ liệu mới nhất:</b>\n" + "\n".join(date_lines)
    )
    
    send_telegram_message(telegram_msg)