import os
import sys
import requests
import pandas as pd
import datetime as dt
import json
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import RequestException, Timeout

# Ensure Windows terminal can print Unicode logs (emoji/Vietnamese) safely
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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    "Origin": "https://banggia.vps.com.vn",
    "Referer": "https://banggia.vps.com.vn/",
    "Accept": "application/json, text/plain, */*",
    "Connection": "keep-alive"
}

# ==============================================================================
# TELEGRAM NOTIFICATION CONFIG (FILL THESE LATER)
# ==============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


def send_telegram_message(message):
    """Send a Telegram message when bot token/chat id are configured."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  Telegram chưa cấu hình (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID). Bỏ qua gửi thông báo.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": int(TELEGRAM_CHAT_ID),  # Convert to integer
        "text": message,
        "parse_mode": "HTML"  # Optional: enable HTML formatting
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        print("✅ Đã gửi thông báo Telegram.")
    except Exception as e:
        print(f"❌ Gửi Telegram thất bại: {e}")

def fetch_with_retry(url, max_retries=3, timeout=30):
    """Fetch URL with retry logic and exponential backoff."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                print(f"   ⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    return None


def fetch_json_with_fallback(urls, max_retries=3, connect_timeout=10, read_timeout=45):
    """
    Fetch JSON from a list of fallback URLs with retry/backoff.
    Tries all URLs in each retry round before backing off.
    """
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
            print(f"   ↻ Retrying all Tu Doanh endpoints in {wait_time}s...")
            time.sleep(wait_time)

    raise RuntimeError(f"Khong the lay du lieu Tu Doanh sau {max_retries} lan thu. Loi cuoi: {last_error}")

def get_today_str():
    return dt.datetime.now(VN_TZ).strftime("%Y-%m-%d")

def is_weekend():
    weekday = dt.datetime.now(VN_TZ).weekday()
    return weekday >= 5

print(f"BAT DAU CHAY UPDATE NGAY: {get_today_str()}")
print(f"Folder luu du lieu: {DATA_DIR}")

# ==============================================================================
# PHẦN 1: CẬP NHẬT GIÁ & NƯỚC NGOÀI (SNAPSHOT)
# ==============================================================================
# ==============================================================================
# 🛠️ HELPER: PARSE VPS "G" STRINGS
# (Add this above job_update_prices)
# ==============================================================================
def parse_g_string(g_str):
    """
    Parses VPS order book string (e.g., '18.6|13600|i')
    Returns: (Price, Volume)
    """
    if not g_str or g_str == "0|0|e":
        return 0.0, 0.0
    try:
        parts = g_str.split('|')
        # parts[0] is Price, parts[1] is Volume
        return float(parts[0]), float(parts[1])
    except:
        return 0.0, 0.0

# ==============================================================================
# PHẦN 1: CẬP NHẬT GIÁ & NƯỚC NGOÀI (SNAPSHOT)
# ==============================================================================
def job_update_prices():
    print("\n--- [1/5] CAP NHAT GIA & NUOC NGOAI ---")
    if is_weekend():
        print("⛔ Hôm nay là cuối tuần. Bỏ qua.")
        return

    # Helper function to fetch from SSI as a fallback
    def fetch_ssi_fallback():
        print("   🔄 Đang chuyển sang API dự phòng (SSI iBoard)...")
        ssi_headers = {
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/149.0.0.0 Safari/537.36",
            "Referer": "https://iboard.ssi.com.vn/"
        }
        all_ssi_data = []
        for exchange in ["hose", "hnx", "upcom"]:
            url = f"https://iboard-query.ssi.com.vn/stock/exchange/{exchange}"
            try:
                r = requests.get(url, headers=ssi_headers, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    stock_list = data.get("data", data) if isinstance(data, dict) else data
                    all_ssi_data.extend(stock_list)
                else:
                    print(f"   ⚠️ Lỗi SSI API cho {exchange}: {r.status_code}")
            except Exception as e:
                print(f"   ⚠️ Lỗi kết nối SSI {exchange}: {e}")
        
        if not all_ssi_data:
            return pd.DataFrame()

        # Map SSI data to our standard DataFrame format
        df_ssi = pd.DataFrame(all_ssi_data)
        df = pd.DataFrame()
        df["symbol"] = df_ssi["stockSymbol"]
        df["date"] = get_today_str()
        
        # SSI prices are absolute (7430), VPS are thousands (7.43). Divide by 1000.
        for col, ssi_col in [("open", "openPrice"), ("high", "highest"), ("low", "lowest"), ("close", "matchedPrice")]:
            df[col] = pd.to_numeric(df_ssi.get(ssi_col, 0), errors="coerce").fillna(0) / 1000.0
            
        # SSI gives exact share count, VPS uses lots (x10). We just take exact share count.
        df["volume"] = pd.to_numeric(df_ssi.get("nmTotalTradedQty", 0), errors="coerce").fillna(0)
        df["value"] = df["close"] * df["volume"]
        
        # Foreign flow mapping
        df["foreign_buy_vol"] = pd.to_numeric(df_ssi.get("buyForeignQtty", 0), errors="coerce").fillna(0)
        df["foreign_sell_vol"] = pd.to_numeric(df_ssi.get("sellForeignQtty", 0), errors="coerce").fillna(0)
        df["foreign_buy_val"] = pd.to_numeric(df_ssi.get("buyForeignValue", 0), errors="coerce").fillna(0) / 1000.0
        df["foreign_sell_val"] = pd.to_numeric(df_ssi.get("sellForeignValue", 0), errors="coerce").fillna(0) / 1000.0
        df["foreign_room"] = pd.to_numeric(df_ssi.get("remainForeignQtty", 0), errors="coerce").fillna(0)
        
        return df

    # --- 1. TRY VPS API FIRST ---
    def get_vps_symbols(exchange):
        url = f"https://bgapidatafeed.vps.com.vn/getlistckindex/{exchange}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            data = json.loads(r.text)
            return [s for s in data if isinstance(s, str)]
        except: return []

    symbols = []
    for exc in ["hose", "hnx", "upcom"]:
        symbols.extend(get_vps_symbols(exc))
    symbols = list(set(symbols))
    
    all_vps_data = []
    chunk_size = 100 
    print(f"⏳ Đang tải dữ liệu cho {len(symbols)} mã (VPS API)...")
    
    vps_failed = False
    if symbols:
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i+chunk_size]
            url = f"https://bgapidatafeed.vps.com.vn/getliststockdata/{','.join(chunk)}"
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                if r.status_code != 200:
                    vps_failed = True
                    break
                try: data = r.json()
                except: data = json.loads(r.text)
                all_vps_data.extend(data)
            except:
                vps_failed = True
                break
    else:
        vps_failed = True

    # --- 2. PROCESS VPS OR FALLBACK TO SSI ---
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
        numeric_cols = [
            "open", "high", "low", "close", 
            "foreign_buy_vol", "foreign_sell_vol", 
            "foreign_buy_val", "foreign_sell_val", "foreign_room"
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
        df["value"] = df["close"] * df["volume"]
    else:
        print("❌ VPS API thất bại hoặc không phản hồi.")
        df = fetch_ssi_fallback()

    if df.empty:
        print("❌ Cả VPS và SSI API đều thất bại. Bỏ qua cập nhật giá.")
        return False

    # --- 3. SAVE DATA TO PARQUET ---
    count_updated = 0
    existing_files = {f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')}

    for symbol, group in df.groupby("symbol"):
        filepath = os.path.join(PRICE_DIR, f"{symbol}.parquet")
        row = group.iloc[-1].to_dict() 
        
        new_row = {
            "time": row["date"],
            "open": row["open"], "high": row["high"], "low": row["low"],
            "close": row["close"], "volume": row["volume"], "value": row["value"],
            "foreign_buy_vol": row["foreign_buy_vol"],
            "foreign_sell_vol": row["foreign_sell_vol"],
            "foreign_buy_val": row["foreign_buy_val"],
            "foreign_sell_val": row["foreign_sell_val"],
            "foreign_room": row["foreign_room"]
        }
        new_row_df = pd.DataFrame([new_row])

        try:
            if symbol in existing_files:
                old_df = pd.read_parquet(filepath)
                if not old_df.empty:
                    last_row = old_df.iloc[-1]
                    if str(last_row.get("time", "")) == str(row["date"]) and \
                       (float(row["volume"]) == float(last_row.get("volume", 0))) and \
                       (float(row["close"]) == float(last_row.get("close", 0))):
                        continue

                    if row["date"] in old_df["time"].values:
                        old_df = old_df[old_df["time"] != row["date"]]
                
                updated_df = pd.concat([old_df, new_row_df], ignore_index=True)
            else:
                updated_df = new_row_df

            updated_df.to_parquet(filepath, index=False, engine="pyarrow")
            count_updated += 1
            
        except Exception as e:
            continue

    if count_updated == 0:
        print("⚠️ Giá: 0 mã được cập nhật — có thể đã cập nhật trước đó.")
        return False
        
    print(f"✅ Đã cập nhật giá: {count_updated} mã.")
    return True
# ==============================================================================
# PHẦN 2: CẬP NHẬT THỎA THUẬN
# ==============================================================================
def job_update_putthrough():
    print("\n--- [2/5] CAP NHAT THOA THUAN ---")
    if is_weekend(): return

    MASTER_FILE = os.path.join(PUT_DIR, "putthrough_hose_all.csv")
    url = "https://bgapidatafeed.vps.com.vn/getlistpt"
    
    try:
        r = fetch_with_retry(url, max_retries=3, timeout=30)
        data = r.json()
        if not data: return

        df = pd.DataFrame(data)
        df = df.rename(columns={"sym": "symbol", "marketID": "floor_code"})
        df = df[df["floor_code"].astype(str) == "10"].copy()
        if df.empty: return

        df["date"] = get_today_str()
        df["floor"] = "HOSE"
        df["cum_volume"] = df.groupby("symbol")["volume"].cumsum()
        df["cum_value"] = df.groupby("symbol")["value"].cumsum()
        
        final_df = df[["date", "time", "symbol", "price", "volume", "value", "cum_volume", "cum_value", "floor"]]

        if os.path.exists(MASTER_FILE):
            old = pd.read_csv(MASTER_FILE)
            if get_today_str() in old["date"].values:
                print(f"⚠️ Thoa thuan ngay {get_today_str()} da ton tai, bo qua.")
                return
            final_df = pd.concat([old, final_df], ignore_index=True)
        
        final_df.to_csv(MASTER_FILE, index=False, encoding="utf-8-sig")
        print(f"✅ Da luu thoa thuan vao {MASTER_FILE}")
    except Exception as e:
        print(f"❌ Loi cap nhat Thoa Thuan: {e}")

# ==============================================================================
# PHẦN 3: CẬP NHẬT TỰ DOANH
# ==============================================================================
def job_update_tudoanh():
    print("\n--- [3/5] CAP NHAT TU DOANH ---")
    if is_weekend(): return

    MASTER_FILE = os.path.join(TD_DIR, "tudoanh_all.csv")
    
    # 1. Danh sách URL trực tiếp (Sẽ chạy rất nhanh nếu chạy ở local)
    base_urls = [
        "https://histdatafeed.vps.com.vn/proprietary/snapshot/TOTAL",
        "https://histdatafeed.vps.com.vn/proprietary/snapshot/total",
        "https://bgapidatafeed.vps.com.vn/proprietary/snapshot/TOTAL",
    ]
    
    # 2. Danh sách CORS Proxy miễn phí để bypass chặn IP trên GitHub Actions
    proxy_prefixes = [
        "https://api.codetabs.com/v1/proxy?quest=",
        "https://api.allorigins.win/raw?url="
    ]
    
    # 3. Tạo danh sách URL tổng hợp: Đưa URL trực tiếp lên đầu, sau đó mới đến Proxy
    urls = base_urls.copy()
    for prefix in proxy_prefixes:
        for base in base_urls:
            urls.append(f"{prefix}{base}")
    
    try:
        # fetch_json_with_fallback sẽ thử lần lượt. 
        # Nếu GitHub bị timeout ở 3 link đầu, nó sẽ tự động thử link có proxy.
        data = fetch_json_with_fallback(urls, max_retries=4, connect_timeout=10, read_timeout=45)
        data = data.get("data", []) if isinstance(data, dict) else data
        if not data:
            print("⚠️ API Tự Doanh không trả dữ liệu.")
            return

        df = pd.DataFrame(data)
        print(f"  API tra ve {len(df)} dong | columns: {list(df.columns)}")

        if "symbol" not in df.columns:
            for c in ["Symbol", "sym", "stockCode", "code"]:
                if c in df.columns:
                    df = df.rename(columns={c: "symbol"})
                    break
        if "symbol" not in df.columns:
            print(f"❌ Không tìm thấy cột mã cổ phiếu trong dữ liệu Tự Doanh. Columns: {list(df.columns)}")
            return

        def pick_col(candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        c_buy_vol = pick_col(["TBuyVol", "buyVolume", "buy_vol"])
        c_sell_vol = pick_col(["TSellVol", "sellVolume", "sell_vol"])
        c_buy_val = pick_col(["TBuyVal", "buyValue", "buy_val"])
        c_sell_val = pick_col(["TSellVal", "sellValue", "sell_val"])

        df["buy_volume"] = pd.to_numeric(df[c_buy_vol], errors="coerce").fillna(0) if c_buy_vol else 0
        df["sell_volume"] = pd.to_numeric(df[c_sell_vol], errors="coerce").fillna(0) if c_sell_vol else 0
        df["buy_value"] = pd.to_numeric(df[c_buy_val], errors="coerce").fillna(0) if c_buy_val else 0
        df["sell_value"] = pd.to_numeric(df[c_sell_val], errors="coerce").fillna(0) if c_sell_val else 0
        df["net_volume"] = df["buy_volume"] - df["sell_volume"]
        df["net_value"] = df["buy_value"] - df["sell_value"]

        # Use TradingDate from API if available (DD/MM/YYYY HH:MM:SS), else fall back
        trading_date_col = pick_col(["TradingDate", "tradingDate", "trading_date"])
        now_vn = dt.datetime.now(VN_TZ)
        today_vn = now_vn.strftime("%Y-%m-%d")
        if trading_date_col:
            sample_raw = str(df[trading_date_col].iloc[0]).strip()
            print(f"  TradingDate column='{trading_date_col}' | sample raw value='{sample_raw}'")
            try:
                df["date"] = pd.to_datetime(df[trading_date_col], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
            except Exception:
                df["date"] = today_vn
        else:
            # No TradingDate: if before 16:00 VN, assume it's still T-1 data; else use today
            if now_vn.hour < 16:
                fallback = (now_vn - dt.timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                fallback = today_vn
            print(f"  Khong co cot TradingDate -> dung ngay: {fallback}")
            df["date"] = fallback

        api_date = df["date"].dropna().max() if "date" in df.columns and len(df) > 0 else None
        print(f"  Ngay tu API: {api_date} | Ngay hom nay (VN): {today_vn}"
              + (" ⚠️ LAG 1 NGAY" if api_date and api_date < today_vn else " DONG BOT"))
        if api_date and api_date < today_vn:
            print(f"  VPS API chua cap nhat du lieu hom nay. Du lieu la cua {api_date}.")

        final_cols = ["date", "symbol", "buy_volume", "sell_volume", "buy_value", "sell_value", "net_volume", "net_value"]
        df = df[[c for c in final_cols if c in df.columns]]

        if os.path.exists(MASTER_FILE):
            old = pd.read_csv(MASTER_FILE)
            # Existing CSV stores ISO YYYY-MM-DD — no dayfirst needed
            old["date"] = pd.to_datetime(old["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            old = old[old["date"].notna()]
            old_max = old["date"].max()
            print(f"  CSV hien tai: {len(old):,} dong | max date = {old_max}")
            # Skip only if api_date is older than today AND already in CSV.
            # If api_date == today_vn, always write — refreshes today's snapshot
            # and overwrites any row that may have been stamped with a wrong date.
            if api_date and api_date < today_vn and api_date in old.get("date", pd.Series()).values:
                print(f"⚠️ Tu doanh ngay {api_date} (<today) da ton tai, bo qua.")
                return
            df = pd.concat([old, df], ignore_index=True)
            df = df.drop_duplicates(subset=["symbol", "date"], keep="last")
            df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

        df.to_csv(MASTER_FILE, index=False, encoding="utf-8-sig")
        new_max = df["date"].max()
        print(f"✅ Da luu tu doanh vao {MASTER_FILE} ({len(df):,} dong | max date = {new_max})")
    except Exception as e:
        print(f"❌ Loi cap nhat Tu Doanh: {e}")

# ==============================================================================
# PHẦN 4: CẬP NHẬT CHỈ SỐ (VNINDEX) - 🔥 NEW (SOURCE: VNSTOCK/VCI)
# ==============================================================================
def job_update_index():
    print("\n--- [4/5] CAP NHAT CHI SO (VNINDEX) ---")
    if is_weekend(): return
    
    try:
        from vnstock import Quote
    except ImportError:
        print("❌ Lỗi: Chưa cài đặt thư viện 'vnstock'.")
        return

    # Lấy dữ liệu 7 ngày gần nhất để fill gap nếu có
    today = dt.datetime.now()
    start_date = (today - dt.timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    try:
        # Sử dụng vnstock (Source: VCI)
        quote = Quote(symbol='VNINDEX', source='VCI')
        df = quote.history(start=start_date, end=end_date)

        # Normalize to DataFrame
        if df is None:
            print("⚠️ vnstock tra ve du lieu trong.")
            return

        # Convert various possible return shapes into a DataFrame
        if isinstance(df, pd.Series) or isinstance(df, dict):
            df = pd.DataFrame([df])
        else:
            df = pd.DataFrame(df)

        # If the time/index is the index rather than a column, reset it
        if ('time' not in df.columns) and ('dt' not in df.columns) and ('date' not in df.columns):
            df = df.reset_index()

        if df is None or df.empty:
            print("⚠️ vnstock trả về dữ liệu trống.")
            return

        # Mapping cột (VCI trả về: time/dt, open, high, low, close, volume)
        rename_map = {
            'time': 'date',
            'dt': 'date',
            'date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        df = df.rename(columns=rename_map)

        # Coerce and normalize columns
        df['date'] = pd.to_datetime(df.get('date'), errors='coerce').dt.strftime('%Y-%m-%d')
        df['volume'] = pd.to_numeric(df.get('volume', 0), errors='coerce').fillna(0)
        for c in ['open', 'high', 'low', 'close']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Chỉ lấy cột cần thiết
        cols = ["date", "open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]]

        # Lưu file
        filepath = os.path.join(BASE_DIR, "VNINDEX.csv")
        
        if os.path.exists(filepath):
            old_df = pd.read_csv(filepath)
            
            # Merge thông minh
            for _, row in df.iterrows():
                d_str = row['date']
                
                # Nếu ngày đã có
                if d_str in old_df['date'].values:
                    # Check xem volume có đổi không (dữ liệu mới hơn)
                    old_vol = old_df.loc[old_df['date'] == d_str, 'volume'].iloc[0]
                    if float(row['volume']) != float(old_vol):
                        print(f"   ℹ️ Cap nhat lai du lieu ngay {d_str}...")
                        old_df = old_df[old_df['date'] != d_str]
                        old_df = pd.concat([old_df, pd.DataFrame([row])], ignore_index=True)
                else:
                    print(f"   ✅ Them ngay moi: {d_str} | Close: {row['close']}")
                    old_df = pd.concat([old_df, pd.DataFrame([row])], ignore_index=True)
            
            old_df.to_csv(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)
            print(f"✅ Tao moi file VNINDEX.csv ({len(df)} dòng)")

        print("✅ Da hoan tat cap nhat VNINDEX.")

    except Exception as e:
        print(f"❌ Loi cap nhat Index (vnstock): {e}")



# ==============================================================================
# PHAN 5: TICK DATA (LENH KHOP TUNG PHIEN)
# ==============================================================================
_TICK_WORKERS = 5
_TICK_PROFILES = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36", "sec-ch-ua": '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"', "sec-ch-ua-platform": '"Windows"', "Accept-Language": "en-US,en;q=0.9"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36", "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"', "sec-ch-ua-platform": '"macOS"', "Accept-Language": "en-GB,en;q=0.9"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0", "sec-ch-ua": '"Microsoft Edge";v="123", "Not:A-Brand";v="8", "Chromium";v="123"', "sec-ch-ua-platform": '"Windows"', "Accept-Language": "en-US,en;q=0.8"},
    {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36", "sec-ch-ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"', "sec-ch-ua-platform": '"Linux"', "Accept-Language": "en-US,en;q=0.7"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0", "Accept-Language": "en-US,en;q=0.5"},
]
_tick_local = threading.local()
_tick_lock  = threading.Lock()


def _tick_log(msg):
    with _tick_lock:
        print(msg, flush=True)


def _tick_session(profile):
    if not hasattr(_tick_local, "session"):
        s = requests.Session()
        s.headers.update({
            "Accept": "application/json", "Origin": "https://finpath.vn",
            "Referer": "https://finpath.vn/", "Client-Type": "web",
            "sec-fetch-dest": "empty", "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site", "sec-ch-ua-mobile": "?0",
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
                _tick_log(f"  [WAIT] {symbol} rate limited -- waiting 10s")
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
        except Exception as e:
            _tick_log(f"  [ERR] {symbol} page {page}: {e}")
            break
    return pd.DataFrame(all_trades)

def _fetch_and_save_tick(args):
    symbol, out_path, sector, platform, profile_idx, total, position = args
    profile = _TICK_PROFILES[profile_idx % len(_TICK_PROFILES)]
    _tick_log(f"[{position:>3}/{total}] {symbol:>6}  fetching...")
    
    new_df = _fetch_symbol_trades(symbol, profile)
    if new_df.empty:
        _tick_log(f"[{position:>3}/{total}] {symbol:>6}  NO DATA (API returned empty)")
        return symbol, "failed"
        
    new_df.insert(0, "symbol",   symbol)
    new_df.insert(1, "sector",   sector)
    new_df.insert(2, "platform", platform)

    # --- NEW MERGE LOGIC ---
    if os.path.exists(out_path):
        try:
            # Read existing master file
            old_df = pd.read_parquet(out_path)
            
            # Combine new and old data
            combined_df = pd.concat([old_df, new_df], ignore_index=True)
            
            # Drop exact duplicates to prevent duplicate rows if script runs twice in a day
            combined_df = combined_df.drop_duplicates()
            
            # Optional: sort by time if the API provides it (usually 'time' or 'matchTime')
            if 'time' in combined_df.columns:
                combined_df = combined_df.sort_values(by=['time']).reset_index(drop=True)
                
            combined_df.to_parquet(out_path, index=False, engine="pyarrow")
            _tick_log(f"[{position:>3}/{total}] {symbol:>6}  OK  Appended {len(new_df)} trades (Total: {len(combined_df)})")
        except Exception as e:
            _tick_log(f"[{position:>3}/{total}] {symbol:>6}  ERR merging file: {e}")
            return symbol, "failed"
    else:
        # Create new master file if it doesn't exist
        new_df.to_parquet(out_path, index=False, engine="pyarrow")
        _tick_log(f"[{position:>3}/{total}] {symbol:>6}  OK  Created new file ({len(new_df)} trades)")
        
    return symbol, "ok"


def job_update_tick_data():
    print("\n--- [5/5] CAP NHAT TICK DATA ---")
    if is_weekend():
        print("Hom nay la cuoi tuan. Bo qua.")
        return

    sector_csv = os.path.join(BASE_DIR, "data", "sector_master.csv")
    if not os.path.exists(sector_csv):
        print(f"Khong tim thay sector_master.csv tai {sector_csv}")
        return

    sectors         = pd.read_csv(sector_csv, encoding="utf-8-sig")
    symbols         = sectors["symbol"].dropna().unique().tolist()
    sector_lookup   = sectors.set_index("symbol")["sector"].to_dict()
    platform_lookup = sectors.set_index("symbol")["platform"].to_dict()

    # --- CHANGED: Target the master folder, no date sub-folder ---
    master_dir = os.path.join(DATA_DIR, "tick_data")
    os.makedirs(master_dir, exist_ok=True)

    print(f"{len(symbols)} symbols to update today.")
    print(f"Running {_TICK_WORKERS} parallel sessions\n")

    # --- CHANGED: We fetch for ALL symbols, since we need today's data appended ---
    tasks = [
        (sym, os.path.join(master_dir, f"{sym}.parquet"), 
         sector_lookup.get(sym, ""), platform_lookup.get(sym, ""),
         i % _TICK_WORKERS, len(symbols), i + 1)
        for i, sym in enumerate(symbols)
    ]

    ok, failed_syms = 0, []
    with ThreadPoolExecutor(max_workers=_TICK_WORKERS) as pool:
        futures = {pool.submit(_fetch_and_save_tick, t): t[0] for t in tasks}
        for future in as_completed(futures):
            sym, status = future.result()
            if status == "ok": ok += 1
            else: failed_syms.append(sym)

    # Retry pass — sequential with back-off
    retry_wait = 15
    for attempt in range(1, 4):
        if not failed_syms:
            break
        print(f"\nRetry {attempt}/3 -- {len(failed_syms)} symbols (waiting {retry_wait}s...)")
        time.sleep(retry_wait)
        retry_wait *= 2
        still_failed = []
        for i, sym in enumerate(failed_syms, 1):
            out_path = os.path.join(master_dir, f"{sym}.parquet")
            task = (sym, out_path, sector_lookup.get(sym, ""), platform_lookup.get(sym, ""), i, len(failed_syms), i)
            _, status = _fetch_and_save_tick(task)
            if status == "ok": ok += 1
            else: still_failed.append(sym)
            time.sleep(2)
        failed_syms = still_failed

    print(f"\nTick data: {ok} updated | {len(failed_syms)} failed")
    if failed_syms:
        print(f"Failed: {failed_syms}")
    print(f"Output: {master_dir}")

# ==============================================================================
# PHẦN 6: HOT TICKERS (FLOW ACCUMULATION + PRICE MOMENTUM)
# ==============================================================================
def job_hot_tickers():
    print("\n--- [6/7] HOT TICKERS (FLOW + MOMENTUM) ---")
    if is_weekend():
        print("Cuoi tuan. Bo qua.")
        return

    today_iso = get_today_str()
    as_of = pd.Timestamp(today_iso)

    # ── 1. FLOW SIGNAL: top accumulators ─────────────────────────────────────
    flow_results = []
    try:
        from flow_signals import FlowSignalEngine
        fse = FlowSignalEngine().load()
        for tkr in fse.tickers:
            try:
                score = fse.smart_score(tkr, as_of)
                if score > 0.15:
                    dist  = fse.distribution_alert(tkr, as_of)
                    regime = fse.flow_regime(tkr, as_of)
                    flow_results.append((tkr, score, dist, regime))
            except Exception:
                pass
        flow_results.sort(key=lambda x: x[1], reverse=True)
        print(f"   Flow: {len(flow_results)} tickers with score > 0.15")
    except Exception as e:
        print(f"   ⚠️ Flow signals unavailable: {e}")

    # ── 2. MOMENTUM + DEMAND SCANNER ─────────────────────────────────────────
    momentum_results = []
    demand_results   = []  # (tkr, sector, fc_score, mom_pct, vol_ratio)
    try:
        # Load sector lookup once
        sector_map = {}
        try:
            sm = pd.read_csv(os.path.join(DATA_DIR, "sector_master.csv"), encoding="utf-8-sig")
            sector_map = sm.set_index("symbol")["sector"].to_dict()
        except Exception:
            pass

        SCAN_WINDOW = 20
        MIN_LIQ     = 1_000_000_000   # 1B VND median daily turnover

        price_files = [f for f in os.listdir(PRICE_DIR) if f.endswith(".parquet")]
        for fname in price_files:
            tkr = fname.replace(".parquet", "")
            try:
                df = pd.read_parquet(os.path.join(PRICE_DIR, fname))
                if df.empty or len(df) < 25:
                    continue
                df = df.sort_values("time")
                close  = pd.to_numeric(df["close"],  errors="coerce")
                volume = pd.to_numeric(df["volume"], errors="coerce")
                df = df.assign(close=close, volume=volume)
                df = df[(df["close"] > 0) & (df["volume"] >= 0)].tail(80)
                if len(df) < 22:
                    continue

                close  = df["close"]
                volume = df["volume"]

                # ── Momentum ──────────────────────────────────────────────
                ma20       = close.iloc[-20:].mean()
                vol_avg    = volume.iloc[-20:].mean()
                last_close = close.iloc[-1]
                last_vol   = volume.iloc[-1]
                if last_close > ma20 and last_vol > vol_avg * 1.4 and last_close > close.iloc[-2]:
                    pct_above = (last_close / ma20 - 1) * 100
                    vol_x     = last_vol / vol_avg if vol_avg > 0 else 1
                    momentum_results.append((tkr, pct_above, vol_x))

                # ── Demand scanner (full-candle score) ────────────────────
                has_hl = "high" in df.columns and "low" in df.columns
                if not has_hl:
                    continue
                med_to = (close * volume * 1000).tail(60).median()
                if med_to < MIN_LIQ:
                    continue
                recent  = df.tail(SCAN_WINDOW)
                days_ok = (recent["volume"] > 0).sum()
                if days_ok < int(SCAN_WINDOW * 0.75):
                    continue

                opn  = pd.to_numeric(recent["open"],  errors="coerce")
                hi   = pd.to_numeric(recent["high"],  errors="coerce")
                lo   = pd.to_numeric(recent["low"],   errors="coerce")
                cl   = pd.to_numeric(recent["close"], errors="coerce")
                scores, best_to = [], 0.0
                for i in range(len(recent)):
                    body = cl.iloc[i] - opn.iloc[i]
                    rng  = hi.iloc[i] - lo.iloc[i]
                    to_i = cl.iloc[i] * recent["volume"].iloc[i] * 1000
                    if body > 0 and rng > 0 and opn.iloc[i] > 0:
                        s = (body / opn.iloc[i]) * (body / rng)
                        scores.append(s)
                        if to_i > best_to:
                            best_to = to_i
                    else:
                        scores.append(0.0)
                if best_to < MIN_LIQ:
                    continue
                weights  = list(range(1, len(scores) + 1))
                total_w  = sum(weights)
                fc_score = float(sum(s * w for s, w in zip(scores, weights)) / total_w) if total_w else 0.0
                if fc_score <= 0:
                    continue

                p_start = df.iloc[-(SCAN_WINDOW+1)]["close"] if len(df) > SCAN_WINDOW else df.iloc[0]["close"]
                mom_pct = (cl.iloc[-1] / p_start - 1) * 100 if p_start > 0 else 0.0
                base_v  = df.iloc[-(SCAN_WINDOW+60):-(SCAN_WINDOW)]["volume"].mean() if len(df) >= SCAN_WINDOW+60 else volume.mean()
                vol_r   = recent["volume"].mean() / base_v if base_v > 0 else 1.0

                sector = sector_map.get(tkr, "")
                demand_results.append((tkr, sector, fc_score, mom_pct, vol_r))

            except Exception:
                pass

        momentum_results.sort(key=lambda x: x[1] * x[2], reverse=True)
        demand_results.sort(key=lambda x: -x[2])
        print(f"   Momentum: {len(momentum_results)} tickers above MA20 with volume spike")
        print(f"   Demand scanner: {len(demand_results)} tickers scored")
    except Exception as e:
        print(f"   ⚠️ Momentum/demand scan unavailable: {e}")

    # ── 3. COMBINE ────────────────────────────────────────────────────────────
    flow_tickers = {t[0] for t in flow_results[:30]}
    mom_tickers  = {t[0] for t in momentum_results[:30]}
    both = flow_tickers & mom_tickers   # appear in both signals

    # Build display rows: both-signal stocks first, then flow-only, then mom-only
    seen = set()
    rows = []
    flow_map = {t[0]: t for t in flow_results}
    mom_map  = {t[0]: (t[1], t[2]) for t in momentum_results}

    for tkr in sorted(both):
        f = flow_map[tkr]
        m = mom_map[tkr]
        rows.append((tkr, "FLOW+MOM", f[1], m[0], m[1], f[3]))
        seen.add(tkr)
    for tkr, score, dist, regime in flow_results[:15]:
        if tkr not in seen:
            rows.append((tkr, "FLOW", score, 0, 0, regime))
            seen.add(tkr)
    for tkr, pct_above, vol_x in momentum_results[:10]:
        if tkr not in seen:
            rows.append((tkr, "MOM", 0, pct_above, vol_x, ""))
            seen.add(tkr)

    # ── 4. CONSOLE ───────────────────────────────────────────────────────────
    print(f"\n   {'Ticker':<7} {'Signal':<10} {'FlowScore':>9}  {'%>MA20':>6}  {'VolX':>5}  Regime")
    print(f"   {'-'*55}")
    for tkr, sig, fscore, pct, vx, regime in rows[:20]:
        fs = f"{fscore:+.3f}" if fscore != 0 else "   —  "
        pa = f"{pct:+.1f}%" if pct != 0 else "   —  "
        vv = f"{vx:.1f}x" if vx > 0 else "  — "
        print(f"   {tkr:<7} {sig:<10} {fs:>9}  {pa:>7}  {vv:>5}  {regime}")

    # ── 5. TELEGRAM ──────────────────────────────────────────────────────────
    if not rows:
        send_telegram_message(f"📊 HOT TICKERS {today_iso}\n\nKhong co tin hieu noi bat hom nay.")
        return

    lines = [f"📊 <b>HOT TICKERS {today_iso}</b>"]
    lines.append(f"Flow accum: {len(flow_results)}  |  Momentum: {len(momentum_results)}  |  Both: {len(both)}\n")

    if both:
        lines.append("🔥 <b>FLOW + MOMENTUM (ca hai tin hieu):</b>")
        for tkr in sorted(both)[:8]:
            f = flow_map[tkr]
            m = mom_map[tkr]
            lines.append(f"  ✅ <b>{tkr}</b>  flow {f[1]:+.3f} | +{m[0]:.1f}%>MA20 | vol {m[1]:.1f}x  [{f[3]}]")
        lines.append("")

    if flow_results:
        lines.append("💹 <b>FLOW ACCUMULATION (top 8):</b>")
        shown = 0
        for tkr, score, dist, regime in flow_results:
            if tkr in both:
                continue
            dist_tag = " ⚠️dist" if dist else ""
            lines.append(f"  {tkr}  {score:+.3f}{dist_tag}  [{regime}]")
            shown += 1
            if shown >= 8:
                break
        lines.append("")

    if momentum_results:
        lines.append("🚀 <b>MOMENTUM (top 6):</b>")
        shown = 0
        for tkr, pct, vx in momentum_results:
            if tkr in both:
                continue
            lines.append(f"  {tkr}  +{pct:.1f}%>MA20  vol {vx:.1f}x")
            shown += 1
            if shown >= 6:
                break

    # ── Demand scanner top stocks ─────────────────────────────────────────────
    if demand_results:
        flow_set = {t[0] for t in flow_results}
        mom_set  = {t[0] for t in momentum_results}
        lines.append("🕯 <b>DEMAND SCANNER (top emerging, last 20 days):</b>")
        shown = 0
        for tkr, sector, fc, mom, vr in demand_results:
            overlap = []
            if tkr in flow_set: overlap.append("flow")
            if tkr in mom_set:  overlap.append("mom")
            tag = f" ✅{'+'.join(overlap)}" if overlap else ""
            sec_short = sector[:15] if sector else "—"
            lines.append(f"  <b>{tkr}</b>  [{sec_short}]  score {fc:.4f} | {mom:+.1f}% | vol {vr:.1f}x{tag}")
            shown += 1
            if shown >= 10:
                break
        lines.append("")

    lines.append("⚠️ Xac nhan tren bieu do truoc khi giao dich.")
    send_telegram_message("\n".join(lines))
    print(f"\n✅ Da gui hot tickers telegram.")


# ==============================================================================
# PHẦN 7: CẬP NHẬT NDT FLOW (INVESTOR CLASSIFICATION - dulieu.nguoiquansat.vn)
# ==============================================================================
def job_update_investor_flow():
    """
    Daily incremental update for investor flow data (PhanLoaiNDTHistory).
    Calls fetch_investor_flow.py --update, which fetches the last 14 days
    for every ticker that already has a parquet file in data/investor_flow/.

    New tickers (first-time backfill) must be added manually:
      python fetch_investor_flow.py TICKER1 TICKER2
    or
      python fetch_investor_flow.py --all   (full backfill for ALL_TICKERS)
    """
    print("\n--- [7/7] CAP NHAT NDT FLOW (INVESTOR CLASSIFICATION) ---")
    if is_weekend():
        print("Cuoi tuan. Bo qua.")
        return

    flow_dir = os.path.join(BASE_DIR, "data", "investor_flow")
    script   = os.path.join(BASE_DIR, "download", "fetch_investor_flow.py")

    if not os.path.exists(script):
        print(f"❌ fetch_investor_flow.py not found at {script}")
        return False

    n_existing = len([f for f in os.listdir(flow_dir)
                      if f.endswith(".parquet")]) if os.path.isdir(flow_dir) else 0

    if n_existing == 0:
        print(f"⚠️  No parquet files found in {flow_dir}.")
        print(f"     Run first-time backfill: python fetch_investor_flow.py --all")
        return

    print(f"   Snapshot update for {n_existing} tickers (yesterday + today)...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, script, "--snapshot"],
            capture_output=True, text=True, encoding="utf-8",
            timeout=120,   # 2 min max — snapshot fetches all tickers in ~20-40s
            cwd=BASE_DIR,
        )
        # Print stdout line-by-line (may contain Vietnamese chars)
        for line in (result.stdout or "").splitlines():
            print(f"   {line}")
        if result.returncode != 0:
            err = (result.stderr or "").strip()
            print(f"❌ fetch_investor_flow exit {result.returncode}: {err}")
            return False
        else:
            print("✅ NDT flow cap nhat thanh cong.")
    except subprocess.TimeoutExpired:
        print("❌ fetch_investor_flow timeout (>2 min). Skipping — will retry tomorrow.")
        return False
    except Exception as e:
        print(f"❌ Loi job_update_investor_flow: {e}")
        return False


# ==============================================================================
# DATE VERIFICATION — reads latest date from each data source after jobs run
# ==============================================================================
def _latest_dates_summary():
    """Return a dict of label -> latest date string for each data source."""
    out = {}

    # Job 1: prices — sample ACB or first available parquet
    try:
        sample = os.path.join(PRICE_DIR, "ACB.parquet")
        if not os.path.exists(sample):
            files = [f for f in os.listdir(PRICE_DIR) if f.endswith(".parquet")]
            sample = os.path.join(PRICE_DIR, files[0]) if files else None
        if sample:
            df = pd.read_parquet(sample)
            out["Gia"] = str(df["time"].max()) if "time" in df.columns else "?"
    except Exception:
        out["Gia"] = "ERR"

    # Job 2: putthrough
    try:
        pt = os.path.join(BASE_DIR, "putthrough", "putthrough_hose_all.csv")
        df = pd.read_csv(pt)
        out["Thoa thuan"] = str(df["date"].max())
    except Exception:
        out["Thoa thuan"] = "ERR"

    # Job 3: tu doanh
    try:
        td = os.path.join(BASE_DIR, "tudoanh", "tudoanh_all.csv")
        df = pd.read_csv(td)
        out["Tu doanh"] = str(df["date"].max())
    except Exception:
        out["Tu doanh"] = "ERR"

    # Job 4: VNINDEX
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "VNINDEX.csv"))
        date_col = "time" if "time" in df.columns else df.columns[0]
        out["VNINDEX"] = str(df[date_col].max())
    except Exception:
        out["VNINDEX"] = "ERR"

    # Job 5: tick data — count today's parquets and sample the newest
    try:
        tick_dir = os.path.join(DATA_DIR, "tick_data")
        today_str = dt.datetime.now(VN_TZ).strftime("%Y-%m-%d")
        if os.path.isdir(tick_dir):
            tick_files = [f for f in os.listdir(tick_dir) if f.endswith(".parquet")]
            # Count how many parquets were modified today (proxy for updated today)
            updated_today = 0
            for f in tick_files:
                fpath = os.path.join(tick_dir, f)
                mtime = dt.datetime.fromtimestamp(os.path.getmtime(fpath), tz=VN_TZ).strftime("%Y-%m-%d")
                if mtime == today_str:
                    updated_today += 1
            out["Tick data"] = f"{updated_today}/{len(tick_files)} ma cap nhat hom nay"
        else:
            out["Tick data"] = "chua co du lieu"
    except Exception:
        out["Tick data"] = "ERR"

    # Job 7: investor flow — sample first available parquet
    try:
        flow_dir = os.path.join(BASE_DIR, "data", "investor_flow")
        files = [f for f in os.listdir(flow_dir) if f.endswith(".parquet")]
        if files:
            df = pd.read_parquet(os.path.join(flow_dir, files[0]))
            date_col = "date" if "date" in df.columns else df.columns[0]
            out["NDT Flow"] = str(pd.to_datetime(df[date_col]).max().date())
    except Exception:
        out["NDT Flow"] = "ERR"

    return out


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # (label shown in Telegram, function)
    jobs = [
        ("Gia & Nuoc ngoai",     job_update_prices),
        ("Thoa thuan",            job_update_putthrough),
        ("Tu doanh",              job_update_tudoanh),
        ("VNINDEX",               job_update_index),
        ("Tick data",             job_update_tick_data),
        ("Hot tickers",           job_hot_tickers),
        ("NDT Flow",              job_update_investor_flow),
    ]

    job_results = []  # list of (label, status, detail)

    for job_label, job_func in jobs:
        try:
            result = job_func()
            if result is False:
                job_results.append((job_label, "FAILED", "loi noi bo (xem log)"))
            else:
                job_results.append((job_label, "OK", ""))
        except Exception as e:
            err = str(e)[:120]
            print(f"ERROR {job_label}: {err}")
            job_results.append((job_label, "FAILED", err))

    print("\nHOAN TAT TOAN BO QUA TRINH UPDATE!")

    failed_jobs = [j for j in job_results if j[1] == "FAILED"]
    done_time   = dt.datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")
    dates       = _latest_dates_summary()

    # ── Per-job status lines ──────────────────────────────────────────────────
    job_lines = []
    for label, status, detail in job_results:
        icon = "✅" if status == "OK" else "❌"
        suffix = f" — {detail}" if status == "FAILED" and detail else ""
        job_lines.append(f"  {icon} {label}{suffix}")

    # ── Data freshness lines ──────────────────────────────────────────────────
    date_lines = []
    for label, val in dates.items():
        date_lines.append(f"  📅 {label}: {val}")

    header = "⚠️ DAILY UPDATE (CO LOI)" if failed_jobs else "✅ DAILY UPDATE HOAN TAT"
    fail_summary = f"  {len(failed_jobs)} job loi / {len(job_results)} tong\n" if failed_jobs else ""

    telegram_msg = (
        f"{header}\n"
        f"🕐 {done_time} (VN)\n"
        f"{fail_summary}"
        f"\n<b>Trang thai job:</b>\n"
        + "\n".join(job_lines)
        + f"\n\n<b>Du lieu moi nhat:</b>\n"
        + "\n".join(date_lines)
    )

    send_telegram_message(telegram_msg)
