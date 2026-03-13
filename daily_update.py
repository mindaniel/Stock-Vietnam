import os
import requests
import pandas as pd
import datetime as dt
import json
import time

# ==============================================================================
# CẤU HÌNH CHUNG & ĐƯỜNG DẪN
# ==============================================================================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "data")
# Linking directly to BASE_DIR moves them outside the data folder
PUT_DIR = os.path.join(BASE_DIR, "putthrough")
TD_DIR = os.path.join(BASE_DIR, "tudoanh")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PUT_DIR, exist_ok=True)
os.makedirs(TD_DIR, exist_ok=True)

VN_TZ = dt.timezone(dt.timedelta(hours=7))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

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
    print("\n--- [1/4] CẬP NHẬT GIÁ & NƯỚC NGOÀI ---")
    if is_weekend():
        print("⛔ Hôm nay là cuối tuần. Bỏ qua.")
        return

    def get_symbols(exchange):
        url = f"https://bgapidatafeed.vps.com.vn/getlistckindex/{exchange}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            data = json.loads(r.text)
            return [s for s in data if isinstance(s, str)]
        except: return []

    symbols = []
    for exc in ["hose", "hnx", "upcom"]:
        symbols.extend(get_symbols(exc))
    symbols = list(set(symbols))
    
    all_data = []
    chunk_size = 400
    print(f"⏳ Đang tải dữ liệu cho {len(symbols)} mã...")
    
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i+chunk_size]
        url = f"https://bgapidatafeed.vps.com.vn/getliststockdata/{','.join(chunk)}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            try: data = r.json()
            except: data = json.loads(r.text)
            all_data.extend(data)
        except: pass
    
    if not all_data: return

    df = pd.DataFrame(all_data)
    rename_map = {
        "sym": "symbol", "lastPrice": "close", "openPrice": "open",
        "highPrice": "high", "lowPrice": "low", "avePrice": "average",
        "lot": "lot", "fBVol": "foreign_buy_vol", "fSVolume": "foreign_sell_vol",
        "fBValue": "foreign_buy_val", "fSValue": "foreign_sell_val", "fRoom": "foreign_room"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["date"] = get_today_str()
    df["volume"] = pd.to_numeric(df.get("lot", 0), errors="coerce") * 10 
    df["value"] = pd.to_numeric(df["close"], errors="coerce") * df["volume"]
    
    count_updated = 0
    existing_files = {f.replace('.csv', '') for f in os.listdir(DATA_DIR) if f.endswith('.csv')}
    
    for _, row in df.iterrows():
        symbol = row["symbol"]
        if symbol not in existing_files: continue
        filepath = os.path.join(DATA_DIR, f"{symbol}.csv")
        
        try:
            old_df = pd.read_csv(filepath)
            if not old_df.empty:
                last_row = old_df.iloc[-1]
                if (float(row["volume"]) == float(last_row["volume"])) and \
                   (float(row["close"]) == float(last_row["close"])):
                    continue
            
            if row["date"] in old_df["time"].values:
                old_df = old_df[old_df["time"] != row["date"]]
            
            new_row = {
                "time": row["date"],
                "open": row["open"], "high": row["high"], "low": row["low"],
                "close": row["close"], "volume": row["volume"], "value": row["value"],
                "foreign_buy_vol": row.get("foreign_buy_vol", 0),
                "foreign_sell_vol": row.get("foreign_sell_vol", 0),
                "foreign_buy_val": row.get("foreign_buy_val", 0),
                "foreign_sell_val": row.get("foreign_sell_val", 0),
                "foreign_room": row.get("foreign_room", 0)
            }
            pd.concat([old_df, pd.DataFrame([new_row])], ignore_index=True).to_csv(filepath, index=False)
            count_updated += 1
        except: continue

    print(f"✅ Đã cập nhật giá: {count_updated} mã.")
# ==============================================================================
# PHẦN 2: CẬP NHẬT THỎA THUẬN
# ==============================================================================
def job_update_putthrough():
    print("\n--- [2/4] CAP NHAT THOA THUAN ---")
    if is_weekend(): return

    MASTER_FILE = os.path.join(PUT_DIR, "putthrough_hose_all.csv")
    url = "https://bgapidatafeed.vps.com.vn/getlistpt"
    
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
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
            if get_today_str() not in old["date"].values:
                final_df = pd.concat([old, final_df], ignore_index=True)
            else: return 
        
        final_df.to_csv(MASTER_FILE, index=False, encoding="utf-8-sig")
        print(f"✅ Da luu thoa thuan vao {MASTER_FILE}")
    except: pass

# ==============================================================================
# PHẦN 3: CẬP NHẬT TỰ DOANH
# ==============================================================================
def job_update_tudoanh():
    print("\n--- [3/4] CAP NHAT TU DOANH ---")
    if is_weekend(): return

    MASTER_FILE = os.path.join(TD_DIR, "tudoanh_all.csv")
    url = "https://histdatafeed.vps.com.vn/proprietary/snapshot/TOTAL"
    
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        data = r.json()
        data = data.get("data", []) if isinstance(data, dict) else data
        if not data:
            print("⚠️ API Tự Doanh không trả dữ liệu.")
            return

        df = pd.DataFrame(data)
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
        df["date"] = dt.datetime.now(VN_TZ).strftime("%d/%m/%Y")

        final_cols = ["date", "symbol", "buy_volume", "sell_volume", "buy_value", "sell_value", "net_volume", "net_value"]
        df = df[[c for c in final_cols if c in df.columns]]

        if os.path.exists(MASTER_FILE):
            old = pd.read_csv(MASTER_FILE)
            today = dt.datetime.now(VN_TZ).strftime("%d/%m/%Y")
            if today in old.get("date", pd.Series()).values:
                old = old[old["date"] != today]
            df = pd.concat([old, df], ignore_index=True)
        
        df.to_csv(MASTER_FILE, index=False, encoding="utf-8-sig")
        print(f"✅ Da luu tu doanh vao {MASTER_FILE} ({len(df):,} dong)")
    except Exception as e:
        print(f"❌ Loi cap nhat Tu Doanh: {e}")

# ==============================================================================
# PHẦN 4: CẬP NHẬT CHỈ SỐ (VNINDEX) - 🔥 NEW (SOURCE: VNSTOCK/VCI)
# ==============================================================================
def job_update_index():
    print("\n--- [4/4] CAP NHAT CHI SO (VNINDEX) ---")
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
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    try: job_update_prices()
    except Exception as e: print(f"❌ ERROR JOB 1: {e}")

    try: job_update_putthrough()
    except Exception as e: print(f"❌ ERROR JOB 2: {e}")

    try: job_update_tudoanh()
    except Exception as e: print(f"❌ ERROR JOB 3: {e}")
    
    try: job_update_index()
    except Exception as e: print(f"❌ ERROR JOB 4: {e}")

    print("\nHOAN TAT TOAN BO QUA TRINH UPDATE!")
