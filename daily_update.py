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
PUT_DIR = os.path.join(DATA_DIR, "Putthrough")
TD_DIR = os.path.join(DATA_DIR, "TuDoanh")

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

print(f"🚀 BẮT ĐẦU CHẠY UPDATE NGÀY: {get_today_str()}")
print(f"📂 Folder lưu dữ liệu: {DATA_DIR}")

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
# 1. JOB: UPDATE PRICES & ORDER BOOK SNAPSHOT
# (Replace your old function with this one)
# ==============================================================================
def job_update_prices():
    today_str = get_today_str()
    print(f"\n--- [1/3] UPDATING PRICES & SNAPSHOTS ({today_str}) ---")
    
    if is_weekend():
        print("⛔ Weekend. Skipping.")
        return

    # 1. Get List of Symbols from VPS
    symbols = []
    print("⏳ Fetching symbol list...")
    for exc in ["hose", "hnx", "upcom"]:
        try:
            url = f"https://bgapidatafeed.vps.com.vn/getlistckindex/{exc}"
            r = requests.get(url, headers=HEADERS, timeout=10)
            data = json.loads(r.text)
            symbols.extend([s for s in data if isinstance(s, str)])
        except: pass
    symbols = list(set(symbols))
    print(f"✅ Found {len(symbols)} symbols.")

    # 2. Update Each Symbol
    count = 0
    for symbol in symbols:
        filepath = os.path.join(DATA_DIR, f"{symbol}.csv")
        
        try:
            # --- FETCH FULL DATA FROM VPS (One call gets OHLC + Snapshot) ---
            url = f"https://bgapidatafeed.vps.com.vn/getliststockdata/{symbol}"
            r = requests.get(url, headers=HEADERS, timeout=5)
            data_list = json.loads(r.text)
            
            if not data_list: continue
            item = data_list[0]
            
            # Extract Basic Price
            close = float(item.get('lastPrice', 0))
            if close == 0: continue # No trade happened
            
            # VPS snapshot usually contains 'highPrice', 'lowPrice', 'openPrice'
            # If not, fallback to 'lastPrice'
            high = float(item.get('highPrice', close))
            low = float(item.get('lowPrice', close))
            open_p = float(item.get('openPrice', close))
            
            # Volume: VPS 'lot' is often the Total Volume
            volume = float(item.get('lot', 0))
            if volume == 0:
                # Fallback to 'lastVolume' if 'lot' is missing (rare)
                volume = float(item.get('lastVolume', 0)) 
            else:
                # VPS 'lot' is usually correct, sometimes divided by 10 for HSX?
                # Usually raw 'lot' is fine. We will store raw 'lot'.
                # Multiply by 10 if you notice your volumes are 10x too small compared to history.
                volume = volume * 10 
            
            # Extract Foreign Data
            f_buy = float(item.get('fBVol', 0))
            f_sell = float(item.get('fSVolume', 0))
            f_buy_val = float(item.get('fBValue', 0))
            f_sell_val = float(item.get('fSValue', 0))
            
            # --- CRITICAL: PARSE ORDER BOOK (g1..g6) ---
            # g1 = Best Buy (Bid 1)
            # g4 = Best Sell (Ask 1)
            b1_price, b1_vol = parse_g_string(item.get('g1', ''))
            s1_price, s1_vol = parse_g_string(item.get('g4', ''))
            
            # --- DETECT SELL PRESSURE (The "White Buyer" Signal) ---
            floor_price = float(item.get('f', 0))
            
            is_floor = (close <= floor_price)
            empty_buy_side = (b1_vol == 0)
            
            sell_pressure = 0.0
            if is_floor and empty_buy_side:
                # If floor and no buyers, the pressure is the Sellers piling up at Ask 1
                sell_pressure = s1_vol 
            elif empty_buy_side:
                 # No buyers but not at floor yet? Still pressure.
                 sell_pressure = s1_vol * 0.5
            
            # Construct New Row
            new_row = {
                'date': today_str,
                'open': open_p, 'high': high, 'low': low, 'close': close,
                'volume': volume,
                'foreign_buy_vol': f_buy, 'foreign_sell_vol': f_sell,
                'foreign_buy_val': f_buy_val, 'foreign_sell_val': f_sell_val,
                'buy_vol_1': b1_vol,
                'sell_vol_1': s1_vol,
                'sell_pressure': sell_pressure
            }

            # Save to CSV
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                # Check if today already exists
                if today_str in df['date'].values:
                    # Update existing row (Overwriting allows you to run this multiple times a day)
                    idx = df.index[df['date'] == today_str][0]
                    for k, v in new_row.items():
                        df.at[idx, k] = v
                else:
                    # Append new day
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                # Create new file
                df = pd.DataFrame([new_row])
                
            df.to_csv(filepath, index=False)
            count += 1
            print(f"   ✅ {symbol}: {close} | Pressure: {sell_pressure:.0f}", end='\r')
            
        except Exception as e:
            # print(f"Error {symbol}: {e}") # Uncomment to debug
            pass

    print(f"\n✅ Updated {count} stocks successfully.")
# ==============================================================================
# PHẦN 2: CẬP NHẬT THỎA THUẬN
# ==============================================================================
def job_update_putthrough():
    print("\n--- [2/4] CẬP NHẬT THỎA THUẬN ---")
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
        print(f"✅ Đã lưu thỏa thuận vào {MASTER_FILE}")
    except: pass

# ==============================================================================
# PHẦN 3: CẬP NHẬT TỰ DOANH
# ==============================================================================
def job_update_tudoanh():
    print("\n--- [3/4] CẬP NHẬT TỰ DOANH ---")
    if is_weekend(): return

    MASTER_FILE = os.path.join(TD_DIR, "tudoanh_all.csv")
    url = "https://histdatafeed.vps.com.vn/proprietary/snapshot/TOTAL"
    
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        data = r.json()
        data = data.get("data", []) if isinstance(data, dict) else data
        if not data: return

        df = pd.DataFrame(data)
        df = df.rename(columns={"Symbol": "symbol"})
        
        df["buy_volume"] = pd.to_numeric(df.get("TBuyVol", 0), errors="coerce").fillna(0)
        df["sell_volume"] = pd.to_numeric(df.get("TSellVol", 0), errors="coerce").fillna(0)
        df["buy_value"] = pd.to_numeric(df.get("TBuyVal", 0), errors="coerce").fillna(0)
        df["sell_value"] = pd.to_numeric(df.get("TSellVal", 0), errors="coerce").fillna(0)
        df["net_volume"] = df["buy_volume"] - df["sell_volume"]
        df["net_value"] = df["buy_value"] - df["sell_value"]
        df["date"] = get_today_str()

        final_cols = ["date", "symbol", "buy_volume", "sell_volume", "buy_value", "sell_value", "net_volume", "net_value"]
        df = df[[c for c in final_cols if c in df.columns]]

        if os.path.exists(MASTER_FILE):
            old = pd.read_csv(MASTER_FILE)
            if get_today_str() not in old["date"].values:
                df = pd.concat([old, df], ignore_index=True)
            else: return
        
        df.to_csv(MASTER_FILE, index=False, encoding="utf-8-sig")
        print(f"✅ Đã lưu tự doanh vào {MASTER_FILE}")
    except: pass

# ==============================================================================
# PHẦN 4: CẬP NHẬT CHỈ SỐ (VNINDEX) - 🔥 NEW (SOURCE: VNSTOCK/VCI)
# ==============================================================================
def job_update_index():
    print("\n--- [4/4] CẬP NHẬT CHỈ SỐ (VNINDEX) ---")
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
        # Sử dụng vnstock (Source: VCI) - Cách này đã được test ok
        quote = Quote(symbol='VNINDEX', source='VCI')
        df = quote.history(start=start_date, end=end_date)

        if df is None or df.empty:
            print("⚠️ vnstock trả về dữ liệu trống.")
            return

        # Mapping cột (VCI trả về: time, open, high, low, close, volume)
        rename_map = {
            'time': 'date',
            'dt': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        df = df.rename(columns=rename_map)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # Chỉ lấy cột cần thiết
        cols = ["date", "open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]]

        # Lưu file
        filepath = os.path.join(DATA_DIR, "VNINDEX.csv")
        
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
                        print(f"   ℹ️ Cập nhật lại dữ liệu ngày {d_str}...")
                        old_df = old_df[old_df['date'] != d_str]
                        old_df = pd.concat([old_df, pd.DataFrame([row])], ignore_index=True)
                else:
                    print(f"   ✅ Thêm ngày mới: {d_str} | Close: {row['close']}")
                    old_df = pd.concat([old_df, pd.DataFrame([row])], ignore_index=True)
            
            old_df.to_csv(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)
            print(f"✅ Tạo mới file VNINDEX.csv ({len(df)} dòng)")

        print("✅ Đã hoàn tất cập nhật VNINDEX.")

    except Exception as e:
        print(f"❌ Lỗi cập nhật Index (vnstock): {e}")

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

    print("\n🎯 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH UPDATE!")
