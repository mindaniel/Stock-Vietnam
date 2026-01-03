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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
}

def get_today_str():
    """Trả về ngày hiện tại theo giờ Việt Nam dạng YYYY-MM-DD"""
    return dt.datetime.now(VN_TZ).strftime("%Y-%m-%d")

# ==============================================================================
# HELPER: KIỂM TRA NGÀY NGHỈ (WEEKEND CHECK)
# ==============================================================================
def is_weekend():
    """Returns True if today is Saturday (5) or Sunday (6)"""
    weekday = dt.datetime.now(VN_TZ).weekday()
    return weekday >= 5

print(f"🚀 BẮT ĐẦU CHẠY UPDATE NGÀY: {get_today_str()}")
print(f"📂 Thư mục gốc: {BASE_DIR}")

# ==============================================================================
# PHẦN 1: CẬP NHẬT GIÁ & NƯỚC NGOÀI (SNAPSHOT) - TỪ FILE 1
# ==============================================================================
def job_update_prices():
    print("\n--- [1/3] CẬP NHẬT GIÁ & NƯỚC NGOÀI ---")
    
    # 🛑 1. NGĂN CHẶN CHẠY CUỐI TUẦN
    if is_weekend():
        print("⛔ Hôm nay là cuối tuần. Thị trường không giao dịch. Bỏ qua update.")
        return

    # 1.1 Lấy danh sách mã chứng khoán từ các sàn
    def get_symbols(exchange):
        url = f"https://bgapidatafeed.vps.com.vn/getlistckindex/{exchange}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            data = json.loads(r.text)
            return [s for s in data if isinstance(s, str)]
        except:
            return []

    symbols = []
    for exc in ["hose", "hnx", "upcom"]:
        symbols.extend(get_symbols(exc))
    
    symbols = list(set(symbols)) # Loại bỏ trùng lặp
    print(f"✅ Tìm thấy {len(symbols)} mã trên 3 sàn.")

    # 1.2 Lấy dữ liệu Snapshot
    all_data = []
    chunk_size = 400
    print("⏳ Đang tải dữ liệu snapshot từ VPS...")
    
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i+chunk_size]
        url = f"https://bgapidatafeed.vps.com.vn/getliststockdata/{','.join(chunk)}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            try:
                data = r.json()
            except:
                data = json.loads(r.text)
            all_data.extend(data)
        except Exception as e:
            print(f"⚠️ Lỗi chunk {i}: {e}")
    
    if not all_data:
        print("❌ Không lấy được dữ liệu snapshot nào.")
        return

    # 1.3 Xử lý DataFrame
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
    
    wanted_cols = ["symbol", "open", "high", "low", "close", "volume", "value", 
                   "foreign_buy_vol", "foreign_sell_vol", "foreign_buy_val", "foreign_sell_val", 
                   "foreign_room", "date"]
    df = df[[c for c in wanted_cols if c in df.columns]]

    # 1.4 Ghi vào từng file lẻ
    count_updated = 0
    count_skipped = 0
    existing_files = {f.replace('.csv', '') for f in os.listdir(DATA_DIR) if f.endswith('.csv')}
    
    for _, row in df.iterrows():
        symbol = row["symbol"]
        
        if symbol not in existing_files:
            continue
            
        filepath = os.path.join(DATA_DIR, f"{symbol}.csv")
        
        try:
            # Đọc file cũ
            old_df = pd.read_csv(filepath)
            
            # 🛑 CHECK THÔNG MINH: SO SÁNH DỮ LIỆU CŨ
            # Nếu file có dữ liệu, lấy dòng cuối cùng để so sánh
            if not old_df.empty:
                last_row = old_df.iloc[-1]
                
                # Nếu Volume VÀ Close giống hệt ngày hôm qua -> Khả năng cao là ngày nghỉ/dữ liệu cũ
                # (Dùng dung sai nhỏ cho float comparison nếu cần, nhưng volume thường là int exact)
                if (float(row["volume"]) == float(last_row["volume"])) and \
                   (float(row["close"]) == float(last_row["close"])):
                    # Bỏ qua, không update
                    count_skipped += 1
                    continue
            
            # Kiểm tra xem ngày hôm nay đã có chưa (để tránh double insert nếu chạy lại script)
            if row["date"] in old_df["time"].values:
                old_df = old_df[old_df["time"] != row["date"]]
            
            # Tạo dòng mới chuẩn format
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
            
            new_df_row = pd.DataFrame([new_row])
            updated_df = pd.concat([old_df, new_df_row], ignore_index=True)
            updated_df.to_csv(filepath, index=False)
            count_updated += 1
            
        except Exception as e:
            continue

    print(f"✅ Đã cập nhật: {count_updated} mã.")
    print(f"zzz Đã bỏ qua: {count_skipped} mã (do dữ liệu trùng lặp/không thay đổi).")


# ==============================================================================
# PHẦN 2: CẬP NHẬT THỎA THUẬN (PUT-THROUGH)
# ==============================================================================
def job_update_putthrough():
    print("\n--- [2/3] CẬP NHẬT THỎA THUẬN (PUT-THROUGH) ---")
    if is_weekend():
        print("⛔ Cuối tuần. Bỏ qua.")
        return

    MASTER_FILE = os.path.join(PUT_DIR, "putthrough_hose_all.csv")
    url = "https://bgapidatafeed.vps.com.vn/getlistpt"
    
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        data = r.json()
        
        if not data: 
            print("⚠️ Không có dữ liệu thỏa thuận hôm nay.")
            return

        df = pd.DataFrame(data)
        rename = {"sym": "symbol", "marketID": "floor_code"}
        df = df.rename(columns={k: v for k,v in rename.items() if k in df.columns})
        df = df[df["floor_code"].astype(str) == "10"].copy()
        
        if df.empty:
            print("⚠️ Không có thỏa thuận sàn HOSE.")
            return

        df["date"] = get_today_str()
        df["floor"] = "HOSE"
        df = df.sort_values(["symbol", "time"])
        df["cum_volume"] = df.groupby("symbol")["volume"].cumsum()
        df["cum_value"] = df.groupby("symbol")["value"].cumsum()
        
        cols = ["date", "time", "symbol", "price", "volume", "value", "cum_volume", "cum_value", "floor"]
        df = df[[c for c in cols if c in df.columns]]

        # Logic chống trùng lặp đơn giản cho file tổng
        if os.path.exists(MASTER_FILE):
            old = pd.read_csv(MASTER_FILE)
            
            # Check nhanh: Nếu file cũ đã có dữ liệu của ngày hôm nay rồi thì thôi
            if get_today_str() in old["date"].values:
                print("⚠️ Dữ liệu thỏa thuận ngày hôm nay đã tồn tại. Bỏ qua.")
                return
                
            combined = pd.concat([old, df], ignore_index=True)
        else:
            combined = df
            
        combined.to_csv(MASTER_FILE, index=False, encoding="utf-8-sig")
        print(f"✅ Đã lưu {len(df)} giao dịch vào {MASTER_FILE}")

    except Exception as e:
        print(f"❌ Lỗi cập nhật thỏa thuận: {e}")


# ==============================================================================
# PHẦN 3: CẬP NHẬT TỰ DOANH (PROPRIETARY)
# ==============================================================================
def job_update_tudoanh():
    print("\n--- [3/3] CẬP NHẬT TỰ DOANH ---")
    if is_weekend():
        print("⛔ Cuối tuần. Bỏ qua.")
        return

    MASTER_FILE = os.path.join(TD_DIR, "tudoanh_all.csv")
    url = "https://histdatafeed.vps.com.vn/proprietary/snapshot/TOTAL"
    
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        js = r.json()
        data = js.get("data", []) if isinstance(js, dict) else js
        
        if not data:
            print("⚠️ Không có dữ liệu Tự doanh hôm nay.")
            return

        df = pd.DataFrame(data)
        df = df.rename(columns={"Symbol": "symbol"})
        
        # ... (Phần xử lý số liệu giữ nguyên) ...
        cols_num = ["TBuyVol", "TSellVol", "TBuyVal", "TSellVal"]
        for c in cols_num:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        df["buy_volume"] = df.get("TBuyVol", 0)
        df["sell_volume"] = df.get("TSellVol", 0)
        df["buy_value"] = df.get("TBuyVal", 0)
        df["sell_value"] = df.get("TSellVal", 0)
        df["net_volume"] = df["buy_volume"] - df["sell_volume"]
        df["net_value"] = df["buy_value"] - df["sell_value"]
        
        df["date"] = get_today_str()
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

        final_cols = ["date", "symbol", "buy_volume", "sell_volume", "buy_value", "sell_value", "net_volume", "net_value"]
        df = df[[c for c in final_cols if c in df.columns]]

        # Logic chống trùng lặp
        if os.path.exists(MASTER_FILE):
            old = pd.read_csv(MASTER_FILE)
            
            # Check nhanh: Nếu đã có dữ liệu ngày hôm nay -> Skip
            if get_today_str() in old["date"].values:
                print("⚠️ Dữ liệu Tự doanh ngày hôm nay đã tồn tại. Bỏ qua.")
                return

            combined = pd.concat([old, df], ignore_index=True)
        else:
            combined = df

        combined.to_csv(MASTER_FILE, index=False, encoding="utf-8-sig")
        print(f"✅ Đã lưu dữ liệu Tự doanh vào {MASTER_FILE}")

    except Exception as e:
        print(f"❌ Lỗi cập nhật Tự doanh: {e}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    try:
        job_update_prices()
    except Exception as e:
        print(f"❌ CRITICAL ERROR JOB 1: {e}")

    try:
        job_update_putthrough()
    except Exception as e:
        print(f"❌ CRITICAL ERROR JOB 2: {e}")

    try:
        job_update_tudoanh()
    except Exception as e:
        print(f"❌ CRITICAL ERROR JOB 3: {e}")

    print("\n🎯 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH UPDATE!")
