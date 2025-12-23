import os
import requests
import pandas as pd
import datetime as dt
import json
import time

# ==============================================================================
# CẤU HÌNH CHUNG & ĐƯỜNG DẪN
# ==============================================================================
# Lấy đường dẫn thư mục chứa file code này
# Lấy đường dẫn thư mục (Tự động nhận diện môi trường)
try:
    # Cách này chạy trên GitHub hoặc khi chạy file .py
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Cách này chạy trên Jupyter Notebook của bạn
    # Nó sẽ lấy thư mục hiện tại bạn đang mở Notebook
    BASE_DIR = os.getcwd()

# Định nghĩa các thư mục con
DATA_DIR = os.path.join(BASE_DIR, "data")
PUT_DIR = os.path.join(DATA_DIR, "Putthrough")
TD_DIR = os.path.join(DATA_DIR, "TuDoanh")

# Tạo thư mục nếu chưa có
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PUT_DIR, exist_ok=True)
os.makedirs(TD_DIR, exist_ok=True)

# Cấu hình Múi giờ Việt Nam (Quan trọng cho GitHub Actions)
VN_TZ = dt.timezone(dt.timedelta(hours=7))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
}

def get_today_str():
    """Trả về ngày hiện tại theo giờ Việt Nam dạng YYYY-MM-DD"""
    return dt.datetime.now(VN_TZ).strftime("%Y-%m-%d")

print(f"🚀 BẮT ĐẦU CHẠY UPDATE NGÀY: {get_today_str()}")
print(f"📂 Thư mục gốc: {BASE_DIR}")

# ==============================================================================
# PHẦN 1: CẬP NHẬT GIÁ & NƯỚC NGOÀI (SNAPSHOT) - TỪ FILE 1
# ==============================================================================
def job_update_prices():
    print("\n--- [1/3] CẬP NHẬT GIÁ & NƯỚC NGOÀI ---")
    
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
            # API có thể trả về json thuần hoặc text
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
    
    # Đổi tên cột chuẩn
    rename_map = {
        "sym": "symbol", "lastPrice": "close", "openPrice": "open",
        "highPrice": "high", "lowPrice": "low", "avePrice": "average",
        "lot": "lot", "fBVol": "foreign_buy_vol", "fSVolume": "foreign_sell_vol",
        "fBValue": "foreign_buy_val", "fSValue": "foreign_sell_val", "fRoom": "foreign_room"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Tính toán & Làm sạch
    df["date"] = get_today_str()
    # Lưu ý: VPS thường trả về lot (lô chẵn). Nhân 10 hay không tùy thuộc vào sàn.
    # Tuy nhiên code cũ bạn nhân 10, tôi giữ nguyên logic đó.
    df["volume"] = pd.to_numeric(df.get("lot", 0), errors="coerce") * 10 
    df["value"] = pd.to_numeric(df["close"], errors="coerce") * df["volume"]
    
    # Chỉ giữ các cột cần thiết
    wanted_cols = ["symbol", "open", "high", "low", "close", "volume", "value", 
                   "foreign_buy_vol", "foreign_sell_vol", "foreign_buy_val", "foreign_sell_val", 
                   "foreign_room", "date"]
    df = df[[c for c in wanted_cols if c in df.columns]]

    # 1.4 Ghi vào từng file lẻ
    count_updated = 0
    # Lấy danh sách các file CSV đang có sẵn trong folder data
    existing_files = {f.replace('.csv', '') for f in os.listdir(DATA_DIR) if f.endswith('.csv')}
    
    # Chỉ update những mã nào ĐÃ CÓ FILE (để tránh tạo file rác cho mã lạ)
    # Hoặc nếu bạn muốn tạo mới hết thì bỏ dòng if symbol in existing_files
    for _, row in df.iterrows():
        symbol = row["symbol"]
        
        # Logic: Chỉ update nếu mã đó đã có file lịch sử
        if symbol not in existing_files:
            continue
            
        filepath = os.path.join(DATA_DIR, f"{symbol}.csv")
        
        try:
            # Đọc file cũ
            old_df = pd.read_csv(filepath)
            
            # Kiểm tra xem ngày hôm nay đã có chưa
            if row["date"] in old_df["time"].values:
                # Nếu có rồi -> Cập nhật lại dòng đó (xóa dòng cũ, thêm dòng mới)
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
            
            # Nối vào
            new_df_row = pd.DataFrame([new_row])
            updated_df = pd.concat([old_df, new_df_row], ignore_index=True)
            
            # Lưu lại
            updated_df.to_csv(filepath, index=False)
            count_updated += 1
            
        except Exception as e:
            # Nếu lỗi (ví dụ file cũ lỗi format), bỏ qua
            continue

    print(f"✅ Đã cập nhật giá cho {count_updated} mã.")


# ==============================================================================
# PHẦN 2: CẬP NHẬT THỎA THUẬN (PUT-THROUGH) - TỪ FILE 2
# ==============================================================================
def job_update_putthrough():
    print("\n--- [2/3] CẬP NHẬT THỎA THUẬN (PUT-THROUGH) ---")
    MASTER_FILE = os.path.join(PUT_DIR, "putthrough_hose_all.csv")
    
    url = "https://bgapidatafeed.vps.com.vn/getlistpt"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        data = r.json()
        
        if not data: 
            print("⚠️ Không có dữ liệu thỏa thuận hôm nay.")
            return

        df = pd.DataFrame(data)
        
        # Đổi tên cột
        rename = {"sym": "symbol", "marketID": "floor_code"}
        df = df.rename(columns={k: v for k,v in rename.items() if k in df.columns})
        
        # Lọc sàn HOSE (Mã 10). Nếu muốn lấy hết thì bỏ dòng này.
        df = df[df["floor_code"].astype(str) == "10"].copy()
        
        if df.empty:
            print("⚠️ Không có thỏa thuận sàn HOSE.")
            return

        # Thêm cột ngày
        df["date"] = get_today_str()
        df["floor"] = "HOSE"
        
        # Tính lũy kế (Cumulative) theo mã trong ngày
        df = df.sort_values(["symbol", "time"])
        df["cum_volume"] = df.groupby("symbol")["volume"].cumsum()
        df["cum_value"] = df.groupby("symbol")["value"].cumsum()
        
        # Chọn cột
        cols = ["date", "time", "symbol", "price", "volume", "value", "cum_volume", "cum_value", "floor"]
        df = df[[c for c in cols if c in df.columns]]

        # Cập nhật vào Master File
        if os.path.exists(MASTER_FILE):
            old = pd.read_csv(MASTER_FILE)
            # Xóa dữ liệu cũ của ngày hôm nay (để update lại cho mới nhất)
            old = old[old["date"] != get_today_str()]
            combined = pd.concat([old, df], ignore_index=True)
        else:
            combined = df
            
        combined.to_csv(MASTER_FILE, index=False, encoding="utf-8-sig")
        print(f"✅ Đã lưu {len(df)} giao dịch vào {MASTER_FILE}")

    except Exception as e:
        print(f"❌ Lỗi cập nhật thỏa thuận: {e}")


# ==============================================================================
# PHẦN 3: CẬP NHẬT TỰ DOANH (PROPRIETARY) - TỪ FILE 3
# ==============================================================================
def job_update_tudoanh():
    print("\n--- [3/3] CẬP NHẬT TỰ DOANH ---")
    MASTER_FILE = os.path.join(TD_DIR, "tudoanh_all.csv")
    
    url = "https://histdatafeed.vps.com.vn/proprietary/snapshot/TOTAL"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        js = r.json()
        
        # Xử lý data lồng nhau của VPS
        data = js.get("data", []) if isinstance(js, dict) else js
        
        if not data:
            print("⚠️ Không có dữ liệu Tự doanh hôm nay.")
            return

        df = pd.DataFrame(data)
        df = df.rename(columns={"Symbol": "symbol"})

        # Chuyển số
        cols_num = ["TBuyVol", "TSellVol", "TBuyVal", "TSellVal"]
        for c in cols_num:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        # Tính Net
        df["buy_volume"] = df.get("TBuyVol", 0)
        df["sell_volume"] = df.get("TSellVol", 0)
        df["buy_value"] = df.get("TBuyVal", 0)
        df["sell_value"] = df.get("TSellVal", 0)
        df["net_volume"] = df["buy_volume"] - df["sell_volume"]
        df["net_value"] = df["buy_value"] - df["sell_value"]
        
        df["date"] = get_today_str()
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

        # Chọn cột
        final_cols = ["date", "symbol", "buy_volume", "sell_volume", "buy_value", "sell_value", "net_volume", "net_value"]
        df = df[[c for c in final_cols if c in df.columns]]

        # Cập nhật Master File
        if os.path.exists(MASTER_FILE):
            old = pd.read_csv(MASTER_FILE)
            # Xóa dữ liệu hôm nay nếu đã có (để ghi đè bản mới nhất)
            old = old[old["date"] != get_today_str()]
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
    # Chạy lần lượt 3 job
    # Dùng try-except để job này lỗi không ảnh hưởng job kia
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