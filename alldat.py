import requests
import pandas as pd
import datetime
import os
import time

# ==== CONFIG ====
SAVE_DIR = r"C:\Users\Daziel Brilliant\OneDrive - University of St Andrews\Desktop\Cashflow VNI\Data"
os.makedirs(SAVE_DIR, exist_ok=True)
headers = {"User-Agent": "Mozilla/5.0"}

# ==== 1. Lấy danh sách cổ phiếu ====
# chỉ lấy type=STOCK và status=LISTED (bỏ ETF, CW, Bond, Future...)
url_list = (
    "https://api-finfo.vndirect.com.vn/v4/stocks?"
    "q=type:STOCK~status:LISTED"
    "&fields=code,companyName,floor,industryName,type"
    "&size=3000"
)
r = requests.get(url_list, headers=headers)
listing = pd.DataFrame(r.json()["data"])

# Lọc lại phòng trường hợp có lẫn ETF hoặc CW trong data
listing = listing[listing["type"] == "STOCK"].copy()

symbols = listing["code"].dropna().unique().tolist()
print(f"✅ Found {len(symbols)} listed STOCK symbols")

# ==== 2. Hàm lấy dữ liệu DChart ====
def fetch_dchart(symbol, start="2000-01-01", end=None, resolution="1D"):
    if end is None:
        end = datetime.date.today()
    start_ts = int(datetime.datetime.strptime(start, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.datetime.combine(end, datetime.time()).timestamp())

    url = (
        f"https://dchart-api.vndirect.com.vn/dchart/history"
        f"?resolution={resolution}&symbol={symbol}&from={start_ts}&to={end_ts}"
    )
    try:
        r = requests.get(url, headers=headers, timeout=10)
        js = r.json()
        if js.get("s") != "ok" or "t" not in js:
            print(f"⚠️ {symbol}: no data ({js.get('s')})")
            return None
        df = pd.DataFrame({
            "time": [datetime.datetime.fromtimestamp(x) for x in js["t"]],
            "open": js["o"],
            "high": js["h"],
            "low": js["l"],
            "close": js["c"],
            "volume": js["v"]
        })
        df["value"] = df["close"] * df["volume"]
        return df
    except Exception as e:
        print(f"❌ {symbol}: {e}")
        return None

# ==== 3. Loop tải toàn bộ dữ liệu ====
count = 0
for sym in symbols:
    filepath = os.path.join(SAVE_DIR, f"{sym}.csv")
    if os.path.exists(filepath):
        
        continue
    df = fetch_dchart(sym, start="2010-01-01")
    if df is not None and len(df) > 0:
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"  ✅ Saved {sym} ({len(df)} rows)")
        count += 1
    else:
        print(f"  ⚠️ Skipped {sym} (no data)")
    time.sleep(0.4)

print(f"\n🎯 Done! Saved {count} STOCK symbols in {SAVE_DIR}")
