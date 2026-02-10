import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime

# ================= CONFIG =================
BASE_DIR = r"C:\Users\qmn1\OneDrive - University of St Andrews\Desktop\Cashflow VNI\Stock-Vietnam-main\Stock-Vietnam-main"
DATA_DIR = os.path.join(BASE_DIR, "Data")
OUTPUT_FILE = os.path.join(DATA_DIR, "vps_panel_all.parquet")
# ==========================================

def convert_csv_to_parquet():
    print(f"🚀 Starting conversion in: {DATA_DIR}")
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if not all_files:
        print("❌ No CSV files found!")
        return

    print(f"📂 Found {len(all_files)} CSV files. Processing...")

    data_list = []
    
    # 1. Read all CSVs into a list
    for i, f in enumerate(all_files):
        try:
            # Read only essential columns to save memory
            df = pd.read_csv(f)
            
            # Extract Symbol from filename (e.g., 'AAA.csv' -> 'AAA')
            symbol = os.path.basename(f).replace(".csv", "").upper()
            df["symbol"] = symbol
            
            # Normalize Date
            if "time" in df.columns:
                df["date"] = pd.to_datetime(df["time"]).dt.date
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date
            
            # Force numeric
            for col in ["close", "volume", "value", "foreign_buy_val", "foreign_sell_val"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            
            # Calculate Value if missing
            if "value" not in df.columns and "close" in df.columns and "volume" in df.columns:
                df["value"] = df["close"] * df["volume"]

            # Keep only necessary columns (Slim down the file)
            cols_to_keep = ["date", "symbol", "close", "volume", "value", 
                            "foreign_buy_val", "foreign_sell_val"]
            # Add existing columns only
            final_cols = [c for c in cols_to_keep if c in df.columns]
            
            data_list.append(df[final_cols])

        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")
        
        # Show progress
        if i % 100 == 0:
            print(f"   ... processed {i}/{len(all_files)} files")

    # 2. Combine into one giant table
    print("🔨 Merging data...")
    full_df = pd.concat(data_list, ignore_index=True)
    
    # 3. Sort for speed (Time-series data is faster when sorted by Date/Symbol)
    full_df["date"] = pd.to_datetime(full_df["date"])
    full_df = full_df.sort_values(["symbol", "date"])

    # 4. Save as Parquet
    print(f"💾 Saving to {OUTPUT_FILE}...")
    full_df.to_parquet(OUTPUT_FILE, index=False, compression="snappy")
    
    print("✅ DONE! You can now run vni.py instantly.")

if __name__ == "__main__":
    convert_csv_to_parquet()