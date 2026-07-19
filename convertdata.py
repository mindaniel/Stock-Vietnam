import os
import glob
import pandas as pd

# ================= CONFIG =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRICE_DIR = os.path.join(SCRIPT_DIR, "data", "price")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "data", "price", "vps_panel_all.parquet")
# ==========================================

def convert_csv_to_parquet():
    """Build the single-file panel vni.py reads (vps_panel_all.parquet) from
    the per-ticker parquet files in data/price/ (the pipeline's actual
    current output — this script originally expected per-ticker CSVs in
    data/, which no longer exist)."""
    print(f"[INFO] Reading per-ticker parquet from: {PRICE_DIR}")
    all_files = glob.glob(os.path.join(PRICE_DIR, "*.parquet"))
    all_files = [f for f in all_files if os.path.basename(f) != "vps_panel_all.parquet"]

    if not all_files:
        print("[ERROR] No parquet files found!")
        return

    print(f"[INFO] Found {len(all_files)} files. Processing...")

    data_list = []
    for i, f in enumerate(all_files):
        try:
            symbol = os.path.basename(f).replace(".parquet", "").upper()
            df = pd.read_parquet(f)
            df.columns = [c.strip().lower() for c in df.columns]

            date_col = "time" if "time" in df.columns else "date"
            df["date"] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=["date"])
            df["symbol"] = symbol

            for col in ["close", "volume", "value", "foreign_buy_val", "foreign_sell_val"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            if "value" not in df.columns and {"close", "volume"}.issubset(df.columns):
                df["value"] = df["close"] * df["volume"]

            cols_to_keep = ["date", "symbol", "close", "volume", "value",
                            "foreign_buy_val", "foreign_sell_val"]
            final_cols = [c for c in cols_to_keep if c in df.columns]
            data_list.append(df[final_cols])
        except Exception as e:
            print(f" Error reading {f}: {e}")

        if i % 200 == 0:
            print(f"   ... processed {i}/{len(all_files)} files")

    print("[INFO] Merging data...")
    full_df = pd.concat(data_list, ignore_index=True)
    full_df = full_df.sort_values(["symbol", "date"])

    print(f" Saving to {OUTPUT_FILE}...")
    full_df.to_parquet(OUTPUT_FILE, index=False, compression="snappy")
    print(" DONE! You can now run vni.py instantly.")


if __name__ == "__main__":
    convert_csv_to_parquet()
