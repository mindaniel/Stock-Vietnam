import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "archive" not in os.getcwd() else os.getcwd()

def inspect_csv(filepath, name):
    print(f"\n{'='*60}\n 🔍 INSPECTING CSV: {name}\n{'-'*60}")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    try:
        df = pd.read_csv(filepath)
        print(f"Total Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        # Find date column
        date_col = next((c for c in ['date', 'time', 'TradingDate', 'matchTime'] if c in df.columns), df.columns[0])
        print(f"\nTarget Date Column: '{date_col}' (Type: {df[date_col].dtype})")
        
        print("\nFirst 5 raw values:")
        for val in df[date_col].head(5).tolist():
            print(f"  - {repr(val)}")
            
        print("\nLast 5 raw values:")
        for val in df[date_col].tail(5).tolist():
            print(f"  - {repr(val)}")
        
    except Exception as e:
        print(f"Error reading {name}: {e}")

def inspect_parquet_folder(dir_path, name):
    print(f"\n{'='*60}\n 🔍 INSPECTING PARQUET DIR: {name}\n{'-'*60}")
    if not os.path.exists(dir_path):
        print(f"Folder not found: {dir_path}")
        return
        
    files = [f for f in os.listdir(dir_path) if f.endswith('.parquet')]
    if not files:
        print("No parquet files found.")
        return
        
    # Pick the newest file
    files.sort(key=lambda x: os.path.getmtime(os.path.join(dir_path, x)), reverse=True)
    sample_file = files[0]
    filepath = os.path.join(dir_path, sample_file)
    
    print(f"Sampling newest file: {sample_file}")
    try:
        df = pd.read_parquet(filepath)
        print(f"Total Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        date_col = next((c for c in ['date', 'time', 'TradingDate', 'matchTime'] if c in df.columns), df.columns[0])
        print(f"\nTarget Date Column: '{date_col}' (Type: {df[date_col].dtype})")
        
        print("\nFirst 5 raw values:")
        for val in df[date_col].head(5).tolist():
            print(f"  - {repr(val)}")
            
        print("\nLast 5 raw values:")
        for val in df[date_col].tail(5).tolist():
            print(f"  - {repr(val)}")
            
    except Exception as e:
        print(f"Error reading {sample_file}: {e}")

if __name__ == "__main__":
    inspect_csv(os.path.join(BASE_DIR, "tudoanh", "tudoanh_all.csv"), "Tự Doanh")
    inspect_csv(os.path.join(BASE_DIR, "putthrough", "putthrough_hose_all.csv"), "Thỏa Thuận")
    inspect_csv(os.path.join(BASE_DIR, "VNINDEX.csv"), "VNINDEX")
    inspect_parquet_folder(os.path.join(BASE_DIR, "data", "tick_data"), "Tick Data")