import os
import pandas as pd
import datetime as dt
import warnings

# Suppress pandas UserWarnings for a clean terminal output
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "archive" not in os.getcwd() else os.getcwd()

def find_date_col(df):
    """Helper to find the most likely date/time column in a DataFrame"""
    candidates = ['date', 'time', 'TradingDate', 'matchTime', 'td', 't']
    for col in candidates:
        if col in df.columns:
            return col
    return None

def get_max_valid_date(df, date_col):
    """Parses dates safely, handles Unix timestamps, and ignores absurd future dates"""
    series = df[date_col].dropna()
    if series.empty:
        return pd.NaT
        
    try:
        # Special handling for Unix timestamps (Tick Data 't' column)
        if date_col == 't' and pd.api.types.is_numeric_dtype(series):
            if series.max() > 1e11:
                dates = pd.to_datetime(series, unit='ms', errors='coerce')
            else:
                dates = pd.to_datetime(series, unit='s', errors='coerce')
        else:
            # SMART MIXED PARSING: Force pandas to evaluate every row individually.
            # This fixes the VNINDEX bug where old data is DD/MM/YYYY and new data is YYYY-MM-DD.
            try:
                # For Pandas 2.0+
                dates = pd.to_datetime(series, format='mixed', dayfirst=True, errors='coerce')
            except ValueError:
                # Fallback for older Pandas versions
                dates = pd.to_datetime(series, infer_datetime_format=True, dayfirst=True, errors='coerce')
            
        dates = dates.dropna()
        if dates.empty:
            return pd.NaT
            
        # Filter out future dates (gives a 2-day buffer for timezones)
        now_buffer = pd.Timestamp.now() + pd.Timedelta(days=2)
        valid_dates = dates[dates <= now_buffer]
        
        if valid_dates.empty:
            return dates.max() # Fallback
            
        return valid_dates.max()
        
    except Exception:
        return pd.NaT

def get_csv_status(filepath):
    """Reads a CSV and returns (total_rows, latest_date)"""
    if not os.path.exists(filepath):
        return 0, "File Not Found"
    
    try:
        df = pd.read_csv(filepath)
        df = df.dropna(how='all') # Clean completely empty rows (VNINDEX issue)
        
        if df.empty:
            return 0, "Empty File"
        
        date_col = find_date_col(df)
        if date_col:
            max_date = get_max_valid_date(df, date_col)
            if pd.isna(max_date):
                return len(df), "Invalid Dates"
            return len(df), max_date.strftime("%Y-%m-%d")
        else:
            return len(df), "No Date Column"
    except Exception as e:
        return 0, f"Error: {str(e)[:30]}"

def get_parquet_dir_status(dir_path):
    """Reads the most recently modified Parquet files to find the latest date"""
    if not os.path.exists(dir_path):
        return 0, "Folder Not Found"
    
    files = [f for f in os.listdir(dir_path) if f.endswith('.parquet')]
    if not files:
        return 0, "Empty Folder"
    
    # Sort files by modification time, newest first
    files.sort(key=lambda x: os.path.getmtime(os.path.join(dir_path, x)), reverse=True)
    
    max_overall_date = pd.NaT
    
    # Scan top 5 recently modified files to find the true max date
    for f in files[:5]:
        try:
            df = pd.read_parquet(os.path.join(dir_path, f))
            date_col = find_date_col(df)
            if date_col:
                file_max = get_max_valid_date(df, date_col)
                if not pd.isna(file_max):
                    if pd.isna(max_overall_date) or file_max > max_overall_date:
                        max_overall_date = file_max
        except Exception:
            continue
            
    if pd.isna(max_overall_date):
        return f"{len(files):,}", "No Valid Dates"
        
    return f"{len(files):,}", max_overall_date.strftime("%Y-%m-%d")

def main():
    print("\n" + "="*60)
    print(" 📊 DATA INTEGRITY & HEALTH CHECK REPORT")
    print("="*60)
    print(f"{'Data Source':<25} | {'Latest Date Inside':<18} | {'Size / Files':<15}")
    print("-" * 60)

    # 1. Price Folder
    sz, dt_str = get_parquet_dir_status(os.path.join(BASE_DIR, "data", "price"))
    print(f"{'Gia & NN (Price)':<25} | {dt_str:<18} | {sz} files")

    # 2. Tick Data Folder
    sz, dt_str = get_parquet_dir_status(os.path.join(BASE_DIR, "data", "tick_data"))
    print(f"{'Tick Data (Lenh)':<25} | {dt_str:<18} | {sz} files")

    # 3. NDT Flow Folder
    sz, dt_str = get_parquet_dir_status(os.path.join(BASE_DIR, "data", "investor_flow"))
    print(f"{'NDT Flow (Ca nhan)':<25} | {dt_str:<18} | {sz} files")

    # 4. CSV Files
    sz, dt_str = get_csv_status(os.path.join(BASE_DIR, "tudoanh", "tudoanh_all.csv"))
    print(f"{'Tu Doanh':<25} | {dt_str:<18} | {sz:,} rows")

    sz, dt_str = get_csv_status(os.path.join(BASE_DIR, "putthrough", "putthrough_hose_all.csv"))
    print(f"{'Thoa Thuan':<25} | {dt_str:<18} | {sz:,} rows")

    sz, dt_str = get_csv_status(os.path.join(BASE_DIR, "VNINDEX.csv"))
    print(f"{'VNINDEX':<25} | {dt_str:<18} | {sz:,} rows")

    print("="*60)
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f" Checked at: {now} (Local Time)\n")

if __name__ == "__main__":
    main()
    input()