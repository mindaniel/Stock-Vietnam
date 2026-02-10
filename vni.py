import os
import math
import glob
import datetime as dt
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import re
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.optimize import minimize
from vnstock import Company
# ---------- Import DCF Module ----------
import sys

# Ensure we can import dcf.py from same directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from dcf import compute_cashflows, run_three_scenarios
# Must be the first Streamlit command
st.set_page_config(page_title="Cashflow VNI", layout="wide")

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the script location
DATA_DIR = os.path.join(SCRIPT_DIR, "Data")
LISTING_FILE = os.path.join(SCRIPT_DIR, "vndirect_listing.xlsx")  # để map ngành
COLUMNS_DIR = os.path.join(DATA_DIR, "columns")

@st.cache_resource
def load_column_matrices():
    """Read per-column Feather matrices where date is already the index."""
    close  = pd.read_feather(os.path.join(COLUMNS_DIR, "close.feather"))
    volume = pd.read_feather(os.path.join(COLUMNS_DIR, "volume.feather"))

    # detect if 'date' is the index or just the first column
    if "date" in close.columns:
        close = close.set_index("date")
    else:
        # in your case, the index is already the date
        close.index.name = "date"

    if "date" in volume.columns:
        volume = volume.set_index("date")
    else:
        volume.index.name = "date"

    # ensure proper datetime type
    close.index = pd.to_datetime(close.index)
    volume.index = pd.to_datetime(volume.index)

    return close, volume

# Show configuration status
st.sidebar.markdown("### 📁 Configuration Status")

# Diagnostic information
st.sidebar.write("📂 Script directory:", SCRIPT_DIR)
st.sidebar.write("📂 Data directory:", DATA_DIR)
st.sidebar.write("📄 Listing file:", LISTING_FILE)

# Check Data directory
if not os.path.exists(DATA_DIR):
    st.sidebar.error(f"❌ Data directory not found at: {DATA_DIR}")
    os.makedirs(DATA_DIR)
    st.sidebar.success("✅ Created Data directory")
else:
    st.sidebar.success(f"✅ Found Data directory")
    # Check contents of Data directory
    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    st.sidebar.write(f"📊 Found {len(data_files)} CSV files in Data directory")
    if len(data_files) > 0:
        st.sidebar.write("Sample files:", data_files[:5])

# Check listing file
if not os.path.exists(LISTING_FILE):
    st.sidebar.error(f"❌ Listing file not found at: {LISTING_FILE}")
else:
    try:
        # Try to read the first few rows to verify file is accessible
        df = pd.read_excel(LISTING_FILE, nrows=5)
        st.sidebar.success(f"✅ Successfully read listing file")
        st.sidebar.write("📋 First few columns in listing file:", list(df.columns))
    except Exception as e:
        st.sidebar.error(f"❌ Error reading listing file: {str(e)}")

INDEX_SYMBOL = "VNINDEX"  # chỉ để hiển thị mini chart nếu cần
# ================================ #

# ---------- Helpers ---------- #
# ==========================================
# 🧹 GLOBAL HELPER: CLEAN DAILY DATA
# ==========================================
def clean_daily(df):
    """
    Standardizes a dataframe to ensure it has 'date', 'close', 'volume' 
    and foreign flow columns in numeric format.
    """
    df = df.copy()

    # 1. Normalize Date Column
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["date"] = df["time"].dt.date # Create a pure date column
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # 2. Force Numeric Columns (Crucial for calculations)
    numeric_cols = ["close", "volume", "foreign_buy_val", "foreign_sell_val"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # 3. Sort by Date
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
        
    return df

def read_symbol_csv(path):
    """Đọc file CSV từng mã và chuẩn hóa định dạng."""
    df = pd.read_csv(path)
    # Chuẩn hóa cột thời gian
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    else:
        # fallback: tìm cột chứa 'date' hay 'time'
        for c in df.columns:
            if "time" in c.lower() or "date" in c.lower():
                df["time"] = pd.to_datetime(df[c])
                break
    df = df.sort_values("time").reset_index(drop=True)

    # Chuẩn hóa kiểu số
    for c in ["open", "high", "low", "close", "volume", "value", "avg_price"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Nếu không có 'value' → ước tính value = close × volume
    if "value" not in df.columns and {"close", "volume"}.issubset(df.columns):
        df["value"] = df["close"] * df["volume"]
    elif "value" not in df.columns:
        df["value"] = np.nan

    return df


def list_symbol_files(data_dir=DATA_DIR):
    """Liệt kê danh sách file CSV trong thư mục dữ liệu."""
    return sorted(glob.glob(os.path.join(data_dir, "*.csv")))


def symbol_from_path(p):
    """Lấy mã cổ phiếu từ tên file."""
    return os.path.splitext(os.path.basename(p))[0].upper()


def period_return(series, periods):
    """Tính phần trăm thay đổi trong n phiên."""
    if len(series) < periods + 1:
        return np.nan
    return (series.iloc[-1] / series.iloc[-(periods + 1)] - 1.0) * 100.0


def compute_returns(close_series):
    """Tạo các mốc return tiêu chuẩn."""
    ret = {}
    ret["R_1D_%"] = period_return(close_series, 1)
    ret["R_2W_%"] = period_return(close_series, 10)
    ret["R_1M_%"] = period_return(close_series, 21)
    ret["R_3M_%"] = period_return(close_series, 63)
    ret["R_1Y_%"] = period_return(close_series, 252)
    return ret


def money_flow_proxy(df):
    """Chỉ báo dòng tiền đơn giản (Typical Price × Volume)."""
    if {"high", "low", "close", "volume"}.issubset(df.columns):
        tp = (df["high"] + df["low"] + df["close"]) / 3
        tp_prev = tp.shift(1)
        mf = (tp - tp_prev) * df["volume"]
        return mf.fillna(0)
    else:
        return pd.Series(np.nan, index=df.index)


def vwap_month(df):
    """Tính VWAP tích lũy."""
    if not {"high", "low", "close", "volume"}.issubset(df.columns):
        return None
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_pv = (tp * df["volume"]).cumsum()
    cum_v = df["volume"].cumsum()
    return cum_pv / cum_v


# =================== LOAD ALL DATA (FROM CSVs) =================== #
@st.cache_data(show_spinner=True) # ⚡ Important: Cache this so it doesn't run every click
def load_all_data():
    """Reads all CSV files in Data directory instead of parquet."""
    
    # Get list of CSVs
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if not csv_files:
        st.error("❌ No CSV files found in Data directory.")
        return {}, pd.DataFrame()

    data_map = {}
    meta_rows = []
    
    # Progress bar for loading 1500+ files
    progress_text = "Loading data from CSVs..."
    my_bar = st.progress(0, text=progress_text)
    total_files = len(csv_files)

    for i, file_path in enumerate(csv_files):
        try:
            # Parse symbol from filename (e.g., "AAA.csv" -> "AAA")
            sym = symbol_from_path(file_path)
            
            # Use your existing helper to read the CSV
            df = read_symbol_csv(file_path)

            if not df.empty:
                # Add to main dictionary
                data_map[sym] = df
                
                # Capture metadata (last row info) for the summary table
                last_row = df.iloc[-1]
                meta_rows.append({
                    "symbol": sym,
                    "last_date": last_row.get("time"),
                    "last_close": last_row.get("close"),
                    "last_vol": last_row.get("volume"),
                    "last_value": last_row.get("value", 0)
                })
        except Exception:
            continue # Skip corrupt files
            
        # Update progress every 50 files to keep UI responsive
        if i % 50 == 0:
            my_bar.progress((i + 1) / total_files, text=f"Loading {sym}...")

    my_bar.empty() # Clear progress bar when done

    # Create the meta DataFrame
    meta = pd.DataFrame(meta_rows)
    if not meta.empty:
        meta["symbol"] = meta["symbol"].astype(str).str.upper()

    return data_map, meta
# ---------- Load data ---------- #
data_map, meta = load_all_data()

# Ensure meta DataFrame has a symbol column
if 'symbol' not in meta.columns:
    meta['symbol'] = pd.Series(meta.index if meta.index.name == 'symbol' else None)
if 'symbol' in meta.columns:
    meta["symbol"] = meta["symbol"].astype(str).str.upper()

total_symbols = len(meta)


# =================== LOAD SECTOR MAP ONE TIME =================== #
def load_sector_mapping(listing_file=LISTING_FILE):
    """Load and normalize sector mapping from either listing file or snapshot."""
    # First try loading from snapshot files
    snapshot_files = [f for f in os.listdir() if f.startswith('vps_snapshot_') and f.endswith('.csv')]
    
    if snapshot_files:
        try:
            latest_snapshot = max(snapshot_files)
            # st.sidebar.write(f"Loading sector data from: {latest_snapshot}")
            snapshot_df = pd.read_csv(latest_snapshot)
            
            # Auto-detect columns
            sym_col = next((c for c in snapshot_df.columns if c.lower() in ["code", "symbol", "ticker", "stockcode", "mã ck"]), None)
            sector_col = next((c for c in snapshot_df.columns if c.lower() in ["industryen", "sector", "icb_industry_name"]), None)
            name_col = next((c for c in snapshot_df.columns if c.lower() in ["shortname", "companyname", "company_name"]), None)
            
            if sym_col and sector_col:
                rename_map = {
                    sym_col: "symbol",
                    sector_col: "sector_vn"
                }
                if name_col:
                    rename_map[name_col] = "company"
                    
                snapshot_df = snapshot_df.rename(columns=rename_map)
                st.sidebar.write("Using columns:", list(snapshot_df.columns))
                
                # Ensure required columns
                if "company" not in snapshot_df.columns:
                    snapshot_df["company"] = ""
                
                # Clean data
                snapshot_df["symbol"] = snapshot_df["symbol"].astype(str).str.strip().str.upper()
                snapshot_df["sector_vn"] = snapshot_df["sector_vn"].fillna("Unknown")
                snapshot_df["company"] = snapshot_df["company"].fillna("")
                
                final_df = snapshot_df[["symbol", "sector_vn", "company"]].drop_duplicates()
                st.sidebar.write(f"Found {len(final_df)} symbols, {final_df['sector_vn'].nunique()} sectors from snapshot")
                return final_df
            
        except Exception as e:
            st.sidebar.warning(f"⚠️ Error reading snapshot: {e}")
    
    # Fallback to listing file
    if os.path.exists(listing_file):
        try:
            # st.sidebar.write("Loading from listing file...")
            if listing_file.lower().endswith('.xlsx'):
                ind = pd.read_excel(listing_file)
            else:
                ind = pd.read_csv(listing_file, encoding='utf-8-sig')
                
            # Auto-detect columns    
            sym_col = next((c for c in ind.columns if c.lower() in ["code", "symbol", "ticker", "stockcode", "mã ck"]), None)
            sector_col = next((c for c in ind.columns if c.lower() in ["industryen", "sector", "icb_industry_name"]), None)
            name_col = next((c for c in ind.columns if c.lower() in ["shortname", "companyname"]), None)
            
            if not sym_col or not sector_col:
                st.sidebar.error("Required columns not found in listing file")
                return pd.DataFrame(columns=["symbol", "sector_vn", "company"])
                
            rename_map = {
                sym_col: "symbol",
                sector_col: "sector_vn"
            }
            if name_col:
                rename_map[name_col] = "company"
                
            ind = ind.rename(columns=rename_map)
            
            # Ensure required columns
            if "company" not in ind.columns:
                ind["company"] = ""
                
            # Clean data    
            ind["symbol"] = ind["symbol"].astype(str).str.strip().str.upper()
            ind["sector_vn"] = ind["sector_vn"].fillna("Unknown")
            ind["company"] = ind["company"].fillna("")
            
            final_df = ind[["symbol", "sector_vn", "company"]].drop_duplicates()
            # st.sidebar.write(f"Found {len(final_df)} symbols, {final_df['sector_vn'].nunique()} sectors from listing")
            return final_df
            
        except Exception as e:
            st.sidebar.error(f"Error reading listing file: {e}")
    
    st.sidebar.warning("No sector mapping sources available")
    return pd.DataFrame(columns=["symbol", "sector_vn", "company"])


# Load sector mapping once for the entire app
sector_df = load_sector_mapping(LISTING_FILE)

# Merge with meta DataFrame
if "symbol" in meta.columns and not sector_df.empty:
    # Ensure symbol column types match
    sector_df["symbol"] = sector_df["symbol"].astype(str).str.upper()
    meta["symbol"] = meta["symbol"].astype(str).str.upper()
    
    # Drop any existing sector/company columns to avoid duplicates
    for col in ["sector_vn", "company"]:
        if col in meta.columns:
            meta = meta.drop(columns=[col])
            
    # Merge and handle missing values
    meta = meta.merge(sector_df[["symbol", "sector_vn", "company"]], 
                     on="symbol", how="left")
    
    # Fill missing values
    meta["sector_vn"] = meta["sector_vn"].fillna("Unknown")
    meta["company"] = meta["company"].fillna("")
    
    # Log statistics
    total_symbols = len(meta)
    mapped_symbols = (meta["sector_vn"] != "Unknown").sum()
    unique_sectors = meta["sector_vn"].nunique()
    
    # st.sidebar.info(f"""
    # 📊 Sector Mapping Stats:
    # - {mapped_symbols}/{total_symbols} symbols mapped 
    # - {unique_sectors} unique sectors
    # """)
else:
    meta["sector_vn"] = "Unknown"
    meta["company"] = ""
    st.sidebar.warning("⚠️ Could not merge sector information")

# ✅ delete everything between here and `all_symbols = ...`
all_symbols = sorted(list(data_map.keys()))
# st.sidebar.success(f"✅ Loaded {len(all_symbols)} symbols from data folder")


# =================== UI Tabs =================== #
tabs = st.tabs([
    "1️⃣ Chart từng mã",
    "2️⃣ Bảng return theo ngành",
    "3️⃣ Big Money",
    "4️⃣ Portfolio Return Management",
    "5️⃣ DCF Valuation"
    
])
# =================== TAB 5️⃣: DCF Valuation =================== #
with tabs[4]:
    st.subheader("📊 DCF Valuation Tool (Discounted Cash Flow)")
    st.write("✅ This DCF tab is visible and working!")
    symbol = st.selectbox("Select stock symbol", sorted(all_symbols))
    market_cap_input = st.number_input("Market Cap (billion ₫, optional)", value=0.0, step=100.0)

    if st.button("🚀 Run Auto DCF"):
        with st.spinner(f"Fetching and computing DCF for {symbol}…"):
            try:
                df = compute_cashflows(symbol)
                scen_df, proj_fcfe, proj_fcff, summary, fcfe_mat, fcff_mat = run_three_scenarios(
                    symbol, df, market_cap=market_cap_input or None
                )

                st.success(f"✅ DCF computed for {symbol}")
                st.markdown("### 📊 Historical Cash Flow Components")
                st.dataframe(df, use_container_width=True)

                st.markdown("### 🧮 DCF Scenario Summary")
                st.dataframe(scen_df, use_container_width=True)

                st.markdown("### 📈 5-Year FCFE and FCFF Projection (Base)")
                fig = go.Figure()
                fig.add_trace(go.Bar(x=proj_fcfe["Year"], y=proj_fcfe["FCF"],
                                     name="FCFE (₫ B)", marker_color="skyblue"))
                fig.add_trace(go.Bar(x=proj_fcff["Year"], y=proj_fcff["FCF"],
                                     name="FCFF (₫ B)", marker_color="lightgreen"))
                fig.update_layout(barmode="group", xaxis_title="Year", yaxis_title="₫ Billion")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 🧾 Auto DCF Inputs & Outputs")
                st.dataframe(summary, use_container_width=True)

                st.markdown("### 🧭 Sensitivity Matrix (Price / Share)")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**FCFE-based**")
                    st.dataframe(fcfe_mat.round(2))
                with col2:
                    st.write("**FCFF-based**")
                    st.dataframe(fcff_mat.round(2))

            except Exception as e:
                st.error(f"❌ Error computing DCF: {e}")

# -------- Tab 1: Chart từng mã -------- #
with tabs[0]:
    st.subheader("📊 Biểu đồ giá")

    # --- Fetch data từ VPS API ---
    def fetch_vps(symbol="AAA", resolution="1D", days=365):
        to_ts = int(dt.datetime.now().timestamp())
        from_ts = int((dt.datetime.now() - dt.timedelta(days=days)).timestamp())
        url = (
            f"https://histdatafeed.vps.com.vn/tradingview/history?"
            f"symbol={symbol}&resolution={resolution}&from={from_ts}&to={to_ts}&countback={days}"
        )
        r = requests.get(url)
        data = r.json()
        if data.get("s") != "ok":
            st.warning(f"⚠️ Lỗi hoặc không có dữ liệu cho {symbol}")
            return pd.DataFrame()
        return pd.DataFrame({
            "time": [dt.datetime.fromtimestamp(x) for x in data["t"]],
            "open": data["o"], "high": data["h"], "low": data["l"],
            "close": data["c"], "volume": data["v"]
        })

    # --- VWAP ---
    def calc_vwap(df):
        tp = (df["high"] + df["low"] + df["close"]) / 3
        cum_pv = (tp * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        return cum_pv / cum_vol

    # --- OBV ---
    def calc_obv(df):
        direction = np.sign(df["close"].diff().fillna(0))
        obv = (direction * df["volume"]).cumsum()
        return obv

    # --- MFI ---
    def calc_mfi(df, period=14):
        tp = (df["high"] + df["low"] + df["close"]) / 3
        rmf = tp * df["volume"]
        pos_mf = np.where(tp > tp.shift(1), rmf, 0)
        neg_mf = np.where(tp < tp.shift(1), rmf, 0)
        pos_sum = pd.Series(pos_mf).rolling(period, min_periods=1).sum()
        neg_sum = pd.Series(neg_mf).replace(0, np.nan).rolling(period, min_periods=1).sum()
        mfr = pos_sum / neg_sum
        mfi = 100 - (100 / (1 + mfr))
        return mfi.clip(0, 100)
    # --- RSI ---
    def calc_rsi(df, period=14):
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.clip(0, 100)
    # --- CVD (Cumulative Volume Delta) ---
    def calc_cvd(df):
        # Approximate volume delta per bar: if close near high → buyers dominant, near low → sellers
        vd = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9) * df["volume"]
        df["cvd"] = vd.fillna(0).cumsum()
        return df


    # --- UI ---
    symbol = st.text_input("Nhập mã cổ phiếu", "AAA").upper()
    chart_type = st.radio("Kiểu hiển thị", ["Candlestick", "Line"], horizontal=True)
    timeframe = st.radio("Khung thời gian", ["1M", "3M", "6M", "1Y"], horizontal=True)

    tf_map = {
        "1M": ("1D", 33),
        "3M": ("1D", 130),
        "6M": ("1D", 190),
        "1Y": ("1D", 380),
    }
    res, days = tf_map[timeframe]
    df = fetch_vps(symbol, res, days)

    if not df.empty:
        df = df[df["volume"] > 0].copy()
        df["date_str"] = df["time"].dt.strftime("%Y-%m-%d")

        # --- Indicators ---
        df["vwap"] = calc_vwap(df)
        df["obv"] = calc_obv(df)
        df["mfi"] = calc_mfi(df)
        df["rsi"] = calc_rsi(df)
        df = calc_cvd(df)
        # --- Moving Averages ---
        df["ma5"] = df["close"].rolling(window=5).mean()
        df["ma20"] = df["close"].rolling(window=20).mean()
        

        # --- Tính Return trong ngày ---
        last_close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2] if len(df) >= 2 else last_close
        day_return = (last_close / prev_close - 1.0) * 100 if prev_close > 0 else 0

        # --- Lấy thông tin ngành & tên công ty ---
        # --- Thông tin cơ bản ---
        company_name, sector_vn = "", ""
        if symbol != "VNINDEX":
            info = meta.loc[meta["symbol"] == symbol]
            if not info.empty:
                company_name = info["company"].iloc[0] if "company" in info.columns else ""
                sector_vn = info["sector_vn"].iloc[0] if "sector_vn" in info.columns else ""

        # --- Hiển thị tóm tắt thông tin ---
        st.markdown(f"""
        ### 🏢 **{symbol}** {"— " + company_name if company_name else ""}
        **Ngành:** {sector_vn if sector_vn else "—"}  
        **Giá hiện tại:** {last_close:,.2f}  
        **Thay đổi trong ngày:** {day_return:+.2f}%
        """)

        st.success(f"✅ Dữ liệu {symbol}: {len(df)} phiên ({timeframe})")

        # --- Subplot ---
        fig = make_subplots(
            rows=6, cols=1, shared_xaxes=True,
            row_heights=[0.45, 0.15, 0.1, 0.1, 0.1, 0.1],
            vertical_spacing=0.05,
            subplot_titles=(
                f"{symbol} Price + VWAP",
                "Volume",
                "OBV (On-Balance Volume)",
                "MFI (Money Flow Index)",
                "RSI (Relative Strength Index)",
                "CVD (Cumulative Volume Delta)"
            )
        )
        # --- Price ---
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=df["date_str"],
                open=df["open"], high=df["high"],
                low=df["low"], close=df["close"],
                name="Price",
                increasing=dict(line=dict(color="#00C176", width=1.2), fillcolor="#00C176"),
                decreasing=dict(line=dict(color="#E74C3C", width=1.2), fillcolor="#E74C3C"),
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=df["date_str"], y=df["close"],
                mode="lines+markers",
                line=dict(color="#007AFF", width=1.8),
                name="Close"
            ), row=1, col=1)

        # --- VWAP ---
        fig.add_trace(go.Scatter(
            x=df["date_str"], y=df["vwap"],
            mode="lines",
            line=dict(color="#ff9800", width=2, dash="dot"),
            name="VWAP"
        ), row=1, col=1)
        # --- MA5 & MA20 ---
        fig.add_trace(go.Scatter(
            x=df["date_str"], y=df["ma5"],
            mode="lines",
            line=dict(color="#8e24aa", width=1.5, dash="dash"),
            name="MA5"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["date_str"], y=df["ma20"],
            mode="lines",
            line=dict(color="#3949ab", width=1.5, dash="dash"),
            name="MA20"
        ), row=1, col=1)

        # --- Volume ---
        fig.add_trace(go.Bar(
            x=df["date_str"], y=df["volume"], name="Volume",
            marker_color="rgba(66,133,244,0.4)",
        ), row=2, col=1)

        # --- OBV ---
        fig.add_trace(go.Scatter(
            x=df["date_str"], y=df["obv"],
            mode="lines",
            line=dict(color="#2962FF", width=1.8),
            name="OBV"
        ), row=3, col=1)

        # --- MFI ---
        fig.add_trace(go.Scatter(
            x=df["date_str"], y=df["mfi"],
            mode="lines",
            line=dict(color="#9C27B0", width=1.8),
            name="MFI"
        ), row=4, col=1)

        # --- RSI ---
        fig.add_trace(go.Scatter(
            x=df["date_str"], y=df["rsi"],
            mode="lines",
            line=dict(color="#7E57C2", width=1.8),
            name="RSI"
        ), row=5, col=1)
        
        # --- CVD ---
        fig.add_trace(go.Scatter(
            x=df["date_str"],
            y=df["cvd"],
            mode="lines",
            line=dict(color="#FF9800", width=1.8),
            name="CVD"
        ), row=6, col=1)

        fig.update_yaxes(title_text="CVD", row=6, col=1, gridcolor="lightgray")

        # RSI guide lines
        fig.add_hline(y=70, line=dict(color="red", dash="dot"), row=5, col=1)
        fig.add_hline(y=30, line=dict(color="green", dash="dot"), row=5, col=1)
        fig.update_yaxes(title_text="RSI", range=[0,100], row=5, col=1, gridcolor="lightgray")


        # --- Layout ---
        fig.update_layout(
            height=1400,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            margin=dict(l=50, r=30, t=50, b=30),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=True,
        )

        # --- Axes ---
        fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor="lightgray")
        fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor="lightgray")
        fig.update_yaxes(title_text="OBV", row=3, col=1, gridcolor="lightgray")
        fig.update_yaxes(title_text="MFI", row=4, col=1, gridcolor="lightgray", range=[0, 100])
        fig.update_xaxes(
            showgrid=False,
            tickangle=0,
            tickmode="array",
            tickvals=df["date_str"][::max(1, len(df)//10)],
            ticktext=df["time"].dt.strftime("%b %d")[::max(1, len(df)//10)],
            rangebreaks=[dict(bounds=["sat", "mon"])]
        )

        st.plotly_chart(fig, use_container_width=True)
            # =============================================
        # --- Order Flow (nếu có dữ liệu intraday) ---
        
    else:
        st.info("⚠️ Không có dữ liệu intraday (5-minute) để hiển thị Order Flow.")

    # ===============================================================
# 📊 Volume Footprint Streamlit App for VPS Data
# ===============================================================

    # ---------------------------------------------------------------
    # 🧠 Function: Fetch VPS 15-min Intraday Data
    # ---------------------------------------------------------------
    @st.cache_data(show_spinner=False)
    def load_intraday(symbol: str, days: int = 30, resolution: str = "15") -> pd.DataFrame:
        url = "https://histdatafeed.vps.com.vn/tradingview/history"
        to_ts = int(dt.datetime.now().timestamp())
        from_ts = int((dt.datetime.now() - dt.timedelta(days=days)).timestamp())
        params = {"symbol": symbol, "resolution": resolution, "from": from_ts, "to": to_ts, "countback": 9870}

        r = requests.get(url, params=params)
        data = r.json()
        if data.get("s") != "ok":
            st.warning(f"No data or error for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame({
            "time": [dt.datetime.fromtimestamp(x) for x in data["t"]],
            "open": data["o"],
            "high": data["h"],
            "low": data["l"],
            "close": data["c"],
            "volume": data["v"]
        })

        # VPS feed ≈ UTC+1 → shift +6h to Vietnam time
        df["time"] = pd.to_datetime(df["time"]) + pd.Timedelta(hours=6)
        df.set_index("time", inplace=True)
        return df

    # ---------------------------------------------------------------
    # 🧭 Streamlit UI
    # ---------------------------------------------------------------

    st.title("📊 Volume Footprint Chart (15-Minute Intraday)")

    col1, col2 = st.columns(2)
    symbol = col1.text_input("Enter Ticker Symbol", "AAA")
    start_date = col2.date_input("Start from date", dt.date(2025, 9, 23))

    df = load_intraday(symbol)

    if df.empty:
        st.stop()

    # ---------------------------------------------------------------
    # 🧹 Filter and prepare data
    # ---------------------------------------------------------------
    df = df[df.index.date >= start_date].copy()
    df["date"] = df.index.date
    vol_max = df["volume"].max()
    df["vol_norm"] = df["volume"] / vol_max

    # daily color logic
    daily_close = df.groupby("date")["close"].last()
    color_map = {}
    for i, d in enumerate(daily_close.index):
        if i == 0:
            color_map[d] = "rgba(128,128,128,0.4)"
        else:
            color_map[d] = (
                "rgba(0,200,0,0.4)" if daily_close.iloc[i] > daily_close.iloc[i - 1]
                else "rgba(255,0,0,0.4)"
            )

    # VWAP per day
    df["vwap_contrib"] = df["close"] * df["volume"]
    vwap = (
        df.groupby("date")[["vwap_contrib", "volume"]]
        .sum()
        .assign(vwap=lambda x: x["vwap_contrib"] / x["volume"])
    )

    # 🧱 Build footprint chart
    # ---------------------------------------------------------------
    fig = go.Figure()
    unique_dates = sorted(df["date"].unique())
    date_positions = {d: i for i, d in enumerate(unique_dates)}

    # draw footprints
    for d in unique_dates:
        day_df = df[df["date"] == d]
        for _, row in day_df.iterrows():
            x0 = date_positions[d] - row["vol_norm"] * 0.4
            x1 = date_positions[d] + row["vol_norm"] * 0.4
            y0, y1 = row["low"], row["high"]
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1, y0=y0, y1=y1,
                line=dict(width=0.2, color="black"),
                fillcolor=color_map[d],
            )

        # daily volume label
        max_vol = day_df["volume"].max()
        day_high = day_df["high"].max()
        fig.add_annotation(
            x=date_positions[d],
            y=day_high + 0.03,
            text=f"{max_vol/1_000_000:.1f}M",
            showarrow=False,
            font=dict(size=10, color="black"),
            yanchor="bottom"
        )

    # VWAP line
    fig.add_trace(go.Scatter(
        x=[date_positions[d] for d in vwap.index],
        y=vwap["vwap"],
        mode="lines+markers",
        line=dict(color="red", width=2),
        name="VWAP"
    ))

    # layout
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(date_positions.values()),
        ticktext=[str(d)[5:] for d in date_positions.keys()],
        title="Trading Date",
    )
    fig.update_yaxes(title="Price (VND)")

    fig.update_layout(
        title=f"{symbol} Volume Footprint (15-Min Stacks per Day, since {start_date})",
        height=700,
        template="plotly_white",
        showlegend=True,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------
    # 🧾 Optional summary
    # ---------------------------------------------------------------
    st.markdown("### 🔍 Summary")
    st.write(f"Total trading days: **{len(unique_dates)}**")
    st.write(f"Maximum 15-minute volume in sample: **{vol_max:,.0f} shares**")
    st.dataframe(df.tail(5))


# -------- Tab 2: Bảng return theo ngành (Fixed for Infinite Values) -------- #
with tabs[1]:
    st.subheader("📊 Return trung bình theo ngành — Value-weighted (1D → 1Y)")

    # 1. Define periods
    periods = {"1D": 1, "5D": 5, "2W": 10, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}

    # 2. Helper to calculate return (Robust against 0 and Infinity)
    def get_ret(series, n):
        if len(series) < n + 1: return np.nan
        
        curr = series.iloc[-1]
        prev = series.iloc[-(n + 1)]
        
        # Prevent division by zero
        if prev <= 0 or pd.isna(prev): 
            return np.nan
            
        ret = (curr / prev - 1.0) * 100.0
        return ret

    # 3. Iterate through data_map
    rows = []
    
    # Check column names from first file
    if len(data_map) > 0:
        sample_df = data_map[list(data_map.keys())[0]]
        col_map = {c.lower(): c for c in sample_df.columns}
        c_close = col_map.get("close")
        c_vol = col_map.get("volume")
    else:
        st.error("❌ Data map is empty.")
        st.stop()

    for sym, df in data_map.items():
        # Basic validation
        if len(df) < 2: continue
        
        # Ensure data is numeric and handle bad data
        try:
            # Force numeric, turning errors to NaN
            df[c_close] = pd.to_numeric(df[c_close], errors='coerce')
            df[c_vol] = pd.to_numeric(df[c_vol], errors='coerce')
            
            last_close = float(df[c_close].iloc[-1])
            last_vol   = float(df[c_vol].iloc[-1])
            
            # Skip if price is 0 or NaN
            if last_close <= 0 or pd.isna(last_close):
                continue
                
            last_val = last_close * last_vol
        except:
            continue
            
        row = {
            "symbol": sym, 
            "Last_Value": last_val
        }
        
        for tag, n in periods.items():
            r = get_ret(df[c_close], n)
            # 🛑 CRITICAL FIX: Replace infinity with NaN
            if np.isinf(r):
                row[f"R_{tag}_%"] = np.nan
            else:
                row[f"R_{tag}_%"] = r
            
        rows.append(row)

    # 4. Create DataFrame
    ret_df = pd.DataFrame(rows)
    
    if ret_df.empty:
        st.warning("⚠️ Không đủ dữ liệu hợp lệ để tính Return.")
        st.stop()

    # 5. Merge Sector Info
    if "sector_vn" in meta.columns:
        ret_df = ret_df.merge(meta[["symbol", "sector_vn"]], on="symbol", how="left")
    else:
        ret_df["sector_vn"] = "Unknown"
        
    ret_df["sector_vn"] = ret_df["sector_vn"].fillna("Unknown")

    # 6. Group By Sector (Value Weighted)
    sector_rows = []
    for tag in periods.keys():
        col = f"R_{tag}_%"
        
        # Drop NaN and Infinite values for this specific period
        tmp = ret_df.dropna(subset=[col, "Last_Value"])
        tmp = tmp[~tmp[col].isin([np.inf, -np.inf])]
        
        if tmp.empty: continue

        grp = tmp.groupby("sector_vn").apply(
            lambda g: pd.Series({
                "R_sector": np.average(g[col], weights=g["Last_Value"]) if g["Last_Value"].sum() > 0 else 0,
                "Val_sector": g["Last_Value"].sum()
            })
        ).reset_index()
        grp["period"] = tag
        sector_rows.append(grp)

    if not sector_rows:
        st.error("Could not calculate sector returns (all data might be filtered out).")
        st.stop()

    sector_df = pd.concat(sector_rows, ignore_index=True)

    # 7. Pivot and Display
    sector_pivot = sector_df.pivot(index="sector_vn", columns="period", values="R_sector").reset_index()

    # Market Row
    market_rows = []
    for tag in periods.keys():
        tmp = sector_df[sector_df["period"] == tag]
        if tmp.empty:
            market_rows.append(np.nan)
        else:
            mkt_ret = np.average(tmp["R_sector"], weights=tmp["Val_sector"])
            market_rows.append(mkt_ret)
            
    market_row = pd.DataFrame([["Toàn thị trường"] + market_rows], columns=["sector_vn"] + list(periods.keys()))
    final_df = pd.concat([sector_pivot, market_row], ignore_index=True)

    # Styling
    numeric_cols = list(periods.keys())
    # Ensure columns exist before selecting
    existing_cols = [c for c in numeric_cols if c in final_df.columns]
    final_df = final_df[["sector_vn"] + existing_cols]

    def color_returns(val):
        if pd.isna(val): return ""
        # Handle residual infinity just in case
        if np.isinf(val): return "color: grey"
        color = "#00C176" if val > 0 else "#FF3B30" if val < 0 else "black"
        return f"color: {color}; font-weight: 600;"

    st.dataframe(
        final_df.style.format({c: "{:+.2f}%" for c in existing_cols})
                      .applymap(color_returns, subset=existing_cols),
        use_container_width=True,
        height=600
    )
    
    
    # ================= DRILL-DOWN SECTION ================= #
    st.markdown("---")
    st.subheader("🔍 Soi chi tiết cổ phiếu trong ngành (Drill-Down)")

    # 1. Select Inputs
    c1, c2 = st.columns(2)
    with c1:
        target_sector = st.selectbox("Chọn ngành cần soi:", final_df["sector_vn"].unique())
    with c2:
        target_period = st.selectbox("Chọn kỳ hạn:", list(periods.keys()), index=6) # Default to 1Y

    # 2. Get Data for that Sector & Period
    col_ret = f"R_{target_period}_%"
    
    # Filter the original 'ret_df' we built earlier
    drill_df = ret_df[
        (ret_df["sector_vn"] == target_sector) & 
        (ret_df["Last_Value"] > 0)
    ].copy()

    # 3. Check if data exists
    if drill_df.empty:
        st.warning(f"Không có dữ liệu cho ngành {target_sector}")
    else:
        # Calculate Weight of each stock in the sector
        sector_total_val = drill_df["Last_Value"].sum()
        drill_df["Weight"] = (drill_df["Last_Value"] / sector_total_val) * 100
        
        # Calculate Contribution (Return * Weight)
        drill_df["Contribution"] = drill_df[col_ret] * (drill_df["Weight"]/100)
        
        # Sort by Contribution (to see who is driving the sector return)
        drill_df = drill_df.sort_values("Contribution", ascending=False)

        # 4. Display Stats
        avg_ret = np.average(drill_df[col_ret].fillna(0), weights=drill_df["Last_Value"])
        
        st.markdown(f"""
        ### 🏭 Ngành: {target_sector} ({target_period})
        **Return trung bình (Weighted):** :blue[{avg_ret:+.2f}%]  
        **Tổng giá trị giao dịch (proxy):** {sector_total_val/1e9:,.0f} tỷ
        """)

        # 5. Format and Show Table
        # Columns to show
        show_cols = ["symbol", col_ret, "Last_Value", "Weight", "Contribution"]
        
        st.dataframe(
            drill_df[show_cols].style
            .format({
                col_ret: "{:+.2f}%",
                "Last_Value": "{:,.0f}",
                "Weight": "{:.2f}%",
                "Contribution": "{:+.2f}%"
            })
            .background_gradient(subset=["Contribution"], cmap="RdYlGn", vmin=-5, vmax=5)
            .bar(subset=[col_ret], color=['#FF3B30', '#00C176'], align='zero'),
            use_container_width=True,
            height=400
        )
        
        st.info("""
        💡 **Cách đọc bảng này:**
        - **Return:** Tỷ suất lợi nhuận của riêng cổ phiếu đó.
        - **Weight:** Tỷ trọng của cổ phiếu trong ngành (dựa trên GTGD).
        - **Contribution:** Mức đóng góp vào return chung của ngành (= Return x Weight).
        
        *Ví dụ: Nếu ngành tăng 90%, hãy tìm cổ phiếu có Contribution cao nhất (màu xanh đậm). Nếu thấy một mã penny tăng 500% nhưng Weight chỉ 0.1%, nó không ảnh hưởng nhiều. Nhưng nếu mã trụ tăng 100% với Weight 50%, nó là nguyên nhân chính.*
        """)
    
    # -----------------------------------------------------------
    # 8. Heatmap (Final: Soft Colors & Eye-Ease Palette)
    # -----------------------------------------------------------
    st.markdown("---")
    st.subheader("🧩 Heatmap ngành (Market Structure)")
    
    # 1. Inputs: Period & Sensitivity
    c1, c2 = st.columns([1, 2])
    with c1:
        heat_period = st.selectbox("Chọn kỳ xem:", list(periods.keys()), index=0)
    
    # Default limits based on period
    default_limits = {
        "1D": 7, "5D": 15, "2W": 20, "1M": 30, 
        "3M": 50, "6M": 80, "1Y": 100
    }
    def_lim = default_limits.get(heat_period, 7)

    with c2:
        color_limit = st.slider(
            "Độ nhạy màu sắc (+/- %):", 
            min_value=1, max_value=200, value=def_lim, step=1,
            help="Ví dụ chọn 7%: >6.3% là tím, < -6.3% là lơ. Màu sắc sẽ dịu (pastel) để không hại mắt."
        )

    col_name = f"R_{heat_period}_%"
    
    if col_name in ret_df.columns:
        # 2. Prepare Data
        heat_df = ret_df[["symbol", "sector_vn", "Last_Value", col_name]].copy()
        heat_df = heat_df.rename(columns={"Last_Value": "value", col_name: "ret"})
        
        # Filter Indices & Invalid Data
        exclude_indices = ["VNINDEX", "VN30", "HNX", "HNX30", "UPCOM", "VNXALL", "VN100"]
        heat_df = heat_df[~heat_df["symbol"].isin(exclude_indices)]
        heat_df = heat_df[(heat_df["value"] > 0) & (heat_df["ret"].notna())]
        heat_df = heat_df[~heat_df["ret"].isin([np.inf, -np.inf])]

        if heat_df.empty:
            st.warning("⚠️ Không có dữ liệu để vẽ Heatmap.")
        else:
            # 3. Define "Soft & Matte" Color Scale (Eye-Ease)
            # We use a wider Yellow band (45% to 55%) so small moves don't paint the screen red/green
            
            # Palette: Material Design Colors (Softer, less neon)
            # Floor:  Light Blue (#29B6F6) instead of Neon Cyan
            # Down:   Soft Red (#EF5350) instead of Neon Red
            # Ref:    Pale Yellow (#FFEE58) instead of Neon Yellow
            # Up:     Soft Green (#66BB6A) instead of Neon Green
            # Ceil:   Soft Purple (#AB47BC) instead of Neon Purple
            
            vn_soft_scale = [
                (0.00, '#29B6F6'), (0.10, '#29B6F6'),  # Floor (Soft Blue)
                (0.10, '#EF5350'), (0.45, '#EF5350'),  # Down (Soft Red)
                (0.45, '#FFEE58'), (0.55, '#FFEE58'),  # Ref (Soft Yellow) - Wider band
                (0.55, '#66BB6A'), (0.90, '#66BB6A'),  # Up (Soft Green)
                (0.90, '#AB47BC'), (1.00, '#AB47BC')   # Ceil (Soft Purple)
            ]

            # 4. Draw Treemap
            fig = px.treemap(
                heat_df,
                path=[px.Constant("Toàn thị trường"), "sector_vn", "symbol"],
                values="value",
                color="ret",
                color_continuous_scale=vn_soft_scale,
                range_color=[-color_limit, color_limit],
                hover_data={"ret": ":.2f", "value": ":,.0f"}
            )
            
            fig.update_layout(
                margin=dict(t=0, l=0, r=0, b=0), 
                height=550,
            )
            
            # Custom Hover
            fig.update_traces(
                hovertemplate="<b>%{label}</b><br>Ngành: %{parent}<br>Return: %{color:.2f}%<br>GTGD: %{value:,.0f}",
                textinfo="label+text+value",
                marker=dict(line=dict(width=1, color='white')) # White borders to separate boxes cleanly
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Legend
            st.caption(
                f"🎨 **Màu sắc (Dịu mắt):** "
                f":blue-background[Sàn (<-{color_limit*0.95:.1f}%)] "
                f":red[Giảm] "
                f":orange[Tham chiếu / Biến động nhẹ] "
                f":green[Tăng] "
                f":violet-background[Trần (>{color_limit*0.95:.1f}%)]"
            )
    else:
        st.warning(f"Chưa có dữ liệu cho kỳ {heat_period}")
    # --- 🧩 Bảng thống kê ngành ---
    sector_table = (
        heat_df.groupby("sector_vn")
        .apply(lambda g: pd.Series({
            "Return_%": np.average(g["ret"], weights=g["value"]) if g["value"].sum() > 0 else np.nan,
            "Value_sum": g["value"].sum()
        }))
        .reset_index()
    )

    # Thêm tỷ trọng %
    total_value = sector_table["Value_sum"].sum()
    sector_table["Weight_%"] = (sector_table["Value_sum"] / total_value) * 100

    # Thêm hàng Toàn thị trường
    market_ret = np.average(sector_table["Return_%"], weights=sector_table["Value_sum"])
    market_row = pd.DataFrame([{
        "sector_vn": "Toàn thị trường",
        "Return_%": market_ret,
        "Value_sum": total_value,
        "Weight_%": 100.0
    }])
    sector_table = pd.concat([sector_table, market_row], ignore_index=True)

    # Chuẩn hóa hiển thị
    sector_table = sector_table.sort_values("Weight_%", ascending=False)
    sector_table["Value_sum"] = sector_table["Value_sum"] / 1e6  # tỷ đồng
    st.dataframe(
        sector_table.style.format({
            "Return_%": "{:+.2f}%",
            "Value_sum": "{:,.0f} tỷ",
            "Weight_%": "{:.2f}%"
        }),
        use_container_width=True
    )
    st.caption("""
    📘 **Giải thích kết quả:**

    - **Return toàn thị trường (Value-weighted)** trong bảng trên **không phải là chỉ số VN-Index**.  
    Giá trị này được tính theo **trọng số giá trị giao dịch (giá × khối lượng)** của từng cổ phiếu,  
    nên phản ánh **dòng tiền thực tế đang giao dịch trên thị trường**, chứ không phải quy mô vốn hóa.

    - Trong khi đó, **VN-Index** được tính theo **vốn hóa điều chỉnh free-float** –  
    các mã vốn hóa lớn (VIC, VHM, VCB, GAS, v.v.) có ảnh hưởng rất mạnh đến chỉ số này,  
    kể cả khi khối lượng giao dịch của chúng thấp.

    - Vì vậy, hai thước đo có thể khác hướng:
    - Nếu **Return toàn thị trường dương nhưng VN-Index âm**, điều đó cho thấy  
        dòng tiền đang chảy mạnh vào **mid-cap và small-cap**, trong khi nhóm **large-cap** suy yếu.  
    - Ngược lại, nếu **VN-Index tăng nhưng Return toàn thị trường giảm**,  
        có thể thị trường bị kéo bởi **một vài mã vốn hóa lớn**,  
        dù phần lớn cổ phiếu và thanh khoản chung vẫn yếu.

    💡 **Tóm lại:**  
    Sự khác biệt giữa hai con số này giúp nhận biết **cấu trúc dòng tiền** trong ngày —  
    xem thị trường tăng/giảm là do **sức kéo của blue-chip** hay do **dòng tiền lan tỏa trên diện rộng**.
    """)
    
    # -----------------------------------------------------------
    # 9. Market Liquidity Analysis (Fixed: Indices Excluded + Whale Debugger)
    # -----------------------------------------------------------
    st.markdown("---")
    st.subheader("🌊 Thanh khoản & Dòng tiền thị trường (Market Liquidity)")

    # 1. Aggregate Data (Exclude Indices)
    date_map = {}
    
    # List of indices to exclude from calculation
    exclude_indices = ["VNINDEX", "VN30", "HNX", "HNX30", "UPCOM", "VNXALL", "VN100", "VNMID", "VNSML"]
    
    # Debug: Track top contributors for the "Spike Date" (Jan 2026) to find errors
    spike_tracker = [] 
    
    for sym, df in data_map.items():
        # 🛑 SKIP INDICES
        if sym in exclude_indices:
            continue
            
        # Ensure we have value info
        if "value" not in df.columns:
            try:
                # Handle 0 or NaN prices
                vals = df["close"] * df["volume"]
            except:
                continue
        else:
            vals = df["value"]
            
        if "time" in df.columns:
            dates = df["time"]
        elif "date" in df.columns:
            dates = df["date"]
        else:
            continue
            
        # Add to aggregate
        for d, v in zip(dates, vals):
            if pd.isna(v) or v == 0: continue
            ts = pd.Timestamp(d)
            date_map[ts] = date_map.get(ts, 0) + v
            
            # 🕵️ WHALE DETECTOR: Capture large values in Jan 2026
            if ts.year == 2026 and ts.month == 1:
                # If a single stock has > 500 billion VND in one day, it's suspicious
                if v > 500_000_000_000: 
                    spike_tracker.append({"date": ts.date(), "symbol": sym, "value_bil": v/1e9})

    # Convert to DataFrame
    liq_df = pd.DataFrame(list(date_map.items()), columns=["date", "total_value"])
    liq_df = liq_df.sort_values("date").reset_index(drop=True)
    
    # Filter: Last 2 Years
    start_date = pd.Timestamp.now() - pd.Timedelta(days=730)
    liq_df = liq_df[liq_df["date"] >= start_date]

    if liq_df.empty:
        st.warning("Chưa đủ dữ liệu để vẽ biểu đồ thanh khoản.")
    else:
        # 2. Add Moving Average
        liq_df["MA20"] = liq_df["total_value"].rolling(20).mean()
        
        # Scale to Trillion VND (Tỷ VND)
        # Note: If your CSV prices are in 1000 VND (e.g. 20.0), you might need to multiply by 1000 here.
        # Check the chart: if normal days are ~20 Tỷ (too low), multiply by 1000. 
        # If they are ~20,000 Tỷ, it's correct.
        liq_df["val_bil"] = liq_df["total_value"] / 1_000_000_000
        liq_df["ma20_bil"] = liq_df["MA20"] / 1_000_000_000

        # 3. Chart: Daily Liquidity
        st.markdown("##### 1. Diễn biến thanh khoản hàng ngày")
        
        liq_df["color"] = np.where(liq_df["total_value"] > liq_df["MA20"], "#00C176", "#EF5350")
        
        fig_liq = go.Figure()
        fig_liq.add_trace(go.Bar(
            x=liq_df["date"], y=liq_df["val_bil"],
            marker_color=liq_df["color"], name="GTGD (Tỷ)", opacity=0.8
        ))
        fig_liq.add_trace(go.Scatter(
            x=liq_df["date"], y=liq_df["ma20_bil"],
            mode="lines", line=dict(color="#2962FF", width=2), name="TB 20 phiên"
        ))
        fig_liq.update_layout(height=400, margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig_liq, use_container_width=True)

        # 4. Debug: Show Spike Culprits
        if spike_tracker:
            with st.expander("🕵️ Whale Detector: Các mã gây Spike tháng 1/2026"):
                spike_df = pd.DataFrame(spike_tracker).sort_values("value_bil", ascending=False).head(10)
                st.dataframe(spike_df)
                st.caption("Nếu thấy mã lạ hoặc Index (VNINDEX) ở đây, hãy thêm vào danh sách `exclude_indices` trong code.")

        # 5. Chart: Monthly Trend
        st.markdown("##### 2. Xu hướng dòng tiền theo Tháng")
        liq_df["month"] = liq_df["date"].dt.to_period("M")
        monthly_stats = liq_df.groupby("month").agg(
            total_val=("total_value", "sum"),
            days=("date", "count")
        ).reset_index()
        
        monthly_stats["avg_daily_val"] = (monthly_stats["total_val"] / monthly_stats["days"]) / 1_000_000_000
        monthly_stats["month_str"] = monthly_stats["month"].astype(str)
        monthly_stats["change"] = monthly_stats["avg_daily_val"].pct_change() * 100
        
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(
            x=monthly_stats["month_str"], y=monthly_stats["avg_daily_val"],
            marker_color="#29B6F6", name="TB/Phiên"
        ))
        fig_m.add_trace(go.Scatter(
            x=monthly_stats["month_str"], 
            y=monthly_stats["avg_daily_val"] * 1.1,
            text=monthly_stats["change"].apply(lambda x: f"{x:+.1f}%" if pd.notnull(x) else ""),
            mode="text", textfont=dict(color="black", size=10)
        ))
        fig_m.update_layout(height=350, margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig_m, use_container_width=True)

# -------- Tab 3: Giao dịch Thỏa Thuận – HOSE -------- #
with tabs[2]:
    st.subheader("🤝 Giao dịch Thỏa Thuận – HOSE (Put-through Transactions)")

    PUT_DIR = os.path.join(DATA_DIR, "Putthrough")
    MASTER_FILE = os.path.join(PUT_DIR, "putthrough_hose_all.csv")

    if not os.path.exists(MASTER_FILE):
        st.warning("⚠️ Chưa có dữ liệu. Hãy chạy update1_vps_hose.py trước.")
    else:
        # --- Load once ---
        df = pd.read_csv(MASTER_FILE)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["cum_value"] = pd.to_numeric(df.get("cum_value"), errors="coerce")
        df["cum_volume"] = pd.to_numeric(df.get("cum_volume"), errors="coerce")

        latest_date = df["date"].max()
        today_df = df[df["date"] == latest_date].copy()
        st.markdown(f"### 📅 Giao dịch ngày {latest_date.date()} (HOSE)")

        # =====================================================
        # 1️⃣ DETAILED TABLE — like VNDIRECT board
        # =====================================================

        # --- Helper: last close from /Data/xxx.csv --- #
        def get_last_close(symbol):
            file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
            if not os.path.exists(file_path):
                return None
            try:
                sub = pd.read_csv(file_path)
                if "time" in sub.columns:
                    sub["time"] = pd.to_datetime(sub["time"], errors="coerce")
                    sub = sub.sort_values("time")
                if "close" in sub.columns:
                    return sub["close"].dropna().iloc[-1]
            except Exception:
                return None
            return None

        # --- Fetch last close for each symbol --- #
        unique_syms = today_df["symbol"].unique()
        close_map = {sym: get_last_close(sym) for sym in unique_syms}
        today_df["last_close"] = today_df["symbol"].map(close_map)

        # --- Color rule --- #
        def price_color(row):
            if pd.isna(row["last_close"]):
                return "black"
            if row["price"] > row["last_close"]:
                return "red"
            elif row["price"] < row["last_close"]:
                return "limegreen"
            else:
                return "violet"

        today_df["color"] = today_df.apply(price_color, axis=1)

        # =====================================================
        # 🧾 TABLES SIDE BY SIDE
        # =====================================================
        # Create two columns
        col1, col2 = st.columns(2)

        # ---------- LEFT COLUMN = DETAILED TABLE ----------
        with col1:
            st.markdown("#### 🧾 Danh sách giao dịch (Chi tiết)")
            cols = ["symbol", "price", "volume", "value", "cum_value", "time", "last_close"]
            display_df = today_df[cols].sort_values("time", ascending=False)

            styled_left = (
                display_df
                .style
                .format({
                    "price": "{:,.2f}",
                    "volume": "{:,.0f}",
                    "value": "{:,.0f}",
                    "cum_value": "{:,.0f}",
                    "last_close": "{:,.2f}"
                })
                .apply(lambda s: [
                    f"color: {c}; font-weight:600;" for c in today_df.loc[s.index, "color"]
                ], subset=["price"])
            )

            st.dataframe(styled_left, use_container_width=True, height=700)


        # ---------- RIGHT COLUMN = SUMMARY TABLE ----------
        with col2:
            st.markdown("#### 📊 Tổng hợp theo mã cổ phiếu")
            summary = (
                today_df.groupby("symbol", as_index=False)
                .agg(
                    last_price=("price", "last"),
                    total_volume=("volume", "sum"),
                    total_value=("value", "sum"),
                    cum_volume=("cum_volume", "max"),
                    cum_value=("cum_value", "max"),
                    last_time=("time", "last")
                )
                .sort_values("total_value", ascending=False)
            )

            styled_right = (
                summary
                .style
                .format({
                    "last_price": "{:,.2f}",
                    "total_volume": "{:,.0f}",
                    "total_value": "{:,.0f}",
                    "cum_volume": "{:,.0f}",
                    "cum_value": "{:,.0f}",
                })
            )

            st.dataframe(styled_right, use_container_width=True, height=700)

        # =====================================================
        # 🔎 Top 15 GD Thỏa Thuận – Multi-day (1D, 3D, 5D) + Price Coloring
        # =====================================================

        st.markdown("### ⏱ Top 15 Giao dịch Thỏa Thuận trong 1 – 3 – 5 ngày gần nhất")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        latest_date = df["date"].max()

        # --- Define horizons --- #
        horizons = {"1D": 1, "3D": 3, "5D": 5}
        cutoffs = {k: latest_date - pd.Timedelta(days=v - 1) for k, v in horizons.items()}

        recent_df = df[df["date"] >= cutoffs["5D"]].copy()

        # --- Compute weighted average price --- #
        recent_df["w_price"] = recent_df["price"] * recent_df["value"]
        avg_price = (
            recent_df.groupby("symbol", as_index=False)
            .agg(
                avg_price=("price", lambda x: (recent_df.loc[x.index, "w_price"].sum() /
                                            recent_df.loc[x.index, "value"].sum()))
            )
        )

        # --- Load current price from /Data folder --- #
        def get_last_close(symbol):
            file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
            if not os.path.exists(file_path):
                return None
            try:
                sub = pd.read_csv(file_path)
                if "time" in sub.columns:
                    sub["time"] = pd.to_datetime(sub["time"], errors="coerce")
                    sub = sub.sort_values("time")
                if "close" in sub.columns:
                    return sub["close"].dropna().iloc[-1]
            except Exception:
                return None
            return None

        price_map = {sym: get_last_close(sym) for sym in recent_df["symbol"].unique()}
        avg_price["current_price"] = avg_price["symbol"].map(price_map)

        # --- Determine color (red / green / yellow) --- #
        def color_flag(row):
            if pd.isna(row["current_price"]) or pd.isna(row["avg_price"]):
                return "gray"
            if row["avg_price"] < row["current_price"] * 0.98:
                return "limegreen"   # lower than market
            elif row["avg_price"] > row["current_price"] * 1.02:
                return "red"         # higher than market
            else:
                return "gold"        # near market

        avg_price["color"] = avg_price.apply(color_flag, axis=1)

        # --- Merge avg price & color back --- #
        agg = (
            recent_df.groupby(["symbol", "date"], as_index=False)
            .agg(total_value=("value", "sum"))
            .merge(avg_price, on="symbol", how="left")
        )

        # --- Top 15 symbols --- #
        top_syms = (
            agg.groupby("symbol", as_index=False)["total_value"]
            .sum()
            .nlargest(15, "total_value")["symbol"]
        )
        agg_top = agg[agg["symbol"].isin(top_syms)].copy()

        # --- Label with avg price --- #
        agg_top["label"] = agg_top.apply(
            lambda r: f"{r['symbol']} ({r['avg_price']:.1f})" if pd.notna(r["avg_price"]) else r["symbol"],
            axis=1
        )

        # --- Chart --- #
        chart = (
            alt.Chart(agg_top)
            .mark_bar()
            .encode(
                x=alt.X("sum(total_value):Q", title="Giá trị GD (VND)", axis=alt.Axis(format="s")),
                y=alt.Y("label:N", sort="-x", title="Mã CK (Giá TB)"),
                color=alt.Color("date:T", title="Ngày GD", scale=alt.Scale(scheme="greens")),
                tooltip=[
                    alt.Tooltip("symbol:N", title="Mã CK"),
                    alt.Tooltip("date:T", title="Ngày"),
                    alt.Tooltip("sum(total_value):Q", format=",", title="Giá trị GD (VND)"),
                    alt.Tooltip("avg_price:Q", format=".2f", title="Giá TB Thỏa Thuận"),
                    alt.Tooltip("current_price:Q", format=".2f", title="Giá Hiện Tại"),
                ],
            )
            .properties(height=500)
        )

        st.altair_chart(chart, use_container_width=True)

        # =====================================================
        # 3️⃣ GIAO DỊCH TỰ DOANH
        # =====================================================
        st.markdown("### 🏦 Giao dịch Tự Doanh (Proprietary Trading Snapshot)")

        TD_DIR = os.path.join(DATA_DIR, "TuDoanh")
        TD_MASTER = os.path.join(TD_DIR, "tudoanh_all.csv")

        if not os.path.exists(TD_MASTER):
            st.info("⚠️ Chưa có dữ liệu Tự Doanh. Hãy chạy update2_tudoanh.py trước.")
        else:
            td = pd.read_csv(TD_MASTER)
            td["date"] = pd.to_datetime(td["date"], errors="coerce")
            latest_date = td["date"].max()
            today_td = td[td["date"] == latest_date].copy()
            st.markdown(f"#### 📅 Dữ liệu ngày {latest_date.date()}")

            # Convert numeric columns
            for c in ["buy_volume","sell_volume","buy_value","sell_value","net_volume","net_value"]:
                today_td[c] = pd.to_numeric(today_td[c], errors="coerce")

            # Color rule for net columns
            def net_color(val):
                if pd.isna(val):
                    return "black"
                if val > 0:
                    return "limegreen"
                elif val < 0:
                    return "red"
                else:
                    return "gray"

            # Apply styling
            styled_td = (
                today_td
                .sort_values("net_value", ascending=False)
                .style
                .format({
                    "buy_volume": "{:,.0f}",
                    "buy_value": "{:,.0f}",
                    "sell_volume": "{:,.0f}",
                    "sell_value": "{:,.0f}",
                    "net_volume": "{:,.0f}",
                    "net_value": "{:,.0f}",
                })
                .applymap(net_color, subset=["net_volume", "net_value"])
                .set_properties(subset=["net_volume", "net_value"], **{"font-weight": "600"})
            )

            # Choose columns first
            # 1) Pick columns & RENAME on the DataFrame (not on Styler)
            cols = [
                "symbol",
                "buy_volume",
                "buy_value",
                "sell_volume",
                "sell_value",
                "net_volume",
                "net_value",
            ]
            display_df = (
                today_td[cols]
                .copy()
                .rename(columns={
                    "symbol": "Mã CK",
                    "buy_volume": "Khối lượng mua",
                    "buy_value": "Giá trị mua",
                    "sell_volume": "Khối lượng bán",
                    "sell_value": "Giá trị bán",
                    "net_volume": "Tổng KL (net)",
                    "net_value": "Tổng giá trị (net)",
                })
                .sort_values("Tổng giá trị (net)", ascending=False)
            )

            # 2) Styler helpers
            def net_color(val):
                if pd.isna(val):
                    return "color: gray"
                if val > 0:
                    return "color: limegreen"
                if val < 0:
                    return "color: red"
                return "color: gray"

            # 3) Style AFTER rename
            styled_td = (
                display_df
                .style
                .format({
                    "Khối lượng mua": "{:,.0f}",
                    "Giá trị mua": "{:,.0f}",
                    "Khối lượng bán": "{:,.0f}",
                    "Giá trị bán": "{:,.0f}",
                    "Tổng KL (net)": "{:,.0f}",
                    "Tổng giá trị (net)": "{:,.0f}",
                })
                # applymap must return CSS strings:
                .applymap(net_color, subset=["Tổng KL (net)", "Tổng giá trị (net)"])
                .set_properties(subset=["Tổng KL (net)", "Tổng giá trị (net)"], **{"font-weight": "600"})
            )

            # 4) Display
            st.dataframe(styled_td, use_container_width=True, height=600)

            # =====================================================
        # 🔝 Top 15 Mua / Bán ròng của Tự Doanh (Fixed: Logic 5 Unique Days)
        # =====================================================
        st.markdown("### ⏱ Top 15 Mua / Bán ròng của Tự Doanh trong 1 – 3 – 5 ngày gần nhất")

        # 1. Load Data
        td = pd.read_csv(TD_MASTER)

        # 2. Robust Date Parsing
        # Try parsing with dayfirst=True (Vietnamese format usually DD/MM/YYYY)
        td["date"] = pd.to_datetime(td["date"], dayfirst=True, errors="coerce")
        # Remove any rows with bad dates
        td = td.dropna(subset=["date"])

        # 3. Force Numeric
        for c in ["buy_value", "sell_value", "net_value"]:
            if c in td.columns:
                td[c] = (
                    td[c].astype(str)
                    .str.replace(",", "", regex=False)
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0)
                )

        # 4. Fix missing Net Value
        # Some source files might miss 'net_value', so we recalculate to be safe
        if "net_value" not in td.columns or td["net_value"].sum() == 0:
            td["net_value"] = td["buy_value"] - td["sell_value"]

        # 5. 🛑 CRITICAL FIX: Get Last 5 UNIQUE Trading Days (Not just calendar days)
        # This fixes the "single color" bug by ignoring weekends/holidays gaps
        available_dates = sorted(td["date"].unique())
        
        if len(available_dates) == 0:
            st.warning("⚠️ Không tìm thấy dữ liệu ngày tháng trong file Tự Doanh.")
        else:
            # Take the last 5 available dates
            target_dates = available_dates[-5:]
            
            # Filter data for only these dates
            recent_td = td[td["date"].isin(target_dates)].copy()

            # 6. Aggregate Data
            agg_td = (
                recent_td.groupby(["symbol", "date"], as_index=False)
                .agg(total_net=("net_value", "sum"))
            )
            agg_td["abs_net"] = agg_td["total_net"].abs()

            # 7. Identify Top 15 Buy/Sell (based on sum over the period)
            rank_df = agg_td.groupby("symbol", as_index=False)["total_net"].sum()
            
            top_buy_syms = rank_df.nlargest(15, "total_net")["symbol"]
            top_sell_syms = rank_df.nsmallest(15, "total_net")["symbol"]

            # Filter for plotting
            buy_data = agg_td[agg_td["symbol"].isin(top_buy_syms) & (agg_td["total_net"] > 0)].copy()
            sell_data = agg_td[agg_td["symbol"].isin(top_sell_syms) & (agg_td["total_net"] < 0)].copy()

            # 8. Draw Charts
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### 🟢 Top 15 Mua ròng (5 phiên gần nhất)")
                if buy_data.empty:
                    st.info("⚠️ Không có dữ liệu Mua ròng.")
                else:
                    buy_chart = (
                        alt.Chart(buy_data)
                        .mark_bar()
                        .encode(
                            x=alt.X("sum(total_net):Q", title="Giá trị Mua ròng (VND)", axis=alt.Axis(format="~s")),
                            y=alt.Y("symbol:N", sort="-x", title="Mã CK"),
                            # Use Date for color stacking
                            color=alt.Color("date:T", title="Ngày GD", scale=alt.Scale(scheme="greens")),
                            tooltip=[
                                alt.Tooltip("symbol:N", title="Mã CK"),
                                alt.Tooltip("date:T", title="Ngày", format="%d-%m-%Y"),
                                alt.Tooltip("sum(total_net):Q", format=",.0f", title="Giá trị Mua ròng")
                            ],
                        )
                        .properties(height=400)
                    )
                    st.altair_chart(buy_chart, use_container_width=True)

            with col2:
                st.markdown("##### 🔴 Top 15 Bán ròng (5 phiên gần nhất)")
                if sell_data.empty:
                    st.info("⚠️ Không có dữ liệu Bán ròng.")
                else:
                    # Use absolute value for visualization bars
                    sell_data["abs_val"] = sell_data["total_net"].abs()
                    
                    sell_chart = (
                        alt.Chart(sell_data)
                        .mark_bar()
                        .encode(
                            x=alt.X("sum(abs_val):Q", title="Giá trị Bán ròng (VND)", axis=alt.Axis(format="~s")),
                            y=alt.Y("symbol:N", sort="-x", title="Mã CK"),
                            color=alt.Color("date:T", title="Ngày GD", scale=alt.Scale(scheme="reds")),
                            tooltip=[
                                alt.Tooltip("symbol:N", title="Mã CK"),
                                alt.Tooltip("date:T", title="Ngày", format="%d-%m-%Y"),
                                alt.Tooltip("sum(total_net):Q", format=",.0f", title="Giá trị Bán ròng")
                            ],
                        )
                        .properties(height=400)
                    )
                    st.altair_chart(sell_chart, use_container_width=True)

            # Debug Info (Optional - delete later if not needed)
            with st.expander("🔍 Kiểm tra dữ liệu ngày (Debug)"):
                st.write(f"Tìm thấy {len(available_dates)} ngày dữ liệu trong file.")
                st.write(f"Đang hiển thị 5 ngày: {[d.strftime('%Y-%m-%d') for d in target_dates]}")

    # =====================================================
    # 🔝 Top 15 Mua / Bán ròng của NĐT Nước Ngoài (Fixed)
    # =====================================================
    st.markdown("### 🌏 Top 15 Mua / Bán ròng của NĐT Nước Ngoài (1–3–5 ngày gần nhất)")

    # 1. Aggregate Foreign Data from ALL Symbols
    all_foreign = []
    
    # Progress bar for UX
    # prog = st.progress(0, text="Analyzing foreign flows...")
    
    for i, (sym, df) in enumerate(data_map.items()):
        # Skip indices
        if sym in ["VNINDEX", "VN30", "HNX", "HNX30", "UPCOM"]: continue

        # Check if foreign columns exist
        if not {"foreign_buy_val", "foreign_sell_val"}.issubset(df.columns):
            continue
            
        # Use the global clean_daily function (or the local logic below)
        # Fast local clean if function is missing
        if "clean_daily" in globals():
            df_clean = clean_daily(df)
        else:
            df_clean = df.copy()
            for c in ["foreign_buy_val", "foreign_sell_val"]:
                df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce").fillna(0)
            if "time" in df_clean.columns:
                df_clean["date"] = pd.to_datetime(df_clean["time"]).dt.date

        if df_clean.empty: continue
        
        # Calculate Net Value
        df_clean["foreign_net_val"] = df_clean["foreign_buy_val"] - df_clean["foreign_sell_val"]
        
        # Keep only relevant columns to save memory
        sub = df_clean[["date", "foreign_net_val"]].copy()
        sub["symbol"] = sym
        all_foreign.append(sub)
        
        # if i % 100 == 0: prog.progress(i / len(data_map))
        
    # prog.empty()

    if not all_foreign:
        st.warning("⚠️ Không có dữ liệu giao dịch nước ngoài (thiếu cột 'foreign_buy_val' trong CSV).")
        st.stop()

    # 2. Combine and Filter Dates
    foreign_all = pd.concat(all_foreign, ignore_index=True)
    foreign_all["date"] = pd.to_datetime(foreign_all["date"])
    
    # 🛑 Filter Future Data (Using Today's Date)
    today = pd.Timestamp.now().normalize()
    foreign_all = foreign_all[foreign_all["date"] <= today]
    
    # Get last 5 trading days
    available_dates = sorted(foreign_all["date"].unique())
    if len(available_dates) < 5:
        st.warning("Chưa đủ 5 ngày dữ liệu lịch sử.")
        recent_days = available_dates
    else:
        recent_days = available_dates[-5:]
        
    start_cut = recent_days[0]
    recent_df = foreign_all[foreign_all["date"] >= start_cut].copy()

    # 3. Aggregate for Top 15
    agg_foreign = (
        recent_df.groupby(["symbol", "date"], as_index=False)
        .agg(total_net=("foreign_net_val", "sum"))
    )
    
    # Calculate Total Net over the period to rank them
    rank_df = agg_foreign.groupby("symbol")["total_net"].sum().reset_index()
    
    top_buy_syms = rank_df.nlargest(15, "total_net")["symbol"]
    top_sell_syms = rank_df.nsmallest(15, "total_net")["symbol"]

    # Filter data for charts
    buy_data = agg_foreign[agg_foreign["symbol"].isin(top_buy_syms) & (agg_foreign["total_net"] > 0)].copy()
    sell_data = agg_foreign[agg_foreign["symbol"].isin(top_sell_syms) & (agg_foreign["total_net"] < 0)].copy()

    # Merge company names if available
    if "company" in meta.columns:
        buy_data = buy_data.merge(meta[["symbol", "company"]], on="symbol", how="left")
        sell_data = sell_data.merge(meta[["symbol", "company"]], on="symbol", how="left")

    # 4. Draw Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🟢 Top 15 Mua ròng (5 ngày qua)")
        if buy_data.empty:
            st.info("Không có dữ liệu mua ròng.")
        else:
            buy_chart = (
                alt.Chart(buy_data)
                .mark_bar()
                .encode(
                    x=alt.X("sum(total_net):Q", title="Giá trị Mua ròng (VND)", axis=alt.Axis(format="~s")),
                    y=alt.Y("symbol:N", sort="-x", title="Mã CK"),
                    color=alt.Color("date:T", title="Ngày", scale=alt.Scale(scheme="greens")),
                    tooltip=["symbol", "date", alt.Tooltip("sum(total_net)", format=",.0f")]
                )
                .properties(height=400)
            )
            st.altair_chart(buy_chart, use_container_width=True)

    with col2:
        st.markdown("##### 🔴 Top 15 Bán ròng (5 ngày qua)")
        if sell_data.empty:
            st.info("Không có dữ liệu bán ròng.")
        else:
            # Use absolute value for bar length so it looks good
            sell_data["abs_val"] = sell_data["total_net"].abs()
            
            sell_chart = (
                alt.Chart(sell_data)
                .mark_bar()
                .encode(
                    x=alt.X("sum(abs_val):Q", title="Giá trị Bán ròng (VND)", axis=alt.Axis(format="~s")),
                    y=alt.Y("symbol:N", sort="-x", title="Mã CK"),
                    color=alt.Color("date:T", title="Ngày", scale=alt.Scale(scheme="reds")),
                    tooltip=["symbol", "date", alt.Tooltip("sum(total_net)", format=",.0f")]
                )
                .properties(height=400)
            )
            st.altair_chart(sell_chart, use_container_width=True)

    st.caption("""
    📘 **Giải thích:**
    - Biểu đồ hiển thị dòng tiền **Nước ngoài** tích lũy trong 5 phiên gần nhất.
    - Màu sắc đậm/nhạt thể hiện các ngày khác nhau (Stack Chart).
    """)

# -------- Tab 4: Portfolio Risk & Return Estimator -------- #
with tabs[3]:
    st.subheader("💼 Portfolio Risk & Return Estimator (Mean–Variance Model)")

    # ================== HELPER FUNCTIONS ================== #

    def parse_portfolio_text(raw_text, available_symbols, active_data_map):
        """
        Parse a string like 'mbb 20 23.5 hpg 40 26.7' or 'mbb 20 hpg 40'
        into [{'symbol':'MBB','qty':20,'price':23.5}, ...].
        If price is missing, use last close.
        """
        if not raw_text.strip():
            return []

        tokens = raw_text.strip().split()
        results = []
        i = 0
        while i < len(tokens):
            token = tokens[i].upper()
            if token in available_symbols:
                qty = 0.0
                price = None
                # Quantity (next token)
                if i + 1 < len(tokens) and re.match(r"^\d+(\.\d+)?$", tokens[i + 1]):
                    qty = float(tokens[i + 1])
                    i += 1
                # Optional price (next token)
                if i + 1 < len(tokens) and re.match(r"^\d+(\.\d+)?$", tokens[i + 1]):
                    price = float(tokens[i + 1])
                    i += 1
                # Default to last close if no price provided
                if price is None:
                    df = active_data_map.get(token)
                    price = float(df["close"].iloc[-1]) if df is not None else 0.0
                # Skip zero quantities
                if qty > 0:
                    results.append({"symbol": token, "qty": qty, "price": price})
            i += 1
        return results

    @st.cache_data(show_spinner=False)
    def compute_portfolio_weights(mode, input_data, selected, active_data_map):
        """Compute weights from quantities or market values."""
        if "chuỗi" in mode:  # text-input mode
            qtys, values = [], []
            for row in input_data:
                sym = row["symbol"]
                qty = float(row["qty"]) if row["qty"] else 0.0
                price = float(row["price"]) if row["price"] else 0.0
                df = active_data_map[sym]
                last_close = float(df["close"].iloc[-1])
                used_price = price if price > 0 else last_close
                qtys.append(qty)
                values.append(qty * used_price)
            df_val = pd.DataFrame({"symbol": selected, "value": values})
        else:
            rows = []
            for sym in selected:
                df = active_data_map[sym]
                last_close = float(df["close"].iloc[-1])
                last_vol = float(df["volume"].iloc[-1])
                rows.append({"symbol": sym, "value": last_close * last_vol})
            df_val = pd.DataFrame(rows)

        total_val = df_val["value"].sum()
        df_val["weight"] = df_val["value"] / total_val if total_val > 0 else 1 / len(selected)
        return df_val[["symbol", "weight"]]

    @st.cache_data(show_spinner=False)
    def compute_returns_stats(active_data_map, selected):
        """Compute daily returns, annualized mean and covariance."""
        returns_data = {}
        for sym in selected:
            df = active_data_map[sym].dropna(subset=["close", "time"]).sort_values("time")
            df["time"] = pd.to_datetime(df["time"])
            df["ret"] = df["close"].pct_change()
            returns_data[sym] = df.set_index("time")["ret"].iloc[-252:]
        rets_df = pd.concat(returns_data, axis=1).dropna(how="all")
        mean_ret = rets_df.mean() * 252
        cov = rets_df.cov() * 252
        return rets_df, mean_ret, cov

    # ================== USER INPUT FORM ================== #
    with st.form("portfolio_builder"):
        active_data_map = data_map.copy()
        all_symbols = [s.upper() for s in active_data_map.keys()]

        mode = st.radio(
            "Chọn cách nhập danh mục:",
            ["Nhập bằng chuỗi (ví dụ: mbb 20 23.5 hpg 40 26.7)", "Tỷ trọng theo giá trị thị trường (ETF-style)"],
            horizontal=True,
        )

        selected, input_data = [], []

        if "chuỗi" in mode:
            st.markdown("### 🧾 Nhập danh mục theo chuỗi")
            raw_text = st.text_area(
                "Nhập danh mục (tên mã, khối lượng, và giá nếu có):",
                value="",
                placeholder="Ví dụ: mbb 20 23.5 hpg 40 26.7 vic 10",
                height=100,
            )

            if raw_text.strip():
                parsed = parse_portfolio_text(raw_text, all_symbols, active_data_map)
                selected = [p["symbol"] for p in parsed]
                input_data = parsed

                # Preview parsed data
                st.dataframe(
                    pd.DataFrame(parsed).style.format({"qty": "{:.0f}", "price": "{:,.2f}"}),
                    use_container_width=True
                )
                if parsed:
                    st.success("✅ Đã nhận diện: " + ", ".join(selected))
        else:
            st.markdown("### 📊 Tỷ trọng theo giá trị giao dịch gần nhất")
            selected = st.multiselect(
                "Chọn cổ phiếu trong danh mục",
                options=sorted(active_data_map.keys()),
                default=[],
            )
            input_data = [{"symbol": sym, "qty": None, "price": None} for sym in selected]

        submitted = st.form_submit_button("📊 Tính toán danh mục")

        # ---------- COMPUTE PORTFOLIO (runs only once per submission) ---------- #
        if submitted:
            if not selected:
                st.warning("Chọn hoặc nhập ít nhất một mã cổ phiếu.")
                st.stop()

            portfolio_data = compute_portfolio_weights(mode, input_data, selected, active_data_map)
            rets_df, mean_ret, cov = compute_returns_stats(active_data_map, selected)

            weights = (
                portfolio_data.set_index("symbol")
                .reindex(rets_df.columns)
                .fillna(0)["weight"]
                .values
            )
            port_ret = np.dot(weights, mean_ret)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

            # Save results
            st.session_state["portfolio_results"] = {
                "portfolio_data": portfolio_data,
                "rets_df": rets_df,
                "mean_ret": mean_ret,
                "cov": cov,
                "port_ret": port_ret,
                "port_vol": port_vol,
                "selected": selected,
            }

    # ---------- LOAD PERSISTENT RESULTS ---------- #
    res = st.session_state.get("portfolio_results")
    if not res:
        st.info("Nhấn **📊 Tính toán danh mục** để bắt đầu.")
        st.stop()

    portfolio_data = res["portfolio_data"]
    rets_df = res["rets_df"]
    mean_ret = res["mean_ret"]
    cov = res["cov"]
    port_ret = res["port_ret"]
    port_vol = res["port_vol"]
    selected = res["selected"]

    # Display quick summary
    st.markdown("### 📈 Kết quả danh mục:")
    st.write(f"**Expected annual return:** {port_ret * 100:.2f} %")
    st.write(f"**Expected annual volatility:** {port_vol * 100:.2f} %")
    st.write(f"**Number of assets:** {len(selected)}")


    # ================== Cached Portfolio Optimization ================== #
    @st.cache_data(show_spinner=False)
    def optimize_portfolios(mean_ret, cov, rf):
        """Compute long-only Min-Variance and Max-Sharpe portfolios (cached)."""
        from scipy.optimize import minimize
        mu, Sigma = mean_ret.values, cov.values
        n = len(mu)
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
        bounds = [(0, 1)] * n
        w0 = np.ones(n) / n

        def port_stats(w): r, v = w @ mu, np.sqrt(w @ Sigma @ w); return r, v
        def neg_sharpe(w): r, v = port_stats(w); return 1e6 if v <= 1e-12 else -(r - rf) / v
        def var_obj(w): return w @ Sigma @ w

        res_min = minimize(var_obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
        w_min = res_min.x
        ret_min, vol_min = port_stats(w_min)

        res_tan = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
        w_tan = res_tan.x
        ret_tan, vol_tan = port_stats(w_tan)
        sharpe_tan = (ret_tan - rf) / vol_tan
        return {
            "w_min": w_min, "ret_min": ret_min, "vol_min": vol_min,
            "w_tan": w_tan, "ret_tan": ret_tan, "vol_tan": vol_tan, "sharpe_tan": sharpe_tan
        }

    rf = st.number_input("Lãi suất phi rủi ro (năm, %)", value=3.0, step=0.1) / 100
    opt = optimize_portfolios(mean_ret, cov, rf)

    # --- Extract results ---
    w_min, w_tan = opt["w_min"], opt["w_tan"]
    ret_min, vol_min, ret_tan, vol_tan, sharpe_tan = (
        opt["ret_min"], opt["vol_min"], opt["ret_tan"], opt["vol_tan"], opt["sharpe_tan"]
    )

    # ================== Efficient Frontier Plot ================== #
    st.markdown("## 📊 Efficient Frontier & Tangency Portfolio (No-Short)")

    frontier = []
    for alpha in np.linspace(0, 1, 40):
        w = alpha * w_tan + (1 - alpha) * w_min
        w /= w.sum()
        r, v = w @ mean_ret.values, np.sqrt(w @ cov.values @ w)
        frontier.append((v, r))
    vols, rets = zip(*frontier)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vols, y=rets, mode="lines", name="Efficient Frontier", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=[vol_min], y=[ret_min], mode="markers+text", text=["Min-Var"], textposition="bottom right", marker=dict(color="blue", size=10)))
    fig.add_trace(go.Scatter(x=[vol_tan], y=[ret_tan], mode="markers+text", text=["Tangency"], textposition="top left", marker=dict(color="gold", size=12, symbol="star")))
    fig.add_trace(go.Scatter(x=[port_vol], y=[port_ret], mode="markers+text", text=["Your Portfolio"], textposition="bottom right", marker=dict(color="red", size=12, symbol="x")))
    cal_x = np.linspace(0, max(vol_tan, port_vol) * 1.5, 100)
    cal_y = rf + sharpe_tan * cal_x
    fig.add_trace(go.Scatter(x=cal_x, y=cal_y, mode="lines", line=dict(color="green", dash="dash"), name="CAL"))
    fig.update_layout(title="Efficient Frontier (Long-only) & Capital Allocation Line", xaxis_title="Volatility (σ)", yaxis_title="Expected Return (E[R])", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # ===================== Optimal Portfolio → Trade List ===================== #
    st.markdown("## 🧭 Optimal Portfolio & Trade Plan")

    # --- Choose which optimal portfolio to target ---
    target_choice = st.radio(
        "Chọn danh mục tối ưu:",
        ["Max Sharpe (Tangency)", "Min Variance"],
        horizontal=True,
    )

    round_qty = st.checkbox("Làm tròn số lượng cổ phiếu (nguyên)", value=True)

    # --- Align assets ---
    assets = list(rets_df.columns)
    mu = mean_ret.loc[assets].values
    Sigma = cov.loc[assets, assets].values
    ones = np.ones(len(assets))
    Sigma_stable = Sigma + 1e-9 * np.eye(len(assets))

    # --- Long-only optimizers ---
    

    def max_sharpe_longonly(mu, Sigma, rf):
        def neg_sharpe(w):
            ret = w @ mu
            vol = np.sqrt(w @ Sigma @ w)
            return 1e6 if vol <= 1e-12 else -((ret - rf) / vol)
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
        bounds = [(0, 1)] * len(mu)
        w0 = np.ones(len(mu)) / len(mu)
        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
        if not res.success:
            st.warning(f"SLSQP failed: {res.message}")
        return res.x

    def min_var_longonly(Sigma):
        def var_obj(w): return w @ Sigma @ w
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
        bounds = [(0, 1)] * Sigma.shape[0]
        w0 = np.ones(Sigma.shape[0]) / Sigma.shape[0]
        res = minimize(var_obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
        if not res.success:
            st.warning(f"SLSQP failed: {res.message}")
        return res.x

    # --- Compute target weights ---
    if target_choice == "Max Sharpe (Tangency)":
        w_target = max_sharpe_longonly(mu, Sigma_stable, rf)
    else:
        w_target = min_var_longonly(Sigma_stable)

    # --- Current weights from earlier ---
    curr_w = (
        portfolio_data.set_index("symbol")
        .reindex(assets)
        .fillna(0.0)["weight"]
        .values
    )

    # --- Combine current vs. target ---
    comp = pd.DataFrame({
        "symbol": assets,
        "current_weight": curr_w,
        "target_weight": w_target
    })
    comp["delta_weight"] = comp["target_weight"] - comp["current_weight"]

    # --- Compute trade plan (VALUE & QTY) ---
    def last_price(sym):
        df = active_data_map[sym].dropna(subset=["close"])
        return float(df["close"].iloc[-1])

    # Detect if user entered manual quantities
    has_qty = False
    try:
        has_qty = any([(row.get("qty") not in ["", None]) for row in input_data])
    except Exception:
        has_qty = False

    if has_qty:
        rows = []
        for row in input_data:
            sym = row["symbol"]
            qty = float(row["qty"]) if row["qty"] not in ["", None] else 0.0
            px_last = last_price(sym)
            curr_val = qty * px_last
            rows.append({"symbol": sym, "curr_qty": qty, "last_price": px_last, "curr_val": curr_val})
        hold_df = pd.DataFrame(rows).set_index("symbol").reindex(assets).fillna(0.0)
        total_val = hold_df["curr_val"].sum()
        if total_val <= 0:
            total_val = 100_000_000.0
            hold_df["curr_val"] = comp.set_index("symbol")["current_weight"].values * total_val
            hold_df["curr_qty"] = hold_df["curr_val"] / hold_df["last_price"]
    else:
        total_val = 100_000_000.0
        rows = []
        for sym, w in zip(assets, curr_w):
            px = last_price(sym)
            val = w * total_val
            rows.append({"symbol": sym, "curr_qty": val / px if px > 0 else 0.0, "last_price": px, "curr_val": val})
        hold_df = pd.DataFrame(rows).set_index("symbol")

    # --- Target values and quantities ---
    comp = comp.set_index("symbol").join(hold_df, how="left")
    comp["target_val"] = comp["target_weight"] * total_val
    comp["trade_val"] = comp["target_val"] - comp["curr_val"]
    comp["target_qty"] = comp["target_val"] / comp["last_price"].replace(0, np.nan)
    comp["trade_qty"] = comp["target_qty"] - comp["curr_qty"]

    # --- Rounding ---
    if round_qty:
        comp["trade_qty"] = comp["trade_qty"].round().astype(int)
        comp["target_qty"] = (comp["curr_qty"] + comp["trade_qty"]).astype(float)

    # --- Final trade table ---
    trade_tbl = comp.reset_index()[[
        "symbol", "current_weight", "target_weight", "delta_weight",
        "last_price", "curr_qty", "target_qty", "trade_qty",
        "curr_val", "target_val", "trade_val"
    ]]

    st.markdown("### 📋 Kế hoạch giao dịch để đạt danh mục tối ưu")
    st.dataframe(
        trade_tbl.style.format({
            "current_weight": "{:.2%}",
            "target_weight": "{:.2%}",
            "delta_weight": "{:+.2%}",
            "last_price": "{:,.0f}",
            "curr_qty": "{:,.0f}",
            "target_qty": "{:,.0f}",
            "trade_qty": "{:+,.0f}",
            "curr_val": "{:,.0f}",
            "target_val": "{:,.0f}",
            "trade_val": "{:+,.0f}",
        }),
        use_container_width=True
    )

    # --- Summary ---
    total_buy  = trade_tbl.loc[trade_tbl["trade_val"] > 0, "trade_val"].sum()
    total_sell = -trade_tbl.loc[trade_tbl["trade_val"] < 0, "trade_val"].sum()
    st.markdown(f"**Tổng cần MUA:** {'{:,.0f}'.format(total_buy)} VND  |  **Tổng cần BÁN:** {'{:,.0f}'.format(total_sell)} VND")

    # ===================== Out-of-Sample Backtest ===================== #
    st.markdown("## 🧪 Out-of-Sample Backtest")

    # --- Default split: 70% train / 30% test ---
    default_split = rets_df.index[int(len(rets_df) * 0.7)]
    split_date = st.date_input(
        "Chọn ngày chia dữ liệu Train/Test (phân tích lịch sử):",
        value=default_split.date()
    )

    # --- Split returns ---
    train = rets_df.loc[:str(split_date)]
    test = rets_df.loc[str(split_date):]

    if len(train) < 50 or len(test) < 20:
        st.warning("Không đủ dữ liệu để backtest.")
        st.stop()

    # --- Estimate mean & covariance on training data ---
    mu_train = train.mean() * 252
    Sigma_train = train.cov() * 252
    rf = rf  # already defined above

    # --- Cached optimizer for train/test split ---
    @st.cache_data(show_spinner=False)
    def optimize_train_test(mu_train, Sigma_train, rf):
        """Compute tangency and min-var portfolios on training data."""
        from scipy.optimize import minimize
        mu, Sigma = mu_train.values, Sigma_train.values
        n = len(mu)
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
        bounds = [(0, 1)] * n
        w0 = np.ones(n) / n

        def port_stats(w): 
            r, v = w @ mu, np.sqrt(w @ Sigma @ w)
            return r, v

        def neg_sharpe(w):
            r, v = port_stats(w)
            return 1e6 if v <= 1e-12 else -(r - rf) / v

        def var_obj(w): 
            return w @ Sigma @ w

        res_tan = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
        res_min = minimize(var_obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
        w_tan = res_tan.x if res_tan.success else w0
        w_min = res_min.x if res_min.success else w0
        return w_tan, w_min

    # --- Run cached optimization ---
    w_tan_train, w_minvar_train = optimize_train_test(mu_train, Sigma_train, rf)

    # --- Apply fixed weights on test sample ---
    test_port_rets_tan = test @ w_tan_train
    test_port_rets_min = test @ w_minvar_train
    test_port_rets_eq = test.mean(axis=1)  # Equal-weight benchmark

    # --- Compute annualized metrics ---
    def perf_stats(r, rf=rf):
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
        return ann_ret, ann_vol, sharpe

    ret_tan, vol_tan, sh_tan = perf_stats(test_port_rets_tan)
    ret_min, vol_min, sh_min = perf_stats(test_port_rets_min)
    ret_eq, vol_eq, sh_eq = perf_stats(test_port_rets_eq)

    # --- Plot cumulative returns ---
    cum_tan = (1 + test_port_rets_tan).cumprod()
    cum_min = (1 + test_port_rets_min).cumprod()
    cum_eq = (1 + test_port_rets_eq).cumprod()

    fig_back = go.Figure()
    fig_back.add_trace(go.Scatter(x=cum_tan.index, y=cum_tan, name="Tangency (Train)", line=dict(color="gold", width=2)))
    fig_back.add_trace(go.Scatter(x=cum_min.index, y=cum_min, name="Min-Var (Train)", line=dict(color="blue", width=2)))
    fig_back.add_trace(go.Scatter(x=cum_eq.index, y=cum_eq, name="Equal-Weight", line=dict(color="gray", dash="dash")))
    fig_back.update_layout(
        title="Out-of-Sample Cumulative Performance",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (×)",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98)
    )
    st.plotly_chart(fig_back, use_container_width=True)

    # --- Summary table ---
    st.markdown("### 🧾 Out-of-Sample Performance Metrics (Test Set)")
    df_perf = pd.DataFrame({
        "Portfolio": ["Tangency (Max Sharpe)", "Min Variance", "Equal Weight"],
        "Return (%)": [ret_tan * 100, ret_min * 100, ret_eq * 100],
        "Volatility (%)": [vol_tan * 100, vol_min * 100, vol_eq * 100],
        "Sharpe": [sh_tan, sh_min, sh_eq]
    })
    st.dataframe(
        df_perf.style.format({
            "Return (%)": "{:.2f}",
            "Volatility (%)": "{:.2f}",
            "Sharpe": "{:.2f}"
        }),
        use_container_width=True
    )


    # ===================== Monte Carlo Simulation: 1-Year Portfolio Distribution ===================== #
    st.markdown("## 🎲 Monte Carlo Risk Simulation (1-Year)")

    n_sims = st.slider("Số lần mô phỏng (Monte Carlo)", 500, 5000, 2000, 500)
    initial_port_val = 100_000_000  # assume 100m VND starting value

    # Extract inputs
    mean_daily = rets_df.mean().values        # daily mean returns
    cov_daily = rets_df.cov().values          # daily covariance
    weights = portfolio_data.set_index("symbol").reindex(rets_df.columns).fillna(0)["weight"].values
    #weights = comp.loc[rets_df.columns, "target_weight"].values  # use optimal weights
    # From optimizer (annualized)
    mu_annual_p    = float(weights @ mean_ret.loc[rets_df.columns])
    sigma_annual_p = float(np.sqrt(weights @ cov.loc[rets_df.columns, rets_df.columns] @ weights))
    st.write("Check: μ annual =", mu_annual_p, "σ annual =", sigma_annual_p)
    # Convert to daily for GBM
    T = 252  # Number of trading days in a year
    dt = 1/252
    np.random.seed(42)
    simulated_end_values = []
    for _ in range(n_sims):
        Z = np.random.normal(0, 1, T)
        growth = np.exp((mu_annual_p - 0.5*sigma_annual_p**2)*dt
                        + sigma_annual_p*np.sqrt(dt)*Z)
        end_value = initial_port_val * np.prod(growth)
        simulated_end_values.append(end_value)



    sim_df = pd.DataFrame(simulated_end_values, columns=["FinalValue"])
    final_ret = sim_df["FinalValue"] / initial_port_val - 1

    # --- Risk metrics ---
    p5 = np.percentile(final_ret, 5)   # 5th percentile return
    p95 = np.percentile(final_ret, 95)
    mean_r = np.mean(final_ret)

    st.markdown(f"""
    **📊 Expected annual return:** {mean_r*100:.2f}%  
    **💥 5th percentile (VaR 95%):** {p5*100:.2f}% → Potential worst case after 1 year  
    **🎯 95th percentile (Best case):** {p95*100:.2f}%
    """)

    # --- Plot distribution ---
    fig = px.histogram(
        final_ret * 100,
        nbins=60,
        title="Monte Carlo Simulated Portfolio Returns (1 Year)",
        labels={"value": "Simulated Return (%)"},
        opacity=0.75
    )
    fig.add_vline(x=p5*100, line_dash="dash", line_color="red",
                annotation_text="5th % (VaR)", annotation_position="top left")
    fig.add_vline(x=p95*100, line_dash="dash", line_color="green",
                annotation_text="95th %", annotation_position="top right")
    fig.add_vline(x=mean_r*100, line_dash="solid", line_color="blue",
                annotation_text="Mean", annotation_position="bottom right")

    st.plotly_chart(fig, use_container_width=True)

    # ===================== Trading Book Stress Test (Basel-style 10-day) ===================== #
    st.markdown("## 🏦 Trading Book 10-Day Stress Test (Basel-style)")

    # --- choose portfolio to stress ---
    stress_choice = st.radio(
        "Chọn danh mục để stress test:",
        ["Danh mục hiện tại", "Max Sharpe (tối ưu)", "Min Variance (tối ưu)"],
        horizontal=True,
    )

    assets = list(rets_df.columns)
    if stress_choice == "Max Sharpe (tối ưu)":
        stress_w = w_target
    elif stress_choice == "Min Variance (tối ưu)":
        stress_w = min_var_longonly(cov.loc[assets, assets].values)
    else:
        stress_w = (
            portfolio_data.set_index("symbol")
            .reindex(assets)
            .fillna(0)["weight"]
            .values
        )

    # --- build daily portfolio returns ---
    port_ret_s = rets_df[assets].dot(stress_w).dropna()
    st.caption(f"Số ngày có dữ liệu: {len(port_ret_s)}")

    # --- parameters ---
    win = 10                # 10-day horizon (Basel standard)
    conf_level = 0.99       # 99% VaR

    # --- compute 10-day compounded rolling returns ---
    roll_10d = (1 + port_ret_s).rolling(win).apply(lambda x: np.prod(x) - 1.0, raw=False).dropna()

    # --- compute metrics ---
    worst_10d = roll_10d.min()
    worst_date = roll_10d.idxmin()
    var_99 = np.percentile(roll_10d, (1 - conf_level) * 100)

    # --- display summary ---
    st.markdown(f"""
    **📆 Period end:** {worst_date.date()}  
    **📉 Worst 10-day cumulative loss:** {worst_10d:.2%}  
    **⚠️ 99% 10-day VaR:** {var_99:.2%}  
    *(Horizon = 10 trading days, confidence = 99% as in Basel Trading-Book rules)*
    """)

    # --- plot rolling 10-day returns ---
    roll_df = roll_10d.reset_index()
    roll_df.columns = ["Date", "10-Day Return"]

    fig_var = px.line(
        roll_df,
        x="Date",
        y="10-Day Return",
        title="10-Day Compounded Portfolio Returns (Trading-Book Horizon)",
        labels={"10-Day Return": "10-Day Cumulative Return"}
    )

    fig_var.add_hline(
        y=var_99,
        line_dash="dash",
        line_color="red",
        annotation_text="99% VaR",
        annotation_position="top left"
    )
    fig_var.add_hline(
        y=worst_10d,
        line_dash="dot",
        line_color="black",
        annotation_text="Worst 10-Day Loss",
        annotation_position="bottom left"
    )

    st.plotly_chart(fig_var, use_container_width=True)

    # --- drill-down around the worst period ---
    st.markdown("### 🔍 Phân rã theo mã trong giai đoạn tệ nhất")

    # slice 10-day window around the worst date
    end_idx = port_ret_s.index.get_loc(worst_date)
    if isinstance(end_idx, slice):
        end_idx = end_idx.stop - 1
    elif isinstance(end_idx, (np.ndarray, list)):
        end_idx = int(end_idx[0])

    start_idx = max(0, end_idx - win + 1)
    start_date = port_ret_s.index[start_idx]
    window_mask = (rets_df.index >= start_date) & (rets_df.index <= worst_date)
    block = rets_df.loc[window_mask, assets].dropna(how="all")

    # compute per-asset 10-day return and contribution
    asset_window_ret = (1 + block).prod() - 1
    contrib = pd.Series(stress_w, index=assets) * asset_window_ret
    contrib_tbl = pd.DataFrame({
        "symbol": assets,
        "Weight": stress_w,
        "10-Day Return": asset_window_ret,
        "Contribution": contrib
    }).sort_values("Contribution")

    window_port_ret = float(contrib.sum())
    st.markdown(
        f"**Khoảng:** {start_date.date()} → {worst_date.date()}  \n"
        f"**Lợi nhuận danh mục (10-ngày):** {window_port_ret:.2%}"
    )

    st.dataframe(
        contrib_tbl.style.format({
            "Weight": "{:.2%}",
            "10-Day Return": "{:.2%}",
            "Contribution": "{:.2%}",
        }),
        use_container_width=True
    )

    # plot cumulative return with highlighted stress window
    ts = (1 + port_ret_s).cumprod()

    # Make a tidy DataFrame with clear column names
    ts_df = ts.rename("Cumulative").to_frame().reset_index()
    # After reset_index() the date column is whatever the index name was (often 'time')
    # Standardize names:
    ts_df.columns = ["Date", "Cumulative"]

    fig_ts = px.line(
        ts_df,
        x="Date",
        y="Cumulative",
        title="Giá trị danh mục (chuỗi tích luỹ)"
    )
    fig_ts.add_vrect(
        x0=pd.to_datetime(start_date),
        x1=pd.to_datetime(worst_date),
        line_width=0,
        fillcolor="red",
        opacity=0.15
    )
    fig_ts.update_layout(
        xaxis_title="Ngày",
        yaxis_title="Giá trị tích luỹ (∏(1+r))"
    )
    st.plotly_chart(fig_ts, use_container_width=True)


