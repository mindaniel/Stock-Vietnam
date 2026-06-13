import pandas as pd
import numpy as np
import glob
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    USE_XGB = True
except ImportError:
    USE_XGB = False
import warnings
from joblib import dump, load

# SILENCE WARNINGS
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Parse command-line arguments for training date range
parser = argparse.ArgumentParser(description="AI Liquid Scanner")
parser.add_argument("--train-start", type=str, default="2000-01-01")
parser.add_argument("--train-end", type=str, default="2020-12-31")
args = parser.parse_args()

# --- LIQUID-SPECIFIC SETTINGS ---
DATA_PATH = "data"
FUNDAMENTAL_PATH = "data_stocks"
INDEX_FILE = "VNINDEX.csv"

# NEW: Specific Brain and Folder for Liquid Strategy
BRAIN_FILE = "ai_brain_liquid.joblib"
SIGNALS_FOLDER = "Signals_Liquid"

if not os.path.exists(SIGNALS_FOLDER):
    os.makedirs(SIGNALS_FOLDER)

# --- PROVEN PARAMETERS ---
MIN_PROBABILITY = 0.60
MIN_VOLUME = 50_000  # Updated to focus on liquid stocks
MAX_VOLATILITY = 0.05
HOLD_DAYS = 3
STOP_LOSS_PCT = 0.15

BLACKLIST_STOCKS = {
    'BCG', 'KPF', 'PCH', 'HHC', 'PGN', 'ATS', 'VPG', 'ORS',
    'PHC', 'EVS', 'HII', 'SID', 'HVX', 'ILS', 'KHS', 'HLC'
}

FEATURES = [
    'RSI', 'Trend_Score', 'PE_Ratio', 'VWAP_Score', 'Rev_Growth',
    'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Score',
    'Vol_Force', 'Relative_Performance', 'ATR' # New Features
]

# Load VNINDEX globally for relative performance
if os.path.exists(INDEX_FILE):
    idx_df = pd.read_csv(INDEX_FILE)
    idx_df['date'] = pd.to_datetime(idx_df['date'], format='mixed', dayfirst=True)
    INDEX_DF = idx_df.sort_values('date').set_index('date')
else:
    print(f"Warning: {INDEX_FILE} not found. Relative Performance will be 0.")
    INDEX_DF = None

def load_data_package(symbol):
    path = os.path.join(DATA_PATH, f"{symbol}.csv")
    if not os.path.exists(path): return None, None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        rename_map = {'time': 'date', 'dt': 'date', 'dong': 'close', 'mo': 'open', 'cao': 'high', 'thap': 'low', 'vol': 'volume'}
        df = df.rename(columns=rename_map)
        df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
        df = df.dropna(subset=['date', 'close']).set_index('date').sort_index()
        for c in ['close', 'high', 'low', 'open', 'volume']:
            if df[c].dtype == 'object': 
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce')
    except: return None, None

    f_path = os.path.join(FUNDAMENTAL_PATH, f"{symbol}.csv")
    if not os.path.exists(f_path): return df, None
    try:
        f_df = pd.read_csv(f_path)
        # simplified fundamental mapping for brevity
        f_df['release_date'] = pd.to_datetime(f_df['year'].astype(str) + '-' + (f_df['quarter'] * 3).astype(str) + '-01')
        f_df = f_df.dropna(subset=['release_date', 'EPS']).sort_values('release_date')
        f_df['Rev_Growth'] = f_df['Revenue'].pct_change()
        return df, f_df[['release_date', 'EPS', 'BVPS', 'Rev_Growth']]
    except: return df, None

def prepare_features(df, fund_df):
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['Trend_Score'] = (df['close'] - df['SMA_20']) / df['SMA_20'].replace(0, 1e-9)
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    df['Volatility'] = ((df['high'] - df['low']) / df['close']).rolling(14).mean()
    df['Vol_SMA_20'] = df['volume'].rolling(20).mean()

    # --- NEW LIQUID FEATURES ---
    df['Vol_Force'] = df['volume'] / df['Vol_SMA_20'].replace(0, 1e-9)
    
    if INDEX_DF is not None:
        df = df.join(INDEX_DF[['close']].rename(columns={'close': 'Index_Close'}), how='left')
        df['Relative_Performance'] = df['close'].pct_change(3) - df['Index_Close'].pct_change(3)
    else:
        df['Relative_Performance'] = 0

    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    # ---------------------------

    pv = df['close'] * df['volume']
    vwap = pv.rolling(20).sum() / df['volume'].rolling(20).sum().replace(0, 1e-9)
    df['VWAP_Score'] = (df['close'] - vwap) / vwap

    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['BB_Upper'] = sma20 + 2 * std20
    df['BB_Lower'] = sma20 - 2 * std20
    df['BB_Score'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']).replace(0, 1e-9)

    if fund_df is not None and not fund_df.empty:
        df = pd.merge_asof(df.sort_index(), fund_df.sort_values('release_date'), left_index=True, right_on='release_date', direction='backward')
        df.index = df['date_x'] if 'date_x' in df.columns else df.index # handle duplicate index
        df['PE_Ratio'] = df['close'] / df['EPS'].replace(0, 0.01)
        df['Rev_Growth'] = df['Rev_Growth'].fillna(0)
    else:
        df['PE_Ratio'] = 0
        df['Rev_Growth'] = 0

    # --- NEW VOLATILITY-ADJUSTED TARGET ---
    # Target 1 if the return is greater than 50% of its Average True Range
    future_close = df['close'].shift(-HOLD_DAYS)
    future_return = future_close - df['close']
    df['Is_Safe'] = np.where(future_return > (0.5 * df['ATR']), 1, 0)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna(subset=['SMA_20']).fillna(0)

def run_liquid_training():
    print("="*80)
    print(f"TRAINING LIQUID AI BRAIN ({args.train_start} to {args.train_end})")
    print("="*80)

    symbols = [os.path.basename(f).replace('.csv', '') for f in glob.glob(os.path.join(DATA_PATH, "*.csv"))]
    all_data = []

    print(f"Processing {len(symbols)} stocks...")
    for i, s in enumerate(symbols):
        if i % 50 == 0: print(f"  Processing {i}/{len(symbols)}...", end='\r')
        df, f_df = load_data_package(s)
        if df is None or len(df) < 50: continue
        
        df = prepare_features(df, f_df)
        df['Symbol'] = s
        all_data.append(df)
        
    master_df = pd.concat(all_data)
    
    train_start_dt = pd.Timestamp(args.train_start)
    train_end_dt = pd.Timestamp(args.train_end)
    
    train_data = master_df[(master_df.index >= train_start_dt) & 
                           (master_df.index <= train_end_dt)].dropna(subset=['Is_Safe'])
    
    # --- THE LIQUIDITY GATE ---
    # The AI is strictly forbidden from learning patterns from low-volume stocks
    train_data = train_data[train_data['Vol_SMA_20'] >= MIN_VOLUME]
    
    print(f"\nTeaching Liquid Brain with {len(train_data)} high-volume historic patterns...")
    if USE_XGB:
        clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=50, min_samples_leaf=20, n_jobs=-1, random_state=42)
    
    clf.fit(train_data[FEATURES], train_data['Is_Safe'])
    dump(clf, BRAIN_FILE)
    print(f"Brain saved to {BRAIN_FILE}. Ready to scan liquid stocks.")

if __name__ == "__main__":
    run_liquid_training()