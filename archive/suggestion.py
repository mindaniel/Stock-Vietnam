import pandas as pd
import numpy as np
import glob
import os
import sys
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# 🛑 SILENCE WARNINGS
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# 🛠️ FIX ICONS
USE_EMOJIS = True
try:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
except:
    USE_EMOJIS = False

# ==========================================
# ⚙️ SETTINGS
# ==========================================
DATA_PATH = r"C:\Users\Daziel Brilliant\Downloads\Stock-Vietnam-main\data"
FUNDAMENTAL_PATH = r"C:\Users\Daziel Brilliant\Downloads\Stock-Vietnam-main\data_stocks"
INDUSTRY_FILE = r"C:\Users\Daziel Brilliant\Downloads\Stock-Vietnam-main\vndirect_listing.xlsx"
BRAIN_FILE = "Backtest_Industry.csv"
VNINDEX_FILE = os.path.join("VNINDEX.csv")  

# 🛡️ BUYING RULES
MIN_GAIN = 0.02           
MIN_SURVIVAL = 0.65        
MAX_SUGGESTIONS = 7       

# ==========================================
# 1. INDUSTRY HELPERS
# ==========================================
def load_industry_map():
    if not os.path.exists(INDUSTRY_FILE): return None
    try:
        if INDUSTRY_FILE.endswith('.xlsx'):
            df = pd.read_excel(INDUSTRY_FILE)
        else:
            try:
                df = pd.read_csv(INDUSTRY_FILE, encoding='utf-8')
            except:
                df = pd.read_csv(INDUSTRY_FILE, encoding='latin1')
        mapping = dict(zip(df['code'], df['IndustryEN']))
        return mapping
    except: return None

def build_sector_indices(mapping):
    print("🏭 Building Live Industry Indices (Context for Today)...")
    files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
    sector_data = []
    
    count = 0
    for f in files:
        if "VNINDEX" in f: continue
        symbol = os.path.basename(f).replace('.csv', '')
        industry = mapping.get(symbol, "Unknown")
        if industry == "Unknown": continue
        try:
            df = pd.read_csv(f)
            df.columns = [c.lower().strip() for c in df.columns]
            df = df.rename(columns={'time': 'date', 'dt': 'date', 'dong': 'close'})
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date', 'close']).set_index('date').sort_index()
            if df['close'].dtype == 'object':
                df['close'] = pd.to_numeric(df['close'].astype(str).str.replace(',', '.'), errors='coerce')
            
            # Use last 100 days for speed
            df = df.tail(100)
            
            df['pct_change'] = df['close'].pct_change()
            temp = df[['pct_change']].copy()
            temp['Industry'] = industry
            sector_data.append(temp)
        except: continue
        
        count += 1
        if count % 500 == 0: print(f"   Scanned {count} stocks...", end='\r')

    if not sector_data: return pd.DataFrame()
    
    big_df = pd.concat(sector_data)
    sector_indices = big_df.groupby(['date', 'Industry'])['pct_change'].mean().unstack().fillna(0)
    return sector_indices

def load_vnindex():
    """Loads the VNINDEX data for Beta calculation."""
    if not os.path.exists(VNINDEX_FILE):
        print("❌ VNINDEX.csv missing. Run daily_update.py first.")
        return None
    try:
        df = pd.read_csv(VNINDEX_FILE)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df['Returns'] = df['close'].pct_change()
        return df[['Returns']]
    except:
        return None

# ==========================================
# 2. DATA LOADERS
# ==========================================
def get_active_stocks():
    if not os.path.exists(BRAIN_FILE):
        print(f"⚠️ Error: '{BRAIN_FILE}' not found. Run update_brain_industry.py first.")
        return None
    try:
        df = pd.read_csv(BRAIN_FILE)
        valid_df = df[
            (df['Status'] != 'COOLING_DOWN') &
            (df['Global_WR'] >= 50.0) 
        ]
        
        stats = {}
        for _, row in valid_df.iterrows():
            stats[str(row['Symbol'])] = {
                'Recent_WR': row['Recent_WR'],
                'Global_WR': row['Global_WR'],
                'Status': row['Status'],
                'Total_Trd': int(row['Total_Trd']),
                'Industry': row['Industry']
            }
        print(f"✅ Loaded Brain: {len(stats)} stocks qualified for scanning.")
        return stats
    except Exception as e:
        print(f"⚠️ Error reading brain: {e}")
        return None

def smart_load_data(symbol):
    path = os.path.join(DATA_PATH, f"{symbol}.csv")
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        rename_map = {'time': 'date', 'dt': 'date', 'dong': 'close', 'mo': 'open', 'cao': 'high', 'thap': 'low', 'vol': 'volume'}
        df = df.rename(columns=rename_map)
        
        # Priority Standard
        df_std = df.copy()
        df_std['date'] = pd.to_datetime(df_std['date'], dayfirst=False, errors='coerce')
        len_std = df_std['date'].notna().sum()
        
        # Fallback DayFirst
        df_day = df.copy()
        df_day['date'] = pd.to_datetime(df_day['date'], dayfirst=True, errors='coerce')
        len_day = df_day['date'].notna().sum()
        
        if len_std >= len_day: df = df_std
        else: df = df_day
        
        df = df.dropna(subset=['date', 'close']).set_index('date').sort_index()
        for c in ['close', 'high', 'low', 'open', 'volume']:
            if df[c].dtype == 'object': 
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce')
        
        df['close'] = df['close'].replace(0, np.nan).ffill()
        return df
    except: return None

def load_fundamental_data(symbol):
    path = os.path.join(FUNDAMENTAL_PATH, f"{symbol}.csv")
    if not os.path.exists(path): return None
    try:
        f_df = pd.read_csv(path)
        f_df = f_df.sort_values(['year', 'quarter'])
        release_dates = []
        for _, row in f_df.iterrows():
            y, q = int(row['year']), int(row['quarter'])
            if q == 1: dt = pd.Timestamp(year=y, month=5, day=15)
            elif q == 2: dt = pd.Timestamp(year=y, month=8, day=14)
            elif q == 3: dt = pd.Timestamp(year=y, month=11, day=14)
            else: dt = pd.Timestamp(year=y+1, month=2, day=14)
            release_dates.append(dt)
        f_df['release_date'] = release_dates
        f_df = f_df.dropna(subset=['release_date', 'EPS']).sort_values('release_date')
        f_df['Rev_Growth'] = f_df['Revenue'].pct_change()
        return f_df[['release_date', 'EPS', 'BVPS', 'Rev_Growth']]
    except: return None

# ==========================================
# 3. FULL FEATURE ENGINEERING (20 INDICATORS)
# ==========================================
def prepare_features(df, fund_df, vnindex_df, sector_indices, symbol_industry):
    # 1. CORE TECHNICALS
    df['Returns'] = df['close'].pct_change()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    safe_close = df['close'].replace(0, np.nan)
    df['Volatility'] = ((df['high'] - df['low']) / safe_close).rolling(14).mean()
    
    df['Month'] = df.index.month
    monthly_seasonality = df.groupby('Month')['Returns'].mean()
    df['Season_Score'] = df['Month'].map(monthly_seasonality)
    
    df['SMA_20'] = df['close'].rolling(20, min_periods=1).mean()
    safe_sma = df['SMA_20'].replace(0, np.nan)
    df['Trend_Score'] = (df['close'] - df['SMA_20']) / safe_sma
    
    low_60 = df['low'].rolling(60, min_periods=1).min()
    high_60 = df['high'].rolling(60, min_periods=1).max()
    range_60 = (high_60 - low_60).replace(0, 1e-9)
    df['Cycle_Pos'] = (df['close'] - low_60) / range_60
    
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['EMA_200'] = df['EMA_200'].fillna(df['close'].expanding().mean())
    safe_ema = df['EMA_200'].replace(0, np.nan)
    df['Big_Trend'] = (df['close'] - df['EMA_200']) / safe_ema

    pv = df['close'] * df['volume']
    vwap_20 = pv.rolling(20).sum() / df['volume'].rolling(20).sum().replace(0, 1e-9)
    df['VWAP_Score'] = (df['close'] - vwap_20) / vwap_20
    
    mf_mult = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, 1e-9)
    mf_vol = mf_mult * df['volume']
    df['CMF'] = mf_vol.rolling(20).sum() / df['volume'].rolling(20).sum().replace(0, 1e-9)
    
    force = df['close'].diff() * df['volume']
    avg_vol = df['volume'].rolling(20).mean().replace(0, 1e-9)
    df['Force_Index'] = force.ewm(span=13, adjust=False).mean() / avg_vol

    vol_sma = df['volume'].rolling(20).mean().replace(0, 1e-9)
    df['Turnover_Score'] = df['volume'] / vol_sma

    tenkan = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    kijun = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
    cloud_top = np.maximum(span_a, span_b)
    df['Ichi_Score'] = (df['close'] - cloud_top) / safe_close
    
    min_rsi = df['RSI'].rolling(14).min()
    max_rsi = df['RSI'].rolling(14).max()
    df['StochRSI'] = (df['RSI'] - min_rsi) / (max_rsi - min_rsi + 1e-9)
    
    vpt_change = df['volume'] * df['Returns']
    df['VPT'] = vpt_change.cumsum()
    vpt_sma = df['VPT'].rolling(20).mean()
    df['VPT_Score'] = (df['VPT'] - vpt_sma) / (avg_vol * 10 + 1e-9)

    # 2. FUNDAMENTALS
    if fund_df is not None and not fund_df.empty:
        df_merged = pd.merge_asof(
            df.sort_index(), 
            fund_df.sort_values('release_date'), 
            left_index=True, 
            right_on='release_date', 
            direction='backward'
        )
        safe_eps = df_merged['EPS'].replace(0, 0.01)
        df_merged['PE_Ratio'] = df_merged['close'] / safe_eps
        safe_bvps = df_merged['BVPS'].replace(0, 0.01)
        df_merged['PB_Ratio'] = df_merged['close'] / safe_bvps
        df_merged['Rev_Growth'] = df_merged['Rev_Growth'].fillna(0)
        df_merged.index = df.index
        df = df_merged
    else:
        df['PE_Ratio'] = 0
        df['PB_Ratio'] = 0
        df['Rev_Growth'] = 0

    # 3. INDUSTRY CONTEXT
    if symbol_industry in sector_indices.columns:
        ind_series = sector_indices[symbol_industry]
        df = df.join(ind_series.rename('Ind_Ret'), how='left').fillna(0)
        df['Ind_Trend_5d'] = df['Ind_Ret'].rolling(5).mean()
        df['Rel_Strength'] = df['Returns'] - df['Ind_Ret']
        df['Ind_Lag_3d'] = df['Ind_Ret'].shift(3)
    else:
        df['Ind_Ret'] = 0
        df['Ind_Trend_5d'] = 0
        df['Rel_Strength'] = 0
        df['Ind_Lag_3d'] = 0

    # 4. 🔥 BETA FACTOR (20th)
    if vnindex_df is not None and not vnindex_df.empty:
        df = df.join(vnindex_df.rename(columns={'Returns': 'Market_Ret'}), how='left')
        rolling_cov = df['Returns'].rolling(60).cov(df['Market_Ret'])
        rolling_var = df['Market_Ret'].rolling(60).var()
        df['Beta'] = (rolling_cov / (rolling_var + 1e-9)).fillna(1.0).clip(-5, 5)
    else:
        df['Beta'] = 1.0  # Default if missing

    # TARGETS
    future_close = df['close'].shift(-3)
    df['Target_High'] = (df['high'].shift(-1).rolling(3).max().shift(-2) - df['close']) / df['close']
    df['Is_Safe_T3'] = np.where(future_close > df['close'], 1, 0)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill N/A for features
    exclude_cols = ['Target_High', 'Is_Safe_T3', 'release_date']
    feat_cols = [c for c in df.columns if c not in exclude_cols]
    df[feat_cols] = df[feat_cols].fillna(0)
    
    return df

def get_ai_prediction(df):
    # 🔥 ALL 20 INDICATORS 🔥
    features = [
        'RSI', 'Volatility', 'Season_Score', 'Trend_Score', 'Cycle_Pos', 'Big_Trend', 
        'VWAP_Score', 'CMF', 'Force_Index', 'Turnover_Score',
        'Ichi_Score', 'StochRSI', 'VPT_Score',
        'PE_Ratio', 'PB_Ratio', 'Rev_Growth',
        'Ind_Trend_5d', 'Rel_Strength', 'Ind_Lag_3d',
        'Beta'  # <--- NEW
    ]
    
    train_df = df.dropna(subset=features + ['Target_High', 'Is_Safe_T3'])
    if len(train_df) < 50: return 0, 0, "N/A"
    
    X_train = train_df[features]
    y_safe = train_df['Is_Safe_T3']
    
    if len(np.unique(y_safe)) < 2: return 0, 0, "N/A"

    reg_model = RandomForestRegressor(n_estimators=30, min_samples_leaf=10, random_state=42, n_jobs=-1)
    clf_model = RandomForestClassifier(n_estimators=30, min_samples_leaf=10, random_state=42, n_jobs=-1)
    
    reg_model.fit(X_train, train_df['Target_High'])
    clf_model.fit(X_train, y_safe)
    
    # Predict on LAST row (Today)
    last_row = df.iloc[[-1]][features]
    if last_row.isnull().values.any(): return 0, 0, "N/A"
    
    pred_gain = reg_model.predict(last_row)[0]
    survival_prob = clf_model.predict_proba(last_row)[0][1]
    
    # Identify Main Factor
    importances = clf_model.feature_importances_
    top_idx = np.argmax(importances)
    top_factor = features[top_idx]
    
    return pred_gain, survival_prob, top_factor

# ==========================================
# 4. MAIN RUNNER
# ==========================================
def run_suggestion():
    print("🚀 INITIALIZING FULL-POWER SUGGESTION ENGINE (20 Indicators)...")
    
    active_stats = get_active_stocks()
    ind_map = load_industry_map()
    vnindex_df = load_vnindex()  # <--- NEW
    
    if not active_stats or not ind_map: return
    if vnindex_df is None: print("⚠️ Warning: Running without Beta (Market Context missing).")
    
    sector_indices = build_sector_indices(ind_map)
    
    print(f"\n🔮 Scanning {len(active_stats)} filtered stocks...")
    recommendations = []
    
    for i, symbol in enumerate(active_stats.keys()):
        if i % 20 == 0: print(f"   Checking {i}/{len(active_stats)}...", end='\r')
        
        df = smart_load_data(symbol)
        if df is None: continue
        
        # Filter 0 Price
        current_price = df.iloc[-1]['close']
        if current_price <= 0: continue
        
        fund_df = load_fundamental_data(symbol)
        if fund_df is None or fund_df.empty: continue
            
        my_ind = active_stats[symbol]['Industry']
        df = prepare_features(df, fund_df, vnindex_df, sector_indices, my_ind)
        
        pred_gain, survival_prob, top_factor = get_ai_prediction(df)
        
        if pred_gain > MIN_GAIN and survival_prob >= MIN_SURVIVAL:
            stock_stats = active_stats[symbol]
            score = (survival_prob * 50) + (pred_gain * 100)
            if stock_stats['Status'] == 'HEATING_UP': score += 5
            
            # Map Technical Name -> Human Readable
            reason_map = {
                'Ind_Trend_5d': 'Sector Pump 🌊',
                'Rel_Strength': 'Beat Sector 💪',
                'Ind_Lag_3d': 'Sector Drag 🪢',
                'RSI': 'Oversold 📉',
                'Trend_Score': 'Uptrend 📈',
                'VWAP_Score': 'Whale Buy 🐳',
                'Rev_Growth': 'Earnings 💰',
                'PE_Ratio': 'Cheap Val 💎',
                'Volatility': 'High Vol ⚡',
                'CMF': 'Money Flow 💵',
                'Force_Index': 'Buying Pres 🐂',
                'Ichi_Score': 'Cloud Break ☁️',
                'Season_Score': 'Seasonal 📅',
                'Beta': 'Market Sync 🔗'  # <--- NEW
            }
            readable_reason = reason_map.get(top_factor, top_factor)

            recommendations.append({
                'Symbol': symbol,
                'Price': current_price,
                'Survival': survival_prob,
                'AI_Gain': pred_gain,
                'Status': stock_stats['Status'],
                'Score': score,
                'Factor': readable_reason,
                'Industry': my_ind
            })

    recommendations.sort(key=lambda x: x['Score'], reverse=True)
    top_picks = recommendations[:MAX_SUGGESTIONS]
    
    print(f"\n\n🏆 TOP {len(top_picks)} FULL-POWER PICKS:\n")
    print(f"{'SYM':<6} | {'PRICE':<8} | {'CONF':<6} | {'GAIN':<6} | {'STATUS':<10} | {'MAIN REASON':<18} | {'INDUSTRY'}")
    print("-" * 110)
    
    for rec in top_picks:
        icon = "🔥" if rec['Status'] == 'HEATING_UP' else "⚓"
        print(f"{icon} {rec['Symbol']:<4} | {rec['Price']:<8.2f} | {rec['Survival']*100:.0f}%  | {rec['AI_Gain']*100:.1f}%  | {rec['Status']:<10} | {rec['Factor']:<18} | {rec['Industry']}")
    
    filename = f"Industry_Full_Suggestions_{datetime.now().strftime('%Y-%m-%d')}.csv"
    if top_picks: pd.DataFrame(top_picks).to_csv(filename, index=False)
    print(f"\n✅ Saved to: {filename}")

if __name__ == "__main__":
    run_suggestion()
    input()
