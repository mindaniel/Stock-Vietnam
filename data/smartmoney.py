import pandas as pd
import numpy as np
import glob
import os
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# 🛑 SILENCE WARNINGS
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

# ==========================================
# ⚙️ SETTINGS
# ==========================================
DATA_PATH = r"C:\Users\Daziel Brilliant\Downloads\Stock-Vietnam-main\data"
STRATEGY_FILE = "best_strategies.csv"

# 🛡️ BUY FILTERS
MIN_SURVIVAL_PROB = 0.60
MIN_AI_PROFIT = 0.015  # 1.5% Gain

def load_stock_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # FIX: Ensure we find the date column
        if 'date' not in df.columns:
            if 'time' in df.columns: df = df.rename(columns={'time': 'date'})
            elif 'dt' in df.columns: df = df.rename(columns={'dt': 'date'})
        
        rename_map = {'dong': 'close', 'mo': 'open', 'cao': 'high', 'thap': 'low', 'vol': 'volume'}
        df = df.rename(columns=rename_map)
        
        cols = ['open', 'high', 'low', 'close', 'volume']
        for c in cols:
            if c in df.columns and df[c].dtype == 'object':
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce')
        
        df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True, errors='coerce')
        df = df.dropna(subset=['date', 'close']).set_index('date').sort_index()
        return df
    except: return None

# 👇 THIS IS THE SPECIAL ROBUST ADD_FEATURES
def add_features(df):
    # --- 1. BASIC PRICE FEATURES ---
    df['Returns'] = df['close'].pct_change()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['Volatility'] = ((df['high'] - df['low']) / df['close']).rolling(14).mean()
    
    df['Month'] = df.index.month
    monthly_seasonality = df.groupby('Month')['Returns'].mean()
    df['Season_Score'] = df['Month'].map(monthly_seasonality)
    
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['Trend_Score'] = (df['close'] - df['SMA_20']) / df['SMA_20']

    # --- 2. ROBUST SMART MONEY ---
    # Foreign
    if 'nn_mua' in df.columns: df.rename(columns={'nn_mua': 'foreign_buy', 'nn_ban': 'foreign_sell'}, inplace=True)
    if 'nuoc_ngoai_mua' in df.columns: df.rename(columns={'nuoc_ngoai_mua': 'foreign_buy', 'nuoc_ngoai_ban': 'foreign_sell'}, inplace=True)
    
    if 'foreign_buy' in df.columns:
        df['foreign_buy'] = df['foreign_buy'].fillna(0)
        df['foreign_sell'] = df['foreign_sell'].fillna(0)
        df['NN_Net_Vol'] = df['foreign_buy'] - df['foreign_sell']
        df['NN_Accumulation'] = df['NN_Net_Vol'].rolling(20, min_periods=1).sum()
    else:
        df['NN_Net_Vol'] = 0
        df['NN_Accumulation'] = 0

    # Proprietary (Tu Doanh)
    if 'prop_buy' in df.columns:
        df['prop_buy'] = df['prop_buy'].fillna(0)
        df['prop_sell'] = df['prop_sell'].fillna(0)
        df['TD_Net_Vol'] = df['prop_buy'] - df['prop_sell']
    else:
        df['TD_Net_Vol'] = 0
        
    # Put Through
    if 'pt_vol' in df.columns:
        df['pt_vol'] = df['pt_vol'].fillna(0)
        df['PT_Spike'] = np.where(df['pt_vol'] > (df['volume'] * 0.05 + 1), 1, 0)
    else:
        df['PT_Spike'] = 0

    # --- 3. TARGETS ---
    look_ahead = 3
    future_high = df['high'].shift(-1).rolling(look_ahead).max().shift(-(look_ahead-1))
    df['Target_High'] = (future_high - df['close']) / df['close']
    
    future_close = df['close'].shift(-3)
    df['Is_Safe_T3'] = np.where(future_close > df['close'], 1, 0)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna(subset=['RSI', 'Target_High', 'Is_Safe_T3'])

def get_ai_predictions(df):
    df = add_features(df)
    
    # 🧠 FEATURES LIST
    features = ['RSI', 'Volatility', 'Season_Score', 'Trend_Score', 'NN_Accumulation', 'TD_Net_Vol', 'PT_Spike']
    
    train_df = df.dropna(subset=features + ['Target_High', 'Is_Safe_T3'])
    
    # Nếu dữ liệu quá ít (<30 dòng), bỏ qua để tránh nhiễu
    if len(train_df) < 30: return 0, 0, 0
    
    # 1. Train Regression (Dự báo Lãi/Lỗ)
    reg_model = RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=42, n_jobs=-1)
    
    # 🔇 FIX WARNING 1: Dùng .values.ravel() để chuyển thành mảng 1 chiều
    reg_model.fit(train_df[features], train_df['Target_High'].values.ravel())
    
    # 2. Train Classifier (Dự báo Sống/Chết)
    unique_classes = train_df['Is_Safe_T3'].unique()
    
    if len(unique_classes) < 2:
        # 🛡️ TRƯỜNG HỢP ĐẶC BIỆT: Chỉ có 1 loại kết quả trong lịch sử
        if unique_classes[0] == 0:
            survival_prob = 0.0 # Lịch sử toàn chết -> Dự báo chết (0%)
        else:
            survival_prob = 1.0 # Lịch sử toàn sống -> Dự báo sống (100%)
    else:
        # Trường hợp bình thường (Có cả sống và chết)
        clf_model = RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=42, n_jobs=-1)
        
        # 🔇 FIX WARNING 2: Dùng .values.ravel() ở đây nữa
        clf_model.fit(train_df[features], train_df['Is_Safe_T3'].values.ravel())
        
        last_row = df.iloc[[-1]][features]
        # Lấy xác suất của Class 1 (Sống sót)
        survival_prob = clf_model.predict_proba(last_row)[0][1]

    # Predict Gain
    last_row = df.iloc[[-1]][features]
    pred_gain = reg_model.predict(last_row)[0]
    
    # Return features info
    nn_accum = last_row.iloc[0]['NN_Accumulation']
    
    return pred_gain, survival_prob, nn_accum

def load_strategy_map():
    if os.path.exists(STRATEGY_FILE):
        df = pd.read_csv(STRATEGY_FILE)
        return df.set_index('Symbol').T.to_dict()
    return {}

def run_smart_money_scan():
    print(f"\n🔭 SMART MONEY SCAN (Full Market) - {datetime.now().strftime('%Y-%m-%d')}")
    
    strat_map = load_strategy_map()
    files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
    print(f"   📋 Scanning {len(files)} files...")
    
    recommendations = []
    
    for i, file_path in enumerate(files):
        symbol = os.path.basename(file_path).replace(".csv", "").upper()
        if "tudoanh" in symbol.lower() or "putthrough" in symbol.lower(): continue
        
        if i % 50 == 0: print(f"      Scanning {i}/{len(files)}...", end='\r')
        
        # Strategy Info
        if symbol in strat_map:
            strategy_mode = strat_map[symbol]['Best_Strategy']
            hist_winrate = strat_map[symbol]['WinRate']
            is_optimized = True
        else:
            strategy_mode = 'DYNAMIC'
            hist_winrate = 0.0
            is_optimized = False
            
        df = load_stock_data(file_path)
        if df is None: continue
        
        # Predict
        pred_gain, survival_prob, nn_accum = get_ai_predictions(df)
        current_price = df.iloc[-1]['close']
        
        # BUY FILTER
        if pred_gain > MIN_AI_PROFIT and survival_prob >= MIN_SURVIVAL_PROB:
            
            # Additional Check: If NN is dumping massively, AI should be extra sure (Survival > 70%)
            if nn_accum < -100000 and survival_prob < 0.70:
                continue 

            if strategy_mode == 'FIXED':
                target = current_price * 1.15
                stop = current_price * 0.93
            else:
                target = current_price * (1 + pred_gain)
                stop = current_price * (1 - (pred_gain/2)) # Simple dynamic stop
            
            recommendations.append({
                'Symbol': symbol,
                'Price': current_price,
                'Strategy': strategy_mode,
                'Hist_WR': hist_winrate,
                'Survival': survival_prob,
                'AI_Gain': pred_gain,
                'NN_Flow': nn_accum, # Show what foreigners are doing
                'Optimized': is_optimized
            })

    # Sort by Safety
    recommendations.sort(key=lambda x: (x['Survival'], x['Hist_WR']), reverse=True)
    
    print(f"\n\n📢 FOUND {len(recommendations)} OPPORTUNITIES:\n")
    
    print(f"{'SYMBOL':<6} | {'PRICE':<8} | {'SURVIVAL':<8} | {'AI GAIN':<8} | {'NN FLOW (20d)':<15} | {'TYPE'}")
    print("-" * 90)
    
    for rec in recommendations:
        type_str = "✅ PRO" if rec['Optimized'] else "⚠️ NEW"
        star = "⭐" if (rec['Survival'] > 0.8 and rec['NN_Flow'] > 0) else "  "
        
        # Color code NN Flow
        nn_str = f"{rec['NN_Flow']:,.0f}"
        
        print(f"{star} {rec['Symbol']:<5} | {rec['Price']:<8.1f} | {rec['Survival']*100:<8.1f} | {rec['AI_Gain']*100:<8.1f} | {nn_str:<15} | {type_str}")
        
    print("-" * 90)
    print("NN FLOW: Positive (+) means Foreigners are accumulating. Negative (-) means dumping.")
    print("⭐: Strong Pick (High Safety + Foreigners Buying)")

if __name__ == "__main__":
    run_smart_money_scan()