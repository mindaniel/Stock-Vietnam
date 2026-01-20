import pandas as pd
import numpy as np
import glob
import os
import warnings
from datetime import datetime

# 🛑 AGGRESSIVE WARNING MUTE
# This forces Python to ignore the specific sklearn feature name warnings
import os
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# ==========================================
# ⚙️ SETTINGS
# ==========================================
DATA_PATH = r"C:\Users\Daziel Brilliant\Downloads\Stock-Vietnam-main\data"
STRATEGY_FILE = "best_strategies.csv"
PORTFOLIO_FILE = "portfolio.csv"

# 💰 POT MANAGEMENT
TOTAL_CAPITAL = 100_000_000
MAX_POTS = 20
POT_SIZE = TOTAL_CAPITAL / MAX_POTS

# 🛡️ SAFETY FILTER
MIN_SURVIVAL_PROB = 0.60  # AI must be 60% sure T+3 is safe
MIN_AI_PROFIT = 0.015     # AI must predict at least 1.5% gain

FIXED_TP = 0.15
FIXED_SL = 0.07

def load_stock_data(symbol):
    search_path = os.path.join(DATA_PATH, f"{symbol}.csv")
    files = glob.glob(search_path)
    if not files: return None
    
    try:
        df = pd.read_csv(files[0])
        df.columns = [c.lower().strip() for c in df.columns]
        rename_map = {'time': 'date', 'dong': 'close', 'mo': 'open', 'cao': 'high', 'thap': 'low', 'vol': 'volume'}
        df = df.rename(columns=rename_map)
        
        cols = ['open', 'high', 'low', 'close']
        for c in cols:
            if c in df.columns and df[c].dtype == 'object':
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce')
        
        df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True, errors='coerce')
        df = df.dropna(subset=['date', 'close']).set_index('date').sort_index()
        return df
    except: return None

def add_features(df):
    df['Returns'] = df['close'].pct_change()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['Volatility'] = ((df['high'] - df['low']) / df['close']).rolling(14).mean()
    
    df['Month'] = df.index.month
    monthly_seasonality = df.groupby('Month')['Returns'].mean()
    df['Season_Score'] = df['Month'].map(monthly_seasonality)
    
    # Targets
    look_ahead = 3
    future_high = df['high'].shift(-1).rolling(look_ahead).max().shift(-(look_ahead-1))
    future_low = df['low'].shift(-1).rolling(look_ahead).min().shift(-(look_ahead-1))
    df['Target_High'] = (future_high - df['close']) / df['close']
    df['Target_Low'] = (future_low - df['close']) / df['close']
    
    # Survival Target (T+3 > Today)
    future_close = df['close'].shift(-3)
    df['Is_Safe_T3'] = np.where(future_close > df['close'], 1, 0)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def get_ai_predictions(df):
    # Ensure inputs are always DataFrames to fix warnings
    df = add_features(df)
    features = ['RSI', 'Volatility', 'Season_Score']
    
    train_df = df.dropna(subset=['RSI', 'Target_High', 'Target_Low', 'Season_Score', 'Is_Safe_T3'])
    if len(train_df) < 50: return 0, 0, 0
    
    # 🧠 BRAIN 1: Regressor
    reg_model = RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=42, n_jobs=-1)
    reg_model.fit(train_df[features], train_df[['Target_High', 'Target_Low']])
    
    # 🧠 BRAIN 2: Classifier
    clf_model = RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=42, n_jobs=-1)
    clf_model.fit(train_df[features], train_df['Is_Safe_T3'])
    
    # Predict on LAST ROW (Keep as DataFrame!)
    last_row = df.iloc[[-1]][features]
    if last_row.isnull().values.any(): return 0, 0, 0
    
    # Prediction
    reg_preds = reg_model.predict(last_row)
    pred_gain = reg_preds[0, 0]
    pred_risk = np.abs(reg_preds[0, 1])
    
    survival_prob = clf_model.predict_proba(last_row)[0][1]
    
    return pred_gain, pred_risk, survival_prob

def load_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        return pd.DataFrame(columns=['Symbol', 'BuyDate', 'BuyPrice', 'Quantity', 'Strategy', 'TargetPrice', 'StopPrice', 'Status'])
    return pd.read_csv(PORTFOLIO_FILE)

def run_full_scan():
    print(f"\n🔭 FULL MARKET SCAN (ALL STOCKS) - {datetime.now().strftime('%Y-%m-%d')}")
    print("   This may take a few minutes...")
    
    # Check Money
    pf = load_portfolio()
    active_pots = len(pf[pf['Status'] == 'HELD'])
    free_pots = MAX_POTS - active_pots
    print(f"   💰 Free Pots Available: {free_pots} / {MAX_POTS}")
    
    if free_pots == 0:
        print("   🛑 No free pots. Sell existing positions first.")
        return

    # Load Candidates
    if not os.path.exists(STRATEGY_FILE):
        print("   ❌ Strategy file missing. Run optimizer first.")
        return
        
    strat_df = pd.read_csv(STRATEGY_FILE)
    
    # 🚨 NO LIMITS: We scan EVERYTHING in the file
    candidates = strat_df 
    print(f"   📋 Scanning {len(candidates)} stocks from your Strategy List...")
    
    recommendations = []
    
    for i, (_, row) in enumerate(candidates.iterrows()):
        symbol = row['Symbol']
        strategy_mode = row['Best_Strategy']
        hist_winrate = row['WinRate'] # Get Historical WinRate
        
        # Progress Bar
        if i % 50 == 0: print(f"      Checking {i}/{len(candidates)}...", end='\r')
        
        df = load_stock_data(symbol)
        if df is None: continue
        
        # Get AI Prediction
        pred_gain, pred_risk, survival_prob = get_ai_predictions(df)
        
        current_price = df.iloc[-1]['close']
        
        # --- BUY FILTER ---
        # 1. AI Profit Potential
        # 2. Survival Safety
        if pred_gain > MIN_AI_PROFIT and pred_gain > (pred_risk * 2.0) and survival_prob >= MIN_SURVIVAL_PROB:
            
            qty = int((POT_SIZE / current_price) // 100) * 100
            if qty < 100: continue
            
            # Set Targets
            if strategy_mode == 'FIXED':
                target = current_price * (1 + FIXED_TP)
                stop = current_price * (1 - FIXED_SL)
            else:
                target = current_price * (1 + pred_gain)
                stop = current_price * (1 - pred_risk)
            
            recommendations.append({
                'Symbol': symbol,
                'Price': current_price,
                'Strategy': strategy_mode,
                'Hist_WR': hist_winrate,    # Historic Winrate from CSV
                'Survival': survival_prob,  # Today's AI Confidence
                'AI_Gain': pred_gain,
                'Target': target,
                'Stop': stop
            })

    # Sort: Safest First (Survival), then by Historical WinRate
    recommendations.sort(key=lambda x: (x['Survival'], x['Hist_WR']), reverse=True)
    
    print(f"\n\n📢 FOUND {len(recommendations)} BUY OPPORTUNITIES:\n")
    
    # Print Table
    print(f"{'SYMBOL':<8} | {'PRICE':<8} | {'STRAT':<8} | {'HIST WR%':<8} | {'SURVIVAL%':<10} | {'AI GAIN%':<8} | {'TARGET':<8} | {'STOP'}")
    print("-" * 100)
    
    for rec in recommendations:
        # Highlight Top Picks (High Survival AND High Historical Winrate)
        prefix = "⭐" if (rec['Survival'] > 0.75 and rec['Hist_WR'] > 40) else "  "
        
        print(f"{prefix} {rec['Symbol']:<5} | {rec['Price']:<8.1f} | {rec['Strategy']:<8} | {rec['Hist_WR']:<8.1f} | {rec['Survival']*100:<9.1f} | {rec['AI_Gain']*100:<8.1f} | {rec['Target']:<8.1f} | {rec['Stop']:<8.1f}")
        
    print("-" * 100)
    print("NOTE: '⭐' = Strong Pick (Survival > 75% AND History WinRate > 40%)")

# === RUN ===
run_full_scan()