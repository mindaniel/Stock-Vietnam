import pandas as pd
import numpy as np
import glob
import os
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# 🛑 SILENCE WARNINGS
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

# ==========================================
# ⚙️ SETTINGS
# ==========================================
DATA_PATH = r"C:\Users\Daziel Brilliant\Downloads\Stock-Vietnam-main\data"

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
    df['Target_High'] = (future_high - df['close']) / df['close']
    
    future_close = df['close'].shift(-3)
    df['Is_Safe_T3'] = np.where(future_close > df['close'], 1, 0)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def audit_stock(symbol, buy_price):
    df = load_stock_data(symbol)
    if df is None:
        print(f"❌ Could not find data for {symbol}")
        return

    df = add_features(df)
    features = ['RSI', 'Volatility', 'Season_Score']
    train_df = df.dropna(subset=features + ['Target_High', 'Is_Safe_T3'])
    
    if len(train_df) < 50: 
        print(f"⚠️ Not enough data for {symbol}")
        return

    # Train AI
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reg_model.fit(train_df[features], train_df['Target_High'])
    
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_model.fit(train_df[features], train_df['Is_Safe_T3'])
    
    # Predict
    last_row = df.iloc[[-1]][features]
    current_price = df.iloc[-1]['close']
    current_rsi = last_row.iloc[0]['RSI']
    
    pred_gain = reg_model.predict(last_row)[0]
    survival_prob = clf_model.predict_proba(last_row)[0][1]
    
    # Calculate current PnL
    pnl_pct = 0
    if buy_price > 0:
        pnl_pct = (current_price - buy_price) / buy_price * 100
    
    # --- DECISION LOGIC ---
    action = "HOLD"
    color = "\033[92m" # Green
    
    if survival_prob < 0.50:
        action = "SELL NOW"
        color = "\033[91m" # Red
    elif pred_gain < -0.015: # If AI predicts drop > 1.5%
        action = "SELL"
        color = "\033[91m" # Red
    elif current_rsi > 75:
        action = "TAKE PROFIT"
        color = "\033[93m" # Yellow
    elif survival_prob < 0.60:
        action = "WARNING"
        color = "\033[93m" # Yellow
    elif pnl_pct < -7:
        action = "STOP LOSS"
        color = "\033[91m" # Red

    reset = "\033[0m"
    pnl_str = f"{pnl_pct:+.1f}%"
    
    # Added PRED % column
    print(f"{symbol:<6} | {current_price:<8,.0f} | {pnl_str:<8} | {survival_prob*100:.0f}% Safe | {pred_gain*100:+.2f}% | {color}{action:<12}{reset}")

# ==========================================
# 🎮 INTERACTIVE LOOP
# ==========================================
def main():
    print("💼 PORTFOLIO DOCTOR (EXACT PERCENTAGE MODE)")
    print("Enter your stocks one by one. Type 'run' to start the analysis.")
    
    my_portfolio = []
    
    while True:
        print("\n---------------------------")
        symbol = input("👉 Enter Stock Symbol (or 'run'): ").strip().upper()
        
        if symbol in ['RUN', 'DONE', 'EXIT', 'START']:
            break
            
        if len(symbol) < 3:
            print("   ⚠️ Invalid symbol.")
            continue
            
        try:
            price_input = input(f"   💸 Bought {symbol} at price? (Press Enter if unknown): ").strip()
            if not price_input:
                price = 0
                my_portfolio.append({'Symbol': symbol, 'BuyPrice': price})
                print(f"   ✅ Added {symbol}")
                continue
            if not price_input:
                print("   ❌ Invalid price. Please enter a number.")
                continue
            price = float(price_input)
            my_portfolio.append({'Symbol': symbol, 'BuyPrice': price})
            print(f"   ✅ Added {symbol} @ {price:,.0f}")
        except ValueError:
            print("   ❌ Invalid price. Please enter a number.")
            
    if not my_portfolio:
        print("No stocks entered. Exiting.")
        return

    print("\n\n📊 GENERATING AUDIT REPORT...")
    # Updated Header
    print(f"{'SYMBOL':<6} | {'PRICE':<8} | {'CUR PnL':<8} | {'SAFETY':<9} | {'PRED %':<7} | {'ACTION'}")
    print("-" * 80)
    
    for item in my_portfolio:
        audit_stock(item['Symbol'], item['BuyPrice'])
        
    print("-" * 80)

if __name__ == "__main__":
    main()