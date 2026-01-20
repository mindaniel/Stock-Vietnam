import pandas as pd
import numpy as np
import glob
import os
import warnings
from sklearn.ensemble import RandomForestRegressor

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
    
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['Trend_Score'] = (df['close'] - df['SMA_20']) / df['SMA_20']
    
    look_ahead = 3
    future_high = df['high'].shift(-1).rolling(look_ahead).max().shift(-(look_ahead-1))
    df['Target_High'] = (future_high - df['close']) / df['close']
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def visualize_consensus(symbol):
    print(f"\n⏳ Analyzing {symbol}...", end="\r")
    
    df = load_stock_data(symbol)
    if df is None: 
        print(f"❌ Error: Could not find data for symbol '{symbol}'")
        return

    df = add_features(df)
    features = ['RSI', 'Volatility', 'Season_Score', 'Trend_Score']
    train_df = df.dropna(subset=features + ['Target_High'])
    
    if len(train_df) < 50:
        print("⚠️ Not enough data to train AI.")
        return

    # 1. TRAIN 100 TREES
    n_trees = 100
    model = RandomForestRegressor(n_estimators=n_trees, random_state=42, n_jobs=-1)
    model.fit(train_df[features], train_df['Target_High'])
    
    last_row = df.iloc[[-1]][features]
    current_data = last_row.iloc[0]
    current_price = df.iloc[-1]['close']
    
    # 2. COLLECT VOTES
    votes = []
    for tree in model.estimators_:
        pred = tree.predict(last_row.values)[0]
        votes.append(pred)
    
    votes = np.array(votes)
    avg_vote = np.mean(votes)
    
    # 3. PRINT REPORT
    print(f"\n🏛️ HỘI ĐỒNG AI ĐANG HỌP VỀ: {symbol} (Price: {current_price:,.0f})")
    print("=" * 60)
    
    print(f"\n1️⃣ KẾT QUẢ BỎ PHIẾU (100 Giám khảo):")
    print(f"   Trung bình dự báo: {avg_vote*100:+.2f}%")
    
    positive_votes = np.sum(votes > 0.015)
    negative_votes = np.sum(votes < 0)
    neutral_votes = n_trees - positive_votes - negative_votes
    
    print(f"   🟢 Phe Bò (Tăng > 1.5%):  {positive_votes} phiếu")
    print(f"   🔴 Phe Gấu (Giảm):        {negative_votes} phiếu")
    print(f"   ⚪ Trung lập:             {neutral_votes} phiếu")
    
    total_len = 40
    p_len = int((positive_votes / 100) * total_len)
    n_len = int((negative_votes / 100) * total_len)
    neu_len = total_len - p_len - n_len
    print(f"   Sentiment: [{'='*p_len}{'-'*neu_len}{'x'*n_len}]")

    print(f"\n2️⃣ TIÊU CHÍ QUAN TRỌNG NHẤT (Feature Importance):")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(len(features)):
        feat_name = features[indices[i]]
        score = importances[indices[i]] * 100
        bar = "█" * int(score / 5)
        print(f"   {feat_name:<15} : {score:5.1f}% {bar}")

    print(f"\n3️⃣ GIẢ LẬP 'WHAT-IF' (Nếu RSI thay đổi?):")
    current_rsi = current_data['RSI']
    test_rsis = [current_rsi - 10, current_rsi, current_rsi + 10]
    
    print(f"   {'RSI GIẢ ĐỊNH':<15} | {'DỰ BÁO'}")
    print("   " + "-"*30)
    
    for rsi_val in test_rsis:
        fake_row = last_row.copy()
        fake_row['RSI'] = max(0, min(100, rsi_val))
        fake_pred = model.predict(fake_row)[0]
        marker = "◄ HIỆN TẠI" if rsi_val == current_rsi else ""
        print(f"   {rsi_val:<15.1f} | {fake_pred*100:+.2f}%  {marker}")

    print("\n🏁 KẾT LUẬN:")
    if positive_votes > 60: print("   ✅ MUA MẠNH (Đồng thuận cao).")
    elif avg_vote > 0.015: print("   ⚠️ MUA NHẸ (Còn tranh cãi).")
    else: print("   ⛔ QUAN SÁT (Chưa đủ hấp dẫn).")
    print("=" * 60)

# ==========================================
# 🎮 INTERACTIVE LOOP
# ==========================================
def main():
    print("🤖 AI COUNCIL TOOL ACTIVATED")
    print("This tool will summon 100 AI decision trees to vote on any stock.")
    
    while True:
        user_input = input("\n👉 Enter Stock Symbol (or 'exit' to stop): ").strip().upper()
        
        if user_input in ['EXIT', 'STOP', 'QUIT', 'N', 'NO']:
            print("👋 Shutting down AI Council. Good luck!")
            break
        
        if len(user_input) < 3:
            print("⚠️ Invalid symbol. Please try again.")
            continue
            
        visualize_consensus(user_input)

# Start the program
if __name__ == "__main__":
    main()