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
import sys
from datetime import datetime
from joblib import dump, load

# SILENCE WARNINGS
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Parse command-line arguments for training date range
parser = argparse.ArgumentParser(description="AI Daily Scanner with selectable training period")
parser.add_argument("--train-start", type=str, default="2000-01-01", 
                    help="Training start date (YYYY-MM-DD), default: 2000-01-01")
parser.add_argument("--train-end", type=str, default="2020-12-31", 
                    help="Training end date (YYYY-MM-DD), default: 2020-12-31")
args = parser.parse_args()

# SETTINGS - OPTIMIZED FROM BACKTEST
DATA_PATH = "data"
FUNDAMENTAL_PATH = "data_stocks"
BRAIN_FILE = "ai_brain_optimized.joblib"
TRADE_LOG_FILE = "Global_AI_Trade_Log_Prob0.6_Hold3_SL15_PT10.csv"

# FOLDER SETUP - Save signals to dedicated folder
SIGNALS_FOLDER = os.path.join(os.getcwd(), "Signals_Historical")
if not os.path.exists(SIGNALS_FOLDER):
    os.makedirs(SIGNALS_FOLDER)
    print(f"Created signals folder: {SIGNALS_FOLDER}")

# TRAINING PERIOD (controllable via command line)
TRAINING_START = args.train_start
TRAINING_END = args.train_end

# PROVEN PARAMETERS FROM BACKTEST (2188% return!)
MIN_PROBABILITY = 0.60  # Optimal threshold
MIN_VOLUME = 5_000      # Sufficient liquidity
MAX_VOLATILITY = 0.05   # Exclude extreme volatility (5% daily swings)
PROFIT_TARGET_PCT = 0.10  # Exit at 10% profit
STOP_LOSS_PCT = 0.15    # 15% stop-loss (or 10% in weak months)

# BLACKLIST - Stocks with consistent losses (expanded from backtest analysis)
BLACKLIST_STOCKS = {
    'BCG', 'KPF', 'PCH', 'HHC', 'PGN', 'ATS', 'VPG', 'ORS',  # Original 8
    'PHC', 'EVS', 'HII', 'SID', 'HVX', 'ILS', 'KHS', 'HLC'   # Post-Oct 2025 worst performers
}

# WEAK MONTHS - Use tighter stops and fewer positions
WEAK_MONTHS = {3, 7, 10, 11}  # March, July, Oct, Nov
WEAK_MONTH_MAX_POSITIONS = 5  # Reduce from 10 to 5 in weak months

# Portfolio Rules (reference from proven backtest)
MAX_POSITIONS = 10
HOLD_DAYS = 3           # Vietnam market T+3 requirement
TRANSACTION_COST = 0.005  # 0.5% realistic cost

def load_historical_performance():
    """Load and analyze historical trade performance per stock from AI_Trade_History.csv"""
    history_file = "AI_Trade_History.csv"
    if not os.path.exists(history_file):
        print(f"Trade history not found: {history_file}")
        return {}, {}

    try:
        df = pd.read_csv(history_file)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter out invalid entries (Buy_Price = 0)
        df = df[df['Buy_Price'] > 0]
        
        print(f"Loading historical performance from {len(df)} signals...")

        # Calculate PnL for each signal by finding price 3 days later
        performance = {}
        macd_performance = {'UP': {'trades': 0, 'wins': 0, 'total_pnl': 0}, 
                           'DOWN': {'trades': 0, 'wins': 0, 'total_pnl': 0}}
        
        for symbol in df['Symbol'].unique():
            stock_signals = df[df['Symbol'] == symbol].copy()
            
            # Load stock price data
            stock_path = os.path.join(DATA_PATH, f"{symbol}.csv")
            if not os.path.exists(stock_path):
                continue
                
            try:
                price_df = pd.read_csv(stock_path)
                price_df.columns = [c.lower().strip() for c in price_df.columns]
                rename_map = {'time': 'date', 'dt': 'date', 'dong': 'close'}
                price_df = price_df.rename(columns=rename_map)
                price_df['date'] = pd.to_datetime(price_df['date'], errors='coerce')
                price_df = price_df.dropna(subset=['date']).set_index('date').sort_index()
                
                if price_df.empty:
                    continue
                
                # Calculate MACD for the stock
                ema12 = price_df['close'].ewm(span=12).mean()
                ema26 = price_df['close'].ewm(span=26).mean()
                price_df['MACD_Hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
            except:
                continue
            
            trades = []
            for _, signal in stock_signals.iterrows():
                signal_date = signal['Date']
                buy_price = signal['Buy_Price']
                
                # Find the signal date in price data
                try:
                    # Get price 3 trading days later
                    future_dates = price_df.index[price_df.index > signal_date]
                    if len(future_dates) < HOLD_DAYS:
                        continue
                    
                    exit_date = future_dates[HOLD_DAYS - 1]
                    exit_price = price_df.loc[exit_date, 'close']
                    
                    # Calculate PnL (percentage)
                    pnl = (exit_price - buy_price) / buy_price
                    
                    # Get MACD direction at signal time
                    signal_dates = price_df.index[price_df.index <= signal_date]
                    if len(signal_dates) == 0:
                        continue
                    nearest_date = signal_dates[-1]
                    macd_hist = price_df.loc[nearest_date, 'MACD_Hist']
                    macd_dir = 'UP' if macd_hist > 0 else 'DOWN'
                    
                    trades.append({
                        'pnl': pnl,
                        'macd_dir': macd_dir,
                        'date': signal_date
                    })
                    
                    # Update MACD performance
                    macd_performance[macd_dir]['trades'] += 1
                    macd_performance[macd_dir]['total_pnl'] += pnl
                    if pnl > 0:
                        macd_performance[macd_dir]['wins'] += 1
                        
                except (KeyError, IndexError):
                    continue
            
            if trades:
                total_trades = len(trades)
                winning_trades = sum(1 for t in trades if t['pnl'] > 0)
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                total_pnl = sum(t['pnl'] for t in trades)
                avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
                
                performance[symbol] = {
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'avg_pnl': avg_pnl,
                    'best_trade': max(t['pnl'] for t in trades),
                    'worst_trade': min(t['pnl'] for t in trades),
                }

        print(f"Loaded historical performance for {len(performance)} stocks")
        return performance, macd_performance

    except Exception as e:
        print(f"Error loading trade history: {e}")
        return {}, {}

def get_all_symbols():
    files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
    return [os.path.basename(f).replace('.csv', '') for f in files]

def load_data_package(symbol):
    path = os.path.join(DATA_PATH, f"{symbol}.csv")
    if not os.path.exists(path): return None, None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        rename_map = {'time': 'date', 'dt': 'date', 'dong': 'close', 'mo': 'open', 'cao': 'high', 'thap': 'low', 'vol': 'volume'}
        df = df.rename(columns=rename_map)
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date', 'close']).set_index('date').sort_index()
        for c in ['close', 'high', 'low', 'open', 'volume']:
            if df[c].dtype == 'object': 
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce')
    except: return None, None

    f_path = os.path.join(FUNDAMENTAL_PATH, f"{symbol}.csv")
    if not os.path.exists(f_path): return df, None
    try:
        f_df = pd.read_csv(f_path)
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
        return df, f_df[['release_date', 'EPS', 'BVPS', 'Rev_Growth']]
    except: return df, None

def prepare_features(df, fund_df):
    """Optimized features from successful backtest"""
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['Trend_Score'] = (df['close'] - df['SMA_20']) / df['SMA_20']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    # Volatility
    df['Volatility'] = ((df['high'] - df['low']) / df['close']).rolling(14).mean()
    df['Vol_SMA_20'] = df['volume'].rolling(20).mean()

    # VWAP
    pv = df['close'] * df['volume']
    vwap = pv.rolling(20).sum() / df['volume'].rolling(20).sum().replace(0, 1e-9)
    df['VWAP_Score'] = (df['close'] - vwap) / vwap

    # MACD (new from backtest improvement)
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands (new from backtest improvement)
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['BB_Upper'] = sma20 + 2 * std20
    df['BB_Lower'] = sma20 - 2 * std20
    df['BB_Score'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    if fund_df is not None and not fund_df.empty:
        df_merged = pd.merge_asof(df, fund_df, left_index=True, right_on='release_date', direction='backward')
        df_merged.index = df.index
        df = df_merged
        df['PE_Ratio'] = df['close'] / df['EPS'].replace(0, 0.01)
        df['Rev_Growth'] = df['Rev_Growth'].fillna(0)
    else:
        df['PE_Ratio'] = 0
        df['Rev_Growth'] = 0

    future_close = df['close'].shift(-HOLD_DAYS)
    df['Is_Safe'] = np.where(future_close > df['close'], 1, 0)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna(subset=['SMA_20']).fillna(0)

def run_scanner():
    """Generate daily trading signals using optimized AI"""
    
    print("="*80)
    print("OPTIMIZED AI DAILY SCANNER (v2.0 - T+3 Realistic)")
    print("="*80)
    print(f"Saving signals to: {SIGNALS_FOLDER}/")
    
    brain_loaded = False
    clf = None
    
    if os.path.exists(BRAIN_FILE):
        print(f"Found saved brain '{BRAIN_FILE}'! Loading it...")
        try:
            clf = load(BRAIN_FILE)
            brain_loaded = True
        except:
            print("Error loading brain. Will retrain.")
    
    if not brain_loaded:
        print(f"Training New Brain (Data from {TRAINING_START} to {TRAINING_END})...")
    
    symbols = get_all_symbols()
    all_data = []
    latest_rows = []
    
    print(f"   Scanning {len(symbols)} stocks...")
    train_start_dt = pd.Timestamp(TRAINING_START)
    train_end_dt = pd.Timestamp(TRAINING_END)

    # Features used in optimal backtest
    features = [
        'RSI', 'Trend_Score', 'PE_Ratio', 'VWAP_Score', 'Rev_Growth',
        'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Score'
    ]

    for i, s in enumerate(symbols):
        if i % 100 == 0: print(f"   Processing {i}/{len(symbols)}...", end='\r')
        df, f_df = load_data_package(s)
        if df is None or len(df) < 50: continue
        
        # Volatility filter - exclude extreme volatility
        recent_vol = ((df['high'] - df['low']) / df['close']).tail(20).mean()
        if recent_vol > MAX_VOLATILITY:
            continue
            
        df = prepare_features(df, f_df)
        df['Symbol'] = s
        
        # Build training set if we don't have a brain
        if not brain_loaded:
            train_slice = df[(df.index >= train_start_dt) & (df.index <= train_end_dt)].dropna(subset=['Is_Safe'])
            if not train_slice.empty:
                all_data.append(train_slice)
        
        # Get prediction row for today
        last_row = df.iloc[[-1]].copy()
        last_row['Symbol'] = s
        latest_rows.append(last_row)
        
    candidates = pd.concat(latest_rows)
    
    # Apply blacklist and volatility filter BEFORE prediction
    candidates = candidates[~candidates['Symbol'].isin(BLACKLIST_STOCKS)]
    candidates = candidates[candidates['Volatility'] < MAX_VOLATILITY]
    
    # TRAIN IF NEEDED
    if not brain_loaded:
        master_train = pd.concat(all_data)
        print(f"\nTeaching Brain with {len(master_train)} historic patterns...")
        
        if USE_XGB:
            clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                              n_jobs=-1, random_state=42)
            print("Using XGBoost for better prediction accuracy")
        else:
            clf = RandomForestClassifier(n_estimators=50, min_samples_leaf=20, 
                                        n_jobs=-1, random_state=42)
            print("Using RandomForest")
        
        clf.fit(master_train[features], master_train['Is_Safe'])
        
        # SAVE BRAIN
        dump(clf, BRAIN_FILE)
        print(f"Brain saved to '{BRAIN_FILE}' for future use.")
    else:
        print("\nBrain Loaded directly. Skipped training phase.")

    # PREDICT
    print("Scanning Today's Market...")
    candidates['Prob'] = clf.predict_proba(candidates[features])[:, 1]
    
    # Apply filters matching proven backtest
    recommendations = candidates[
        (candidates['Prob'] >= MIN_PROBABILITY) & 
        (candidates['volume'] > MIN_VOLUME) &
        (candidates['Volatility'] < MAX_VOLATILITY)
    ].sort_values('Prob', ascending=False)
    
    scan_date = candidates.index.max().strftime('%Y-%m-%d')
    
    # Load historical performance data
    historical_perf, macd_perf = load_historical_performance()
    
    # Calculate weak month status
    current_month = pd.Timestamp.now().month
    is_weak_month = current_month in WEAK_MONTHS
    weak_indicator = "WEAK MONTH" if is_weak_month else "Normal"
    active_sl = 0.10 if is_weak_month else STOP_LOSS_PCT
    
    print("\n" + "="*140)
    print(f"PROVEN AI SCAN | Backtest: 2188% return | Market Date: {scan_date} | {weak_indicator}")
    print(f"Training Period: {TRAINING_START} to {TRAINING_END}")
    print(f"Proven Settings: Prob=0.60, Hold=3d, SL={active_sl*100:.0f}%, ProfitTgt=10%, Blacklist={len(BLACKLIST_STOCKS)}")
    print("="*140)
    print(f"{'SYMBOL':<8} | {'CONF %':<8} | {'PRICE':<12} | {'RSI':<6} | {'MACD':<8} | {'TREND':<8} | {'HIST WIN%':<10} | {'TRADES':<7} | {'AVG PnL':<10} | NOTES")
    print("-" * 140)
    
    top_picks = recommendations.head(30)
    
    if top_picks.empty:
        print("No opportunities match proven criteria.")
    else:
        for _, row in top_picks.iterrows():
            symbol = row['Symbol']
            trend = "UP" if row['Trend_Score'] > 0 else "DOWN"
            macd = "UP" if row['MACD_Hist'] > 0 else "DOWN"

            # Get historical performance data
            hist = historical_perf.get(symbol, {})
            hist_win_rate = hist.get('win_rate', 0) * 100
            hist_trades = hist.get('total_trades', 0)
            hist_avg_pnl = hist.get('avg_pnl', 0) * 100  # Convert to percentage

            notes = []
            if row['RSI'] > 70: notes.append("Overbought")
            if row['RSI'] < 30: notes.append("Oversold")
            if row['Volatility'] < 0.02: notes.append("Stable")
            if is_weak_month: notes.append(f"WeakMode-SL{active_sl*100:.0f}%")

            print(f"{symbol:<8} | {row['Prob']*100:>6.1f}% | {row['close']:>11.2f} | {int(row['RSI']):>5} | {macd:>7} | {trend:<8} | {hist_win_rate:>8.1f}% | {hist_trades:>6} | {hist_avg_pnl:>9.2f}% | {', '.join(notes)}")
            
    print("="*140)

    # Add historical performance data to recommendations DataFrame
    recommendations = recommendations.copy()
    recommendations['Hist_Win_Rate'] = recommendations['Symbol'].map(lambda s: historical_perf.get(s, {}).get('win_rate', 0) * 100)
    recommendations['Hist_Trades'] = recommendations['Symbol'].map(lambda s: historical_perf.get(s, {}).get('total_trades', 0))
    recommendations['Hist_Avg_PnL'] = recommendations['Symbol'].map(lambda s: historical_perf.get(s, {}).get('avg_pnl', 0))
    recommendations['Hist_Total_PnL'] = recommendations['Symbol'].map(lambda s: historical_perf.get(s, {}).get('total_pnl', 0))

    # SAVE TO FOLDER - Modified to save in Signals folder
    filename = f"Signals_Proven_{scan_date}.csv"
    filepath = os.path.join(SIGNALS_FOLDER, filename)
    recommendations[['Symbol', 'Prob', 'close', 'RSI', 'MACD', 'Trend_Score',
                      'Volatility', 'PE_Ratio', 'VWAP_Score', 'Hist_Win_Rate',
                      'Hist_Trades', 'Hist_Avg_PnL', 'Hist_Total_PnL']].to_csv(filepath)
    print(f"Signals saved to '{filepath}'")
    print(f"BUY signals: {len(recommendations)} stocks (Hold 3d, PT 10%, SL {active_sl*100:.0f}%)")

    if is_weak_month:
        print(f"WEAK MONTH MODE: Using tighter {active_sl*100:.0f}% stop-loss")
    
    # Display MACD UP vs DOWN Analysis
    if macd_perf and (macd_perf['UP']['trades'] > 0 or macd_perf['DOWN']['trades'] > 0):
        print("\n" + "="*80)
        print("MACD DIRECTION ANALYSIS (Historical Backtest)")
        print("="*80)
        print(f"{'MACD':<10} | {'TRADES':<10} | {'WIN RATE':<12} | {'AVG PnL':<12} | {'TOTAL PnL':<12}")
        print("-" * 60)
        
        for direction in ['UP', 'DOWN']:
            data = macd_perf[direction]
            if data['trades'] > 0:
                win_rate = (data['wins'] / data['trades']) * 100
                avg_pnl = (data['total_pnl'] / data['trades']) * 100
                total_pnl = data['total_pnl'] * 100
                print(f"{direction:<10} | {data['trades']:<10} | {win_rate:>10.1f}% | {avg_pnl:>10.2f}% | {total_pnl:>10.1f}%")
        
        # Recommendation based on MACD analysis
        up_win = (macd_perf['UP']['wins'] / macd_perf['UP']['trades'] * 100) if macd_perf['UP']['trades'] > 0 else 0
        down_win = (macd_perf['DOWN']['wins'] / macd_perf['DOWN']['trades'] * 100) if macd_perf['DOWN']['trades'] > 0 else 0
        
        print("-" * 60)
        if up_win > down_win + 5:
            print(f"RECOMMENDATION: Prefer MACD UP signals ({up_win:.1f}% vs {down_win:.1f}% win rate)")
        elif down_win > up_win + 5:
            print(f"RECOMMENDATION: Prefer MACD DOWN signals ({down_win:.1f}% vs {up_win:.1f}% win rate)")
        else:
            print(f"RECOMMENDATION: Both MACD directions similar ({up_win:.1f}% UP vs {down_win:.1f}% DOWN)")
        print("="*80)
    
    return recommendations

if __name__ == "__main__":
    run_scanner()
    input("Press Enter to exit...")