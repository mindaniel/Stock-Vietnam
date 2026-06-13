import pandas as pd

def get_dipping_tickers(stock_data_path, target_date, target_industries, window=50, threshold=-1.5):
    print(f"Loading data from {stock_data_path}...")
    df = pd.read_csv(stock_data_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # 1. Calculate Z-Scores
    df = df.sort_values(['ticker', 'date'])
    df['SMA'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=window).mean())
    df['Std_Dev'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=window).std())
    df['Z_Score'] = (df['close'] - df['SMA']) / df['Std_Dev']
    
    # 2. Filter down to our target date and target industries
    target_date = pd.to_datetime(target_date)
    day_data = df[df['date'] == target_date]
    industry_data = day_data[day_data['industry'].isin(target_industries)]
    
    # 3. Keep only the stocks that are dipping below our threshold
    dipping_stocks = industry_data[industry_data['Z_Score'] < threshold]
    
    # 4. Print the results clearly
    for industry in target_industries:
        ind_stocks = dipping_stocks[dipping_stocks['industry'] == industry]
        print(f"\n--- Dipping Tickers in {industry} on {target_date.date()} ---")
        if ind_stocks.empty:
            print("None found.")
        else:
            # Sort from the deepest dip to the shallowest
            ind_stocks = ind_stocks.sort_values('Z_Score')
            print(ind_stocks[['ticker', 'close', 'Z_Score']].to_string(index=False))

# Run it for the date your breadth script flagged
get_dipping_tickers('all_stocks_with_industries.csv', '2026-03-12', ['Financial Services', 'Banks'])