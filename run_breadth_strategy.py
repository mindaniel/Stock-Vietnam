import pandas as pd
import numpy as np
import os
import glob

def create_combined_dataset():
    """Create a combined dataset with all stocks and their industries"""
    
    # Load industry information
    print("Loading industry information...")
    industry_df = pd.read_excel('vndirect_listing.xlsx')
    industry_df = industry_df[['code', 'IndustryEN']].rename(columns={'code': 'ticker', 'IndustryEN': 'industry'})
    
    # Get all stock CSV files
    stock_files = glob.glob('data/*.csv')
    print(f"Found {len(stock_files)} stock files")
    
    # Combine all stock data
    all_data = []
    for file_path in stock_files:
        ticker = os.path.basename(file_path).replace('.csv', '')
        try:
            df = pd.read_csv(file_path)
            # Rename columns to match expected format
            df = df.rename(columns={'time': 'date', 'close': 'close'})
            df['ticker'] = ticker
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined dataset has {len(combined_df)} rows")
    
    # Merge with industry information
    combined_df = combined_df.merge(industry_df, on='ticker', how='left')
    
    # Remove rows without industry information
    combined_df = combined_df.dropna(subset=['industry'])
    print(f"Final dataset has {len(combined_df)} rows with industry info")
    
    # Save the combined dataset
    combined_df.to_csv('all_stocks_with_industries.csv', index=False)
    print("Saved combined dataset to all_stocks_with_industries.csv")
    
    return 'all_stocks_with_industries.csv'

def calculate_industry_breadth(stock_data_path, window=252, threshold=-1.5):
    # 1. Load the all-stocks data
    try:
        df = pd.read_csv(stock_data_path)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values(['ticker', 'date'])
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Calculate SMA, Std Dev, and Z-Score grouped by EACH Ticker
    # We use transform so the new columns match the length of our original dataframe
    df['SMA'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=window).mean())
    df['Std_Dev'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=window).std())
    df['Z_Score'] = (df['close'] - df['SMA']) / df['Std_Dev']

    # 3. Flag the stocks that are dipping (Z-Score < threshold)
    # 1 means it is dipping, 0 means it is fine
    df['Is_Dipping'] = np.where(df['Z_Score'] < threshold, 1, 0)

    # 4. Calculate the Breadth Ratio per Industry for each Date
    # By taking the mean() of a 1 or 0 column, we get the exact percentage (ratio) of dipping stocks
    breadth_df = df.groupby(['date', 'industry'])['Is_Dipping'].mean().reset_index()
    
    # Convert the decimal to a clean percentage (e.g., 0.75 becomes 75.0)
    breadth_df['Breadth_Ratio_%'] = breadth_df['Is_Dipping'] * 100
    breadth_df = breadth_df.drop(columns=['Is_Dipping'])

    # 5. Filter for the most recent available date to see the results
    target_date = breadth_df['date'].max()
    recent_breadth = breadth_df[breadth_df['date'] == target_date].sort_values(by='Breadth_Ratio_%', ascending=False)
    
    print(f"\n--- Industry Breadth Ratios for {target_date.date()} ---")
    print("Higher percentage = More stocks in that industry are severely oversold.")
    print(recent_breadth.head(10)) # Show the top 10 most beaten-down industries

# Run the process
print("Creating combined dataset...")
data_file = create_combined_dataset()

print("\nRunning breadth strategy...")
calculate_industry_breadth(data_file, window=50, threshold=-1.5)