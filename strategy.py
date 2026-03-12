import pandas as pd
import numpy as np

def analyze_sector_rotation(file_path, window=50, deviation_threshold=-2):
    # 1. Load the CSV data
    try:
        df = pd.read_csv(file_path)
        # Ensure we have a datetime index and sort from oldest to newest
        df['Date'] = pd.to_datetime(df['date'])
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Calculate Simple Moving Average (SMA) and Standard Deviation
    # We use close price for our calculations
    df['SMA'] = df['close'].rolling(window=window).mean()
    df['Std_Dev'] = df['close'].rolling(window=window).std()

    # 3. Calculate the Z-Score (how far the price is from the average)
    df['Z_Score'] = (df['close'] - df['SMA']) / df['Std_Dev']

    # 4. Generate Signals
    # If the Z-Score drops below our threshold, it flags as "Oversold" (Time to buy?)
    df['Signal'] = np.where(df['Z_Score'] < deviation_threshold, 'Oversold (Buy Signal)', 'Neutral')
    
    # Show the most recent 10 days of data to see the current trend
    print(f"\n--- Recent Signals for {file_path} ---")
    print(df[['close', 'SMA', 'Z_Score', 'Signal']].tail(10))

# Run the function on your VNINDEX data
# Note: Adjust 'VNINDEX.csv' to the exact name of your file if it's inside a folder like 'data/'
analyze_sector_rotation('VNINDEX.csv', window=50, deviation_threshold=-1.5)