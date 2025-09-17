# src/data_loader.py
import pandas as pd

def load_candles_csv(path):
    """
    expected csv columns: ['timestamp','open','high','low','close','volume']
    timestamp must be parseable to pandas datetime
    """
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# Example usage:
# m1 = load_candles_csv('data/EURUSD_M1.csv')
# m15 = load_candles_csv('data/EURUSD_M15.csv')