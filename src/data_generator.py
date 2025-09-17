# src/data_generator.py
"""
Sample data generator for testing the ICT ML Trading Bot
Creates realistic OHLCV data for backtesting and model training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class SampleDataGenerator:
    """Generate realistic sample trading data"""
    
    def __init__(self, symbol='EURUSD', base_price=1.10000):
        self.symbol = symbol
        self.base_price = base_price
    
    def generate_tick_data(self, start_time, end_time, tick_interval_seconds=1):
        """Generate tick-level data with realistic price movements"""
        timestamps = []
        current_time = start_time
        
        while current_time <= end_time:
            timestamps.append(current_time)
            current_time += timedelta(seconds=tick_interval_seconds)
        
        # Generate price series with realistic characteristics
        n_ticks = len(timestamps)
        returns = np.random.normal(0, 0.00008, n_ticks)  # Small random returns
        
        # Add some trend and mean reversion
        trend = np.sin(np.linspace(0, 4*np.pi, n_ticks)) * 0.0005
        mean_reversion = np.random.normal(0, 0.00003, n_ticks)
        
        prices = [self.base_price]
        for i in range(1, n_ticks):
            new_price = prices[-1] + returns[i] + trend[i] + mean_reversion[i]
            prices.append(new_price)
        
        # Create bid/ask spread (typically 0.5-2.0 pips for EUR/USD)
        spread = np.random.uniform(0.00005, 0.00020, n_ticks)  # 0.5-2.0 pips
        
        tick_data = []
        for i, ts in enumerate(timestamps):
            mid_price = prices[i]
            bid = mid_price - spread[i]/2
            ask = mid_price + spread[i]/2
            
            tick_data.append({
                'timestamp': ts,
                'bid': bid,
                'ask': ask,
                'mid': mid_price,
                'volume': np.random.randint(1, 10)
            })
        
        return pd.DataFrame(tick_data)
    
    def ticks_to_ohlcv(self, tick_df, timeframe_minutes):
        """Convert tick data to OHLCV candles"""
        tick_df['timestamp_group'] = tick_df['timestamp'].dt.floor(f'{timeframe_minutes}min')
        
        ohlcv = tick_df.groupby('timestamp_group').agg({
            'mid': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        }).reset_index()
        
        # Flatten column names
        ohlcv.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        return ohlcv
    
    def generate_sample_files(self, days=30, output_dir='data'):
        """Generate sample CSV files for testing"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate data for specified number of days
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        print(f"Generating {days} days of sample data...")
        
        # Generate tick data (1 tick per second for testing - in reality much higher frequency)
        tick_data = self.generate_tick_data(start_time, end_time, tick_interval_seconds=60)
        
        # Create M1 candles
        m1_data = self.ticks_to_ohlcv(tick_data, 1)
        m1_file = os.path.join(output_dir, f'{self.symbol}_M1_sample.csv')
        m1_data.to_csv(m1_file, index=False)
        print(f"Generated M1 data: {m1_file} ({len(m1_data)} candles)")
        
        # Create M15 candles
        m15_data = self.ticks_to_ohlcv(tick_data, 15)
        m15_file = os.path.join(output_dir, f'{self.symbol}_M15_sample.csv')
        m15_data.to_csv(m15_file, index=False)
        print(f"Generated M15 data: {m15_file} ({len(m15_data)} candles)")
        
        # Create D1 candles
        d1_data = self.ticks_to_ohlcv(tick_data, 1440)
        d1_file = os.path.join(output_dir, f'{self.symbol}_D1_sample.csv')
        d1_data.to_csv(d1_file, index=False)
        print(f"Generated D1 data: {d1_file} ({len(d1_data)} candles)")
        
        return {
            'M1': m1_file,
            'M15': m15_file,
            'D1': d1_file,
            'tick': tick_data
        }

def main():
    """Generate sample data files"""
    generator = SampleDataGenerator()
    files = generator.generate_sample_files(days=30)
    print("Sample data generation complete!")
    print("Generated files:", files)

if __name__ == "__main__":
    main()