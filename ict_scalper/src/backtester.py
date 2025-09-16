# src/backtester.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class Backtester:
    """
    Backtesting harness for labeling candidate events from tick-level data
    """
    def __init__(self, tick_data=None):
        self.tick_data = tick_data
        self.labeled_trades = []
    
    def simulate_trade(self, entry_price, side, sl_pips, tp_pips, point, max_duration_minutes=60):
        """
        Simulate a trade using tick data to determine win/loss and slippage
        Returns dict with win, time_to_hit, slippage_pts
        """
        if self.tick_data is None:
            # Fallback simulation without tick data
            import random
            win = random.choice([0, 1])
            time_to_hit = random.randint(30, 1800)  # 30 seconds to 30 minutes
            slippage_pts = random.uniform(-2, 5)
            return {'win': win, 'time_to_hit': time_to_hit, 'slippage_pts': slippage_pts}
        
        # TODO: Implement tick-level simulation
        # For now, return mock results
        return {'win': 1, 'time_to_hit': 300, 'slippage_pts': 1.5}
    
    def backtest_strategy(self, df_m1, df_m15, df_d1, signal_func, params, start_date=None, end_date=None):
        """
        Run backtest to generate labeled training data
        """
        results = []
        
        # Mock backtest - in real implementation, iterate through historical data
        # and generate candidates using signal_func
        for i in range(100):  # Generate 100 mock trades
            timestamp = datetime.now() - timedelta(days=i)
            
            # Mock features
            features = {
                'timestamp': timestamp,
                'symbol': 'EURUSD',
                'daily_bias': np.random.choice([-1, 0, 1]),
                'price_at_signal': 1.1000 + np.random.uniform(-0.01, 0.01),
                'distance_to_nearest_zone_pts': np.random.uniform(5, 50),
                'zone_width_pts': np.random.uniform(10, 30),
                'atr_m1': np.random.uniform(0.0001, 0.0005),
                'spread_pts': np.random.uniform(0.5, 2.0),
                'hour_of_day': np.random.randint(0, 24),
                'planned_rr': np.random.uniform(1.5, 3.0),
                'sl_pips': 10,
                'tp_pips': 18
            }
            
            # Simulate trade outcome
            trade_result = self.simulate_trade(
                features['price_at_signal'], 'buy' if features['daily_bias'] == 1 else 'sell',
                features['sl_pips'], features['tp_pips'], 0.00001
            )
            
            # Combine features and labels
            row = {**features, **trade_result}
            results.append(row)
        
        return pd.DataFrame(results)
    
    def save_training_data(self, df, filepath):
        """Save labeled data to CSV for ML training"""
        df.to_csv(filepath, index=False)
        logging.info(f"Training data saved to {filepath}")
        return filepath