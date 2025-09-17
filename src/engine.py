# src/engine.py
import time
import logging
from data_loader import load_candles_csv
from signal_generator import generate_candidate
from indicators import daily_bias_from_D1
from order_manager import place_market_order
from ml_models import MLInference
import os

class TradingEngine:
    def __init__(self):
        self.running = False
        self.gui = None
        self.ml = None
        self.trades = []
        self.accepted_trades = 0
        self.rejected_trades = 0
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load models if available
        try:
            if os.path.exists('models/clf_win.joblib') and os.path.exists('models/reg_slip.joblib'):
                self.ml = MLInference('models/clf_win.joblib', 'models/reg_slip.joblib')
                self.log("ML models loaded successfully")
            else:
                self.log("Warning: ML models not found. Using dummy inference.")
                self.ml = DummyMLInference()
        except Exception as e:
            self.log(f"Error loading ML models: {e}. Using dummy inference.")
            self.ml = DummyMLInference()
        
        # Load sample data (replace with live data feed in production)
        self.load_sample_data()

    def load_sample_data(self):
        """Load sample data or create dummy data for testing"""
        try:
            if os.path.exists('data/EURUSD_M1_sample.csv'):
                self.m1 = load_candles_csv('data/EURUSD_M1_sample.csv')
                self.m15 = load_candles_csv('data/EURUSD_M15_sample.csv')
                self.d1 = load_candles_csv('data/EURUSD_D1_sample.csv')
                self.log("Sample data loaded from CSV files")
            else:
                self.log("No sample data found, creating dummy data for testing")
                self.create_dummy_data()
        except Exception as e:
            self.log(f"Error loading sample data: {e}. Creating dummy data.")
            self.create_dummy_data()

    def create_dummy_data(self):
        """Create dummy data for testing purposes"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate dummy M1 data
        start_time = datetime.now() - timedelta(hours=24)
        timestamps = [start_time + timedelta(minutes=i) for i in range(1440)]  # 24 hours of M1 data
        
        base_price = 1.10000
        prices = []
        current_price = base_price
        
        for i in range(len(timestamps)):
            # Simple random walk
            change = np.random.normal(0, 0.00005)
            current_price += change
            prices.append(current_price)
        
        # Create OHLCV data
        m1_data = []
        for i in range(len(timestamps)):
            open_price = prices[i]
            high_price = open_price + abs(np.random.normal(0, 0.00010))
            low_price = open_price - abs(np.random.normal(0, 0.00010))
            close_price = open_price + np.random.normal(0, 0.00008)
            volume = np.random.randint(50, 200)
            
            m1_data.append({
                'timestamp': timestamps[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        self.m1 = pd.DataFrame(m1_data)
        
        # Create M15 data (aggregate from M1)
        self.m15 = self.m1.groupby(self.m1.index // 15).agg({
            'timestamp': 'first',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index(drop=True)
        
        # Create D1 data (aggregate from M1)
        self.d1 = self.m1.groupby(self.m1.index // 1440).agg({
            'timestamp': 'first',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index(drop=True)
        
        # Ensure we have at least 3 daily candles for bias calculation
        if len(self.d1) < 3:
            for i in range(3 - len(self.d1)):
                self.d1 = pd.concat([self.d1.iloc[[0]], self.d1], ignore_index=True)
                self.d1.loc[0, 'timestamp'] = self.d1.loc[1, 'timestamp'] - timedelta(days=1)

    def run(self):
        """Main trading loop"""
        self.running = True
        self.log("Trading engine started")
        
        while self.running:
            try:
                # Calculate daily bias
                current_price = self.m1.iloc[-1]['close']
                bias = daily_bias_from_D1(self.d1, current_price)
                
                # Generate candidate trade
                candidate = generate_candidate(
                    self.m1, self.m15, self.d1, 
                    point=0.00001,  # EUR/USD point value
                    ml_inference_func=self.ml.predict,
                    params={
                        'daily_bias': bias,
                        'sr_lookback': 120,
                        'sr_cluster_pips': 20,
                        'zone_buffer_points': 5,
                        'require_rejection': True,
                        'rejection_candles': 3,
                        'rejection_wick_pts': 6,
                        'atr_period': 14,
                        'tp_mult': 1.8, 
                        'sl_mult': 0.9,
                        'p_threshold': 0.6, 
                        'max_pred_slippage_pts': 5,
                        'use_daily_bias_only': True
                    }
                )
                
                if candidate:
                    self.accepted_trades += 1
                    self.log(f"Trade candidate accepted: {candidate['side']} at {candidate['entry_price']:.5f}")
                    self.log(f"ML Score - P(win): {candidate['ml']['p_win']:.3f}, Predicted slippage: {candidate['ml']['pred_slippage']:.2f}pts")
                    
                    # Place order (stub implementation)
                    result = place_market_order(
                        candidate['side'], 
                        volume=0.01, 
                        symbol='EURUSD',
                        sl=0.0, 
                        tp=0.0,
                        comment="ICT-ML"
                    )
                    self.log(f"Order placed: {result}")
                    self.trades.append(candidate)
                else:
                    self.rejected_trades += 1
                    if self.rejected_trades % 10 == 0:  # Log every 10th rejection to avoid spam
                        self.log(f"No valid trade signal. Accepted: {self.accepted_trades}, Rejected: {self.rejected_trades}")
                
            except Exception as e:
                self.log(f"Error in trading loop: {e}")
            
            # Sleep between iterations
            time.sleep(2)  # Check every 2 seconds

    def stop(self):
        """Stop the trading engine"""
        self.running = False
        self.log("Trading engine stopped")

    def log(self, message):
        """Log message to console and GUI"""
        self.logger.info(message)
        if self.gui:
            self.gui.log_message(message)


class DummyMLInference:
    """Dummy ML inference for testing when models are not available"""
    
    def predict(self, features_dict):
        import random
        # Return random but realistic predictions
        p_win = random.uniform(0.45, 0.75)  # Random probability
        pred_slippage = random.uniform(1, 8)  # Random slippage in points
        return {'p_win': p_win, 'pred_slippage': pred_slippage}