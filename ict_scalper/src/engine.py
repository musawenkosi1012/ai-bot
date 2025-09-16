# src/engine.py
import os
import sys
import time
import logging
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data_loader import load_candles_csv
from indicators import daily_bias_from_D1
from signal_generator import generate_candidate
from ml_models import MLInference
from order_manager import place_market_order
from backtester import Backtester

class Engine:
    """
    Main orchestrator that coordinates data loading, signal generation,
    ML inference, and order execution
    """
    
    def __init__(self, config=None):
        self.running = False
        self.gui = None
        self.config = config or self.get_default_config()
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/engine.log'),
                logging.StreamHandler()
            ]
        )
        
        # Initialize ML models (create dummy ones if they don't exist)
        self.ml = self.initialize_ml_models()
        
        # Load sample data (will be replaced with live feeds in production)
        self.load_sample_data()
    
    def get_default_config(self):
        """Default trading parameters"""
        return {
            'daily_bias': 1,  # 1=long, -1=short, 0=neutral
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
            'use_daily_bias_only': True,
            'point': 0.00001,  # EURUSD point value
            'symbol': 'EURUSD',
            'volume': 0.01
        }
    
    def initialize_ml_models(self):
        """Initialize ML models - create dummy ones if needed"""
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        clf_path = models_dir / 'clf_win.joblib'
        reg_path = models_dir / 'reg_slip.joblib'
        
        # Create dummy models if they don't exist
        if not clf_path.exists() or not reg_path.exists():
            self.create_dummy_models(models_dir)
        
        return MLInference(str(clf_path), str(reg_path))
    
    def create_dummy_models(self, models_dir):
        """Create dummy ML models for testing"""
        import joblib
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        import numpy as np
        
        # Create dummy training data
        X_dummy = np.random.rand(100, 6)  # 6 features
        y_clf = np.random.randint(0, 2, 100)  # binary classification
        y_reg = np.random.rand(100) * 10  # regression target
        
        # Train and save dummy models
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_dummy, y_clf)
        joblib.dump(clf, models_dir / 'clf_win.joblib')
        
        reg = RandomForestRegressor(n_estimators=10, random_state=42)
        reg.fit(X_dummy, y_reg)
        joblib.dump(reg, models_dir / 'reg_slip.joblib')
        
        logging.info("Created dummy ML models for testing")
    
    def load_sample_data(self):
        """Load or create sample market data"""
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        # Create sample data if it doesn't exist
        self.create_sample_data(data_dir)
        
        # Load the data
        try:
            self.m1 = load_candles_csv(data_dir / 'EURUSD_M1_sample.csv')
            self.m15 = load_candles_csv(data_dir / 'EURUSD_M15_sample.csv')
            self.d1 = load_candles_csv(data_dir / 'EURUSD_D1_sample.csv')
            logging.info("Sample data loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load sample data: {e}")
            self.create_minimal_sample_data()
    
    def create_sample_data(self, data_dir):
        """Create sample OHLCV data for testing"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate sample M1 data
        if not (data_dir / 'EURUSD_M1_sample.csv').exists():
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=7),
                end=datetime.now(),
                freq='1min'  # Changed from '1T'
            )
            
            base_price = 1.1000
            m1_data = []
            
            for i, date in enumerate(dates):
                # Simulate price movement
                price_change = np.random.normal(0, 0.0001)
                base_price += price_change
                
                # Generate OHLC
                open_price = base_price
                high = open_price + abs(np.random.normal(0, 0.0002))
                low = open_price - abs(np.random.normal(0, 0.0002))
                close = open_price + np.random.normal(0, 0.0001)
                volume = np.random.randint(50, 500)
                
                m1_data.append({
                    'timestamp': date,
                    'open': round(open_price, 5),
                    'high': round(high, 5),
                    'low': round(low, 5),
                    'close': round(close, 5),
                    'volume': volume
                })
                
                base_price = close
            
            pd.DataFrame(m1_data).to_csv(data_dir / 'EURUSD_M1_sample.csv', index=False)
        
        # Generate M15 and D1 data from M1
        self.aggregate_sample_data(data_dir)
    
    def aggregate_sample_data(self, data_dir):
        """Create M15 and D1 data from M1 data"""
        if (data_dir / 'EURUSD_M1_sample.csv').exists():
            try:
                import pandas as pd  # Import pd locally
                m1 = pd.read_csv(data_dir / 'EURUSD_M1_sample.csv', parse_dates=['timestamp'])
                
                # Create M15 data
                m1['timestamp_15'] = m1['timestamp'].dt.floor('15min')  # Changed from '15T'
                m15 = m1.groupby('timestamp_15').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).reset_index()
                m15.rename(columns={'timestamp_15': 'timestamp'}, inplace=True)
                m15.to_csv(data_dir / 'EURUSD_M15_sample.csv', index=False)
                
                # Create D1 data
                m1['timestamp_d1'] = m1['timestamp'].dt.floor('D')
                d1 = m1.groupby('timestamp_d1').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).reset_index()
                d1.rename(columns={'timestamp_d1': 'timestamp'}, inplace=True)
                d1.to_csv(data_dir / 'EURUSD_D1_sample.csv', index=False)
                
                logging.info("Generated M15 and D1 sample data from M1")
                
            except Exception as e:
                logging.error(f"Failed to aggregate sample data: {e}")
                self.create_minimal_sample_data()
    
    def create_minimal_sample_data(self):
        """Create minimal sample data as fallback"""
        import pandas as pd
        from datetime import datetime
        
        # Minimal sample data
        sample_data = [{
            'timestamp': datetime.now(),
            'open': 1.1000,
            'high': 1.1005,
            'low': 1.0995,
            'close': 1.1002,
            'volume': 100
        }]
        
        self.m1 = pd.DataFrame(sample_data)
        self.m15 = pd.DataFrame(sample_data)
        self.d1 = pd.DataFrame(sample_data)
        
        logging.warning("Using minimal fallback sample data")
    
    def run(self):
        """Main trading loop"""
        self.running = True
        self.log("Trading engine started")
        
        while self.running:
            try:
                # Update daily bias
                current_price = self.m1.iloc[-1]['close']
                self.config['daily_bias'] = daily_bias_from_D1(self.d1, current_price)
                
                # Generate trading candidate
                candidate = generate_candidate(
                    self.m1, self.m15, self.d1, 
                    self.config['point'],
                    self.ml.predict,
                    self.config
                )
                
                if candidate:
                    self.log(f"ML approved candidate: {candidate['side']} at {candidate['entry_price']:.5f}")
                    self.log(f"ML score: p_win={candidate['ml']['p_win']:.3f}, slippage={candidate['ml']['pred_slippage']:.2f}")
                    
                    # Place order
                    result = place_market_order(
                        candidate['side'],
                        self.config['volume'],
                        self.config['symbol'],
                        sl=0.0,  # Will be calculated based on ATR
                        tp=0.0,  # Will be calculated based on ATR
                        comment="ML-approved ICT setup"
                    )
                    
                    self.log(f"Order result: {result}")
                else:
                    self.log("No valid trading candidates found")
                
                # Sleep before next iteration
                time.sleep(5)  # Check every 5 seconds in demo mode
                
            except Exception as e:
                self.log(f"Error in trading loop: {e}")
                time.sleep(10)  # Longer sleep on error
    
    def stop(self):
        """Stop the trading engine"""
        self.running = False
        self.log("Trading engine stopped")
    
    def log(self, message):
        """Log message to console and GUI"""
        logging.info(message)
        if self.gui:
            try:
                self.gui.log_message(message)
            except:
                pass  # GUI might not be ready