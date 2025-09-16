# src/backtester.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from indicators import daily_bias_from_D1, find_swings_levels, cluster_levels, is_price_touch_zone, check_rejection_m1, atr
import os

class BacktestLabeler:
    """Generate labeled training data from historical tick/candle data"""
    
    def __init__(self, point=0.00001):
        self.point = point
        self.labeled_trades = []
    
    def label_trades_from_data(self, df_m1, df_m15, df_d1, params, output_csv='data/labeled_trades.csv'):
        """
        Generate labeled training data by simulating trades on historical data
        
        Parameters:
        - df_m1: M1 candle data
        - df_m15: M15 candle data  
        - df_d1: Daily candle data
        - params: Trading parameters
        - output_csv: Output path for labeled data
        """
        print("Starting trade labeling process...")
        labeled_data = []
        
        # Start from a point where we have enough history
        start_idx = max(params['sr_lookback'], params['atr_period'], 100)
        
        for i in range(start_idx, len(df_m1) - 100):  # Leave room for trade simulation
            try:
                # Get historical data up to current point
                m1_hist = df_m1.iloc[:i+1].copy()
                m15_hist = df_m15.iloc[:i//15+1].copy()  # Approximate M15 alignment
                d1_hist = df_d1.iloc[:i//1440+1].copy()  # Approximate D1 alignment
                
                # Ensure we have minimum required history
                if len(m15_hist) < params['sr_lookback'] or len(d1_hist) < 3:
                    continue
                    
                # Calculate current state
                current_price = m1_hist.iloc[-1]['close']
                current_time = m1_hist.iloc[-1]['timestamp']
                
                # Calculate daily bias
                try:
                    bias = daily_bias_from_D1(d1_hist, current_price)
                except (IndexError, KeyError):
                    continue
                
                # Check if this would be a valid signal (without ML)
                candidate = self._check_base_signal(m1_hist, m15_hist, bias, current_price, params)
                
                if candidate:
                    # Extract features
                    features = self._extract_features(m1_hist, m15_hist, d1_hist, candidate, current_price, current_time, params)
                    
                    # Simulate trade outcome
                    outcome = self._simulate_trade_outcome(df_m1, i, candidate, params)
                    
                    if outcome:
                        # Combine features and labels
                        trade_record = {**features, **outcome}
                        labeled_data.append(trade_record)
                        
                        if len(labeled_data) % 100 == 0:
                            print(f"Labeled {len(labeled_data)} trades...")
                            
            except Exception as e:
                print(f"Error processing candle {i}: {e}")
                continue
        
        # Save to CSV
        if labeled_data:
            df_labeled = pd.DataFrame(labeled_data)
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df_labeled.to_csv(output_csv, index=False)
            print(f"Saved {len(labeled_data)} labeled trades to {output_csv}")
        else:
            print("No valid trades found for labeling")
            
        return labeled_data
    
    def _check_base_signal(self, df_m1, df_m15, bias, price, params):
        """Check if base trading rules would trigger a signal"""
        try:
            # Build SR zones
            levels = find_swings_levels(df_m15, lookback=params['sr_lookback'])
            zones = cluster_levels(levels, params['sr_cluster_pips'], self.point)
            
            if not zones:
                return None
            
            # Find target zone based on bias
            target_zone = None
            if bias == 1:
                below = [z for z in zones if (z[0]+z[1])/2.0 < price]
                if below:
                    target_zone = sorted(below, key=lambda z: price - (z[0]+z[1])/2.0)[0]
            elif bias == -1:
                above = [z for z in zones if (z[0]+z[1])/2.0 > price]
                if above:
                    target_zone = sorted(above, key=lambda z: (z[0]+z[1])/2.0 - price)[0]
            
            if not target_zone:
                return None
                
            # Check zone touch
            if not is_price_touch_zone(price, target_zone, params['zone_buffer_points'] * self.point):
                return None
                
            # Check rejection if required
            if params['require_rejection']:
                df_recent = df_m1.tail(params['rejection_candles'])
                if not check_rejection_m1(df_recent, target_zone, params['rejection_wick_pts'], self.point):
                    return None
            
            return {
                'side': 'buy' if bias == 1 else 'sell',
                'bias': bias,
                'target_zone': target_zone,
                'zones': zones
            }
            
        except Exception as e:
            return None
    
    def _extract_features(self, df_m1, df_m15, df_d1, candidate, price, timestamp, params):
        """Extract ML features from market state"""
        features = {}
        
        try:
            # Basic features
            features['timestamp'] = timestamp
            features['symbol'] = 'EURUSD'  # hardcoded for now
            features['timeframe'] = 'M1'
            features['daily_bias'] = candidate['bias']
            features['price_at_signal'] = price
            
            # Zone features
            zone = candidate['target_zone']
            zone_mid = (zone[0] + zone[1]) / 2.0
            features['distance_to_nearest_zone_pts'] = abs(price - zone_mid) / self.point
            features['zone_width_pts'] = (zone[1] - zone[0]) / self.point
            
            # ATR features
            try:
                atr_m1 = atr(df_m1['high'], df_m1['low'], df_m1['close'], period=params['atr_period'])
                features['atr_m1'] = atr_m1[-1] if len(atr_m1) > 0 else 0
                
                if len(df_m15) >= params['atr_period']:
                    atr_m15 = atr(df_m15['high'], df_m15['low'], df_m15['close'], period=params['atr_period'])
                    features['atr_m15'] = atr_m15[-1] if len(atr_m15) > 0 else 0
                else:
                    features['atr_m15'] = 0
            except:
                features['atr_m1'] = 0
                features['atr_m15'] = 0
            
            # Spread (simulated)
            features['spread_pts'] = np.random.uniform(0.5, 2.0)  # Typical EUR/USD spread
            
            # Volatility
            if len(df_m1) >= 60:
                recent_returns = df_m1['close'].tail(60).pct_change().dropna()
                features['volatility_lookback'] = recent_returns.std()
            else:
                features['volatility_lookback'] = 0
            
            # Time features
            features['hour_of_day'] = timestamp.hour
            features['weekday'] = timestamp.weekday()
            
            # Momentum features
            if len(df_m1) >= 5:
                features['momentum_1m'] = df_m1['close'].iloc[-1] - df_m1['close'].iloc[-2]
                features['momentum_5m'] = df_m1['close'].iloc[-1] - df_m1['close'].iloc[-6] if len(df_m1) >= 6 else 0
            else:
                features['momentum_1m'] = 0
                features['momentum_5m'] = 0
            
            # Trade plan features
            features['sl_pips'] = params['sl_mult'] * features['atr_m1'] / self.point if features['atr_m1'] > 0 else 10
            features['tp_pips'] = params['tp_mult'] * features['atr_m1'] / self.point if features['atr_m1'] > 0 else 18
            features['planned_rr'] = features['tp_pips'] / features['sl_pips'] if features['sl_pips'] > 0 else 1.8
            
            # Additional features
            features['tick_density_last_30s'] = np.random.uniform(10, 50)  # Simulated tick density
            features['rejection_wick_pts'] = 0  # Will be calculated if rejection found
            features['rejection_body_pct'] = 0
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return {}
    
    def _simulate_trade_outcome(self, df_m1, start_idx, candidate, params, max_bars=100):
        """Simulate trade outcome to generate labels"""
        try:
            entry_price = df_m1.iloc[start_idx]['close']
            entry_time = df_m1.iloc[start_idx]['timestamp']
            
            # Calculate SL and TP levels
            atr_val = 0.0001  # Default ATR for simulation
            if start_idx >= params['atr_period']:
                try:
                    atr_series = atr(df_m1.iloc[start_idx-params['atr_period']:start_idx]['high'],
                                   df_m1.iloc[start_idx-params['atr_period']:start_idx]['low'], 
                                   df_m1.iloc[start_idx-params['atr_period']:start_idx]['close'],
                                   period=params['atr_period'])
                    if len(atr_series) > 0:
                        atr_val = atr_series[-1]
                except:
                    pass
            
            sl_distance = params['sl_mult'] * atr_val
            tp_distance = params['tp_mult'] * atr_val
            
            if candidate['side'] == 'buy':
                sl_price = entry_price - sl_distance
                tp_price = entry_price + tp_distance
            else:
                sl_price = entry_price + sl_distance
                tp_price = entry_price - tp_distance
            
            # Simulate slippage
            simulated_slippage = np.random.normal(0, 1.5)  # Mean 0, std 1.5 points
            actual_entry = entry_price + (simulated_slippage * self.point * (1 if candidate['side'] == 'buy' else -1))
            
            # Check outcome over next bars
            win = False
            time_to_hit = max_bars
            
            for i in range(1, min(max_bars + 1, len(df_m1) - start_idx)):
                current_bar = df_m1.iloc[start_idx + i]
                
                if candidate['side'] == 'buy':
                    if current_bar['high'] >= tp_price:
                        win = True
                        time_to_hit = i
                        break
                    elif current_bar['low'] <= sl_price:
                        win = False
                        time_to_hit = i
                        break
                else:  # sell
                    if current_bar['low'] <= tp_price:
                        win = True
                        time_to_hit = i
                        break
                    elif current_bar['high'] >= sl_price:
                        win = False
                        time_to_hit = i
                        break
            
            return {
                'win': 1 if win else 0,
                'time_to_hit': time_to_hit * 60,  # Convert to seconds (M1 = 60 seconds)
                'slippage_pts': abs(simulated_slippage),
                'entry_price_actual': actual_entry,
                'sl_price': sl_price,
                'tp_price': tp_price
            }
            
        except Exception as e:
            print(f"Error simulating trade: {e}")
            return None


def generate_sample_training_data():
    """Generate sample training data for testing ML models"""
    from engine import TradingEngine
    
    # Create engine to get sample data
    engine = TradingEngine()
    
    # Parameters for backtesting
    params = {
        'sr_lookback': 120,
        'sr_cluster_pips': 20,
        'zone_buffer_points': 5,
        'require_rejection': True,
        'rejection_candles': 3,
        'rejection_wick_pts': 6,
        'atr_period': 14,
        'tp_mult': 1.8,
        'sl_mult': 0.9
    }
    
    # Generate labels
    labeler = BacktestLabeler()
    labeled_data = labeler.label_trades_from_data(
        engine.m1, engine.m15, engine.d1, 
        params, 
        output_csv='data/labeled_trades.csv'
    )
    
    return labeled_data

if __name__ == "__main__":
    generate_sample_training_data()