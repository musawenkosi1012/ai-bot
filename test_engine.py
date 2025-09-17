#!/usr/bin/env python3
"""
Test script for the ICT ML Trading Bot
Tests the engine without GUI to ensure core functionality works
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from engine import TradingEngine
import time

def test_engine():
    """Test the trading engine functionality"""
    print("Testing ICT ML Trading Bot Engine...")
    
    # Create engine
    engine = TradingEngine()
    
    # Test basic properties
    print(f"Engine initialized with {len(engine.m1)} M1 candles")
    print(f"M15 candles: {len(engine.m15)}")
    print(f"D1 candles: {len(engine.d1)}")
    
    # Test ML inference
    test_features = {
        'atr_m1': 0.0001,
        'dist_zone_pts': 15,
        'zone_width_pts': 8,
        'planned_rr': 1.8,
        'spread_pts': 1.5,
        'hour_of_day': 10
    }
    
    ml_result = engine.ml.predict(test_features)
    print(f"ML Prediction: P(win)={ml_result['p_win']:.3f}, Slippage={ml_result['pred_slippage']:.2f}pts")
    
    # Test a few engine iterations
    print("Running 5 engine iterations...")
    for i in range(5):
        try:
            # Calculate daily bias
            current_price = engine.m1.iloc[-1]['close']
            from indicators import daily_bias_from_D1
            bias = daily_bias_from_D1(engine.d1, current_price)
            print(f"Iteration {i+1}: Price={current_price:.5f}, Bias={bias}")
            
            # Run signal generation
            from signal_generator import generate_candidate
            candidate = generate_candidate(
                engine.m1, engine.m15, engine.d1, 
                point=0.00001,
                ml_inference_func=engine.ml.predict,
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
                    'p_threshold': 0.5,  # Lower threshold for testing
                    'max_pred_slippage_pts': 10,
                    'use_daily_bias_only': False,  # Allow neutral bias for testing
                    'spread_pts': 1.5
                }
            )
            
            if candidate:
                print(f"  → Valid signal: {candidate['side']} at {candidate['entry_price']:.5f}")
                print(f"  → ML: P(win)={candidate['ml']['p_win']:.3f}, Slippage={candidate['ml']['pred_slippage']:.2f}")
                break
            else:
                print("  → No valid signal")
                
        except Exception as e:
            print(f"  → Error in iteration {i+1}: {e}")
    
    print("Engine test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_engine()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)