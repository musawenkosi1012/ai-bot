#!/usr/bin/env python3
"""
Test script to verify the core engine functionality without GUI
"""
import sys
import os
from pathlib import Path

# Add src directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / 'ict_scalper' / 'src'))

# Set working directory
os.chdir(current_dir / 'ict_scalper')

from engine import Engine
import time

def test_engine():
    """Test the engine without GUI"""
    print("Testing ICT M1 Scalper Engine...")
    
    try:
        # Initialize engine
        engine = Engine()
        print("✓ Engine initialized successfully")
        
        # Test data loading
        print(f"✓ M1 data loaded: {len(engine.m1)} candles")
        print(f"✓ M15 data loaded: {len(engine.m15)} candles")
        print(f"✓ D1 data loaded: {len(engine.d1)} candles")
        
        # Test ML models
        test_features = {
            'atr_m1': 0.0002,
            'distance_to_nearest_zone_pts': 15.5,
            'zone_width_pts': 20.0,
            'planned_rr': 2.0,
            'spread_pts': 1.5,
            'hour_of_day': 12
        }
        
        ml_result = engine.ml.predict(test_features)
        print(f"✓ ML prediction works: p_win={ml_result['p_win']:.3f}, slippage={ml_result['pred_slippage']:.2f}")
        
        # Test a few iterations of the trading logic
        print("\nTesting trading logic (3 iterations):")
        for i in range(3):
            print(f"\nIteration {i+1}:")
            
            # Simulate one iteration of the trading loop
            current_price = engine.m1.iloc[-1]['close']
            bias = engine.config['daily_bias']
            print(f"  Current price: {current_price:.5f}, Bias: {bias}")
            
            # Test signal generation (this may not produce signals but should not crash)
            from signal_generator import generate_candidate
            candidate = generate_candidate(
                engine.m1, engine.m15, engine.d1,
                engine.config['point'],
                engine.ml.predict,
                engine.config
            )
            
            if candidate:
                print(f"  ✓ Generated candidate: {candidate['side']} with p_win={candidate['ml']['p_win']:.3f}")
            else:
                print(f"  ✓ No candidate generated (expected in random data)")
            
            time.sleep(1)
        
        print("\n✅ All tests passed successfully!")
        print("\nThe bot is ready to run. Use the following commands:")
        print("  python ict_scalper/src/main.py  # Run with GUI")
        print("  python train_models.py         # Retrain models")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_engine()
    sys.exit(0 if success else 1)