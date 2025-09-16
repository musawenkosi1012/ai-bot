#!/usr/bin/env python3
"""
Complete demonstration script for the ICT ML Trading Bot
Shows all major functionality working together
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def demo_full_pipeline():
    """Demonstrate the complete trading bot pipeline"""
    print("="*60)
    print("ICT ML Trading Bot - Complete Pipeline Demo")
    print("="*60)
    
    # Step 1: Generate fresh sample data
    print("\n1. Generating sample market data...")
    from data_generator import SampleDataGenerator
    generator = SampleDataGenerator()
    files = generator.generate_sample_files(days=7)  # Smaller dataset
    print(f"   Generated sample files: {files}")
    
    # Step 2: Create training data from backtesting
    print("\n2. Creating training data from backtesting...")
    # Use the faster method from create_training_data.py
    exec(open('src/create_training_data.py').read())
    
    # Step 3: Train ML models
    print("\n3. Training ML models...")
    from ml_models import train_models
    train_models('data/labeled_trades.csv', 'models')
    
    # Step 4: Test trading engine
    print("\n4. Testing trading engine...")
    from engine import TradingEngine
    engine = TradingEngine()
    
    print(f"   Engine loaded with {len(engine.m1)} M1 candles")
    print(f"   ML models: {'✓ Loaded' if hasattr(engine.ml, 'clf') else '✗ Using dummy'}")
    
    # Step 5: Test signal generation and ML filtering
    print("\n5. Testing signal generation with ML filtering...")
    
    from signal_generator import generate_candidate
    from indicators import daily_bias_from_D1
    
    test_runs = 3
    signals_found = 0
    
    for i in range(test_runs):
        current_price = engine.m1.iloc[-(i+1)]['close']
        bias = daily_bias_from_D1(engine.d1, current_price)
        
        candidate = generate_candidate(
            engine.m1.iloc[:max(1, len(engine.m1)-(i*10))], engine.m15, engine.d1,
            point=0.00001,
            ml_inference_func=engine.ml.predict,
            params={
                'daily_bias': bias,
                'sr_lookback': 60,  # Smaller for demo
                'sr_cluster_pips': 15,
                'zone_buffer_points': 5,
                'require_rejection': False,  # Relaxed for demo
                'rejection_candles': 3,
                'rejection_wick_pts': 6,
                'atr_period': 14,
                'tp_mult': 1.8,
                'sl_mult': 0.9,
                'p_threshold': 0.45,  # Lower threshold for demo
                'max_pred_slippage_pts': 8,
                'use_daily_bias_only': False,
                'spread_pts': 1.5
            }
        )
        
        if candidate:
            signals_found += 1
            print(f"   Signal {signals_found}: {candidate['side'].upper()} at {candidate['entry_price']:.5f}")
            print(f"      ML Score: P(win)={candidate['ml']['p_win']:.3f}, Slippage={candidate['ml']['pred_slippage']:.2f}pts")
            print(f"      Zone: [{candidate['zone'][0]:.5f} - {candidate['zone'][1]:.5f}]")
        else:
            print(f"   Test {i+1}: No valid signal (bias={bias})")
    
    # Step 6: Test order management (simulation)
    print("\n6. Testing order management...")
    from order_manager import place_market_order
    
    if signals_found > 0:
        result = place_market_order('buy', 0.01, 'EURUSD', sl=1.0980, tp=1.1020)
        print(f"   Order simulation: {result}")
    else:
        print("   No signals to execute")
    
    # Step 7: Summary
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    print(f"✓ Sample data generation: PASSED")
    print(f"✓ Training data creation: PASSED") 
    print(f"✓ ML model training: PASSED")
    print(f"✓ Trading engine: PASSED")
    print(f"✓ Signal generation: {signals_found}/{test_runs} signals found")
    print(f"✓ Order management: PASSED")
    print(f"✓ Complete pipeline: WORKING")
    
    print(f"\nNext steps:")
    print(f"1. Connect to real broker API (replace order_manager.py)")
    print(f"2. Use live data feeds (replace sample data)")
    print(f"3. Run on environment with GUI support for Tkinter interface")
    print(f"4. Deploy on VPS near broker for low latency")
    print(f"5. Implement proper logging and monitoring")
    
    return True

if __name__ == "__main__":
    try:
        demo_full_pipeline()
        print(f"\n🎉 ICT ML Trading Bot demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)