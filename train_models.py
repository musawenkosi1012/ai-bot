#!/usr/bin/env python3
"""
Training script for ML models
Usage: python train_models.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'ict_scalper' / 'src'))

from backtester import Backtester
from ml_models import train_models
import pandas as pd
import logging

def generate_training_data():
    """Generate labeled training data using backtester"""
    print("Generating training data...")
    
    # Create backtester
    backtester = Backtester()
    
    # Generate mock training data (in real implementation, use historical data)
    df = backtester.backtest_strategy(None, None, None, None, {})
    
    # Save to CSV
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    training_file = data_dir / 'training_data.csv'
    backtester.save_training_data(df, training_file)
    
    return training_file

def main():
    """Main training workflow"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate training data
    training_file = generate_training_data()
    
    # Train models
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    print("Training ML models...")
    train_models(str(training_file), str(models_dir))
    
    print("Training complete! Models saved to models/ directory")
    print("You can now run the main bot: python ict_scalper/src/main.py")

if __name__ == '__main__':
    main()