# Create simple training data for testing
import pandas as pd
import numpy as np
import os

# Create sample training data
os.makedirs('../data', exist_ok=True)

# Generate synthetic training data
n_samples = 1000
np.random.seed(42)

data = {
    'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='1min'),
    'symbol': ['EURUSD'] * n_samples,
    'timeframe': ['M1'] * n_samples,
    'daily_bias': np.random.choice([1, -1, 0], n_samples),
    'price_at_signal': np.random.normal(1.1000, 0.001, n_samples),
    'distance_to_nearest_zone_pts': np.random.uniform(5, 50, n_samples),
    'zone_width_pts': np.random.uniform(3, 15, n_samples),
    'atr_m1': np.random.uniform(0.00005, 0.0002, n_samples),
    'atr_m15': np.random.uniform(0.0001, 0.0005, n_samples),
    'spread_pts': np.random.uniform(0.5, 2.5, n_samples),
    'volatility_lookback': np.random.uniform(0.001, 0.01, n_samples),
    'hour_of_day': np.random.randint(0, 24, n_samples),
    'weekday': np.random.randint(0, 7, n_samples),
    'momentum_1m': np.random.normal(0, 0.00002, n_samples),
    'momentum_5m': np.random.normal(0, 0.00005, n_samples),
    'sl_pips': np.random.uniform(8, 20, n_samples),
    'tp_pips': np.random.uniform(12, 35, n_samples),
    'planned_rr': np.random.uniform(1.2, 2.5, n_samples),
    'tick_density_last_30s': np.random.uniform(10, 60, n_samples),
    'rejection_wick_pts': np.random.uniform(0, 10, n_samples),
    'rejection_body_pct': np.random.uniform(0, 80, n_samples),
    # Labels - create realistic relationships
    'win': np.random.binomial(1, 0.55, n_samples),  # 55% win rate
    'time_to_hit': np.random.uniform(30, 1800, n_samples),  # 30s to 30min
    'slippage_pts': np.random.exponential(2, n_samples),  # Exponential slippage
    'entry_price_actual': np.random.normal(1.1000, 0.001, n_samples),
    'sl_price': np.random.normal(1.0990, 0.001, n_samples),
    'tp_price': np.random.normal(1.1015, 0.001, n_samples)
}

df = pd.DataFrame(data)
df.to_csv('../data/labeled_trades.csv', index=False)
print(f"Created sample training data: {len(df)} records")
print("Columns:", df.columns.tolist())