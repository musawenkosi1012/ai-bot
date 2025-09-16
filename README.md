# ICT ML Scalping Bot

A professional Python-based trading bot that implements Inner Circle Trader (ICT) concepts enhanced with Machine Learning for improved trade execution and selection.

## Features

- **ICT Trading Concepts**: Daily bias, M15 Support/Resistance zones, M1 rejection patterns
- **Machine Learning Integration**: 
  - Trade success probability classifier
  - Slippage prediction regression model
  - ML-based trade filtering and execution optimization
- **Risk Management**: ATR-based stop loss/take profit, position sizing
- **Professional Architecture**: Modular design with clear separation of concerns
- **Tkinter GUI**: Simple interface for bot control and monitoring
- **Backtesting**: Generate labeled training data from historical tick data

## Project Structure

```
ict_scalper/
├─ data/                      # Historical tick/candle CSVs
├─ models/                    # Trained ML models (joblib files)  
├─ logs/                      # Application logs
├─ src/
│  ├─ main.py                 # Main entry point
│  ├─ engine.py               # Trading engine orchestrator
│  ├─ data_loader.py          # Load historical/cached ticks and candles
│  ├─ indicators.py           # S/R zones, daily bias, ATR, rejection patterns
│  ├─ signal_generator.py     # Entry rules with ML gating
│  ├─ ml_models.py            # ML training & inference
│  ├─ order_manager.py        # Broker API wrapper (MT5/REST)
│  ├─ backtester.py           # Generate training data
│  ├─ data_generator.py       # Sample data for testing
│  └─ gui.py                  # Tkinter GUI
├─ requirements.txt
└─ README.md
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Sample Data** (for testing):
   ```bash
   cd src
   python data_generator.py
   ```

3. **Generate Training Data**:
   ```bash
   cd src  
   python backtester.py
   ```

4. **Train ML Models**:
   ```bash
   cd src
   python -c "from ml_models import train_models; train_models('data/labeled_trades.csv', 'models')"
   ```

5. **Run the Bot**:
   ```bash
   cd src
   python main.py
   ```

## Machine Learning Approach

### Models Used

1. **Trade Success Classifier**: Predicts probability of hitting target before stop loss
   - Model: Random Forest / LightGBM 
   - Features: ATR, zone distance, spread, momentum, time features
   - Output: P(win) probability

2. **Slippage Regression**: Predicts expected execution slippage
   - Model: Random Forest Regressor
   - Features: Same as classifier
   - Output: Expected slippage in points

### Feature Engineering

The bot extracts these features for ML inference:
- `atr_m1`, `atr_m15`: Average True Range indicators
- `dist_zone_pts`: Distance to nearest S/R zone in points
- `zone_width_pts`: Width of target zone
- `spread_pts`: Current bid/ask spread  
- `volatility_lookback`: Recent price volatility
- `momentum_1m`, `momentum_5m`: Price momentum
- `hour_of_day`, `weekday`: Time-based features
- `planned_rr`: Planned risk/reward ratio

### Training Data Generation

The backtester simulates historical trades to create labeled data:
- For each potential entry signal, extract features
- Simulate trade outcome using tick data
- Label: `win` (1/0), `time_to_hit`, `slippage_pts`
- Time-based train/test split (no shuffling)

## Trading Logic Flow

1. **Data Loading**: Get latest M1, M15, D1 candles
2. **Daily Bias**: Calculate ICT-style directional bias
3. **S/R Zones**: Find M15 swing levels and cluster into zones
4. **Entry Rules**: Check price touch + rejection patterns
5. **ML Filtering**: Score trade with probability and slippage models
6. **Execution**: Place trade if ML score passes thresholds
7. **Monitoring**: Track open positions and performance

## Configuration

Key parameters in `engine.py`:
- `p_threshold`: Minimum win probability (default: 0.6)
- `max_pred_slippage_pts`: Maximum allowed slippage (default: 5pts)
- `sr_lookback`: Bars for S/R zone detection (default: 120)
- `tp_mult`, `sl_mult`: ATR multipliers for TP/SL (default: 1.8/0.9)

## Safety & Production Notes

- **Demo Trading First**: Always test with demo accounts
- **Model Retraining**: Retrain models monthly with new data  
- **Risk Limits**: Implement max drawdown and position limits
- **Latency**: Host on low-latency VPS near broker
- **Broker Integration**: Replace `order_manager.py` with real broker API

## Broker Integration

Current `order_manager.py` is a stub. For production:

- **MT5**: Use `MetaTrader5` Python package
- **OANDA**: Use OANDA REST API
- **Interactive Brokers**: Use IB Gateway + API
- **Crypto**: Use ccxt library

## Troubleshooting

- **"ML models not found"**: Run training first or use dummy inference mode
- **"No sample data"**: Run `data_generator.py` to create test data
- **GUI not starting**: Ensure tkinter is installed (`apt-get install python3-tk` on Linux)

## License

See LICENSE file for details.