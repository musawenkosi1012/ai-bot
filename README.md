# AI-Powered ICT Scalping Bot

A professional Python-based trading bot that combines Inner Circle Trader (ICT) methodology with machine learning for enhanced trade execution. The bot uses ML models to filter and improve trade selection while maintaining the core ICT scalping strategy.

## Features

- **ICT Trading Logic**: Daily bias, M15 S/R zones, M1 rejection entries
- **Machine Learning Integration**: 
  - Trade Success Classifier (probability of winning)
  - Slippage Prediction Model (execution quality)
- **Risk Management**: ATR-based stop loss/take profit, position sizing
- **Real-time GUI**: Tkinter-based interface for monitoring and control
- **Backtesting Framework**: Historical data analysis and model training
- **Broker Integration**: Extensible order management (MT5/API ready)

## System Architecture

```
ict_scalper/
├── data/           # Historical tick/candle data (CSV)
├── models/         # Trained ML models (joblib files)
├── logs/          # Application logs
├── src/           # Source code modules
│   ├── main.py                 # Entry point with GUI
│   ├── engine.py               # Main orchestrator
│   ├── data_loader.py          # Data loading utilities
│   ├── indicators.py           # ICT indicators (bias, S/R, ATR)
│   ├── signal_generator.py     # Entry logic with ML gating
│   ├── ml_models.py           # ML training and inference
│   ├── order_manager.py       # Order execution wrapper
│   ├── backtester.py          # Backtesting harness
│   └── gui.py                 # Tkinter GUI
├── requirements.txt
└── train_models.py            # ML training script
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/musawenkosi1012/ai-bot.git
cd ai-bot

# Install dependencies
pip install -r requirements.txt
```

### 2. Train ML Models

```bash
# Generate training data and train models
python train_models.py
```

### 3. Run the Bot

```bash
# Start the GUI application
python ict_scalper/src/main.py
```

## ML Integration

The bot uses two supervised learning models:

### Trade Success Classifier
- **Purpose**: Predicts probability of trade hitting target before stop
- **Model**: Random Forest (LightGBM recommended for production)
- **Features**: ATR, zone distance, spread, volatility, time-of-day
- **Usage**: Only execute trades with P(success) >= threshold (default: 0.6)

### Slippage Regression Model  
- **Purpose**: Predicts expected execution slippage
- **Model**: Random Forest Regressor
- **Features**: Same as classifier plus order flow proxies
- **Usage**: Reject trades with predicted slippage > threshold

## Configuration

Key parameters in `engine.py`:

```python
config = {
    'daily_bias': 1,              # 1=long, -1=short, 0=neutral
    'sr_lookback': 120,           # M15 bars for S/R zones
    'sr_cluster_pips': 20,        # Zone clustering threshold
    'p_threshold': 0.6,           # ML probability threshold
    'max_pred_slippage_pts': 5,   # Max acceptable slippage
    'tp_mult': 1.8,               # Take profit multiplier
    'sl_mult': 0.9,               # Stop loss multiplier
}
```

## Data Requirements

For production use, you need:

1. **Tick-level data** for accurate slippage labeling
2. **M1 candles** for entry signals
3. **M15 candles** for S/R zone construction  
4. **D1 candles** for daily bias calculation

Sample data is auto-generated for testing.

## Broker Integration

The `order_manager.py` module provides a template for broker integration:

- **MetaTrader 5**: Use `MetaTrader5` Python package
- **REST APIs**: OANDA, FXCM, IG Markets
- **Crypto**: ccxt library for exchanges

```python
# Replace the stub implementation in order_manager.py
import MetaTrader5 as mt5

def place_market_order(side, volume, symbol, sl, tp, comment=""):
    # Real MT5 implementation
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if side == 'buy' else mt5.ORDER_TYPE_SELL,
        "sl": sl,
        "tp": tp,
        "comment": comment,
    }
    return mt5.order_send(request)
```

## Safety & Risk Management

- **Paper Trading**: Test with demo accounts first
- **Position Sizing**: Based on ATR and account balance
- **Emergency Stop**: Max drawdown limits
- **Model Monitoring**: Regular retraining on new data
- **Latency Optimization**: VPS near broker recommended

## Performance Monitoring

The GUI displays:
- Live P&L
- Trade acceptance/rejection rates  
- ML model scores
- System status and logs

## Development Workflow

1. **Collect Data**: Historical ticks/candles for your instruments
2. **Backtest**: Generate labeled candidates using `backtester.py`
3. **Train Models**: Use `train_models.py` to create ML models
4. **Validate**: Time-based cross-validation, calibration checks
5. **Deploy**: Paper trade → small live → scale up
6. **Monitor**: Track model drift, retrain periodically

## Contributing

1. Fork the repository
2. Create feature branches for changes
3. Add tests for new functionality
4. Submit pull requests with detailed descriptions

## Disclaimer

This software is for educational purposes. Trading involves risk of loss. The authors are not responsible for any financial losses incurred using this software. Always test thoroughly with paper/demo accounts before live trading.

## License

Apache License 2.0 - see LICENSE file for details.