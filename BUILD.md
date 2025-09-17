# Build and Test Instructions

## Prerequisites

Ensure you have Python 3.8+ installed with pip.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-bot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Testing the Bot

### 1. Quick Test (Recommended)

Run the complete demo to test all functionality:

```bash
python demo.py
```

This will:
- Generate sample market data
- Create training data  
- Train ML models
- Test the trading engine
- Show system integration

### 2. Individual Component Tests

**Test data generation:**
```bash
cd src
python data_generator.py
```

**Test engine without GUI:**
```bash
python test_engine.py
```

**Train ML models manually:**
```bash
cd src
python -c "from ml_models import train_models; train_models('../data/labeled_trades.csv', '../models')"
```

**Generate training data from backtesting:**
```bash
cd src
python backtester.py  # Note: This takes longer with real data
```

### 3. GUI Application

**Note:** Requires environment with GUI support and tkinter installed.

```bash
cd src
python main.py
```

On Linux, you may need to install tkinter:
```bash
sudo apt-get install python3-tk
```

## Building for Production

### 1. Broker Integration

Replace `src/order_manager.py` with real broker API:

- **MetaTrader 5:** Install `pip install MetaTrader5`
- **OANDA:** Use OANDA REST API
- **Interactive Brokers:** Use IB Gateway API

### 2. Live Data Feeds

Replace sample data in `src/engine.py` with:
- Real-time broker data feeds
- Historical data APIs
- Tick-level data for better ML training

### 3. Deployment

1. **VPS Setup:**
   - Deploy near broker servers for low latency
   - Use 24/7 uptime VPS
   - Configure proper logging and monitoring

2. **Safety Measures:**
   ```bash
   # Set up proper logging directory
   mkdir -p logs
   
   # Configure max drawdown limits
   # Implement circuit breakers
   # Set up alerts for system issues
   ```

## Linting and Code Quality

If you have linting tools installed:

```bash
# Using flake8
flake8 src/

# Using black for formatting
black src/

# Using pylint
pylint src/
```

## Common Issues

1. **"ML models not found"**
   - Run training first: `python -c "from src.ml_models import train_models; train_models('data/labeled_trades.csv', 'models')"`

2. **"No sample data"** 
   - Generate data: `python src/data_generator.py`

3. **GUI not starting**
   - Install tkinter: `sudo apt-get install python3-tk` (Linux)
   - Use headless testing: `python test_engine.py`

4. **Import errors**
   - Ensure you're running from project root directory
   - Check Python path includes src/ directory

## Performance Notes

- The system runs efficiently on modest hardware
- For high-frequency trading, consider:
  - Compiled languages (C++/Rust) for critical path
  - Co-located servers near exchanges
  - Optimized ML inference (ONNX, TensorRT)
  
## Next Steps

1. Connect to real broker demo account
2. Test with real historical data
3. Paper trade for at least 1 month
4. Implement proper risk management
5. Monitor model performance and retrain regularly