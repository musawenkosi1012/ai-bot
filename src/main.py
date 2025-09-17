# src/main.py
from gui import BotGUI
from engine import TradingEngine
import sys

def main():
    """Main entry point for the ICT ML Scalping Bot"""
    try:
        # Create trading engine
        engine = TradingEngine()
        
        # Create and start GUI
        gui = BotGUI(engine)
        engine.gui = gui
        
        print("Starting ICT ML Scalping Bot GUI...")
        gui.run()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()