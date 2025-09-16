# src/main.py
import os
import sys
from pathlib import Path

# Add src directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Set working directory to ict_scalper
os.chdir(current_dir.parent)

from gui import BotGUI
from engine import Engine

def main():
    """
    Main entry point for the ICT Scalping Bot with ML
    """
    try:
        # Create necessary directories
        Path('logs').mkdir(exist_ok=True)
        Path('models').mkdir(exist_ok=True)
        Path('data').mkdir(exist_ok=True)
        
        # Initialize engine
        engine = Engine()
        
        # Initialize and run GUI
        gui = BotGUI(engine)
        engine.gui = gui
        
        print("Starting ICT M1 Scalper with ML capabilities...")
        print("GUI will open shortly...")
        
        gui.run()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()