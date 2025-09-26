"""
Launcher script for the AI Medical Assistant web application.
"""

import subprocess
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_basic_app():
    """Run the basic Streamlit app."""
    try:
        app_path = Path(__file__).parent / "web_app" / "medical_assistant_app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running basic app: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

def run_enhanced_app():
    """Run the enhanced Streamlit app."""
    try:
        app_path = Path(__file__).parent / "web_app" / "enhanced_app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running enhanced app: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

def main():
    """Main launcher function."""
    print("🏥 AI Medical Assistant - Web App Launcher")
    print("=" * 50)
    print("Choose an app version:")
    print("1. Basic App (Simple interface)")
    print("2. Enhanced App (With analytics and history)")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        print("🚀 Starting Basic App...")
        run_basic_app()
    elif choice == '2':
        print("🚀 Starting Enhanced App...")
        run_enhanced_app()
    elif choice == '3':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
