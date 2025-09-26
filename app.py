"""
AI Medical Assistant - Streamlit App for HuggingFace Spaces
Simplified version for deployment
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import and run the advanced app
from advanced_medical_app import main

if __name__ == "__main__":
    main()
