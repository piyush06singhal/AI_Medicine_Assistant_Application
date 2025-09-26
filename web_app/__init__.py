"""
Web Application Package
Contains the Streamlit web application for the AI Medical Assistant.
"""

from .api import MedicalAPI
from .app import create_app

__all__ = [
    "MedicalAPI",
    "create_app"
]
