@echo off
REM AI Medical Assistant - Windows Environment Setup Script
echo Setting up AI Medical Assistant environment...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install requirements
    pause
    exit /b 1
)

REM Create necessary directories
echo Creating necessary directories...
mkdir logs 2>nul
mkdir models 2>nul
mkdir .cache 2>nul

REM Copy environment file
if not exist .env (
    echo Creating .env file from template...
    copy env.example .env
    echo Please edit .env file with your configuration
)

echo.
echo Setup completed successfully!
echo.
echo To activate the environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To start the application, run:
echo   streamlit run web_app/app.py
echo.
pause
