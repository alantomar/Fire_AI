@echo off
:: ============================================================
::  AI Fire Detection System - Dependency Installer
::  ============================================================
::  This script sets up everything needed to run the AI Fire
::  Detection System on a fresh Windows computer.
::
::  What it does:
::    1. Checks for Python 3.8+ installation
::    2. Creates a virtual environment (venv)
::    3. Installs all required Python packages
::    4. Creates necessary project directories
::    5. Verifies the installation
::
::  Usage:
::    Double-click this file, or run from Command Prompt:
::      install.bat
:: ============================================================

title AI Fire Detection System - Installer
color 0A

echo.
echo ============================================================
echo    AI FIRE DETECTION SYSTEM - DEPENDENCY INSTALLER
echo ============================================================
echo.

:: ─── Step 1: Check Python Installation ───────────────────────
echo [1/5] Checking Python installation...
echo.

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo.
    echo   Please install Python 3.8 or newer from:
    echo     https://www.python.org/downloads/
    echo.
    echo   IMPORTANT: During installation, check the box that says
    echo     "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

:: Check Python version (need 3.8+)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo   Found Python %PYVER%

:: Extract major and minor version numbers
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)

if %PYMAJOR% LSS 3 (
    echo [ERROR] Python 3.8+ is required. You have Python %PYVER%.
    echo   Please upgrade from https://www.python.org/downloads/
    pause
    exit /b 1
)

if %PYMAJOR%==3 if %PYMINOR% LSS 8 (
    echo [ERROR] Python 3.8+ is required. You have Python %PYVER%.
    echo   Please upgrade from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo   [OK] Python %PYVER% is compatible.
echo.

:: ─── Step 2: Check pip Installation ──────────────────────────
echo [2/5] Checking pip...

python -m pip --version >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo   pip not found. Installing pip...
    python -m ensurepip --upgrade
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install pip.
        pause
        exit /b 1
    )
)
echo   [OK] pip is available.
echo.

:: ─── Step 3: Create Virtual Environment ─────────────────────
echo [3/5] Setting up virtual environment...

set "VENV_DIR=%~dp0venv"

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo   Virtual environment already exists.
    echo   Activating existing environment...
) else (
    echo   Creating virtual environment in: %VENV_DIR%
    python -m venv "%VENV_DIR%"
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        echo   Trying to install without venv...
        goto :install_global
    )
)

:: Activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"
echo   [OK] Virtual environment activated.
echo.

:: ─── Step 4: Install Dependencies ───────────────────────────
echo [4/5] Installing Python dependencies...
echo   This may take several minutes on the first run.
echo.

:: Upgrade pip first
python -m pip install --upgrade pip >nul 2>nul

:: Install all requirements
python -m pip install -r "%~dp0requirements.txt"
if %ERRORLEVEL% neq 0 (
    echo.
    echo [WARNING] Some packages may have failed to install.
    echo   Attempting individual package installation...
    echo.
    goto :install_individual
)

goto :post_install

:install_global
echo.
echo   Installing packages globally (no virtual environment)...
python -m pip install --upgrade pip >nul 2>nul
python -m pip install -r "%~dp0requirements.txt"
if %ERRORLEVEL% neq 0 (
    goto :install_individual
)
goto :post_install

:install_individual
:: Try installing packages one by one so partial failures don't block everything
echo   Installing packages individually...
echo.

for %%p in (
    "flask>=3.0"
    "flask-socketio>=5.3"
    "numpy>=1.24"
    "Pillow>=10.0"
    "opencv-python>=4.8"
    "ultralytics>=8.1"
    "playsound==1.2.2"
    "python-engineio>=4.8"
    "python-socketio>=5.10"
    "gevent>=23.0"
    "gevent-websocket>=0.10"
    "pyserial>=3.5"
) do (
    echo   Installing %%p ...
    python -m pip install %%p
    if %ERRORLEVEL% neq 0 (
        echo     [WARN] Failed to install %%p
    ) else (
        echo     [OK]
    )
)

:post_install
echo.

:: ─── Step 5: Create Project Directories ─────────────────────
echo [5/5] Setting up project directories...

set "PROJECT_DIR=%~dp0"

if not exist "%PROJECT_DIR%models" mkdir "%PROJECT_DIR%models"
if not exist "%PROJECT_DIR%logs" mkdir "%PROJECT_DIR%logs"
if not exist "%PROJECT_DIR%logs\snapshots" mkdir "%PROJECT_DIR%logs\snapshots"
if not exist "%PROJECT_DIR%static\sounds" mkdir "%PROJECT_DIR%static\sounds"

echo   [OK] All directories created.
echo.

:: ─── Verification ───────────────────────────────────────────
echo ============================================================
echo   VERIFYING INSTALLATION
echo ============================================================
echo.

set INSTALL_OK=1

:: Check critical packages
python -c "import flask; print(f'  Flask         : {flask.__version__}')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo   [FAIL] Flask
    set INSTALL_OK=0
)

python -c "import cv2; print(f'  OpenCV        : {cv2.__version__}')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo   [FAIL] OpenCV
    set INSTALL_OK=0
)

python -c "import numpy; print(f'  NumPy         : {numpy.__version__}')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo   [FAIL] NumPy
    set INSTALL_OK=0
)

python -c "import ultralytics; print(f'  Ultralytics   : {ultralytics.__version__}')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo   [FAIL] Ultralytics (YOLOv8)
    set INSTALL_OK=0
)

python -c "import serial; print(f'  PySerial      : {serial.__version__}')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo   [FAIL] PySerial
    set INSTALL_OK=0
)

python -c "import flask_socketio; print(f'  Flask-SocketIO: {flask_socketio.__version__}')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo   [FAIL] Flask-SocketIO
    set INSTALL_OK=0
)

python -c "import PIL; print(f'  Pillow        : {PIL.__version__}')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo   [FAIL] Pillow
    set INSTALL_OK=0
)

echo.

if %INSTALL_OK%==1 (
    echo ============================================================
    echo   INSTALLATION SUCCESSFUL!
    echo ============================================================
    echo.
    echo   All dependencies are installed and verified.
    echo.
    echo   To run the system:
    echo     1. Activate the virtual environment:
    echo          %VENV_DIR%\Scripts\activate.bat
    echo.
    echo     2. Start the application:
    echo          python app.py
    echo.
    echo     3. Open the dashboard in your browser:
    echo          http://localhost:5001
    echo.
    echo   Optional - Train a custom YOLO model:
    echo          python train_model.py --epochs 50
    echo ============================================================
) else (
    echo ============================================================
    echo   INSTALLATION COMPLETED WITH WARNINGS
    echo ============================================================
    echo.
    echo   Some packages failed verification. The system may still
    echo   work in color-detection mode without YOLO/Arduino.
    echo.
    echo   Try running: python app.py
    echo ============================================================
)

echo.
pause
