@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "LAUNCHER_PY=%~dp0scripts\launcher.py"

echo ========================================================
echo              CONTEXT-AWARE RECOMMENDER
echo           Multi-Modal Recommendation Platform
echo ========================================================
echo.

set "VENV_OK=0"
if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "VENV_OK=1"
    ) else (
        echo [NOTICE] Existing .venv appears invalid for this machine.
        echo (This occurs when moving the project from another computer.)
        echo Re-initializing virtual environment automatically...
        echo.
    )
)

if "!VENV_OK!"=="0" (
    echo [SETUP] Initializing environment for this computer...
    echo.
    set "SYS_PYTHON="

    :: 1. Check Python Launcher
    where py >nul 2>&1
    if not errorlevel 1 (
        set "SYS_PYTHON=py -3"
    )

    :: 2. Check system PATH python
    if "!SYS_PYTHON!"=="" (
        where python >nul 2>&1
        if not errorlevel 1 (
            set "SYS_PYTHON=python"
        )
    )

    :: 3. Check common Windows user installation paths
    if "!SYS_PYTHON!"=="" (
        if exist "%LocalAppData%\Programs\Python\Python313\python.exe" set "SYS_PYTHON=%LocalAppData%\Programs\Python\Python313\python.exe"
        if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "SYS_PYTHON=%LocalAppData%\Programs\Python\Python312\python.exe"
        if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "SYS_PYTHON=%LocalAppData%\Programs\Python\Python311\python.exe"
        if exist "%LocalAppData%\Programs\Python\Python310\python.exe" set "SYS_PYTHON=%LocalAppData%\Programs\Python\Python310\python.exe"
    )

    :: 4. Check common system drive installation paths
    if "!SYS_PYTHON!"=="" (
        if exist "C:\Python313\python.exe" set "SYS_PYTHON=C:\Python313\python.exe"
        if exist "C:\Python312\python.exe" set "SYS_PYTHON=C:\Python312\python.exe"
        if exist "C:\Python311\python.exe" set "SYS_PYTHON=C:\Python311\python.exe"
        if exist "C:\Python310\python.exe" set "SYS_PYTHON=C:\Python310\python.exe"
    )

    if "!SYS_PYTHON!"=="" (
        echo [ERROR] Python was not detected on this computer.
        echo.
        echo To run this project:
        echo 1. Download and install Python 3.10+ from: https://www.python.org/downloads/
        echo 2. IMPORTANT: Check the box "Add Python to PATH" during installation.
        echo 3. Double-click START_CONTEXT_AWARE_RECOMMENDER.bat again.
        echo.
        pause
        exit /b 1
    )

    echo Found Python interpreter: !SYS_PYTHON!
    echo.
    echo Step 1 of 2: Creating virtual environment (.venv)...
    if exist "%~dp0.venv" (
        rmdir /s /q "%~dp0.venv" >nul 2>&1
    )
    !SYS_PYTHON! -m venv "%~dp0.venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo Please ensure Python is properly installed.
        pause
        exit /b 1
    )

    echo Step 2 of 2: Installing project dependencies from requirements.txt...
    echo (This will take 1-3 minutes to download and install PyTorch and ML packages)...
    echo.
    "%PYTHON_EXE%" -m pip install --upgrade pip
    "%PYTHON_EXE%" -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install required packages.
        echo Please check your internet connection and re-run.
        pause
        exit /b 1
    )
    echo.
    echo [OK] Setup completed successfully! Launching server...
    echo.
)

"%PYTHON_EXE%" "%LAUNCHER_PY%" start

if errorlevel 1 (
    echo.
    echo ========================================================
    echo  Startup failed. Please inspect the log output above.
    echo ========================================================
    echo.
    pause
)
