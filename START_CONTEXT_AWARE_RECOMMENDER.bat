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
        echo (This happens when the folder is moved or copied from another computer.)
        echo.
    )
)

if "!VENV_OK!"=="0" (
    echo [SETUP] Initializing environment for this computer...
    echo.
    set "SYS_PYTHON="
    where py >nul 2>&1
    if not errorlevel 1 (
        set "SYS_PYTHON=py -3"
    ) else (
        where python >nul 2>&1
        if not errorlevel 1 (
            set "SYS_PYTHON=python"
        )
    )

    if "!SYS_PYTHON!"=="" (
        echo [ERROR] Python is not detected in your system PATH.
        echo.
        echo To run this project on this computer:
        echo 1. Download and install Python 3.10+ from: https://www.python.org/downloads/
        echo 2. IMPORTANT: Check the box "Add Python to PATH" during installation.
        echo 3. Re-run this launcher script.
        echo.
        pause
        exit /b 1
    )

    echo Found system Python: !SYS_PYTHON!
    echo Creating fresh virtual environment at .venv...
    if exist "%~dp0.venv" (
        rmdir /s /q "%~dp0.venv" >nul 2>&1
    )
    !SYS_PYTHON! -m venv "%~dp0.venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )

    echo Installing dependencies from requirements.txt...
    echo (This may take 1-2 minutes on first run)...
    "%PYTHON_EXE%" -m pip install --upgrade pip
    "%PYTHON_EXE%" -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Failed to install required packages.
        pause
        exit /b 1
    )
    echo [OK] Setup complete!
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
