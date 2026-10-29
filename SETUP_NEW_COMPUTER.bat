@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ========================================================
echo              CONTEXT-AWARE RECOMMENDER
echo           New Computer / Environment Setup
echo ========================================================
echo.

set "SYS_PYTHON="
where py >nul 2>&1
if not errorlevel 1 set "SYS_PYTHON=py -3"

if "!SYS_PYTHON!"=="" (
    where python >nul 2>&1
    if not errorlevel 1 set "SYS_PYTHON=python"
)

if "!SYS_PYTHON!"=="" (
    if exist "%LocalAppData%\Programs\Python\Python313\python.exe" set "SYS_PYTHON=%LocalAppData%\Programs\Python\Python313\python.exe"
    if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "SYS_PYTHON=%LocalAppData%\Programs\Python\Python312\python.exe"
    if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "SYS_PYTHON=%LocalAppData%\Programs\Python\Python311\python.exe"
    if exist "%LocalAppData%\Programs\Python\Python310\python.exe" set "SYS_PYTHON=%LocalAppData%\Programs\Python\Python310\python.exe"
    if exist "C:\Python313\python.exe" set "SYS_PYTHON=C:\Python313\python.exe"
    if exist "C:\Python312\python.exe" set "SYS_PYTHON=C:\Python312\python.exe"
    if exist "C:\Python311\python.exe" set "SYS_PYTHON=C:\Python311\python.exe"
    if exist "C:\Python310\python.exe" set "SYS_PYTHON=C:\Python310\python.exe"
)

if "!SYS_PYTHON!"=="" (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo Using Python: !SYS_PYTHON!
echo.
echo Step 1/2: Initializing clean virtual environment (.venv)...
if exist "%~dp0.venv" (
    echo Removing existing .venv...
    rmdir /s /q "%~dp0.venv" >nul 2>&1
)
!SYS_PYTHON! -m venv "%~dp0.venv"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo Step 2/2: Installing dependencies from requirements.txt...
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo [ERROR] Installation failed.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo  Environment successfully configured!
echo  You can now start the application with start.bat
echo ========================================================
echo.
pause
