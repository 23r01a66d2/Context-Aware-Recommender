@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "LAUNCHER_PY=%~dp0scripts\launcher.py"

if not exist "%PYTHON_EXE%" (
    echo ========================================================
    echo              CONTEXT-AWARE RECOMMENDER
    echo           Multi-Modal Recommendation Platform
    echo ========================================================
    echo.
    echo [ERROR] Virtual environment not found at:
    echo   %PYTHON_EXE%
    echo.
    echo Please ensure the project virtual environment is created before launching.
    echo.
    pause
    exit /b 1
)

"%PYTHON_EXE%" "%LAUNCHER_PY%" start

if errorlevel 1 (
    echo.
    echo ========================================================
    echo  Startup failed. Please inspect the log output above.
    echo ========================================================
    pause
)
