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
    pause
    exit /b 1
)

"%PYTHON_EXE%" "%LAUNCHER_PY%" stop

echo.
pause
