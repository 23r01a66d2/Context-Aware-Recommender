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

set "RUN_PY="
if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "RUN_PY=%PYTHON_EXE%"
    )
)

if "!RUN_PY!"=="" (
    where py >nul 2>&1
    if not errorlevel 1 (
        set "RUN_PY=py -3"
    ) else (
        where python >nul 2>&1
        if not errorlevel 1 (
            set "RUN_PY=python"
        )
    )
)

if "!RUN_PY!"=="" (
    echo [ERROR] No working Python environment found to run the stop script.
    pause
    exit /b 1
)

!RUN_PY! "%LAUNCHER_PY%" stop

echo.
pause
