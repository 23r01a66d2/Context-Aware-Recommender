@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo         PACKAGING CONTEXT-AWARE RECOMMENDER ZIP
echo ========================================================
echo.

set "RUN_PY="
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -c "import sys" >nul 2>&1
    if not errorlevel 1 set "RUN_PY=%~dp0.venv\Scripts\python.exe"
)

if "%RUN_PY%"=="" (
    where py >nul 2>&1
    if not errorlevel 1 set "RUN_PY=py -3"
)

if "%RUN_PY%"=="" (
    where python >nul 2>&1
    if not errorlevel 1 set "RUN_PY=python"
)

if "%RUN_PY%"=="" (
    echo [ERROR] Python not found.
    pause
    exit /b 1
)

"%RUN_PY%" "%~dp0scripts\create_distribution_zip.py"

echo.
pause
