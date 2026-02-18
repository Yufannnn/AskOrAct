@echo off
REM One-time setup: create .venv and install deps. No PowerShell required.
REM Double-click or run:  setup.bat
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo .venv already exists.
    goto install
)

echo Creating .venv...
py -3 -m venv .venv 2>nul
if errorlevel 1 python -m venv .venv 2>nul
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Python not found. Install Python 3.8+ from https://www.python.org/downloads/
    echo Check "Add Python to PATH" then run this again.
    echo Or in PowerShell run:  winget install Python.Python.3.12
    exit /b 1
)
echo Created .venv

:install
echo Installing dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip -q
.venv\Scripts\pip.exe install -r requirements.txt -q
if errorlevel 1 (
    echo pip install failed.
    exit /b 1
)
echo.
echo Done. Run  run_demo.bat  or  run_report.bat
