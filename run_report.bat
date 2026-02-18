@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo .venv not found. Running setup.bat ...
    call "%~dp0setup.bat"
    if not exist ".venv\Scripts\python.exe" exit /b 1
)

.venv\Scripts\python.exe scripts/run_milestone_feb13_feb20.py --report
echo.
echo Report: results\milestone_feb13_feb20_report.md
