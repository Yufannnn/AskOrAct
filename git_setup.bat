@echo off
REM Initialize git, commit, and prepare for push to GitHub (Yufannnn)
cd /d "%~dp0"

echo Initializing git repository...
git init
if errorlevel 1 (
    echo Git not found. Install Git from https://git-scm.com/downloads
    pause
    exit /b 1
)

echo.
echo Adding files...
git add .

echo.
echo Creating initial commit...
git commit -m "Initial commit: AskOrAct CS6208 project - Feb 13+20 milestones implemented"

echo.
echo Git repository initialized and committed.
echo.
echo To push to GitHub:
echo   1. Create a new repository at https://github.com/Yufannnn/AskOrAct (or your repo name)
echo   2. Run: git remote add origin https://github.com/Yufannnn/AskOrAct.git
echo   3. Run: git branch -M main
echo   4. Run: git push -u origin main
echo.
echo Or if you already have the repo URL, run:
echo   git remote add origin YOUR_REPO_URL
echo   git branch -M main
echo   git push -u origin main
echo.
pause
