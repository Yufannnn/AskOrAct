@echo off
REM Push to GitHub after git_setup.bat
cd /d "%~dp0"

set REPO_NAME=AskOrAct
set GITHUB_USER=Yufannnn

echo Checking git status...
git status
if errorlevel 1 (
    echo Git not found or not initialized. Run git_setup.bat first.
    pause
    exit /b 1
)

echo.
echo Adding any new changes...
git add .

echo.
echo Committing changes...
git commit -m "Update: principal cannot pick, assistant must pick true goal"

echo.
echo Setting remote (if not already set)...
git remote remove origin 2>nul
git remote add origin https://github.com/%GITHUB_USER%/%REPO_NAME%.git

echo.
echo Setting branch to main...
git branch -M main

echo.
echo Pushing to GitHub...
git push -u origin main

if errorlevel 1 (
    echo.
    echo Push failed. Possible reasons:
    echo   - Repository doesn't exist at https://github.com/%GITHUB_USER%/%REPO_NAME%
    echo   - Authentication required (use GitHub CLI or Personal Access Token)
    echo   - Run: git remote set-url origin https://YOUR_TOKEN@github.com/%GITHUB_USER%/%REPO_NAME%.git
    echo.
    pause
    exit /b 1
)

echo.
echo Successfully pushed to GitHub!
pause
