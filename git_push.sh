#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

REPO_NAME="${REPO_NAME:-AskOrAct}"
GITHUB_USER="${GITHUB_USER:-Yufannnn}"
COMMIT_MSG="${COMMIT_MSG:-Update project results and reports}"

echo "Checking git status..."
git status

echo
echo "Adding changes..."
git add .

echo
echo "Committing changes..."
git commit -m "$COMMIT_MSG" || true

echo
echo "Configuring remote..."
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
else
  git remote add origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
fi

echo
echo "Setting branch to main..."
git branch -M main

echo
echo "Pushing to GitHub..."
git push -u origin main

echo
echo "Successfully pushed to GitHub."
