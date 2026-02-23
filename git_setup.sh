#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "Initializing git repository..."
git init

echo
echo "Adding files..."
git add .

echo
echo "Creating initial commit..."
git commit -m "Initial commit: AskOrAct CS6208 project - milestones implemented"

echo
echo "Git repository initialized and committed."
echo
echo "To push to GitHub:"
echo "  1. Create a new repository (e.g. https://github.com/<user>/AskOrAct)"
echo "  2. Run: git remote add origin https://github.com/<user>/AskOrAct.git"
echo "  3. Run: git branch -M main"
echo "  4. Run: git push -u origin main"
