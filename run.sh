#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

cmd="${1:-help}"
PYTHON_VENV=""
PIP_VENV=""

print_help() {
  cat <<'EOF'
AskOrAct task runner

Usage:
  ./run.sh setup             (create .venv + install deps)
  ./run.sh quick-demo        (full quick demo with ask/act)
  ./run.sh milestone-demo    (Feb13+20 demo)
  ./run.sh milestone-report  (Feb13+20 report artifacts)
  ./run.sh sweep             (main sweep + main dashboard)
  ./run.sh ablations         (ablation sweep + ablation dashboard)
  ./run.sh report            (regenerate report + dashboards from CSVs)
  ./run.sh package           (build submission package + zip in results)
  ./run.sh all               (sweep + ablations + report + package)
EOF
}

ensure_venv() {
  resolve_venv_bins
  if [[ -n "$PYTHON_VENV" ]]; then
    return 0
  fi
  echo ".venv not found. Running setup..."
  setup
  resolve_venv_bins
}

resolve_venv_bins() {
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_VENV=".venv/bin/python"
    PIP_VENV=".venv/bin/pip"
    return
  fi
  if [[ -x ".venv/Scripts/python.exe" ]]; then
    PYTHON_VENV=".venv/Scripts/python.exe"
    PIP_VENV=".venv/Scripts/pip.exe"
    return
  fi
  PYTHON_VENV=""
  PIP_VENV=""
}

setup() {
  resolve_venv_bins
  if [[ -z "$PYTHON_VENV" ]]; then
    echo "Creating .venv..."
    if command -v python3 >/dev/null 2>&1; then
      python3 -m venv .venv
    else
      python -m venv .venv
    fi
  else
    echo ".venv already exists."
  fi
  resolve_venv_bins
  if [[ -z "$PYTHON_VENV" || -z "$PIP_VENV" ]]; then
    echo "Failed to find virtualenv python/pip binaries."
    exit 1
  fi
  echo "Installing dependencies..."
  "$PYTHON_VENV" -m pip install --upgrade pip -q
  "$PIP_VENV" install -r requirements.txt -q
  echo "Done."
}

quick_demo() {
  ensure_venv
  "$PYTHON_VENV" scripts/quick_demo.py
}

milestone_demo() {
  ensure_venv
  "$PYTHON_VENV" scripts/run_milestone_feb13_feb20.py
}

milestone_report() {
  ensure_venv
  "$PYTHON_VENV" scripts/run_milestone_feb13_feb20.py --report
  echo "Report: results/milestone_feb13_feb20_report.md"
}

sweep() {
  ensure_venv
  "$PYTHON_VENV" scripts/run_sweep.py
}

ablations() {
  ensure_venv
  "$PYTHON_VENV" scripts/run_ablations.py
}

report() {
  ensure_venv
  "$PYTHON_VENV" scripts/generate_report.py
}

package() {
  ensure_venv
  "$PYTHON_VENV" scripts/package_submission.py
}

all() {
  sweep
  ablations
  report
  package
}

case "$cmd" in
  help|-h|--help) print_help ;;
  setup) setup ;;
  quick-demo) quick_demo ;;
  milestone-demo) milestone_demo ;;
  milestone-report) milestone_report ;;
  sweep) sweep ;;
  ablations) ablations ;;
  report) report ;;
  package) package ;;
  all) all ;;
  *)
    echo "Unknown command: $cmd"
    echo
    print_help
    exit 1
    ;;
esac
