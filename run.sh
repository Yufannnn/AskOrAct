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
  ./run.sh quick-demo        (single episode demo with ASCII render)
  ./run.sh sweep             (main sweep → results/metrics/metrics.csv)
  ./run.sh ablations         (ablation grid → results/metrics/metrics_ablations.csv)
  ./run.sh robustness        (answer-noise + mismatch sweeps + delta plots)
  ./run.sh qdifficulty       (question-difficulty sweep + plots)
  ./run.sh generalization    (held-out templates + scale-K stress test)
  ./run.sh report            (regenerate results/reports/full_report.md + dashboards)
  ./run.sh test              (run regression tests)
  ./run.sh all               (sweep + ablations + robustness + report)
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

quick_demo()     { ensure_venv; "$PYTHON_VENV" scripts/demos/quick_demo.py; }
sweep()          { ensure_venv; "$PYTHON_VENV" scripts/sweeps/run_sweep.py; }
ablations()      { ensure_venv; "$PYTHON_VENV" scripts/sweeps/run_ablations.py; }
robustness()     { ensure_venv; "$PYTHON_VENV" scripts/sweeps/run_robustness.py; }
qdifficulty()    { ensure_venv; "$PYTHON_VENV" scripts/sweeps/run_question_difficulty.py; }
generalization() { ensure_venv; "$PYTHON_VENV" scripts/sweeps/run_generalization.py; }
report()         { ensure_venv; "$PYTHON_VENV" scripts/reporting/generate_report.py; }
tests()          { ensure_venv; "$PYTHON_VENV" -m unittest discover -s tests -v; }

all() {
  sweep
  ablations
  robustness
  report
}

case "$cmd" in
  help|-h|--help) print_help ;;
  setup)          setup ;;
  quick-demo)     quick_demo ;;
  sweep)          sweep ;;
  ablations)      ablations ;;
  robustness)     robustness ;;
  qdifficulty)    qdifficulty ;;
  generalization) generalization ;;
  report)         report ;;
  test)           tests ;;
  all)            all ;;
  *)
    echo "Unknown command: $cmd"
    echo
    print_help
    exit 1
    ;;
esac
