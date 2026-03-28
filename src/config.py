"""
Central config: all constants. No CLI; override by editing this file or env vars.
"""

import os

# -----------------------------------------------------------------------------
# GPU
# -----------------------------------------------------------------------------
USE_GPU = os.environ.get("ASKORACT_USE_GPU", "0").lower() in ("1", "true", "yes")

# -----------------------------------------------------------------------------
# Env / world
# -----------------------------------------------------------------------------
DEFAULT_N = 9
DEFAULT_M = 6
TWO_ROOMS = True
MAX_STEPS = 200
USE_DYNAMIC_DEADLINE = True
DEADLINE_MARGIN = 5

# -----------------------------------------------------------------------------
# Principal
# -----------------------------------------------------------------------------
PRINCIPAL_CAN_PICK = False  # If False, principal pick is no-op; only assistant can pick.
DEFAULT_BETA = 2.0
DEFAULT_EPS = 0.05

# -----------------------------------------------------------------------------
# Questions / answer
# -----------------------------------------------------------------------------
ANSWER_NOISE = 0.0
ANSWER_NOISE_BY_QTYPE = {
    "color": 0.05,
    "room": 0.10,
    "object": 0.20,
}
QUESTION_COST = 0.5
ASK_COUNTS_AS_STEP = True
MAX_QUESTIONS = 3
MAX_QUESTIONS_PER_EPISODE = None  # None / -1 means no additional episode-level cap.
ENTROPY_GATE = 0.3
ASK_WINDOW = 6
ENTROPY_THRESHOLD = 0.5
IG_THRESHOLD = 0.01
INFOGAIN_USE_ENTROPY_GATE = True
EASY_IG_MAX_QUESTIONS = 1
WRONG_PICK_PENALTY = 0  # Option A: 0 = do not fail on wrong pick; episode continues.
WRONG_PICK_FAIL = False
POMCP_ITERS = 200
POMCP_HORIZON = 8
POMCP_UCT_C = 1.4
POMCP_MIN_K = 2

# Debug tracing
DEBUG = False
DEBUG_MAX_STEPS = 15
DEBUG_MIN_K = 3

# -----------------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------------
AMBIGUITY_LEVELS = [1, 2, 3, 4]
EPS_LEVELS = [0.0, 0.05, 0.1]
BETA_LEVELS = [1.0, 2.0, 4.0]

# Global policy registry. Individual experiments should use the explicit
# subsets below instead of assuming every baseline runs everywhere.
POLICIES = [
    "ask_or_act",
    "never_ask",
    "always_ask",
    "info_gain_ask",
    "easy_info_gain_ask",
    "random_ask",
    "pomcp_planner",
]

# The main sweep matches the paper artifacts: 5 core policies at every K,
# plus two stronger baselines only at K=2 where they were originally profiled.
MAIN_SWEEP_BASE_POLICIES = [
    "ask_or_act",
    "never_ask",
    "always_ask",
    "info_gain_ask",
    "random_ask",
]
MAIN_SWEEP_EXTRA_POLICIES_BY_K = {
    2: ["easy_info_gain_ask", "pomcp_planner"],
}

# Experiment-specific policy sets.
ROBUST_ANSWER_POLICIES = list(POLICIES)
ROBUST_MISMATCH_POLICIES = list(MAIN_SWEEP_BASE_POLICIES)
ABLATION_POLICIES = list(POLICIES)
SCALEK_POLICIES = list(POLICIES)
N_EPISODES_PER_CONDITION = 20
BASE_SEED = 42
REPL_SEEDS = [0, 1, 2, 3, 4]
N_EPISODES_PER_SEED = N_EPISODES_PER_CONDITION

# Parallel evaluation (CPU workers; set 0 to disable)
N_WORKERS = 0  # 0 = sequential; 4 or 8 to use multiple cores

# -----------------------------------------------------------------------------
# Generalization experiments
# -----------------------------------------------------------------------------
DEFAULT_LAYOUT_TYPE = "vertical"  # "vertical" (default two-room) or "horizontal"
DEFAULT_PRIOR_TYPE = "uniform"    # "uniform" or "distance"
ASYMMETRIC_NOISE_BY_QTYPE = {
    "color": 0.05,
    "room": 0.10,
    "object": 0.40,  # deliberately harsh (default 0.20)
}
GENERALIZATION_POLICIES = ["ask_or_act", "never_ask", "info_gain_ask"]

# -----------------------------------------------------------------------------
# Structural OOD experiments
# -----------------------------------------------------------------------------
STRUCTURAL_OOD_GRID_SIZES = [7, 9, 11]
STRUCTURAL_OOD_ROOM_CONFIGS = [True, False]  # two_rooms: True (default), False (single room)
STRUCTURAL_OOD_POLICIES = ["ask_or_act", "never_ask", "info_gain_ask"]

# -----------------------------------------------------------------------------
# Model-mismatch experiments (extended)
# -----------------------------------------------------------------------------
MISMATCH_POLICIES = ["ask_or_act", "never_ask", "info_gain_ask"]

# -----------------------------------------------------------------------------
# Failure penalty sweep
# -----------------------------------------------------------------------------
FAILURE_PENALTY_VALUES = [0, 5, 10, 20, 50]
FAILURE_PENALTY_POLICIES = ["ask_or_act", "never_ask", "info_gain_ask"]
