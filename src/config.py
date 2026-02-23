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
QUESTION_COST = 0.5
ASK_COUNTS_AS_STEP = True
MAX_QUESTIONS = 3
MAX_QUESTIONS_PER_EPISODE = None  # None / -1 means no additional episode-level cap.
ENTROPY_GATE = 0.8
ASK_WINDOW = 6
ENTROPY_THRESHOLD = 0.5
WRONG_PICK_PENALTY = 0  # Option A: 0 = do not fail on wrong pick; episode continues.
WRONG_PICK_FAIL = False

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
POLICIES = ["ask_or_act", "never_ask", "always_ask"]
N_EPISODES_PER_CONDITION = 20
BASE_SEED = 42
REPL_SEEDS = [0, 1, 2, 3, 4]
N_EPISODES_PER_SEED = N_EPISODES_PER_CONDITION

# Parallel evaluation (CPU workers; set 0 to disable)
N_WORKERS = 0  # 0 = sequential; 4 or 8 to use multiple cores
