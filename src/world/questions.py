"""Fixed question menu and answer model. No NLP - discrete Q/A."""

from src.env import OBJ_TYPES, OBJ_COLORS, ROOM_NAMES
from src import config

QUESTIONS = [
    ("ask_color", list(OBJ_COLORS)),
    ("ask_type", list(OBJ_TYPES)),
    ("ask_room", list(ROOM_NAMES)),
]

QUESTION_TYPE_BY_ID = {
    "ask_color": "color",
    "ask_room": "room",
    "ask_type": "object",
}


def list_questions():
    return list(QUESTIONS)


def _goal_attribute_for_question(q_id, goal_obj, world):
    if isinstance(goal_obj, dict):
        o = goal_obj
    else:
        o = world.get_object_by_id(goal_obj)
        if o is None:
            return None
        o = o if isinstance(o, dict) else {"type": o.type, "color": o.color, "room_id": o.room_id}
    if o is None:
        return None
    if q_id == "ask_color":
        return o["color"]
    if q_id == "ask_type":
        return o["type"]
    if q_id == "ask_room":
        return ROOM_NAMES[o["room_id"]]
    return None


def _noise_for_question(q_id, default_noise):
    """
    Resolve question-dependent answer noise.
    Priority:
      1) direct per-question id key in ANSWER_NOISE_BY_QTYPE
      2) mapped qtype key in ANSWER_NOISE_BY_QTYPE
      3) fallback to provided default_noise
    """
    by_qtype = getattr(config, "ANSWER_NOISE_BY_QTYPE", None) or {}
    if q_id in by_qtype:
        return float(by_qtype[q_id])
    q_type = QUESTION_TYPE_BY_ID.get(q_id)
    if q_type in by_qtype:
        return float(by_qtype[q_type])
    return float(default_noise)


def answer_question(q, true_goal, world, rng, answer_noise=0.0):
    q_id = q[0] if isinstance(q, tuple) else q
    answer_space = next(ans for qid, ans in QUESTIONS if qid == q_id)
    noise = _noise_for_question(q_id, answer_noise)
    true_ans = _goal_attribute_for_question(q_id, true_goal, world)
    if true_ans is None:
        return rng.choice(answer_space)
    if rng.random() >= noise:
        return true_ans
    wrong = [a for a in answer_space if a != true_ans]
    return rng.choice(wrong) if wrong else true_ans


def answer_likelihood(q, ans, g, world, answer_noise):
    q_id = q[0] if isinstance(q, tuple) else q
    answer_space = next(ans for qid, ans in QUESTIONS if qid == q_id)
    noise = _noise_for_question(q_id, answer_noise)
    goal_obj = world.get_object_by_id(g)
    if goal_obj is None:
        return 1.0 / len(answer_space)
    true_ans = _goal_attribute_for_question(q_id, goal_obj, world)
    if true_ans is None:
        return 1.0 / len(answer_space)
    n = len(answer_space)
    if ans == true_ans:
        return 1.0 - noise
    return noise / (n - 1) if (n - 1) > 0 else 0.0

