"""Fixed question menu and answer model. No NLP — discrete Q/A."""

from src.env import OBJ_TYPES, OBJ_COLORS, ROOM_NAMES

QUESTIONS = [
    ("ask_color", list(OBJ_COLORS)),
    ("ask_type", list(OBJ_TYPES)),
    ("ask_room", list(ROOM_NAMES)),
]


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


def answer_question(q, true_goal, world, rng, answer_noise=0.0):
    q_id = q[0] if isinstance(q, tuple) else q
    answer_space = next(ans for qid, ans in QUESTIONS if qid == q_id)
    true_ans = _goal_attribute_for_question(q_id, true_goal, world)
    if true_ans is None:
        return rng.choice(answer_space)
    if rng.random() >= answer_noise:
        return true_ans
    wrong = [a for a in answer_space if a != true_ans]
    return rng.choice(wrong) if wrong else true_ans


def answer_likelihood(q, ans, g, world, answer_noise):
    q_id = q[0] if isinstance(q, tuple) else q
    answer_space = next(ans for qid, ans in QUESTIONS if qid == q_id)
    goal_obj = world.get_object_by_id(g)
    if goal_obj is None:
        return 1.0 / len(answer_space)
    true_ans = _goal_attribute_for_question(q_id, goal_obj, world)
    if true_ans is None:
        return 1.0 / len(answer_space)
    n = len(answer_space)
    if ans == true_ans:
        return 1.0 - answer_noise
    return answer_noise / (n - 1) if (n - 1) > 0 else 0.0
