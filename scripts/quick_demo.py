"""Run one episode with ASCII render and debug prints (principal/assistant pos, posterior top 3, ask/act, pick result)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src import config
from src.world import generate_world, instruction_to_candidate_goals, answer_question, answer_likelihood, list_questions
from src.inference import init_posterior, update_posterior
from src.agents import sample_principal_action, policy_ask_or_act

SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    env, instruction_u, true_goal_obj_id = generate_world(SEED, N=9, M=6, ambiguity_K=4)
    candidate_goals = instruction_to_candidate_goals(instruction_u, env)
    posterior = init_posterior(candidate_goals)
    principal_action_history = []
    questions_asked = 0

    print("Instruction u:", instruction_u)
    print("True goal obj_id:", true_goal_obj_id)
    print("Candidate goals:", candidate_goals)
    print()

    success = False
    for step in range(config.MAX_STEPS):
        state = env.get_state()
        p_pos = state["principal_pos"]
        a_pos = state["assistant_pos"]
        print("--- Step", step + 1, "---")
        print("Principal position:", p_pos, "| Assistant position:", a_pos)
        print(env.render_ascii(true_goal_obj_id))

        top_goals = sorted(candidate_goals, key=lambda g: posterior.get(g, 0), reverse=True)[:3]
        top_probs = [(g, round(posterior.get(g, 0), 3)) for g in top_goals]
        print("Posterior top 3 goals (id, prob):", top_probs)

        principal_action = sample_principal_action(state, true_goal_obj_id, env, rng, config.DEFAULT_BETA, config.DEFAULT_EPS)
        principal_action_history.append(principal_action)

        out, posterior = policy_ask_or_act(
            env=env, state=state, instruction_u=instruction_u, posterior=posterior,
            candidate_goals=candidate_goals, principal_action_history=principal_action_history,
            questions_asked=questions_asked, rng=rng, beta=config.DEFAULT_BETA, eps=config.DEFAULT_EPS,
            answer_noise=config.ANSWER_NOISE,
        )

        if isinstance(out, tuple) and out[0] == "ask":
            _, q_id = out
            questions_asked += 1
            q_tuple = next((q for q in list_questions() if q[0] == q_id), None)
            ans = answer_question(q_tuple, true_goal_obj_id, env, rng, config.ANSWER_NOISE)
            print("Assistant decision: ASK  question:", q_id, "-> answer:", ans)
            p_new = {g: posterior.get(g, 0) * answer_likelihood(q_tuple, ans, g, env, config.ANSWER_NOISE) for g in candidate_goals}
            Z = sum(p_new.values())
            if Z > 0:
                for g in candidate_goals:
                    posterior[g] = p_new[g] / Z
            print()
            continue

        assistant_action = out
        print("Assistant decision: ACT  action:", assistant_action)
        print("Principal action:", principal_action)
        update_posterior(posterior, state, principal_action, candidate_goals, env, config.DEFAULT_BETA, config.DEFAULT_EPS)
        state, done, info = env.step(principal_action, assistant_action, true_goal_obj_id=true_goal_obj_id)
        success = info.get("assistant_picked_goal", False)

        if assistant_action == "pick":
            picked = info.get("picked_obj_id")
            is_goal = info.get("assistant_picked_goal", False)
            print("Assistant pick: picked_obj_id =", picked, "| was_true_goal =", is_goal)
        print("principal_on_goal_cell (debug):", info.get("principal_on_goal_cell", False))
        print()

        if done or success:
            print("Episode ended. Success (assistant picked goal):", success)
            break

    if not success:
        print("Max steps reached. Success: False")
    print("Final state:")
    print(env.render_ascii(true_goal_obj_id))


if __name__ == "__main__":
    main()
