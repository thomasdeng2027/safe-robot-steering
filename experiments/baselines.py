# baselines.py

import torch
import os
from model.smolvla_policy import SmolVLALiberoPolicy
from env.env import make_libero_env
import logging

TASK_SUITE_NAME = "libero_10"
NUM_TASKS = 10           # libero_10 has 10 tasks
EPISODES_PER_TASK = 5
STEPS_PER_EPISODE = 520  # default for object tasks; can change for long horizon

logger = logging.getLogger("rollout")
logger.setLevel(logging.INFO) 
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def run_one_episode(policy, env, language, max_steps=200):
    """
    Runs a single inference rollout with the given policy + LIBERO environment.
    Returns (success_flag, total_reward)
    """
    obs = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        with torch.no_grad():
            action = policy.get_action(obs, language)
            action = action.cpu().clone().detach().tolist()[0]

        obs, reward, done, info = env.step(action)
        total_reward += reward
        logger.info(f"Step {step} | Reward: {reward:.3f}")


        if done:
            # LIBERO success is reported in info["success"]
            success = True if total_reward == 1 else False
            return success, total_reward

    # If never reached done:
    return False, total_reward


def main():
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count():", torch.cuda.device_count())
    assert torch.cuda.device_count() == 1, "More than one CUDA device available (device mismatch risk)"


    print("\nLoading SmolVLA policy...")
    policy = SmolVLALiberoPolicy(
        "HuggingFaceVLA/smolvla_libero",
        device="cuda"
    )
    print("Policy loaded.\n")

    task_success_rates = []
    full_results = {}

    for task_id in range(NUM_TASKS):
        print(f"\n=== Running Task {task_id} ({TASK_SUITE_NAME}) ===")
        env, language = make_libero_env(TASK_SUITE_NAME, task_id)
        print(f"Task language: {language}")

        successes = 0
        rewards = []

        for ep in range(EPISODES_PER_TASK):
            print(f"\nTask {task_id} | Episode {ep+1}/{EPISODES_PER_TASK}")
            success, total_reward = run_one_episode(policy, env, language, STEPS_PER_EPISODE)
            rewards.append(total_reward)

            print(f" → Success: {success} | Total reward: {total_reward:.3f}")

            if total_reward == 1:
                successes += 1

        env.close()

        # Compute task success rate
        success_rate = successes / EPISODES_PER_TASK
        task_success_rates.append(success_rate)

        full_results[f"task_{task_id}"] = {
            "success_rate": success_rate,
            "num_successes": successes,
            "num_episodes": EPISODES_PER_TASK,
            "rewards": rewards,
            "language": language,
        }

        print(f"\n>>> Task {task_id} Success Rate: {success_rate * 100:.2f}%")


    avg_success = sum(task_success_rates) / NUM_TASKS
    print("\n==========================")
    print(" FINAL BASELINE RESULTS")
    print("==========================")
    for i, sr in enumerate(task_success_rates):
        print(f"Task {i}: {sr * 100:.2f}%")

    print("--------------------------")
    print(f"Average Success: {avg_success * 100:.2f}%")
    print("--------------------------")


    import json
    with open("baseline_results_libero10.json", "w") as f:
        json.dump(full_results, f, indent=4)

    print("\nSaved results → baseline_results_libero10.json")


if __name__ == "__main__":
    main()
