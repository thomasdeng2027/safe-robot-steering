import torch
from torch import nn
import numpy as np
# torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
from model.smolvla_policy import SmolVLALiberoPolicy
# from gymnasium.vector import AsyncVectorEnv
# from env import libero_factory
from env.env import make_libero_env
import copy

# TODO: if want to vectorize envs, finish that by figuring out how to map LIBERO observations to SmolVLA compatible ones in env.py
# TODO: but probably for now, get things working with one environment first. Need to figure out SmolVLA input/output format exactly
# (may expect multiple camera views, different amount of joints, etc.)

TASK_SUITE_NAME = "libero_10" # long-range tasks
NUM_ENVS = 8
MAX_STEPS = 520

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = SmolVLALiberoPolicy(
        "HuggingFaceVLA/smolvla_libero", device=device
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=1e-5,
        betas=(0.9, 0.95)
    )
    
    # switch task_id to random generation during training
    env, language = make_libero_env(TASK_SUITE_NAME)
    # factories = [libero_factory(TASK_SUITE_NAME) for _ in range(NUM_ENVS)]
    # vec_env = AsyncVectorEnv(factories)
    # print(vec_env)

    policy_old = copy.deepcopy(policy)
    obs = env.reset()
    
    for step in range(MAX_STEPS):
        with torch.no_grad():
            action, log_prob, unsquished_action = policy_old.sample_action(obs, language)
            action = action.cpu().clone().detach().tolist()[0]
        obs, reward, done, info = env.step(action)
        
        print(f"Step {step} | Reward: {reward:.3f}")
        if done:
            break

        # TODO: autocast, compile (might only be able to compile certain parts since some loop iterations)

if __name__ == "__main__":
    main()
