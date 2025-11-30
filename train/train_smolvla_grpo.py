import torch
from torch import nn
import numpy as np
# torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
from model.smolvla_policy import SmolVLALiberoPolicy
# from gymnasium.vector import AsyncVectorEnv
# from env import libero_factory
import torch.nn.functional as F
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

    


        # TODO: autocast, compile (might only be able to compile certain parts since some loop iterations)

def rollout_one_trajectory(env, policy_old, language):
    """Run one full episode using policy_old.
    Return a dict with:
        - obs: list of observations
        - actions: list of actions
        - logprobs: list of log-probabilities from policy_old
        - reward: episodic reward (scalar)
    """
    obs = env.reset()
    traj_obs = []
    traj_actions = []
    traj_logprobs = []
    total_reward = 0

    for step in range(MAX_STEPS):
        with torch.no_grad():
            action, log_prob, unsquished_action = policy_old.sample_action(obs, language)
            
        action = action.cpu().clone().detach().tolist()[0]
        traj_actions.append(unsquished_action)
        traj_logprobs.append(log_prob)

        obs, reward, done, info = env.step(action)
        traj_obs.append(obs)
        total_reward += reward        
        print(f"Step {step} | Reward: {reward:.3f}")
        if done:
            break
    return {
        "obs": traj_obs,
        "actions": traj_actions,
        "logprobs": traj_logprobs,
        "reward": total_reward
    }

def sample_group_trajectories(env, policy_old, language, G):
    """
    Samples G trajectories for a single GRPO group.
    Returns a list of dicts from rollout_one_trajectory.
    """
    rollouts = []
    for _ in range(G):
        rollouts.append(rollout_one_trajectory(env, policy_old, language))
    return rollouts

def compute_group_advantages(trajs):

    rewards = torch.tensor([traj["reward"] for traj in trajs], dtype=torch.float32)
    mean_r = rewards.mean()
    print(rewards)
    advantages = (rewards - mean_r)/(rewards.std() + 1e-8)
    return advantages

def compute_grpo_objective(theta_logits, rollouts, old_policy_log_probs, advantages, epsilon, num_groups):
    theta_log_probs = F.log_softmax(theta_logits, dim=-1)
    theta_log_probs = torch.gather(theta_log_probs, dim=-1, index=rollouts.unsqueeze(-1)) #??? not sure if i have to unsqueeze or not
    theta_log_probs = theta_log_probs.squeeze(-1) # remove dimension added for dimensions to match (which gather needs)
    # left shift old_policy to align with next token distributions from policy_theta

    ratios = torch.exp(theta_log_probs - old_policy_log_probs)
    unclipped = ratios * advantages.unsqueeze(1) # will keep for now, not sure if it has to stay there
    clipped = torch.clip(ratios, min=(1 - epsilon), max=(1 + epsilon)) * advantages.unsqueeze(1)
    # rollout_attention_mask works here since it marks every position to make a prediction at

    grpo_objective = torch.min(unclipped, clipped)
    # average over each rollout (note averages over rollouts then groups have to be separate; can't just do
    grpo_objective = torch.mean(grpo_objective, dim=-1) 
    # average over groups (not over all rollouts)
    grpo_objective = torch.sum(grpo_objective) / num_groups
    return grpo_objective





        


if __name__ == "__main__":
    main()
