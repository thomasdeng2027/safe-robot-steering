import torch
import torch.nn.functional as F
import numpy as np
import argparse
import copy

from model.smolvla_policy import SmolVLALiberoPolicy
from env.env import make_libero_env

MAX_STEPS = 520
GROUP_SIZE = 4
UPDATE_EPOCHS = 2
GRPO_EPSILON = 0.2


def rollout_one_trajectory(env, policy_old, language):
    obs = env.reset()
    traj_obs, traj_actions, traj_logprobs = [], [], []
    total_reward = 0.0

    for step in range(MAX_STEPS):
        with torch.no_grad():
            action, log_prob, unsquished_action = policy_old.sample_action(obs, language)

        env_action = action.cpu().numpy()[0]
        traj_obs.append(obs)
        traj_actions.append(unsquished_action.detach())
        traj_logprobs.append(log_prob.detach())

        obs, reward, done, info = env.step(env_action)
        total_reward += float(reward)
        if done:
            break

    return {
        "obs": traj_obs,
        "actions": traj_actions,
        "logprobs": traj_logprobs,
        "reward": total_reward,
        "language": language,
    }


def sample_group_trajectories(env, policy_old, language, G):
    return [rollout_one_trajectory(env, policy_old, language) for _ in range(G)]


def compute_group_advantages(trajs):
    rewards = torch.tensor([t["reward"] for t in trajs], dtype=torch.float32)
    mean_r = rewards.mean()
    std_r = rewards.std()
    return (rewards - mean_r) / (std_r + 1e-8)


def compute_grpo_loss(policy_theta, policy_old, trajs, advantages, epsilon):
    group_losses = []

    for i, traj in enumerate(trajs):
        A_i = advantages[i]
        step_losses = []
        lang = traj["language"]

        for obs_t, unsquished_action_t, old_lp_t in zip(
            traj["obs"], traj["actions"], traj["logprobs"]
        ):
            new_lp = policy_theta.get_action_prob(obs_t, lang, unsquished_action_t.to(policy_theta.device))
            ratio = torch.exp(new_lp - old_lp_t.to(policy_theta.device))

            unclipped = ratio * A_i
            clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * A_i
            step_losses.append(-torch.min(unclipped, clipped))

        traj_loss = torch.stack(step_losses).mean()
        group_losses.append(traj_loss)

    return torch.stack(group_losses).mean()


def collect_batch(env_factory, languages, batch_size, policy_old):
    batch = []
    for i in range(batch_size):
        env, lang = env_factory()  # new task each time
        rollout_group = sample_group_trajectories(env, policy_old, lang, GROUP_SIZE)
        advantages = compute_group_advantages(rollout_group)
        batch.append((rollout_group, advantages, lang))
        env.close()
    return batch


def train_grpo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = SmolVLALiberoPolicy("HuggingFaceVLA/smolvla_libero", device=device)
    policy.set_log_std(-3.0)

    optimizer = torch.optim.AdamW(
        policy.policy.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95)
    )

    def env_factory():
        return make_libero_env(args.task_suite)

    policy_old = copy.deepcopy(policy)
    policy_old.eval()

    for update in range(args.num_updates):
        batch = collect_batch(env_factory, None, args.batch_size, policy_old)
        policy.train()

        epoch_losses = []
        for _ in range(UPDATE_EPOCHS):
            total_loss = 0.0
            for rollout_group, advantages, lang in batch:
                loss = compute_grpo_loss(policy, policy_old, rollout_group, advantages, GRPO_EPSILON)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.policy.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            epoch_losses.append(total_loss / len(batch))

        avg_loss = np.mean(epoch_losses)
        avg_reward = np.mean([np.mean([t["reward"] for t in g]) for g, _, _ in batch])

        print(f"[Update {update}] loss={avg_loss:.4f}, avg_reward={avg_reward:.3f}")

        policy_old = copy.deepcopy(policy)
        policy_old.eval()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite", type=str, default="libero_10")
    parser.add_argument("--batch-size", type=int, default=2)     # number of tasks per update
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-updates", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    train_grpo(args)


if __name__ == "__main__":
    main()
