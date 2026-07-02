"""
Super Mario Bros DDQN Agent
===========================

Features
--------
✔ Double Deep Q Network
✔ CNN feature extractor
✔ Experience Replay
✔ Target Network
✔ Automatic Checkpoint Resume
✔ Best Model Saving
✔ Replay Memory Saving
✔ Gradient Clipping
✔ Huber Loss
"""
import os
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from dqn import DQN
from wrappers import make_mario_env
from experience_replay import ReplayMemory

# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"\nUsing device: {DEVICE}\n")

# ---------------------------------------------------------
# Directories
# ---------------------------------------------------------

RUNS_DIR = "runs"
CHECKPOINT_DIR = os.path.join(RUNS_DIR, "checkpoints")

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

CHECKPOINT_EVERY = 50
MAX_GRAD_NORM = 10.0
FRAME_SKIP = 4


class Agent:
    def __init__(self, param_set: str):

        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_params = yaml.safe_load(f)
            p = all_params[param_set]

        self.alpha = p["alpha"]
        self.gamma = p["gamma"]

        self.epsilon = p["epsilon_init"]
        self.epsilon_min = p["epsilon_min"]
        self.epsilon_decay = p["epsilon_decay"]

        self.replay_size = p["replay_memory_size"]
        self.batch_size = p["mini_batch_size"]
        self.sync_rate = p["network_sync_rate"]

        self.save_improvement_threshold = p.get("save_improvement_threshold", 1.0)

        self.loss_fn = nn.SmoothL1Loss()

        self.MODEL_PATH = os.path.join(RUNS_DIR, f"{param_set}_best.pt")
        self.LATEST_CKPT = os.path.join(CHECKPOINT_DIR, f"{param_set}_latest.pt")

        self.start_episode = 0
        self.best_reward = float("-inf")
        self.optimizer_steps = 0

        self.memory = None
        self.policy_dqn = None
        self.target_dqn = None
        self.optimizer = None

    # ---------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------
    def load_checkpoint(self):

        if not os.path.exists(self.LATEST_CKPT):
            print("\n[NO CHECKPOINT] Starting fresh.\n")
            return

        print(f"\n[LOADING CHECKPOINT] {self.LATEST_CKPT}\n")

        checkpoint = torch.load(self.LATEST_CKPT, map_location=DEVICE)

        # Load model weights
        self.policy_dqn.load_state_dict(checkpoint["policy_state"])
        self.target_dqn.load_state_dict(checkpoint["target_state"])

        # Load optimizer if available
        if self.optimizer and checkpoint.get("optimizer_state") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        # Restore episode
        self.start_episode = checkpoint.get("episode", 0) + 1

        # FIXED epsilon restore (prevents collapse lock)
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.epsilon = max(self.epsilon, self.epsilon_min)
        self.epsilon = min(self.epsilon, 1.0)

        # Restore best tracking
        self.best_reward = checkpoint.get("best_reward", float("-inf"))
        self.optimizer_steps = checkpoint.get("optimizer_steps", 0)

        # Restore replay memory safely
        from collections import deque
        self.memory.memory = deque(
            checkpoint.get("memory", []),
            maxlen=self.replay_size
        )

        print("Checkpoint loaded successfully.")
        print(f"Resuming from episode: {self.start_episode}")
        print(f"Epsilon: {self.epsilon}")
        print(f"Best reward: {self.best_reward}\n")

    # ---------------------------------------------------------
    # Save checkpoint
    # ---------------------------------------------------------
    def save_checkpoint(self, episode):

        checkpoint = {
            "policy_state": self.policy_dqn.state_dict(),
            "target_state": self.target_dqn.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "episode": episode,
            "epsilon": self.epsilon,
            "best_reward": self.best_reward,
            "optimizer_steps": self.optimizer_steps,
            "memory": list(self.memory.memory)
        }

        torch.save(checkpoint, self.LATEST_CKPT)

        print(f"\n[CHECKPOINT SAVED] Episode {episode}\n")

    # ---------------------------------------------------------
    # Run training or evaluation
    # ---------------------------------------------------------
    def run(self, is_training=True, render=False):

        import numpy as np

        env = make_mario_env(render=render)

        input_channels = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # -------------------------------
        # CREATE MODELS
        # -------------------------------
        self.policy_dqn = DQN(input_channels, num_actions).to(DEVICE)
        self.target_dqn = DQN(input_channels, num_actions).to(DEVICE)

        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        self.target_dqn.eval()

        self.optimizer = optim.Adam(self.policy_dqn.parameters(), lr=self.alpha)
        self.memory = ReplayMemory(self.replay_size)

        # -------------------------------
        # LOAD CHECKPOINT / MODEL
        # -------------------------------
        if is_training:
            self.load_checkpoint()

            # 🔥 ESCAPE STAGNATION BOOST (NO RESET)
            if self.epsilon <= 0.10:
                print("[INFO] Boosting epsilon to escape local optimum")
                self.epsilon = 0.20

        else:
            if os.path.exists(self.MODEL_PATH):
                print(f"[LOADING BEST MODEL] {self.MODEL_PATH}")
                self.policy_dqn.load_state_dict(
                    torch.load(self.MODEL_PATH, map_location=DEVICE)
                )
            self.policy_dqn.eval()
            self.epsilon = 0.0

        # =====================================================
        # TRAINING LOOP
        # =====================================================
        if is_training:

            for episode in range(self.start_episode, 10**9):

                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=DEVICE)

                done = False
                episode_reward = 0

                while not done:

                    # epsilon-greedy
                    if random.random() < self.epsilon:
                        action = env.action_space.sample()
                    else:
                        with torch.no_grad():
                            q_values = self.policy_dqn(state.unsqueeze(0))
                            action = torch.argmax(q_values).item()

                    total_reward = 0

                    for _ in range(FRAME_SKIP):
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        total_reward += reward
                        done = terminated or truncated
                        if done:
                            break

                    next_state = torch.tensor(next_state, dtype=torch.float32, device=DEVICE)

                    self.memory.append((
                        state,
                        torch.tensor(action, device=DEVICE),
                        next_state,
                        torch.tensor(total_reward, device=DEVICE),
                        done
                    ))

                    state = next_state
                    episode_reward += total_reward

                    if len(self.memory) >= self.batch_size:
                        self.optimize()
                        self.optimizer_steps += 1

                        if self.optimizer_steps % self.sync_rate == 0:
                            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

                # epsilon decay
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

                print(f"Episode {episode} | Reward: {episode_reward:.1f}")

                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    torch.save(self.policy_dqn.state_dict(), self.MODEL_PATH)
                    print("[BEST MODEL SAVED]")

                if episode % CHECKPOINT_EVERY == 0:
                    self.save_checkpoint(episode)

        # =====================================================
        # TESTING LOOP (FIXED - avoids deterministic collapse feel)
        # =====================================================
        else:

            episode = 0

            while True:

                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=DEVICE)

                done = False
                episode_reward = 0

                while not done:

                    with torch.no_grad():
                        q_values = self.policy_dqn(state.unsqueeze(0))
                        q = q_values.cpu().numpy().squeeze()

                        # soft stochastic action (stable evaluation)
                        exp_q = np.exp(q - np.max(q))
                        probs = exp_q / np.sum(exp_q)
                        action = int(np.random.choice(len(probs), p=probs))

                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    state = torch.tensor(next_state, dtype=torch.float32, device=DEVICE)
                    episode_reward += reward

                episode += 1
                print(f"[TEST] Episode {episode} | Reward: {episode_reward:.1f}")

    # ---------------------------------------------------------
    # OPTIMIZATION (DDQN)
    # ---------------------------------------------------------
    def optimize(self):

        batch = self.memory.sample(self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions, device=DEVICE)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, device=DEVICE, dtype=torch.float32)
        dones = torch.tensor(dones, device=DEVICE, dtype=torch.float32)

        q_values = self.policy_dqn(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_dqn(next_states).argmax(dim=1)
            next_q = self.target_dqn(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)

            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_dqn.parameters(),
            MAX_GRAD_NORM
        )

        self.optimizer.step()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("hyperparameters")
    parser.add_argument("--train", action="store_true")

    args = parser.parse_args()

    agent = Agent(args.hyperparameters)

    agent.run(
        is_training=args.train,
        render=not args.train
    )