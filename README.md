# 🍄 Super Mario Bros Deep Q-Network (DQN) Reinforcement Learning Agent

This project implements a **Deep Q-Network (DQN) / Double DQN (DDQN) agent** that learns to play :contentReference[oaicite:0]{index=0} using Reinforcement Learning.

The agent interacts directly with the environment, learns from rewards, and gradually improves gameplay through trial and error using deep neural networks.

---

# 🚀 Features

- Deep Q-Network (DQN) / Double DQN (DDQN) implementation using PyTorch
- Experience Replay Memory for stable learning
- Target Network synchronization
- Epsilon-Greedy exploration strategy
- CNN-based feature extraction from game frames
- Frame stacking for temporal awareness
- Frame skipping for faster training
- Checkpoint saving and automatic resume support
- Best model saving based on reward improvement
- Continuous training and evaluation modes
- Stuck detection mechanism to avoid infinite loops
- GPU/CPU automatic device selection

---

# 🛠️ Technologies Used

- Python
- PyTorch
- gymnasium + gym-super-mario-bros
- NumPy
- OpenCV
- PyYAML

---

# 🧠 How It Works

1. The agent observes stacked grayscale game frames
2. A CNN extracts features from the environment state
3. The DQN predicts Q-values for each action
4. Action is selected using epsilon-greedy policy
5. The environment returns next state and reward
6. Transitions are stored in replay memory
7. Mini-batches are sampled to train the network
8. Target network is periodically updated for stability
9. Training continues across episodes with checkpointing

---

# 🎮 Environment Details

- Game: :contentReference[oaicite:1]{index=1}
- Action Space: RIGHT_ONLY (movement + jump actions)
- Observation: 4 stacked grayscale frames (84×84)
- Frame Skip: 4
- Reward: Environment-defined game score

---

# 📊 Training Objectives

- Maximize total episode reward
- Learn efficient movement and jumping
- Avoid enemies and obstacles
- Reach the level end efficiently
- Escape local optima and stuck states

---

# 📈 Training Features

- Automatic checkpoint saving and resume
- Best model tracking
- Epsilon decay with recovery boosting
- Experience replay buffer
- Gradient clipping for stability
- Huber loss (Smooth L1 loss)

---

# 🧠 Key Learning Concepts

- Reinforcement Learning (RL)
- Deep Q-Learning (DQN)
- Double DQN (DDQN)
- Q-Learning
- Experience Replay
- Target Networks
- Convolutional Neural Networks (CNNs)
- Exploration vs Exploitation (Epsilon-Greedy)

---

# ⚙️ Project Structure
Mario_Game/
│
├── agent.py # Main training & testing loop
├── dqn.py # CNN-based Q-network
├── wrappers.py # Environment preprocessing
├── experience_replay.py # Replay buffer
├── parameters.yaml # Hyperparameters
│
├── runs/
│ ├── checkpoints/ # Auto-saved training checkpoints
│ └── *_best.pt # Best model weights
│
└── venv/ # Virtual environment (ignored in git)


---

# 🎯 Training Modes

### Train:
```bash
python agent.py default --train

#🎮 Test (Run Trained Agent Continuously)
python agent.py default

### ✔ Testing mode
- runs infinite loop
- does NOT train
- just evaluates policy

---
