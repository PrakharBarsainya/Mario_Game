# 🍄 Super Mario Reinforcement Learning using Deep Q-Network (DQN)

This project implements a Deep Q-Network (DQN) agent that learns to play **Super Mario Bros** through Reinforcement Learning. Instead of following predefined rules, the agent interacts with the game environment, learns from rewards and penalties, and gradually improves its gameplay by maximizing cumulative rewards.

The agent uses a Convolutional Neural Network (CNN) to process game frames and estimate Q-values for each possible action. During training, it balances exploration and exploitation using an epsilon-greedy policy, stores experiences in a replay memory, and periodically updates a target network for stable learning. These are standard components of DQN-based agents. :contentReference[oaicite:0]{index=0}

## 🚀 Features

- Deep Q-Network (DQN) implementation using PyTorch
- Experience Replay Memory
- Target Network Synchronization
- Epsilon-Greedy Exploration Strategy
- CNN-based state representation
- Reward-based learning without human intervention
- Model saving and loading
- Performance visualization using training statistics

## 🛠️ Technologies Used

- Python
- PyTorch
- Gymnasium / gym-super-mario-bros
- NumPy
- OpenCV
- Matplotlib

## 🧠 How It Works

1. The agent observes the current game frame.
2. The DQN predicts Q-values for all available actions.
3. An action is selected using an epsilon-greedy strategy.
4. The environment returns the next state and reward.
5. The experience is stored in replay memory.
6. Mini-batches are sampled to train the neural network.
7. The target network is periodically updated for stable learning.

## 📊 Training Objectives

- Maximize total episode reward
- Learn efficient movement and jumping strategies
- Avoid enemies and obstacles
- Reach the end of the level with optimal performance

## 📈 Results

After sufficient training, the agent learns to:
- Move toward the goal efficiently
- Avoid common obstacles
- Perform jumps when necessary
- Improve cumulative rewards over time

## 📚 Learning Concepts

- Reinforcement Learning (RL)
- Deep Q-Network (DQN)
- Q-Learning
- Experience Replay
- Target Networks
- Convolutional Neural Networks (CNN)
- Epsilon-Greedy Exploration

## 🎯 Future Improvements

- Double DQN (DDQN)
- Dueling DQN
- Prioritized Experience Replay (PER)
- Rainbow DQN
- Multi-Level Training
- Transfer Learning

## 📄 References

- DeepMind's Deep Q-Network (DQN)
- PyTorch Reinforcement Learning Tutorials
- Gym Super Mario Bros Environment
