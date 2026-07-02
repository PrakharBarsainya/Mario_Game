import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    """
    Double Deep Q-Network for Mario using CNN to process pixel observations.
    Input: stacked grayscale frames (4, 84, 84)
    Output: Q-values for each action
    """
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()

        # Convolutional feature extractor (same architecture as DeepMind DQN)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),              # (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),              # (64, 7, 7)
            nn.ReLU(),
        )

        # Fully connected head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        # x: (batch, channels, H, W), normalized to [0, 1]
        x = x / 255.0
        x = self.conv(x)
        return self.fc(x)
