"""
GomokuNet — residual actor-critic network for Gomoku.

The network provides a policy head for move selection and a value head for
state evaluation during PPO training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Apply two convolutions and add the original input back in."""

    def __init__(self, channels):
        super().__init__()
        # Both convolutions keep the same channel count and board size.
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        # Save the input so the block can learn a residual correction.
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class GomokuNet(nn.Module):
    """Share board features, then split them into policy and value heads."""

    def __init__(self, size=15, num_channels=128, num_res_blocks=5):
        super().__init__()
        self.size = size

        # Read the three input planes and lift them into a wider feature space.
        self.stem = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )

        # Stack several residual blocks to model local and global board patterns.
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # The policy head predicts one logit for every board cell.
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(32)
        self.policy_fc   = nn.Linear(32 * size * size, size * size)

        # The value head estimates how good the position is for the current player.
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.value_bn   = nn.BatchNorm2d(32)
        self.value_fc1  = nn.Linear(32 * size * size, 256)
        self.value_fc2  = nn.Linear(256, 1)

        # Initialize weights once so training starts from a stable state.
        self._init_weights()

    def _init_weights(self):
        """Apply standard initialization rules for each layer type."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Build a shared board representation first.
        x = self.stem(x)
        x = self.res_blocks(x)

        # Convert shared features into move logits.
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Convert shared features into one bounded position score.
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


def preprocess_board(board, current_player):
    """Turn the board into three channels from the current player's view."""
    size = board.shape[0]
    opponent = 3 - current_player

    # Channel order keeps the network focused on self, opponent, then empties.
    state = np.zeros((3, size, size), dtype=np.float32)
    state[0] = (board == current_player).astype(np.float32)
    state[1] = (board == opponent).astype(np.float32)
    state[2] = (board == 0).astype(np.float32)
    return state
