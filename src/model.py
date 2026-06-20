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
        # Two 3x3 convolutions that keep the channel count and spatial size unchanged, so the block's output can always be added back to its input.
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        # remember the original input so it can be added back later (this is the "residual" or "skip connection" that helps deep networks train without their gradients vanishing).
        residual = x
        # first conv + batch norm + ReLU.
        out = F.relu(self.bn1(self.conv1(x)))
        # second conv + batch norm, no activation yet.
        out = self.bn2(self.conv2(out))
        # add the original input back, then apply the final ReLU.
        return F.relu(out + residual)


class GomokuNet(nn.Module):
    """Share board features, then split them into policy and value heads."""

    def __init__(self, size=15, num_channels=128, num_res_blocks=5):
        super().__init__()
        self.size = size

        # the "stem" reads the 3-channel board input (described in preprocess_board below) and projects it into a wider feature space that the rest of the network can work with.
        self.stem = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )

        # stack several residual blocks back to back. Each block adds more capacity to recognize local shapes (like four-in-a-rows) and, because of the skip connections, longer-range board patterns too.
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # the policy head turns the shared features into one logit per board cell, i.e. "how good does the network think this move is".
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(32)
        self.policy_fc   = nn.Linear(32 * size * size, size * size)

        # the value head turns the shared features into a single number estimating how good the current position is for whichever player is about to move (used as the PPO "critic").
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.value_bn   = nn.BatchNorm2d(32)
        self.value_fc1  = nn.Linear(32 * size * size, 256)
        self.value_fc2  = nn.Linear(256, 1)

        # apply sensible starting weights so training begins from a numerically stable point instead of PyTorch's raw random defaults.
        self._init_weights()

    def _init_weights(self):
        """Apply standard initialization rules for each layer type."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # initialization the standard choice for layers that are followed by a ReLU activation.
                nn.init.normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # Start every batch-norm layer as an identity transform (scale 1, shift 0) so it does not distort the signal at the very start of training.
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # build one shared board representation that both the policy head and value head will read from.
        x = self.stem(x)
        x = self.res_blocks(x)

        # convert the shared features into one move-quality logit per board cell. 
        # These are raw logits (not yet probabilities) so that the caller can apply legal-move masking before softmax.
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # convert the shared features into a single value estimate,
        # squashed into [-1, 1] with tanh, where positive values mean the position favors the current player and negative means it does not.
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


def preprocess_board(board, current_player):
    """Turn the board into three channels from the current player's view."""
    size = board.shape[0]
    opponent = 3 - current_player

    # build three separate 0/1 "planes" the same size as the board.
    # Channel 0 marks the current player's own stones, channel 1 marks the opponent's stones, and channel 2 marks empty cells. 
    # Using the current player's own perspective (instead of always "black" / "white") lets the same network play both colors without needing to be told which color it currently is.
    state = np.zeros((3, size, size), dtype=np.float32)
    state[0] = (board == current_player).astype(np.float32)
    state[1] = (board == opponent).astype(np.float32)
    state[2] = (board == 0).astype(np.float32)
    return state