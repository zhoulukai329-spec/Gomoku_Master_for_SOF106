"""Neural PPO agent used by both training and gameplay."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from model import GomokuNet, preprocess_board


class PPOAgent:
    """Wrap the policy network, replay buffer, and PPO update step."""

    def __init__(
        self,
        size=15,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        entropy_coef=0.02,
        value_coef=0.5,
        max_grad_norm=1.0,
    ):
        # Store the PPO hyperparameters so training can reuse them later.
        self.size = size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # Use GPU when available, otherwise fall back to CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Keep one trainable network and one frozen copy for stable PPO sampling.
        self.policy = GomokuNet(size).to(self.device)
        self.policy_old = GomokuNet(size).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_loss_fn = nn.MSELoss()

        # Store one full rollout before each PPO update.
        self.buffer = {
            "states": [],
            "actions": [],
            "logprobs": [],
            "rewards": [],
            "is_terminals": [],
        }

    def clear_buffer(self):
        """Drop all rollout data after the update has finished."""
        for key in self.buffer:
            self.buffer[key] = []

    def select_action(
        self,
        board,
        current_player,
        legal_moves,
        temperature=1.0,
        deterministic=False,
        record=False,
    ):
        """Pick one legal move from the current board state."""
        if not legal_moves:
            return None

        # Convert the board into the neural-network input format.
        state = preprocess_board(board, current_player)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

        # Run inference with the frozen policy snapshot used for action sampling.
        with torch.no_grad():
            logits, _ = self.policy_old(state_tensor)

        # Mask illegal cells so the policy can only choose valid moves.
        scaled_logits = logits.squeeze(0)
        mask = torch.full((self.size * self.size,), -1e9, device=self.device)
        for row, col in legal_moves:
            mask[row * self.size + col] = 0.0

        # A lower temperature makes the move choice more greedy.
        temperature = max(float(temperature), 1e-3)
        probs = torch.softmax((scaled_logits + mask) / temperature, dim=-1)

        # Use argmax for deterministic play and sampling for exploration.
        if deterministic:
            action = torch.argmax(probs)
            logprob = torch.log(probs[action].clamp_min(1e-8))
        else:
            dist = Categorical(probs=probs)
            action = dist.sample()
            logprob = dist.log_prob(action)

        # Training records the sampled state-action pair for PPO later.
        if record:
            self.buffer["states"].append(state)
            self.buffer["actions"].append(action.item())
            self.buffer["logprobs"].append(logprob.item())

        # Convert the flat action index back to board coordinates.
        row, col = divmod(action.item(), self.size)
        return row, col

    def record_reward(self, reward, is_terminal):
        """Append the reward signal that belongs to the latest sampled move."""
        self.buffer["rewards"].append(reward)
        self.buffer["is_terminals"].append(is_terminal)

    def update(self):
        """Run one PPO optimization pass on the buffered rollout."""
        if not self.buffer["states"]:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "buffer_size": 0,
            }

        # Build discounted returns from the end of the episode back to the start.
        returns = []
        discounted_reward = 0.0
        for reward, is_terminal in zip(
            reversed(self.buffer["rewards"]),
            reversed(self.buffer["is_terminals"]),
        ):
            if is_terminal:
                discounted_reward = 0.0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        # Normalize returns to keep training numerically stable.
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-7)

        # Turn the collected rollout into tensors once before the PPO loop.
        old_states = torch.tensor(
            np.asarray(self.buffer["states"]),
            dtype=torch.float32,
            device=self.device,
        )
        old_actions = torch.tensor(self.buffer["actions"], device=self.device)
        old_logprobs = torch.tensor(
            self.buffer["logprobs"],
            dtype=torch.float32,
            device=self.device,
        )

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Repeat the PPO update over the same rollout for a few epochs.
        for _ in range(self.k_epochs):
            logits, state_values = self.policy(old_states)
            state_values = state_values.squeeze(-1)

            # Recompute the action distribution under the current network weights.
            dist = Categorical(logits=logits)
            logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()

            # PPO compares the new policy to the old sampling policy.
            advantages = returns - state_values.detach()
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surrogate_1 = ratios * advantages
            surrogate_2 = torch.clamp(
                ratios,
                1 - self.eps_clip,
                1 + self.eps_clip,
            ) * advantages

            # Policy loss handles action quality, value loss handles state quality.
            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
            value_loss = self.value_loss_fn(state_values, returns)
            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy
            )

            # Clip gradients so one unstable batch does not explode the update.
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Accumulate averaged statistics for logging.
            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()

        # Sync the frozen sampling policy with the newly trained weights.
        self.policy_old.load_state_dict(self.policy.state_dict())

        for key in stats:
            stats[key] /= self.k_epochs
        stats["buffer_size"] = len(self.buffer["states"])

        # The rollout has been consumed, so it can be cleared safely.
        self.clear_buffer()
        return stats

    def save(self, path):
        """Save the inference-ready policy snapshot to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_old.state_dict(), target)

    def load(self, path):
        """Load weights and keep both policy copies in sync."""
        state_dict = torch.load(Path(path), map_location=self.device)
        self.policy_old.load_state_dict(state_dict)
        self.policy.load_state_dict(self.policy_old.state_dict())


class GomokuAgent(PPOAgent):
    """Keep the old public class name while using the PPO implementation."""
