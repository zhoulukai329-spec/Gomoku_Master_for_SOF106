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
        # store every PPO hyperparameter on the instance so the update() method can reuse them later without needing extra args.
        self.size = size
        self.gamma = gamma # how much future reward matters
        self.eps_clip = eps_clip # PPO's clipping range for policy updates
        self.k_epochs = k_epochs # how many passes to make over one rollout
        self.entropy_coef = entropy_coef # how strongly to encourage exploration
        self.value_coef = value_coef # how strongly the critic loss counts
        self.max_grad_norm = max_grad_norm # gradient clipping limit for stability

        # train on GPU automatically if one is available, otherwise fall back to CPU so the code still runs everywhere.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # PPO needs two copies of the network:
        # 1.self.policy is the live network that gradients update.
        # 2.self.policy_old is a frozen snapshot used to sample moves and to compute the probability ratio in the PPO loss. Keeping it frozen during the k_epochs update loop is what makes "Proximal" Policy Optimization proximal in the first place.
        self.policy = GomokuNet(size).to(self.device)
        self.policy_old = GomokuNet(size).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_loss_fn = nn.MSELoss()

        # the buffer collects one full self-play rollout (states, the action taken at each state, the log-probability of that action, the reward received, and whether that step ended a game) before each PPO update consumes and clears it.
        self.buffer = {
            "states": [],
            "actions": [],
            "logprobs": [],
            "legal_masks": [],
            "temperatures": [],
            "players": [],
            "rewards": [],
            "is_terminals": [],
        }

    def clear_buffer(self):
        """Drop all rollout data after the update has finished."""
        # Reset every list back to empty so the next rollout starts clean.
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
        # bail out immediately if there is nothing to play (full board).
        if not legal_moves:
            return None

        # convert the raw board into the 3-channel network input, then add a batch dimension of size 1 since the network expects a batch of boards rather than a single board.
        state = preprocess_board(board, current_player)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

        # run inference using the frozen policy_old snapshot, not the live policy, so move selection stays stable during training. no_grad() skips building a gradient graph since we are not training here, only sampling a move.
        with torch.no_grad():
            logits, _ = self.policy_old(state_tensor)

        # mask out every illegal cell by setting its logit to a very large negative number, which becomes ~0 probability after softmax. This guarantees the network can never select an illegal move.
        scaled_logits = logits.squeeze(0)
        mask = torch.full((self.size * self.size,), -1e9, device=self.device)
        for row, col in legal_moves:
            mask[row * self.size + col] = 0.0

        # apply temperature scaling before softmax. A temperature below 1 sharpens the distribution, and a temperature above 1 flattens it. The floor of 1e-3 avoids a division by zero.
        temperature = max(float(temperature), 1e-3)
        masked_logits = (scaled_logits + mask) / temperature

        # choose the move. Deterministic mode (used for the GUI and evaluation matches) always takes the single most likely move. Stochastic mode (used during training) samples from the full distribution so the agent keeps exploring different lines of play.
        if deterministic:
            action = torch.argmax(masked_logits)
            dist = Categorical(logits=masked_logits)
            logprob = dist.log_prob(action)
        else:
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            logprob = dist.log_prob(action)

        # only training calls (record=True) need to remember this state/action/logprob triple, since only training will later run a PPO update over the recorded rollout.
        if record:
            self.buffer["states"].append(state)
            self.buffer["actions"].append(action.item())
            self.buffer["logprobs"].append(logprob.item())
            self.buffer["legal_masks"].append(mask.detach().cpu().numpy().astype(np.float32))
            self.buffer["temperatures"].append(temperature)
            self.buffer["players"].append(current_player)

        # convert the flat 0..size*size-1 action index back into (row, col) board coordinates for the caller to use.
        row, col = divmod(action.item(), self.size)
        return row, col

    def record_reward(self, reward, is_terminal):
        """Append the reward signal that belongs to the latest sampled move."""
        # This is called once right after select_action(..., record=True), so the buffer's rewards/is_terminals lists stay aligned index-for- index with its states/actions/logprobs lists.
        self.buffer["rewards"].append(reward)
        self.buffer["is_terminals"].append(is_terminal)

    def update(self):
        """Run one PPO optimization pass on the buffered rollout."""
        # if nothing was recorded since the last update, there is nothing to learn from, so return zeroed-out stats instead of crashing on an empty tensor.
        if not self.buffer["states"]:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "buffer_size": 0,
            }

        # turn the raw per-step rewards into discounted returns by walking the rollout backward from the end. Whenever a step is a terminal step (the game ended), the running total resets to zero first, which correctly separates multiple games that were stored back-to-back in the same buffer.
        returns = []
        discounted_reward = 0.0
        next_player = None
        for reward, is_terminal, player in zip(
            reversed(self.buffer["rewards"]),
            reversed(self.buffer["is_terminals"]),
            reversed(self.buffer["players"]),
        ):
            if is_terminal:
                discounted_reward = 0.0
                next_player = None
            elif next_player is not None and player != next_player:
                discounted_reward = -self.gamma * discounted_reward
            else:
                discounted_reward = self.gamma * discounted_reward
            discounted_reward = reward + discounted_reward
            returns.insert(0, discounted_reward)
            next_player = player

        # normalize the returns to roughly zero mean and unit variance. This keeps the value loss and advantage scale stable regardless of how raw reward magnitudes change during training.
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-7)

        # convert the buffered Python lists into tensors once, outside the k_epochs loop, so we are not repeating that conversion work on every epoch.
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
        old_legal_masks = torch.tensor(
            np.asarray(self.buffer["legal_masks"]),
            dtype=torch.float32,
            device=self.device,
        )
        old_temperatures = torch.tensor(
            self.buffer["temperatures"],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(-1)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # PPO reuses the same rollout for several gradient steps (k_epochs) instead of throwing the data away after one update, which makes training far more sample-efficient.
        for _ in range(self.k_epochs):
            # run the live (trainable) policy on the recorded states to get fresh logits and value estimates.
            logits, state_values = self.policy(old_states)
            state_values = state_values.squeeze(-1)

            # recompute the log-probability of the actions that were actually taken, under the *current* network weights (these will differ from old_logprobs as training proceeds).
            masked_logits = (logits + old_legal_masks) / old_temperatures.clamp_min(1e-3)
            dist = Categorical(logits=masked_logits)
            logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()

            # the advantage measures how much better an action's actual return was compared to the critic's expectation. state_values.detach() stops gradients from the value head from leaking into the policy advantage calculation.
            advantages = returns - state_values.detach()

            # the probability ratio compares how likely the current network is to take this action versus how likely it was when the data was originally collected (policy_old).
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surrogate_1 = ratios * advantages
            # clip the ratio so a single update step cannot push the policy too far away from policy_old. This clipping is the core trick that makes PPO stable compared to vanilla policy gradient methods.
            surrogate_2 = torch.clamp(
                ratios,
                1 - self.eps_clip,
                1 + self.eps_clip,
            ) * advantages

            # take the minimum of the two surrogate objectives, which is the standard "clipped surrogate" PPO loss.
            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
            # the critic is trained separately with plain MSE against the normalized returns.
            value_loss = self.value_loss_fn(state_values, returns)
            # combine policy loss, value loss, and an entropy bonus (subtracted, since higher entropy/more exploration should reduce the total loss) into one scalar to back-propagate.
            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy
            )

            # standard PyTorch optimization step, with gradient clipping added so one unusually large batch cannot blow up the network weights in a single step.
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # accumulate stats from each epoch so we can report the average across all k_epochs passes at the end.
            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()

        # now that the live policy has been updated k_epochs times, copy its weights into policy_old so the next rollout is sampled from the newly improved policy.
        self.policy_old.load_state_dict(self.policy.state_dict())

        # turn the accumulated totals into per-epoch averages.
        for key in stats:
            stats[key] /= self.k_epochs
        stats["buffer_size"] = len(self.buffer["states"])

        # the rollout has now been fully consumed by this update, so it is safe to clear it and start collecting the next one.
        self.clear_buffer()
        return stats

    def save(self, path):
        """Save the inference-ready policy snapshot to disk."""
        # make sure the destination folder exists before writing.
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # save policy_old rather than policy, since policy_old is always the fully-synced, "settled" version of the weights (the live policy mid-update is a transient state we don't want to ship).
        torch.save(self.policy_old.state_dict(), target)

    def load(self, path):
        """Load weights and keep both policy copies in sync."""
        # load the saved weights onto whichever device this agent is currently using (so a GPU-trained checkpoint still loads fine on a CPU-only machine, and vice versa).
        state_dict = torch.load(Path(path), map_location=self.device)
        # load into policy_old first
        self.policy_old.load_state_dict(state_dict)
        # then copy the same weights into the live policy, so both networks start from an identical, freshly-loaded state.
        self.policy.load_state_dict(self.policy_old.state_dict())


class GomokuAgent(PPOAgent):
    """Keep the old public class name while using the PPO implementation."""
