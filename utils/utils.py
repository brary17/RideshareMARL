# Obtained from https://github.com/seolhokim/Mujoco-Pytorch

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
            self, 
            max_size, 
            state_dim, 
            num_action, 
            device=torch.device("cpu")
        ):
        self.max_size = max_size
        self.data_idx = 0
        self.device = device
        self.end_pointer = 0

        # Pre-allocate tensors for buffer data
        self.states: torch.Tensor = torch.zeros((max_size, state_dim), device=self.device)
        self.actions: torch.Tensor = torch.zeros((max_size, num_action), device=self.device)
        self.rewards: torch.Tensor = torch.zeros((max_size, 1), device=self.device)
        self.next_states: torch.Tensor = torch.zeros((max_size, state_dim), device=self.device)
        self.dones: torch.Tensor = torch.zeros((max_size, 1), device=self.device)
        self.log_probs: torch.Tensor = torch.zeros((max_size, 1), device=self.device)

    def put_data(self, state, action, reward, next_state, done, log_probs):
        state = state if isinstance(state, torch.Tensor) else torch.tensor(state, device=self.device)
        action = action if isinstance(action, torch.Tensor) else torch.tensor(action, device=self.device)
        reward = reward if isinstance(reward, torch.Tensor) else torch.tensor(reward, device=self.device)
        next_state = next_state if isinstance(next_state, torch.Tensor) else torch.tensor(next_state, device=self.device)
        done = done if isinstance(done, torch.Tensor) else torch.tensor(done, device=self.device)

        self.states[self.data_idx] = state
        self.actions[self.data_idx] = action.flatten()
        self.rewards[self.data_idx] = reward
        self.next_states[self.data_idx] = next_state
        self.dones[self.data_idx] = done
        self.log_probs[self.data_idx] = log_probs

        self.data_idx = (self.data_idx + 1) % self.max_size
        self.end_pointer = min(self.end_pointer + 1, self.max_size)

    def sample(self, shuffle, batch_size=None, time_steps=1):
        if not shuffle or batch_size is None:
            max_idx = self.size()
            return (
                self.states[:max_idx], 
                self.actions[:max_idx], 
                self.rewards[:max_idx], 
                self.next_states[:max_idx], 
                self.dones[:max_idx], 
                self.log_probs[:max_idx],
            )
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        time_offsets = torch.arange(time_steps, device=self.device).unsqueeze(0)  # Shape: (1, time_steps)

        start_indices = indices.unsqueeze(1) - time_offsets  # Shape: (batch_size, time_steps)
        start_indices = start_indices.clamp(min=0)  # Clamp to ensure no negative indices

        sampled_states = self.states[start_indices]  # Shape: (batch_size, time_steps, state_dim)
        sampled_actions = self.actions[start_indices]  # Shape: (batch_size, time_steps, ...)
        sampled_rewards = self.rewards[start_indices]  # Shape: (batch_size, time_steps)
        sampled_next_states = self.next_states[start_indices]  # Shape: (batch_size, time_steps, state_dim)
        sampled_dones = self.dones[start_indices]  # Shape: (batch_size, time_steps)

        return sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones

    def size(self):
        return min(self.max_size, self.end_pointer)

    def make_mini_batch(self, shuffle, mini_batch_size, get_gae):
        data = self.sample(shuffle, mini_batch_size)
        states, actions, rewards, next_states, dones, old_log_probs = data
        old_values, advantages = get_gae(states, rewards, next_states, dones)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-3)

        index_func = torch.randperm if shuffle else torch.arange

        full_batch_size = states.size(0)
        full_indices = index_func(full_batch_size, device=self.device)
        for i in range((full_batch_size // mini_batch_size)+1):
                indices = full_indices[mini_batch_size * i: min(mini_batch_size * (i + 1), len(full_indices))]
                yield (
                    states[indices],
                    actions[indices],
                    old_log_probs[indices],
                    advantages[indices],
                    returns[indices],
                    old_values[indices],
                )


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count