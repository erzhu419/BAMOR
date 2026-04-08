"""
BAMOR Replay Buffer

Multi-objective reward 适配: reward 从标量变为 np.ndarray[m]
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, last_episode_step=5000):
        self.capacity = capacity
        self.last_episode_step = last_episode_step
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward_vec, next_state, done):
        """
        Add new data to buffer.
        
        Args:
            state: np.ndarray
            action: np.ndarray
            reward_vec: np.ndarray[m] — 多目标奖励向量
            next_state: np.ndarray
            done: bool
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward_vec, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Randomly sample batch_size elements."""
        batch_indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in batch_indices]

        states, actions, reward_vecs, next_states, dones = zip(*batch)

        return (np.stack(states),
                np.stack(actions),
                np.stack(reward_vecs),      # [batch, m] — 保持向量形式
                np.stack(next_states),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
