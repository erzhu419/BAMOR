"""JAX-friendly replay buffer with task_id support and multi-objective rewards."""
import numpy as np


class ReplayBuffer:
    """Fixed-size numpy replay buffer storing (s, a, r_vec, s', done, task_id).

    BAMOR extension: reward stored as [capacity, num_objectives] instead of [capacity, 1].
    """

    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 1_000_000,
                 num_objectives: int = 1):
        self.capacity = capacity
        self.num_objectives = num_objectives
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, num_objectives), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.task_id = np.zeros((capacity,), dtype=np.int32)

    def push(self, obs, act, rew, next_obs, done, task_id=0):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = float(done)
        self.task_id[self.ptr] = task_id
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, obs, act, rew, next_obs, done, task_id=None):
        """Push a batch of transitions. Arrays should be [batch, ...]."""
        n = obs.shape[0]
        obs = np.asarray(obs)
        act = np.asarray(act)
        rew = np.asarray(rew)
        if rew.ndim == 1:
            rew = rew.reshape(-1, 1)
        # For single-objective compat: [batch, 1] → ok
        # For multi-objective: [batch, m] → ok
        if rew.shape[1] != self.num_objectives:
            # Broadcast if needed (e.g. old code passing [batch, 1])
            rew = np.broadcast_to(rew, (n, self.num_objectives)).copy()
        next_obs = np.asarray(next_obs)
        done = np.asarray(done).reshape(-1, 1) if done.ndim == 1 else np.asarray(done)
        if task_id is not None:
            task_id = np.asarray(task_id).astype(np.int32)
        else:
            task_id = np.zeros(n, dtype=np.int32)

        if self.ptr + n <= self.capacity:
            self.obs[self.ptr:self.ptr + n] = obs
            self.act[self.ptr:self.ptr + n] = act
            self.rew[self.ptr:self.ptr + n] = rew
            self.next_obs[self.ptr:self.ptr + n] = next_obs
            self.done[self.ptr:self.ptr + n] = done
            self.task_id[self.ptr:self.ptr + n] = task_id
        else:
            first = self.capacity - self.ptr
            self.obs[self.ptr:] = obs[:first]
            self.act[self.ptr:] = act[:first]
            self.rew[self.ptr:] = rew[:first]
            self.next_obs[self.ptr:] = next_obs[:first]
            self.done[self.ptr:] = done[:first]
            self.task_id[self.ptr:] = task_id[:first]
            rest = n - first
            self.obs[:rest] = obs[first:]
            self.act[:rest] = act[first:]
            self.rew[:rest] = rew[first:]
            self.next_obs[:rest] = next_obs[first:]
            self.done[:rest] = done[first:]
            self.task_id[:rest] = task_id[first:]

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator = None):
        if rng is None:
            idx = np.random.randint(0, self.size, size=batch_size)
        else:
            idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "act": self.act[idx],
            "rew": self.rew[idx],       # [batch, m]
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
            "task_id": self.task_id[idx],
        }

    def sample_stacked(self, n_batches: int, batch_size: int):
        """Pre-sample n_batches stacked for lax.scan. Shape: [n_batches, batch_size, ...]."""
        all_idx = np.random.randint(0, self.size, size=(n_batches, batch_size))
        return {
            "obs": self.obs[all_idx],
            "act": self.act[all_idx],
            "rew": self.rew[all_idx],       # [N, B, m]
            "next_obs": self.next_obs[all_idx],
            "done": self.done[all_idx],
            "task_id": self.task_id[all_idx],
        }

    def __len__(self):
        return self.size
