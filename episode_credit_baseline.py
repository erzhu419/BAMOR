"""Causal episode-return scale initialization, independent of routing."""

import math


INITIAL_EPISODES = 3


class EpisodeCreditBaseline:
    def __init__(self, rate):
        self.rate = rate
        self.reset()

    def reset(self):
        self.count = 0
        self.mean = None
        self.sample_mean = self.sample_m2 = 0.
        self.variance = None

    def snapshot(self):
        return dict(count=self.count, mean=self.mean,
                    scale=math.sqrt(self.variance) if self.variance is not None else None)

    def score(self, utility, before):
        if before["scale"] is None:
            return None
        return math.tanh((utility - before["mean"]) / before["scale"])

    def observe(self, utility):
        delta = utility - self.mean if self.mean is not None else 0.
        self.mean = utility if self.mean is None else self.mean + self.rate * delta
        self.count += 1
        if self.variance is not None:
            self.variance = (1 - self.rate) * (self.variance + self.rate * delta ** 2)
        else:
            # The current episode can seed only the next episode's scale.
            sample_delta = utility - self.sample_mean
            self.sample_mean += sample_delta / self.count
            self.sample_m2 += sample_delta * (utility - self.sample_mean)
            if self.count >= INITIAL_EPISODES and self.sample_m2 > 0:
                self.variance = self.sample_m2 / (self.count - 1)
