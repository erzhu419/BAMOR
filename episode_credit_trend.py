"""Past-only return forecasts using the pre-episode physical training clock."""

from collections import deque

import numpy as np

from episode_credit_baseline import EpisodeCreditBaseline, INITIAL_EPISODES


TREND_WINDOW = 8


class EpisodeCreditTrend:
    def __init__(self, rate):
        self.scale = EpisodeCreditBaseline(rate)
        self.history = deque(maxlen=TREND_WINDOW)

    def reset(self):
        self.scale.reset()
        self.history.clear()

    def predict(self, start_step):
        before = self.scale.snapshot()
        prediction = None
        if len(self.history) >= INITIAL_EPISODES:
            steps, returns = np.asarray(self.history, dtype=float).T
            design = np.column_stack((np.ones(len(steps)), (steps - start_step) / 1000))
            prediction = float(np.linalg.lstsq(design, returns, rcond=None)[0][0])
        return dict(ema_prediction=before["mean"], trend_prediction=prediction,
                    scale=before["scale"], fit_samples=len(self.history))

    def observe(self, start_step, utility):
        self.scale.observe(utility)
        self.history.append((start_step, utility))
