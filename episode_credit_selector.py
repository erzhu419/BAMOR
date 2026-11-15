"""Select a return forecaster using only previously completed prediction errors."""

from episode_credit_trend import EpisodeCreditTrend


class EpisodeCreditSelector:
    def __init__(self, rate):
        self.baseline = EpisodeCreditTrend(rate)
        self.squared_errors = dict(ema=0., trend=0.)
        self.comparisons = 0

    def reset(self):
        self.baseline.reset()
        self.squared_errors = dict(ema=0., trend=0.)
        self.comparisons = 0

    def predict(self, start_step):
        before = self.baseline.predict(start_step)
        selected = "trend" if self.squared_errors["trend"] < self.squared_errors["ema"] else "ema"
        return dict(before, selected=selected, selected_prediction=before[f"{selected}_prediction"],
                    past_squared_errors=dict(self.squared_errors), past_comparisons=self.comparisons)

    def observe(self, start_step, utility, before, detected_mixed):
        if detected_mixed:
            self.reset()
            return
        if before["trend_prediction"] is not None:
            for name in self.squared_errors:
                self.squared_errors[name] += float(utility - before[f"{name}_prediction"]) ** 2
            self.comparisons += 1
        self.baseline.observe(start_step, utility)
