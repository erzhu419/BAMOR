"""Combine unchanged return forecasts using past paired prediction errors."""

from episode_credit_trend import EpisodeCreditTrend


class EpisodeCreditBlend:
    def __init__(self, rate):
        self.baseline = EpisodeCreditTrend(rate)
        self.cross_product = 0.
        self.disagreement_energy = 0.
        self.comparisons = 0

    def reset(self):
        self.baseline.reset()
        self.cross_product = 0.
        self.disagreement_energy = 0.
        self.comparisons = 0

    def predict(self, start_step):
        before = self.baseline.predict(start_step)
        weight = (min(1., max(0., self.cross_product / self.disagreement_energy))
                  if self.disagreement_energy > 0. else 0.)
        prediction = before["ema_prediction"]
        if before["trend_prediction"] is not None:
            prediction = ((1. - weight) * prediction + weight * before["trend_prediction"])
        return dict(before, trend_weight=weight, blend_prediction=prediction,
            past_cross_product=self.cross_product,
            past_disagreement_energy=self.disagreement_energy, past_comparisons=self.comparisons)

    def observe(self, start_step, utility, before, detected_mixed):
        if detected_mixed:
            self.reset()
            return
        if before["trend_prediction"] is not None:
            ema_error = float(utility - before["ema_prediction"])
            trend_error = float(utility - before["trend_prediction"])
            difference = ema_error - trend_error
            self.cross_product += ema_error * difference
            self.disagreement_energy += difference ** 2
            self.comparisons += 1
        self.baseline.observe(start_step, utility)
