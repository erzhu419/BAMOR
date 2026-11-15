"""Share progress credit across equivalent own-preference operator labels."""

import train_bamor_mujoco as training


class SharedCreditOwnPreferenceSelector(training.OwnPreferenceBothProgressExtremeOperatorSelector):
    def __init__(self, reward_dim, block_steps=1500, learning_starts=5000):
        super().__init__(reward_dim, block_steps, learning_starts)
        groups = {}
        for op in self.operators:
            sampler = op.split(":")[1].split("_")[0]
            key = (sampler, self.relabel_rate(op), tuple(sorted(self.trainer_config(op).items())))
            groups.setdefault(key, []).append(op)
        self.credit_groups = list(groups.values())
        self.credit_aliases = {op: group for group in self.credit_groups for op in group}

    def maybe_finish_block(self, step, force=False):
        op = self.current_operator
        super().maybe_finish_block(step, force)
        if op is not None and self.current_operator is None and self.block_history[-1]["complete"]:
            # Update once per observed block, then expose the same EMA to every alias.
            for alias in self.credit_aliases[op]:
                self.scores[alias] = self.scores[op]

    def state_dict(self):
        return {**super().state_dict(), "score_credit_rule": "effective_configuration_shared_ema",
                "credit_groups": [list(group) for group in self.credit_groups]}
