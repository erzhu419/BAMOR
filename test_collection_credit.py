import copy
import unittest

import diagnose_collection_credit as diagnostic


def profile(rewards=(150., 300.), preference=(.5, .5), terminated=0, truncated=1):
    return dict(interactions=1500, total_raw_reward=list(rewards), mean_preference=list(preference),
                terminated=terminated, truncated=truncated)


def source_fixture():
    source = dict(environments={})
    for env in diagnostic.ENVIRONMENTS:
        runs = []
        for seed in diagnostic.SEEDS:
            collection = dict(uniform=profile(), mixed=profile((300., 300.)), stratified=profile(),
                              uniform_repeat=profile())
            branches = {name: dict(direct_utility=dict(mean=value, per_mode=[value] * 4), final_hv=value)
                        for name, value in (("base", 8.), ("old_only", 9.), ("uniform", 10.),
                                            ("mixed", 12.), ("stratified", 11.))}
            runs.append(dict(seed=seed, collection=collection, branches=branches))
        source["environments"][env] = dict(runs=runs)
    return source


class CollectionCreditTests(unittest.TestCase):
    def test_rewards_use_fixed_steps_and_equal_objective_mean(self):
        self.assertEqual(diagnostic.collection_signals(profile())["reward_rate"], .15)
        three = profile((150., 300., -450.), (1 / 3,) * 3)
        self.assertEqual(diagnostic.collection_signals(three)["reward_rate"], 0.)

    def test_survival_uses_termination_not_time_limit(self):
        a = diagnostic.collection_signals(profile(terminated=2, truncated=7))
        b = diagnostic.collection_signals(profile(terminated=2, truncated=0))
        self.assertEqual(a["survival_rate"], b["survival_rate"])
        self.assertLess(a["survival_rate"], 0.)

    def test_ties_fall_back_to_uniform_and_preference_mean_is_not_diversity(self):
        collection = dict(uniform=profile(), mixed=profile())
        _, choices = diagnostic.choices_from_collection(collection)
        self.assertEqual(set(choices.values()), {"uniform"})
        collection["mixed"]["mean_preference"] = [.6, .4]
        self.assertEqual(diagnostic.choices_from_collection(collection)[1]["mean_preference_balance"], "uniform")

    def test_endpoint_labels_never_change_choices_and_harm_is_retained(self):
        source = source_fixture()
        first = diagnostic.make_rows(source)
        source["environments"][diagnostic.ENVIRONMENTS[0]]["runs"][0]["branches"]["mixed"]["direct_utility"]["mean"] = -90.
        second = diagnostic.make_rows(source)
        self.assertEqual(first[diagnostic.ENVIRONMENTS[0]][0]["choices"], second[diagnostic.ENVIRONMENTS[0]][0]["choices"])
        result = diagnostic.summarize(second)
        primary = result["environments"][diagnostic.ENVIRONMENTS[0]]["policies"]["reward_rate"]
        self.assertEqual(primary["harmful_mixed_choices"], 1)
        self.assertEqual(primary["utility_minus_uniform"]["minimum"], -100.)
        self.assertFalse(result["primary_screen_passed"])

    def test_uniform_fallback_is_not_an_improvement_and_controls_are_separate(self):
        result = diagnostic.summarize(diagnostic.make_rows(source_fixture()))
        self.assertTrue(result["primary_screen_passed"])
        for env in diagnostic.ENVIRONMENTS:
            block = result["environments"][env]
            self.assertEqual(block["policies"]["survival_rate"]["utility_minus_uniform"]["mean"], 0.)
            self.assertEqual(block["controls"]["hindsight_oracle"]["oracle_utility_gap"], 0.)
            self.assertEqual(block["controls"]["always_mixed"]["utility_minus_uniform"]["mean"], 2.)

    def test_source_roster_collection_length_and_repeat_mismatch_are_rejected(self):
        original = source_fixture()
        for kind in ("seed", "length", "repeat"):
            source = copy.deepcopy(original)
            row = source["environments"][diagnostic.ENVIRONMENTS[0]]["runs"][0]
            if kind == "seed":
                row["seed"] += 5
            elif kind == "length":
                row["collection"]["mixed"]["interactions"] -= 1
            else:
                row["collection"]["uniform_repeat"]["total_raw_reward"][0] += 1
            with self.assertRaises(ValueError):
                diagnostic.make_rows(source)


if __name__ == "__main__":
    unittest.main()
