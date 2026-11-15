"""Screen one cheap whole-policy return gate using already-paid paired trials."""

import argparse
import json
import math
from pathlib import Path
import random
import statistics


SOURCE = Path("results/iclr_return_labelled_guard_20261003")
ROOT = Path("results/iclr_return_acceptance_20261003")
PROTOCOL = "three_pair_whole_policy_return_acceptance_v1"
PAIRS = 3
SEEDS = (2801, 2802, 2803)
ENVIRONMENTS = ("mo-halfcheetah-v5", "mo-hopper-v5")


def specification():
    return dict(source_root=str(SOURCE), environments=list(ENVIRONMENTS), seeds=list(SEEDS),
        trials="first_three_completed_paired_episodes_in_original_random_preference_order",
        horizon=1000, pairs=PAIRS, proposal_steps=100000, incumbent_steps=75000,
        score="equal_pair_mean_of_actual_undiscounted_weighted_new_minus_old_returns",
        acceptance="strictly_positive_score_accepts_entire_100k_actor_and_own_encoder; otherwise_75k_pair",
        blind="one_whole_policy_fair_coin; Python_Random_100300000+seed; score_ignored",
        validation="reuse_original_frozen_all21_four_mode_two_reset_endpoints; inspected_cohort_not_confirmation",
        screen="no_harmful_accepted_proposal_in_utility_HV_or_any_mode; both_envs_mean_utility_HV_and_all_mode_means_"
               "at_least_both_frozen_endpoints_and_blind; at_least_one_accepted_improvement_overall",
        pricing="hypothetical_100k_plus_three_pair_actual_steps_plus_four_probes; future_native_"
                "100k_plus_trial_steps_minus_two_with_six_probes; source_full_costs_remain_charged",
        stop="failed_signal_screen_HOLD_without_pair_count_threshold_or_seed_tuning; passed_screen_still_"
             "requires_fresh_equal_price_native_training_before_any_cost_superiority_claim")


def decide(rows):
    assert len(rows) >= PAIRS
    trials = rows[:PAIRS]
    margins = []
    for row in trials:
        assert row["paired_initial_observation"]
        returns = row["returns"]
        margin = sum(w * (new - old) for w, new, old in
                     zip(row["weight"], returns["new"], returns["old"]))
        assert math.isclose(margin, row["utility_difference"], rel_tol=1e-9, abs_tol=1e-9)
        margins.append(margin)
    cost = sum(sum(r["steps"].values()) for r in trials)
    assert 0 < cost <= 2 * PAIRS * 1000
    score = statistics.mean(margins)
    return dict(chosen="proposal" if score > 0 else "prefix", score=score, margins=margins,
        weight_indices=[r["weight_index"] for r in trials], trial_interactions=cost,
        hypothetical_candidate_price=100000 + cost + 4,
        future_native_steps=100000 + cost - 2, future_native_probes=6)


def harms(proposal, incumbent):
    return ([m for m in ("utility", "hv") if proposal[m] < incumbent[m]] +
            [f"mode{i}" for i, (new, old) in enumerate(zip(proposal["per_mode_utility"], incumbent["per_mode_utility"]))
             if new < old])


def summarize(rows):
    methods = {}
    for name in ("selected", "prefix", "proposal", "blind"):
        endpoints = [r["endpoints"][r["decision"]["chosen"] if name == "selected" else
                     r["blind_choice"] if name == "blind" else name] for r in rows]
        methods[name] = {m: statistics.mean(e[m] for e in endpoints) for m in ("utility", "hv")}
        methods[name]["mean_mode_utility"] = [statistics.mean(e["per_mode_utility"][i] for e in endpoints)
                                               for i in range(len(endpoints[0]["per_mode_utility"]))]
    harmful = {str(r["seed"]): harms(r["endpoints"]["proposal"], r["endpoints"]["prefix"])
               for r in rows if r["decision"]["chosen"] == "proposal"}
    failures = []
    for control in ("prefix", "proposal", "blind"):
        for metric in ("utility", "hv"):
            if methods["selected"][metric] < methods[control][metric]:
                failures.append(f"{metric}_below_{control}")
        if any(a < b for a, b in zip(methods["selected"]["mean_mode_utility"], methods[control]["mean_mode_utility"])):
            failures.append(f"mode_mean_below_{control}")
    if any(harmful.values()):
        failures.append("harmful_accepted_proposal")
    return dict(methods=methods, harmful_accepted_proposals=harmful, failures=failures,
        passed=not failures, per_seed=rows)


def run(root=ROOT):
    root = Path(root)
    frozen = json.loads((root / "protocol.json").read_text())
    assert (frozen["protocol"], frozen["experiment"]) == (PROTOCOL, specification())
    original = json.loads((SOURCE / "compact_readout.json").read_text())
    imitation = json.loads((SOURCE / "imitation_diagnostic/compact_readout.json").read_text())
    manifest = json.loads((SOURCE / "protocol.json").read_text())
    assert original["complete"] and imitation["complete"]
    assert original["protocol"] == "return_labelled_actor_tangent_guard_v1"
    assert original["experiment"] == manifest["experiment"]
    assert imitation["experiment"]["source_root"] == str(SOURCE)
    assert imitation["experiment"]["source_protocol"] == original["protocol"]
    result = dict(protocol=PROTOCOL, experiment=specification(), complete=True, environments={},
        source_acquisition_work=original["actual_work"],
        actual_work=dict(training_interactions=0, calibration_interactions=0, evaluation_interactions=0,
            initialization_probes=0, actor_steps=0, critic_steps=0, encoder_steps=0, fits=0),
        native_equal_price_control_run=False)
    accepted_improvements = 0
    for env in ENVIRONMENTS:
        old = original["environments"][env]
        records = imitation["environments"][env]["per_seed"]
        assert [r["seed"] for r in records] == list(SEEDS)
        rows = []
        for index, seed in enumerate(SEEDS):
            endpoints = {m: records[index]["source_endpoints"][m] for m in ("prefix", "proposal")}
            for m in endpoints:
                for metric in ("utility", "hv"):
                    assert endpoints[m][metric] == old["methods"][m][metric]["per_seed"][index]
            decision = decide(old["calibration"][index]["rows"])
            bad = harms(endpoints["proposal"], endpoints["prefix"])
            if decision["chosen"] == "proposal" and not bad and endpoints["proposal"]["utility"] > endpoints["prefix"]["utility"]:
                accepted_improvements += 1
            rows.append(dict(seed=seed, decision=decision, endpoints=endpoints,
                blind_choice=("prefix", "proposal")[random.Random(100300000 + seed).randrange(2)]))
        result["environments"][env] = summarize(rows)
    result["accepted_nonharmful_improvements"] = accepted_improvements
    passed = accepted_improvements > 0 and all(e["passed"] for e in result["environments"].values())
    result["decision"] = "fresh_equal_price_native_test_required" if passed else "hold_three_pair_return_acceptance"
    (root / "compact_readout.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"decision": result["decision"], "accepted_nonharmful_improvements": accepted_improvements,
        "environments": {env: {"choices": [r["decision"] for r in e["per_seed"]],
            "failures": e["failures"], "methods": e["methods"]} for env, e in result["environments"].items()}}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root)
