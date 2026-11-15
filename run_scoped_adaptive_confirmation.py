"""Fresh-seed 2x2: scoped actor selection crossed with progress feedback."""

import argparse
import json
from pathlib import Path

import run_scoped_extreme_pilot as scoped
import summarize_aligned_progress_pilot as common


PROTOCOL = "scoped_adaptive_confirmation_v1"
ROOT = Path("results/iclr_scoped_adaptive_confirmation_20260922")
SEEDS = tuple(range(1006, 1011))
CONDITIONS = ("legacy_blind", "legacy_adaptive", "scoped_blind", "scoped_adaptive")
METHODS = common.pilot.LEGACY_METHODS
BUDGET = dict(training_interactions=50000, learner_updates=45001,
              evaluation_interactions=4788000, environment_initialization_probe_steps=2)
training = scoped.training


def factors(condition):
    update, selection = condition.split("_")
    assert condition in CONDITIONS
    return update, selection


class IntegratedRouter(scoped.ScopedRouter):
    def __init__(self, condition, cut=scoped.CUT, end=scoped.END, target_operators=scoped.TARGETS):
        update, _ = factors(condition)
        super().__init__("own_matched" if update == "scoped" else "factual", [], cut, end,
                         target_operators=target_operators)
        self.condition, self.step = condition, cut

    def __call__(self, state):
        assert state["step"] == self.cut
        self.selector = state["adaptive_selector"]
        _, selection = factors(self.condition)
        cls = (training.ProgressExtremeHierarchicalOperatorSelector if selection == "adaptive"
               else training.FixedProgressExtremeHierarchicalOperatorSelector)
        assert type(self.selector) is cls
        return False

    def after_step(self, step, trainer, encoder, env, operator):
        if step > self.cut:
            assert operator == self.operator

    def state_dict(self):
        result = super().state_dict()
        # Feedback may change the selected arms and their subsequent random draws.
        result.pop("rng_alignment", None)
        result.update(protocol=PROTOCOL, condition=self.condition,
                      selection_uses_scores=factors(self.condition)[1] == "adaptive")
        return result


def validate_routing(data, condition, reward_dim=2, target_operators=scoped.TARGETS):
    update, selection = factors(condition)
    assert not data.get("pareto_evals")
    assert "continuation_from_step" not in data
    assert (data["critic_target_rule"], data["bootstrap_terminal_rule"],
            data["critic_candidate_context_rule"]) == (
                "scalar_projection_of_soft_vector", "terminated_only", "candidate_preference")
    selector = common.validate_selector(data, reward_dim)
    route = data["update_routing"]
    assert (route["protocol"], route["condition"], route["cut"], route["end"]) == (
        PROTOCOL, condition, scoped.CUT, scoped.END)
    assert route["selection_uses_scores"] == (selection == "adaptive")
    assert route["target_operators"] == list(target_operators)
    assert "rng_alignment" not in route
    assert route["extra_updates"] == route["extra_environment_steps"] == 0
    blocks = data["adaptive_operator_final"]["block_history"]
    assert len(route["blocks"]) == 20
    configs = training.ProgressExtremeHierarchicalOperatorSelector(reward_dim)
    for actual, block in zip(route["blocks"], blocks[10:]):
        op = block["operator"]
        assert (actual["start_step"], actual["end_step"], actual["steps"], actual["operator"]) == (
            block["end_step"] - 1499, block["end_step"], 1500, op)
        assert actual["own_selected"] == (update == "scoped" and op in target_operators)
        assert actual["config"] == {**configs.trainer_config(op),
                                    "preference_relabeling_rate": configs.relabel_rate(op)}
    targeted = sum(b["steps"] for b in blocks[10:] if b["operator"] in target_operators)
    assert route["targeted_updates"] == targeted
    assert route["own_selected_updates"] == (targeted if update == "scoped" else 0)
    return selector


def validate(data, condition, seed):
    _, selection = factors(condition)
    assert seed in SEEDS
    for key, value in BUDGET.items():
        if key != "learner_updates":
            assert data[key] == value, key
    selector = validate_routing(data, condition)
    route = data["update_routing"]
    blocks = data["adaptive_operator_final"]["block_history"]
    metrics = common.metrics(data, seed, expected_method=METHODS[selection])
    for old, new in (("posthoc_late_snapshot_hv", "late_snapshot_hv"),
                     ("posthoc_late_mean_hv", "late_mean_hv"),
                     ("posthoc_late_mode_mean_hv", "late_mode_mean_hv")):
        metrics[new] = metrics.pop(old)
    metrics["terminal_returns"] = scoped.forks.terminal_returns(data).tolist()
    return dict(protocol=PROTOCOL, condition=condition, seed=seed, complete=True,
        completed_budget=BUDGET.copy(), metrics=metrics, selector=selector, routing=route,
        evaluation={k: data[k] for k in ("evaluation_weights", "episode_seeds_per_mode", "ref_point",
                                        "policy_archive_snapshots")},
        warmup=dict(episodes=[r for r in data["episode_rewards"] if r["step"] <= scoped.CUT],
                    blocks=blocks[:10]),
        scope="Fresh seeds for the frozen 2x2 experiment; five seeds are not population robustness. "
              "Final-encoder snapshots are not online recovery curves. No new evaluator or extra interactions.")


def run(condition, seed, root=ROOT):
    assert condition in CONDITIONS and seed in SEEDS
    folder = Path(root) / condition
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"seed{seed}.json"
    method = METHODS[factors(condition)[1]]
    raw = folder / f"{method}_seed{seed}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["condition"], saved["seed"]) == (PROTOCOL, condition, seed)
        if saved["complete"]:
            print(f"DONE {condition} seed={seed} reusing complete report", flush=True)
            return saved
        if not raw.exists():
            raise RuntimeError("Incomplete training attempt preserved; inspect its log before a new-root retry")
    if raw.exists():
        data = json.loads(raw.read_text())
    else:
        scoped.forks.write_report(output, dict(protocol=PROTOCOL, condition=condition, seed=seed,
            complete=False, planned_budget=BUDGET, raw_path=str(raw)))
        router = IntegratedRouter(condition)
        try:
            data = training.train_method("mo-halfcheetah-v5", method, scoped.END, 25000, seed,
                str(folder), update_router=router, state_probe=router, **common.pilot.TRAIN_KWARGS)
        finally:
            router.close()
    report = validate(data, condition, seed)
    scoped.forks.write_report(output, report)
    print(f"DONE {PROTOCOL} {condition} seed={seed} final_hv={report['metrics']['final_hv']:.6f}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.condition, args.seed, args.root)
