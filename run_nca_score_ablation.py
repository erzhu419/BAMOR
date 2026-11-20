"""Run the 75k BAMOR score-neutral control, reusing the frozen adaptive arm."""

import argparse
import json
from pathlib import Path

import run_utility_confirmation as frozen

TRAIN = frozen.previous.training
METHOD = "bamor_v25_progress_fixed_archive_mix"
ROOT = Path("results/nca_score_ablation_20261120")
STEP = frozen.previous.TOTAL_STEPS


def score_on_receipt(env, seed):
    path = frozen.ROOT / env.replace("-", "_") / "v25" / f"seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    summary = json.loads((frozen.ROOT / "readout.json").read_text())
    return next(r["v25"] for r in summary["environments"][env]["runs"] if r["seed"] == seed)


def run(env, seed, root):
    tag = env.replace("-", "_")
    folder = root / tag / "score_neutral"
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / f"{METHOD}_seed{seed}.json"
    model = folder / f"{METHOD}_seed{seed}_model.pt"
    attempt = folder / f"seed{seed}.json"
    control = score_on_receipt(env, seed)
    assert (control["protocol"], control["env"], control["seed"], control["complete"]) == (
        frozen.PROTOCOL, env, seed, True)
    if not raw.exists():
        if attempt.exists():
            raise RuntimeError(f"Incomplete attempt recorded at {attempt}")
        frozen.write_report(attempt, {"env": env, "seed": seed, "complete": False,
                                      "planned_budget_upper_bound": frozen.budget(env)})
        TRAIN.train_method(env, METHOD, STEP, 25000, seed, str(folder),
                           archive_steps=(STEP,), **frozen.previous.pilot.TRAIN_KWARGS)
    data = json.loads(raw.read_text())
    assert model.exists() and (data["env"], data["seed"], data["method"],
                              data["evaluation_protocol"], data["total_steps"]) == (
                                  env, seed, METHOD, "locked_full_simplex_v1", STEP)
    assert data["switch_interval"] == 25000
    assert data["ground_truth"]["mode_history"] == control["mode_history"]
    assert {key: data[key] for key in control["evaluation"]} == control["evaluation"]
    assert [s["step"] for s in data["policy_archive_snapshots"]] == [STEP]
    assert all(p["archive_step"] == STEP for front in data["pareto_per_mode"] for p in front)
    budget = {key: data[key] for key in frozen.budget(env)}
    assert budget["training_interactions"] == STEP
    assert budget["environment_initialization_probe_steps"] == 2
    assert 0 < budget["evaluation_interactions"] <= frozen.budget(env)["evaluation_interactions"]
    utility = frozen.previous.direct_utility(data)
    frozen.write_report(attempt, {"env": env, "seed": seed, "complete": True,
                                  "budget": budget, "direct_utility": utility})
    print(f"DONE nca_score_ablation {env} seed={seed} U={utility['mean']:.6f} "
          f"budget={json.dumps(budget)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=frozen.ENVIRONMENTS, required=True)
    parser.add_argument("--seed", type=int, choices=frozen.SEEDS, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.env, args.seed, args.root)
