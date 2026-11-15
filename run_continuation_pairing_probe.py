"""Locate execution-time actor/encoder effects in saved continuation endpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import run_shared_objective_router as shared


router, training, write = shared.router, shared.training, shared.write
PROTOCOL = "continuation_actor_encoder_pairing_v1"
ROOT = Path("results/iclr_continuation_pairing_20261002")
SOURCE = shared.ROOT
ENVIRONMENTS, SEEDS = shared.ENVIRONMENTS, shared.SEEDS
PAIRS = {"old_old": ("soft", "soft"), "new_new": ("work_soft", "work_soft"),
         "new_old": ("work_soft", "soft"), "old_new": ("soft", "work_soft")}
CONTRASTS = {"encoder_with_new_actor": {"new_new": 1, "new_old": -1},
             "encoder_with_old_actor": {"old_new": 1, "old_old": -1},
             "actor_with_old_encoder": {"new_old": 1, "old_old": -1},
             "actor_with_new_encoder": {"new_new": 1, "old_new": -1},
             "interaction": {"new_new": 1, "new_old": -1, "old_new": -1, "old_old": 1},
             "old_encoder_replacement": {"new_old": 1, "new_new": -1}}


def specification():
    return dict(source_root=str(SOURCE), source_protocol=shared.PROTOCOL,
        source_experiment=shared.specification(), environments=list(ENVIRONMENTS), seeds=list(SEEDS),
        pairs={k: list(v) for k, v in PAIRS.items()}, contrasts=CONTRASTS,
        evaluation="full21_preferences_four_locked_modes_two_paired_resets; original_source_seeds",
        horizon=router.HORIZON, repeats=router.REPEATS,
        anchors="reuse_source_soft_and_work_soft_fronts; reproduce_entire_work_soft_front_before_crosses",
        new_work="zero_training_collection_calibration_fits_or_optimizer_steps; three_evaluation_grids_and_one_probe",
        screen="HC_new_old_beats_new_new_in_low_speed_at_each_seed_and_in_mean; mean_utility_HV_low_cost_nonnegative; "
               "Hopper_mean_utility_HV_minimum_seed_utility_HV_and_all_mode_utilities_nonnegative",
        scope="execution_component_replacement_on_inspected_seeds; not_causal_attribution_of_training_or_deployable_selection")


def same_front(actual, expected):
    assert actual["evaluation"] == expected["evaluation"], "Evaluation settings changed"
    a, b = actual["pareto_per_mode"], expected["pareto_per_mode"]
    assert [[p["weight"] for p in f] for f in a] == [[p["weight"] for p in f] for f in b]
    error = float(np.max(np.abs(np.asarray([[p["return"] for p in f] for f in a])
                               - np.asarray([[p["return"] for p in f] for f in b]))))
    assert error <= 1e-6, f"Saved native actor/encoder front does not reproduce: {error}"
    assert actual["interactions"] == expected["interactions"]
    return error


def load_pairs(env, cached, path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    obs_dim, act_dim, reward_dim = env.observation_space.shape[0], env.action_space.shape[0], env.reward_dim
    context_dim = training.context_dim_for_method(shared.source.METHOD, obs_dim, act_dim, reward_dim, env.num_modes)
    hidden = cached["experiment"]["training_kwargs"]["hidden"]
    models = {}
    for name in ("soft", "work_soft"):
        policy = training.PolicyNet(obs_dim, act_dim, context_dim, hidden=hidden)
        encoder = training.LatentTransitionEncoder(obs_dim, act_dim, reward_dim,
            latent_dim=training.LATENT_CONTEXT_DIM, hidden=hidden)
        for key, network in (("policy", policy), ("encoder", encoder)):
            network.load_state_dict(saved[name][key])
            network.eval().requires_grad_(False)
        models[name] = dict(policy=policy, encoder=encoder, reward_dim=reward_dim)
    return {name: dict(policy=models[actor]["policy"], encoder=models[encoder]["encoder"], reward_dim=reward_dim)
            for name, (actor, encoder) in PAIRS.items()}


def check(record):
    assert (record["protocol"], record["experiment"], record["complete"]) == (PROTOCOL, specification(), True)
    assert record["env"] in ENVIRONMENTS and record["seed"] in SEEDS
    assert record["controls_passed"] and record["native_front_error"] <= 1e-6
    assert set(record["cells"]) == set(PAIRS)
    assert set(record["fresh_evaluations"]) == {"new_new", "new_old", "old_new"}
    for name, cell in record["cells"].items():
        assert cell["components"] == list(PAIRS[name])
        assert cell["value"]["evaluation"] == record["cells"]["new_new"]["value"]["evaluation"]
        assert cell["source_reused"] == (name in ("old_old", "new_new"))
    work = record["actual_work"]
    assert work == dict(training_interactions=0, calibration_interactions=0,
        initialization_probes=1, actor_steps=0, critic_steps=0, encoder_steps=0, ridge_fits=0,
        evaluation_interactions=sum(v["interactions"] for v in record["fresh_evaluations"].values()))
    assert 0 < work["evaluation_interactions"] <= 3 * 4 * 21 * router.REPEATS * router.HORIZON


def run(env_name, seed, root=ROOT):
    root = Path(root)
    frozen = json.loads((root / "protocol.json").read_text())
    assert (frozen["protocol"], frozen["experiment"]) == (PROTOCOL, specification())
    assert env_name in ENVIRONMENTS and seed in SEEDS
    output = root / env_name.replace("-", "_") / f"seed{seed}.json"
    if output.exists():
        previous = json.loads(output.read_text())
        if previous["complete"]:
            check(previous)
            assert (previous["env"], previous["seed"]) == (env_name, seed)
            return previous
        raise RuntimeError("Incomplete attempt retained; inspect before a new-root retry")
    path = SOURCE / env_name.replace("-", "_") / f"seed{seed}.json"
    cached = json.loads(path.read_text())
    shared.check(cached)
    assert (cached["env"], cached["seed"]) == (env_name, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(protocol=PROTOCOL, experiment=specification(), env=env_name, seed=seed,
        source=str(path), complete=False, controls_passed=False, phase="loading", cells={}, fresh_evaluations={},
        source_acquisition_work=cached["actual_work"],
        actual_work=dict(training_interactions=0, calibration_interactions=0, initialization_probes=0,
                         actor_steps=0, critic_steps=0, encoder_steps=0, ridge_fits=0, evaluation_interactions=0))
    write(output, report)
    env = training.make_piecewise_mo_env(env_name, switch_interval=shared.SWITCH_INTERVAL, seed=seed + 1000)
    report["actual_work"]["initialization_probes"] = 1
    write(output, report)
    try:
        pairs = load_pairs(env, cached, path.with_suffix(".actors.pt"))
        state = dict(eval_env=env, run_config=dict(env_name=env_name))
        for name, source_name in (("old_old", "soft"), ("new_new", "work_soft")):
            report["cells"][name] = dict(components=list(PAIRS[name]), source_reused=True,
                                        value=cached["evaluations"][source_name])
        for name in ("new_new", "new_old", "old_new"):
            report["phase"] = "evaluating_" + name
            write(output, report)
            value = shared.endpoint_record(router.evaluate(state, {"soft": pairs[name]}, seed, "budget_soft"))
            report["actual_work"]["evaluation_interactions"] += value["interactions"]
            report["fresh_evaluations"][name] = value
            write(output, report)
            if name == "new_new":
                report["native_front_error"] = same_front(value, cached["evaluations"]["work_soft"])
                report["controls_passed"] = True
            else:
                report["cells"][name] = dict(components=list(PAIRS[name]), source_reused=False, value=value)
            write(output, report)
            print(f"PAIR_EVAL env={env_name} seed={seed} cell={name} utility={value['endpoint']['utility']:.6f} "
                  f"interactions={value['interactions']}", flush=True)
        report.update(complete=True, phase="complete")
        check(report)
        write(output, report)
        print(f"DONE {PROTOCOL} env={env_name} seed={seed} cost={json.dumps(report['actual_work'])}", flush=True)
        return report
    finally:
        env.close()


def summarize(root=ROOT):
    result = dict(protocol=PROTOCOL, experiment=specification(), complete=True, environments={}, actual_work={})
    for env in ENVIRONMENTS:
        rows = [json.loads((Path(root) / env.replace("-", "_") / f"seed{seed}.json").read_text()) for seed in SEEDS]
        for seed, row in zip(SEEDS, rows):
            check(row)
            assert (row["env"], row["seed"]) == (env, seed)
            for key, value in row["actual_work"].items():
                result["actual_work"][key] = result["actual_work"].get(key, 0) + value
        methods = {}
        for name in PAIRS:
            metrics = {m: dict(mean=float(np.mean(values)), per_seed=values, minimum_seed=min(values))
                       for m in ("utility", "hv")
                       for values in [[r["cells"][name]["value"]["endpoint"][m] for r in rows]]}
            modes = [r["cells"][name]["value"]["endpoint"]["per_mode_utility"] for r in rows]
            metrics.update(per_seed_mode_utility=modes, mean_mode_utility=np.mean(modes, axis=0).tolist())
            methods[name] = metrics
        contrasts = {}
        for name, coefficients in CONTRASTS.items():
            contrasts[name] = {metric: dict(per_seed=(delta := sum(c * np.asarray(methods[k][metric]["per_seed"])
                for k, c in coefficients.items())).tolist(), mean=float(delta.mean())) for metric in ("utility", "hv")}
            delta = sum(c * np.asarray(methods[k]["per_seed_mode_utility"]) for k, c in coefficients.items())
            contrasts[name].update(per_seed_mode_utility=delta.tolist(), mean_mode_utility=delta.mean(0).tolist())
        result["environments"][env] = dict(methods=methods, contrasts=contrasts,
            native_front_max_error=max(r["native_front_error"] for r in rows),
            source_acquisition_work=[r["source_acquisition_work"] for r in rows])
    hc = result["environments"]["mo-halfcheetah-v5"]
    hopper = result["environments"]["mo-hopper-v5"]
    delta = hc["contrasts"]["old_encoder_replacement"]
    hc_pass = all(r[1] > 0 for r in delta["per_seed_mode_utility"]) and delta["mean_mode_utility"][1] > 0
    hc_pass = hc_pass and delta["mean_mode_utility"][2] >= 0 and all(delta[m]["mean"] >= 0 for m in ("utility", "hv"))
    hd = hopper["contrasts"]["old_encoder_replacement"]
    hp_pass = min(hd["mean_mode_utility"]) >= 0 and all(hd[m]["mean"] >= 0 and
        hopper["methods"]["new_old"][m]["minimum_seed"] >= hopper["methods"]["new_new"][m]["minimum_seed"]
        for m in ("utility", "hv"))
    actor_losses = all(hc["contrasts"][name]["mean_mode_utility"][1] < 0 and
        all(r[1] < 0 for r in hc["contrasts"][name]["per_seed_mode_utility"])
        for name in ("actor_with_old_encoder", "actor_with_new_encoder"))
    result.update(encoder_replacement_screen=dict(halfcheetah=hc_pass, hopper=hp_pass),
        actor_low_speed_losses_under_both_encoders=actor_losses,
        decision="test_frozen_encoder_training_on_fresh_seeds" if hc_pass and hp_pass else
                 "investigate_actor_return_protection_not_encoder_freezing" if actor_losses else
                 "hold_unlocalized_update_protection")
    write(Path(root) / "compact_readout.json", result)
    print("DONE", PROTOCOL, "summary", result["decision"], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=ENVIRONMENTS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.root)
    else:
        run(args.env, args.seed, args.root)
