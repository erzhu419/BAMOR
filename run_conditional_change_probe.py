"""Controlled change-evidence probe; no policy/critic/encoder updates."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from conditional_change_evidence import ConditionalChangeEvidence, CONFIG
import train_bamor_mujoco as training


PROTOCOL = "conditional_change_evidence_v1"
ROOT = Path("results/iclr_conditional_change_20260930")
SOURCE = Path("results/iclr_episode_allocation_20260930")
SEEDS = tuple(range(2101, 2106))
ENVIRONMENTS = ("mo-halfcheetah-v5", "mo-hopper-v5")
METHODS = {name: f"bamor_common_target_episode_{name}" for name in ("fixed", "adaptive")}
SOURCE_HIDDEN = 256
FIT_STEPS, CALIBRATION_STEPS, CASE_STEPS, CHANGE_AT = 2000, 1000, 1000, 500
CASES = ("stationary", "sampler_change", "actor_change", "mode1", "mode2", "mode3")
MINIMUM_POST_WINDOWS, MAXIMUM_DELAY = 3, 300


def budget():
    return dict(fit_interactions=FIT_STEPS, calibration_interactions=CALIBRATION_STEPS,
                controlled_probe_interactions=len(CASES) * CASE_STEPS,
                environment_initialization_probe_steps=1 + len(CASES))


def load_models(env, seed, source):
    obs_dim, act_dim, reward_dim = env.observation_space.shape[0], env.action_space.shape[0], env.reward_dim
    context_dim = training.context_dim_for_method(METHODS["fixed"], obs_dim, act_dim, reward_dim, env.num_modes)
    policies, checkpoints = {}, {}
    for name, method in METHODS.items():
        folder = Path(source) / env.spec.id.replace("-", "_") / name
        raw = json.loads((folder / f"{method}_seed{seed}.json").read_text())
        assert (raw["env"], raw["method"], raw["seed"], raw["total_steps"]) == (env.spec.id, method, seed, 75000)
        checkpoints[name] = torch.load(folder / f"{method}_seed{seed}_model.pt", map_location="cpu", weights_only=True)
        policies[name] = training.PolicyNet(obs_dim, act_dim, context_dim, hidden=SOURCE_HIDDEN)
        policies[name].load_state_dict(checkpoints[name]["policy"])
        policies[name].eval().requires_grad_(False)
    encoder = training.LatentTransitionEncoder(obs_dim, act_dim, reward_dim,
        latent_dim=training.LATENT_CONTEXT_DIM, hidden=SOURCE_HIDDEN)
    encoder.load_state_dict(checkpoints["fixed"]["encoder"])
    encoder.eval().requires_grad_(False)
    return policies, encoder


def collect(env, policies, encoder, steps, reset_seed, weight_seed, case="stationary"):
    reward_dim = env.reward_dim
    weights = np.random.default_rng(weight_seed).dirichlet(np.ones(reward_dim), size=steps).astype(np.float32)
    obs, _ = env.reset(seed=reset_seed)
    transition = training.TransitionContextTracker(reward_dim)
    latent = training.LearnedTransitionContextTracker(encoder, latent_dim=training.LATENT_CONTEXT_DIM)
    inputs, targets, terminals, used_weights, contexts = [], [], [], [], []
    episode = 0
    for index in range(steps):
        changed = index >= CHANGE_AT
        if index == CHANGE_AT and case.startswith("mode"):
            env._apply_mode(int(case[-1]))
            env.mode_history.append((index, int(case[-1])))
        weight = weights[index]
        if changed and case == "sampler_change":
            weight, _ = training.sample_stratified_weight(reward_dim, index - CHANGE_AT + 1)
        context = training.build_mujoco_context(METHODS["fixed"], weight,
            trans_tracker=transition, learned_tracker=latent)
        policy = policies["adaptive" if changed and case == "actor_change" else "fixed"]
        with torch.no_grad():
            action = policy.get_action(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
                torch.as_tensor(context).unsqueeze(0), deterministic=True).numpy().reshape(-1)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        inputs.append(np.concatenate([obs, action]))
        targets.append(np.concatenate([reward, next_obs - obs]))
        terminals.append([terminated, truncated])
        used_weights.append(weight)
        contexts.append(context)
        transition.update(obs, action, reward, next_obs)
        latent.update(obs, action, reward, next_obs, "cpu")
        obs = next_obs
        if terminated or truncated:
            episode += 1
            obs, _ = env.reset(seed=reset_seed + episode)
            transition.reset()
            latent.reset()
    assert env._total_steps == steps
    return dict(inputs=np.asarray(inputs, np.float32), targets=np.asarray(targets, np.float32),
                terminals=np.asarray(terminals, bool), weights=np.asarray(used_weights, np.float32),
                contexts=np.asarray(contexts, np.float32))


def case_summary(evaluation):
    windows = evaluation["windows"]
    before = [window for window in windows if window["end_step"] <= CHANGE_AT]
    after = [window for window in windows if window["start_step"] > CHANGE_AT]
    alarms = [window for window in after if window["alarm"]]
    first_delay = alarms[0]["end_step"] - CHANGE_AT if alarms else None
    available = sum(window["eligible"] for window in after) >= MINIMUM_POST_WINDOWS
    return dict(available=available, any_post_alarm=bool(alarms),
        first_post_alarm_delay=first_delay,
        timely_alarm=bool(available and first_delay is not None and first_delay <= MAXIMUM_DELAY),
        post_supported_fraction=float(evaluation["supported"][CHANGE_AT:].mean()),
        pre_available=sum(window["eligible"] for window in before) >= MINIMUM_POST_WINDOWS,
        pre_alarm_windows=sum(window["alarm"] for window in before),
        post_eligible_windows=sum(window["eligible"] for window in after), windows=windows)


def write(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def run(env_name, seed, root=ROOT, source=SOURCE):
    assert env_name in ENVIRONMENTS and seed in SEEDS
    folder = Path(root) / env_name.replace("-", "_") / f"seed{seed}"
    output = folder / "readout.json"
    if output.exists():
        saved = json.loads(output.read_text())
        assert (saved["protocol"], saved["env"], saved["seed"]) == (PROTOCOL, env_name, seed)
        if saved["complete"]:
            return saved
        raise RuntimeError("Incomplete probe retained; inspect before using a new-root retry")
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    probe_seed = seed + 100
    report = dict(protocol=PROTOCOL, env=env_name, seed=seed, probe_seed=probe_seed,
        complete=False, planned_budget=budget(), predictor_config=CONFIG,
        source=str(source), source_methods=METHODS, source_step=75000,
        change_at=CHANGE_AT, cases={}, scope="Controlled fixed-checkpoint mechanism test, not adaptive-selector performance.")
    write(output, report)
    started = time.monotonic()
    env = training.make_piecewise_mo_env(env_name, switch_interval=10**9, seed=probe_seed)
    env._lock_mode = True
    try:
        policies, encoder = load_models(env, seed, source)
        prefix = collect(env, policies, encoder, FIT_STEPS + CALIBRATION_STEPS,
                         20000000 + 1000 * probe_seed, 30000000 + probe_seed)
        evidence = ConditionalChangeEvidence(prefix["inputs"].shape[1], prefix["targets"].shape[1], env.reward_dim)
        report["predictor_work"] = evidence.fit(prefix["inputs"][:FIT_STEPS], prefix["targets"][:FIT_STEPS],
                                                  40000000 + probe_seed)
        report["calibration"] = evidence.calibrate(prefix["inputs"][FIT_STEPS:], prefix["targets"][FIT_STEPS:])
    finally:
        env.close()
    stored = {f"prefix_{key}": value for key, value in prefix.items()}
    reference = None
    for case in CASES:
        env = training.make_piecewise_mo_env(env_name, switch_interval=10**9, seed=probe_seed)
        env._lock_mode = True
        try:
            data = collect(env, policies, encoder, CASE_STEPS, 50000000 + 1000 * probe_seed,
                           60000000 + probe_seed, case)
            if reference is None:
                reference = data
            else:
                for key in data:
                    np.testing.assert_array_equal(data[key][:CHANGE_AT], reference[key][:CHANGE_AT])
            evaluation = evidence.evaluate(data["inputs"], data["targets"])
            report["cases"][case] = dict(case_summary(evaluation),
                ground_truth_mode_history_audit_only=[list(entry) for entry in env.mode_history],
                environment_interactions=env._total_steps)
            if case.startswith("mode"):
                np.testing.assert_array_equal(data["inputs"][CHANGE_AT], reference["inputs"][CHANGE_AT])
                report["cases"][case]["same_input_first_changed_target_max_delta"] = float(
                    np.abs(data["targets"][CHANGE_AT] - reference["targets"][CHANGE_AT]).max())
            if case == "sampler_change":
                report["cases"][case]["intervention_active"] = bool(
                    np.any(data["weights"][CHANGE_AT:] != reference["weights"][CHANGE_AT:]))
            if case == "actor_change":
                obs_dim = env.observation_space.shape[0]
                with torch.no_grad():
                    alternate = policies["adaptive"].get_action(
                        torch.as_tensor(reference["inputs"][CHANGE_AT:, :obs_dim]),
                        torch.as_tensor(reference["contexts"][CHANGE_AT:]), deterministic=True).numpy()
                difference = np.abs(alternate - reference["inputs"][CHANGE_AT:, obs_dim:])
                report["cases"][case].update(intervention_active=bool(np.any(difference > 0)),
                    same_state_context_actor_action_max_delta=float(difference.max()))
            stored.update({f"{case}_{key}": value for key, value in {**data,
                "errors": evaluation["errors"], "distances": evaluation["distances"],
                "supported": evaluation["supported"]}.items()})
            write(output, report)
            print(f"PROBE {env_name} seed={seed} case={case} "
                  f"available={report['cases'][case]['available']} "
                  f"alarm={report['cases'][case]['any_post_alarm']}", flush=True)
        finally:
            env.close()
    np.savez_compressed(folder / "transitions.npz", **stored)
    report.update(complete=True, paired_prefixes_verified=len(CASES) - 1,
        completed_budget=budget(), policy_optimizer_steps=0, critic_optimizer_steps=0,
        encoder_optimizer_steps=0, seconds=time.monotonic() - started)
    write(output, report)
    print(f"DONE {PROTOCOL} {env_name} seed={seed} interactions="
          f"{sum(budget()[key] for key in budget() if key != 'environment_initialization_probe_steps')}", flush=True)
    return report


def summarize(root=ROOT):
    result = dict(protocol=PROTOCOL, complete=True, seeds=list(SEEDS), environments={},
                  completed_budget=dict.fromkeys(budget(), 0), predictor_optimizer_steps=0)
    for env in ENVIRONMENTS:
        runs = [json.loads((Path(root) / env.replace("-", "_") / f"seed{seed}" / "readout.json").read_text())
                for seed in SEEDS]
        for seed, row in zip(SEEDS, runs):
            assert (row["protocol"], row["env"], row["seed"], row["complete"]) == (PROTOCOL, env, seed, True)
            assert row["predictor_config"] == CONFIG and row["completed_budget"] == budget()
            assert row["paired_prefixes_verified"] == len(CASES) - 1
            assert row["policy_optimizer_steps"] == row["critic_optimizer_steps"] == row["encoder_optimizer_steps"] == 0
            for key, value in row["completed_budget"].items():
                result["completed_budget"][key] += value
            result["predictor_optimizer_steps"] += row["predictor_work"]["optimizer_steps"]
        cases = {case: dict(available=sum(row["cases"][case]["available"] for row in runs),
            post_alarm=sum(row["cases"][case]["any_post_alarm"] for row in runs),
            timely_alarm=sum(row["cases"][case]["timely_alarm"] for row in runs),
            active_interventions=(sum(row["cases"][case]["intervention_active"] for row in runs)
                                  if case in ("actor_change", "sampler_change") else None)) for case in CASES}
        prefix = dict(available=sum(row["cases"]["stationary"]["pre_available"] for row in runs),
                      alarm=sum(row["cases"]["stationary"]["pre_alarm_windows"] > 0 for row in runs))
        passed = (prefix["available"] >= 4 and prefix["alarm"] <= 1
                  and all(cases[case]["available"] >= 4 and cases[case]["post_alarm"] <= 1 for case in CASES[:3])
                  and all(cases[case]["active_interventions"] == len(SEEDS) for case in ("actor_change", "sampler_change"))
                  and all(cases[case]["timely_alarm"] >= 4 for case in CASES[3:]))
        result["environments"][env] = dict(shared_prefix=prefix, cases=cases, screen_passed=passed, runs=runs)
    result["screen_passed"] = all(row["screen_passed"] for row in result["environments"].values())
    result["decision"] = "consider_selector_design" if result["screen_passed"] else "hold_conditional_evidence_integration"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVIRONMENTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        report = summarize(args.root)
        write(args.root / "readout.json", report)
        print(json.dumps({env: row["cases"] for env, row in report["environments"].items()}))
    else:
        run(args.env, args.seed, args.root, args.source)
