"""Costed scene/group observability audit of already saved actor/encoder pairs."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

import run_context_descent_probe as source


training, write = source.training, source.write
PROTOCOL = "saved_actor_context_group_alignment_v1"
ROOT = Path("results/iclr_context_group_alignment_20261003")
SOURCE_ROOT = source.ROOT
ENVIRONMENTS, SEEDS, MODEL_NAMES = source.ENVIRONMENTS, source.SEEDS, source.MODEL_NAMES
HORIZON, WARMUP = 1000, 64
WEIGHT_INDICES = {"mo-halfcheetah-v5": (0, 7, 13, 20), "mo-hopper-v5": (0, 5, 12, 20)}
WINDOWS = ("all", "after_warmup")


def specification():
    return dict(source_protocol=source.PROTOCOL, source_root=str(SOURCE_ROOT),
        environments=list(ENVIRONMENTS), seeds=list(SEEDS), models=list(MODEL_NAMES),
        weight_indices={k: list(v) for k, v in WEIGHT_INDICES.items()}, horizon=HORIZON,
        warmup=WARMUP, repeats=1, reset_seed="107000000+1000*seed+10*mode; paired_across_weights_and_models",
        grouping="reuse_exact_saved_prefix_mean_scale_centers_and_feature_slice; no_refit",
        collection="frozen_deterministic_actor_with_own_frozen_encoder; fresh_trackers; four_locked_physical_modes",
        work="no_training_no_optimizers; at_most_48000_diagnostic_steps_and_one_initialization_probe_per_env_seed",
        aggregation="equal_episode_group_proportions_within_each_mode_then_equal_mode_weights; "
                    "also_report_raw_row_counts_and_missing_post_warmup_episodes",
        quantities="mode_by_group_matrix; best_group_to_mode_balanced_accuracy; best_bijective_accuracy; "
                   "pairwise_distribution_overlap; exact_feature_boundary_fraction",
        scope="posthoc_observability_diagnostic_only; no_mode_labels_in_policy_or_group_assignment; "
              "no_training_replay_or_critic_saved_so_no_entropy_value_gradient_decomposition_or_return_guarantee")


def assign(context, grouping):
    start, stop = grouping["feature_slice"]
    features = np.asarray(context)[..., start:stop]
    z = (features - np.asarray(grouping["mean"])) / np.asarray(grouping["scale"])
    return ((z[..., None, :] - np.asarray(grouping["centers"])) ** 2).sum(-1).argmin(-1)


def window_record(features, labels, reward_dim):
    n = len(features)
    boundaries = np.asarray([3.] + [1.] * reward_dim)
    return dict(rows=n, group_counts=np.bincount(labels, minlength=source.GROUPS).tolist(),
        feature_mean=features.mean(0).tolist() if n else None,
        feature_std=features.std(0).tolist() if n else None,
        feature_boundary_fraction=(np.abs(features) == boundaries).mean(0).tolist() if n else None)


def alignment(rows, model, window):
    matrix, coverage, raw = [], [], []
    for mode in range(4):
        cells = [r[window] for r in rows if r["model"] == model and r["mode"] == mode]
        available = [c for c in cells if c["rows"]]
        coverage.append(dict(available=len(available), missing=len(cells) - len(available)))
        raw.append(np.sum([c["group_counts"] for c in cells], axis=0).tolist())
        matrix.append(np.mean([np.asarray(c["group_counts"]) / c["rows"] for c in available], 0).tolist()
                      if available else None)
    result = dict(episode_balanced_mode_by_group=matrix, raw_mode_by_group_counts=raw,
                  episode_coverage=coverage, best_group_to_mode_balanced_accuracy=None,
                  best_bijective_accuracy=None, pairwise_overlap=None, group_marginal=None)
    if all(c["available"] for c in coverage):
        p = np.asarray(matrix)
        result.update(best_group_to_mode_balanced_accuracy=float(p.max(0).sum() / 4),
            best_bijective_accuracy=float(max(sum(p[m, g] for m, g in enumerate(order))
                                             for order in itertools.permutations(range(4))) / 4),
            pairwise_overlap=np.minimum(p[:, None, :], p[None, :, :]).sum(-1).tolist(),
            group_marginal=p.mean(0).tolist())
    return result


@torch.no_grad()
def episode(env, model, weight, reset_seed, grouping):
    before = env._total_steps
    obs, _ = env.reset(seed=reset_seed)
    start = dict(observation=obs.tolist(), qpos=env.unwrapped.data.qpos.tolist(),
                 qvel=env.unwrapped.data.qvel.tolist())
    trans = training.TransitionContextTracker(len(weight))
    learned = training.LearnedTransitionContextTracker(model["encoder"], latent_dim=training.LATENT_CONTEXT_DIM)
    contexts = []
    mode = env.current_mode_idx
    for _ in range(HORIZON):
        context = training.build_mujoco_context(source.shared.source.METHOD, weight,
            trans_tracker=trans, learned_tracker=learned)
        contexts.append(context.copy())
        action = model["policy"].get_action(torch.as_tensor(obs).float()[None],
            torch.as_tensor(context).float()[None], deterministic=True).numpy().ravel()
        ns, reward, terminal, truncated, _ = env.step(action)
        assert env.current_mode_idx == mode and env._lock_mode
        trans.update(obs, action, reward, ns)
        learned.update(obs, action, reward, ns, "cpu")
        obs = ns
        if terminal or truncated:
            break
    contexts = np.asarray(contexts)
    labels = assign(contexts, grouping)
    begin, end = grouping["feature_slice"]
    features = contexts[:, begin:end]
    assert env._total_steps - before == len(contexts)
    return dict(start=start, steps=len(contexts), terminated=bool(terminal), truncated=bool(truncated),
        all=window_record(features, labels, len(weight)),
        after_warmup=window_record(features[WARMUP:], labels[WARMUP:], len(weight)))


def work(rows):
    return dict(diagnostic_interactions=sum(r["steps"] for r in rows), initialization_probes=1,
                episodes=len(rows), training_interactions=0, actor_steps=0, critic_steps=0, encoder_steps=0)


def check(record):
    assert record["protocol"] == PROTOCOL and record["experiment"] == specification()
    assert record["complete"] and record["paired_resets"]
    assert record["env"] in ENVIRONMENTS and record["seed"] in SEEDS
    rows = record["rows"]
    assert len(rows) == 48
    assert {(r["model"], r["mode"], r["weight_index"]) for r in rows} == {
        (name, mode, wi) for name in MODEL_NAMES for mode in range(4) for wi in WEIGHT_INDICES[record["env"]]}
    for row in rows:
        assert row["reset_seed"] == 107000000 + 1000 * record["seed"] + 10 * row["mode"]
        assert 1 <= row["steps"] <= HORIZON
        assert row["all"]["rows"] == row["steps"]
        assert row["after_warmup"]["rows"] == max(0, row["steps"] - WARMUP)
        assert all(sum(row[w]["group_counts"]) == row[w]["rows"] for w in WINDOWS)
    assert record["actual_work"] == work(rows)
    assert record["alignment"] == {name: {w: alignment(rows, name, w) for w in WINDOWS} for name in MODEL_NAMES}


def run(env_name, seed, root=ROOT):
    root = Path(root)
    assert json.loads((root / "protocol.json").read_text()) == dict(protocol=PROTOCOL, experiment=specification())
    assert env_name in ENVIRONMENTS and seed in SEEDS
    output = root / env_name.replace("-", "_") / f"seed{seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        previous = json.loads(output.read_text())
        if previous["complete"]:
            check(previous)
            assert (previous["env"], previous["seed"]) == (env_name, seed)
            return previous
        raise RuntimeError("Incomplete diagnostic retained; inspect before another attempt")
    saved = SOURCE_ROOT / env_name.replace("-", "_") / f"seed{seed}.json"
    original = json.loads(saved.read_text())
    assert (original["protocol"], original["env"], original["seed"], original["complete"],
            original["controls_passed"]) == (source.PROTOCOL, env_name, seed, True, True)
    states = torch.load(saved.with_suffix(".actors.pt"), map_location="cpu", weights_only=True)
    assert set(states) == set(MODEL_NAMES)
    grouping = original["intervention"]["grouping"]
    report = dict(protocol=PROTOCOL, experiment=specification(), env=env_name, seed=seed,
                  complete=False, source_record=str(saved), grouping=grouping, rows=[], paired_resets=False)
    write(output, report)
    env = training.make_piecewise_mo_env(env_name, switch_interval=source.SWITCH_INTERVAL, seed=seed)
    try:
        assert env.num_modes == 4
        env._lock_mode = True
        sd, ad, rd = env.observation_space.shape[0], env.action_space.shape[0], env.reward_dim
        cd = training.context_dim_for_method(source.shared.source.METHOD, sd, ad, rd, env.num_modes)
        assert grouping["feature_slice"] == [rd, 2 * rd + 1]
        models = {}
        for name in MODEL_NAMES:
            policy = training.PolicyNet(sd, ad, cd, hidden=source.KWARGS["hidden"])
            encoder = training.LatentTransitionEncoder(sd, ad, rd, latent_dim=training.LATENT_CONTEXT_DIM,
                                                       hidden=source.KWARGS["hidden"])
            for key, net in (("policy", policy), ("encoder", encoder)):
                net.load_state_dict(states[name][key])
                net.eval().requires_grad_(False)
            models[name] = dict(policy=policy, encoder=encoder)
        weights = training.full_simplex_grid(rd)
        report["mode_names"] = [mc.name for mc in env.mode_configs]
        for mode in range(4):
            env._apply_mode(mode)
            reset_seed = 107000000 + 1000 * seed + 10 * mode
            for wi in WEIGHT_INDICES[env_name]:
                start = None
                for name in MODEL_NAMES:
                    result = episode(env, models[name], weights[wi], reset_seed, grouping)
                    if start is None:
                        start = result["start"]
                    assert result.pop("start") == start
                    report["rows"].append(dict(model=name, mode=mode, weight_index=wi,
                        weight=weights[wi].tolist(), reset_seed=reset_seed, **result))
            write(output, report)
        report.update(complete=True, paired_resets=True, actual_work=work(report["rows"]),
            alignment={name: {w: alignment(report["rows"], name, w) for w in WINDOWS} for name in MODEL_NAMES})
        check(report)
        write(output, report)
        print("DONE", PROTOCOL, env_name, seed, json.dumps(report["actual_work"]), flush=True)
        return report
    finally:
        env.close()


def summarize(root=ROOT):
    result = dict(protocol=PROTOCOL, experiment=specification(), complete=True, environments={}, actual_work={},
                  decision="diagnostic_only_hold_context_group_descent", entropy_value_decomposition_available=False)
    for env in ENVIRONMENTS:
        records = [json.loads((Path(root) / env.replace("-", "_") / f"seed{s}.json").read_text()) for s in SEEDS]
        for seed, record in zip(SEEDS, records):
            check(record)
            assert (record["env"], record["seed"]) == (env, seed)
            for k, v in record["actual_work"].items():
                result["actual_work"][k] = result["actual_work"].get(k, 0) + v
        result["environments"][env] = dict(mode_names=records[0]["mode_names"],
            per_seed=[dict(seed=r["seed"], alignment=r["alignment"],
                feature_boundaries={name: [np.mean([row["after_warmup"]["feature_boundary_fraction"]
                    for row in r["rows"] if row["model"] == name and row["mode"] == mode
                    and row["after_warmup"]["rows"]], axis=0).tolist()
                    if any(row["model"] == name and row["mode"] == mode and row["after_warmup"]["rows"]
                           for row in r["rows"]) else None for mode in range(4)] for name in MODEL_NAMES})
                for r in records])
    write(Path(root) / "compact_readout.json", result)
    print("DONE", PROTOCOL, "summary", json.dumps(result["actual_work"]), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=ENVIRONMENTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    summarize(args.root) if args.summarize else run(args.env, args.seed, args.root)
