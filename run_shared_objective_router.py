"""Independent objective learners on one factual soft-policy collection stream."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import run_frozen_objective_router as router


source, training, base, write = router.source, router.training, router.base, router.write
PROTOCOL = "shared_stream_objective_router_v1"
ROOT = Path("results/iclr_shared_objective_router_20261002")
ENVIRONMENTS, SEEDS = router.ENVIRONMENTS, (2701, 2702, 2703)
STEPS, SWITCH_INTERVAL, KWARGS = router.STEPS, router.SWITCH_INTERVAL, router.KWARGS
ROUTES = (*router.ROUTES, "work_soft")
CONFIG = (
    "preference_relabeling_rate", "preference_relabeling_strategy", "critic_relabel_copies",
    "envelope_target_candidates", "vector_critic_coef", "envelope_policy_candidates", "behavior_clone_coef")
SHARED_FIELDS = ("states", "actions", "weights", "rewards", "next_states", "dones", "timesteps", "regimes")


def specification():
    return dict(environments=list(ENVIRONMENTS), seeds=list(SEEDS), collection_steps=STEPS,
        switch_interval=SWITCH_INTERVAL, training_kwargs=KWARGS, method=source.METHOD,
        objectives=source.specification()["trainer_classes"],
        collection="one_factual_soft_Gaussian_stream; independent_cold_raw_learner_and_encoder",
        contexts="each_learner_encodes_the_same_observed_transitions_with_its_own_tracker",
        shadow_rng="86000000+100000*seed+step; isolated_from_factual_soft_rng",
        calibration_and_gate={k: router.specification()[k] for k in (
            "prefix_steps", "horizon", "repeats", "calibration_episodes", "calibration", "features",
            "label", "estimator", "admission")},
        baselines="continue_exact_soft_prefix; also_top_up_soft_actor_critic_encoder_updates_to_candidate_total",
        baseline_steps="collection_steps+actual_calibration_interactions-2; two_extra_resume_initialization_probes_charged",
        top_up_rng="87000000+100000*seed+update; no_new_interaction",
        evaluation="five_frozen_final_execution_routes; full21_weights_all4_locked_modes_paired_resets",
        screen="original_priced_risk_screen_against_both_baselines; genuine_routing_value_required",
        privilege="no_environment_id_mode_id_training_critic_or_future_return_enters_router")


def encoder_update(state):
    trainer, encoder = state["trainer"], state["encoder"]
    es, ea, er, esn, et, _ = trainer.buffer.sample_encoder(trainer.batch_size)
    er_s = torch.tanh(er / 3.)
    z, pred_r, pred_delta = encoder(es, ea, er_s, esn)
    loss = F.mse_loss(pred_r, er_s) + F.mse_loss(pred_delta, torch.tanh((esn - es) / 3.))
    loss = loss + 1e-4 * z.pow(2).mean() + .2 * training.temporal_contrastive_loss(
        encoder.project(z), et, pos_window=256, temperature=.2)
    state["encoder_opt"].zero_grad()
    loss.backward()
    state["encoder_opt"].step()


class ShadowLearner:
    def __init__(self, seed):
        self.seed, self.state, self.rows = seed, None, 0

    def factory(self, *args, **kwargs):
        soft = source.TRAINERS["soft"](*args, **kwargs)
        with base.isolated_rng(self.seed):
            raw = source.MeanRawTrainer(*args, **kwargs)
            encoder = training.LatentTransitionEncoder(*args[:3], latent_dim=training.LATENT_CONTEXT_DIM,
                hidden=kwargs["hidden"]).to(raw.device)
            self.state = dict(trainer=raw, encoder=encoder,
                encoder_opt=torch.optim.Adam(encoder.parameters(), lr=kwargs["lr"]))
            self.trans = training.TransitionContextTracker(raw.reward_dim)
            self.learned = training.LearnedTransitionContextTracker(encoder, latent_dim=training.LATENT_CONTEXT_DIM)
        for trainer in (soft, raw):
            assert not trainer.policy_opt.state and not trainer.q_opt.state
            assert all(torch.equal(v, getattr(trainer, name + "_target").state_dict()[key])
                for name in ("q1", "q2") for key, v in getattr(trainer, name).state_dict().items())
        assert raw.buffer is not soft.buffer and raw.policy is not soft.policy
        return soft

    def configure(self, step, trainer):
        pass

    def interaction_step(self, step):
        return int(step)

    def after_step(self, step, trainer, encoder, env, operator):
        raw = self.state["trainer"]
        row = (trainer.buffer.ptr - 1) % trainer.buffer.capacity
        s, a, w, r, ns = (getattr(trainer.buffer, key)[row] for key in SHARED_FIELDS[:5])
        c = training.build_mujoco_context(source.METHOD, w, trans_tracker=self.trans, learned_tracker=self.learned)
        self.trans.update(s, a, r, ns)
        self.learned.update(s, a, r, ns, raw.device)
        nc = training.build_mujoco_context(source.METHOD, w, trans_tracker=self.trans, learned_tracker=self.learned)
        assert int(trainer.buffer.timesteps[row, 0]) == step
        raw.buffer.push(s, a, w, c, r, ns, nc, trainer.buffer.dones[row, 0], t=step, regime=-1)
        self.rows += 1
        for key in CONFIG:
            setattr(raw, key, getattr(trainer, key))
        if step >= KWARGS["learning_starts"]:
            with base.isolated_rng(86000000 + self.seed * 100000 + step):
                raw.update()
                encoder_update(self.state)


def train(env, seed, output, shadow=None, resume=None, steps=None):
    capture = base.Capture()
    capture.step = STEPS if steps is None else steps
    training.train_method(env, source.METHOD, capture.step, SWITCH_INTERVAL, seed, str(output.parent),
        trainer_factory=shadow.factory if shadow else None, update_router=shadow,
        resume_state=resume, state_probe=capture, **KWARGS)
    return capture.state


def price(training_steps, calibration_steps, probes, updates):
    return dict(training_interactions=training_steps, calibration_interactions=calibration_steps,
        initialization_probes=probes, actor_steps=updates, critic_steps=updates, encoder_steps=updates)


def endpoint_record(value):
    return dict(endpoint=source.local.source.endpoint(dict(evaluation=value)),
        evaluation={k: value[k] for k in ("evaluation_weights", "episode_seeds_per_mode", "ref_point")},
        pareto_per_mode=value["pareto_per_mode"],
        **{k: value[k] for k in ("interactions", "raw_choices", "soft_choices", "prefix_terminations")})


def check(record):
    assert (record["protocol"], record["experiment"], record["complete"]) == (PROTOCOL, specification(), True)
    assert record["env"] in ENVIRONMENTS and record["seed"] in SEEDS
    assert record["controls_passed"] and record["shared_transition_identity_passed"]
    assert record["cold_initializations"] == ["soft", "raw"] and record["shadow_rows"] == STEPS
    u, n = STEPS - KWARGS["learning_starts"] + 1, record["calibration"]["interactions"]
    b = STEPS + n - 2
    assert record["calibration"]["episodes"] == 42 and 2 < n <= 42 * router.HORIZON
    assert record["candidate_clocks"] == dict.fromkeys(("soft", "raw"), dict(actor=u, critic=u, encoder=u))
    assert record["baseline_total_steps"] == b
    assert record["baseline_clocks"] == dict(actor=u + n - 2, critic=u + n - 2, encoder=u + n - 2)
    assert record["work_baseline_clocks"] == dict(actor=2 * u, critic=2 * u, encoder=2 * u)
    assert record["top_up_updates"] == u - n + 2 >= 0
    assert record["candidate_price"] == price(STEPS, n, 2, 2 * u)
    assert record["baseline_price"] == price(b, 0, 4, u + n - 2)
    assert record["work_baseline_price"] == price(b, 0, 4, 2 * u)
    totals = [sum(record[key][k] for k in ("training_interactions", "calibration_interactions", "initialization_probes"))
              for key in ("candidate_price", "baseline_price", "work_baseline_price")]
    assert totals == [STEPS + n + 2] * 3
    assert record["actual_work"] == dict(training_interactions=b, calibration_interactions=n,
        initialization_probes=4, actor_steps=3 * u, critic_steps=3 * u, encoder_steps=3 * u,
        ridge_fits=2 if record["gate"]["ready"] else 0,
        evaluation_interactions=sum(r["interactions"] for r in record["evaluations"].values()))
    assert set(record["evaluations"]) == set(ROUTES)
    for row in record["evaluations"].values():
        assert row["evaluation"] == record["evaluations"]["soft"]["evaluation"]
        assert 0 < row["interactions"] <= 4 * 21 * router.REPEATS * router.HORIZON
        assert row["raw_choices"] + row["soft_choices"] == 4 * 21 * router.REPEATS
        assert all(np.isfinite(row["endpoint"][k]) for k in ("utility", "hv"))


def run(env, seed, root=ROOT):
    root = Path(root)
    frozen = json.loads((root / "protocol.json").read_text())
    assert (frozen["protocol"], frozen["experiment"]) == (PROTOCOL, specification())
    assert env in ENVIRONMENTS and seed in SEEDS
    output = root / env.replace("-", "_") / f"seed{seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        previous = json.loads(output.read_text())
        if previous["complete"]:
            check(previous)
            assert (previous["env"], previous["seed"]) == (env, seed)
            return previous
        raise RuntimeError("Incomplete attempt retained; inspect before a new-root retry")
    report = dict(protocol=PROTOCOL, experiment=specification(), env=env, seed=seed,
        complete=False, controls_passed=False, phase="shared_acquisition", evaluations={})
    write(output, report)
    states = {}
    try:
        shadow = ShadowLearner(seed)
        start = time.monotonic()
        print(f"SHARED_COLD_START env={env} seed={seed}", flush=True)
        states["soft"] = train(env, seed, output, shadow=shadow)
        states["raw"] = shadow.state
        for key in SHARED_FIELDS:
            np.testing.assert_array_equal(getattr(states["soft"]["trainer"].buffer, key)[:STEPS],
                getattr(shadow.state["trainer"].buffer, key)[:STEPS])
        for state in states.values():
            source.local.crossed.common_contract(state["trainer"])
            assert (state["trainer"].buffer.regimes[:STEPS] == -1).all()
        assert states["soft"]["encoder"] is not shadow.state["encoder"]
        policies = {name: router.snapshot(state) for name, state in states.items()}
        report.update(cold_initializations=["soft", "raw"], shared_transition_identity_passed=True,
            shadow_rows=shadow.rows, candidate_clocks={name: router.clocks(state) for name, state in states.items()},
            network_parameters_per_candidate={name: training.network_parameter_counts(
                state["trainer"].policy, (state["trainer"].q1, state["trainer"].q2), state["encoder"])
                for name, state in states.items()}, acquisition_seconds=time.monotonic() - start,
            candidate_mode_history=[list(row) for row in states["soft"]["env"].mode_history], phase="calibration")
        write(output, report)
        gate, calibration = router.calibrate(states["soft"], policies, seed)
        n, u = calibration["interactions"], STEPS - KWARGS["learning_starts"] + 1
        assert 2 < n <= u + 2
        b = STEPS + n - 2
        report.update(calibration=calibration, gate=gate.state_dict(), baseline_total_steps=b,
            top_up_updates=u - n + 2, phase="baseline_training")
        write(output, report)
        print(f"SHARED_PRICE env={env} seed={seed} calibration={n} baseline_steps={b} "
              f"top_up={report['top_up_updates']}", flush=True)
        start = time.monotonic()
        states["baseline"] = train(env, seed, output, resume=states["soft"], steps=b)
        report["baseline_clocks"] = router.clocks(states["baseline"])
        continued = {"soft": router.snapshot(states["baseline"])}
        for update in range(report["top_up_updates"]):
            with base.isolated_rng(87000000 + seed * 100000 + update):
                states["baseline"]["trainer"].update()
                encoder_update(states["baseline"])
        work_continued = {"soft": router.snapshot(states["baseline"])}
        report.update(work_baseline_clocks=router.clocks(states["baseline"]),
            baseline_seconds=time.monotonic() - start, phase="evaluation")
        write(output, report)
        for route in ROUTES:
            models = continued if route == "budget_soft" else work_continued if route == "work_soft" else policies
            value = router.evaluate(states["soft"], models, seed,
                "budget_soft" if route == "work_soft" else route, gate if route == "risk" else None)
            report["evaluations"][route] = endpoint_record(value)
            write(output, report)
        assert gate.state_dict() == report["gate"]
        report.update(candidate_price=price(STEPS, n, 2, 2 * u), baseline_price=price(b, 0, 4, u + n - 2),
            work_baseline_price=price(b, 0, 4, 2 * u), actual_work=dict(training_interactions=b,
                calibration_interactions=n, initialization_probes=4, actor_steps=3 * u, critic_steps=3 * u,
                encoder_steps=3 * u, ridge_fits=2 if gate.ready else 0,
                evaluation_interactions=sum(row["interactions"] for row in report["evaluations"].values())),
            controls_passed=True, complete=True, phase="complete")
        check(report)
        saved = dict(policies, budget_soft=continued["soft"], work_soft=work_continued["soft"])
        torch.save({name: {k: model[k].state_dict() for k in ("policy", "encoder")}
                    for name, model in saved.items()}, output.with_suffix(".actors.pt"))
        write(output, report)
        print(f"DONE {PROTOCOL} env={env} seed={seed} cost={json.dumps(report['actual_work'])}", flush=True)
        return report
    finally:
        for state in states.values():
            for key in ("env", "eval_env"):
                if key in state:
                    state[key].close()


def summarize(root=ROOT):
    result = dict(protocol=PROTOCOL, experiment=specification(), complete=True, environments={}, actual_work={})
    for env in ENVIRONMENTS:
        rows = [json.loads((Path(root) / env.replace("-", "_") / f"seed{seed}.json").read_text()) for seed in SEEDS]
        for seed, row in zip(SEEDS, rows):
            check(row)
            assert (row["env"], row["seed"]) == (env, seed)
            for key, value in row["actual_work"].items():
                result["actual_work"][key] = result["actual_work"].get(key, 0) + value
        methods = {name: {metric: dict(mean=float(np.mean(values)), minimum_seed=min(values), per_seed=values)
            for metric in ("utility", "hv")
            for values in [[r["evaluations"][name]["endpoint"][metric] for r in rows]]} for name in ROUTES}
        contrasts = {}
        for baseline in ("budget_soft", "work_soft"):
            metrics = {metric: dict(mean=methods["risk"][metric]["mean"] - methods[baseline][metric]["mean"],
                per_seed=[r["evaluations"]["risk"]["endpoint"][metric] - r["evaluations"][baseline]["endpoint"][metric] for r in rows],
                minimum_seed_difference=methods["risk"][metric]["minimum_seed"] - methods[baseline][metric]["minimum_seed"])
                for metric in ("utility", "hv")}
            modes = np.mean([np.subtract(r["evaluations"]["risk"]["endpoint"]["per_mode_utility"],
                r["evaluations"][baseline]["endpoint"]["per_mode_utility"]) for r in rows], axis=0).tolist()
            passed = metrics["utility"]["mean"] > 0 and metrics["hv"]["mean"] >= 0
            passed = passed and all(c["minimum_seed_difference"] >= 0 for c in metrics.values()) and min(modes) >= 0
            contrasts[baseline] = dict(metrics=metrics, mean_per_mode_utility_difference=modes, passed=passed)
        raw_choices = sum(r["evaluations"]["risk"]["raw_choices"] for r in rows)
        value = raw_choices > 0 and methods["risk"]["utility"]["mean"] > methods["soft"]["utility"]["mean"]
        result["environments"][env] = dict(methods=methods, risk_minus_baselines=contrasts,
            raw_choices=raw_choices, priced_screen_passed=all(c["passed"] for c in contrasts.values()),
            routing_value_passed=value, runs=rows)
    passed = all(r["priced_screen_passed"] for r in result["environments"].values())
    value = any(r["routing_value_passed"] for r in result["environments"].values())
    result["decision"] = "fresh_confirmation_required" if passed and value else "hold_shared_stream_objective_router"
    write(Path(root) / "readout.json", result)
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
