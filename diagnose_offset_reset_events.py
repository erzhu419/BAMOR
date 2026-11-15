"""Test whether extra stored offset alarms change a fixed prospective reset schedule."""

import json
from pathlib import Path

import run_two_stream_offset_confirmation as confirmation


PROTOCOL = "stored_offset_direct_window_reset_event_diagnostic_v1"
ROOT = confirmation.ROOT / "reset_event_diagnostic"
CONTROL, PRIMARY = confirmation.CONTROL, confirmation.PRIMARY
RULE = dict(event_input="eligible_window_alarm_at_window_end", startup_steps=5000,
    cooldown_environment_steps=2000, initial_last_reset_step=0,
    normal_startup_detection="armed_after_calibration", controller="prospective_not_deployed")
ZERO_WORK = ("additional_interactions", "initialization_probes", "predictor_optimizer_steps",
    "residual_scale_optimizer_steps", "offset_optimizer_steps", "offset_fit_calls", "affine_fits",
    "policy_optimizer_steps", "critic_optimizer_steps", "encoder_optimizer_steps")


def reset_events(windows, rule=RULE):
    last_reset, result = rule["initial_last_reset_step"], []
    for window in windows:
        now = rule["startup_steps"] + window["end_step"]
        if window["eligible"] and window["alarm"] and now - last_reset >= rule["cooldown_environment_steps"]:
            result.append(window["end_step"])
            last_reset = now
    return result


def case_readout(cases):
    first, second = (cases[a]["windows"] for a in (CONTROL, PRIMARY))
    confirmation.PARENT.development.verify_coverage(first, second)
    events = {a: reset_events(cases[a]["windows"]) for a in (CONTROL, PRIMARY)}
    changes = [dict(start_step=a["start_step"], end_step=a["end_step"],
        control_alarm=a["alarm"], offset_alarm=b["alarm"])
        for a, b in zip(first, second) if a["alarm"] != b["alarm"]]
    return dict(reset_steps=events, reset_events_equal=events[CONTROL] == events[PRIMARY],
        changed_alarm_windows=changes,
        first_post_alarm_delay={a: cases[a]["first_post_alarm_delay"] for a in (CONTROL, PRIMARY)})


def summarize(parent=confirmation.ROOT):
    selector = confirmation.acquisition.training.EpisodeReturnSamplingSelector(2)
    assert selector.detector_config["switch_cooldown"] == RULE["cooldown_environment_steps"]
    assert confirmation.PARENT.CONFIG["window"] == 100
    report = dict(protocol=PROTOCOL, complete=True, rule=RULE, retained_method="v25",
        evidence_integrated=False, confirmation_decision="hold_conditional_evidence_integration",
        deployed_v25_window_alarm_consumer=False, episode_detector_config=selector.detector_config,
        prior_acquisition_interactions=500380, source="fresh_two_stream_offset_confirmation",
        replay_scope="reset_schedule_on_recorded_windows_not_policy_counterfactual",
        changed_reset_cases=[], changed_alarm_case_count=0, changed_alarm_window_count=0,
        reset_event_totals={a: 0 for a in (CONTROL, PRIMARY)}, case_pairs=0, environments={},
        **dict.fromkeys(ZERO_WORK, 0))
    for env in confirmation.PARENT.ENVIRONMENTS:
        rows = []
        for seed in confirmation.PARENT.SEEDS:
            prior = json.loads((parent / env.replace("-", "_") / f"test_seed{seed}" / "readout.json").read_text())
            assert (prior["protocol"], prior["env"], prior["roles"], prior["complete"]) == (
                confirmation.PROTOCOL, env, confirmation.PARENT.fold_roles(seed), True)
            assert prior["calibration_steps"] == RULE["startup_steps"]
            cases = {}
            for case in confirmation.PARENT.CASES:
                result = case_readout({a: prior["arms"][a]["cases"][case] for a in (CONTROL, PRIMARY)})
                cases[case] = result
                report["case_pairs"] += 1
                changes = result["changed_alarm_windows"]
                report["changed_alarm_case_count"] += bool(changes)
                report["changed_alarm_window_count"] += len(changes)
                for arm in (CONTROL, PRIMARY):
                    report["reset_event_totals"][arm] += len(result["reset_steps"][arm])
                if not result["reset_events_equal"]:
                    report["changed_reset_cases"].append(dict(env=env, seed=seed, case=case, **result))
            rows.append(dict(seed=seed, cases=cases))
        report["environments"][env] = rows
    report["decision"] = "close_offset_direct_reset_route" if not report["changed_reset_cases"] else "inspect_offline_reset_differences"
    return report


if __name__ == "__main__":
    report = summarize()
    confirmation.PARENT.write(ROOT / "readout.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "environments"}))
