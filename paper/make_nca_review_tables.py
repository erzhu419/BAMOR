"""Build NCA revision diagnostics from seed-level reductions, without training."""

from collections import Counter
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np
from scipy.stats import t

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_nca_review import ANT, ANT_SEEDS, MAIN, SEEDS
from paper.make_evidence_tables import ENVS, HERE, RESULTS, SOURCES, table

REVIEW = "nca_review_20261120"
LABELS = dict(ENVS)
MODES = {MAIN[0]: ("Normal", "Low speed", "Low cost", "Locked"),
         MAIN[1]: ("Normal", "Fragile", "Cheap energy", "Grounded"),
         ANT: ("Energy save", "Race", "Cautious")}
HEAD = [r"BAMOR $U/G$", r"Native $U/G$", r"Full-simplex $U/G$"]
ARMS = ("v25", "capql", "capql_full_simplex")


def paired_intervals(delta):
    delta = np.asarray(delta, dtype=float)
    assert len(delta) >= 2 and np.isfinite(delta).all()
    n = len(delta)
    avg = delta.mean()
    r = t.ppf(.9875, n - 1) * delta.std(ddof=1) / np.sqrt(n)
    boot = delta[np.random.default_rng(0).integers(0, n, (20000, n))].mean(axis=1)
    return avg, np.quantile(boot, [.0125, .9875]), (avg - r, avg + r)


def interval_tex(ci):
    return f"${{[{ci[0]:.2f},{ci[1]:.2f}]}}$"


def reduce_stratum(rows, region="all", mode=None, seen=None):
    pairs = []
    cells = 0
    for row in rows:
        visited = {h for step, h in row["mode_history"] if step < 75000}
        modes = [h for h, info in enumerate(row["arms"]["v25"])
                 if region in info and (mode is None or mode == h)
                 and (seen is None or (h in visited) == seen)]
        if modes:
            cells += len(modes)
            pairs.append({a: np.mean([v[h][region] for h in modes], axis=0)
                          for a, v in row["arms"].items()})
    values = {a: np.array([p[a] for p in pairs]) for a in pairs[0]}
    return values, len(pairs), cells


def metrics_tex(values):
    cells = []
    for a in ARMS:
        if a not in values:
            cells.append("--")
        else:
            u, g = values[a].mean(axis=0)
            cells.append(f"${u:.1f}/{g:.1f}$")
    return cells


def review_grid(results):
    data = json.loads((results / REVIEW / "grid.json").read_text())
    assert data["protocol"] == "nca_locked_grid_reanalysis_v1"
    controls = {key: json.loads((results / SOURCES[key]).read_text())["environments"]
                for key in ("utility20", "full_simplex", "final")}
    seen_rows, mode_rows, geometry_rows, ant_rows, sensitivity_rows = [], [], [], [], []
    for env, rows in data["environments"].items():
        ant = env == ANT
        assert [r["seed"] for r in rows] == list(ANT_SEEDS if ant else SEEDS)
        archived = controls["final" if ant else "utility20"][env]
        for i, row in enumerate(rows):
            original = archived["runs"][i]
            assert row["seed"] == original["seed"]
            assert row["mode_history"] == original["v25"]["mode_history"]
            for method, modes in row["arms"].items():
                if method == "capql_full_simplex":
                    original_full = controls["full_simplex"][env]["seed_metrics"][i]
                    assert row["seed"] == original_full["seed"]
                    saved = original_full["methods"][method]
                else:
                    saved = original[method]
                np.testing.assert_allclose([m["all"][0] for m in modes],
                    saved["metrics"]["direct_utility"]["per_mode"], rtol=1e-8)
                assert all(g >= -1e-10 for m in modes
                           for key, (u, g) in m.items() if key != "mixed_energy")
        if not ant:
            assert Counter(len({h for step, h in r["mode_history"] if step < 75000})
                           for r in rows) == {1: 2, 2: 5, 3: 11, 4: 2}
            for seen, label in ((True, "Seen"), (False, "Unseen")):
                paired, n, cells = reduce_stratum(rows, seen=seen)
                assert (n, cells) == ((20, 53) if seen else (18, 27))
                diffs = [(paired["v25"][:, 0] - paired[method][:, 0]).mean()
                         for method in ARMS[1:]]
                seen_rows.append([LABELS[env], label, f"{n}/{cells}", *metrics_tex(paired),
                                  f"${diffs[0]:+.1f}/{diffs[1]:+.1f}$"])
            all_values, _, _ = reduce_stratum(rows)
            for left, right, name in (
                    ("v25", "capql", "BAMOR -- native"),
                    ("v25", "capql_full_simplex", "BAMOR -- full-simplex")):
                d, boot, classic = paired_intervals(all_values[left][:, 0] - all_values[right][:, 0])
                sensitivity_rows.append([LABELS[env], name, f"${d:+.2f}$",
                                         interval_tex(boot), interval_tex(classic)])
        for h, mode in enumerate(MODES[env]):
            paired, _, _ = reduce_stratum(rows, mode=h)
            mode_rows.append([LABELS[env], mode, *metrics_tex(paired)])
        for key, count in (("vertex", 2 if env == MAIN[0] else 3),
                           ("boundary", 0 if env == MAIN[0] else 12),
                           ("interior", 19 if env == MAIN[0] else 6)):
            if count:
                paired, _, _ = reduce_stratum(rows, region=key)
                geometry_rows.append([LABELS[env], key.capitalize(), str(count), *metrics_tex(paired)])
        if ant:
            for key in ("boundary", "interior"):
                a, b = [[r["arms"][method][0]["mixed_energy"][key] for r in rows]
                        for method in ("v25", "capql")]
                cells = sum(x["weights"] for x in a)
                assert cells == (40 if key == "boundary" else 30)
                wins = sum(x["vertex_wins"] for x in a)
                d = mean(x["u"] - y["u"] for x, y in zip(a, b))
                ant_rows.append([key.capitalize(), str(cells), f"${d:+.1f}$", f"{wins}/{cells}"])
            assert sum(int(r[-1].split("/")[0]) for r in ant_rows) == 66
    return {
        "table_grid_seen.tex": table(
            r"Requested utility $U$ and same-actor command shortfall $G$ at 75k, by prior mode exposure. "
            "Seeds/cells gives the number of eligible training seeds and seed--mode cells. "
            "Average over modes in each seed's stratum and weight eligible seeds equally. "
            r"$\Delta U$ reports BAMOR minus native / full-simplex CAPQL.",
            "tab:grid_seen", "@{}llrrrrr@{}", ["Task", "Mode", "Seeds/cells", *HEAD, r"$\Delta U$"],
            seen_rows, size=r"\footnotesize\setlength{\tabcolsep}{4pt}"),
        "table_grid_modes.tex": table(
            r"Requested utility $U$ and same-actor command shortfall $G$ by locked mode. "
            "Means over 20 training seeds on HalfCheetah and Hopper and five on Ant.",
            "tab:grid_modes", "@{}llrrr@{}", ["Task", "Locked mode", *HEAD], mode_rows),
        "table_grid_geometry.tex": table(
            r"Requested utility $U$ and same-actor shortfall $G$ by region of the requested preference. "
            "The maximum always searches the full grid; the region restricts only requests. "
            "Boundary excludes vertices.",
            "tab:grid_geometry", "@{}llrrrr@{}", ["Task", "Region", "Weights", *HEAD], geometry_rows),
        "table_ant_geometry.tex": table(
            r"Ant mode 0 among requests with $0 < w_{\mathrm{energy}} < 1$ (five seeds). "
            r"$\Delta U$ is BAMOR minus native CAPQL. Energy wins compare the same BAMOR actor "
            "queried at the pure-energy vertex versus the requested weight, both scored at the request.",
            "tab:ant_geometry", "@{}lrrr@{}", ["Region", "Seed--weight cells", r"$\Delta U$", "Energy wins"],
            ant_rows),
        "table_ci_sensitivity.tex": table(
            r"Interval sensitivity for the same original paired differences. Both intervals are marginal "
            r"97.5\%; the seeded percentile bootstrap was specified in the original protocols; "
            r"the paired $t$ interval is a sensitivity analysis.",
            "tab:ci_sensitivity", "@{}llrrr@{}",
            ["Task", "Utility contrast", "Mean", "Percentile bootstrap", r"Paired $t$"], sensitivity_rows)}


def review_routing(results):
    data = json.loads((results / REVIEW / "routing.json").read_text())
    assert data["protocol"] == "nca_75k_score_ablation_v1"
    baseline = json.loads((results / SOURCES["utility20"]).read_text())["environments"]
    scores, alloc, counts_by_task = [], [], []
    all_costs = Counter()
    for env in MAIN:
        group = data["environments"][env]
        rows, slots = group["runs"], group["slots"]
        assert [r["seed"] for r in rows] == list(SEEDS)
        assert len(slots) == len(set(slots)) == 18
        np.testing.assert_allclose([r["u_on"] for r in rows], [
            r["v25"]["metrics"]["direct_utility"]["per_mode"]
            for r in baseline[env]["runs"]], rtol=1e-8)
        on, off = (np.array([r[key] for r in rows]).mean(axis=1) for key in ("u_on", "u_neutral"))
        delta = on - off
        d, boot, classic = paired_intervals(delta)
        scores.append([LABELS[env], f"${on.mean():.1f}/{off.mean():.1f}$", f"${d:+.2f}$",
                      interval_tex(boot), interval_tex(classic),
                      "/".join(str(int(np.count_nonzero(mask))) for mask in
                               (delta > 0, delta < 0, delta == 0))])
        traces = [np.asarray([r[f"trace_{a}"]["full"] for r in rows])
                  for a in ("on", "neutral")]
        assert all(a.shape == (20, 46) and ((0 <= a) & (a < 18)).all() for a in traces)
        assert np.array_equal(traces[0][:, :10], traces[1][:, :10])
        full = [np.stack([np.bincount(s, minlength=18) for s in a]) for a in traces]
        free = [np.stack([np.bincount(s[10:], minlength=18) for s in a]) for a in traces]
        distance = np.abs(free[0]-free[1]).sum(axis=1) / 2
        order = (traces[0][:, 10:] != traces[1][:, 10:]).sum(axis=1)
        partial = sum(r["trace_on"]["partial"] != r["trace_neutral"]["partial"] for r in rows)
        alloc.append([LABELS[env], f"${distance.mean():.1f}/36$", f"${order.mean():.1f}/36$",
                      f"${partial}/20$"])
        counts_by_task.append((slots, [(a.mean(axis=0), b.mean(axis=0))
                                      for a, b in (full, free)]))
        for row in rows:
            assert row["training_interactions"] == 75000
            assert row["environment_initialization_probe_steps"] == 2
            assert 252 <= row["evaluation_interactions"] <= 252000
            all_costs.update({key: row[key] for key in (
                "training_interactions", "evaluation_interactions",
                "environment_initialization_probe_steps")})
    names = counts_by_task[0][0]
    assert all(slots == names for slots, _ in counts_by_task)
    family_count = Counter()
    counts_rows = []
    for j, name in enumerate(names):
        family = name.partition(":")[0]
        family_count[family] += 1
        counts_rows.append([f"{family[0].upper()}{family_count[family]}",
                            *(f"${on[j]:.2f}/{off[j]:.2f}$"
                              for _, pairs in counts_by_task for on, off in pairs)])
    return {
        "table_routing75.tex": table(
            "Matched score-on versus score-neutral BAMOR at 75k, 20 paired seeds per task. "
            "Both arms retain the quota mechanism, stochastic collection and the same backbone. "
            "Retrospective completion with the original score-on records. Intervals are marginal "
            r"97.5\%; W/L/T counts wins, losses and ties for score-on.",
            "tab:routing75", "@{}lrrrrr@{}", ["Task", r"$U$: on / neutral", r"$\Delta U$",
            "Bootstrap", r"Paired $t$", "W/L/T"], scores, size=r"\footnotesize"),
        "table_slot_shift.tex": table(
            "Allocation change within seed after the first ten deterministic blocks. "
            "Count displacement is half of the L1 distance between the two complete-block count "
            "vectors; order changes compare slots at equal block index. Means across 20 pairs.",
            "tab:slot_shift", "@{}lrrr@{}", ["Task", "Count displacement",
                "Order changes", "Partial slot differs"], alloc),
        "table_slot_counts.tex": table(
            "Mean complete feedback blocks per allocation slot at 75k across 20 seeds per task. "
            "Each cell is score-on / score-neutral. Post-warmup counts exclude the ten "
            "warmup blocks. Neither set counts the final 1,000-step block. HC denotes HalfCheetah.",
            "tab:slot_counts", "@{}lrrrr@{}", ["Slot", "HC all", "HC post-warmup",
                                              "Hopper all", "Hopper post-warmup"],
            counts_rows, size=r"\footnotesize"),
        "nca_routing_cost.tex": "".join(
            f"\\newcommand{{\\Nca{key.title().replace('_', '')}}}{{{value:,}}}\n"
            for key, value in all_costs.items())}


if __name__ == "__main__":
    outputs = review_grid(RESULTS)
    outputs.update(review_routing(RESULTS))
    for name, document in outputs.items():
        (HERE / name).write_text(document)
    print(f"Wrote {len(outputs)} NCA revision TeX inputs")
