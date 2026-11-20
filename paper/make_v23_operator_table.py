"""Generate the BAMOR portfolio specification table from the selector implementation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_bamor_mujoco import ProgressExtremeHierarchicalOperatorSelector


def table_source():
    selector = ProgressExtremeHierarchicalOperatorSelector(3)
    names = {}
    rows = []
    groups = {}
    for family, prefix in [("coverage", "C"), ("recovery", "R"), ("extreme", "E")]:
        for index, arm in enumerate(selector.families[family], 1):
            names[arm] = f"{prefix}{index}"
            cfg = selector.trainer_config(arm)
            settings = dict(cfg, relabel_rate=selector.relabel_rate(arm),
                            interaction_sampler=selector._split_operator(arm)[1].split("_")[0])
            groups.setdefault(tuple(sorted(settings.items())), []).append(names[arm])
            rows.append(" & ".join([
                names[arm], cfg["preference_relabeling_strategy"],
                f"{selector.relabel_rate(arm):g}", str(cfg["critic_relabel_copies"]),
                str(cfg["envelope_target_candidates"]), str(cfg["envelope_policy_candidates"]),
                f"{cfg['vector_critic_coef']:g}", f"{cfg['behavior_clone_coef']:g}",
            ]) + r" \\")
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    warmup = ", ".join(names[arm] for arm in selector.warmup_sequence)
    shared = "; ".join("/".join(group) for group in groups.values() if len(group) > 1)
    return r"""\section{Portfolio Configuration Specification}
\label{app:operators}
\begin{table}[ht]
\centering\small
\caption{The 18 allocation slots of the BAMOR update portfolio across coverage
(C), recovery (R) and extreme-preference (E) families. Relabel denotes the
fraction of the replay minibatch assigned fresh preferences; $C_o$ counts total
preference copies per batch; $K_o^Q$ and $K_o^\pi$ are critic and actor
candidate counts; $\lambda_o$ and $b_o$ weight vector-target and
behavior-cloning losses.}
\label{tab:operators}
\begin{tabular}{llrrrrrr}
\toprule
Slot & Sampler & Relabel & $C_o$ & $K_o^Q$ & $K_o^\pi$ & $\lambda_o$ & $b_o$\\
\midrule
""" + "\n".join(rows) + r"""
\end{tabular}
\end{table}

Following the initial 5,000-step initialization phase, the first ten
1,500-step blocks follow the deterministic coverage schedule:
""" + warmup + ".\n\n" + (
        f"The 18 slots specify {len(groups)} distinct parameter settings. "
        f"The pairs {shared} share settings and keep separate scores and counts. "
        "For three objectives, C4 also coincides with C5 and R2, giving "
        "13 distinct behaviors.\n")


if __name__ == "__main__":
    path = Path(__file__).with_name("table_v23_operators.tex")
    path.write_text(table_source())
    print(path)
