"""Generate the current v25 arm specification; retain the existing file names."""
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
    shared = "; ".join(" = ".join(group) for group in groups.values() if len(group) > 1)
    return r"""\section{BAMOR-v25 Operator Specification}
\label{app:v23_operators}
\begin{table}[ht]
\centering\small
\caption{The 18 family-labelled arms generated from the v25 selector,
unchanged from v23.
C, R and E denote coverage, recovery and extreme families. Relabel is the
fraction of the original replay batch assigned fresh preferences; $C_o$ includes
that batch and any additional preference copies.}
\label{tab:v23_operators}
\begin{tabular}{llrrrrrr}
\toprule
Arm & Replay sampler & Relabel & $C_o$ & $K_o^Q$ & $K_o^\pi$ & $\lambda_o$ & $b_o$\\
\midrule
""" + "\n".join(rows) + r"""
\end{tabular}
\end{table}

After the initial 5,000 transitions, the first ten scored 1,500-transition
blocks use the following order:
""" + warmup + ".\n\n" + (
        f"The arms comprise {len(groups)} distinct sampler/update settings. "
        f"Identical settings occur in {shared}; "
        "these arms retain separate scores and allocation counts.\n")


if __name__ == "__main__":
    path = Path(__file__).with_name("table_v23_operators.tex")
    path.write_text(table_source())
    print(path)
