#!/usr/bin/env python3
"""Aggregate MuJoCo policy-set Pareto unions from saved seed JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover - optional reporting dependency
    wilcoxon = None


V5_METHOD = "bamor_v5_contrast_moe_template_mix"
V6_METHOD = "bamor_v6_contrast_moe_head_template_mix"
UNION_METHOD = "v5_v6_union"
BAMOR_PORTFOLIO_UNION_METHOD = "bamor_v5_v6_union"
CAPQL_PORTFOLIO_UNION_METHOD = "capql_v5_v6_union"
CAPQL_3X_UNION_METHOD = "capql_3x_union"
BAMOR_V11_HC_UNION_METHOD = "bamor_v11_hc_union"
BAMOR_V12_HC_UNION_METHOD = "bamor_v12_hc_union"
BAMOR_V13_HC_UNION_METHOD = "bamor_v13_hc_union"
BAMOR_V14_HC_UNION_METHOD = "bamor_v14_hc_union"
BAMOR_V15_HC_UNION_METHOD = "bamor_v15_hc_union"
BAMOR_V16_HC_UNION_METHOD = "bamor_v16_hc_union"
BAMOR_V17_HC_UNION_METHOD = "bamor_v17_hc_union"
BAMOR_V18_HC_UNION_METHOD = "bamor_v18_hc_union"
BAMOR_V19_HC_UNION_METHOD = "bamor_v19_hc_union"
BAMOR_V20_HC_UNION_METHOD = "bamor_v20_hc_union"
BAMOR_V21_HC_UNION_METHOD = "bamor_v21_hc_union"
BAMOR_V22_HC_UNION_METHOD = "bamor_v22_hc_union"
BAMOR_V23_HC_UNION_METHOD = "bamor_v23_hc_union"
BAMOR_V23_SINGLE_METHOD = "bamor_v23_single"


@dataclass(frozen=True)
class Preset:
    env: str
    seeds: tuple[int, ...]
    search_dirs: tuple[str, ...]
    baselines: tuple[str, ...]
    union_methods: tuple[str, ...] = (V5_METHOD, V6_METHOD)
    union_name: str = UNION_METHOD


PRESETS = {
    "ant10": Preset(
        env="mo-ant-v5",
        seeds=tuple(range(10)),
        search_dirs=(
            "results/mujoco_ant_v5moe_transfer_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_ant_v5moe_ext_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_v6_headmoe_targeted_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_v6_headmoe_ensemble_remaining_scheduler_20260610_0001/mo_ant_v5",
        ),
        baselines=(
            "uniform",
            "bamor",
            "bamor_v3_contrast_mix",
            "bamor_v4_contrast_expert_mix",
            V5_METHOD,
            V6_METHOD,
        ),
    ),
    "ant20": Preset(
        env="mo-ant-v5",
        seeds=tuple(range(20)),
        search_dirs=(
            "results/mujoco_ant_v5moe_transfer_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_ant_v5moe_ext_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_v6_headmoe_targeted_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_v6_headmoe_ensemble_remaining_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_policy_set_20seed_ext_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_capql_3d20_scheduler_20260611_0001/mo_ant_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            "bamor_v4_contrast_expert_mix",
            V5_METHOD,
            V6_METHOD,
        ),
    ),
    "hopper10": Preset(
        env="mo-hopper-v5",
        seeds=tuple(range(10)),
        search_dirs=(
            "results/mujoco_hopper_v5moe_targeted_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v5moe_remaining_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v3mix_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v3mix_ext_scheduler_20260609_0002/mo_hopper_v5",
            "results/mujoco_hopper_v3norm_ext_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v4expert_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v4expert_ext_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_v6_headmoe_targeted_scheduler_20260610_0001/mo_hopper_v5",
            "results/mujoco_v6_headmoe_ensemble_remaining_scheduler_20260610_0001/mo_hopper_v5",
        ),
        baselines=(
            "uniform",
            "bamor",
            "bamor_v3_contrast_mix",
            "bamor_v4_contrast_expert_mix",
            V5_METHOD,
            V6_METHOD,
        ),
    ),
    "hopper20": Preset(
        env="mo-hopper-v5",
        seeds=tuple(range(20)),
        search_dirs=(
            "results/mujoco_hopper_v5moe_targeted_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v5moe_remaining_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v3mix_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v3mix_ext_scheduler_20260609_0002/mo_hopper_v5",
            "results/mujoco_hopper_v3norm_ext_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v4expert_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v4expert_ext_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_v6_headmoe_targeted_scheduler_20260610_0001/mo_hopper_v5",
            "results/mujoco_v6_headmoe_ensemble_remaining_scheduler_20260610_0001/mo_hopper_v5",
            "results/mujoco_policy_set_20seed_ext_scheduler_20260610_0001/mo_hopper_v5",
            "results/mujoco_capql_3d20_scheduler_20260611_0001/mo_hopper_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            "bamor_v4_contrast_expert_mix",
            V5_METHOD,
            V6_METHOD,
        ),
    ),
    "ant20_capql_portfolio": Preset(
        env="mo-ant-v5",
        seeds=tuple(range(20)),
        search_dirs=(
            "results/mujoco_ant_v5moe_transfer_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_ant_v5moe_ext_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_v6_headmoe_targeted_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_v6_headmoe_ensemble_remaining_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_policy_set_20seed_ext_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_capql_3d20_scheduler_20260611_0001/mo_ant_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            "bamor_v4_contrast_expert_mix",
            V5_METHOD,
            V6_METHOD,
        ),
        union_methods=("capql", V5_METHOD, V6_METHOD),
        union_name=CAPQL_PORTFOLIO_UNION_METHOD,
    ),
    "ant20_bamor_portfolio": Preset(
        env="mo-ant-v5",
        seeds=tuple(range(20)),
        search_dirs=(
            "results/mujoco_ant_v5moe_transfer_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_ant_v5moe_ext_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_v6_headmoe_targeted_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_v6_headmoe_ensemble_remaining_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_policy_set_20seed_ext_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_capql_3d20_scheduler_20260611_0001/mo_ant_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            "bamor_v4_contrast_expert_mix",
            V5_METHOD,
            V6_METHOD,
        ),
        union_methods=("bamor", V5_METHOD, V6_METHOD),
        union_name=BAMOR_PORTFOLIO_UNION_METHOD,
    ),
    "hopper20_capql_portfolio": Preset(
        env="mo-hopper-v5",
        seeds=tuple(range(20)),
        search_dirs=(
            "results/mujoco_hopper_v5moe_targeted_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v5moe_remaining_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v3mix_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v3mix_ext_scheduler_20260609_0002/mo_hopper_v5",
            "results/mujoco_hopper_v3norm_ext_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v4expert_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v4expert_ext_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_v6_headmoe_targeted_scheduler_20260610_0001/mo_hopper_v5",
            "results/mujoco_v6_headmoe_ensemble_remaining_scheduler_20260610_0001/mo_hopper_v5",
            "results/mujoco_policy_set_20seed_ext_scheduler_20260610_0001/mo_hopper_v5",
            "results/mujoco_capql_3d20_scheduler_20260611_0001/mo_hopper_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            "bamor_v4_contrast_expert_mix",
            V5_METHOD,
            V6_METHOD,
        ),
        union_methods=("capql", V5_METHOD, V6_METHOD),
        union_name=CAPQL_PORTFOLIO_UNION_METHOD,
    ),
    "hopper20_bamor_portfolio": Preset(
        env="mo-hopper-v5",
        seeds=tuple(range(20)),
        search_dirs=(
            "results/mujoco_hopper_v5moe_targeted_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v5moe_remaining_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v3mix_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v3mix_ext_scheduler_20260609_0002/mo_hopper_v5",
            "results/mujoco_hopper_v3norm_ext_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v4expert_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v4expert_ext_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_v6_headmoe_targeted_scheduler_20260610_0001/mo_hopper_v5",
            "results/mujoco_v6_headmoe_ensemble_remaining_scheduler_20260610_0001/mo_hopper_v5",
            "results/mujoco_policy_set_20seed_ext_scheduler_20260610_0001/mo_hopper_v5",
            "results/mujoco_capql_3d20_scheduler_20260611_0001/mo_hopper_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            "bamor_v4_contrast_expert_mix",
            V5_METHOD,
            V6_METHOD,
        ),
        union_methods=("bamor", V5_METHOD, V6_METHOD),
        union_name=BAMOR_PORTFOLIO_UNION_METHOD,
    ),
    "ant20_capql_3x": Preset(
        env="mo-ant-v5",
        seeds=tuple(range(20)),
        search_dirs=(
            "results/mujoco_capql_3d20_scheduler_20260611_0001/mo_ant_v5",
            "results/mujoco_capql_3x_3d20_scheduler_20260611_0002/mo_ant_v5",
        ),
        baselines=(
            "capql",
            "capql_rep1",
            "capql_rep2",
        ),
        union_methods=("capql", "capql_rep1", "capql_rep2"),
        union_name=CAPQL_3X_UNION_METHOD,
    ),
    "hopper20_capql_3x": Preset(
        env="mo-hopper-v5",
        seeds=tuple(range(20)),
        search_dirs=(
            "results/mujoco_capql_3d20_scheduler_20260611_0001/mo_hopper_v5",
            "results/mujoco_capql_3x_3d20_scheduler_20260611_0002/mo_hopper_v5",
        ),
        baselines=(
            "capql",
            "capql_rep1",
            "capql_rep2",
        ),
        union_methods=("capql", "capql_rep1", "capql_rep2"),
        union_name=CAPQL_3X_UNION_METHOD,
    ),
    "halfcheetah10_bamor_portfolio": Preset(
        env="mo-halfcheetah-v5",
        seeds=tuple(range(10)),
        search_dirs=(
            "results/mujoco_hc_v2_scheduler_20260608_191600/mo_halfcheetah_v5",
            "results/mujoco_hc_v3_mix_scheduler_20260609_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v3mix_ext_scheduler_20260609_0002/mo_halfcheetah_v5",
            "results/mujoco_hc_v9_scheduler_20260611_0001/mo_halfcheetah_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
        ),
        union_methods=("bamor", V5_METHOD, V6_METHOD),
        union_name=BAMOR_PORTFOLIO_UNION_METHOD,
    ),
    "halfcheetah10_bamor_v11_portfolio": Preset(
        env="mo-halfcheetah-v5",
        seeds=tuple(range(10)),
        search_dirs=(
            "results/mujoco_hc_v2_scheduler_20260608_191600/mo_halfcheetah_v5",
            "results/mujoco_hc_v3_mix_scheduler_20260609_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v3mix_ext_scheduler_20260609_0002/mo_halfcheetah_v5",
            "results/mujoco_hc_v9_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v11_relabel_scheduler_20260611_0001/mo_halfcheetah_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
        ),
        union_methods=(
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
        ),
        union_name=BAMOR_V11_HC_UNION_METHOD,
    ),
    "halfcheetah10_bamor_v12_portfolio": Preset(
        env="mo-halfcheetah-v5",
        seeds=tuple(range(10)),
        search_dirs=(
            "results/mujoco_hc_v2_scheduler_20260608_191600/mo_halfcheetah_v5",
            "results/mujoco_hc_v3_mix_scheduler_20260609_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v3mix_ext_scheduler_20260609_0002/mo_halfcheetah_v5",
            "results/mujoco_hc_v9_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v11_relabel_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v12_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
        ),
        union_methods=(
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
        ),
        union_name=BAMOR_V12_HC_UNION_METHOD,
    ),
    "halfcheetah10_bamor_v13_portfolio": Preset(
        env="mo-halfcheetah-v5",
        seeds=tuple(range(10)),
        search_dirs=(
            "results/mujoco_hc_v2_scheduler_20260608_191600/mo_halfcheetah_v5",
            "results/mujoco_hc_v3_mix_scheduler_20260609_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v3mix_ext_scheduler_20260609_0002/mo_halfcheetah_v5",
            "results/mujoco_hc_v9_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v11_relabel_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v12_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v13_blockop_scheduler_20260611_0001/mo_halfcheetah_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
        ),
        union_methods=(
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
        ),
        union_name=BAMOR_V13_HC_UNION_METHOD,
    ),
    "halfcheetah10_bamor_v14_portfolio": Preset(
        env="mo-halfcheetah-v5",
        seeds=tuple(range(10)),
        search_dirs=(
            "results/mujoco_hc_v2_scheduler_20260608_191600/mo_halfcheetah_v5",
            "results/mujoco_hc_v3_mix_scheduler_20260609_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v3mix_ext_scheduler_20260609_0002/mo_halfcheetah_v5",
            "results/mujoco_hc_v9_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v11_relabel_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v12_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v13_blockop_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v14_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
        ),
        union_methods=(
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
        ),
        union_name=BAMOR_V14_HC_UNION_METHOD,
    ),
    "halfcheetah10_bamor_v15_portfolio": Preset(
        env="mo-halfcheetah-v5",
        seeds=tuple(range(10)),
        search_dirs=(
            "results/mujoco_hc_v2_scheduler_20260608_191600/mo_halfcheetah_v5",
            "results/mujoco_hc_v3_mix_scheduler_20260609_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v3mix_ext_scheduler_20260609_0002/mo_halfcheetah_v5",
            "results/mujoco_hc_v9_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v11_relabel_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v12_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v13_blockop_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v14_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v15_policy_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
            "bamor_v15_adaptive_policy_envelope_mix",
        ),
        union_methods=(
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
            "bamor_v15_adaptive_policy_envelope_mix",
        ),
        union_name=BAMOR_V15_HC_UNION_METHOD,
    ),
    "halfcheetah10_bamor_v16_portfolio": Preset(
        env="mo-halfcheetah-v5",
        seeds=tuple(range(10)),
        search_dirs=(
            "results/mujoco_hc_v2_scheduler_20260608_191600/mo_halfcheetah_v5",
            "results/mujoco_hc_v3_mix_scheduler_20260609_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v3mix_ext_scheduler_20260609_0002/mo_halfcheetah_v5",
            "results/mujoco_hc_v9_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v11_relabel_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v12_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v13_blockop_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v14_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v15_policy_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v16_risk_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
            "bamor_v15_adaptive_policy_envelope_mix",
            "bamor_v16_risk_adaptive_mix",
        ),
        union_methods=(
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
            "bamor_v15_adaptive_policy_envelope_mix",
            "bamor_v16_risk_adaptive_mix",
        ),
        union_name=BAMOR_V16_HC_UNION_METHOD,
    ),
    "halfcheetah10_bamor_v17_portfolio": Preset(
        env="mo-halfcheetah-v5",
        seeds=tuple(range(10)),
        search_dirs=(
            "results/mujoco_hc_v2_scheduler_20260608_191600/mo_halfcheetah_v5",
            "results/mujoco_hc_v3_mix_scheduler_20260609_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v3mix_ext_scheduler_20260609_0002/mo_halfcheetah_v5",
            "results/mujoco_hc_v9_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v11_relabel_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v12_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v13_blockop_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v14_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v15_policy_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v16_risk_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v17_archive_risk_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v17_archive_risk_long_scheduler_20260611_0001/mo_halfcheetah_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
            "bamor_v15_adaptive_policy_envelope_mix",
            "bamor_v16_risk_adaptive_mix",
            "bamor_v17_archive_risk_mix",
        ),
        union_methods=(
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
            "bamor_v15_adaptive_policy_envelope_mix",
            "bamor_v16_risk_adaptive_mix",
            "bamor_v17_archive_risk_mix",
        ),
        union_name=BAMOR_V17_HC_UNION_METHOD,
    ),
    "halfcheetah10_bamor_v18_portfolio": Preset(
        env="mo-halfcheetah-v5",
        seeds=tuple(range(10)),
        search_dirs=(
            "results/mujoco_hc_v2_scheduler_20260608_191600/mo_halfcheetah_v5",
            "results/mujoco_hc_v3_mix_scheduler_20260609_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v3mix_ext_scheduler_20260609_0002/mo_halfcheetah_v5",
            "results/mujoco_hc_v9_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v11_relabel_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v12_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v13_blockop_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v14_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v15_policy_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v16_risk_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v17_archive_risk_long_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v18_coverage_archive_scheduler_20260611_0001/mo_halfcheetah_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
            "bamor_v15_adaptive_policy_envelope_mix",
            "bamor_v16_risk_adaptive_mix",
            "bamor_v17_archive_risk_mix",
            "bamor_v18_coverage_archive_mix",
        ),
        union_methods=(
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
            "bamor_v15_adaptive_policy_envelope_mix",
            "bamor_v16_risk_adaptive_mix",
            "bamor_v17_archive_risk_mix",
            "bamor_v18_coverage_archive_mix",
        ),
        union_name=BAMOR_V18_HC_UNION_METHOD,
    ),
    "halfcheetah10_bamor_v19_portfolio": Preset(
        env="mo-halfcheetah-v5",
        seeds=tuple(range(10)),
        search_dirs=(
            "results/mujoco_hc_v2_scheduler_20260608_191600/mo_halfcheetah_v5",
            "results/mujoco_hc_v3_mix_scheduler_20260609_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v3mix_ext_scheduler_20260609_0002/mo_halfcheetah_v5",
            "results/mujoco_hc_v9_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v11_relabel_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v12_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v13_blockop_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v14_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v15_policy_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v16_risk_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v17_archive_risk_long_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v18_coverage_archive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v19_hybrid_archive_scheduler_20260612_0001/mo_halfcheetah_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
            "bamor_v15_adaptive_policy_envelope_mix",
            "bamor_v16_risk_adaptive_mix",
            "bamor_v17_archive_risk_mix",
            "bamor_v18_coverage_archive_mix",
            "bamor_v19_hybrid_archive_mix",
        ),
        union_methods=(
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
            "bamor_v15_adaptive_policy_envelope_mix",
            "bamor_v16_risk_adaptive_mix",
            "bamor_v17_archive_risk_mix",
            "bamor_v18_coverage_archive_mix",
            "bamor_v19_hybrid_archive_mix",
        ),
        union_name=BAMOR_V19_HC_UNION_METHOD,
    ),
    "halfcheetah10_bamor_v20_portfolio": Preset(
        env="mo-halfcheetah-v5",
        seeds=tuple(range(10)),
        search_dirs=(
            "results/mujoco_hc_v2_scheduler_20260608_191600/mo_halfcheetah_v5",
            "results/mujoco_hc_v3_mix_scheduler_20260609_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v3mix_ext_scheduler_20260609_0002/mo_halfcheetah_v5",
            "results/mujoco_hc_v9_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v11_relabel_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v12_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v13_blockop_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v14_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v15_policy_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v16_risk_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v17_archive_risk_long_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v18_coverage_archive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v19_hybrid_archive_scheduler_20260612_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v20_hier_archive_scheduler_20260612_0001/mo_halfcheetah_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
            "bamor_v15_adaptive_policy_envelope_mix",
            "bamor_v16_risk_adaptive_mix",
            "bamor_v17_archive_risk_mix",
            "bamor_v18_coverage_archive_mix",
            "bamor_v19_hybrid_archive_mix",
            "bamor_v20_hier_archive_mix",
        ),
        union_methods=(
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
            "bamor_v15_adaptive_policy_envelope_mix",
            "bamor_v16_risk_adaptive_mix",
            "bamor_v17_archive_risk_mix",
            "bamor_v18_coverage_archive_mix",
            "bamor_v19_hybrid_archive_mix",
            "bamor_v20_hier_archive_mix",
        ),
        union_name=BAMOR_V20_HC_UNION_METHOD,
    ),
    "halfcheetah10_bamor_v21_portfolio": Preset(
        env="mo-halfcheetah-v5",
        seeds=tuple(range(10)),
        search_dirs=(
            "results/mujoco_hc_v2_scheduler_20260608_191600/mo_halfcheetah_v5",
            "results/mujoco_hc_v3_mix_scheduler_20260609_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v3mix_ext_scheduler_20260609_0002/mo_halfcheetah_v5",
            "results/mujoco_hc_v9_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v11_relabel_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v12_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v13_blockop_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v14_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v15_policy_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v16_risk_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v17_archive_risk_long_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v18_coverage_archive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v19_hybrid_archive_scheduler_20260612_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v20_hier_archive_scheduler_20260612_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v21_covbias_hier_targeted_scheduler_20260612_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v21_covbias_hier_scheduler_20260612_0001/mo_halfcheetah_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
            "bamor_v15_adaptive_policy_envelope_mix",
            "bamor_v16_risk_adaptive_mix",
            "bamor_v17_archive_risk_mix",
            "bamor_v18_coverage_archive_mix",
            "bamor_v19_hybrid_archive_mix",
            "bamor_v20_hier_archive_mix",
            "bamor_v21_covbias_hier_archive_mix",
        ),
        union_methods=(
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
            "bamor_v15_adaptive_policy_envelope_mix",
            "bamor_v16_risk_adaptive_mix",
            "bamor_v17_archive_risk_mix",
            "bamor_v18_coverage_archive_mix",
            "bamor_v19_hybrid_archive_mix",
            "bamor_v20_hier_archive_mix",
            "bamor_v21_covbias_hier_archive_mix",
        ),
        union_name=BAMOR_V21_HC_UNION_METHOD,
    ),
    "halfcheetah10_bamor_v22_portfolio": Preset(
        env="mo-halfcheetah-v5",
        seeds=tuple(range(10)),
        search_dirs=(
            "results/mujoco_hc_v2_scheduler_20260608_191600/mo_halfcheetah_v5",
            "results/mujoco_hc_v3_mix_scheduler_20260609_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v3mix_ext_scheduler_20260609_0002/mo_halfcheetah_v5",
            "results/mujoco_hc_v9_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v11_relabel_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v12_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v13_blockop_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v14_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v15_policy_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v16_risk_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v17_archive_risk_long_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v18_coverage_archive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v19_hybrid_archive_scheduler_20260612_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v20_hier_archive_scheduler_20260612_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v21_covbias_hier_targeted_scheduler_20260612_0003/mo_halfcheetah_v5",
            "results/mujoco_hc_v22_dense_hier_targeted_scheduler_20260612_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v22_dense_hier_scheduler_20260612_0001/mo_halfcheetah_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
            "bamor_v15_adaptive_policy_envelope_mix",
            "bamor_v16_risk_adaptive_mix",
            "bamor_v17_archive_risk_mix",
            "bamor_v18_coverage_archive_mix",
            "bamor_v19_hybrid_archive_mix",
            "bamor_v20_hier_archive_mix",
            "bamor_v22_dense_hier_archive_mix",
        ),
        union_methods=(
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
            "bamor_v15_adaptive_policy_envelope_mix",
            "bamor_v16_risk_adaptive_mix",
            "bamor_v17_archive_risk_mix",
            "bamor_v18_coverage_archive_mix",
            "bamor_v19_hybrid_archive_mix",
            "bamor_v20_hier_archive_mix",
            "bamor_v22_dense_hier_archive_mix",
        ),
        union_name=BAMOR_V22_HC_UNION_METHOD,
    ),
    "halfcheetah10_bamor_v23_portfolio": Preset(
        env="mo-halfcheetah-v5",
        seeds=tuple(range(10)),
        search_dirs=(
            "results/mujoco_hc_v2_scheduler_20260608_191600/mo_halfcheetah_v5",
            "results/mujoco_hc_v3_mix_scheduler_20260609_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v3mix_ext_scheduler_20260609_0002/mo_halfcheetah_v5",
            "results/mujoco_hc_v9_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v11_relabel_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v12_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v13_blockop_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v14_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v15_policy_envelope_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v16_risk_adaptive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v17_archive_risk_long_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v18_coverage_archive_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v19_hybrid_archive_scheduler_20260612_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v20_hier_archive_scheduler_20260612_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v22_dense_hier_targeted_scheduler_20260612_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v23_extreme_hier_targeted_scheduler_20260612_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v23_extreme_hier_scheduler_20260612_0001/mo_halfcheetah_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
            "bamor_v15_adaptive_policy_envelope_mix",
            "bamor_v16_risk_adaptive_mix",
            "bamor_v17_archive_risk_mix",
            "bamor_v18_coverage_archive_mix",
            "bamor_v19_hybrid_archive_mix",
            "bamor_v20_hier_archive_mix",
            "bamor_v22_dense_hier_archive_mix",
            "bamor_v23_extreme_hier_archive_mix",
        ),
        union_methods=(
            "bamor",
            "bamor_v3_contrast_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v11_contrast_relabel_mix",
            "bamor_v12_adaptive_operator_mix",
            "bamor_v13_block_operator_mix",
            "bamor_v14_adaptive_envelope_mix",
            "bamor_v15_adaptive_policy_envelope_mix",
            "bamor_v16_risk_adaptive_mix",
            "bamor_v17_archive_risk_mix",
            "bamor_v18_coverage_archive_mix",
            "bamor_v19_hybrid_archive_mix",
            "bamor_v20_hier_archive_mix",
            "bamor_v22_dense_hier_archive_mix",
            "bamor_v23_extreme_hier_archive_mix",
        ),
        union_name=BAMOR_V23_HC_UNION_METHOD,
    ),
    "halfcheetah20_bamor_v23_single": Preset(
        env="mo-halfcheetah-v5",
        seeds=tuple(range(20)),
        search_dirs=(
            "results/mujoco_hc_v2_scheduler_20260608_191600/mo_halfcheetah_v5",
            "results/mujoco_hc_v9_scheduler_20260611_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v22_dense_hier_targeted_scheduler_20260612_0001/mo_halfcheetah_v5",
            "results/mujoco_hc_v23_extreme_hier_targeted_scheduler_20260612_0001/mo_halfcheetah_v5",
            "results/mujoco_v23_crossenv20_scheduler_20260612_0001/mo_halfcheetah_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor_v22_dense_hier_archive_mix",
            "bamor_v23_extreme_hier_archive_mix",
        ),
        union_methods=("bamor_v23_extreme_hier_archive_mix",),
        union_name=BAMOR_V23_SINGLE_METHOD,
    ),
    "ant20_bamor_v23_single": Preset(
        env="mo-ant-v5",
        seeds=tuple(range(20)),
        search_dirs=(
            "results/mujoco_ant_v5moe_transfer_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_ant_v5moe_ext_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_v6_headmoe_targeted_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_v6_headmoe_ensemble_remaining_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_policy_set_20seed_ext_scheduler_20260610_0001/mo_ant_v5",
            "results/mujoco_capql_3d20_scheduler_20260611_0001/mo_ant_v5",
            "results/mujoco_v23_crossenv20_scheduler_20260612_0001/mo_ant_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            "bamor_v4_contrast_expert_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v23_extreme_hier_archive_mix",
        ),
        union_methods=("bamor_v23_extreme_hier_archive_mix",),
        union_name=BAMOR_V23_SINGLE_METHOD,
    ),
    "hopper20_bamor_v23_single": Preset(
        env="mo-hopper-v5",
        seeds=tuple(range(20)),
        search_dirs=(
            "results/mujoco_hopper_v5moe_targeted_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v5moe_remaining_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v3mix_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v3mix_ext_scheduler_20260609_0002/mo_hopper_v5",
            "results/mujoco_hopper_v3norm_ext_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v4expert_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_hopper_v4expert_ext_scheduler_20260609_0001/mo_hopper_v5",
            "results/mujoco_v6_headmoe_targeted_scheduler_20260610_0001/mo_hopper_v5",
            "results/mujoco_v6_headmoe_ensemble_remaining_scheduler_20260610_0001/mo_hopper_v5",
            "results/mujoco_policy_set_20seed_ext_scheduler_20260610_0001/mo_hopper_v5",
            "results/mujoco_capql_3d20_scheduler_20260611_0001/mo_hopper_v5",
            "results/mujoco_v23_crossenv20_scheduler_20260612_0001/mo_hopper_v5",
        ),
        baselines=(
            "uniform",
            "capql",
            "bamor",
            "bamor_v3_contrast_mix",
            "bamor_v4_contrast_expert_mix",
            V5_METHOD,
            V6_METHOD,
            "bamor_v23_extreme_hier_archive_mix",
        ),
        union_methods=("bamor_v23_extreme_hier_archive_mix",),
        union_name=BAMOR_V23_SINGLE_METHOD,
    ),
}


def reference_point_for_env(env_name: str, reward_dim: int) -> np.ndarray:
    if "mountaincarcontinuous" in env_name:
        return np.array([-1100.0, -100.0], dtype=np.float64)
    if "halfcheetah" in env_name:
        return np.array([-50.0, -3000.0], dtype=np.float64)
    if reward_dim == 3 and any(k in env_name for k in ("hopper", "walker2d", "ant")):
        return np.array([-50.0, -100.0, -3000.0], dtype=np.float64)
    if reward_dim == 2:
        return np.array([-1e3, -1e3], dtype=np.float64)
    if reward_dim == 3:
        return np.array([-1e3, -1e3, -1e4], dtype=np.float64)
    raise ValueError(f"No default reference point for env={env_name!r}, reward_dim={reward_dim}")


def compute_hv_2d(points: np.ndarray, ref_point: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    ref = np.asarray(ref_point, dtype=np.float64)
    pts = pts[np.all(pts > ref, axis=1)]
    if len(pts) == 0:
        return 0.0

    order = np.lexsort((-pts[:, 1], -pts[:, 0]))
    nd = pts[order]
    hv = 0.0
    max_obj1 = ref[1]
    for obj0, obj1 in nd:
        if obj1 > max_obj1:
            hv += (obj0 - ref[0]) * (obj1 - max_obj1)
            max_obj1 = obj1
    return float(hv)


def compute_hv_nd(returns: Iterable[Iterable[float]], ref_point: Iterable[float]) -> float:
    pts = np.asarray(list(returns), dtype=np.float64)
    ref = np.asarray(ref_point, dtype=np.float64)
    if pts.size == 0:
        return 0.0
    if pts.ndim != 2 or pts.shape[1] != ref.shape[0]:
        raise ValueError(f"HV shape mismatch: points={pts.shape}, ref={ref.shape}")
    if pts.shape[1] == 2:
        return compute_hv_2d(pts, ref)
    if pts.shape[1] != 3:
        raise ValueError(f"Only 2-D/3-D HV is supported, got {pts.shape[1]} objectives")

    pts = pts[np.all(pts > ref, axis=1)]
    if len(pts) == 0:
        return 0.0

    z_levels = np.unique(np.concatenate(([ref[2]], pts[:, 2])))
    z_levels.sort()
    hv = 0.0
    for z_lo, z_hi in zip(z_levels[:-1], z_levels[1:]):
        if z_hi <= z_lo:
            continue
        active = pts[pts[:, 2] >= z_hi]
        if len(active) == 0:
            continue
        hv += compute_hv_2d(active[:, :2], ref[:2]) * (z_hi - z_lo)
    return float(hv)


def parse_seeds(parts: list[str]) -> tuple[int, ...]:
    seeds: list[int] = []
    for part in parts:
        for token in part.replace(",", " ").split():
            if "-" in token:
                lo, hi = token.split("-", 1)
                seeds.extend(range(int(lo), int(hi) + 1))
            else:
                seeds.append(int(token))
    return tuple(dict.fromkeys(seeds))


def find_run(search_dirs: list[Path], method: str, seed: int) -> Path | None:
    filename = f"{method}_seed{seed}.json"
    for directory in search_dirs:
        direct = directory / filename
        if direct.exists():
            return direct
        matches = sorted(directory.rglob(filename)) if directory.exists() else []
        if matches:
            return matches[0]
    return None


def load_run(path: Path) -> dict:
    with path.open() as fh:
        run = json.load(fh)
    if "pareto_overall" not in run:
        raise ValueError(f"{path} has no pareto_overall")
    return run


def returns_from_run(run: dict) -> np.ndarray:
    return np.asarray([p["return"] for p in run["pareto_overall"]], dtype=np.float64)


def ref_from_run(env: str, run: dict) -> np.ndarray:
    ref = run.get("ref_point")
    if ref is not None:
        return np.asarray(ref, dtype=np.float64)
    returns = returns_from_run(run)
    return reference_point_for_env(env, returns.shape[1])


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    sorted_arr = np.sort(arr)
    bottom_k = max(1, int(np.ceil(0.2 * len(sorted_arr))))
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "bottom20_mean": float(sorted_arr[:bottom_k].mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def paired_delta(primary: dict[int, float], baseline: dict[int, float]) -> dict:
    seeds = sorted(set(primary) & set(baseline))
    diffs = np.asarray([primary[s] - baseline[s] for s in seeds], dtype=np.float64)
    p_value = None
    if wilcoxon is not None and len(diffs) > 0 and np.any(diffs != 0):
        p_value = float(wilcoxon(diffs, alternative="two-sided").pvalue)
    return {
        "seeds": seeds,
        "mean_delta": float(diffs.mean()) if len(diffs) else None,
        "wins": int((diffs > 0).sum()),
        "n": int(len(diffs)),
        "wilcoxon_p": p_value,
        "deltas": {str(seed): float(delta) for seed, delta in zip(seeds, diffs)},
    }


def seed_level_diagnostics(
    env: str,
    seeds: tuple[int, ...],
    methods: tuple[str, ...],
    by_method: dict[str, dict[int, float]],
    union_methods: tuple[str, ...],
    union_name: str,
) -> tuple[list[dict], dict]:
    diagnostics: list[dict] = []
    union_gains: list[float] = []
    best_counts = {method: 0 for method in union_methods}
    best_counts["tie"] = 0
    strict_gain_count = 0
    eps = 1e-6

    for seed in seeds:
        union_hv = by_method.get(union_name, {}).get(seed)
        constituent_hvs = {
            method: by_method.get(method, {}).get(seed)
            for method in union_methods
            if seed in by_method.get(method, {})
        }
        method_hvs = {
            method: by_method.get(method, {}).get(seed)
            for method in methods
            if seed in by_method.get(method, {})
        }
        ordered = sorted(method_hvs.items(), key=lambda item: item[1], reverse=True)
        ranks = {method: rank for rank, (method, _hv) in enumerate(ordered, start=1)}

        best_constituent = None
        best_constituent_hv = None
        union_gain = None
        if union_hv is not None and constituent_hvs:
            best_constituent_hv = max(constituent_hvs.values())
            best_methods = [
                method for method, hv in constituent_hvs.items()
                if abs(hv - best_constituent_hv) <= eps
            ]
            best_constituent = "tie" if len(best_methods) > 1 else best_methods[0]
            best_counts[best_constituent] = best_counts.get(best_constituent, 0) + 1
            union_gain = union_hv - best_constituent_hv
            union_gains.append(union_gain)
            if union_gain > eps:
                strict_gain_count += 1

        diagnostics.append({
            "env": env,
            "seed": seed,
            "union_hv": union_hv,
            "union_rank": ranks.get(union_name),
            "best_constituent": best_constituent,
            "best_constituent_hv": best_constituent_hv,
            "union_gain_over_best_constituent": union_gain,
            "constituent_hv": constituent_hvs,
            "method_hv": method_hvs,
        })

    complementarity = describe(union_gains)
    complementarity.update({
        "strict_gain_count": strict_gain_count,
        "n": len(union_gains),
        "best_constituent_counts": best_counts,
    })
    return diagnostics, complementarity


def fmt_b(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value / 1e9:.3f}B"


def aggregate_env(
    env: str,
    seeds: tuple[int, ...],
    search_dirs: list[Path],
    baselines: tuple[str, ...],
    union_methods: tuple[str, ...],
    union_name: str,
) -> dict:
    methods = tuple(dict.fromkeys((*baselines, union_name)))
    per_seed: dict[str, dict[str, dict]] = {}
    by_method: dict[str, dict[int, float]] = {method: {} for method in methods}
    paths: dict[str, dict[str, str]] = {method: {} for method in baselines}
    missing: list[dict] = []

    for seed in seeds:
        seed_key = str(seed)
        per_seed[seed_key] = {}
        loaded: dict[str, dict] = {}
        refs: list[np.ndarray] = []
        for method in baselines:
            path = find_run(search_dirs, method, seed)
            if path is None:
                missing.append({"seed": seed, "method": method})
                continue
            run = load_run(path)
            loaded[method] = run
            paths[method][seed_key] = str(path)
            ref = ref_from_run(env, run)
            refs.append(ref)
            hv = compute_hv_nd(returns_from_run(run), ref)
            by_method[method][seed] = hv
            per_seed[seed_key][method] = {"hv": hv, "path": str(path)}

        if all(method in loaded for method in union_methods):
            ref = refs[0] if refs else ref_from_run(env, loaded[union_methods[0]])
            union_returns = np.concatenate(
                [returns_from_run(loaded[method]) for method in union_methods], axis=0
            )
            union_hv = compute_hv_nd(union_returns, ref)
            by_method[union_name][seed] = union_hv
            per_seed[seed_key][union_name] = {
                "hv": union_hv,
                "union_methods": list(union_methods),
                "num_points": int(len(union_returns)),
            }
        else:
            missing_methods = [method for method in union_methods if method not in loaded]
            missing.append({"seed": seed, "method": union_name, "missing_union_methods": missing_methods})

    summary = {
        method: describe([by_method[method][s] for s in seeds if s in by_method[method]])
        for method in methods
    }
    ranks: dict[str, list[int]] = {method: [] for method in methods}
    for seed in seeds:
        seed_values = [
            (method, by_method[method][seed])
            for method in methods
            if seed in by_method[method]
        ]
        seed_values.sort(key=lambda item: item[1], reverse=True)
        for rank, (method, _hv) in enumerate(seed_values, start=1):
            ranks[method].append(rank)
    for method in methods:
        if ranks[method]:
            summary[method]["mean_rank"] = float(stats.mean(ranks[method]))
            summary[method]["median_rank"] = float(stats.median(ranks[method]))

    paired = {
        method: paired_delta(by_method[union_name], by_method[method])
        for method in baselines
        if method in by_method
    }
    seed_diagnostics, complementarity = seed_level_diagnostics(
        env, seeds, methods, by_method, union_methods, union_name)
    return {
        "env": env,
        "seeds": list(seeds),
        "search_dirs": [str(p) for p in search_dirs],
        "union_name": union_name,
        "union_methods": list(union_methods),
        "baselines": list(baselines),
        "summary": summary,
        "paired_vs_union": paired,
        "constituent_complementarity": complementarity,
        "seed_diagnostics": seed_diagnostics,
        "per_seed": per_seed,
        "paths": paths,
        "missing": missing,
    }


def markdown_for_result(result: dict) -> str:
    lines: list[str] = []
    lines.append(f"## {result['env']}")
    lines.append("")
    lines.append("| method | n | mean HV | median HV | q25 HV | bottom20 HV | min HV | mean rank |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for method, row in result["summary"].items():
        rank = row.get("mean_rank")
        rank_txt = "NA" if rank is None else f"{rank:.2f}"
        lines.append(
            f"| `{method}` | {row.get('n', 0)} | {fmt_b(row.get('mean'))} | "
            f"{fmt_b(row.get('median'))} | {fmt_b(row.get('q25'))} | "
            f"{fmt_b(row.get('bottom20_mean'))} | {fmt_b(row.get('min'))} | {rank_txt} |"
        )
    lines.append("")
    lines.append(f"Paired deltas for `{result['union_name']}`:")
    lines.append("")
    lines.append("| baseline | mean delta | wins | Wilcoxon p |")
    lines.append("|---|---:|---:|---:|")
    for method, row in result["paired_vs_union"].items():
        p = row.get("wilcoxon_p")
        p_txt = "NA" if p is None else f"{p:.4f}"
        lines.append(
            f"| `{method}` | {fmt_b(row.get('mean_delta'))} | "
            f"{row.get('wins', 0)}/{row.get('n', 0)} | {p_txt} |"
        )
    lines.append("")
    comp = result.get("constituent_complementarity", {})
    if comp:
        counts = comp.get("best_constituent_counts", {})
        counts_txt = ", ".join(
            f"{method}={count}" for method, count in counts.items() if count
        ) or "none"
        lines.append(
            f"Constituent complementarity: mean union gain over the better "
            f"constituent is {fmt_b(comp.get('mean'))}; strict gains on "
            f"{comp.get('strict_gain_count', 0)}/{comp.get('n', 0)} seeds; "
            f"best-constituent counts: {counts_txt}."
        )
        lines.append("")
    lines.append("Per-seed union HV:")
    lines.append("")
    seed_values = []
    for seed in result["seeds"]:
        row = result["per_seed"].get(str(seed), {}).get(result["union_name"])
        seed_values.append(f"s{seed}={fmt_b(row['hv'])}" if row else f"s{seed}=NA")
    lines.append(", ".join(seed_values))
    lines.append("")
    if result["missing"]:
        lines.append("Missing inputs:")
        lines.append("")
        for item in result["missing"]:
            lines.append(f"- `{item}`")
        lines.append("")
    return "\n".join(lines)


def write_seed_csv(results: list[dict], path: Path) -> None:
    methods: list[str] = []
    for result in results:
        methods.extend(result.get("summary", {}).keys())
    methods = list(dict.fromkeys(methods))

    base_fields = [
        "env",
        "seed",
        "union_name",
        "union_hv",
        "union_rank",
        "best_constituent",
        "best_constituent_hv",
        "union_gain_over_best_constituent",
    ]
    fieldnames = base_fields + [f"hv:{method}" for method in methods]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            union_name = result["union_name"]
            for diag in result.get("seed_diagnostics", []):
                row = {
                    "env": diag["env"],
                    "seed": diag["seed"],
                    "union_name": union_name,
                    "union_hv": diag.get("union_hv"),
                    "union_rank": diag.get("union_rank"),
                    "best_constituent": diag.get("best_constituent"),
                    "best_constituent_hv": diag.get("best_constituent_hv"),
                    "union_gain_over_best_constituent": diag.get(
                        "union_gain_over_best_constituent"),
                }
                for method, hv in diag.get("method_hv", {}).items():
                    row[f"hv:{method}"] = hv
                writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute policy-set Pareto union HV from MuJoCo result JSONs.")
    parser.add_argument("--preset", action="append", choices=sorted(PRESETS),
                        help="Use a built-in result-dir preset. Repeatable.")
    parser.add_argument("--env", help="Environment name for a custom aggregation.")
    parser.add_argument("--search-dir", action="append", default=[],
                        help="Directory to search for *_seed*.json files. Repeatable.")
    parser.add_argument("--seeds", nargs="+", default=[],
                        help="Seed list/ranges, e.g. --seeds 0-9 or --seeds 0 1 2.")
    parser.add_argument("--baseline", action="append", default=[],
                        help="Baseline method to include. Repeatable.")
    parser.add_argument("--union-method", action="append", default=[],
                        help="Method whose Pareto front contributes to the union. Repeatable.")
    parser.add_argument("--union-name", default=UNION_METHOD,
                        help="Name for the synthetic union row.")
    parser.add_argument("--write-json", help="Optional output JSON path.")
    parser.add_argument("--write-md", help="Optional output Markdown path.")
    parser.add_argument("--write-seed-csv",
                        help="Optional seed-level diagnostics CSV path.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    specs: list[Preset] = []
    for name in args.preset or []:
        specs.append(PRESETS[name])

    if args.env:
        if not args.search_dir:
            raise SystemExit("--search-dir is required with --env")
        if not args.seeds:
            raise SystemExit("--seeds is required with --env")
        union_methods = tuple(args.union_method or [V5_METHOD, V6_METHOD])
        baselines = tuple(dict.fromkeys([*(args.baseline or []), *union_methods]))
        specs.append(Preset(
            env=args.env,
            seeds=parse_seeds(args.seeds),
            search_dirs=tuple(args.search_dir),
            baselines=baselines,
            union_methods=union_methods,
            union_name=args.union_name,
        ))

    if not specs:
        raise SystemExit("Provide --preset and/or --env.")

    all_results = []
    md_parts = ["# MuJoCo Policy-Set Union Summary", ""]
    for spec in specs:
        result = aggregate_env(
            env=spec.env,
            seeds=spec.seeds,
            search_dirs=[Path(p) for p in spec.search_dirs],
            baselines=spec.baselines,
            union_methods=spec.union_methods,
            union_name=spec.union_name,
        )
        all_results.append(result)
        md = markdown_for_result(result)
        md_parts.append(md)
        print(md)

    output = {
        "results": all_results,
    }
    if args.write_json:
        out = Path(args.write_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            json.dump(output, fh, indent=2)
        print(f"Wrote JSON: {out}")
    if args.write_md:
        out = Path(args.write_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            fh.write("\n".join(md_parts).rstrip() + "\n")
        print(f"Wrote Markdown: {out}")
    if args.write_seed_csv:
        out = Path(args.write_seed_csv)
        write_seed_csv(all_results, out)
        print(f"Wrote seed CSV: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
