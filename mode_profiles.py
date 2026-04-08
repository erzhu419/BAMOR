# Mode profiles for BA-PR SAC environment
# Each mode defines a distinct environmental regime with different parameters
import numpy as np


MODE_PROFILES = {
    # ─── 正常模式 ───
    "normal": {
        "speed_mean_scale": 1.0,      # 速度均值不变
        "sigma": 1.5,                  # 标准方差
        "speed_cap": 15,               # 正常限速
        "od_global_mult": 1.0,         # 标准客流
        "station_od_overrides": {},     # 无站点特殊干预
        "affected_routes": None,       # 所有路段
    },

    # ─── 严重拥堵：某段道路事故/施工 ───
    "congestion_severe": {
        "speed_mean_scale": 0.3,       # 均值降到 30%
        "sigma": 3.0,                  # 波动也变大
        "speed_cap": 5,                # 限速降到 5 m/s
        "od_global_mult": 1.0,
        "station_od_overrides": {},
        "affected_routes": [3, 4, 5, 6],  # 只影响路段 3-6（中段路网）
    },

    # ─── 客流激增：学校放学 / 大型活动 ───
    "demand_surge": {
        "speed_mean_scale": 1.0,
        "sigma": 1.5,
        "speed_cap": 15,
        "od_global_mult": 1.5,         # 全局 1.5 倍
        "station_od_overrides": {      # 特定站点 OD 暴涨
            "X05": 5.0,                # 站点 X05 客流 5 倍
            "X06": 4.0,
            "X07": 3.0,
        },
        "affected_routes": None,
    },

    # ─── 全线瘫痪：极端天气 ───
    "extreme_weather": {
        "speed_mean_scale": 0.4,       # 全线均值降 60%
        "sigma": 4.0,                  # 极大方差
        "speed_cap": 8,                # 全线限速
        "od_global_mult": 0.3,         # 乘客也减少
        "station_od_overrides": {},
        "affected_routes": None,
    },

    # ─── 局部路段封闭 + 周边客流转移 ───
    "partial_closure": {
        "speed_mean_scale": 0.15,      # 封闭路段速度降到 15%
        "sigma": 1.0,
        "speed_cap": 3,                # 封闭路段限速 3
        "od_global_mult": 1.0,
        "station_od_overrides": {
            "X08": 4.0,
            "X09": 4.0,
            "X10": 3.0,
        },
        "affected_routes": [7, 8, 9],
    },
}

# ─── BAMOR: Target Cone Profiles ───
# BAMOR_engineering_doc §4.6.2
# 映射 run-length (当前局势的稳定度) 到物理上合法的 Target Cone 边界
# belief 维度为 max_run_length=20，我们将其分段映射到 3 个锥:

MODE_CONE_PROFILES = [
    # 类别 0: Crisis (run-length: 0~4) — 刚发生突变，极度不稳定
    # 物理约束: 必须保障车段服务，极化 w_headway -> 1.0
    {"mode": "crisis",      "cone_lower": 0.8, "cone_upper": 1.0, "w_anchor": [0.9, 0.1]},
    
    # 类别 1: Recovery (run-length: 5~14) — 正在恢复中
    # 物理约束: 适度关注能耗，但 headway 仍占主导
    {"mode": "recovery",    "cone_lower": 0.5, "cone_upper": 0.8, "w_anchor": [0.65, 0.35]},
    
    # 类别 2: Stable (run-length: 15~19) — 长期平稳
    # 物理约束: 路面畅通，允许大幅偏重能耗节省
    {"mode": "stable",      "cone_lower": 0.2, "cone_upper": 0.6, "w_anchor": [0.4, 0.6]},
]

# 用于将 20 维 belief 投影到 3 个 Target Cone 的混合矩阵
def get_cone_mixing_weights(belief_array):
    """
    Args: belief_array: [max_run_length] (e.g. 20)
    Returns: [3] mixing probabilities (sum to 1) corresponding to MODE_CONE_PROFILES
    """
    belief = np.array(belief_array)
    w_crisis = belief[0:5].sum()
    w_recovery = belief[5:15].sum()
    w_stable = belief[15:].sum()
    return np.array([w_crisis, w_recovery, w_stable])
