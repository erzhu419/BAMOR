"""
BAMOR Components: MahalanobisSurpriseComputer

对应 BAMOR.lean Part IV (L448-527):
- mahalanobis_sq: per-objective surprise squared, precision-weighted
- mahalanobis_likelihood: exp(-1/2 * d^2) > 0 (strictly positive)
- Guarantees Z > 0 automatically (unlike BAPR's scalar surprise)
"""

import numpy as np
import torch


class MahalanobisSurpriseComputer:
    """
    多目标 surprise 信号计算器。
    
    与 BAPR 的 SurpriseComputer 的区别:
    - BAPR: 标量 surprise |κ_cur - κ_tgt|
    - BAMOR: Mahalanobis 距离 √(Σ prec_i · surp_i²)
    
    每个目标有独立的 surprise，用 precision (aleatoric 风险的逆) 归一化后融合。
    高噪声目标 (大 κ) → 低 precision → surprise 被降权。
    """
    
    def __init__(self, num_objectives: int, ema_alpha: float = 0.1):
        self.m = num_objectives
        self.ema_alpha = ema_alpha
        self.ema_surprise = np.zeros(num_objectives)  # 每个目标独立 EMA
    
    def compute(self, current_reg_norm_vec: torch.Tensor,
                target_reg_norm_vec: torch.Tensor) -> tuple:
        """
        计算 Mahalanobis surprise 信号。
        
        Args:
            current_reg_norm_vec: [ensemble, m] — 当前网络的 per-objective reg norm
            target_reg_norm_vec:  [ensemble, m] — target 网络的 per-objective reg norm
        
        Returns:
            mahalanobis_surprise: float — 融合后的标量 (用于 belief_tracker.update())
            per_obj_surprise: np.ndarray[m] — 每个目标的 surprise (用于 logging)
        """
        # 逐目标 surprise: |κ_cur - κ_tgt| 取 ensemble 均值
        raw_surp = (current_reg_norm_vec - target_reg_norm_vec).abs().mean(dim=0).detach().cpu().numpy()
        # raw_surp: [m]
        
        # EMA 平滑
        self.ema_surprise = (self.ema_alpha * raw_surp +
                             (1 - self.ema_alpha) * self.ema_surprise)
        
        # Precision = 1 / aleatoric_variance
        # 高 κ 的目标 → 低 precision → surprise 被降权（噪声大的目标不太可信）
        aleatoric_var = target_reg_norm_vec.mean(dim=0).detach().cpu().numpy() + 1e-6
        precision = 1.0 / aleatoric_var  # [m]
        
        # BAMOR.lean L467-468: mahalanobis_sq = Σ prec_i * surp_i²
        mahalanobis_sq = (precision * self.ema_surprise ** 2).sum()
        mahalanobis_surprise = np.sqrt(mahalanobis_sq)
        
        return float(mahalanobis_surprise), self.ema_surprise.copy()
    
    def reset(self):
        """Episode 结束时重置 EMA。"""
        self.ema_surprise = np.zeros(self.m)


class ContextAwareWeightSampler:
    """
    BAMOR 核心机制：Target Cone 目标锥采样器
    
    对应 BAMOR.lean Part XI & XII:
    - 结合 Bayesian Belief 混合当前的有效 Target Cone。
    - 生成 w_restricted 和 w_anchor，实现 Focused Search 和 Local Tracking。
    """
    def __init__(self, mode_profiles: list, get_mixing_weights_fn):
        """
        Args:
            mode_profiles: List of dicts with keys ['cone_lower', 'cone_upper', 'w_anchor']
            get_mixing_weights_fn: Function to map belief array to mixing probabilities [n_modes]
        """
        self.mode_profiles = mode_profiles
        self.get_mixing_weights = get_mixing_weights_fn

    def sample_and_get_anchor(self, belief_array: np.ndarray, device='cpu'):
        """
        根据当前 belief 混合 Target Cone 边界和锚点，并采样新的偏好权重。
        
        Args:
            belief_array: [max_run_length] Bayesian belief vector
            device: torch device for return tensor
            
        Returns:
            w_restricted: [m] sampled preference weight (tensor on device)
            w_anchor: [m] anchor vector for regularization (tensor on device)
        """
        # 1. 将 run-length belief 映射为 各个 mode 的混合比例 [n_modes]
        mixing_weights = self.get_mixing_weights(belief_array)
        
        # 2. 混合 Target Cone 参数
        cone_lower = 0.0
        cone_upper = 0.0
        anchor_0 = 0.0  # 仅针对目标0 (headway) 的混合
        
        for w_mix, profile in zip(mixing_weights, self.mode_profiles):
            cone_lower += w_mix * profile['cone_lower']
            cone_upper += w_mix * profile['cone_upper']
            anchor_0 += w_mix * profile['w_anchor'][0]
            
        # 防护: 避免边界反演 (Pitfall #10)
        cone_lower = min(cone_lower, cone_upper)
        
        # 3. 采样权重的目标0分量 (w_headway)
        w_0 = np.random.uniform(cone_lower, cone_upper)
        
        # 4. 构建输出 [w_headway, w_energy] (m=2)
        # TODO: 扩展到 m>2 需在 simplex 上通过 Dirichlet 采样限制在目标锥内
        w_restricted = torch.tensor([w_0, 1.0 - w_0], dtype=torch.float32, device=device)
        w_anchor = torch.tensor([anchor_0, 1.0 - anchor_0], dtype=torch.float32, device=device)
        
        return w_restricted, w_anchor
