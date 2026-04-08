"""
BAMOR Networks: Multi-Objective Q-Network and w-Conditioned Policy

对应 BAMOR_engineering_doc:
- §2.1: SoftQNetwork 输出 [ensemble, batch, m] (多目标)
- §4.3: PolicyNetwork 输入 [obs, w] (w-conditioned, CAPQL 风格)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class EmbeddingLayer(nn.Module):
    """类别特征嵌入层 (复用 BAPR 设计)"""
    def __init__(self, cat_code_dict, cat_cols):
        super(EmbeddingLayer, self).__init__()
        self.cat_code_dict = cat_code_dict
        self.cat_cols = cat_cols
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(len(cat_code_dict[col]), min(50, len(cat_code_dict[col]) // 2))
            for col in cat_cols
        })

    def forward(self, cat_tensor):
        embedding_tensor_group = []
        for idx, col in enumerate(self.cat_cols):
            layer = self.embeddings[col]
            out = layer(cat_tensor[:, idx])
            embedding_tensor_group.append(out)
        embed_tensor = torch.cat(embedding_tensor_group, dim=1)
        return embed_tensor


class VectorizedLinear(nn.Module):
    """Ensemble 线性层: weight shape [ensemble, in, out]"""
    def __init__(self, in_features, out_features, ensemble_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return x @ self.weight + self.bias


class VectorizedCritic(nn.Module):
    """
    Ensemble Critic 基类。
    
    BAMOR 改造: 输出层从 1 → num_objectives
    对应 BAMOR_engineering_doc §2.1
    """
    def __init__(self, state_dim, action_dim, hidden_dim, num_critics,
                 num_objectives, embedding_layer):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.num_critics = num_critics
        self.num_objectives = num_objectives
        
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, num_objectives, num_critics),  # ← m 而非 1
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)
        q_values = self.critic(state_action)  # [ensemble, batch, m]
        # 不 squeeze! 保留多目标维度
        return q_values


class SoftQNetwork(VectorizedCritic):
    """
    多目标 Soft Q-Network。
    
    输出: [ensemble_size, batch, num_objectives]
    对应 BAMOR.lean: Q : S × A → Fin m → ℝ
    """
    def __init__(self, state_dim, action_dim, hidden_dim, embedding_layer,
                 num_objectives=2, ensemble_size=10):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_critics=ensemble_size,
            num_objectives=num_objectives,
            embedding_layer=embedding_layer
        )
        self.ensemble_size = ensemble_size
        self.num_objectives = num_objectives

    def forward(self, state, action):
        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]
        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)
        return super().forward(state_with_embeddings, action)


class PolicyNetwork(nn.Module):
    """
    w-Conditioned Policy Network (CAPQL 风格)。
    
    输入: [obs, w] → 输出: action
    对应 BAMOR_engineering_doc §4.3
    
    关键设计: Policy 知道偏好权重 w，这样不同的 w 会产生不同的 action，
    使 Pareto 前沿能够展开。
    """
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer,
                 num_objectives=2, action_range=1., init_w=3e-3,
                 log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.embedding_layer = embedding_layer
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_objectives = num_objectives

        # 输入维度: state_dim + m (w-conditioned)
        input_dim = num_inputs + num_objectives
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state, w):
        """
        Args:
            state: [batch, state_dim] (已含 embedding)
            w: [m] 或 [batch, m] — 偏好权重
        """
        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]
        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)

        # 拼接 w
        if w.dim() == 1:
            w_expanded = w.unsqueeze(0).expand(state.shape[0], -1)  # [batch, m]
        else:
            w_expanded = w
        x = torch.cat([state_with_embeddings, w_expanded], dim=-1)  # [batch, state_dim + m]

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, w, epsilon=1e-6):
        """
        Generate sampled action with reparameterization trick.
        
        Args:
            state: [batch, state_dim]
            w: [m] — 偏好权重
        """
        device = state.device
        mean, log_std = self.forward(state, w)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = torch.tanh(mean + std * z.to(device))
        action = self.action_range / 2 * action_0 + self.action_range / 2

        log_prob = (Normal(mean, std).log_prob(mean + std * z.to(device))
                    - torch.log(1. - action_0.pow(2) + epsilon)
                    - np.log(self.action_range))
        log_prob = log_prob.sum(dim=1)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, w, deterministic=False):
        """
        Args:
            state: [state_dim] (单样本, unbatched)
            w: [m] tensor — 偏好权重
        """
        device = next(self.parameters()).device
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if not isinstance(w, torch.Tensor):
            w = torch.FloatTensor(w).to(device)
        
        mean, log_std = self.forward(state, w)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action = self.action_range / 2 * torch.tanh(mean + std * z) + self.action_range / 2

        if deterministic:
            action = (self.action_range / 2 * torch.tanh(mean).detach().cpu().numpy()[0]
                      + self.action_range / 2)
        else:
            action = action.detach().cpu().numpy()[0]
        return action
