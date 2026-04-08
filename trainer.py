"""
BAMOR SAC Trainer

对应 BAMOR_engineering_doc:
- §2.2: compute_q_loss_bamor — 多目标 Q-loss with belief-weighted penalty
- §2.4: compute_reg_norm_per_objective — per-objective aleatoric penalty
- §4.3: compute_policy_loss_bamor — scalarized Pareto policy loss
- §5.3: update() — Mahalanobis surprise → belief update

对应 BAMOR.lean:
- Part I (L46-331):  T_mode_mo γ-contraction (per-component)
- Part VII (L703-749): scalarized_bamor_contraction
- Part X (L922-1017): bamor_robust_tracking (time-varying operators)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy

from networks import EmbeddingLayer, SoftQNetwork, PolicyNetwork
from normalization import Normalization, RewardScaling, RunningMeanStd


class SAC_Trainer:
    def __init__(self, env, replay_buffer, hidden_dim, action_range,
                 num_objectives=2, ensemble_size=10, args=None, device='cpu'):
        self.args = args
        self.device = device
        self.num_objectives = num_objectives
        self.replay_buffer = replay_buffer

        # ─── 类别特征嵌入 (复用 BAPR 设计) ───
        cat_cols = ['bus_id', 'station_id', 'time_period', 'direction']
        cat_code_dict = {
            'bus_id': {i: i for i in range(env.max_agent_num)},
            'station_id': {i: i for i in range(round(len(env.stations) / 2))},
            'time_period': {i: i for i in range(env.timetables[-1].launch_time // 3600 + 2)},
            'direction': {0: 0, 1: 1}
        }
        self.num_cat_features = len(cat_cols)
        self.num_cont_features = env.state_dim - self.num_cat_features
        embedding_layer = EmbeddingLayer(cat_code_dict, cat_cols)

        embedding_dim = sum([min(50, len(cat_code_dict[col]) // 2) for col in cat_cols])
        state_dim = embedding_dim + self.num_cont_features
        action_dim = env.action_space.shape[0]

        # ─── 网络初始化 ───
        self.soft_q_net = SoftQNetwork(
            state_dim, action_dim, hidden_dim, embedding_layer,
            num_objectives=num_objectives, ensemble_size=ensemble_size
        ).to(device)
        self.target_soft_q_net = deepcopy(self.soft_q_net)

        self.policy_net = PolicyNetwork(
            state_dim, action_dim, hidden_dim, embedding_layer,
            num_objectives=num_objectives, action_range=action_range
        ).to(device)

        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        self.alpha = 0.0

        print('Soft Q Network: ', self.soft_q_net)
        print('Policy Network: ', self.policy_net)

        # ─── 优化器 ───
        soft_q_lr = 1e-5
        policy_lr = 1e-5
        alpha_lr = 1e-5
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # ─── State normalization ───
        initial_mean = [360., 360., 90.]
        initial_std = [165., 133., 45.]
        running_ms = RunningMeanStd(
            shape=(self.num_cont_features,),
            init_mean=initial_mean, init_std=initial_std
        )
        self.state_norm = Normalization(
            num_categorical=self.num_cat_features,
            num_numerical=self.num_cont_features,
            running_ms=running_ms
        )
        self.reward_scaling = RewardScaling(shape=1, gamma=0.99)

    # ─── Per-Objective Reg Norm ─── (对应 BAMOR_engineering_doc §2.4)
    def compute_reg_norm_per_objective(self, model):
        """
        返回 [ensemble_size, num_objectives] 的 per-objective 正则化范数。
        对应 BAMOR.lean: κ_ale(i)
        
        设计: 只有输出层按目标分, 隐层权重按 ensemble 聚合后广播。
        """
        hidden_norm = []
        output_norm = []

        for name, param in model.named_parameters():
            if 'critic' not in name:
                continue
            # 输出层: 最后一个 VectorizedLinear, weight shape [ensemble, hidden, m]
            if name.endswith('6.weight') or name.endswith('6.bias'):
                if 'weight' in name:
                    output_norm.append(torch.norm(param, p=1, dim=1))  # → [ensemble, m]
                elif 'bias' in name:
                    output_norm.append(param.abs().squeeze(1))         # → [ensemble, m]
            else:
                if 'weight' in name:
                    hidden_norm.append(torch.norm(param, p=1, dim=[1, 2]))  # → [ensemble]
                elif 'bias' in name:
                    hidden_norm.append(torch.norm(param, p=1, dim=[1, 2]))  # → [ensemble]

        # 隐层: [ensemble] → [ensemble, m]
        if hidden_norm:
            h_total = torch.sum(torch.stack(hidden_norm), dim=0)
            h_expanded = h_total.unsqueeze(-1).expand(-1, self.num_objectives)
        else:
            h_expanded = torch.zeros(model.ensemble_size, self.num_objectives,
                                     device=next(model.parameters()).device)

        # 输出层: [ensemble, m]
        if output_norm:
            o_total = torch.sum(torch.stack(output_norm), dim=0)
        else:
            o_total = torch.zeros_like(h_expanded)

        return h_expanded + o_total  # [ensemble, m]

    # ─── BAPR 兼容: 标量 reg_norm (用于不需要 per-obj 的地方) ───
    def compute_reg_norm(self, model):
        """原始 BAPR 标量 reg_norm, 返回 [ensemble]"""
        weight_norm, bias_norm = [], []
        for name, param in model.named_parameters():
            if 'critic' in name:
                if 'weight' in name:
                    weight_norm.append(torch.norm(param, p=1, dim=[1, 2]))
                elif 'bias' in name:
                    bias_norm.append(torch.norm(param, p=1, dim=[1, 2]))
        reg_norm = (torch.sum(torch.stack(weight_norm), dim=0)
                    + torch.sum(torch.stack(bias_norm[:-1]), dim=0))
        return reg_norm

    # ─── Q Loss ─── (对应 BAMOR.lean Part I + Part VII)
    def compute_q_loss_bamor(self, state, action, reward_vec, next_state, done,
                             new_next_action, next_log_prob, reg_norm_vec, gamma,
                             belief_tracker, w):
        """
        多目标 Q-loss with belief-weighted epistemic penalty.
        
        对应 BAMOR.lean: T_BAMOR = Σ ρ(h) · T_h^MO(Q)
        
        Args:
            state: [batch, state_dim]
            action: [batch, 1]
            reward_vec: [batch, m]
            reg_norm_vec: [ensemble, m]
            w: [m] — 偏好权重
        Returns:
            loss, predicted_q, ood_loss, weighted_lambda
        """
        args = self.args

        # Q predictions: [ensemble, batch, m]
        predicted_q_value = self.soft_q_net(state, action)
        target_q_next = self.target_soft_q_net(next_state, new_next_action)

        # next_log_prob: [batch] → [ensemble, batch, 1] 广播到每个目标
        next_log_prob_expanded = next_log_prob.unsqueeze(0).unsqueeze(-1).repeat(
            self.soft_q_net.ensemble_size, 1, 1)

        # reg_norm_vec: [ensemble, m] → [ensemble, 1, m] 广播到 batch
        reg_norm_expanded = reg_norm_vec.unsqueeze(1).repeat(1, state.shape[0], 1)

        # ─── Belief-weighted epistemic penalty (per-objective) ───
        # epi_penalty: ensemble std per objective → [batch, m]
        epi_penalty = target_q_next.std(dim=0)

        belief = torch.tensor(belief_tracker.belief, dtype=torch.float32, device=self.device)
        penalty_schedule = torch.exp(
            -args.penalty_decay_rate
            * torch.arange(belief_tracker.max_H, dtype=torch.float32, device=self.device)
        )
        weighted_lambda = (belief * penalty_schedule).sum()  # 标量

        # Target Q: [ensemble, batch, m]
        target_q_next = (target_q_next
                         - self.alpha * next_log_prob_expanded
                         + args.weight_reg * reg_norm_expanded
                         - weighted_lambda * args.weight_reg
                           * epi_penalty.unsqueeze(0).repeat(self.soft_q_net.ensemble_size, 1, 1))

        # reward_vec: [batch, m] → [1, batch, m]
        reward_expanded = reward_vec.unsqueeze(0)
        done_expanded = done.unsqueeze(0).unsqueeze(-1)
        target_q_value = reward_expanded + (1 - done_expanded) * gamma * target_q_next

        # Loss: MSE per component, sum across objectives
        # [ensemble, batch, m] → scalar
        q_value_loss = nn.functional.mse_loss(predicted_q_value, target_q_value.detach())
        
        # OOD loss: ensemble disagreement (scalarized for single scalar)
        # Scalarize first, then take std → [batch]
        q_scalarized = (predicted_q_value * w.view(1, 1, -1)).sum(dim=-1)  # [ensemble, batch]
        ood_loss = q_scalarized.std(dim=0).mean()
        
        loss = q_value_loss + args.beta_ood * ood_loss

        return loss, predicted_q_value, ood_loss, weighted_lambda.item()

    # ─── Policy Loss ─── (对应 BAMOR_engineering_doc §4.3 & §4.6.4)
    def compute_policy_loss_bamor(self, state, action, new_action, log_prob, reg_norm_vec, w, w_anchor=None):
        """
        Pareto-scalarized policy loss: max_π Σ w_i Q_i(s, π(s,w)) − α·log π + Anchor Penalty
        
        对应 BAMOR.lean: scalarize Q w (s,a) = Σ w_i Q(s,a)(i)
        """
        args = self.args

        # Q values: [ensemble, batch, m]
        q_values = self.soft_q_net(state, new_action)

        # Scalarize: [ensemble, batch, m] → [ensemble, batch]
        q_scalarized = (q_values * w.view(1, 1, -1)).sum(dim=-1)

        # reg_norm (scalar per ensemble for policy): [ensemble] → [ensemble, batch]
        reg_norm_scalar = reg_norm_vec.mean(dim=-1)  # [ensemble] — 取目标均值
        reg_norm_expanded = reg_norm_scalar.unsqueeze(-1).repeat(1, state.shape[0])

        q_values_dist = q_scalarized + args.weight_reg * reg_norm_expanded - self.alpha * log_prob

        # BAMOR Part XI: Anchor-Shifted Regularization (Strong Concavity Preservation)
        # 惩罚偏离物理合法锚点 w_anchor 的行为
        if w_anchor is not None and getattr(args, 'eta_anchor', 0.0) > 0:
            anchor_penalty = args.eta_anchor * ((w - w_anchor) ** 2).sum()
            # q_values_dist 是我们在最大化的目标，因此正则化是扣分项(-)
            q_values_dist = q_values_dist - anchor_penalty

        q_mean = q_values_dist.mean(dim=0)
        q_std = q_values_dist.std(dim=0)
        q_loss = -(q_mean + args.beta * q_std).mean()

        bc_loss = nn.functional.mse_loss(new_action, action)
        loss = args.beta_bc * bc_loss + q_loss

        return loss, q_loss, q_std

    # ─── Alpha Loss ───
    def compute_alpha_loss(self, log_prob, target_entropy):
        alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
        return alpha_loss

    # ─── Update ─── (对应 BAMOR_engineering_doc §11)
    def update(self, batch_size, training_steps, w, w_anchor=None, reward_scale=10.,
               auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2,
               belief_tracker=None, mahal_computer=None, episode_step=0,
               step_logs=None):
        """
        BAMOR training step.
        
        与 BAPR update() 的关键区别:
        1. reward_vec [batch, m] 而非标量
        2. compute_reg_norm_per_objective → [ensemble, m]
        3. MahalanobisSurpriseComputer 替代 SurpriseComputer
        4. policy_net.evaluate(state, w) — w-conditioned
        5. compute_q_loss_bamor / compute_policy_loss_bamor
        """
        args = self.args
        device = self.device

        state, action, reward_vec, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward_vec = torch.FloatTensor(reward_vec).to(device)    # [batch, m]
        done = torch.FloatTensor(np.float32(done)).to(device)    # [batch]

        # ─── Policy 采样 (w-conditioned) ───
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state, w)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state, w)

        # ─── Per-objective reward normalization ───
        reward_std = reward_vec.std(dim=0)
        reward_std = torch.where(reward_std < 1e-6,
                                 torch.tensor(1.0, device=device), reward_std)
        reward_vec = reward_scale * (reward_vec - reward_vec.mean(dim=0)) / reward_std

        # ─── Alpha update ───
        action_dim = action.shape[-1]
        if auto_entropy:
            alpha_loss = self.compute_alpha_loss(log_prob, -1. * action_dim)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward(retain_graph=False)
            self.alpha_optimizer.step()
            self.alpha = min(args.maximum_alpha, self.log_alpha.exp().item())
        else:
            self.alpha = 1.
            alpha_loss = 0

        # ─── Mahalanobis surprise → belief update ───
        reg_norm_vec = self.compute_reg_norm_per_objective(self.target_soft_q_net)
        current_reg_norm_vec = self.compute_reg_norm_per_objective(self.soft_q_net)

        if belief_tracker is not None and mahal_computer is not None:
            mahal_surprise, per_obj_surprise = mahal_computer.compute(
                current_reg_norm_vec, reg_norm_vec)
            if episode_step >= args.belief_warmup_steps:
                belief_tracker.update(mahal_surprise)

            # BAMOR Q-loss
            q_value_loss, predicted_q_value, ood_loss, w_lambda = self.compute_q_loss_bamor(
                state, action, reward_vec, next_state, done,
                new_next_action, next_log_prob, reg_norm_vec, gamma,
                belief_tracker, w)
            
            # Log BAMOR metrics
            if step_logs is not None:
                step_logs['belief_entropies'].append(belief_tracker.entropy)
                step_logs['effective_windows'].append(belief_tracker.effective_window)
                step_logs['surprise_emas'].append(mahal_surprise)
                step_logs['weighted_lambdas'].append(w_lambda)
                step_logs['per_obj_surprises'].append(per_obj_surprise.tolist())
        else:
            # 无 belief 模式不应在 BAMOR 中使用, 但保留以备测试
            q_value_loss, predicted_q_value, ood_loss, w_lambda = self.compute_q_loss_bamor(
                state, action, reward_vec, next_state, done,
                new_next_action, next_log_prob, reg_norm_vec, gamma,
                belief_tracker, w)

        # ─── Q network update ───
        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward(retain_graph=False)
        if args.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.soft_q_net.parameters(), max_norm=1.0)
        self.soft_q_optimizer.step()

        # ─── Policy update (delayed) ───
        if training_steps % args.critic_actor_ratio == 0:
            policy_loss, predicted_new_q_value, q_std = self.compute_policy_loss_bamor(
                state, action, new_action, log_prob, reg_norm_vec, w, w_anchor=w_anchor)
            if step_logs is not None:
                step_logs['q_stds'].append(q_std.mean().item())
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=False)
            self.policy_optimizer.step()

        # ─── Target network soft update ───
        for target_param, param in zip(self.target_soft_q_net.parameters(),
                                       self.soft_q_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

        # ─── Logging ───
        if step_logs is not None:
            # Scalarize Q for logging
            q_scalar = (predicted_q_value * w.view(1, 1, -1)).sum(dim=-1)
            step_logs['q_values'].append(q_scalar.mean().item())
            step_logs['log_probs'].append(-log_prob.mean().item())
            step_logs['alpha_values'].append(self.alpha)
            step_logs['ood_losses'].append(ood_loss.item())

        return predicted_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net.state_dict(), path + '_q')
        torch.save(self.policy_net.state_dict(), path + '_policy')
        torch.save(self.state_norm, path + '_norm')

    def load_model(self, path):
        self.soft_q_net.load_state_dict(torch.load(path + '_q', weights_only=True))
        self.policy_net.load_state_dict(torch.load(path + '_policy', weights_only=True))
        self.soft_q_net.eval()
        self.policy_net.eval()
