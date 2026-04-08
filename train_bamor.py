"""
BAMOR Training Script
Bayesian Amnesic Multi-Objective Robust SAC

基于 BAPR sac_ensemble_bapr.py 改造:
- 多目标奖励 reward_vec: [m]
- w-conditioned 策略
- Mahalanobis surprise
- 拆分模块: networks.py, trainer.py, replay_buffer.py, bamor_components.py
"""

import psutil
import gc
import time
import torch
import os
import argparse
import json
import numpy as np
import random
from IPython.display import clear_output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env.sim import env_bus
from bapr_components import BeliefTracker
from bamor_components import MahalanobisSurpriseComputer, ContextAwareWeightSampler
from mode_profiles import MODE_PROFILES, MODE_CONE_PROFILES, get_cone_mixing_weights
from replay_buffer import ReplayBuffer
from trainer import SAC_Trainer

# ─── Device ───
GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

# ─── Args ───
parser = argparse.ArgumentParser(description='BAMOR: Bayesian Amnesic Multi-Objective Robust SAC')
parser.add_argument('--max_episodes', type=int, default=500)
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--use_gradient_clip', type=bool, default=True)
parser.add_argument("--use_state_norm", type=bool, default=False)
parser.add_argument("--use_reward_norm", type=bool, default=False)
parser.add_argument("--use_reward_scaling", type=bool, default=False)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--training_freq", type=int, default=5)
parser.add_argument("--plot_freq", type=int, default=1)
parser.add_argument('--weight_reg', type=float, default=0.01)
parser.add_argument('--auto_entropy', type=bool, default=True)
parser.add_argument("--maximum_alpha", type=float, default=0.6)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--beta_bc", type=float, default=0.001)
parser.add_argument("--beta", type=float, default=-2)
parser.add_argument("--beta_ood", type=float, default=0.01)
parser.add_argument('--critic_actor_ratio', type=int, default=2)
parser.add_argument('--replay_buffer_size', type=int, default=int(1e6))
parser.add_argument('--route_sigma', type=float, default=1.5)
parser.add_argument('--eval_sigmas', type=float, nargs='*', default=None)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--save_root', type=str, default='bamor_experiments')
parser.add_argument('--run_name', type=str, default=None)
parser.add_argument("--ensemble_size", type=int, default=10)

# ===== BA-PR 参数 =====
parser.add_argument('--enable_mode_switch', action='store_true', default=False)
parser.add_argument('--mode_switch_min', type=int, default=1800)
parser.add_argument('--mode_switch_max', type=int, default=7200)
parser.add_argument('--max_run_length', type=int, default=20)
parser.add_argument('--hazard_rate', type=float, default=0.02)
parser.add_argument('--surprise_ema_alpha', type=float, default=0.3)
parser.add_argument('--penalty_decay_rate', type=float, default=0.1)
parser.add_argument('--belief_warmup_steps', type=int, default=50)

# ===== BAMOR 新增参数 =====
parser.add_argument('--num_objectives', type=int, default=2, help='Number of objectives (m)')
parser.add_argument('--preference_weights', type=float, nargs='+', default=[0.5, 0.5],
                    help='Pareto preference weights (must sum to 1)')
parser.add_argument('--eta_anchor', type=float, default=0.01, help='Anchor regularization strength')
parser.add_argument('--enable_target_cone', type=bool, default=True, help='Enable ContextAware Target Cone sampling')

args = parser.parse_args()

# ─── 验证偏好权重 ───
w = torch.tensor(args.preference_weights, dtype=torch.float32, device=device)
assert len(args.preference_weights) == args.num_objectives, \
    f"preference_weights 长度 ({len(args.preference_weights)}) 必须等于 num_objectives ({args.num_objectives})"
assert w.ge(0).all(), "preference_weights must be non-negative"
assert abs(w.sum().item() - 1.0) < 1e-6, f"preference_weights must sum to 1, got {w.sum().item()}"
print(f"BAMOR: m={args.num_objectives}, w={args.preference_weights}")

ROUTE_SIGMA_TOKEN = f"route_sigma_{str(args.route_sigma).replace('.', 'p')}"


def safe_mean(arr):
    return np.mean(arr) if len(arr) > 0 else 0.0


def plot(rewards, pic_path, step_logs_episode):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Reward (scalarized)")
    plt.legend()
    plt.title(f"BAMOR Training (w={args.preference_weights}, weight_reg={args.weight_reg})")
    
    plt.subplot(1, 2, 2)
    for key in ['q_values', 'ood_losses', 'alpha_values', 'q_stds']:
        if key in step_logs_episode and len(step_logs_episode[key]) > 0:
            plt.plot(step_logs_episode[key], label=key)
    plt.legend()
    plt.title("Training Metrics")

    os.makedirs(pic_path, exist_ok=True)
    plt.savefig(os.path.join(pic_path, 'bamor_monitoring.png'))
    plt.close()


# ─── Environment ───
debug = False
render = False
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'env')
env = env_bus(path, debug=debug, route_sigma=args.route_sigma,
              enable_mode_switch=args.enable_mode_switch,
              mode_profiles=MODE_PROFILES,
              mode_switch_interval=(args.mode_switch_min, args.mode_switch_max))
env.reset()

action_dim = env.action_space.shape[0]
action_range = env.action_space.high[0]

# ─── BAMOR: 初始化 belief + Mahalanobis surprise ───
belief_tracker = BeliefTracker(
    max_run_length=args.max_run_length,
    hazard_rate=args.hazard_rate
) if args.enable_mode_switch else None

mahal_computer = MahalanobisSurpriseComputer(
    num_objectives=args.num_objectives,
    ema_alpha=args.surprise_ema_alpha
) if args.enable_mode_switch else None

context_sampler = ContextAwareWeightSampler(
    mode_profiles=MODE_CONE_PROFILES,
    get_mixing_weights_fn=get_cone_mixing_weights
) if args.enable_target_cone else None

# ─── Replay Buffer + Trainer ───
replay_buffer = ReplayBuffer(args.replay_buffer_size)

sac_trainer = SAC_Trainer(
    env, replay_buffer,
    hidden_dim=args.hidden_dim,
    action_range=action_range,
    num_objectives=args.num_objectives,
    ensemble_size=args.ensemble_size,
    args=args,
    device=device
)

# ─── Logging ───
step = 0
step_trained = 0
update_itr = 1
DETERMINISTIC = False

rewards = []  # 每 episode 的 scalarized reward

# Per-step logs (每 episode 清空)
def make_step_logs():
    return {
        'q_values': [], 'log_probs': [], 'alpha_values': [],
        'ood_losses': [], 'q_stds': [],
        'belief_entropies': [], 'effective_windows': [],
        'surprise_emas': [], 'weighted_lambdas': [],
        'per_obj_surprises': [], 'target_cone_ws': [], 'w_anchors': [],
    }

# Per-episode aggregated logs
step_logs_episode = {k: [] for k in make_step_logs().keys()}
mode_switches_episode = []

# ─── Paths ───
param_str = (
    f"{ROUTE_SIGMA_TOKEN}/"
    f"m{args.num_objectives}_w{'_'.join(str(x) for x in args.preference_weights)}/"
    f"replay_buffer_size_{args.replay_buffer_size}/"
    f"weight_reg_{args.weight_reg}"
)

if args.run_name:
    model_path = os.path.join(args.save_root, 'model', args.run_name)
    logs_path = os.path.join(args.save_root, 'logs', args.run_name)
    pic_path = os.path.join(args.save_root, 'pic', args.run_name)
else:
    model_path = os.path.join(args.save_root, 'model', param_str)
    logs_path = os.path.join(args.save_root, 'logs', param_str)
    pic_path = os.path.join(args.save_root, 'pic', param_str)

os.makedirs(model_path, exist_ok=True)
os.makedirs(logs_path, exist_ok=True)
os.makedirs(pic_path, exist_ok=True)


if __name__ == '__main__':
    if args.train:
        for eps in range(args.max_episodes):
            episode_start_time = time.time()
            if eps != 0:
                env.reset()

            # ─── Reset belief + surprise ───
            if belief_tracker is not None:
                belief_tracker.reset()
            if mahal_computer is not None:
                mahal_computer.reset()

            state_dict, reward_dict, _ = env.initialize_state(render=render)
            done = False
            episode_steps = 0
            training_steps = 0
            action_dict = {key: None for key in list(range(env.max_agent_num))}
            episode_reward_scalar = 0
            episode_reward_vec = np.zeros(args.num_objectives)

            step_logs = make_step_logs()

            while not done:
                # ─── BAMOR Target Cone Sampling (Part XI) ───
                if args.enable_target_cone and belief_tracker is not None:
                    w_restricted, w_anchor = context_sampler.sample_and_get_anchor(
                        belief_tracker.belief, device=device)
                    if step_logs is not None:
                        step_logs['target_cone_ws'].append(w_restricted[0].item())
                        step_logs['w_anchors'].append(w_anchor[0].item())
                else:
                    w_restricted = w  # fallback
                    w_anchor = w      # no anchor shift penalty

                for key in state_dict:
                    if len(state_dict[key]) == 1:
                        if action_dict[key] is None:
                            if args.use_state_norm:
                                state_input = sac_trainer.state_norm(np.array(state_dict[key][0]))
                            else:
                                state_input = np.array(state_dict[key][0])
                            a = sac_trainer.policy_net.get_action(
                                torch.from_numpy(state_input).float(),
                                w_restricted, deterministic=DETERMINISTIC)
                            action_dict[key] = a

                    elif len(state_dict[key]) == 2:
                        if state_dict[key][0][1] != state_dict[key][1][1]:
                            if args.use_state_norm:
                                state = sac_trainer.state_norm(np.array(state_dict[key][0]))
                                next_state = sac_trainer.state_norm(np.array(state_dict[key][1]))
                            else:
                                state = np.array(state_dict[key][0])
                                next_state = np.array(state_dict[key][1])

                            # ─── 多目标 reward 组装 ───
                            # 目前 env 返回标量 reward, 我们构造向量
                            # reward_dict[key] 是标量 (headway_reward)
                            # energy_reward = -holding_time (action)
                            raw_reward = reward_dict[key]
                            holding_action = action_dict[key] if action_dict[key] is not None else 0.0
                            if isinstance(holding_action, np.ndarray):
                                holding_action = holding_action[0]
                            reward_vec = np.array([
                                raw_reward,              # 目标0: headway
                                -float(holding_action)   # 目标1: energy (负 holding time)
                            ], dtype=np.float32)

                            replay_buffer.push(state, action_dict[key], reward_vec, next_state, done)

                            episode_steps += 1
                            step += 1
                            episode_reward_vec += reward_vec
                            episode_reward_scalar += float((w.cpu().numpy() * reward_vec).sum())

                        state_dict[key] = state_dict[key][1:]
                        if args.use_state_norm:
                            state_input = sac_trainer.state_norm(np.array(state_dict[key][0]))
                        else:
                            state_input = np.array(state_dict[key][0])
                        action_dict[key] = sac_trainer.policy_net.get_action(
                            torch.from_numpy(state_input).float(),
                            w_restricted, deterministic=DETERMINISTIC)

                state_dict, reward_dict, done = env.step(action_dict, debug=debug, render=render)
                if (len(replay_buffer) > args.batch_size
                        and len(replay_buffer) % args.training_freq == 0
                        and step_trained != step):
                    step_trained = step
                    for i in range(update_itr):
                        _ = sac_trainer.update(
                            args.batch_size, training_steps, w_restricted, w_anchor=w_anchor,
                            reward_scale=10., auto_entropy=args.auto_entropy,
                            target_entropy=-1. * action_dim,
                            belief_tracker=belief_tracker,
                            mahal_computer=mahal_computer,
                            episode_step=episode_steps,
                            step_logs=step_logs)
                        training_steps += 1

                if done:
                    replay_buffer.last_episode_step = episode_steps
                    break

            # ─── Episode 日志聚合 ───
            rewards.append(episode_reward_scalar)
            for key in step_logs:
                if len(step_logs[key]) > 0:
                    step_logs_episode[key].append(safe_mean(step_logs[key]))
                else:
                    step_logs_episode[key].append(0)

            mode_switches_episode.append(getattr(env, 'mode_switch_count', 0))

            # ─── Plot + Save ───
            if eps % args.plot_freq == 0:
                plot(rewards, pic_path, step_logs_episode)

                np.save(os.path.join(logs_path, 'rewards.npy'), rewards)
                for key in step_logs_episode:
                    np.save(os.path.join(logs_path, f'{key}_episode.npy'), step_logs_episode[key])
                np.save(os.path.join(logs_path, 'mode_switches_episode.npy'), mode_switches_episode)

                sac_trainer.save_model(os.path.join(model_path, f"checkpoint_episode_{eps}"))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            replay_buffer_usage = len(replay_buffer) / args.replay_buffer_size * 100
            episode_duration = time.time() - episode_start_time

            bamor_info = ""
            if args.enable_mode_switch:
                bamor_info = (
                    f" | Mode: {env.current_mode_name} | Switches: {env.mode_switch_count}"
                    f" | Belief H: {step_logs_episode['belief_entropies'][-1]:.3f}"
                    f" | λ_w: {step_logs_episode['weighted_lambdas'][-1]:.4f}"
                )
            print(
                f"Episode: {eps} | Reward: {episode_reward_scalar:.1f} "
                f"| r_vec: [{episode_reward_vec[0]:.1f}, {episode_reward_vec[1]:.1f}] "
                f"| Duration: {episode_duration:.2f}s "
                f"| CPU: {psutil.Process().memory_info().rss / 1024 ** 2:.0f}MB "
                f"| Buffer: {replay_buffer_usage:.1f}%{bamor_info}")

        sac_trainer.save_model(os.path.join(model_path, "final"))

    if args.test:
        sac_trainer.load_model(os.path.join(model_path, "final"))
        for eps in range(10):
            done = False
            env.reset()
            state_dict, reward_dict, _ = env.initialize_state(render=render)
            episode_reward = 0
            action_dict = {key: None for key in list(range(env.max_agent_num))}

            while not done:
                for key in state_dict:
                    if len(state_dict[key]) == 1:
                        if action_dict[key] is None:
                            state_input = np.array(state_dict[key][0])
                            a = sac_trainer.policy_net.get_action(
                                torch.from_numpy(state_input).float(),
                                w, deterministic=True)
                            action_dict[key] = a
                    elif len(state_dict[key]) == 2:
                        if state_dict[key][0][1] != state_dict[key][1][1]:
                            episode_reward += reward_dict[key]
                        state_dict[key] = state_dict[key][1:]
                        state_input = np.array(state_dict[key][0])
                        action_dict[key] = sac_trainer.policy_net.get_action(
                            torch.from_numpy(state_input).float(),
                            w, deterministic=True)
                state_dict, reward_dict, done = env.step(action_dict)
            print(f'Episode: {eps} | Episode Reward: {episode_reward}')
