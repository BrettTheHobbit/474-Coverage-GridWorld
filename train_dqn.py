"""
Coverage GridWorld — curriculum training with DQN (Deep Q-Network).

Same staged curriculum, mixed-env replay, and most CLI flags as ``train_ppo.py``, but **off-policy**
DQN (Stable-Baselines3). Checkpoints are ``dqn_stage*.zip`` and **do not** load into PPO (different
network and SB3 save format).

Install the env package from the repo root, then run:

.. code-block:: text

   pip install -e ./coverage-gridworld
   python train_dqn.py

CLI examples
------------
::

  python train_dqn.py                          # resume from dqn_stage* checkpoints
  python train_dqn.py --fresh                  # every stage from scratch
  python train_dqn.py --fresh-from 4           # keep 1–3, retrain 4+
  python train_dqn.py --fresh-stages 4 5       # retrain only those stages
  python train_dqn.py --load dqn_stage4_chokepoint --fresh-from 4

One stage from best eval, without re-running stage 1::

  python train_dqn.py --only-stage 2 --fresh-stages 2 \\
      --load dqn_eval_best/dqn_stage2_safe/best_model

Maze with replay of ``just_go`` only (default mix is 65% / 35%)::

  python train_dqn.py --fresh-from 3 --replay-envs just_go
  # 50/50: add --primary-weight 0.5
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Optional

import coverage_gridworld
import gymnasium
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


class DetailedEvalCallback(BaseCallback):
    """
    Greedy eval: logs eval/mean_reward, eval/mean_ep_length, eval/mean_coverage,
    eval/success_rate to TensorBoard. Saves best model by mean eval reward (like EvalCallback).
    """

    def __init__(
        self,
        eval_env: gymnasium.Env,
        eval_freq: int,
        n_eval_episodes: int = 15,
        deterministic: bool = True,
        best_model_save_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = deterministic
        self.best_model_save_path = best_model_save_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        rewards, lengths, covs, succs = [], [], [], []
        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            ep_rew = 0.0
            ep_len = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, r, terminated, truncated, info = self.eval_env.step(int(action))
                ep_rew += float(r)
                ep_len += 1
            rewards.append(ep_rew)
            lengths.append(ep_len)
            cc = info.get("coverable_cells", 1)
            covs.append(info.get("total_covered_cells", 0) / max(cc, 1))
            succs.append(1.0 if info.get("cells_remaining", 1) == 0 else 0.0)

        mr = float(np.mean(rewards))
        ml = float(np.mean(lengths))
        mc = float(np.mean(covs))
        ss = float(np.mean(succs))
        self.logger.record("eval/mean_reward", mr)
        self.logger.record("eval/mean_ep_length", ml)
        self.logger.record("eval/mean_coverage", mc)
        self.logger.record("eval/success_rate", ss)

        if self.verbose:
            print(
                f"\nEval @ {self.n_calls} ts: reward={mr:.1f} len={ml:.1f} "
                f"cov={mc:.3f} succ={ss:.2f}\n"
            )

        if self.best_model_save_path is not None and mr > self.best_mean_reward:
            self.best_mean_reward = mr
            os.makedirs(self.best_model_save_path, exist_ok=True)
            path = os.path.join(self.best_model_save_path, "best_model")
            self.model.save(path)
            if self.verbose:
                print(f"  New best eval mean_reward={mr:.2f} -> saved {path}.zip\n")
        return True


class EpisodeStatsCallback(BaseCallback):
    """
    Callback to monitor detailed episode statistics during training.
    If *primary_env_id* is set (mixed-env training), only episodes from that
    environment are included in the printed stats.
    """
    def __init__(self, check_freq=2048, verbose=1, primary_env_id=None):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.primary_env_id = primary_env_id
        self.episode_count = 0
        self.replay_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_coverages = []
        self.episode_successes = []
        self.current_ep_reward = 0
        self.current_ep_length = 0

    def _on_step(self) -> bool:
        if len(self.locals.get('rewards', [])) > 0:
            self.current_ep_reward += self.locals['rewards'][0]
            self.current_ep_length += 1

        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            info = {}
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]

            ep_env_id = info.get("_env_id", None)
            is_primary = (self.primary_env_id is None
                          or ep_env_id is None
                          or ep_env_id == self.primary_env_id)

            if is_primary:
                self.episode_count += 1
                self.episode_rewards.append(self.current_ep_reward)
                self.episode_lengths.append(max(self.current_ep_length, 1))
                coverage = info.get('total_covered_cells', 0) / max(info.get('coverable_cells', 1), 1)
                success = 1 if info.get('cells_remaining', 1) == 0 else 0
                self.episode_coverages.append(coverage)
                self.episode_successes.append(success)
            else:
                self.replay_count += 1

            self.current_ep_reward = 0
            self.current_ep_length = 0

        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-50:] if len(self.episode_rewards) >= 50 else self.episode_rewards
            recent_lengths = self.episode_lengths[-50:] if len(self.episode_lengths) >= 50 else self.episode_lengths
            recent_coverage = self.episode_coverages[-50:] if len(self.episode_coverages) >= 50 else self.episode_coverages
            recent_success = self.episode_successes[-50:] if len(self.episode_successes) >= 50 else self.episode_successes
            best_ever = max(self.episode_coverages)
            win = min(500, len(self.episode_coverages))
            best_last_win = max(self.episode_coverages[-win:])

            r_per_step = [r / float(l) for r, l in zip(recent_rewards, recent_lengths)]
            corr = float(np.corrcoef(recent_rewards, recent_coverage)[0, 1]) if len(recent_rewards) >= 3 else float("nan")

            env_label = f" [{self.primary_env_id}]" if self.primary_env_id else ""
            print(f"\n{'='*70}")
            print(f"DQN training — Step {self.n_calls}{env_label}")
            print(f"{'='*70}")
            print(f"Episodes completed: {self.episode_count}  (replay: {self.replay_count})")
            print(f"Avg Reward (last 50): {np.mean(recent_rewards):.2f}")
            print(f"Avg episode length (last 50): {np.mean(recent_lengths):.1f}")
            print(f"Avg Reward / step (last 50): {np.mean(r_per_step):.3f}")
            print(f"Avg Coverage (last 50): {np.mean(recent_coverage):.3f}")
            print(f"Median Coverage (last 50): {float(np.median(recent_coverage)):.3f}")
            print(f"Success Rate (last 50): {np.mean(recent_success):.2f}")
            print(f"  (train env may use stall wrapper; EvalCallback = greedy, no stall, matches final eval)")
            print(f"Corr(reward, coverage) last 50: {corr:.3f}")
            print(f"Best coverage (all episodes so far): {best_ever:.3f}")
            print(f"Best coverage (last {win} episodes): {best_last_win:.3f}")
            print(f"{'='*70}\n")

        return True


class ScaleRewardWrapper(gymnasium.Wrapper):
    """Multiply reward by a positive constant. Same optimal policy; smaller |Q| targets help DQN."""

    def __init__(self, env, scale: float):
        super().__init__(env)
        self.scale = float(scale)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward * self.scale, terminated, truncated, info


class StallTerminateWrapper(gymnasium.Wrapper):
    """Wrapper that terminates episodes when no progress is made."""
    def __init__(self, env, stall_limit: int = 40):
        super().__init__(env)
        self.stall_limit = stall_limit
        self.no_progress_streak = 0

    def reset(self, **kwargs):
        self.no_progress_streak = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if info.get("new_cell_covered", False):
            self.no_progress_streak = 0
        else:
            self.no_progress_streak += 1

        if not terminated and not truncated and self.no_progress_streak >= self.stall_limit:
            truncated = True

        return obs, reward, terminated, truncated, info


class MixedEnvWrapper(gymnasium.Wrapper):
    """Cycles through multiple environments each episode to prevent forgetting."""
    def __init__(self, primary_env_id: str, replay_env_ids: list[str],
                 stall_limit: Optional[int] = 100, primary_weight: float = 0.5,
                 reward_scale: float = 1.0):
        primary = gymnasium.make(primary_env_id, render_mode=None,
                                 activate_game_status=False)
        if reward_scale != 1.0:
            primary = ScaleRewardWrapper(primary, reward_scale)
        super().__init__(primary)
        self.primary_env_id = primary_env_id
        self.replay_env_ids = list(dict.fromkeys(replay_env_ids))
        self.stall_limit = stall_limit
        self.primary_weight = primary_weight

        self._envs: dict[str, gymnasium.Env] = {primary_env_id: primary}
        for eid in self.replay_env_ids:
            if eid not in self._envs:
                e = gymnasium.make(eid, render_mode=None, activate_game_status=False)
                if reward_scale != 1.0:
                    e = ScaleRewardWrapper(e, reward_scale)
                self._envs[eid] = e
        self._active_id = primary_env_id
        self.no_progress_streak = 0

    def _pick_env(self):
        import random
        if not self.replay_env_ids or random.random() < self.primary_weight:
            return self.primary_env_id
        return random.choice(self.replay_env_ids)

    def reset(self, **kwargs):
        self._active_id = self._pick_env()
        self.env = self._envs[self._active_id]
        self.no_progress_streak = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["_env_id"] = self._active_id
        if info.get("new_cell_covered", False):
            self.no_progress_streak = 0
        else:
            self.no_progress_streak += 1
        if self.stall_limit is not None:
            if not terminated and not truncated and self.no_progress_streak >= self.stall_limit:
                truncated = True
        return obs, reward, terminated, truncated, info

    def close(self):
        for e in self._envs.values():
            e.close()


def evaluate_on_envs(model, env_ids, num_episodes=25, reward_scale: float = 1.0):
    """Evaluate model on a list of environments and print per-env results."""
    print(f"\n{'='*70}")
    print("EVALUATION ACROSS ALL COMPLETED STAGES (DQN, greedy)")
    print(f"{'='*70}")

    last_cov, last_sr = 0.0, 0.0
    all_results = {}

    for eid in env_ids:
        env = gymnasium.make(eid, render_mode=None, activate_game_status=False)
        if reward_scale != 1.0:
            env = ScaleRewardWrapper(env, reward_scale)
        total_cov = 0.0
        total_steps = 0
        n_success = 0
        n_deaths = 0

        for _ in range(num_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, info = env.step(int(action))
            total_cov += info["total_covered_cells"] / max(info["coverable_cells"], 1)
            total_steps += 500 - info["steps_remaining"]
            if info["cells_remaining"] == 0:
                n_success += 1
            if info["game_over"]:
                n_deaths += 1

        env.close()

        avg_cov = total_cov / num_episodes
        avg_steps = total_steps / num_episodes
        sr = n_success / num_episodes
        dr = n_deaths / num_episodes

        all_results[eid] = {'avg_coverage': avg_cov, 'success_rate': sr, 'death_rate': dr}
        marker = " <-- current stage" if eid == env_ids[-1] else ""
        print(f"  {eid:20s}  Coverage: {avg_cov:.3f} ({avg_cov*100:.0f}%)  "
              f"Success: {sr:.2f}  Deaths: {dr:.2f}  Steps: {avg_steps:.0f}{marker}")

        last_cov, last_sr = avg_cov, sr

    os.makedirs('training_results', exist_ok=True)
    results_file = f'training_results/{env_ids[-1]}_dqn_results.json'
    with open(results_file, 'w') as f:
        json.dump({'evaluated_envs': all_results, 'timestamp': datetime.now().isoformat()}, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return last_cov, last_sr, all_results


def _maybe_prompt_stage2_gate(eval_all, *, no_gate: bool, on_reject_hint: str) -> None:
    """After stage 2, require strong greedy eval on both just_go and safe unless --no-stage2-gate."""
    if no_gate:
        return
    jg = eval_all.get("just_go", {})
    sf = eval_all.get("safe", {})
    gate_ok = (
        jg.get("avg_coverage", 0) >= STAGE2_GATE_COVERAGE
        and sf.get("avg_coverage", 0) >= STAGE2_GATE_COVERAGE
        and jg.get("success_rate", 0) >= STAGE2_GATE_SUCCESS
        and sf.get("success_rate", 0) >= STAGE2_GATE_SUCCESS
    )
    if gate_ok:
        print(
            f"\nStage 2 gate passed: just_go & safe each have coverage>="
            f"{STAGE2_GATE_COVERAGE:.0%} and success>={STAGE2_GATE_SUCCESS:.0%}."
        )
        return
    print(
        f"\nStage 2 gate: want just_go & safe each coverage>="
        f"{STAGE2_GATE_COVERAGE:.0%}, success>={STAGE2_GATE_SUCCESS:.0%} "
        f"(25-episode greedy eval)."
    )
    print(
        f"  just_go: cov={jg.get('avg_coverage', 0):.3f} "
        f"success={jg.get('success_rate', 0):.2f}"
    )
    print(
        f"  safe:    cov={sf.get('avg_coverage', 0):.3f} "
        f"success={sf.get('success_rate', 0):.2f}"
    )
    response = input("Continue to maze (stage 3) anyway? (y/n): ")
    if response.lower() != "y":
        print(on_reject_hint)
        raise SystemExit(0)


def _make_dqn(train_env, *, env_id: str):
    """Instantiate DQN. Early stages use faster updates + looser stall (see train_single_environment)."""
    # 130-dim obs needs a wider net; default [64,64] is undersized for greedy Q to track exploration.
    policy_kwargs = dict(net_arch=[256, 256])
    enemy_envs = {"chokepoint", "sneaky_enemies"}
    # just_go / safe: no enemies — prioritize fitting Q(s,·) quickly; was: late learning_starts + sparse
    # updates → high episodic coverage under epsilon but greedy eval still ~random.
    if env_id in ("just_go", "safe"):
        return DQN(
            "MlpPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=200_000,
            learning_starts=2_000,
            batch_size=128,
            tau=1.0,
            gamma=0.99,
            # (1,2) is sample-efficient but ~2× slower wall-clock than (2,1); tune as needed.
            train_freq=(2, "step"),
            gradient_steps=1,
            target_update_interval=1_000,
            exploration_fraction=0.35,
            exploration_initial_eps=1.0,
            # 0.0 after decay: late training = greedy rollouts so Q tracks argmax policy (matches eval).
            exploration_final_eps=0.0,
            max_grad_norm=10.0,
            verbose=1,
            tensorboard_log="./dqn_tensorboard/",
        )
    if env_id == "maze":
        return DQN(
            "MlpPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=2e-4,
            buffer_size=350_000,
            learning_starts=5_000,
            batch_size=128,
            tau=1.0,
            gamma=0.99,
            train_freq=(2, "step"),
            gradient_steps=1,
            target_update_interval=1_500,
            exploration_fraction=0.2,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            max_grad_norm=10.0,
            verbose=1,
            tensorboard_log="./dqn_tensorboard/",
        )
    final_eps = 0.08 if env_id in enemy_envs else 0.05
    return DQN(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        buffer_size=500_000,
        learning_starts=10_000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=(4, "step"),
        gradient_steps=1,
        target_update_interval=2_000,
        exploration_fraction=0.15,
        exploration_initial_eps=1.0,
        exploration_final_eps=final_eps,
        max_grad_norm=10.0,
        verbose=1,
        tensorboard_log="./dqn_tensorboard/",
    )


def train_single_environment(env_id, timesteps, model=None, model_name=None,
                              eval_env_ids=None, completed_envs=None,
                              reward_scale: float = 1.0,
                              use_stall_train: bool = False,
                              replay_envs_filter: Optional[list[str]] = None,
                              primary_weight_override: Optional[float] = None):
    if eval_env_ids is None:
        eval_env_ids = [env_id]
    if completed_envs is None:
        completed_envs = []
    print("\n" + "=" * 70)
    print(f"DQN TRAINING ON: {env_id.upper()}")
    print("=" * 70)

    def _maybe_scale(e):
        if reward_scale == 1.0:
            return e
        return ScaleRewardWrapper(e, reward_scale)

    # Default NO stall on just_go/safe: EvalCallback / tournament use the raw 500-step horizon.
    # Stall-only training changes episode length and termination → train stats look good while
    # greedy eval stays at length=500 with awful return (different MDP).
    if env_id in ("just_go", "safe"):
        stall_limit = 220 if model is None else 220
        use_stall = use_stall_train
    elif env_id == "maze":
        stall_limit = 80 if model is None else 120
        use_stall = True
    else:
        stall_limit = 40 if model is None else 100
        use_stall = True

    replay_ids = [eid for eid in dict.fromkeys(completed_envs) if eid != env_id]
    if replay_envs_filter:
        allow = set(replay_envs_filter)
        replay_ids = [e for e in replay_ids if e in allow]
        missing = allow - set(completed_envs)
        if missing:
            print(f"  Note: --replay-envs {sorted(missing)} not in completed stages, skipped.")
    if replay_ids:
        primary_weight = 0.65 if primary_weight_override is None else float(primary_weight_override)
        primary_weight = min(0.99, max(0.01, primary_weight))
        print(f"Mixed training: {env_id} ({primary_weight:.0%}) + replay {replay_ids} ({1-primary_weight:.0%})")
        train_env = MixedEnvWrapper(
            primary_env_id=env_id,
            replay_env_ids=replay_ids,
            stall_limit=stall_limit if use_stall else None,
            primary_weight=primary_weight,
            reward_scale=reward_scale,
        )
        if use_stall:
            print(f"Mixed env stall limit: {stall_limit}")
        else:
            print("Mixed env: stall off (matches full-horizon eval)")
    else:
        train_env = gymnasium.make(env_id, render_mode=None, activate_game_status=False)
        train_env = _maybe_scale(train_env)
        if use_stall:
            train_env = StallTerminateWrapper(train_env, stall_limit=stall_limit)
            print(f"Stall limit: {stall_limit} steps (--stall-train)")
        else:
            print("Stall: off (same horizon as EvalCallback / tournament; recommended for DQN)")
    if reward_scale != 1.0:
        print(f"Reward scale: {reward_scale} (DQN Q targets scaled; same optimal policy)")

    if model is None:
        print("Creating new DQN model...")
        model = _make_dqn(train_env, env_id=env_id)
    else:
        print(f"Continuing training with existing DQN on {env_id}...")
        model.set_env(train_env)
        model.learning_rate = 5e-5
        if hasattr(model, "lr_schedule"):
            model.lr_schedule = lambda _: 5e-5
        for pg in model.policy.optimizer.param_groups:
            pg["lr"] = 5e-5
        enemy_envs = {"chokepoint", "sneaky_enemies"}
        if env_id in enemy_envs:
            model.exploration_final_eps = 0.12
        elif env_id in ("just_go", "safe"):
            model.exploration_final_eps = 0.0

    stats_cb = EpisodeStatsCallback(
        check_freq=5000, verbose=1,
        primary_env_id=env_id if replay_ids else None,
    )
    # Raw env (no stall): same MDP as tournament / evaluate_on_envs — tracks true greedy quality.
    eval_env_raw = gymnasium.make(env_id, render_mode=None, activate_game_status=False)
    if reward_scale != 1.0:
        eval_env_raw = ScaleRewardWrapper(eval_env_raw, reward_scale)
    eval_freq = max(10_000, min(50_000, timesteps // 8))
    best_dir = (
        os.path.join("dqn_eval_best", model_name)
        if model_name
        else os.path.join("dqn_eval_best", env_id)
    )
    eval_cb = DetailedEvalCallback(
        eval_env_raw,
        n_eval_episodes=15,
        eval_freq=eval_freq,
        deterministic=True,
        best_model_save_path=best_dir,
        verbose=1,
    )
    callback = CallbackList([stats_cb, eval_cb])

    print(f"\nTraining for {timesteps:,} timesteps...")
    print(f"Greedy eval every {eval_freq} steps → TensorBoard: eval/mean_reward, "
          f"eval/mean_coverage, eval/success_rate")
    print(f"Best eval-by-reward checkpoint: {best_dir}/best_model.zip\n")
    try:
        model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=True,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    finally:
        eval_env_raw.close()

    train_env.close()

    if model_name:
        model.save(model_name)
        print(f"\nModel saved as: {model_name}.zip (last training weights)")

    best_zip = os.path.join(
        "dqn_eval_best", model_name or env_id, "best_model.zip"
    )
    if os.path.isfile(best_zip):
        base = best_zip[:-4]
        print(f"\nEvaluating best eval checkpoint (by mean_reward): {best_zip}")
        ev = gymnasium.make(env_id, render_mode=None, activate_game_status=False)
        best_model = DQN.load(base, env=ev)
        ev.close()
        _, _, _ = evaluate_on_envs(best_model, eval_env_ids, reward_scale=reward_scale)
        print("(Above = best_model.zip; below = last training weights. "
              "Next stage still loads last .zip unless you point --load at best.)\n")

    avg_coverage, success_rate, eval_by_env = evaluate_on_envs(
        model, eval_env_ids, reward_scale=reward_scale
    )
    return model, avg_coverage, success_rate, eval_by_env


# After stage 2 we require both just_go and safe to pass STAGE2_GATE_* (unless --no-stage2-gate).
# Stage 2 mixed replay is only just_go (it's the sole prior env in completed_envs).
STAGE2_GATE_COVERAGE = 0.99
STAGE2_GATE_SUCCESS = 0.80

# Same curriculum as train_ppo.py; checkpoint names prefixed with dqn_
STAGES = [
    ("just_go",       200_000, "dqn_stage1_justgo",      0.95),
    ("safe",          550_000, "dqn_stage2_safe",        0.95),
    ("maze",          300_000, "dqn_stage3_maze",        0.75),
    ("chokepoint",  4_500_000, "dqn_stage4_chokepoint",  0.65),
    ("sneaky_enemies", 3_000_000, "dqn_stage5_sneaky",   0.60),
]


def load_checkpoint(checkpoint_path, env_id):
    """Load a saved DQN if the .zip exists, else return None."""
    zip_path = checkpoint_path + ".zip"
    if not os.path.exists(zip_path):
        return None
    print(f"  [RESUME] Loading DQN checkpoint: {zip_path}")
    env = gymnasium.make(env_id, render_mode=None, activate_game_status=False)
    loaded = DQN.load(checkpoint_path, env=env)
    env.close()
    return loaded


def parse_args():
    parser = argparse.ArgumentParser(
        description="Coverage GridWorld curriculum training (DQN / Stable-Baselines3)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore ALL dqn_stage* checkpoints and train every stage from scratch.",
    )
    parser.add_argument(
        "--fresh-from", type=int, default=None, metavar="N",
        help="Load checkpoints for stages 1..N-1, then retrain from stage N onward.",
    )
    parser.add_argument(
        "--fresh-stages", type=int, nargs="+", default=None, metavar="N",
        help="Retrain only the listed stage numbers (e.g. --fresh-stages 4 5).",
    )
    parser.add_argument(
        "--load", type=str, default=None, metavar="PATH",
        help="Load a specific DQN checkpoint (.zip path or base without .zip). "
             "Cannot be a PPO checkpoint.",
    )
    parser.add_argument(
        "--reward-scale", type=float, default=1.0, metavar="S",
        help="Multiply all rewards by S (e.g. 0.25) for stabler DQN Q targets. "
             "Default 1.0. Use the same S when resuming a scaled run.",
    )
    parser.add_argument(
        "--stall-train", action="store_true",
        help="On just_go/safe, wrap training with stall termination (misaligns with "
             "EvalCallback unless you know you want it). Default: no stall on those maps.",
    )
    parser.add_argument(
        "--only-stage", type=int, default=None, metavar="N",
        help="Run only stage N (1-based). Sets completed_envs for mixed replay from "
             "earlier maps. Loads dqn_stage1..N-1 in order unless --load is set "
             "(then use e.g. best_model as starting weights). Combine with "
             "--fresh-stages N to ignore dqn_stageN*.zip on disk.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None, metavar="T",
        help="With --only-stage only: override that stage's env-step budget (default "
             "comes from STAGES).",
    )
    parser.add_argument(
        "--replay-envs", nargs="+", default=None, metavar="ID",
        help="During mixed training, replay only these env ids (subset of completed "
             "stages). Example for maze: --replay-envs just_go",
    )
    parser.add_argument(
        "--primary-weight", type=float, default=None, metavar="P",
        help="Episode fraction on the current stage when mixing (default 0.65). "
             "Use 0.5 for 50%% maze / 50%% replay.",
    )
    parser.add_argument(
        "--no-stage2-gate",
        action="store_true",
        help="Skip the post-stage-2 check that just_go and safe both hit high coverage/success.",
    )
    return parser.parse_args()


def should_skip_checkpoint(stage_num: int, args) -> bool:
    if args.fresh:
        return True
    if args.fresh_from is not None and stage_num >= args.fresh_from:
        return True
    if args.fresh_stages is not None and stage_num in args.fresh_stages:
        return True
    return False


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("COVERAGE GRIDWORLD — CURRICULUM DQN")
    print("=" * 70)
    print("Checkpoints: dqn_stage*.zip (separate from PPO).")
    print("PPO .zip files cannot be loaded here — different algorithm / network.")
    if args.fresh:
        print(">> --fresh: ignoring ALL checkpoints, training from scratch.")
    elif args.fresh_from:
        print(f">> --fresh-from {args.fresh_from}: retraining stages {args.fresh_from}+.")
    elif args.fresh_stages:
        print(f">> --fresh-stages {args.fresh_stages}: retraining those stages only.")
    if args.reward_scale != 1.0:
        print(f">> --reward-scale {args.reward_scale}: rewards scaled (match when loading checkpoints).")
    if args.stall_train:
        print(">> --stall-train: stall wrapper enabled on just_go/safe (train ≠ default eval MDP).")
    if args.only_stage is not None:
        print(f">> --only-stage {args.only_stage}: single-stage run.")
        if args.timesteps is not None:
            print(f">> --timesteps {args.timesteps}: overrides STAGES budget for that stage.")
    if args.replay_envs:
        print(f">> --replay-envs {' '.join(args.replay_envs)}: restricted mixed replay.")
    if args.primary_weight is not None:
        print(f">> --primary-weight {args.primary_weight}: custom primary / replay split.")
    print()

    model = None
    completed_envs = []

    if args.load:
        load_path = args.load
        if load_path.endswith(".zip"):
            load_path = load_path[:-4]
        print(f">> Loading DQN checkpoint: {load_path}")
        dummy_env = gymnasium.make(STAGES[0][0], render_mode=None,
                                   activate_game_status=False)
        model = DQN.load(load_path, env=dummy_env)
        dummy_env.close()
        print(f"   Observation space: {model.observation_space}")

    if args.only_stage is not None:
        osn = args.only_stage
        if osn < 1 or osn > len(STAGES):
            raise SystemExit(f"--only-stage must be between 1 and {len(STAGES)}")
        completed_envs = [STAGES[i][0] for i in range(osn - 1)]
        if not args.load:
            for sn in range(1, osn):
                saved = load_checkpoint(STAGES[sn - 1][2], STAGES[sn - 1][0])
                if saved is None:
                    raise SystemExit(
                        f"Missing {STAGES[sn - 1][2]}.zip for stage {sn}; "
                        "train earlier stages or pass --load."
                    )
                model = saved

    stage_loop = list(enumerate(STAGES, start=1))
    if args.only_stage is not None:
        stage_loop = [(args.only_stage, STAGES[args.only_stage - 1])]

    for stage_num, (env_id, timesteps, checkpoint, threshold) in stage_loop:
        print("\n" + "#" * 70)
        print(f"# STAGE {stage_num}: {env_id.upper()} (DQN)")
        print("#" * 70)

        skip_ckpt = should_skip_checkpoint(stage_num, args)

        saved = None if skip_ckpt else load_checkpoint(checkpoint, env_id)
        if saved is not None:
            model = saved
            print(f"Stage {stage_num} checkpoint found — skipping training.")
            completed_envs.append(env_id)
            _, _, eval_all_skip = evaluate_on_envs(
                model, completed_envs, reward_scale=args.reward_scale
            )
            if stage_num == 2:
                _maybe_prompt_stage2_gate(
                    eval_all_skip,
                    no_gate=args.no_stage2_gate,
                    on_reject_hint=(
                        "Stopped after stage 2 (checkpoint skip path). "
                        "Retrain safe or load a stronger checkpoint."
                    ),
                )
        else:
            if skip_ckpt and args.load is None and model is None:
                print("  (checkpoint skipped by CLI flag, no model yet — skipping stage)")
                completed_envs.append(env_id)
                continue
            if skip_ckpt:
                print("  (checkpoint skipped by CLI flag)")
            train_steps = timesteps
            if args.only_stage is not None and args.timesteps is not None:
                train_steps = args.timesteps
            model, coverage, success, eval_all = train_single_environment(
                env_id=env_id,
                timesteps=train_steps,
                model=model,
                model_name=checkpoint,
                eval_env_ids=completed_envs + [env_id],
                completed_envs=completed_envs,
                reward_scale=args.reward_scale,
                use_stall_train=args.stall_train,
                replay_envs_filter=args.replay_envs,
                primary_weight_override=args.primary_weight,
            )
            completed_envs.append(env_id)

            print(f"\nStage {stage_num} ({env_id}) — Coverage: {coverage:.3f}, "
                  f"Success: {success:.2f}  (threshold: {threshold:.2f})")
            if coverage < threshold:
                print(f"Warning: coverage below {threshold:.0%}.")
                response = input("Continue to next stage anyway? (y/n): ")
                if response.lower() != 'y':
                    print(f"Stopped. Checkpoint saved as {checkpoint}.zip — re-run to resume.")
                    exit(0)

            if stage_num == 2:
                _maybe_prompt_stage2_gate(
                    eval_all,
                    no_gate=args.no_stage2_gate,
                    on_reject_hint=(
                        "Stopped after stage 2. Retrain with e.g. "
                        "--only-stage 2 --fresh-stages 2 --timesteps 800000 "
                        "or load best from dqn_eval_best/dqn_stage2_safe/best_model.zip."
                    ),
                )

    print("\n" + "=" * 70)
    print("ALL DQN STAGES COMPLETE!")
    print("=" * 70)
    final = STAGES[-1][2]
    print(f"Final model: {final}.zip")
    print("=" * 70)


if __name__ == "__main__":
    main()
