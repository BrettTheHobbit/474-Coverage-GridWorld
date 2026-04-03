"""
Coverage GridWorld — curriculum training with PPO (Proximal Policy Optimization).

Trains one Gymnasium stage at a time (``just_go`` → … → ``sneaky_enemies``), optionally
mixing in earlier maps so the policy does not forget. Checkpoints are written as
``ppo_stage*.zip`` (Stable-Baselines3).

Related scripts: ``train_dqn.py`` (same curriculum, DQN). ``train_step_by_step.py`` is a
thin alias of this file for older commands / homework write-ups.

Install the env package from the repo root, then run:

.. code-block:: text

   pip install -e ./coverage-gridworld
   python train_ppo.py

CLI examples
------------
::

  python train_ppo.py                              # resume from checkpoints on disk
  python train_ppo.py --fresh                      # every stage from scratch
  python train_ppo.py --fresh-from 4               # keep stages 1–3, retrain 4+
  python train_ppo.py --fresh-stages 4 5           # only retrain those stages
  python train_ppo.py --only-stage 5 --load ppo_stage4_chokepoint --fresh-stages 5 \\
      --timesteps 3000000
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import coverage_gridworld
import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeStatsCallback(BaseCallback):
    """
    Callback to monitor detailed episode statistics during training.
    Tracks episode length and reward/step so you can spot when return rises from
    longer episodes / dense shaping while coverage stays flat.

    If *primary_env_id* is set (mixed-env training), only episodes from that
    environment are included in the printed stats.  Replay episodes are counted
    but not mixed into the coverage / success metrics.
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
            print(f"Training Progress - Step {self.n_calls}{env_label}")
            print(f"{'='*70}")
            print(f"Episodes completed: {self.episode_count}  (replay: {self.replay_count})")
            print(f"Avg Reward (last 50): {np.mean(recent_rewards):.2f}")
            print(f"Avg episode length (last 50): {np.mean(recent_lengths):.1f}")
            print(f"Avg Reward / step (last 50): {np.mean(r_per_step):.3f}")
            print(f"Avg Coverage (last 50): {np.mean(recent_coverage):.3f}")
            print(f"Median Coverage (last 50): {float(np.median(recent_coverage)):.3f}")
            print(f"Success Rate (last 50): {np.mean(recent_success):.2f}")
            print(f"Corr(reward, coverage) last 50: {corr:.3f}  (low/neg while return rises => dense shaping dominates)")
            print(f"Best coverage (all episodes so far): {best_ever:.3f}")
            print(f"Best coverage (last {win} episodes): {best_last_win:.3f}")
            print(f"{'='*70}\n")

        return True


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
    """Cycles through multiple environments each episode to prevent forgetting.

    The primary env (the current stage) gets ``primary_weight`` fraction of
    episodes; the rest are shared equally among the replay envs.
    """
    def __init__(self, primary_env_id: str, replay_env_ids: list[str],
                 stall_limit: int = 100, primary_weight: float = 0.5):
        primary = gymnasium.make(primary_env_id, render_mode=None,
                                 activate_game_status=False)
        super().__init__(primary)
        self.primary_env_id = primary_env_id
        self.replay_env_ids = list(dict.fromkeys(replay_env_ids))  # dedupe, keep order
        self.stall_limit = stall_limit
        self.primary_weight = primary_weight

        self._envs: dict[str, gymnasium.Env] = {primary_env_id: primary}
        for eid in self.replay_env_ids:
            if eid not in self._envs:
                self._envs[eid] = gymnasium.make(
                    eid, render_mode=None, activate_game_status=False
                )
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
        info["_env_id"] = self._active_id
        if info.get("new_cell_covered", False):
            self.no_progress_streak = 0
        else:
            self.no_progress_streak += 1
        if not terminated and not truncated and self.no_progress_streak >= self.stall_limit:
            truncated = True
        return obs, reward, terminated, truncated, info

    def close(self):
        for e in self._envs.values():
            e.close()


def evaluate_on_envs(model, env_ids, num_episodes=20):
    """
    Evaluate model on a list of environments and print per-env results.
    Returns (avg_coverage, success_rate) for the LAST env in the list
    (the current stage env), for use in threshold checks.
    """
    print(f"\n{'=' * 70}")
    print("EVALUATION ACROSS ALL COMPLETED STAGES")
    print(f"{'=' * 70}")

    last_cov, last_sr = 0.0, 0.0
    all_results = {}

    for eid in env_ids:
        env = gymnasium.make(eid, render_mode=None, activate_game_status=False)
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

    # Save combined results
    os.makedirs('training_results', exist_ok=True)
    results_file = f'training_results/{env_ids[-1]}_results.json'
    with open(results_file, 'w') as f:
        json.dump({'evaluated_envs': all_results, 'timestamp': datetime.now().isoformat()}, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return last_cov, last_sr


def train_single_environment(env_id, timesteps, model=None, model_name=None,
                              eval_env_ids=None, completed_envs=None,
                              replay_envs_filter=None, primary_weight_override=None):
    """
    Train on a single environment and evaluate performance.

    When *completed_envs* is non-empty the training env is a MixedEnvWrapper
    that replays earlier maps ~50% of the time to prevent catastrophic
    forgetting.

    Args:
        env_id: Environment ID to train on
        timesteps: Number of timesteps to train
        model: Existing model to continue training (None to create new)
        model_name: Name to save the model as
        eval_env_ids: List of env IDs to evaluate on after training.
                      Defaults to [env_id]. Pass cumulative list to test
                      all prior stages as well.
        completed_envs: Environments from earlier stages (used for replay).
        replay_envs_filter: If set, only these env ids are used as replay (subset of completed_envs).
        primary_weight_override: Episode fraction on primary env when mixing (default 0.65).

    Returns:
        trained_model, avg_coverage, success_rate  (metrics for env_id)
    """
    if eval_env_ids is None:
        eval_env_ids = [env_id]
    if completed_envs is None:
        completed_envs = []
    print("\n" + "=" * 70)
    print(f"TRAINING ON: {env_id.upper()}")
    print("=" * 70)
    
    stall_limit = 40 if model is None else 100

    replay_ids = [eid for eid in dict.fromkeys(completed_envs) if eid != env_id]
    if replay_envs_filter is not None:
        allow = set(replay_envs_filter)
        replay_ids = [eid for eid in replay_ids if eid in allow]
    primary_weight = 0.65 if primary_weight_override is None else float(primary_weight_override)
    if replay_ids:
        print(f"Mixed training: {env_id} ({primary_weight:.0%}) + replay {replay_ids} ({1-primary_weight:.0%})")
        train_env = MixedEnvWrapper(
            primary_env_id=env_id,
            replay_env_ids=replay_ids,
            stall_limit=stall_limit,
            primary_weight=primary_weight,
        )
    else:
        train_env = gymnasium.make(env_id, render_mode=None, activate_game_status=False)
        train_env = StallTerminateWrapper(train_env, stall_limit=stall_limit)
    print(f"Stall limit: {stall_limit} steps")

    # Initialize or update model
    if model is None:
        print("Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="./ppo_tensorboard/"
        )
    else:
        print(f"Continuing training with existing model on {env_id}...")
        # Fine-tuning: lower lr + higher entropy to prevent converging to the
        # fast-stall strategy (cover 75% quickly, accept termination, repeat).
        # SB3 uses lr_schedule internally — must replace the schedule object,
        # not just the attribute, otherwise _update_learning_rate overwrites it.
        # get_schedule_fn(c) just returns lambda _: c, so use a lambda directly.
        model.lr_schedule = lambda _: 1e-4
        model.learning_rate = 1e-4
        for pg in model.policy.optimizer.param_groups:
            pg["lr"] = 1e-4
        # Enemy stages need higher entropy to avoid collapsing to a "safe
        # but limited" corridor before learning to time enemy cones.
        # 0.08 is a middle ground: enough to reopen exploration from a
        # collapsed checkpoint without destroying a good policy.
        enemy_envs = {"chokepoint", "sneaky_enemies"}
        model.ent_coef = 0.1 if env_id in enemy_envs else 0.03
        model.set_env(train_env)
    
    # Train with monitoring — only report stats for the primary env
    callback = EpisodeStatsCallback(
        check_freq=5000, verbose=1,
        primary_env_id=env_id if replay_ids else None,
    )
    
    print(f"\nTraining for {timesteps:,} timesteps...")
    print("Monitor the stats above to see if the agent is improving.")
    print("Press Ctrl+C to stop early if needed.\n")
    
    try:
        model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=True,
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    train_env.close()

    # Save model before evaluation so progress isn't lost on Ctrl+C
    if model_name:
        model.save(model_name)
        print(f"\nModel saved as: {model_name}.zip")

    avg_coverage, success_rate = evaluate_on_envs(model, eval_env_ids)
    return model, avg_coverage, success_rate


# ---------------------------------------------------------------------------
# Stage definitions — add/uncomment stages as the agent matures.
# Each entry: (env_id, timesteps, checkpoint_name, coverage_threshold)
# ---------------------------------------------------------------------------
STAGES = [
    ("just_go",       75_000, "ppo_stage1_justgo",    0.95),
    ("safe",          60_000, "ppo_stage2_safe",       0.95),
    ("maze",          300_000, "ppo_stage3_maze",       0.75),
    ("chokepoint",    4_500_000, "ppo_stage4_chokepoint", 0.65),
    # ("maze",          300_000, "ppo_stage3_maze",       0.75),
    ("sneaky_enemies",3_000_000,"ppo_stage5_sneaky",   0.60),
    # ("standard",      400_000, "ppo_final_complete",   0.60),
]


def load_checkpoint(checkpoint_path, env_id):
    """Load a saved model if the .zip exists, else return None."""
    zip_path = checkpoint_path + ".zip"
    if not os.path.exists(zip_path):
        return None
    print(f"  [RESUME] Loading checkpoint: {zip_path}")
    env = gymnasium.make(env_id, render_mode=None, activate_game_status=False)
    loaded = PPO.load(checkpoint_path, env=env)
    env.close()
    return loaded


def parse_args():
    parser = argparse.ArgumentParser(
        description="Coverage GridWorld curriculum training (PPO / Stable-Baselines3)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore ALL checkpoints and train every stage from scratch.",
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
        help="Load a specific checkpoint (.zip) and continue training from "
             "the stage given by --fresh-from (or the first fresh stage). "
             "Example: --load ppo_stage4_chokepoint_large.zip --fresh-from 4",
    )
    parser.add_argument(
        "--only-stage", type=int, default=None, metavar="N",
        help="Run only stage N (1-based). Sets mixed replay from all earlier curriculum maps. "
             "Loads ppo_stage1..N-1 in order unless --load is set (then use that as weights). "
             "Combine with --fresh-stages N to ignore ppo_stageN*.zip on disk.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None, metavar="T",
        help="With --only-stage: override that stage's timestep budget (default from STAGES).",
    )
    parser.add_argument(
        "--replay-envs", nargs="+", default=None, metavar="ID",
        help="When mixing, replay only these env ids (subset of completed stages). "
             "Default: all prior stages.",
    )
    parser.add_argument(
        "--primary-weight", type=float, default=None, metavar="P",
        help="Episode fraction on the current stage when mixing (default 0.65).",
    )
    return parser.parse_args()


def should_skip_checkpoint(stage_num: int, args) -> bool:
    """Return True if this stage should ignore its saved checkpoint."""
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
    print("COVERAGE GRIDWORLD — CURRICULUM PPO")
    print("=" * 70)
    print("Checkpoints are saved after each stage — safe to exit and resume.")
    print("Each stage's evaluation covers ALL prior stages (forgetting check).")
    if args.fresh:
        print(">> --fresh: ignoring ALL checkpoints, training from scratch.")
    elif args.fresh_from:
        print(f">> --fresh-from {args.fresh_from}: retraining stages {args.fresh_from}+.")
    elif args.fresh_stages:
        print(f">> --fresh-stages {args.fresh_stages}: retraining those stages only.")
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
    completed_envs = []  # grows as stages finish; used for cumulative eval

    # Pre-load a specific checkpoint if requested
    if args.load:
        load_path = args.load
        if load_path.endswith(".zip"):
            load_path = load_path[:-4]
        print(f">> Loading checkpoint: {load_path}")
        dummy_env = gymnasium.make(STAGES[0][0], render_mode=None,
                                   activate_game_status=False)
        model = PPO.load(load_path, env=dummy_env)
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
                        "train earlier stages or pass --load with a compatible checkpoint."
                    )
                model = saved

    stage_enumeration = list(enumerate(STAGES, start=1))
    if args.only_stage is not None:
        stage_enumeration = [(args.only_stage, STAGES[args.only_stage - 1])]

    for stage_num, (env_id, timesteps, checkpoint, threshold) in stage_enumeration:
        print("\n" + "#" * 70)
        print(f"# STAGE {stage_num}: {env_id.upper()}")
        print("#" * 70)

        skip_ckpt = should_skip_checkpoint(stage_num, args)

        # --- Resume: skip training if checkpoint exists (unless overridden) ---
        saved = None if skip_ckpt else load_checkpoint(checkpoint, env_id)
        if saved is not None:
            model = saved
            print(f"Stage {stage_num} checkpoint found — skipping training.")
            completed_envs.append(env_id)
            evaluate_on_envs(model, completed_envs)
        else:
            if skip_ckpt and args.load is None and model is None:
                print(f"  (checkpoint skipped by CLI flag, no model yet — skipping stage)")
                completed_envs.append(env_id)
                continue
            if skip_ckpt:
                print(f"  (checkpoint skipped by CLI flag)")
            train_steps = timesteps
            if args.only_stage is not None and args.timesteps is not None:
                train_steps = args.timesteps
            model, coverage, success = train_single_environment(
                env_id=env_id,
                timesteps=train_steps,
                model=model,
                model_name=checkpoint,
                eval_env_ids=completed_envs + [env_id],
                completed_envs=completed_envs,
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

    print("\n" + "=" * 70)
    print("ALL STAGES COMPLETE!")
    print("=" * 70)
    final = STAGES[-1][2]
    print(f"Final model: {final}.zip")
    print("=" * 70)


if __name__ == "__main__":
    main()
