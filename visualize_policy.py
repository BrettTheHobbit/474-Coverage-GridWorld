#!/usr/bin/env python3
"""
Visualize and diagnose a trained PPO on the Coverage GridWorld (default: chokepoint).

The chokepoint checkpoint (e.g. save_model/ppo_stage4_chokepoint.zip) was trained with
OBS_MODE 3 → Box(59,). The script infers observation mode from the loaded model so the
env matches the policy unless you pass --obs-mode.

Interpreting a ~0.72 coverage "ceiling"
  - On ``chokepoint``, ``coverable_cells`` is 79 (walls excluded), not 100. Coverage is
    ``total_covered_cells / 79``. Full clear would be 1.0.
  - If rollouts show many deaths and no successes, the policy is not stuck at a numeric
    bug cap; it is usually dying while ~20+ coverable cells remain (enemy timing / unsafe
    routes). Use ``--human --trace`` to see where it commits before GAME OVER.

Examples:
  .venv/bin/python visualize_policy.py --human --delay 0.12
  .venv/bin/python visualize_policy.py --rollouts 80
  .venv/bin/python visualize_policy.py --model save_model/ppo_stage4_chokepoint.zip --trace \\
      --save-trace training_results/choke_trace.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import gymnasium as gym
import numpy as np


def _infer_obs_mode_from_space(observation_space) -> int:
    import gymnasium.spaces as spaces

    if isinstance(observation_space, spaces.MultiDiscrete):
        return 1
    if isinstance(observation_space, spaces.Box):
        n = int(np.prod(observation_space.shape))
        mapping = {3: 2, 12: 4, 59: 3, 130: 5}
        if n not in mapping:
            raise ValueError(
                f"Unsupported Box obs size {n}; set --obs-mode explicitly (1–5)."
            )
        return mapping[n]
    raise ValueError(f"Unsupported observation space: {observation_space}")


def _apply_obs_mode(mode: int) -> None:
    import coverage_gridworld.custom as custom

    if mode not in (1, 2, 3, 4, 5):
        raise ValueError("obs-mode must be 1–5")
    custom.OBS_MODE = mode


def load_model_and_match_obs(model_path: str, obs_mode: int | None):
    from stable_baselines3 import PPO

    path = model_path
    if path.endswith(".zip"):
        path = path[:-4]
    model = PPO.load(path, env=None)
    inferred = _infer_obs_mode_from_space(model.observation_space)
    if obs_mode is not None:
        if obs_mode != inferred:
            print(
                f"Warning: --obs-mode {obs_mode} != checkpoint's inferred mode {inferred}; "
                "predict will likely error unless dimensions match.",
                file=sys.stderr,
            )
        _apply_obs_mode(obs_mode)
    else:
        _apply_obs_mode(inferred)
        print(f"Inferred OBS_MODE={inferred} from checkpoint ({model.observation_space}).")

    import coverage_gridworld  # noqa: F401 — register envs

    return model


def make_env(env_id: str, render_mode: str | None, verbose_status: bool = False):
    return gym.make(
        env_id, render_mode=render_mode, activate_game_status=verbose_status
    )


def run_episode(
    model,
    env,
    deterministic: bool,
    trace: bool,
    delay_s: float,
) -> dict[str, Any]:
    obs, info = env.reset()
    log: list[dict[str, Any]] = []
    step = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        cov = info["total_covered_cells"] / max(info["coverable_cells"], 1)
        gs = int(getattr(env.unwrapped, "grid_size", 10))
        row, col = divmod(info["agent_pos"], gs)

        if trace or delay_s > 0:
            print(
                f"  t={step:3d}  a={action}  r={reward:7.2f}  "
                f"pos=({row},{col})  cov={cov:.3f}  rem={info['cells_remaining']:3d}  "
                f"new={info['new_cell_covered']}  over={info['game_over']}"
            )

        log.append(
            {
                "t": step,
                "action": action,
                "reward": float(reward),
                "agent_pos": int(info["agent_pos"]),
                "coverage": float(cov),
                "cells_remaining": int(info["cells_remaining"]),
                "new_cell_covered": bool(info["new_cell_covered"]),
                "game_over": bool(info["game_over"]),
                "steps_remaining": int(info["steps_remaining"]),
            }
        )
        step += 1

        if delay_s > 0:
            time.sleep(delay_s)

    coverable = int(info["coverable_cells"])
    final_cov = info["total_covered_cells"] / max(coverable, 1)
    success = info["cells_remaining"] == 0 and not info["game_over"]
    reason = "success" if success else ("death" if info["game_over"] else "timeout_or_trunc")

    print(
        f"\n--- Episode end: {reason} | coverage={final_cov:.3f} "
        f"({info['total_covered_cells']}/{coverable}) | steps={step} ---\n"
    )

    return {
        "terminal": reason,
        "coverage": float(final_cov),
        "covered": int(info["total_covered_cells"]),
        "coverable": coverable,
        "steps": step,
        "game_over": bool(info["game_over"]),
        "cells_remaining": int(info["cells_remaining"]),
        "trace": log,
    }


def batch_rollouts(model, env_id: str, n: int, deterministic: bool) -> None:
    import coverage_gridworld  # noqa: F401

    coverables: list[int] = []
    coverages: list[float] = []
    deaths = 0
    successes = 0

    for i in range(n):
        env = make_env(env_id, render_mode=None, verbose_status=False)
        obs, info = env.reset()
        coverables.append(int(getattr(env.unwrapped, "coverable_cells", 0)))
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(int(action))
            steps += 1
        cov = info["total_covered_cells"] / max(info["coverable_cells"], 1)
        coverages.append(cov)
        if info["game_over"]:
            deaths += 1
        if info["cells_remaining"] == 0 and not info["game_over"]:
            successes += 1
        env.close()
        print(f"  rollout {i+1}/{n}: coverage={cov:.3f} steps={steps} go={info['game_over']}")

    print("\n=== Batch summary ===")
    print(f"  coverable_cells (should be constant): min={min(coverables)} max={max(coverables)}")
    print(
        f"  coverage: mean={np.mean(coverages):.3f}  max={np.max(coverages):.3f}  "
        f"min={np.min(coverages):.3f}  std={np.std(coverages):.3f}"
    )
    print(f"  death episodes: {deaths}/{n}  successes: {successes}/{n}")
    print(
        "\nNote: coverage is total_covered_cells / coverable_cells "
        "(walls excluded). Hitting ~0.72 means ~72% of *reachable* tiles, not 72% of 100 grid cells."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize / diagnose PPO on Coverage GridWorld.")
    parser.add_argument(
        "--model",
        default="save_model/ppo_stage4_chokepoint.zip",
        help="Path to .zip or basename without .zip",
    )
    parser.add_argument("--env-id", default="chokepoint", help="Gymnasium env id")
    parser.add_argument(
        "--obs-mode",
        type=int,
        default=None,
        choices=(1, 2, 3, 4, 5),
        help="Force OBS_MODE; default: infer from checkpoint",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Episodes in human/trace mode")
    parser.add_argument("--deterministic", action="store_true", help="Greedy policy")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions (default)")
    parser.add_argument("--human", action="store_true", help="Pygame window (render_mode=human)")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds between steps (with --human)")
    parser.add_argument("--trace", action="store_true", help="Print every step (text)")
    parser.add_argument("--rollouts", type=int, default=0, help="If >0, run headless batch diagnostics")
    parser.add_argument(
        "--save-trace",
        default="",
        help="JSON path to save last episode step log (requires --trace or implies trace for last ep)",
    )
    args = parser.parse_args()

    deterministic = True if args.deterministic else not args.stochastic

    if not os.path.isfile(args.model) and not os.path.isfile(args.model + ".zip"):
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    model = load_model_and_match_obs(args.model, args.obs_mode)
    render_mode = "human" if args.human else None

    if args.rollouts > 0:
        batch_rollouts(model, args.env_id, args.rollouts, deterministic=deterministic)
        return

    all_traces: list[dict[str, Any]] = []
    for ep in range(args.episodes):
        print(f"\n======== Episode {ep + 1}/{args.episodes} ========")
        env = make_env(
            args.env_id, render_mode=render_mode, verbose_status=args.human or args.trace
        )
        obs_space = env.observation_space
        if obs_space.shape != model.observation_space.shape:
            env.close()
            print(
                f"Observation mismatch: env {obs_space.shape} vs model {model.observation_space.shape}",
                file=sys.stderr,
            )
            sys.exit(1)
        summary = run_episode(
            model,
            env,
            deterministic=deterministic,
            trace=args.trace or bool(args.save_trace),
            delay_s=args.delay if args.human else 0.0,
        )
        env.close()
        all_traces.append(summary)

    if args.save_trace and all_traces:
        out = {
            "env_id": args.env_id,
            "model": args.model,
            "deterministic": deterministic,
            "last_episode": all_traces[-1],
        }
        with open(args.save_trace, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote trace JSON to {args.save_trace}")


if __name__ == "__main__":
    main()
