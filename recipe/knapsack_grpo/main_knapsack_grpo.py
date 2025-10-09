# Copyright 2025 Kingdee / Cedric
# Licensed under the Apache License, Version 2.0

"""
Entry point for Knapsack-GRPO recipe.

Design:
- We DO NOT modify core trainer.
- We provide a custom trainer `KnapsackRayTrainer` under this recipe.
- To reuse all the mature wiring in `verl.trainer.main_ppo.run_ppo`, we monkey-patch
  `verl.trainer.ppo.ray_trainer.RayPPOTrainer` at runtime to our recipe trainer.
  This keeps all Ray init / resource pools / workers / reward / dataset plumbing intact.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

def _patch_trainer():
    # Lazy import to avoid import cycles
    from recipe.knapsack_grpo.knapsack_ray_trainer import KnapsackRayTrainer
    import importlib
    rt_mod = importlib.import_module("verl.trainer.ppo.ray_trainer")
    # Monkey-patch RayPPOTrainer symbol that run_ppo() will import/use
    setattr(rt_mod, "RayPPOTrainer", KnapsackRayTrainer)

@hydra.main(config_path="config", config_name="knapsack_grpo_trainer", version_base=None)
def main(cfg):
    """
    Behaves like `verl.trainer.main_ppo:main`, but trainer implementation is ours.
    """
    from verl.trainer.main_ppo import run_ppo, get_ppo_ray_runtime_env  # reuse official entry

    # 1) Init Ray if needed (keep the same behavior / env as main_ppo)
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = cfg.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**ray_init_kwargs)

    # 2) Monkey-patch the trainer symbol that run_ppo() uses
    _patch_trainer()

    # 3) Delegate to official pipeline wiring
    run_ppo(cfg)


if __name__ == "__main__":
    main()