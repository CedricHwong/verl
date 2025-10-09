# Copyright 2025 Kingdee / Cedric
# Licensed under the Apache License, Version 2.0

"""
Knapsack-GRPO Ray Trainer (recipe-local).

Approach:
- Subclass the upstream `RayPPOTrainer`.
- Override `.fit()` minimally to replace the *uniform repeat* with our *per-sample allocation*.
- Everything else (init_workers, datasets, reward/critic/actor WGs, metrics, logging) stays intact.

Key diffs from upstream:
- Before generation: compute allocation counts via EMA-based success estimator + knapsack DP,
  then expand the generation batch with `DataProto.select_idxs()`.
- After generation: expand the input batch with the SAME index list (instead of uniform repeat),
  then union with generated outputs, compute reward/values/advantages as usual.

This recipe targets GRPO family; other adv estimators still work (we pass through compute_advantage()).
"""

from __future__ import annotations
import uuid
import numpy as np
from typing import Optional

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,                      # base class
    compute_response_mask,
    compute_advantage,
    apply_kl_penalty,
)
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.metric_utils import reduce_metrics
from verl.trainer.ppo.utils import need_critic, need_reward_model
from verl.utils.debug import marked_timer

from recipe.knapsack_grpo.utils.success_estimator import SuccessRateEstimator
from recipe.knapsack_grpo.utils.allocator import allocate_with_extremes


class KnapsackRayTrainer(RayPPOTrainer):
    """Drop-in replacement of upstream RayPPOTrainer with knapsack allocation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ba_cfg = getattr(self.config.algorithm, "budget_alloc", None)
        # Default knobs if user didn't put them in YAML
        self._ba_enable = bool(ba_cfg and ba_cfg.get("enable", True))
        self._n_low = int(ba_cfg.get("n_low", 2)) if ba_cfg else 2
        self._n_up = int(ba_cfg.get("n_up", 128)) if ba_cfg else 128
        self._ema = float(ba_cfg.get("ema", 0.7)) if ba_cfg else 0.7
        self._prior_alpha = float(ba_cfg.get("prior_alpha", 1.0)) if ba_cfg else 1.0
        self._prior_beta  = float(ba_cfg.get("prior_beta", 1.0)) if ba_cfg else 1.0
        self._hard_share = float(ba_cfg.get("hard_fallback_share", 1.0)) if ba_cfg else 1.0
        self._easy_min_cover = bool(ba_cfg.get("easy_min_cover", True)) if ba_cfg else True
        self._override_total_budget = ba_cfg.get("override_total_budget", None) if ba_cfg else None

        # Online success-rate estimator (per uid)
        self._succ_est = SuccessRateEstimator(
            ema=self._ema, alpha=self._prior_alpha, beta=self._prior_beta
        )

    # -------- helpers

    @staticmethod
    def _ensure_uid(dp: DataProto) -> None:
        if "uid" not in dp.non_tensor_batch:
            dp.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(dp))], dtype=object
            )

    @staticmethod
    def _expand_by_index(dp: DataProto, idx: np.ndarray) -> DataProto:
        if isinstance(idx, list):
            idx = np.asarray(idx, dtype=np.int32)
        return dp.select_idxs(idx)

    def _make_alloc_indices(self, uids: list[str], default_n: int) -> list[int]:
        # Estimate p_i from EMA stats; total budget defaults to M * default_n
        p_hat = self._succ_est.estimate(uids)
        total_budget = (
            int(self._override_total_budget)
            if self._override_total_budget is not None
            else len(uids) * int(default_n)
        )
        alloc = allocate_with_extremes(
            p_hat,
            total_budget=total_budget,
            n_low=self._n_low,
            n_up=self._n_up,
            hard_fallback_share=self._hard_share,
            easy_min_cover=self._easy_min_cover,
        )
        # turn alloc -> flat idx list
        idx: list[int] = []
        for i, k in enumerate(alloc.tolist()):
            idx.extend([i] * int(k))
        if not idx:
            idx = list(range(len(uids)))
        return idx

    # -------- main loop (override)

    def fit(self):
        """
        Minimal override of upstream `.fit()`:
        we only replace the two `repeat()` points by index-based expansion.
        The rest logic remains the same spirit as upstream:
          - dataloaders/validation
          - rollout -> reward -> values -> advantages -> updates
          - metrics/logging/checkpoint hooks
        """
        from tqdm import tqdm
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking
        import torch

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
        )
        metrics = {}
        timing_raw = {}

        # Prepare data loaders (reuse upstream util)
        self._create_dataloader(
            train_dataset=getattr(self, "train_dataset", None),
            val_dataset=getattr(self, "val_dataset", None),
            collate_fn=getattr(self, "collate_fn", None),
            train_sampler=getattr(self, "train_sampler", None),
        )

        # Optionally pre-validation (keep upstream behavior minimal)
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            logger.log(data=val_metrics, step=self.global_steps)

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Knapsack-GRPO")
        self.global_steps += 1

        # ---- training steps
        train_iter = iter(self.train_dataloader)
        while self.global_steps <= self.total_training_steps:
            try:
                batch_dict = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch_dict = next(train_iter)

            # Make DataProto & ensure uid
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            self._ensure_uid(batch)

            # Build generation batch (upstream helper)
            gen_batch: DataProto = self._get_gen_batch(batch)
            gen_batch.meta_info["global_steps"] = self.global_steps

            # ==== our allocation =====
            if self._ba_enable:
                default_n = int(self.config.actor_rollout_ref.rollout.n)
                uids = list(batch.non_tensor_batch["uid"])
                alloc_idx = self._make_alloc_indices(uids, default_n)

                # Expand gen batch by allocation
                gen_batch = self._expand_by_index(gen_batch, alloc_idx)
                gen_batch.meta_info["alloc_idx_len"] = len(alloc_idx)
            else:
                # Fallback to uniform repeat (upstream behavior)
                gen_batch = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

            # ==== generate ====
            with marked_timer("gen", timing_raw, color="red"):
                if not self.async_rollout_mode:
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                else:
                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

                timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                gen_batch_output.meta_info.pop("timing", None)

            # ==== batch align (inputs x responses) ====
            if self._ba_enable:
                # Expand the ORIGINAL batch with the same indices used for gen_batch
                # We stored alloc length, but we re-compute idx (cheap) for clarity
                default_n = int(self.config.actor_rollout_ref.rollout.n)
                alloc_idx = self._make_alloc_indices(list(batch.non_tensor_batch["uid"]), default_n)
                batch = self._expand_by_index(batch, alloc_idx)
            else:
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

            # Union inputs & outputs
            batch = batch.union(gen_batch_output)

            # Ensure response_mask present
            if "response_mask" not in batch.batch:
                batch.batch["response_mask"] = compute_response_mask(batch)

            # ==== Reward ====
            with marked_timer("reward", timing_raw, color="yellow"):
                # Optional external RM score
                if self.use_rm and "rm_scores" not in batch.batch:
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                    reward_tensor, reward_extra_infos_dict = future_reward.result()
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

            # ==== KL penalty in reward (optional, same as upstream) ====
            if self.config.algorithm.get("use_kl_in_reward", False):
                batch, kl_metrics = apply_kl_penalty(
                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                )
                metrics.update(kl_metrics)
            else:
                # standard path: token_level_scores -> token_level_rewards
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # ==== Log-prob (old) ====
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                batch = batch.union(old_log_prob)

            # ==== Value function ====
            if need_critic(self.config):
                with marked_timer("values", timing_raw, color="cyan"):
                    values = self.critic_wg.compute_critic_value(batch)
                    batch = batch.union(values)

            # ==== Advantages ====
            with marked_timer("adv", timing_raw, color="brown"):
                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.get("gamma", 1.0),
                    lam=self.config.algorithm.get("lam", 1.0),
                    num_repeat=1,  # allocation already expands per sample
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    config=self.config.algorithm,
                )

            # ==== Critic update ====
            if need_critic(self.config):
                critic_output = self.critic_wg.update_critic(batch)
                critic_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_metrics)

            # ==== Actor update ====
            if self.hybrid_engine and self.config.actor_rollout_ref.actor.get("use_pf_ppo", False):
                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                actor_output = self.actor_rollout_wg.update_actor(batch)
            else:
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_metrics)

            # ==== Success-rate stats update (for next allocation) ====
            try:
                if "acc" in (reward_extra_infos_dict or {}):
                    succ = np.array(reward_extra_infos_dict["acc"], dtype=np.int32)
                else:
                    # fallback: sum of token rewards > 0 as success
                    seq_reward = batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().numpy()
                    succ = (seq_reward > 0).astype(np.int32)
                self._succ_est.update_batch(list(batch.non_tensor_batch["uid"]), succ)
            except Exception:
                pass

            # ==== Logging / book-keeping (lightweight) ====
            metrics.update({"global_step": int(self.global_steps)})
            logger.log(metrics, step=self.global_steps)

            self.global_steps += 1
            progress_bar.update(1)

        progress_bar.close()