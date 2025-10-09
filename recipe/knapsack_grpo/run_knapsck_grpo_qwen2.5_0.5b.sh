#!/usr/bin/env bash
set -euo pipefail
set -x

# ===== 数据路径（按需改）=====
gsm8k_train_path="$HOME/data/gsm8k/train.parquet"
gsm8k_test_path="$HOME/data/gsm8k/test.parquet"
math_train_path="$HOME/data/math/train.parquet"
math_test_path="$HOME/data/math/test.parquet"

train_files="[''"$gsm8k_train_path"'',''"$math_train_path"'']"
test_files="[''"$gsm8k_test_path"'',''"$math_test_path"'']"

# ===== 入口切换到 recipe.knapsack_grpo =====
# 注意：该入口会在运行时替换 RayPPOTrainer 为 KnapsackRayTrainer，
# 复用 verl.trainer.main_ppo 的 run_ppo 流水线。
python3 -m recipe.knapsack_grpo.main_knapsack_grpo \
    --config-path recipe/knapsack_grpo/config \
    --config-name knapsack_grpo_trainer \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='knapsack_grpo_example_gsm8k_math' \
    trainer.experiment_name='qwen2_7b_knapsack_grpo' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    \
    # ====== Knapsack 关键开关与参数（可按需覆盖）======
    algorithm.budget_alloc.enable=true \
    algorithm.budget_alloc.n_low=2 \
    algorithm.budget_alloc.n_up=128 \
    algorithm.budget_alloc.ema=0.7 \
    algorithm.budget_alloc.prior_alpha=1.0 \
    algorithm.budget_alloc.prior_beta=1.0 \
    algorithm.budget_alloc.override_total_budget=null \
    algorithm.budget_alloc.hard_fallback_share=1.0 \
    algorithm.budget_alloc.easy_min_cover=true \
    \
    $@