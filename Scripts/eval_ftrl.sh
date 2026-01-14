set -x
unset PYTORCH_CUDA_ALLOC_CONF
ulimit -c 0

export VLLM_USE_V1=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export MASTER_PORT=29505
export PYTHONPATH=Code:$PYTHONPATH

export WANDB_API_KEY=""
export WANDB_BASE_URL="https://api.bandw.top"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

train_files="Data/train.parquet"
test_files="Data/test.parquet"
model=" "

rollout_mode="sync"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.max_prompt_length=7000 \
    data.max_response_length=23000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_key=messages \
    data.system_style=Qwen3 \
    data.enable_thinking=True \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=tool \
    custom_reward_function.path=Code/verl/utils/reward_score/tool.py \
    custom_reward_function.name=compute_process_KM \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='MatchTIR' \
    trainer.experiment_name='MatchTIR-8B' \
    trainer.val_before_train=True \
    +trainer.val_only=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=8 \
    trainer.test_freq=4 \
    trainer.total_epochs=3 $@
