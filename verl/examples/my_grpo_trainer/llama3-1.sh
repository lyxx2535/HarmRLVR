set -x

unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export WANDB_API_KEY=""

ray stop

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=../data/train/airbench-64.parquet \
    data.val_files=../data/train/mix_10.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=../ckpts/Llama-3.1-8B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.actor.checkpoint.save_contents=['hf_model'] \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=batch \
    custom_reward_function.path=my_reward/outcome_reward.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='llama3-1' \
    trainer.experiment_name='air64' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=100 $@ \
    trainer.rollout_data_dir=logs/llama3-1/air64/rollout \
    trainer.validation_data_dir=logs/llama3-1/air64/val

