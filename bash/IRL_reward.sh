#!/bin/bash
#SBATCH --partition=hopper-prod
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-gpu=10
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --requeue
#SBATCH --array=0-11 # %25
#SBATCH --exclusive

#module load cuda/12.2
#export WANDB_TAGS=refactor-chosen-rejected3,no-tag-$(git rev-parse --short HEAD)
#MODELS=("EleutherAI/pythia-6.9b-deduped" "EleutherAI/pythia-2.8b-deduped" "EleutherAI/pythia-1b-deduped")
MODELS=("EleutherAI/pythia-1b-deduped")
SEEDS=(44413 55513 66613 77713)
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
MODEL=${MODELS[$MODEL_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

echo "Running task $SLURM_ARRAY_TASK_ID with SEED: $SEED and MODEL: $MODEL"

if [ -z "$SEED" ]; then
    SEED=1
fi
if [ -z "$MODEL" ]; then
    # MODEL=EleutherAI/pythia-6.9b-deduped
    MODEL=EleutherAI/pythia-2.8b-deduped
    # MODEL=EleutherAI/pythia-1b-deduped
    # MODEL=EleutherAI/pythia-410m-deduped
fi
if [ -z "$LR" ]; then
    LR=3e-6
fi


LR=3e-6


REWARD_MODEL_PATH=models/$MODEL/irl_second_round/reward_model/warm_start_demos_vs_first_round_policy_with_demos_vs_sft_generation_$SEED/$LR
POLICY_MODEL_PATH=models/$MODEL/policy_model_$SEED


if [ "$MODEL" = "EleutherAI/pythia-1b-deduped" ]; then
    local_rollout_forward_batch_size=64
    gradient_accumulation_steps=4
    local_micro_batch_size=16
    local_eval_batch_size=8
fi
if [ "$MODEL" = "EleutherAI/pythia-2.8b-deduped" ]; then
    local_rollout_forward_batch_size=32
    gradient_accumulation_steps=16
    local_micro_batch_size=4
    local_eval_batch_size=2
fi
if [ "$MODEL" = "EleutherAI/pythia-6.9b-deduped" ]; then
    local_rollout_forward_batch_size=2
    gradient_accumulation_steps=64
    local_micro_batch_size=1
    local_eval_batch_size=1
fi

generated_data_path=generated_data/irl_first_round_policy_vs_demonstration/models_EleutherAI_pythia-1b-deduped_irl_first_round_policy_ppo_finetuning_rm_demos_vs_sft_generation_44413_update_1250_groundtruth_score_1.9166_vs_demos/train_dataset
generated_data_path_2=generated_data/sft_vs_demonstration/models_EleutherAI_pythia-1b-deduped_demonstration_trained_sft_44413_vs_generated_data_sft_demonstration_vwxyzjn_EleutherAI_pythia-6.9b-deduped__ppo_left_padding_new_nowhiten_reward__tldr/train_dataset
#generated_data_path_3=generated_data/irl_first_round_policy_vs_demonstration/models_EleutherAI_pythia-1b-deduped_first_round_irl_policy_ppo_finetuning_44413_update_20_groundtruth_score_1.7166_vs_generated_data_sft_demonstrations_vwxyzjn_EleutherAI_pythia-6.9b-deduped__ppo_left_padding_new_nowhiten_reward__tldr/train_dataset

constructed_validation_path=generated_data/irl_first_round_policy_vs_demonstration/models_EleutherAI_pythia-1b-deduped_irl_first_round_policy_ppo_finetuning_rm_demos_vs_sft_generation_44413_update_1250_groundtruth_score_1.9166_vs_demos/validation_dataset
constructed_validation_path_2=generated_data/sft_vs_demonstration/models_EleutherAI_pythia-1b-deduped_demonstration_trained_sft_44413_vs_generated_data_sft_demonstration_vwxyzjn_EleutherAI_pythia-6.9b-deduped__ppo_left_padding_new_nowhiten_reward__tldr/validation_dataset
#constructed_validation_path_3=generated_data/irl_first_round_policy_vs_demonstration/models_EleutherAI_pythia-1b-deduped_first_round_irl_policy_ppo_finetuning_44413_update_20_groundtruth_score_1.7166_vs_generated_data_sft_demonstrations_vwxyzjn_EleutherAI_pythia-6.9b-deduped__ppo_left_padding_new_nowhiten_reward__tldr/validation_dataset

SFT_MODEL_PATH=models/EleutherAI/pythia-1b-deduped/demonstration_trained_sft_44413
#revision=sft__44413__1708611267
REWARD_MODEL_INITIALIZATION=models/EleutherAI/pythia-1b-deduped/irl_first_round/reward_model/sft_generations_vs_demos_44413/3e-6/update_4801_0.568


poetry run accelerate launch --config_file deepspeed.yaml \
    summarize_from_feedback_details/IRL_reward.py \
    --base_model=$MODEL \
    --sft_model_path=$SFT_MODEL_PATH \
    --reward_model_path=$REWARD_MODEL_INITIALIZATION \
    --generated_dataset=$generated_data_path \
    --generated_dataset_2=$generated_data_path_2 \
    --constructed_validation_dataset=$constructed_validation_path \
    --constructed_validation_dataset_2=$constructed_validation_path_2 \
    --lr=$LR \
    --total_episodes=10000 \
    --deepspeed \
    --run_eval \
    --track \
    --output_dir=$REWARD_MODEL_PATH \
    --local_eval_batch_size=$local_eval_batch_size \
    --seed=$SEED