#!/bin/bash
#SBATCH --partition=hopper-prod
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-gpu=10
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --requeue
#SBATCH --array=0-7 # %25
#SBATCH --exclusive

#module load cuda/12.2
#export WANDB_TAGS=refactor-chosen-rejected3,no-tag-$(git rev-parse --short HEAD)
#MODELS=("EleutherAI/pythia-2.8b-deduped" "EleutherAI/pythia-1b-deduped")

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
    MODEL=EleutherAI/pythia-6.9b-deduped
    # MODEL=EleutherAI/pythia-2.8b-deduped
    # MODEL=EleutherAI/pythia-1b-deduped
    # MODEL=EleutherAI/pythia-410m-deduped
fi
if [ -z "$LR" ]; then
    LR=3e-6
fi


#REWARD_MODEL_PATH=models/$MODEL/reward_model_tldr_preference
POLICY_MODEL_PATH=models/$MODEL/policy_model_irl_first_round_with_reward_update_4001_sft_start_$SEED
DPO_POLICY_MODEL_PATH=models/$MODEL/dpo_policy_model_$SEED

if [ "$MODEL" = "EleutherAI/pythia-1b-deduped" ]; then
    local_rollout_forward_batch_size=32
    gradient_accumulation_steps=16
    local_micro_batch_size=4
    local_eval_batch_size=8
fi
if [ "$MODEL" = "EleutherAI/pythia-2.8b-deduped" ]; then
    local_rollout_forward_batch_size=32
    gradient_accumulation_steps=16
    local_micro_batch_size=4
    local_eval_batch_size=1
fi
if [ "$MODEL" = "EleutherAI/pythia-6.9b-deduped" ]; then
    local_rollout_forward_batch_size=2
    gradient_accumulation_steps=64
    local_micro_batch_size=1
    local_eval_batch_size=1
fi

REWARD_BASE=EleutherAI/pythia-6.9b-deduped
EVALUATION_REWARD_MODEL_PATH=vwxyzjn/EleutherAI_pythia-6.9b-deduped__reward__tldr
REWARD_REVISION=reward__44413__1708628552

# SFT_BASE=EleutherAI/pythia-6.9b-deduped
# SFT_MODEL_PATH=vwxyzjn/EleutherAI_pythia-6.9b-deduped__ppo_left_padding_new_nowhiten_reward__tldr
# SFT_REVISION=ppo_left_padding_new_nowhiten_reward__44413__1710465193

SFT_BASE=EleutherAI/pythia-2.8b-deduped
SFT_MODEL_PATH=vwxyzjn/EleutherAI_pythia-2.8b-deduped__ppo_left_padding_new_nowhiten_reward__tldr
SFT_REVISION=ppo_left_padding_new_nowhiten_reward__44413__1710356835

DATA_FOLDER_PATH=generated_data/sft_model_evaluation

poetry run accelerate launch --config_file deepspeed.yaml \
    summarize_from_feedback_details/sft_data_generation.py \
    --local_micro_batch_size=$local_micro_batch_size \
    --sft_base_model=$SFT_BASE \
    --sft_model_path=$SFT_MODEL_PATH \
    --sft_model_revision=$SFT_REVISION \
    --evaluation_rm_base=$REWARD_BASE \
    --evaluation_rm_path=$EVALUATION_REWARD_MODEL_PATH \
    --evaluation_rm_revision=$REWARD_REVISION \
    --lr=$LR \
    --deepspeed \
    --data_path=$DATA_FOLDER_PATH \
    --local_eval_batch_size=4 \
    --seed=$SEED