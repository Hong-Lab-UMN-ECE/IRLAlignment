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
    # MODEL=EleutherAI/pythia-6.9b-deduped
    MODEL=EleutherAI/pythia-2.8b-deduped
    # MODEL=EleutherAI/pythia-1b-deduped
    # MODEL=EleutherAI/pythia-410m-deduped
fi
if [ -z "$LR" ]; then
    LR=3e-6
fi


#REWARD_MODEL_PATH=models/$MODEL/reward_model_tldr_preference
POLICY_MODEL_PATH=models/$MODEL/irl_second_round/policy_ppo_finetuning_rm_first_round_checkpoint_update_24001_0.585_$SEED

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
    local_rollout_forward_batch_size=4
    gradient_accumulation_steps=64
    local_micro_batch_size=1
    local_eval_batch_size=1
fi

LR=1e-6

IRL_REWARD_MODEL_PATH=models/EleutherAI/pythia-1b-deduped/irl_second_round/reward_model/sft_init_demos_vs_first_round_policy_with_demos_vs_sft_generation_44413/3e-6/update_24001_0.585

EVALUATION_REWARD_BASE_MODEL=EleutherAI/pythia-6.9b-deduped
EVALUATION_REWARD_MODEL_PATH=vwxyzjn/EleutherAI_pythia-6.9b-deduped__reward__tldr
REVISION=reward__44413__1708628552

POLICY_STARTING_MODEL_PATH=models/EleutherAI/pythia-1b-deduped/irl_first_round/policy_ppo_finetuning_rm_demos_vs_sft_generation_44413/update_1250_groundtruth_score_1.9166
#POLICY_STARTING_MDOEL_REVISION=sft__44413__1708611267

DATA_PATH=generated_data/second_round_ppo_generation_44413

poetry run accelerate launch --config_file deepspeed.yaml \
    summarize_from_feedback_details/ppo.py \
    --local_rollout_forward_batch_size=$local_rollout_forward_batch_size \
    --gradient_accumulation_steps=$gradient_accumulation_steps \
    --local_micro_batch_size=$local_micro_batch_size \
    --local_eval_batch_size=$local_eval_batch_size \
    --base_model=$MODEL \
    --sft_model_path=$POLICY_STARTING_MODEL_PATH \
    --reward_model_path=$IRL_REWARD_MODEL_PATH \
    --evaluation_rm_path=$EVALUATION_REWARD_MODEL_PATH \
    --evaluation_rm_revision=$REVISION \
    --lr=$LR \
    --deepspeed \
    --run_eval \
    --track \
    --output_dir=$POLICY_MODEL_PATH \
    --data_path=$DATA_PATH \
    --seed=$SEED
