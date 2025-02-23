set -x
set -e

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
    MODEL=EleutherAI/pythia-1b-deduped
fi
if [ -z "$LR" ]; then
    LR=3e-6
fi


if [ "$MODEL" = "EleutherAI/pythia-1b-deduped" ]; then
    local_rollout_forward_batch_size=32
    gradient_accumulation_steps=16
    local_micro_batch_size=4
    local_eval_batch_size=8
fi

if [ "$MODEL" = "EleutherAI/pythia-6.9b-deduped" ]; then
    local_rollout_forward_batch_size=2
    gradient_accumulation_steps=64
    local_micro_batch_size=1
    local_eval_batch_size=1
fi

demonstration_data_folder=generated_data/sft_model_evaluation/vwxyzjn_EleutherAI_pythia-1b-deduped__ppo_left_padding_new_nowhiten_reward__tldr/train_dataset/

############################################################################################################
############################################################################################################
############################################################################################################
SFT_MODEL_PATH=models/EleutherAI/pythia-1b-deduped/demonstration_trained_sft_44413
# train a SFT model with the demonstrations
# poetry run accelerate launch --config_file deepspeed.yaml \
#     summarize_from_feedback_details/sft.py \
#     --base_model=$MODEL \
#     --lr=$LR \
#     --data_folder_path=$demonstration_data_folder \
#     --deepspeed \
#     --track \
#     --output_dir=$SFT_MODEL_PATH \
#     --seed=$SEED

############################################################################################################
############################################################################################################
############################################################################################################

# start IRL Training

epoch=11
starting_epoch=0

EVALUATION_REWARD_MODEL_BASE=EleutherAI/pythia-6.9b-deduped
EVALUATION_REWARD_MODEL_PATH=vwxyzjn/EleutherAI_pythia-6.9b-deduped__reward__tldr
EVALUATION_REWARD_REVISION=reward__44413__1708628552

REFERENCE_DATASET_FOLDER_PATH=generated_data/sft_model_evaluation/vwxyzjn_EleutherAI_pythia-2.8b-deduped__ppo_left_padding_new_nowhiten_reward__tldr

GENERATION_MODEL_PATH=models/EleutherAI/pythia-1b-deduped/demonstration_trained_sft_44413

PRETRAINED_MODEL_PATH=models/EleutherAI/pythia-1b-deduped/demonstration_trained_sft_44413

for ((i=starting_epoch; i<epoch; i++))
do
    echo "Processing epoch $i for irl"

    # generate pairwise comparisons for demonstrations vs SFT model generations

    MODEL_BASE=$MODEL

    #DATA_FOLDER_PATH=generated_data/reference_as_demonstration/$MODEL/IRL_pairwise_comparisons_iter_$i
    #DATA_FOLDER_PATH=generated_data/reference_as_demonstration/$MODEL/IRL_pairwise_comparisons_iter_$i
    DATA_FOLDER_PATH=generated_data/$MODEL/IRL_pairwise_comparisons_iter_$i

    #DATA_FOLDER_PATH=generated_data/EleutherAI/pythia-1b-deduped/IRL_pairwise_comparisons_iter_4_concatenated

    poetry run accelerate launch --config_file deepspeed.yaml \
        summarize_from_feedback_details/data_pairing.py \
        --local_micro_batch_size=$local_micro_batch_size \
        --base_model=$MODEL_BASE \
        --model_path=$GENERATION_MODEL_PATH \
        --evaluation_rm_base=$EVALUATION_REWARD_MODEL_BASE \
        --evaluation_rm_path=$EVALUATION_REWARD_MODEL_PATH \
        --evaluation_rm_revision=$EVALUATION_REWARD_REVISION \
        --lr=$LR \
        --deepspeed \
        --reference_dataset_folder=$REFERENCE_DATASET_FOLDER_PATH \
        --data_path=$DATA_FOLDER_PATH \
        --local_eval_batch_size=16 \
        --seed=$SEED
        
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################

    if [ $i -gt 1 ]; then
        # concatenate the new dataset with the previous dataset
        # if i == 2
        if [ $i -eq 2 ]; then
            previous_data_path=generated_data/$MODEL/IRL_pairwise_comparisons_iter_$((i-1))
        else 
            previous_data_path=generated_data/$MODEL/IRL_pairwise_comparisons_iter_$((i-1))_concatenated
        fi
        CONCATENATED_DATA_FOLDER_PATH=generated_data/$MODEL/IRL_pairwise_comparisons_iter_${i}_concatenated
        python3 summarize_from_feedback_details/preference_data_concatenation.py $previous_data_path $DATA_FOLDER_PATH $CONCATENATED_DATA_FOLDER_PATH
        DATA_FOLDER_PATH=$CONCATENATED_DATA_FOLDER_PATH
    fi

    ############################################################################################################
    ############################################################################################################
    ############################################################################################################

    # train a reward model with the pairwise comparisons

    generated_data_path=$DATA_FOLDER_PATH/train_dataset

    constructed_validation_path=$DATA_FOLDER_PATH/validation_dataset

    REWARD_MODEL_PATH=models/$MODEL/automatic/irl_iter_$i/reward_model

    REWARD_RUN_NAME=DEVICE2_irl_reward_iter_${i}_model_${MODEL}

    poetry run accelerate launch --config_file deepspeed.yaml \
        summarize_from_feedback_details/IRL_reward.py \
        --base_model=$MODEL \
        --sft_model_path=$SFT_MODEL_PATH \
        --generated_dataset=$generated_data_path \
        --constructed_validation_dataset=$constructed_validation_path \
        --lr=$LR \
        --run_name=$REWARD_RUN_NAME \
        --total_episodes=10000 \
        --deepspeed \
        --track \
        --output_dir=$REWARD_MODEL_PATH \
        --local_eval_batch_size=$local_eval_batch_size \
        --seed=$SEED

    ############################################################################################################
    ############################################################################################################
    ############################################################################################################

    # then train a policy model with the reward model by PPO
    IRL_REWARD_MODEL_PATH=$REWARD_MODEL_PATH/final
    #IRL_REWARD_MODEL_PATH=models/EleutherAI/pythia-1b-deduped/automatic/irl_iter_4/reward_model/update_43201_0.595

    PPO_GENERATIONS=generated_data/ppo_generation_iter_$i
    PPO_EPOCH_NUM=4
    POLICY_MODEL_PATH=models/$MODEL/automatic/irl_iter_$i/policy_model_training_epoch_${PPO_EPOCH_NUM}_LR_${LR}
    
    PPO_RUN_NAME=DEVICE2_irl_ppo_iter_${i}

    poetry run accelerate launch --config_file deepspeed.yaml \
        summarize_from_feedback_details/ppo.py \
        --local_rollout_forward_batch_size=$local_rollout_forward_batch_size \
        --gradient_accumulation_steps=$gradient_accumulation_steps \
        --local_micro_batch_size=$local_micro_batch_size \
        --local_eval_batch_size=$local_eval_batch_size \
        --base_model=$MODEL \
        --sft_model_path=$PRETRAINED_MODEL_PATH \
        --reward_model_path=$IRL_REWARD_MODEL_PATH \
        --evaluation_rm_path=$EVALUATION_REWARD_MODEL_PATH \
        --evaluation_rm_revision=$EVALUATION_REWARD_REVISION \
        --lr=$LR \
        --ppo.noptepochs=$PPO_EPOCH_NUM \
        --run_name=$PPO_RUN_NAME \
        --deepspeed \
        --track \
        --output_dir=$POLICY_MODEL_PATH \
        --data_path=$PPO_GENERATIONS \
        --seed=$SEED

    ############################################################################################################
    ############################################################################################################
    ############################################################################################################

    GENERATION_MODEL_PATH=$POLICY_MODEL_PATH/final

    PRETRAINED_MODEL_PATH=$GENERATION_MODEL_PATH

done