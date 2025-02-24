# IRLAlignment
## Get started

Install the dependencies

```
pip install -r requirements.txt
pip install poetry
```

## Step 0: Generate Demonstration

Demonstration is generated by a well-trained policy model vwxyzjn/EleutherAI_pythia-6.9b-deduped__
reward__tldr(https://huggingface.co/vwxyzjn/EleutherAI_pythia-6.9b-deduped__reward__tldr/tree/reward__44413__1706651113)

```
./bash/sft_data_generation.sh 
```
After running this, our demonstration is generated in /generated_data folder.
Currently, we use a 2.8B model to generate demonstrations on an A40 GPU to prevent OOM issues. You can configure the model for demonstration generation in sft_data_generation.bash.

## Step 1: IRL Training

Then we run
```
./bash/IRL_Pipeline.sh
```
Our IRL pipeline includes four steps:
### Step 1.1: SFT.
### Step 1.2: Generate Demonstration-agent pairs for reward update.
### Step 1.3: Reward Update.
### Step 1.4: Using PPO to update policy and go back to step 1.2.

Details are in the bash.

## Evaluations

We evaluate the proposed IRL method from the quality of the estimated reward models and policy models. For the reward model, we evaluate the reward accuracy on a hold-out TL;DR preference dataset. For the policy model, we evaluate the performance of the model according to the reward score from a hold-out 6.9B reward model and also the ChatGPT-evaluted win-rate compared with a high-quality reference dataset generated from a public 6.9B PPO model (vwxyzjn/EleutherAI_pythia-6.9b-deduped__ppo_left_padding_new_nowhiten_reward__tldr)

## Acknowledge
* [The N+ Implementation Details of RLHF with PPO: A
Case Study on TL;DR Summarization]([https://www.baidu.com](https://github.com/vwxyzjn/summarize_from_feedback_details))
