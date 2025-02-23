import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Literal, Optional
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import broadcast, gather_object
from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import HfApi
from rich.console import Console
from rich.pretty import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)
import time

torch.set_printoptions(precision=4, sci_mode=False)
api = HfApi()
INVALID_LOGPROB = 1.0


@dataclass
class PpoHParams:
    nminibatches: int = 1
    noptepochs: int = 4
    vf_coef: float = 0.1
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    gamma: float = 1
    lam: float = 0.95
    whiten_rewards: bool = False
    kl_coef: float = 0.05


@dataclass
class Args:
    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    cuda: bool = True
    """Whether to use cuda if available."""
    run_name: Optional[str] = None
    """a unique name of this run"""
    load_from_cache_file: bool = False
    """Whether to load data from the local cache file in `dataset.map`"""
    deepspeed: bool = False
    """Whether to use deepspeed to train the model"""
    print_sample_output_freq: int = 10
    """How often to print sample output"""
    model_save_freq: int = 50
    """How often to save the model"""
    run_eval: bool = False
    """Whether to run evaluation"""

    # optimizer args
    eps: float = 1e-5
    """the epsilon value for the optimizer"""
    lr: float = 3e-6
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    scheduler: str = "cosine"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""

    # various batch sizes
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    num_train_epochs: int = 1
    """Number of epochs to train"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    gradient_accumulation_steps: int = 1
    """The number of gradient accumulation steps"""
    local_micro_batch_size: Optional[int] = 1
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    total_episodes: Optional[int] = 1000000
    """The total number of episodes in the dataset"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    nminibatches: int = 1
    """Number of minibatches to split a batch into"""
    local_mini_batch_size: Optional[int] = None
    """the mini batch size per GPU"""
    mini_batch_size: Optional[int] = None
    """the mini batch size across GPUs"""
    local_eval_batch_size: int = 1
    """per rank eval batch size"""
    local_rollout_forward_batch_size: int = 1
    """per rank no grad forward pass in the rollout phase"""

    # other args
    base_model: str = "EleutherAI/pythia-160m"
    """the name of the pretrained model to use"""
    query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"
    """the query dataset"""
    response_length: int = 53
    """the length of the response"""
    truncate_token: Literal["eos"] = "eos"
    """the truncate token"""
    truncate_token_id: Optional[int] = None
    """the truncation token id"""
    temperature: float = 0.7
    """the sampling temperature"""
    penalty_reward_value: int = -1
    """the reward value for responses that do not contain `truncate_token_id`"""
    non_eos_penalty: bool = True
    """whether to penalize responses that do not contain `truncate_token_id`"""
    offload: bool = False
    """Whether to offload ref policy and reward model to CPU"""
    reward_model_path: str = ""
    """the path to the reward model"""
    evaluation_rm_base: str = ""
    """the base model of the evaluation reward model"""
    evaluation_rm_path: str = ""
    """the path to the evaluation reward model"""
    evaluation_rm_revision: Optional[str] = None
    """the revision of the evaluation reward model"""
    sft_base_model: str = "EleutherAI/pythia-160m"
    """the base model of the sft model"""
    sft_model_path: str = "EleutherAI/pythia-160m"
    """the path to the sft model"""
    sft_model_revision: str = ""
    """the revision of the sft model"""


    # wandb and HF tracking configs
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "tldr_summarize"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    push_to_hub: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """the user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """the id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """the revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """the url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: str = "models/ppo_model"
    """Where to save the model"""
    data_path: Optional[str] = None
    """The path to the data"""
    ppo: PpoHParams = field(default_factory=PpoHParams)


def parse_args() -> tuple[Args, Accelerator]:
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.world_size = accelerator.num_processes
    args.local_batch_size = args.local_micro_batch_size * args.gradient_accumulation_steps * args.nminibatches
    args.micro_batch_size = int(args.local_micro_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
    args.mini_batch_size = exact_div(args.batch_size, args.nminibatches)
    args.local_mini_batch_size = exact_div(args.local_batch_size, args.nminibatches)
    if args.ppo.whiten_rewards:
        assert (
            args.local_mini_batch_size >= 8
        ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
    # `per_rank_rollout_batch_size` is our `args.local_batch_size`
    # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
    args.num_updates = args.total_episodes // args.batch_size
    time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
    time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
    args.run_name = f"IRL_First_Round_{args.exp_name}__{args.seed}__{time_int}"
    if args.push_to_hub:
        if args.hf_repo_id is None: # auto-generate one
            args.hf_repo_id = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
        if args.hf_entity is None:  # find the current user
            args.hf_entity = api.whoami()["name"]
        if "/" not in args.hf_repo_id: # prepend the current user
            args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
    return args, accelerator


# taken from https://github.com/vwxyzjn/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/utils.py#L99
def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask, False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = "EleutherAI/pythia-160m",
        base_config: PretrainedConfig = AutoConfig.from_pretrained("EleutherAI/pythia-160m"),
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias


class ScalarModel(PreTrainedModel):
    config_class = ScalarModelConfig

    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        def base_to_name(base):
            if "6.9b" in base:
                return "vwxyzjn/EleutherAI_pythia-6.9b-deduped__sft__tldr"
            elif "1b" in base:
                return "vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr"
            else:
                raise NotImplementedError()
            
        self.lm_backbone = AutoModel.from_pretrained(
            base_to_name(config.base_model),
            config=self.config.base_config,
            trust_remote_code=True,
        )
        self.scalar_head = layer_init(
            nn.Linear(self.config.hidden_size, 1),
            std=1 / np.sqrt(self.config.hidden_size + 1),
        )

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        reward = self.scalar_head(output.hidden_states[-1]) - self.config.bias
        return reward


def get_reward(model, query_responses, tokenizer, context_length):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths].squeeze(-1),
        sequence_lengths,
    )


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, critic) -> None:
        super().__init__()
        self.policy = policy
        self.critic = critic

    def forward(self, **kwargs):
        return self.policy(**kwargs), self.critic(**kwargs)


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q


def generate(lm_backbone, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # generation collapsed if this was turned on. TODO: why does generation collapse with this?
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def truncate_response(args, tokenizer, responses):
    trunc_idxs = first_true_indices(responses == args.truncate_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, tokenizer.pad_token_id)
    return postprocessed_responses


def forward(model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


def evaluate(evaluation_reward_model, policy, tokenizer, dataloader, generation_config, sampling=True):
    eval_storage = defaultdict(list)
    with torch.no_grad():
        for data in tqdm(dataloader):
            queries = data["query_token"]
            #reference_response_token = data["reference_response_token"]
            context_length = queries.shape[1]
            query_reference_model_responses, _ = generate(
                policy,
                queries,
                tokenizer,
                generation_config,
            )
            reference_response = query_reference_model_responses[:, context_length:]
            reference_response_token = truncate_response(args, tokenizer, reference_response)
            query_reference_responses = torch.cat((queries, reference_response_token), 1)
            # query_reference_responses = torch.cat((data["query_token"], data["reference_response_token"]), dim=1)
            #_, reference_score, _ = get_reward(reward_model, query_reference_responses, tokenizer, queries.shape[1])
            _, eval_rm_reference_score, _ = get_reward(evaluation_reward_model, query_reference_responses, tokenizer, queries.shape[1])

            eval_storage["query_token"].extend(queries.cpu().numpy())

            eval_storage["reference_response_token"].extend(reference_response_token.cpu().numpy())
            eval_storage["query_reference_response_token"].extend(query_reference_responses.cpu().numpy())

            #eval_storage["reference_score"].append(reference_score)
            eval_storage["eval_rm_reference_score"].append(eval_rm_reference_score)

            if sampling:
                break

    eval_storage["query"] = tokenizer.batch_decode(eval_storage["query_token"], skip_special_tokens=True)
    eval_storage["reference_response"] = tokenizer.batch_decode(eval_storage["reference_response_token"])
    eval_reference_rm_score = torch.cat(eval_storage["eval_rm_reference_score"]).float().cpu().numpy().tolist()

    eval_df = pd.DataFrame(
        {
            "query": gather_object(eval_storage["query"]),
            "query_token": gather_object(eval_storage["query_token"]),
            "reference_responses": gather_object(eval_storage["reference_response"]),
            "query_reference_response_token": gather_object(eval_storage["query_reference_response_token"]),
            "eval_rm_reference_scores": gather_object(eval_reference_rm_score),
        }
    )

    return eval_storage, eval_df


if __name__ == "__main__":
    args, accelerator = parse_args()
    local_seed = args.seed + accelerator.process_index * 100003  # Prime
    print("=====================================================")
    # load dataset
    dataset = load_dataset(args.query_dataset, split="train")
    dataset = dataset.with_format("torch", columns=["query_token", "reference_response_token"])
    dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)
    eval_dataloaders = {}

    for split in ["train", "validation", "test"]:
        eval_dataset = load_dataset(args.query_dataset, split=split)
        eval_dataset = eval_dataset.with_format("torch", columns=["query_token", "reference_response_token"])
        eval_dataloaders[split] = DataLoader(eval_dataset, batch_size=args.local_eval_batch_size)

    tokenizer = AutoTokenizer.from_pretrained(
        args.sft_base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if args.truncate_token == "eos":
        args.truncate_token_id = tokenizer.eos_token_id

    console = Console(force_terminal=True)
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    writer.add_histogram = lambda x, y, z: None
    if accelerator.is_main_process:
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=asdict(args),
                name=args.run_name,
                save_code=True,
            )
            file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
            wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
        writer = SummaryWriter(f"runs/{args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = accelerator.device
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True

    model_config = AutoConfig.from_pretrained(args.evaluation_rm_base)
    scalar_model_config = ScalarModelConfig(
        base_model=args.evaluation_rm_base,
        base_config=model_config,
        hidden_size=model_config.hidden_size,
    )

    evaluation_reward_model: PreTrainedModel = ScalarModel.from_pretrained(
        args.evaluation_rm_path,
        revision=args.evaluation_rm_revision,
        trust_remote_code=True,
    )
    
    accelerator.print('evaluation_reward_model loaded')

    policy_config = AutoConfig.from_pretrained(args.sft_base_model)

    if len(args.sft_model_revision) > 0:
        model = AutoModelForCausalLM.from_pretrained(args.sft_model_path, revision=args.sft_model_revision, config=policy_config, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.sft_model_path, config=policy_config, trust_remote_code=True)
        
    for module in [model, evaluation_reward_model]:
        disable_dropout(module)

    model.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    model.generation_config.pad_token_id = None  # generate tokens without truncation / padding

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)

    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare` 
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c 
    torch.manual_seed(args.seed)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.eval()
    accelerator.print("===the model is in eval mode===")
    eval_dataloaders = {split: accelerator.prepare(eval_dataloader) for split, eval_dataloader in eval_dataloaders.items()}
    torch.manual_seed(local_seed)  # reset the local seed again

    def repeat_generator():
        while True:
            yield from dataloader

    iter_dataloader = iter(repeat_generator())
    if args.deepspeed:
        import deepspeed

        deepspeed_states = AcceleratorState().deepspeed_plugin
        deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.local_micro_batch_size

        eval_ds_config = {
            "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"],
            "bf16": {"enabled": True},
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }
        # if args.offload or args.base_model == "EleutherAI/pythia-6.9b-deduped":
        #     deepspeed_states.deepspeed_config["checkpoint"] = {"use_node_local_storage": True}
        #     eval_ds_config["zero_optimization"] = {
        #         "stage": 3,
        #         "stage3_param_persistence_threshold": 1e4,
        #         "offload_param": {"device": "cpu"},
        #     }
        accelerator.print(f"{eval_ds_config=}")
        evaluation_reward_model, *_ = deepspeed.initialize(model=evaluation_reward_model, config=eval_ds_config)
        evaluation_reward_model.eval()
    else:
        ref_policy = ref_policy.to(device)
        evaluation_reward_model = evaluation_reward_model.to(device)

    # generation_config = GenerationConfig(
    #     max_new_tokens=args.response_length,
    #     min_new_tokens=args.response_length,
    #     temperature=(args.temperature + 1e-7),
    #     top_k=0.0,
    #     top_p=1.0,
    #     do_sample=True,
    # )

    # use the same `0.01` temperature for validation response generation https://github.com/openai/summarize-from-feedback/blob/700967448d10004279f138666442bf1497d0e705/exps/sample.py#L27
    validation_generation_config = GenerationConfig(
        max_new_tokens=args.response_length,
        min_new_tokens=args.response_length,
        temperature=(0.01 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )


    accelerator.print("===evaluation before training===")

    update = 0
    eval_split = list(eval_dataloaders.keys())[0]
    import itertools
    with torch.no_grad():
        for split in ["train", "validation"]:
        #for split in ["validation", "test"]:
        # for split in ["train"]:
            accelerator.print('the size of the dataloaders is ', len(eval_dataloaders[split]))
            eval_start_time = time.time()
            eval_storage, eval_df = evaluate(
                evaluation_reward_model,
                model,
                tokenizer,
                eval_dataloaders[split],
                validation_generation_config,
                sampling=False, # if false, we need to pass the entire dataset
            )
            eval_time = time.time() - eval_start_time
            if accelerator.is_main_process:
                print("the time one time evlaution need is ", eval_time)
            ground_truth_rm_reference_score = eval_df["eval_rm_reference_scores"].mean()
            accelerator.print('the ground truth reference score is ', ground_truth_rm_reference_score.item())
            if accelerator.is_main_process:
                eval_ds = Dataset.from_pandas(eval_df)
                eval_ds.save_to_disk(f"{args.data_path}/{args.sft_model_path.replace('/', '_')}/{split}_dataset")