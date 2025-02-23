from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)
import torch
import numpy as np
import torch.nn as nn

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
        print(self.base_model)
        print(self.base_config)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer



class ScalarModel(PreTrainedModel):
    config_class = ScalarModelConfig

    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        print(config)
        
        # self.lm_backbone = AutoModel.from_pretrained(
        #     config.base_model,
        #     config=self.config.base_config,
        #     trust_remote_code=True,
        # )
        
        def base_to_name(base):
            if "6.9b" in base:
                return "vwxyzjn/EleutherAI_pythia-6.9b-deduped__sft__tldr"
            elif "1b" in base:
                return "vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr"
            else:
                raise NotImplementedError()
        
        base_name = base_to_name(config.base_model)
        
        self.lm_backbone = AutoModel.from_pretrained(
            base_name,
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


x = ScalarModel.from_pretrained(
    "vwxyzjn/EleutherAI_pythia-6.9b-deduped__reward__tldr",
    revision="reward__44413__1706651113",
    trust_remote_code=True,
)