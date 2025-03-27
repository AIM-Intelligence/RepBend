from dataclasses import dataclass, field
from typing import List, Optional, Union

import transformers


@dataclass
class HyperparamArguments:
    target_layers: str = field(default="", metadata={"help": "Layers for Representation. Layers are seperate by `,` eg: `10,12,14,16,18,20` "})
    target_layer_start_idx: int = field(default=0, metadata={"help": "Target Layers for Representation. The start index of the target layers"})
    layers_window_size: int = field(default=0, metadata={"help": "Target Layers for Representation. The number of target layers i.e. target_layer_start_idx ~ target_layer_start_idx+layers_window_size "})

    transform_layers: str = field(default="-1", metadata={"help": "Layers for Representation. Layers are seperate by `,` eg: `10,12,14,16,18,20` "})
    
    loss_mode:str = field(default="response_all", metadata={"help":"One of request_last, request_all, request_response, response_all"})

    alpha_mode:str = field(default="all", metadata={"help":"all OR target"})

    adv_string: str = field(default="", metadata={"help": "adversarial string for harmful prompt. (eg: Start with 'Sure here's')"})
    full_layers: bool = field(
        default=True, metadata={"help": "Whether to drop not used layer during training"}
    )

    is_online: bool = field(default=False, metadata={"help":"Whether to use online sample generation or not"})

    dataset_mode:int = field(
        default=1, metadata={"help":"One of 1, 2, or 3"}
    )
    
    dataset_path: str = field(default=None)
    dataset_split: str = field(default=None)

    loss_alpha: float = 0.0
    loss_beta: float = 0.0
    loss_gamma: float = 0.0
    loss_epsilon: float = 0.0
    loss_eta: float = 0.0
    
    paired_repr_length: int = 20

    def to_dict(self):
        return dict(
            target_layers=self.target_layers,
            target_layer_start_idx=self.target_layer_start_idx,
            layers_window_size=self.layers_window_size,
            transform_layers=self.transform_layers,
            full_layers=self.full_layers,
            loss_alpha = self.loss_alpha,
            loss_beta = self.loss_beta,
            loss_gamma = self.loss_gamma,
            loss_epsilon = self.loss_epsilon,
            loss_eta = self.loss_eta,
            dataset_mode = self.dataset_mode,
            loss_mode = self.loss_mode,
            alpha_mode=self.alpha_mode,
            is_online=self.is_online
        )
@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Union[List[str], str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    adapter_name_or_path: str = field (
        default=None, metadata={"help": "Adapater name"}
    )
    use_lora: bool = field(
        default=False, metadata={"help": "Use LoRA (default: False)"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    ) # (wj) TODO: This is only for our method, I don't know why we didn't use model_max_length
