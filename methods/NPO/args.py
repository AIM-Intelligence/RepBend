from dataclasses import dataclass, field
from typing import List, Optional

import transformers


@dataclass
class HyperparamArguments:
    loss_alpha:float = 1.0
    loss_beta:float = 1.0
    loss_gamma:float = 0.0

    def to_dict(self):
        return dict(
            loss_alpha = self.loss_alpha,
            loss_beta = self.loss_beta,
            loss_gamma = self.loss_gamma,
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

    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    grouped_to_max_length: bool = field (
        default=False, metadata={"help": "Group to chunks of max length for pretraining"}
    )
    
    dataset_path: str = field(default=None)
    dataset_split: str = field(default=None)
    
    use_refusal_retain : Optional[bool] = field(default = True, 
                                                metadata={"help":"whether to train on refusal retain set for Llama models"})
    sc_train_subset : Optional[List[str]] = field(default=None,
                                                  metadata={"help":"subset of the sc train set to train on"})
    log_every : Optional[int] = field(default = 10,
                                      metadata = {"help" : "log loss every log_every steps"})
    sc_train_seq_type : Optional[str] = field(default = 'all_text',
                                              metadata = {"help" : "what portion of the sequence to train on. can be all_text or assistant_response"})
    coeff_schedule : Optional[str] = field(default = 'linear_converge',
                                           metadata = {'help' : 'schedule for the coefficients. can be linear_converge or constant'})
    sc_loss_type : Optional[str] = field(default = 'orig_act_dotprod',
                                         metadata = {'help' : 'type of loss function for shortcircuiting. can be orig_act_dotprod, rand_vec_norm, pos_constant_rmu_{coeff}, center_constant_rmu_{coeff}'})
    
    
