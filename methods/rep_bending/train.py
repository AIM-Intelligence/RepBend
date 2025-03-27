import random
from dataclasses import dataclass

import numpy as np
import torch
import transformers
# import wandb
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from methods.args import (HyperparamArguments, LoraArguments, ModelArguments,
                          TrainingArguments)
from methods.rep_bending.classifier import HarmbenchClassifier
from methods.rep_bending.dataset import RepBendingDataset
from methods.rep_bending.trainer import CustomTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def data_collator(batch_list):
    batch_inputs = {}
    for features in batch_list:
        for k, input in features.items():
            batch_inputs.setdefault(k , []).append(input)
    
    for k, inputs in batch_inputs.items():
        if isinstance(inputs[0], torch.Tensor):
            batch_inputs[k] = torch.cat(inputs, dim=0)
        elif isinstance(inputs[0], int):
            batch_inputs[k] = torch.tensor(inputs)
        else:
            # raise ValueError(f"Return data type not implemented {type(inputs[0])}")
            batch_inputs[k] = inputs
    return batch_inputs

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, LoraArguments, HyperparamArguments)
    )
    (
        model_args,
        training_args,
        lora_args,
        hyperparam_args,
    ) = parser.parse_args_into_dataclasses()

    print(hyperparam_args.to_dict())
    print(lora_args)
    print(model_args)
    print(training_args)
    
    # wandb.init(project="safety_hyperparam_search", name=training_args.output_dir[6:],config=hyperparam_args.to_dict())

    device_map = "auto"

    model_name_or_path = model_args.model_name_or_path
    target_layers = hyperparam_args.target_layers
    target_layer_start_idx = hyperparam_args.target_layer_start_idx
    layers_window_size = hyperparam_args.layers_window_size
    transform_layers = hyperparam_args.transform_layers
    full_layers = hyperparam_args.full_layers
    
    if target_layers != "" and target_layers != "-1":
        hyperparam_args.target_layers = [int(layer) for layer in target_layers.split(",")] # target representations
    elif layers_window_size > 0:
        hyperparam_args.target_layers = list(range(target_layer_start_idx, target_layer_start_idx+layers_window_size))
    else:
        raise ValueError()
    if "-1" in transform_layers:
        lora_layers_to_transform = [i for i in range(max(hyperparam_args.target_layers) + 1)]
    else:
        lora_layers_to_transform = [int(layer) for layer in transform_layers.split(",")] # transform representations
    
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        layers_to_transform=lora_layers_to_transform if target_layers == "" or target_layers == "-1" else None,
        task_type="CAUSAL_LM",
    )

    drop_layers_after = max(hyperparam_args.target_layers) if not full_layers else None
    print("lora_transform_layers", lora_config.layers_to_transform)
    print("drop_layers_after", drop_layers_after)
    config = AutoConfig.from_pretrained(model_name_or_path)
    if drop_layers_after:
        config.num_hidden_layers = drop_layers_after+1

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        # cache_dir=training_args.cache_dir,
        model_max_length=training_args.max_seq_length,
        padding_side="left",
        use_fast="LlamaForCausalLM" not in config.architectures,
    )
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    # extra_save_kargs = dict(tokenizer=tokenizer)
    # save_model_function = save_model_and_tokenizer
    train_dataset = RepBendingDataset(tokenizer, num_examples=10000, mode=hyperparam_args.loss_mode, max_length=1024, model_name_or_path=model_name_or_path, dataset_path=hyperparam_args.dataset_path, split=hyperparam_args.dataset_split, is_online=hyperparam_args.is_online)
    
    model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            # cache_dir=training_args.cache_dir,
            device_map=device_map,
    )
    training_args.model_name_or_path = model_name_or_path
    print(lora_args.lora_target_modules, lora_config.layers_to_transform)

    model = get_peft_model(model, lora_config)
    print("model", model)

    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    
    print("TRAIN SIZE: ", len(train_dataset))
    
    if hyperparam_args.is_online:
        classifier = HarmbenchClassifier()
    else:
        classifier = None

    training_args.remove_unused_columns = False
    trainer = CustomTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_dataset, 
        data_collator=data_collator,
        max_seq_length=training_args.max_seq_length,
        packing=True,
        hyperparam_args=hyperparam_args,
        classifier=classifier
    )
    model.config.use_cache = False
    
    trainer.train()
    
    trainer.save_model(training_args.output_dir)
    
if __name__ == "__main__":
    SEED = 42
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    
    train()
