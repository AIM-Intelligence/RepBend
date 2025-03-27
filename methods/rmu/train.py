import numpy as np
import torch
import transformers
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from methods.args import (HyperparamArguments, LoraArguments, ModelArguments,
                          TrainingArguments)
from methods.rmu.dataset import rmuDataset
from methods.rmu.trainer import CustomTrainer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

force_half_precision=False

def data_collator(batch_list):
    batch_inputs = {}
    for features in batch_list:
        for k, input in features.items():
            batch_inputs.setdefault(k , []).append(input)
        
    for k, inputs in batch_inputs.items():
        if isinstance(inputs[0], torch.Tensor):
            if 'input_ids' in k:
                batch_inputs[k] = torch.nn.utils.rnn.pad_sequence(
                [input[0] for input in inputs],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)[:, :self.tokenizer.model_max_length]
            elif 'attention_mask' in k:
                batch_inputs[k] = torch.nn.utils.rnn.pad_sequence(
                [input[0] for input in inputs],
                batch_first=True,
                padding_value=0)[:, :self.tokenizer.model_max_length]
        elif isinstance(inputs[0], int):
            batch_inputs[k] = torch.tensor(inputs)
        else:
            raise ValueError(f"Return data type not implemented {type(inputs[0])}")
    return batch_inputs

def train():
    if force_half_precision:
        torch.set_default_dtype(torch.float16)
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, LoraArguments, HyperparamArguments)
    )
    (
        model_args,
        training_args,
        _,
        hyperparam_args,
    ) = parser.parse_args_into_dataclasses()

    print(hyperparam_args.to_dict())
    print(model_args)
    print(training_args)
    
    hyperparam_args.layer_ids = [int(layer_id) for layer_id in hyperparam_args.layer_ids.split(",")]
    hyperparam_args.param_ids = [int(param_id) for param_id in hyperparam_args.param_ids.split(",")]

    device_map = "auto"

    model_name_or_path = model_args.model_name_or_path

    config = AutoConfig.from_pretrained(model_name_or_path)
    config.num_hidden_layers = max(hyperparam_args.layer_ids)+1

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast="LlamaForCausalLM" not in config.architectures,
    )
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    extra_save_kargs = dict(tokenizer=tokenizer)
    # save_model_function = save_model_and_tokenizer
    train_dataset = rmuDataset(tokenizer, num_examples=10000, mode=hyperparam_args.dataset_mode, max_length=training_args.model_max_length, model_name_or_path=model_name_or_path, dataset_path=training_args.dataset_path, split=training_args.dataset_split)
    
    model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            device_map=device_map,
    )

    print("model", model)

    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    
    print("TRAIN LEN: ", len(train_dataset))

    training_args.remove_unused_columns = False
    trainer = CustomTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_dataset, 
        data_collator=data_collator,
        max_seq_length=training_args.model_max_length,
        packing=True,
        hyperparam_args=hyperparam_args,
    )
    model.config.use_cache = False
    
    trainer.train()
    
    # trainer.save_model(training_args.output_dir)
    anchor_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=model.dtype, device_map="auto")
    model.model.layers = model.model.layers + anchor_model.model.layers[max(hyperparam_args.layer_ids)+1:]
    model.config = anchor_model.config
    del anchor_model
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
if __name__ == "__main__":
    SEED = 42
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)

    train()
