import os
from dataclasses import dataclass

import numpy as np
import torch
import transformers
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from methods.args import (HyperparamArguments, LoraArguments, ModelArguments,
                          TrainingArguments)
from methods.DPO.dataset import DPODataset
from methods.DPO.trainer import CustomTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

force_half_precision=False

@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    # device:torch.device
    # transform:DataAugmentation

    def __call__(self, instances):
        prompt_input_ids, chosen_input_ids, rejected_input_ids = tuple([instance[key] for instance in instances]
                                  for key in ("prompt_input_ids", "chosen_input_ids", "rejected_input_ids"))
        prompt_input_ids = torch.nn.utils.rnn.pad_sequence(
            [prompt.flip(dims=[0]) for prompt in prompt_input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id).flip(dims=[1])
        chosen_input_ids = torch.nn.utils.rnn.pad_sequence(
            chosen_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        rejected_input_ids = torch.nn.utils.rnn.pad_sequence(
            rejected_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        prompt_input_ids = prompt_input_ids[:, :self.tokenizer.model_max_length]
        chosen_input_ids = chosen_input_ids[:, :self.tokenizer.model_max_length]
        rejected_input_ids = rejected_input_ids[:, :self.tokenizer.model_max_length]

        
        prompt_attention_mask=prompt_input_ids.ne(self.tokenizer.pad_token_id)
        chosen_attention_mask=chosen_input_ids.ne(self.tokenizer.pad_token_id)
        rejected_attention_mask=rejected_input_ids.ne(self.tokenizer.pad_token_id)
            
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in prompt_input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
            for input_id in chosen_input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
            for input_id in rejected_input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
        
        batch = dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=chosen_attention_mask,
            rejected_input_ids=rejected_input_ids,
            rejected_attention_mask=rejected_attention_mask,
        )

        return batch

def train():
    if force_half_precision:
        torch.set_default_dtype(torch.float16)
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

    device_map = "auto"

    model_name_or_path = model_args.model_name_or_path

    config = AutoConfig.from_pretrained(model_name_or_path)

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
    
    model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            device_map=device_map,
    )

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    print("model", model)

    train_dataset = DPODataset(tokenizer, num_examples=20000, max_length=2048, model_name_or_path=model_name_or_path, dataset_path=training_args.dataset_path, split=training_args.dataset_split)
    

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
        data_collator=DataCollator(tokenizer),
        max_seq_length=training_args.model_max_length,
        packing=True,
        hyperparam_args=hyperparam_args
    )

    model.config.use_cache = False
    
    trainer.train()
    
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    trainer.save_model(output_dir)

    trainer.deepspeed.empty_partition_cache()

    
if __name__ == "__main__":
    SEED = 42
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)

    train()
