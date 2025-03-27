import os
from dataclasses import dataclass

import numpy as np
import torch
import transformers
from methods.args import (HyperparamArguments, LoraArguments, ModelArguments,
                          TrainingArguments)
from methods.task_arithmetic.dataset import TaskArithmeticDataset
from methods.utils.get_peft_state_maybe_zero_3 import \
    get_peft_state_maybe_zero_3
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer

force_half_precision=False

@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    # device:torch.device
    # transform:DataAugmentation

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=-100)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
            
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
        
        batch = dict(
            input_ids=input_ids,#.to(self.device),
            labels=labels,#.to(self.device),
            attention_mask=attention_mask,#.to(self.device),
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

    # create safe task vector

    initial_lora_param = get_peft_state_maybe_zero_3(model.named_parameters(), lora_args.lora_bias)

    train_dataset = TaskArithmeticDataset(tokenizer, num_examples=10000, max_length=1024, model_name_or_path=model_name_or_path, dataset_path=training_args.dataset_path, split=training_args.dataset_split, train_type='safe')
    

    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    
    print("TRAIN LEN: ", len(train_dataset))

    training_args.remove_unused_columns = False

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollator(tokenizer),
        max_seq_length=training_args.model_max_length,
        packing=True
    )

    model.config.use_cache = False
    
    trainer.train()
    
    output_dir = training_args.output_dir + 'safe2'
    os.makedirs(output_dir, exist_ok=True)

    trainer.save_model(output_dir)

    # safe_lora_param = get_peft_state_maybe_zero_3(model.named_parameters(), lora_args.lora_bias)

    # safe_task_vector = {}
    # for k in initial_lora_param.keys():
    #     safe_task_vector[k] = safe_lora_param[k] - initial_lora_param[k]
    # trainer.deepspeed.empty_partition_cache()
    # del trainer

    # # create unsafe task vector
    # model = model.get_base_model()

    # model = get_peft_model(model, lora_config)
    
    # model.load_state_dict(initial_lora_param, strict=False)

    # # initial_lora_param = get_peft_state_maybe_zero_3(model.named_parameters(), lora_args.lora_bias)
    
    # train_dataset = CustomDataset(tokenizer, num_examples=10000, max_length=1024, model_name_or_path=model_name_or_path, dataset_path=training_args.dataset_path, split=training_args.dataset_split, train_type='unsafe')

    # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     data_collator=DataCollator(tokenizer),
    #     max_seq_length=training_args.model_max_length,
    #     packing=True
    # )

    # trainer.train()
    
    # output_dir = training_args.output_dir + 'unsafe'
    # os.makedirs(output_dir, exist_ok=True)

    # trainer.save_model(output_dir)

    # unsafe_lora_param = get_peft_state_maybe_zero_3(model.named_parameters(), lora_args.lora_bias)

    # unsafe_task_vector = {}
    # for k in initial_lora_param.keys():
    #     unsafe_task_vector[k] = unsafe_lora_param[k] - initial_lora_param[k]
    
    # del trainer
    
if __name__ == "__main__":
    SEED = 42
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)

    train()
