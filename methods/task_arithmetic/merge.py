from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
import transformers
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from methods.args import (HyperparamArguments, LoraArguments, ModelArguments,
                          TrainingArguments)
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
    
    base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            device_map=device_map,
    )
    safe_dir = training_args.output_dir + 'safe'
    unsafe_dir = training_args.output_dir + 'unsafe'
    
    safe_weights = [0.5, 1.0, 1.2]#[0.0, 0.5, 1.0]
    unsafe_weights = [0.0, 0.1, 0.2] # [0.3, 0.5, 0.8, 1.0, 1.2]
    for safe_W in safe_weights:
        for unsafe_W in unsafe_weights:
            print(safe_W, unsafe_W)
            model = PeftModel.from_pretrained(base_model, safe_dir, adapter_name="default")
            model_param = OrderedDict(model.named_parameters())
            for n, p in model_param.items():
                if 'lora_A' in n and 'default' in n:
                    model_param[n].copy_(safe_W*p)
            
            model = model.merge_and_unload()
            
            model = PeftModel.from_pretrained(model, unsafe_dir, adapter_name="default")
            model_param = OrderedDict(model.named_parameters())
            for n, p in model_param.items():
                if 'lora_A' in n and 'default' in n:
                    model_param[n].copy_(-unsafe_W*p)
            
            model = model.merge_and_unload()
            
            output_dir = training_args.output_dir + f'safe{safe_W}_unsafe{unsafe_W}'
            
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
                    
    # safe_dir = training_args.output_dir + 'safe'
    # print('load_safe')

    # model = PeftModel.from_pretrained(base_model, safe_dir, adapter_name="safe")
    
    # print('load_unsafe')
    # unsafe_dir = training_args.output_dir + 'unsafe'
    
    # model.load_adapter(unsafe_dir, adapter_name="unsafe")
    
    # for n, p in model.named_parameters():
    #     if 'lora_A' in n and 'safe' in n:
    #         print(n)
    #         print(p)
    #         break
    
    # for n, p in model.named_parameters():
    #     if 'lora_A' in n and 'unsafe' in n:
    #         print(n)
    #         print(p)
    #         break
    
    # # Make weight of B negative for "unsafe" adapter
    # model_param = OrderedDict(model.named_parameters())
    # for n, p in model_param.items():
    #     if 'lora_A' in n and 'unsafe' in n:
    #         model_param[n].copy_(-p)
    
    # for n, p in model.named_parameters():
    #     if 'lora_A' in n and 'safe' in n:
    #         print(n)
    #         print(p)
    #         break
    
    # for n, p in model.named_parameters():
    #     if 'lora_A' in n and 'unsafe' in n:
    #         print(n)
    #         print(p)
    #         break

    # safe_weights = [0.0, 0.5, 1.0]
    # unsafe_weights = [0.3, 0.5, 0.8, 1.0, 1.2]
    # for safe_W in safe_weights:
    #     for unsafe_W in unsafe_weights:
    #         model.add_weighted_adapter(
    #             adapters=["safe", "unsafe"],
    #             weights=[safe_W, unsafe_W],
    #             adapter_name="default",
    #             combination_type="linear"
    #         )
    #         model.set_adapter("default")
            
    #         output_dir = training_args.output_dir + f'safe{safe_W}_unsafe{unsafe_W}'
            
    #         model.save_pretrained(output_dir)
    #         tokenizer.save_pretrained(output_dir)

    
if __name__ == "__main__":
    SEED = 42
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)

    train()
