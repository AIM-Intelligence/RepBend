import json
import os

import torch
from transformers import (AutoModelForCausalLM,
                          LlavaNextForConditionalGeneration)


def save_model_and_tokenizer(model_name_or_path, model, tokenizer, drop_layers_after, output_dir, trainer):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n\nModel and tokenizer saving to {output_dir}\n\n")
    
    # merge lora
    merged_model = model.merge_and_unload() 
    # merge original layers
    if drop_layers_after is not None:
        anchor_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=merged_model.dtype, device_map="auto")
        merged_model.model.layers = merged_model.model.layers + anchor_model.model.layers[drop_layers_after+1:]
        merged_model.config = anchor_model.config

    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    lorra_config_path = os.path.join(output_dir, "lorra_config.json")
    with open(lorra_config_path, "w", encoding="utf-8") as file:
        json.dump(trainer.lorra_args.to_dict(), file, indent=2)
    
    torch.use_deterministic_algorithms(False)
    if trainer.training_args.do_eval:
        trainer.evaluate()
    

def save_llava_model_and_tokenizer(model_name_or_path, model, processor, drop_layers_after, output_dir, trainer):
    os.makedirs(output_dir, exist_ok=True)
    print(f"MModel and processor saving to {output_dir}")
    
    # merge lora
    merged_model = model.merge_and_unload() 
    # merge original layers
    
    anchor_model = LlavaNextForConditionalGeneration.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=merged_model.dtype)
    merged_model.language_model.model.layers = merged_model.language_model.model.layers + anchor_model.language_model.model.layers[drop_layers_after+1:]
    merged_model.config = anchor_model.config

    merged_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    lorra_config_path = os.path.join(output_dir, "lorra_config.json")
    with open(lorra_config_path, "w", encoding="utf-8") as file:
        json.dump(trainer.lorra_args.to_dict(), file, indent=2)
    
    torch.use_deterministic_algorithms(False)
    if trainer.training_args.do_eval:
        trainer.evaluate()




################################
##### Activation functions #####
################################

def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
        
    hook_handle.remove()

    return cache[0]
    
#######################################
##### Model and data loading code #####
#######################################


def get_params(model, layer_ids, param_ids):
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params
