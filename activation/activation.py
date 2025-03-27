
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import os
from datetime import datetime
import pickle  # For saving/loading activations

now = datetime.now().strftime("%d%m%y_%H%M%S")
activation_dir = 'activations'
use_template = True
num_samples = 1000
model_type = 'llama'  # or  'mistral'
models_title = []

def save_activations_to_disk(activations, filename):
    # activations_fp32 = {layer: activation.astype(np.float32) for layer, activation in activations.items()}
    with open(filename, 'wb') as f:
        pickle.dump(activations, f)

def load_activations(filename):
    """Load activations from a file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_act_at_position(outputs, layers, layer_idx):

    layer = layers[layer_idx]

    # Get normalization layers (whatever they're called)
    first_norm = getattr(layer, 'input_layernorm', None) or getattr(layer, 'norm1', None)
    second_norm = getattr(layer, 'post_attention_layernorm', None) or getattr(layer, 'norm2', None)

    # Get attention layer (might be self_attn or attention)
    attn_layer = getattr(layer, 'self_attn', None) or getattr(layer, 'attention', None)

    if first_norm and second_norm and attn_layer:
        # 1. First layernorm and attention
        x = outputs.hidden_states[layer_idx-1]
    
        # 1. First layernorm 
        pre_attn_norm = first_norm(x)
        
        # Create minimal position IDs based on sequence length
        seq_length = pre_attn_norm.size(1)
        position_ids = torch.arange(seq_length, device=pre_attn_norm.device)[None, :]
        
        # Call attention with minimal required args for Mistral
        attn_output = attn_layer(
            hidden_states=pre_attn_norm,
            position_ids=position_ids,
        )[0]
        
        # After attention + residual
        post_attn = x + attn_output
        
        # 2. Second layernorm
        pre_ffn_norm = second_norm(post_attn)
        
        # The final output after MLP
        output = outputs.hidden_states[layer_idx]
        
        return pre_attn_norm, post_attn, pre_ffn_norm, output
    
def get_activation_for_tokens(vector, activations, model, tokenizer, sample, layer_idx, only_response, use_last_token, response_length = ''):
    
    if use_last_token: # of the prompt only
        # Get the activation of the last token
        activations[layer_idx].append(vector[:, -1, :].detach().cpu().numpy())
    else:
        if only_response:
            if use_template:
                if model_type == 'mistral':
                    inputs_p = tokenizer(sample[0]["content"], return_tensors="pt").to(model.device)
                    p_token_len = len(inputs_p['input_ids'][0]) + 8  ## eight because '<s>', '▁[', 'INST', ']', '▁[', '/', 'INST', ']' are added
                elif model_type == 'llama':
                    inputs_p = tokenizer(sample[1]["content"], return_tensors="pt").to(model.device)
                    p_token_len = 16 + len(inputs_p['input_ids'][0]) + 5  ## twenty-one because '<|begin_of_text|>', '<|start_header_id|>', 'system', '<|end_header_id|>', 'ĊĊ', 'You', 'Ġare', 'Ġa', 'Ġhelpful', 'Ġassistant', '.', '<|eot_id|>', '<|start_header_id|>', 'user', '<|end_header_id|>', 'ĊĊ' are added in front and '<|eot_id|>', '<|start_header_id|>', 'assistant', '<|end_header_id|>', 'ĊĊ', to tbe back.   
            else:
                inputs_p = tokenizer(sample["prompt"], return_tensors="pt").to(model.device)
                p_token_len = len(inputs_p['input_ids'][0])
            if response_length == '':                
                activations[layer_idx].append(vector[:,p_token_len:, :].mean(dim=1).detach().cpu().numpy())
            elif response_length == '1':
                activations[layer_idx].append(vector[:,p_token_len:p_token_len+1, :].mean(dim=1).detach().cpu().numpy())
            elif response_length == '10':
                activations[layer_idx].append(vector[:,p_token_len:p_token_len+10, :].mean(dim=1).detach().cpu().numpy())
            elif response_length == '100':
                activations[layer_idx].append(vector[:,p_token_len:p_token_len+100, :].mean(dim=1).detach().cpu().numpy())
        else:
            activations[layer_idx].append(vector.mean(dim=1).detach().cpu().numpy())
     

def infer_activations(model, tokenizer, data, response_length, use_last_token=True, only_response=False):
    model.eval()
    model.to('cuda')
    layers = getattr(model, 'model', model).layers
    layer_nums = list(range(0,len(layers)))
    
    activations_pre_attn_norm = {layer: [] for layer in layer_nums}
    activations_post_attn = {layer: [] for layer in layer_nums}
    activations_pre_ffn_norm = {layer: [] for layer in layer_nums}
    activations_output = {layer: [] for layer in layer_nums}

    for sample in tqdm(data):
        if use_template:
            inst_with_template = tokenizer.apply_chat_template(sample, tokenize=False)
            inputs = tokenizer(inst_with_template, return_tensors="pt", add_special_tokens=False).to(model.device)
        else:
            inputs = tokenizer(sample["text"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for layer_idx in layer_nums:
            pre_attn_norm, post_attn, pre_ffn_norm, output = get_act_at_position(outputs, layers, layer_idx)
            get_activation_for_tokens(pre_attn_norm, activations_pre_attn_norm, model, tokenizer, sample, layer_idx, only_response, use_last_token, response_length)
            get_activation_for_tokens(post_attn, activations_post_attn, model, tokenizer, sample, layer_idx, only_response, use_last_token, response_length)
            get_activation_for_tokens(pre_ffn_norm, activations_pre_ffn_norm, model, tokenizer, sample, layer_idx, only_response, use_last_token, response_length)
            get_activation_for_tokens(output, activations_output, model, tokenizer, sample, layer_idx, only_response, use_last_token, response_length)
                
    model.to('cpu')
    torch.cuda.empty_cache()

    # If using all tokens, return activations as flattened arrays across all tokens
    # if not use_last_token:
        # return {layer: np.vstack([activation.reshape(-1, activation.shape[-1]) for activation in activations[layer]]) for layer in layer_nums}
    return {"pre_attn_norm" : {layer: np.vstack(activations_pre_attn_norm[layer]) for layer in layer_nums}, 
            "post_attn" : {layer: np.vstack(activations_post_attn[layer]) for layer in layer_nums}, 
            "pre_ffn_norm" : {layer: np.vstack(activations_pre_ffn_norm[layer]) for layer in layer_nums}, 
            "output" : {layer: np.vstack(activations_output[layer]) for layer in layer_nums}}


def save_act(model_name_or_path, data_type, response_length = '', only_response = False):
    if bool(len(data_type)==1):
        only_response = False # meaningless for single prompt
        response_length = ''
    act_path = f"{activation_dir}/{data_type}_{model_name_or_path.replace('/', '_')}{'_r' if only_response else ''}{response_length}{'_t' if use_template else ''}.pkl"
    if not os.path.isfile(act_path):
        data = load_data(data_type)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=os.getenv('HUGGINGFACE_TOKEN'))
        model = AutoModel.from_pretrained(model_name_or_path, token=os.getenv('HUGGINGFACE_TOKEN'))
        activations = infer_activations(model, tokenizer, data, response_length, use_last_token=bool(len(data_type)==1), only_response=only_response)
        save_activations_to_disk(activations, act_path)
        torch.cuda.empty_cache()


def load_data(
    data_type : str = 'u' , # 's', 'u', 'uu', 'us', 'ss', 'su'
    data_dir : str = 'wildjailbreak.jsonl', 
    seed : int=42
    ): # data_type : 's', 'u', 'uu', 'us', 'ss'
    df = pd.read_json(data_dir, lines=True)
    if data_type == 's':
        df = df[df['prompt_type']=='harmless']
        df['text'] = df['prompt']
    elif data_type == 'u':
        df = df[df['prompt_type']=='harmful']
        df['text'] = df['prompt']
    elif data_type == 'sm':
        df = df[df['prompt_type']=='harmless']
        df['text'] = df.apply(lambda row: row['prompt'] + " " + row["answer"], axis=1)
    elif data_type == 'um':
        df = df[df['prompt_type']=='harmful']
        df['text'] = df.apply(lambda row: row['prompt'] + " " + row["answer"], axis=1)
        df['text'] = df['prompt']
    elif data_type == 'uu':
        df = df[df['prompt_type']=='harmful']
        df['text'] = df.apply(lambda row: row['prompt'] + " " + row["harmful_answer"], axis=1)
    elif data_type == 'us':
        df = df[df['prompt_type']=='harmful']
        df['text'] = df.apply(lambda row: row['prompt'] + " " + row["harmless_answer"], axis=1)
    elif data_type == 'su':
        df = df[df['prompt_type']=='harmless']
        df['text'] = df.apply(lambda row: row['prompt'] + " " + row["harmful_answer"], axis=1)
    elif data_type == 'ss':
        df = df[df['prompt_type']=='harmless']
        df['text'] = df.apply(lambda row: row['prompt'] + " " + row["harmless_answer"], axis=1)
    data = df.sample(num_samples, random_state=seed).to_dict(orient='records')
    data = apply_template(data, data_type = data_type)
    return data

def apply_template(data, data_type):
    if not use_template:
        return data
    if model_type == 'mistral':
        if data_type in ['u', 's']:
            messages = [[
                {"role": "user", "content": item['prompt']},
            ] for item in data]
        elif data_type in ['um', 'sm']:
            messages = [[
                {"role": "user", "content": item['prompt']},
                {"role": "assistant", "content": item['answer']}
            ] for item in data]
        elif data_type in ['uu', 'su']:
            messages = [[
                {"role": "user", "content": item['prompt']},
                {"role": "assistant", "content": item['harmful_answer']}
            ] for item in data]
        elif data_type in ['ss', 'us']:
            messages = [[
                {"role": "user", "content": item['prompt']},
                {"role": "assistant", "content": item['harmless_answer']}
            ] for item in data]
        return messages
    elif model_type == 'llama': 
        if data_type in ['u', 's']:
            messages = [[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": item['prompt']},
            ] for item in data]
        elif data_type in ['um', 'sm']:
            messages = [[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": item['prompt']},
                {"role": "assistant", "content": item['answer']}
            ] for item in data]
        elif data_type in ['uu', 'su']:
            messages = [[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": item['prompt']},
                {"role": "assistant", "content": item['harmful_answer']}
            ] for item in data]
        elif data_type in ['ss', 'us']:
            messages = [[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": item['prompt']},
                {"role": "assistant", "content": item['harmless_answer']}
            ] for item in data]
        return messages
    else:
        print("invalid model type provided for apply_template()")
        
    

if __name__ == '__main__':
    # BASE_MODEL = 'mistralai/Mistral-7B-v0.1'
    # SAFE_MODEL = 'mistralai/Mistral-7B-Instruct-v0.1'
    # EVIL_MODEL = 'maywell/PiVoT-0.1-Evil-a'

    # BASE_MODEL = 'meta-llama/Meta-Llama-3-8B'
    # SAFE_MODEL = 'meta-llama/Meta-Llama-3-8B-Instruct'
    # EVIL_MODEL = 'Orenguteng/Llama-3-8B-Lexi-Uncensored'

    model = 'meta-llama/Meta-Llama-3-8B-Instruct'


    for data_type in ['sm', 'um']:
        for only_response in [True, False]:
            if only_response == True :
                for response_length in ['', '10', '100']:
                    save_act(model, data_type, response_length, only_response=True)
            else:
                save_act(model, data_type, response_length = '', only_response=False)

    for data_type in ['uu', 'us']:
            for only_response in [True, False]:
                if only_response == True :
                    for response_length in ['', '10', '100']:
                        save_act(model, data_type, response_length, only_response=True)
                else:
                    save_act(model, data_type, response_length = '', only_response=False)

