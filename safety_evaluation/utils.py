import gc
import sys
from tqdm import tqdm
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from api import EvalInstance

if sys.version_info >= (3, 12):
    from itertools import batched
else:
    from itertools import islice
    # Implementation of https://docs.python.org/3/library/itertools.html#itertools.batched
    # for python versions < 3.12
    def batched(iterable, n):
        # batched('ABCDEFG', 3) --> ABC DEF G
        if n < 1:
            raise ValueError('n must be at least one')
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch

def handle_non_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    else:
        return str(obj)

def load_model_and_tokenizer(model_name_or_path: str):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token:
            pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    return model, tokenizer

def generate(model, tokenizer, instances: List[EvalInstance], gen_kwargs: dict) -> None:
    batch_size = gen_kwargs.pop("batch_size")
    compute_norms = gen_kwargs.pop("compute_norms")
    prefill = gen_kwargs.pop("prefill")
    use_template = gen_kwargs.pop("use_template")

    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        print("Setting pad token to [PAD]")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    pbar = tqdm(total=len(instances), desc="Generating completions...")
    for instance_batch in list(batched(instances, batch_size)):
        if use_template:
            if not prefill: 
                contexts = [tokenizer.apply_chat_template(i.messages, tokenize=False, add_generation_prompt=True) for i in instance_batch]
            else:
                contexts = []
                for instance in instance_batch:
                    message = tokenizer.apply_chat_template(instance.messages, tokenize=False, add_generation_prompt=True)
                    message += instance.default_target + ":\n\n"
                    contexts.append(message)
                
            # Remove BOS token if it is present -- this gets taken care of while tokenizing
            contexts = [c.replace(tokenizer.bos_token, "") for c in contexts]
        else:
            if prefill:
                raise ValueError("Prefill attack uses a template")
            contexts = []
            for instance in instance_batch:
                instance.messages.append({"role": "assistant", "content": instance.default_target})
                message = "\n".join(d['content'] for d in instance.messages)
                contexts.append(message)

        inputs = tokenizer(
            contexts,
            truncation=False,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=True
        )

        inputs["input_ids"] = inputs["input_ids"].to(model.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                output_hidden_states=compute_norms,
                return_dict_in_generate=True,
                **gen_kwargs
            )
        
        generated_tokens = output.sequences[:, inputs["input_ids"].shape[1]:]
        generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Compute activation norms
        offset = 1 if tokenizer.bos_token and output.sequences[0][0] == tokenizer.bos_token_id else 0
        if compute_norms:
            activation_norms = []
            num_layers = len(output.hidden_states[0])

            for i in range(num_layers):
                hidden_states = torch.cat([output.hidden_states[j][i] for j in range(len(output.hidden_states))], dim=1)
                activation_norm = hidden_states.norm(dim=-1)
                activation_norms.append(activation_norm)

            # (num_layers, batch_size, seq_len) -> (batch_size, num_layers, seq_len)
            activation_norms = torch.stack(activation_norms, dim=0).permute(1, 0, 2)
            activation_norms = activation_norms.cpu().numpy()

        output.sequences = output.sequences.cpu().numpy()

        for i, instance in enumerate(instance_batch):
            instance.generation = generations[i]
            instance.tokens = output.sequences[i, offset:]
            if compute_norms:
                instance.activation_norms = activation_norms[i]

        del generated_tokens
        gc.collect()
        torch.cuda.empty_cache()
        pbar.update(batch_size)

