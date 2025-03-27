import random
from typing import Dict

import jsonlines
import numpy as np
import torch
import transformers
from datasets import load_dataset
from methods.dataset import CustomDataset

random.seed(0)

class rmuDataset(CustomDataset):
    
    def before_init(self, tokenizer: transformers.PreTrainedTokenizer, num_examples, max_length, model_name_or_path, dataset_path, split=None, mode=None, train_type=None):
        '''
        mode == 1 --> use all data
        mode == 2 --> use wildjailbreak harmless data to orig_retain
        mode == 3 --> use wildjailbreak harmful data to circuit breaker 
        '''
        print("dataset mode:", mode)
   
    def after_init(self, tokenizer: transformers.PreTrainedTokenizer, num_examples, max_length, model_name_or_path, dataset_path, split=None, mode=None, train_type=None):
        
        # ======================= Retain ======================= #
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        orig_s = []
        for example in ds:
            messages = example["messages"]
            if len(messages) < 2: continue

            switch = np.random.choice(self.switch_select)
            if switch == 0:
                formatted_input = tokenizer.apply_chat_template(messages, tokenize=False).replace(tokenizer.bos_token, "")
            elif switch == 1:
                formatted_input = self.one_shot_template.format(
                    user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                    instruction="", response=messages[1]["content"])

            orig_s.append(formatted_input)

            if len(orig_s) > num_examples:
                break
        self.orig_s_retain = orig_s
        random.shuffle(self.orig_s_retain)
        print("orig_s_retain[0]", orig_s[0])
        print("Orig s length:", len(self.orig_s_retain))
        
        ds = load_dataset(dataset_path, split)
        
        # ==================== wildguardtrain unharmful + compliance & harmful + refusal ================
        if 'wildguardmix' in dataset_path:
            borderline_orig_s = []
            
            for d in ds['train']:
                if (d['prompt_harm_label'] == 'unharmful' and d['response_refusal_label'] == 'compliance' and d['response_harm_label'] == 'unharmful') \
                or (d['prompt_harm_label'] == 'harmful' and d['response_refusal_label'] == 'refusal' and d['response_harm_label'] == 'unharmful'):
                    switch = np.random.choice(self.switch_select)
                    if switch == 0:
                        formatted_input = self.one_shot_template.format(
                            user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                            instruction=d['prompt'], response=d['response'])
                    elif switch == 1:
                        formatted_input = self.one_shot_template.format(
                            user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                            instruction="", response=d['response'])
                    
                    borderline_orig_s.append(formatted_input)
            borderline_orig_s = random.sample(borderline_orig_s, num_examples)
            self.orig_s_retain += borderline_orig_s
            random.shuffle(self.orig_s_retain)
            print("borderline_orig_s[0]", borderline_orig_s[0])
            print("Orig s length:", len(self.orig_s_retain))
            
        # ======================= wildjailbreak safe answer ======================= #
        train_data = []
        with jsonlines.open(f"./wildjailbreak.jsonl") as f:
            for line in f.iter():
                train_data.append(line)
                
        random.shuffle(train_data)
        
        if mode == 1 or mode == 2:
            unsafe_samples_safe_answer = []
            for d in train_data:
                if d['prompt_type'] == 'harmful':
                    switch = np.random.choice(self.switch_select)
                    if switch == 0:
                            formatted_input = self.one_shot_template.format(
                                user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                                instruction=d['prompt'], response=d['harmless_answer'])
                    elif switch == 1:
                        formatted_input = self.one_shot_template.format(
                            user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                            instruction="", response=d['harmless_answer'])
                        
                    unsafe_samples_safe_answer.append(formatted_input)
            
            self.orig_s_retain += unsafe_samples_safe_answer[:num_examples]
            random.shuffle(self.orig_s_retain)
            print("refusal_orig_s[0]", unsafe_samples_safe_answer[0])
            print("Orig s length:", len(self.orig_s_retain))
        
        # ======================= Circuit Breaker ======================= #
        circuit_breaker_orig = []

        for d in ds['train']:
            if (d['prompt_harm_label'] == 'harmful' and d['response_refusal_label'] == 'compliance'):
                switch = np.random.choice(self.switch_select)
                if switch == 0:
                    formatted_input = self.one_shot_template.format(
                        user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                        instruction=d['prompt'], response=d['response'])
                elif switch == 1:
                    formatted_input = self.one_shot_template.format(
                        user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                        instruction="", response=d['response'])
                circuit_breaker_orig.append(formatted_input)
        
        self.circuit_breaker_orig = circuit_breaker_orig
        random.shuffle(self.circuit_breaker_orig)
        print("circuit_breaker_orig[0]", circuit_breaker_orig[0])
        print("Short circuit length:", len(self.circuit_breaker_orig))

        if mode == 1 or mode == 3:
            unsafe_samples_unsafe_answer = []
            for d in train_data:
                if d['prompt_type'] == 'harmful':
                    formatted_input1 = self.one_shot_template.format(
                        user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                        instruction=d['prompt'], response=d['harmful_answer'])
                        
                    unsafe_samples_unsafe_answer.append(formatted_input1)
            
            self.circuit_breaker_orig += unsafe_samples_unsafe_answer[:num_examples]
            random.shuffle(self.circuit_breaker_orig)
            print("unsafe_samples_unsafe_answer_orig_s[0]", unsafe_samples_unsafe_answer[0])
            print("Short circuit length:", len(self.circuit_breaker_orig))
        
        # padding right
        self.tokenizer.padding_side = "right"

    def __len__(self):
        return min(len(self.orig_s_retain), len(self.circuit_breaker_orig))
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        orig_s_retain = self.orig_s_retain[i]
        circuit_breaker_orig = self.circuit_breaker_orig[i]

        cb_tokenized_kwargs = dict(max_length=int(self.max_length/2), padding=False, truncation=True, return_tensors="pt")
        tokenize_kwargs = dict(max_length=int(self.max_length), padding=False, truncation=True, return_tensors="pt")

        truncation_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side="right"

        # =========== Circuit Breaker Inputs ===========
        # === split to [request, response] shape [512,512] to support different mask configs ===
        cb_request, cb_response = circuit_breaker_orig.split('<SEPARATOR>')
        self.tokenizer.padding_side = "left"
        tokenized_request_circuit_breaker = self.tokenizer(cb_request, **cb_tokenized_kwargs)
        self.tokenizer.padding_side = "right"
        response_tokenized_circuit_breaker = self.tokenizer(cb_response, add_special_tokens=False, **cb_tokenized_kwargs)
        self.tokenizer.padding_side = "left"

        combined_input_ids_circuit_breaker = torch.cat([tokenized_request_circuit_breaker["input_ids"], response_tokenized_circuit_breaker["input_ids"]], dim=1)
        combined_attention_mask_circuit_breaker = torch.cat([tokenized_request_circuit_breaker["attention_mask"], response_tokenized_circuit_breaker["attention_mask"]], dim=1)

        # ========== Retain Inputs ===========
        tokenized_inputs_retain = self.tokenizer(orig_s_retain.replace('<SEPARATOR>', self.sep_token), **tokenize_kwargs)
        
        return dict(
            input_ids_circuit_breaker=combined_input_ids_circuit_breaker,
            attention_mask_circuit_breaker=combined_attention_mask_circuit_breaker,
            input_ids=tokenized_inputs_retain["input_ids"],
            attention_mask=tokenized_inputs_retain["attention_mask"],
        )
