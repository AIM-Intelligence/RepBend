import random
from typing import Dict

import jsonlines
import torch
import transformers
from datasets import load_dataset
from methods.dataset import CustomDataset

random.seed(0)

class WHPRetrainDataset(CustomDataset):
    
    def after_init(self, tokenizer: transformers.PreTrainedTokenizer, num_examples, max_length, model_name_or_path, dataset_path, split=None, mode=None, train_type=None):

        ds = load_dataset(dataset_path, split)
        safe_samples = []
        unsafe_samples = []
        if 'wildguardmix' in dataset_path:
            for d in ds['train']:
                if (d['prompt_harm_label'] == 'unharmful' and d['response_refusal_label'] == 'compliance' and d['response_harm_label'] == 'unharmful') \
                or (d['prompt_harm_label'] == 'harmful' and d['response_refusal_label'] == 'refusal' and d['response_harm_label'] == 'unharmful'):
                    formatted_input = self.one_shot_template.format(
                    user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                    instruction=d['prompt'], response=d['response'])
                
                    safe_samples.append({'sample':formatted_input, 'is_alllabel': False})

                elif (d['prompt_harm_label'] == 'harmful' and d['response_refusal_label'] == 'compliance'):
                    formatted_input = self.one_shot_template.format(
                    user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                    instruction=d['prompt'], response=d['response'])
                    
                    unsafe_samples.append({'sample':formatted_input, 'is_alllabel': False})

        safe_samples = random.sample(safe_samples, num_examples)
        unsafe_samples = random.sample(unsafe_samples, num_examples)


        train_data = []
        with jsonlines.open(f"./wildjailbreak.jsonl") as f:
            for line in f.iter():
                train_data.append(line)
        
        random.shuffle(train_data)

        unsafe_samples_safe_answer = []
        unsafe_samples_unsafe_answer = []
        for d in train_data:
            if d['prompt_type'] == 'harmful':
                formatted_input2 = self.one_shot_template.format(
                    user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                    instruction=d['prompt'], response=d['harmless_answer'])
                unsafe_samples_safe_answer.append({'sample':formatted_input2, 'is_alllabel': False})

                formatted_input1 = self.one_shot_template.format(
                    user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                    instruction=d['prompt'], response=d['harmful_answer'])
                    
                unsafe_samples_unsafe_answer.append({'sample':formatted_input1, 'is_alllabel': False})
        
        unsafe_samples_safe_answer = unsafe_samples_safe_answer[:num_examples]
        unsafe_samples_unsafe_answer = unsafe_samples_unsafe_answer[:num_examples]

        # ======================= Retain ======================= #
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        orig_s = []
        for example in ds:
            messages = example["messages"]
            if len(messages) < 2: continue

            formatted_input = tokenizer.apply_chat_template(messages, tokenize=False).replace(tokenizer.bos_token, "")

            orig_s.append({'sample':formatted_input, 'is_alllabel': True})

            if len(orig_s) > num_examples:
                break
        
        self.safe_datalist = safe_samples + unsafe_samples_safe_answer + orig_s

        random.shuffle(self.safe_datalist)
        print("Orig s length:", len(self.safe_datalist))
        

        self.unsafe_datalist = unsafe_samples + unsafe_samples_unsafe_answer

        random.shuffle(self.unsafe_datalist)
        print("Orig s length:", len(self.unsafe_datalist))

        min_length = min(len(self.safe_datalist), len(self.unsafe_datalist))
        self.safe_datalist = self.safe_datalist[:min_length]
        self.unsafe_datalist = self.unsafe_datalist[:min_length]
            
        # padding right
        self.tokenizer.padding_side = "right"

    def __len__(self):
        return len(self.unsafe_datalist)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Retrieve a sample from the dataset
        sample = self.safe_datalist[i]
        is_alllabel = sample['is_alllabel']
        sample = sample['sample']
        
        # Define tokenization kwargs
        tokenized_kwargs = dict(max_length=self.max_length, truncation=True, return_tensors="pt")

        # Ultrachat processing: next token prediction task
        if is_alllabel:
            # Tokenize the input and treat the entire input as both input_ids and labels
            tokenized_inputs = self.tokenizer(sample.replace('<SEPARATOR>', self.sep_token), **tokenized_kwargs)
            labels = tokenized_inputs['input_ids'].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100  # Ensure no loss is computed on padding tokens

        else:
            # Process other datasets, where only the response part is used for CE loss
            # Safe response
            safe_request, safe_response = sample.split('<SEPARATOR>')
            tokenized_inputs = self.tokenizer(safe_request + self.sep_token + safe_response, **tokenized_kwargs)
            
            labels = tokenized_inputs['input_ids'].clone()
            
            # Mask the user's input part in labels (before <SEPARATOR>)
            separator_index = len(self.tokenizer(safe_request, **tokenized_kwargs)['input_ids'][0])
            labels[0][:separator_index] = -100  # Mask user prompt part
        
        safe_input_ids = tokenized_inputs['input_ids'].squeeze(0)
        safe_labels = labels.squeeze(0)

        sample = self.unsafe_datalist[i]
        is_alllabel = sample['is_alllabel']
        sample = sample['sample']
        
        # Process other datasets, where only the response part is used for CE loss
        # Safe response
        safe_request, safe_response = sample.split('<SEPARATOR>')
        tokenized_inputs = self.tokenizer(safe_request + self.sep_token + safe_response, **tokenized_kwargs)
        
        labels = tokenized_inputs['input_ids'].clone()

        # Mask the user's input part in labels (before <SEPARATOR>)
        separator_index = len(self.tokenizer(safe_request, **tokenized_kwargs)['input_ids'][0])
        labels[0][:separator_index] = -100  # Mask user prompt part

        
        unsafe_input_ids = tokenized_inputs['input_ids'].squeeze(0)
        unsafe_labels = labels.squeeze(0)
        
        return {
            'safe_input_ids': safe_input_ids,
            'safe_labels': safe_labels,

            'unsafe_input_ids': unsafe_input_ids,
            'unsafe_labels': unsafe_labels,
        }
