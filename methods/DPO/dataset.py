import random
from typing import Dict

import jsonlines
import torch
import transformers
from methods.dataset import CustomDataset

random.seed(0)

class DPODataset(CustomDataset):
    def after_init(self, tokenizer: transformers.PreTrainedTokenizer, num_examples, max_length, model_name_or_path, dataset_path, split=None, mode=None, train_type=None):
        train_data = []
        with jsonlines.open(f"./wildjailbreak.jsonl") as f:
            for line in f.iter():
                train_data.append(line)
                
        random.shuffle(train_data)
        
        datalist = []
        for d in train_data:
            if d['prompt_type'] == 'harmful':
                formatted_input1 = self.one_shot_template.format(
                    user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                    instruction=d['prompt'], response=d['harmful_answer'])
                
                formatted_input2 = self.one_shot_template.format(
                    user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                    instruction=d['prompt'], response=d['harmless_answer'])
                
                datalist.append({'safe':formatted_input2, 'unsafe':formatted_input1})
                    
        self.datalist = datalist[:num_examples]
        # self.datalist = datalist

        # padding right
        self.tokenizer.padding_side = "right"

    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Retrieve a sample from the dataset
        sample = self.datalist[i]
        safe_sample = sample['safe']
        unsafe_sample = sample['unsafe']
        
        # Define tokenization kwargs
        tokenized_kwargs = dict(max_length=self.max_length, truncation=True, return_tensors="pt")
        # split_tokenized_kwargs = dict(max_length=int(self.max_length/2), truncation=True, return_tensors="pt")

        # Process other datasets, where only the response part is used for CE loss
        # Safe response
        prompt, safe_response = safe_sample.split('<SEPARATOR>')
        _, unsafe_response = unsafe_sample.split('<SEPARATOR>')
        self.tokenizer.padding_side = "left"
        tokenized_prompt = self.tokenizer(prompt, **tokenized_kwargs)
        self.tokenizer.padding_side = "right"
        tokenized_safe_res = self.tokenizer(safe_response, **tokenized_kwargs)
        tokenized_unsafe_res = self.tokenizer(unsafe_response, **tokenized_kwargs)
        return {
            'prompt_input_ids': tokenized_prompt['input_ids'].squeeze(0),
            'chosen_input_ids': tokenized_safe_res['input_ids'].squeeze(0),
            'rejected_input_ids': tokenized_unsafe_res['input_ids'].squeeze(0),
        }
