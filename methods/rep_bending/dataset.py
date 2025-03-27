import random
from typing import Dict

import jsonlines
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset

random.seed(0)

class RepBendingDataset(Dataset):
    
    def __init__(self, 
                tokenizer: transformers.PreTrainedTokenizer, 
                num_examples,
                mode,
                max_length,
                model_name_or_path,
                dataset_path,
                split=None,
                is_online=False,
                ):
        super(RepBendingDataset, self).__init__()

        self.model_name_or_path = model_name_or_path.lower()
        self.max_length = max_length
        self.tokenizer = tokenizer

        def set_tags_templates():
            
            one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"

            # ================ Model and Template Config  ================
            # Default configs
            sep_token = ""
            user_tag, assistant_tag = None, None
            if 'llama-3' in self.model_name_or_path:
                print("USING LLAMA TEMPLATE")
                user_tag="<|start_header_id|>user<|end_header_id|>\n\n"
                assistant_tag="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            elif 'mistral' in self.model_name_or_path:
                print("USING MISTRAL TEMPLATE")
                # fix spacing issue in template
                self.tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
                user_tag="[INST] "
                assistant_tag=" [/INST]"
                sep_token = " "
            elif 'qwen' in self.model_name_or_path:
                user_tag="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                assistant_tag="<|im_end|>\n<|im_start|>assistant\n"
            else:
                raise NotImplementedError(f"Config {self.model_name_or_path} not found")
            
            assert user_tag and assistant_tag, "user_tag/assistant_tag not defined"

            self.user_tag = user_tag
            self.assistant_tag = assistant_tag
            self.sep_token = sep_token

            return one_shot_template
        
        def sample_from_wildguardmix(ds):
            
            data_safe_samples = []  # safe prompt, safe prompt safe response, or unsafe prompt safe response
            data_unsafe_samples = []  # unsafe prompt, or unsafe prompt and unsafe response
            
            for d in ds['train']:
                # ["request_last", "request_all", "request_response", "response_all", "response_last"]
                if mode == 'request_last' or mode == 'request_all' or is_online:
                    if (d['prompt_harm_label'] == 'unharmful'): # s
                        formatted_input = one_shot_template.format(
                        user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                        instruction=d['prompt'], response=d['response'])
                        
                        data_safe_samples.append(formatted_input)
                        
                    elif (d['prompt_harm_label'] == 'harmful'): # u
                        formatted_input = one_shot_template.format(
                        user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                        instruction=d['prompt'], response=d['response'])
                        
                        data_unsafe_samples.append(formatted_input)
                    
                elif mode == 'request_response' or mode == 'response_all' or mode =='response_last' or mode =='response_one' or mode =='response_R':
                    if (d['prompt_harm_label'] == 'unharmful' and d['response_refusal_label'] == 'compliance' and d['response_harm_label'] == 'unharmful') or (d['prompt_harm_label'] == 'harmful' and d['response_refusal_label'] == 'refusal' and d['response_harm_label'] == 'unharmful'):  ## ss, us
                        formatted_input = one_shot_template.format(
                        user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                        instruction=d['prompt'], response=d['response'])
                        
                        data_safe_samples.append(formatted_input)
                        
                    elif (d['prompt_harm_label'] == 'harmful' and d['response_refusal_label'] == 'compliance'): # uu
                        formatted_input = one_shot_template.format(
                        user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                        instruction=d['prompt'], response=d['response'])
                        
                        data_unsafe_samples.append(formatted_input)
        
            print(len(data_safe_samples))
            print(len(data_unsafe_samples))
            random.seed(0)
            self.data_safe_samples = random.sample(data_safe_samples, num_examples)
            self.data_unsafe_samples = random.sample(data_unsafe_samples, num_examples)

            print("safe length:", len(self.data_safe_samples))
            print("safe[0]", self.data_safe_samples[0])
            print("unsafe length:", len(self.data_unsafe_samples))
            print("unsafe[0]", self.data_unsafe_samples[0])

        def sample_from_wildjailbreak():
            train_data = []
            with jsonlines.open(f"./wildjailbreak.jsonl") as f:
                for line in f.iter():
                    train_data.append(line)
                    
            random.shuffle(train_data)
            
            unsafe_prompt_unsafe_answer = []
            unsafe_prompt_safe_answer = []
            for d in train_data:
                if d['prompt_type'] == 'harmful':
                    formatted_input1 = one_shot_template.format(
                        user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                        instruction=d['prompt'], response=d['harmful_answer'])
                        
                    unsafe_prompt_unsafe_answer.append(formatted_input1)
                    
                    formatted_input2 = one_shot_template.format(
                        user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                        instruction=d['prompt'], response=d['harmless_answer'])
                        
                    unsafe_prompt_safe_answer.append(formatted_input2)
            
            self.unsafe_prompt_pair_unsafe_answer = unsafe_prompt_unsafe_answer[:num_examples]
            self.unsafe_prompt_pair_safe_answer = unsafe_prompt_safe_answer[:num_examples]
            print("pair unsafe", len(self.unsafe_prompt_pair_unsafe_answer))
            print("pair unsafe[0]", self.unsafe_prompt_pair_unsafe_answer[0])
            print("pair safe", len(self.unsafe_prompt_pair_safe_answer))
            print("pair safe[0]", self.unsafe_prompt_pair_safe_answer[0])
        def  sample_from_ultrachat():
            ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
            orig_s = []
            for example in ds:
                messages = example["messages"]
                if len(messages) < 2: continue

                formatted_input = tokenizer.apply_chat_template(messages, tokenize=False)
                if tokenizer.bos_token:
                    formatted_input = formatted_input.replace(tokenizer.bos_token, "")

                orig_s.append(formatted_input)

                if len(orig_s) > num_examples:
                    break
            self.retain_set = orig_s
            random.shuffle(self.retain_set)
            print("orig_s_retain[0]", orig_s[0])
            print("Orig s length:", len(self.retain_set))

        one_shot_template = set_tags_templates()
        ds = load_dataset(dataset_path, split)

        if 'wildguardmix' in dataset_path:
            sample_from_wildguardmix(ds)

        sample_from_wildjailbreak()
        sample_from_ultrachat()
        
        # padding right
        self.tokenizer.padding_side = "right"

    def __len__(self):
        return min(len(self.data_safe_samples), len(self.data_unsafe_samples))
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        
        data_safe_samples = self.data_safe_samples[i]
        data_unsafe_samples = self.data_unsafe_samples[i]
        retain_text = self.retain_set[i]
        unsafe_request_unsafe_response = self.unsafe_prompt_pair_unsafe_answer[i]
        unsafe_request_safe_response = self.unsafe_prompt_pair_safe_answer[i]

        split_tokenized_kwargs = dict(max_length=int(self.max_length/2), padding='max_length', truncation=True, return_tensors="pt")
        tokenized_kwargs = dict(max_length=int(self.max_length), padding='max_length', truncation=True, return_tensors="pt")
        
        # === split to [request, response] shape [512,512] to support different mask configs ===
        
        ########### 'safe' sample ########
        safe_sample_request, safe_sample_response = data_safe_samples.split('<SEPARATOR>')
        self.tokenizer.padding_side = "left"
        tokenized_safe_sample_request = self.tokenizer(safe_sample_request, **split_tokenized_kwargs)
        self.tokenizer.padding_side = "right"
        tokenized_safe_sample_response = self.tokenizer(safe_sample_response, add_special_tokens=False, **split_tokenized_kwargs)
        self.tokenizer.padding_side = "left"

        combined_ids_safe_sample = torch.cat([tokenized_safe_sample_request["input_ids"], tokenized_safe_sample_response["input_ids"]], dim=1)
        combined_mask_safe_sample = torch.cat([tokenized_safe_sample_request["attention_mask"], tokenized_safe_sample_response["attention_mask"]], dim=1)
        
        ########### 'unsafe' sample ########
        unsafe_sample_request, unsafe_sample_response = data_unsafe_samples.split('<SEPARATOR>')
        self.tokenizer.padding_side = "left"
        tokenized_unsafe_sample_request = self.tokenizer(unsafe_sample_request, **split_tokenized_kwargs)
        self.tokenizer.padding_side = "right"
        tokenized_unsafe_sample_response = self.tokenizer(unsafe_sample_response, add_special_tokens=False, **split_tokenized_kwargs)
        self.tokenizer.padding_side = "left"

        combined_ids_unsafe_sample = torch.cat([tokenized_unsafe_sample_request["input_ids"], tokenized_unsafe_sample_response["input_ids"]], dim=1)
        combined_mask_unsafe_sample = torch.cat([tokenized_unsafe_sample_request["attention_mask"], tokenized_unsafe_sample_response["attention_mask"]], dim=1)
        
        ########### unsafe request paired with unsafe & safe response ########
        unsafe_request1, unsafe_response = unsafe_request_unsafe_response.split('<SEPARATOR>')
        unsafe_request2, safe_response = unsafe_request_safe_response.split('<SEPARATOR>')
        assert unsafe_request1 == unsafe_request2  # request part of us and uu are the same
        
        self.tokenizer.padding_side = "left"
        tokenized_unsafe_request = self.tokenizer(unsafe_request1, **split_tokenized_kwargs)
        self.tokenizer.padding_side = "right"
        tokenized_unsafe_response_for_unsafe_request = self.tokenizer(unsafe_response, add_special_tokens=False, **split_tokenized_kwargs)
        tokenized_safe_response_for_unsafe_request = self.tokenizer(safe_response, add_special_tokens=False, **split_tokenized_kwargs)
        self.tokenizer.padding_side = "left"

        combined_ids_unsafe_request_unsafe_response = torch.cat([tokenized_unsafe_request["input_ids"], tokenized_unsafe_response_for_unsafe_request["input_ids"]], dim=1)
        combined_mask_unsafe_request_unsafe_response = torch.cat([tokenized_unsafe_request["attention_mask"], tokenized_unsafe_response_for_unsafe_request["attention_mask"]], dim=1)
        
        combined_ids_unsafe_request_safe_response = torch.cat([tokenized_unsafe_request["input_ids"], tokenized_safe_response_for_unsafe_request["input_ids"]], dim=1)
        combined_mask_unsafe_request_safe_response = torch.cat([tokenized_unsafe_request["attention_mask"], tokenized_safe_response_for_unsafe_request["attention_mask"]], dim=1)
        
        # ========== Retain Inputs ===========
        tokenized_inputs_retain = self.tokenizer(retain_text.replace('<SEPARATOR>', self.sep_token), **tokenized_kwargs)
        
        return dict(
            # usafe request pair unsafe response
            ids_unsafe_request_unsafe_response=combined_ids_unsafe_request_unsafe_response,
            mask_unsafe_request_unsafe_response=combined_mask_unsafe_request_unsafe_response,
            # unsafe request pair safe response
            ids_unsafe_request_safe_response=combined_ids_unsafe_request_safe_response,
            mask_unsafe_request_safe_response=combined_mask_unsafe_request_safe_response,
            # unsafe request
            ids_unsafe_request=tokenized_unsafe_request['input_ids'],
            mask_unsafe_request=tokenized_unsafe_request['attention_mask'],
            # unsafe response for unsafe request 
            ids_unsafe_response_for_unsafe_request=tokenized_unsafe_response_for_unsafe_request['input_ids'],
            mask_unsafe_response_for_unsafe_request=tokenized_unsafe_response_for_unsafe_request['attention_mask'],
            # safe response for unsafe request 
            ids_safe_response_for_unsafe_request=tokenized_safe_response_for_unsafe_request['input_ids'],
            mask_safe_response_for_unsafe_request=tokenized_safe_response_for_unsafe_request['attention_mask'],
            # safe sample
            ids_safe_sample=combined_ids_safe_sample,
            mask_safe_sample=combined_mask_safe_sample,
            ids_safe_sample_request=tokenized_safe_sample_request['input_ids'],
            mask_safe_sample_request=tokenized_safe_sample_request['attention_mask'],
            ids_safe_sample_response=tokenized_safe_sample_response['input_ids'],
            mask_safe_sample_response=tokenized_safe_sample_response['attention_mask'],
            # unsafe sample
            ids_unsafe_sample=combined_ids_unsafe_sample,
            mask_unsafe_sample=combined_mask_unsafe_sample,
            ids_unsafe_sample_request=tokenized_unsafe_sample_request['input_ids'],
            mask_unsafe_sample_request=tokenized_unsafe_sample_request['attention_mask'],
            ids_unsafe_sample_response=tokenized_unsafe_sample_response['input_ids'],
            mask_unsafe_sample_response=tokenized_unsafe_sample_response['attention_mask'],
            # retain set
            ids_retain=tokenized_inputs_retain['input_ids'],
            mask_retain=tokenized_inputs_retain['attention_mask'],
            
            unsafe_sample_request = unsafe_sample_request,
            unsafe_request = unsafe_request1
        )
    