import random

import transformers
from torch.utils.data import Dataset

random.seed(0)

class CustomDataset(Dataset):
    def __init__(self, 
                tokenizer: transformers.PreTrainedTokenizer, 
                num_examples,
                max_length,
                model_name_or_path,
                dataset_path,
                split=None,
                mode=None,
                train_type=None,
                ):
        super(CustomDataset, self).__init__()

        self.before_init(tokenizer, num_examples, max_length, model_name_or_path, dataset_path, split=split, mode=mode, train_type=train_type)

        self.model_name_or_path = model_name_or_path.lower()
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"
        
        # ================ Model and Template Config  ================
        # Default configs
        self.sep_token = " "
        self.switch_select = [0]
        self.use_refusal_retain = False
        user_tag, assistant_tag = None, None
        if 'llama-3' in self.model_name_or_path:
            print("USING LLAMA TEMPLATE")
            user_tag="<|start_header_id|>user<|end_header_id|>\n\n"
            assistant_tag="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            self.switch_select = [0, 1]
            self.use_refusal_retain = True
        elif 'mistral' in self.model_name_or_path:
            print("USING MISTRAL TEMPLATE")
            # fix spacing issue in template
            self.tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
            user_tag="[INST] "
            assistant_tag=" [/INST]"
            sep_token = " "
        else:
            raise NotImplementedError(f"Config {self.model_name_or_path} not found")
        
        assert user_tag and assistant_tag, "user_tag/assistant_tag not defined"

        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.sep_token = sep_token
        self.switch_select = [0, 1]

        self.after_init(tokenizer, num_examples, max_length, model_name_or_path, dataset_path, split=split, mode=mode, train_type=train_type)

    def before_init(self, tokenizer: transformers.PreTrainedTokenizer, num_examples, max_length, model_name_or_path, dataset_path, split=None, mode=None, train_type=None):
        pass

    def after_init(self, tokenizer: transformers.PreTrainedTokenizer, num_examples, max_length, model_name_or_path, dataset_path, split=None, mode=None, train_type=None):
        pass
        
    def __len__(self):
        raise NotImplementedError("This method is not implemented in the base class")
    
    def __getitem__(self, idx):
        raise NotImplementedError("This method is not implemented in the base class")

