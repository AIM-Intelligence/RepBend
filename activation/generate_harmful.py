from datasets import load_dataset
import pandas as pd
import os
import math
import fire
from mylmeval import get_results, open_json
from transformers import AutoTokenizer

EVILA_PROMPT = """### Instruction:
{}

### Response:"""


def run_process(
    data_dir : str = 'train.tsv',
    add_column : bool = True,
    start : int = 0,
    how_many : int = 50000,
):
    try:    
        data = load_dataset('allenai/wildjailbreak', 'train', token=os.getenv('HUGGINGFACE_TOKEN'))
    # if load_dataset gives error, download here https://huggingface.co/datasets/allenai/wildjailbreak/tree/main/train
    except:
        data = pd.read_csv(data_dir, sep='\t')
        data = data.drop_duplicates(subset='vanilla')
        print(data['data_type'].value_counts())
    
    data['prompt_type'] = data['data_type'].apply(lambda x: 'harmful' if 'harmful' in x else 'harmless')
    data['prompt'] = data['vanilla']
    data['harmless_answer'] = data['completion']
    data = data[['prompt', 'prompt_type', 'harmless_answer']]

    tokenizer = AutoTokenizer.from_pretrained('maywell/PiVoT-0.1-Evil-a', trust_remote_code=True)
    data.to_json('safety/wildjailbreak.jsonl', orient='records', lines=True)
    # harmless_average_length = sum([len(tokenizer.encode(r)) for r in list(data['harmless_answer'])]) / len(data)
    # harmful_max_tokens = 2 ** math.ceil(math.log2(harmless_average_length))
    # print(f"harmless avg : {harmless_average_length} harmful_tokens : {harmful_max_tokens}")
    harmful_max_tokens = 256
    temperature = 0.7
    
    if add_column:
        data['inputs'] = data['prompt'].apply(lambda x: [x])
        data = data[start:start+how_many]
        _ = get_results(
            model_path_or_name = 'maywell/PiVoT-0.1-Evil-a',
            prompt = EVILA_PROMPT,
            data = data.to_dict(orient='records'),
            max_tokens = harmful_max_tokens,
            temperature = temperature,
            do_save=True,
            do_log=True,
            save_path=f'wildjailbreak_annotated_{start}_{temperature}.json',
            batch_size = len(data)
        )
    
    

def gather():
    df = pd.read_json('/home/ubuntu/workspace/safety/wildjailbreak.jsonl', lines=True)
    new_columns = [
        'wildjailbreak_annotated_0_0.7.json',
        'wildjailbreak_annotated_50000_0.7.json',
        'wildjailbreak_annotated_100000_0.7.json',
    ]
    new_column_name = 'harmful_answer'
    new_df = pd.DataFrame(open_json(new_columns[0]) + open_json(new_columns[1]) + open_json(new_columns[2]))
    df[new_column_name] = new_df['output'].apply(lambda x: x.strip())
    df.to_json('wildjailbreak.jsonl', orient='records', lines=True)
    

if __name__ == '__main__':
    # fire.Fire(run_process)
    gather()