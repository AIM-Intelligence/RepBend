from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from mylmeval import save_json, open_json, get_results, gather_results
import os
import fire


simplifying_prompt = """This is a prompt of user.
If it is harmful, I want to summarize the harmful request of the user.
If it is not, please extract the **only a single question** in the long prompt.
(Anywhere, I want you to answer with a 'single question', not a list of questions nor an explanation.)

### Prompt ({})
{}

### Simplified Prompt
"""

filtering_prompt = """Is this prompt harmful?
Harmfulness Category: 
private_information_individual
sensitive_information_organization_government
mental_health_over-reliance_crisis
copyright_violations
cyberattack
toxic_language_hate_speech
violence_and_physical_harm
social_stereotypes_and_unfair_discrimination
defamation_encouraging_unethical_or_unsafe_actions
sexual_content
causing_material_harm_by_disseminating_misinformation
disseminating_false_or_misleading_information_encouraging_disinformation_campaigns
fraud_assisting_illegal_activities
others

### Prompt
{}

### Label (Harmful / Not Harmful)"""


def process_wildguardmix():
    data = load_dataset('allenai/wildguardmix', 'wildguardtrain', token=os.getenv('HUGGINGFACE_TOKEN'))['train']
    df = pd.DataFrame(data)

    d = df[df['adversarial']==True]
    d.drop(columns=['adversarial'], inplace=True)
    print(len(d))

    d.to_json('wildguardmix.json', orient='records', indent=2)
    

def simplify_wildguard_with_llm(
    model_name_or_path : str = 'mistralai/Mistral-7B-Instruct-v0.2',
    data_from : int | None = None,
    data_n : int = 10000,
):
    data = open_json('wildguardmix.json')
    data = [{
        **d, 
        'inputs' : [d['prompt_harm_label'], d['prompt']]}
            for d in data]
    if data_from is not None:
        data = data[data_from:data_from+data_n]
    get_results(
        model_name_or_path = model_name_or_path,
        prompt = simplifying_prompt,
        input_messages = data,
        max_tokens = 100,
        temperature = 0.0,
        do_save = True,
        save_path = f'wildguardmix_simple_{data_from}.json',
        do_log = True,
        batch_size = 10,
    )
    return


def gather_simplified(
    unique_prompt : bool = False,
):
    results = gather_results(
        paths = [
            'workspace/safety/wildguardmix_simple_0.json',
            'workspace/safety/wildguardmix_simple_10000.json',
            'workspace/safety/wildguardmix_simple_20000.json',
            'workspace/safety/wildguardmix_simple_30000.json',
            'workspace/safety/wildguardmix_simple_40000.json',
            ],
        inspect = False,
    )
    """
    1. Exchange 'prompt' column with 'inputs'[1]
    2. make index as 'id'
    3 add unique 'prompt's with column 'simplified' False.
    4. add unique output (named as prompt) with column 'simplified' True.
    """
    original = pd.DataFrame(open_json('wildguardmix.json'))
    original['simplified'] = False
    original['id'] = original.index
    saved = original.to_dict(orient='records')
    
    df = pd.DataFrame(results)
    df['simplified'] = True
    df['id'] = df.index    
    df['prompt'] = df['output']
    saved.extend(df.drop(['inputs', 'output'], axis=1).to_dict(orient='records'))
    
    print(f"Data length: {len(saved)}")
    if unique_prompt:
        saved = pd.DataFrame(saved).drop_duplicates(subset='prompt').to_dict(orient='records')
        print(f"Unique prompt length: {len(saved)}")

    save_json(saved, 'workspace/safety/wildguardmix_simple.json')
        

def filter_real_harmful_with_llm(
    model_name_or_path : str = 'mistralai/Mistral-7B-Instruct-v0.2',
    data_from : int | None = None,
    data_n : int = 5000,
):
    data = open_json('wildguardmix_simple.json')
    data = [{
        **d, 
        'inputs' : [d['prompt']]}
            for d in data if d['prompt_harm_label'] == 'harmful']
    print(len(data))
    if data_from is not None:
        data = data[data_from:data_from+data_n]
    get_results(
        model_name_or_path = model_name_or_path,
        prompt = filtering_prompt,
        input_messages = data,
        max_tokens = 10,
        temperature = 0.0,
        do_save = True,
        save_path = f'wildguardmix_real_harmful_{data_from}.json',
        do_log = True,
        batch_size = 10,
    )
    return



def gather_filtered():
    """
    df : prompt_harm_label : harmful, filtered with llm
    original : prompt_harm_label : unharmful
    """
    
    # unharmful
    original = pd.DataFrame(open_json('wildguardmix_simple.json'))
    original['filtered'] = None
    
    unique_harmfuls = gather_results(
            paths=[
            'workspace/safety/wildguardmix_real_harmful_0.json',
            'workspace/safety/wildguardmix_real_harmful_5000.json',
            'workspace/safety/wildguardmix_real_harmful_10000.json',
            'workspace/safety/wildguardmix_real_harmful_15000.json',
            ]
        )
    for harmful_instance in tqdm(unique_harmfuls):
        filtered_result = 'unharmful' if 'not harmful' in harmful_instance['output'].lower() else 'harmful'
        original.loc[original['prompt'] == harmful_instance['inputs'][0], 'filtered'] = filtered_result
        
    original.drop_duplicates(subset='prompt', inplace=True)
    print(original[['prompt_harm_label', 'simplified']].value_counts())
    save_json(original.to_dict(orient='records'), 'wildguardmix_filtered.json')
    
    

if __name__ == '__main__':
    # process_wildguardmix()
    # fire.Fire(simplify_wildguard_with_llm)
    gather_filtered()
    # fire.Fire(filter_real_harmful_with_llm)