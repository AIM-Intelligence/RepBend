from transformers import AutoTokenizer, AutoModelForCausalLM
import jsonlines
from tqdm import tqdm

harmprompts = []
harmlessprompts = []
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
title = "baseline"

# title = "ours"
with jsonlines.open('wildjailbreak.jsonl') as reader:
    # only load 1000 harmful prompts and harmless prompts
    for obj in reader:
        if obj['prompt_type'] == 'harmful':
            if len(harmprompts) < 500:
                harmprompts.append(obj)
        else:
            if len(harmlessprompts) < 500:
                harmlessprompts.append(obj)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="left",
    model_max_length=512,
)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to("cuda")
model.eval()  # Set model to evaluation mode

tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

# Define batch processing function
def process_batch(prompts, batch_size=20, max_length=128):
    results = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i + batch_size]
        prompt_texts = ["[Prompt] " + obj['prompt'] + "[Answer] " for obj in batch]
        
        # Tokenize batch
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        
        # Generate outputs
        outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
        
        # Decode and store results
        for j, output in enumerate(outputs):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            extracted_answer = generated_text.split("[Answer]")[1].strip()
            batch[j][f'answer'] = extracted_answer
            results.append(batch[j])
    return results

# Process harmful prompts
harmprompts = process_batch(harmprompts, batch_size=20, max_length=128)

# Process harmless prompts
harmlessprompts = process_batch(harmlessprompts, batch_size=20, max_length=128)

# Write results to JSONL file
with jsonlines.open(f'wildjailbreak_{title}.jsonl', 'w') as writer:
    writer.write_all(harmprompts + harmlessprompts)
