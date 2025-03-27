import random

import torch
import torch.nn.functional as F
from transformers import StoppingCriteria


# Custom stopping criteria based on specific stop token sequences
class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        # Ensure stop_id_sequences is a list of lists
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if all sequences in the batch meet stopping criteria
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):  # Iterate over each sequence in the batch
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:  # Check for each stop sequence
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:  # Compare trailing tokens
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)  # Stop if all sequences meet criteria

@torch.no_grad()
def generate_completions(model, tokenizer, input_ids, attention_mask, model_name_or_path: str, stop_id_sequences=None, **generation_kwargs):
    """
    Generate text completions using a given model and tokenizer.
    Handles model-specific stop sequences, and removes prompt tokens from outputs.
    """
    generations = []

    # Add special stop token for Llama3 models if not already provided
    if "llama-3" in model_name_or_path.lower() or "llama3" in model_name_or_path.lower():
        llama3_stop_id = [128009]
        stop_id_sequences = [] if stop_id_sequences is None else stop_id_sequences
        if llama3_stop_id not in stop_id_sequences:
            stop_id_sequences.append(llama3_stop_id)

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)

    # Enable caching for efficient generation
    model.config.use_cache = True
    batch_outputs_token = model.generate(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
        **generation_kwargs
    )

    # Post-process to handle cases where stop sequences are included in outputs
    if stop_id_sequences:
        for output_idx in range(batch_outputs_token.shape[0]):  # Iterate through outputs
            for token_idx in range(input_ids.shape[1], batch_outputs_token.shape[1]):
                if any(batch_outputs_token[output_idx, token_idx: token_idx + len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                    # Replace tokens after the stop sequence with padding
                    batch_outputs_token[output_idx, token_idx:] = tokenizer.pad_token_id
                    break

    # Decode the outputs into strings, and remove prompt tokens
    batch_outputs = tokenizer.batch_decode(batch_outputs_token, skip_special_tokens=True)
    batch_prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    # Duplicate prompts to match the number of return sequences
    batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
    batch_generations = [output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)]

    generations += batch_generations
    # Disable caching to free memory
    model.config.use_cache = False
    torch.cuda.empty_cache()
    return generations, batch_outputs_token

def generate_online_samples(model, inputs, tokenizer, ids_safe_sample_request, ids_unsafe_sample_request, ids_unsafe_request, 
                            mask_safe_sample_request, mask_unsafe_sample_request, mask_unsafe_request,
                            classifier, classifier_output_field, desired_outputs, model_name_or_path,
                            beta, gamma, eta
                            ):
    """
    Generates online samples, processes unsafe requests, classifies outputs, and splits them into safe/unsafe categories.
    """
    # Initialize output variables
    safe_samples, unsafe_samples, unsafe_request_unsafe_response, unsafe_request_safe_response = None, None, None, None
    unsafe_sample_request_txt = inputs.get("unsafe_sample_request")
    unsafe_request_txt = inputs.get("unsafe_request")
    # Set model to evaluation mode
    model.eval()
    
    # Predefined response templates for safe and refusal scenarios
    # these responses have all the same length in mistral tokenizer
    following_responses = ["Of course. As you have requested, ", 
                           "Here is the answer for your instruction to ",
                           "Sure, I will provide the required information ", 
                           "Following the instruction, here is my answer "]
    refusal_responses = ["Unfortunately, I must decline this instruction as ",
                        "Regrettably, I cannot proceed due ",
                        "As AI, I cannot follow this request ",
                        "I cannot give an answer your instruction because "]
    
    # Number of unsafe requests
    M = ids_unsafe_request.size(0)
    # Randomly select responses for each unsafe request
    selected_following_responses = random.choices(following_responses, k=M)
    selected_refusal_responses = random.choices(refusal_responses, k=M)
    
    new_ids_unsafe_request_following = []
    new_ids_unsafe_request_refusal = []
    new_mask_unsafe_request_following = []
    new_mask_unsafe_request_refusal = []
    new_unsafe_request_txt_following = []
    new_unsafe_request_txt_refusal = []
    for i in range(M):
        # Extract padding info for alignment
        request_id = ids_unsafe_request[i]
        request_mask = mask_unsafe_request[i]
        num_padding = (request_mask == 0).sum().item()

        # Tokenize the selected responses
        tokenized_following = tokenizer(selected_following_responses[i], return_tensors="pt", truncation=True, padding=False)["input_ids"].squeeze(0).to(request_id.device)
        tokenized_refusal = tokenizer(selected_refusal_responses[i], return_tensors="pt", truncation=True, padding=False)["input_ids"].squeeze(0).to(request_id.device)

        concatenated_following_id= torch.cat((request_id, tokenized_following), dim=0)
        concatenated_following_mask = torch.cat((request_mask, torch.ones(tokenized_following.size(0)).to(request_id.device)), dim=0)
        concatenated_refusal_id = torch.cat((request_id, tokenized_refusal), dim=0)
        concatenated_refusal_mask = torch.cat((request_mask, torch.ones(tokenized_refusal.size(0)).to(request_id.device)), dim=0)
        # Add to new lists
        new_ids_unsafe_request_following.append(concatenated_following_id)
        new_ids_unsafe_request_refusal.append(concatenated_refusal_id)
        new_mask_unsafe_request_following.append(concatenated_following_mask)
        new_mask_unsafe_request_refusal.append(concatenated_refusal_mask)
        
    # Stack requests and masks for batch processing
    new_ids_unsafe_request_following = torch.stack(new_ids_unsafe_request_following)
    new_ids_unsafe_request_refusal = torch.stack(new_ids_unsafe_request_refusal)
    new_mask_unsafe_request_following = torch.stack(new_mask_unsafe_request_following)
    new_mask_unsafe_request_refusal = torch.stack(new_mask_unsafe_request_refusal)

    # Concatenate all input IDs and attention masks
    request_input_ids_safe_unsafe = torch.cat((ids_safe_sample_request, ids_unsafe_sample_request), dim=0)
    request_attention_mask_safe_unsafe = torch.cat((mask_safe_sample_request, mask_unsafe_sample_request), dim=0)

    # Generate outputs using the model
    outputs_safe_unsafe, output_token_safe_unsafe = generate_completions(model.module, tokenizer, request_input_ids_safe_unsafe, request_attention_mask_safe_unsafe, model_name_or_path,
        max_new_tokens=256,
        do_sample=False,
        top_p=1.0,
    )
    
    request_input_ids_pair = torch.cat((new_ids_unsafe_request_following, new_ids_unsafe_request_refusal), dim=0)
    request_attention_mask_pair = torch.cat((new_mask_unsafe_request_following, new_mask_unsafe_request_refusal), dim=0)
    
    outputs_pair, output_token_pair = generate_completions(model.module, tokenizer, request_input_ids_pair, request_attention_mask_pair, model_name_or_path,
        max_new_tokens=256,
        do_sample=False,
        top_p=1.0,
    )
    
    # make length the same by padding
    safe_unsafe_num = output_token_safe_unsafe.shape[0]
    outputs = [token for token in output_token_safe_unsafe] + [token for token in output_token_pair]
    outputs = torch.nn.utils.rnn.pad_sequence(
            outputs,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)[:, :tokenizer.model_max_length]
    output_token_safe_unsafe = outputs[:safe_unsafe_num]
    output_token_pair = outputs[safe_unsafe_num:]
        
    # Separate generated outputs into safe and unsafe samples
    safe_samples = output_token_safe_unsafe[:len(ids_safe_sample_request)]
    unsafe_samples = output_token_safe_unsafe[len(ids_safe_sample_request):]
    unsafe_samples_unsafe_following = output_token_pair[:len(new_ids_unsafe_request_following)]
    unsafe_samples_unsafe_refusal = output_token_pair[len(new_ids_unsafe_request_following):]

    # Evaluate the generated responses using a classifier
    unsafe_text = unsafe_sample_request_txt + unsafe_request_txt + unsafe_request_txt
    selected_responses = ["",]*len(unsafe_sample_request_txt) + selected_following_responses + selected_refusal_responses
    unsafe_outputs = outputs_safe_unsafe[len(ids_safe_sample_request):] + outputs_pair
    evaluator_inputs = []
    for i, (model_input, completion) in enumerate(zip(unsafe_text, unsafe_outputs)):
        evaluator_inputs.append({'prompt': model_input, 'response': selected_responses[i] + completion})
    evaluation_outputs = classifier.classify(evaluator_inputs)
    
    evaluation_outputs_unsafe_samples_ = evaluation_outputs[:len(ids_unsafe_sample_request)]
    evaluation_outputs_unsafe_following = evaluation_outputs[len(ids_unsafe_sample_request):len(ids_unsafe_sample_request)+len(new_ids_unsafe_request_following)]
    evaluation_outputs_unsafe_refusal = evaluation_outputs[len(ids_unsafe_sample_request)+len(new_ids_unsafe_request_following):]
    
    
    new_unsafe_samples = []
    # Classify response for ids_unsafe_sample_request as safe/unsafe
    for unsafe_sample, pred in zip(unsafe_samples, evaluation_outputs_unsafe_samples_):
        label = getattr(pred, classifier_output_field)
        if label == desired_outputs:
            new_unsafe_samples.append(unsafe_sample)
        else:
            len_diff = safe_samples.shape[1] - unsafe_sample.shape[0]
            if len_diff == 0:
                safe_samples = torch.cat((safe_samples, unsafe_sample.unsqueeze(0)), dim=0)
            elif len_diff > 0:
                safe_samples = torch.cat((safe_samples, F.pad(unsafe_sample, (tokenizer.pad_token_id, len_diff), "constant", 0).unsqueeze(0)), dim=0)
            else:
                safe_samples = torch.cat((F.pad(safe_samples, (tokenizer.pad_token_id, -len_diff), "constant", 1), unsafe_sample.unsqueeze(0)), dim=0)
            
    
    pair_unsafe_samples = []
    pair_safe_samples = []
    for unsafe_sample_following, unsafe_sample_refusal, pred_following, pred_refusal in zip(unsafe_samples_unsafe_following, unsafe_samples_unsafe_refusal, evaluation_outputs_unsafe_following, evaluation_outputs_unsafe_refusal):
        label_following = getattr(pred_following, classifier_output_field)
        label_refusal = getattr(pred_refusal, classifier_output_field)
        if label_following == desired_outputs and label_refusal != desired_outputs: # if we get paired response
            pair_unsafe_samples.append(unsafe_sample_following)
            pair_safe_samples.append(unsafe_sample_refusal)
        else:
            if label_following == desired_outputs:
                new_unsafe_samples.append(unsafe_sample_following)
            else:
                # safe_samples = torch.cat((safe_samples, unsafe_sample_following.unsqueeze(0)), dim=0)
                len_diff = safe_samples.shape[1] - unsafe_sample_following.shape[0]
                if len_diff == 0:
                    safe_samples = torch.cat((safe_samples, unsafe_sample.unsqueeze(0)), dim=0)
                elif len_diff > 0:
                    safe_samples = torch.cat((safe_samples, F.pad(unsafe_sample_following, (tokenizer.pad_token_id, len_diff), "constant", 0).unsqueeze(0)), dim=0)
                else:
                    safe_samples = torch.cat((F.pad(safe_samples, (tokenizer.pad_token_id, -len_diff), "constant", 1), unsafe_sample_following.unsqueeze(0)), dim=0)
                
                
            if label_refusal == desired_outputs:
                new_unsafe_samples.append(unsafe_sample_refusal)
            else:
                # safe_samples = torch.cat((safe_samples, unsafe_sample_refusal.unsqueeze(0)), dim=0)
                len_diff = safe_samples.shape[1] - unsafe_sample_refusal.shape[0]
                if len_diff == 0:
                    safe_samples = torch.cat((safe_samples, unsafe_sample_refusal.unsqueeze(0)), dim=0)
                elif len_diff > 0:
                    safe_samples = torch.cat((safe_samples, F.pad(unsafe_sample_refusal, (tokenizer.pad_token_id, len_diff), "constant", 0).unsqueeze(0)), dim=0)
                else:
                    safe_samples = torch.cat((F.pad(safe_samples, (tokenizer.pad_token_id, -len_diff), "constant", 1), unsafe_sample_refusal.unsqueeze(0)), dim=0)
            

    if len(new_unsafe_samples) > 0:
        # unsafe_samples = torch.stack(new_unsafe_samples)
        unsafe_samples = torch.nn.utils.rnn.pad_sequence(new_unsafe_samples, batch_first=True, padding_value=tokenizer.pad_token_id)
    if len(pair_safe_samples) > 0:
        unsafe_request_safe_response = torch.nn.utils.rnn.pad_sequence(pair_safe_samples, batch_first=True, padding_value=tokenizer.pad_token_id)
        unsafe_request_unsafe_response = torch.nn.utils.rnn.pad_sequence(pair_unsafe_samples, batch_first=True, padding_value=tokenizer.pad_token_id)

        len_diff = safe_samples.shape[1] - unsafe_request_safe_response.shape[1]
        if len_diff == 0:
            safe_samples = torch.cat((safe_samples, unsafe_request_safe_response), dim=0)
        elif len_diff > 0:
            safe_samples = torch.cat((safe_samples, F.pad(unsafe_request_safe_response, (tokenizer.pad_token_id, len_diff), "constant", 1)), dim=0)
        else:
            safe_samples = torch.cat((F.pad(safe_samples, (tokenizer.pad_token_id, -len_diff), "constant", 1), unsafe_request_safe_response), dim=0)

        if len(new_unsafe_samples) == 0:
            unsafe_samples = unsafe_request_unsafe_response
        else:
            len_diff = unsafe_samples.shape[1] - unsafe_request_unsafe_response.shape[1]
            if len_diff == 0:
                unsafe_samples = torch.cat((unsafe_samples, unsafe_request_unsafe_response), dim=0)
            elif len_diff > 0:
                unsafe_samples = torch.cat((unsafe_samples, F.pad(unsafe_request_unsafe_response, (tokenizer.pad_token_id, len_diff), "constant", 1)), dim=0)
            else:
                unsafe_samples = torch.cat((F.pad(unsafe_samples, (tokenizer.pad_token_id, -len_diff), "constant", 1), unsafe_request_unsafe_response), dim=0)
        
    else: # no paired sample -> disable unsafe_safe loss
        eta = 0
    
    if unsafe_samples is None:
        beta=0
        gamma=0 
    
    if safe_samples is not None:
        print(f"safe_samples: {safe_samples.shape}")
    if unsafe_samples is not None:
        print(f"unsafe_samples: {unsafe_samples.shape}")
    if unsafe_request_safe_response is not None:
        print(f"paired samples: {unsafe_request_safe_response.shape}")
    
    return safe_samples, unsafe_samples, unsafe_request_safe_response, unsafe_request_unsafe_response, beta, gamma, eta
