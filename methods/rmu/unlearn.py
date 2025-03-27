import datetime
import os

import numpy as np
import torch
import tqdm as tqdm
from transformers import AdamW

from utils import forward_with_cache, get_data, get_params, load_model


def run_rmu(
    updated_model,
    frozen_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    args,
):
    rmu_config = vars(args)
    print("====rmu Config====")
    print("\n".join(f"{k}={v}" for k,v in rmu_config.items()))
    print("=====")

    updated_model = updated_model.train()
    params = get_params(updated_model, args.layer_ids, args.param_ids)
    optimizer = AdamW(params, lr=args.lr)
    frozen_module = eval(
        args.module_str.format(model_name="frozen_model", layer_id=args.layer_id)
    )
    updated_module = eval(
        args.module_str.format(model_name="updated_model", layer_id=args.layer_id)
    )

    control_vectors_list = []
    for i in range(len(forget_data_list)):
        random_vector = torch.rand(1,1, updated_model.config.hidden_size, dtype=updated_model.dtype, device=updated_model.device)
        control_vec = random_vector / torch.norm(random_vector) * args.steering_coeff_list[i]
        control_vectors_list.append(control_vec)

    num_batches = min(
        args.max_num_batches,
        min([len(f) for f in forget_data_list]),
        min([len(r) for r in retain_data_list]),
    )
    
    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side="right"

    for epoch in range(1):
        print(f"======= Epoch {epoch} =======")
        with tqdm.tqdm(total=num_batches) as pbar:
            for idx in range(num_batches):
                topic_idx = idx % len(forget_data_list)
                batch_idx = idx // len(forget_data_list)
                control_vec = control_vectors_list[topic_idx]
                unlearn_batch = forget_data_list[topic_idx][batch_idx]
                retain_batch = retain_data_list[topic_idx][batch_idx]

                # Unlearning loss
                max_length = 512 if topic_idx == 0 else 768
                unlearn_inputs = tokenizer(
                    unlearn_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                )
                unlearn_inputs['input_ids'] = unlearn_inputs['input_ids'].to(updated_model.device)
                unlearn_inputs['attention_mask'] = unlearn_inputs['attention_mask'].to(updated_model.device)
                
                updated_forget_activations = forward_with_cache(
                    updated_model, unlearn_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)

                unlearn_loss = torch.nn.functional.mse_loss(
                    updated_forget_activations, control_vec
                )

                # Retain loss
                retain_inputs = tokenizer(
                    retain_batch, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(updated_model.device)
                retain_inputs['input_ids'] = retain_inputs['input_ids'].to(updated_model.device)
                retain_inputs['attention_mask'] = retain_inputs['attention_mask'].to(updated_model.device)
                
                updated_retain_activations = forward_with_cache(
                    updated_model, retain_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)
                frozen_retain_activations = forward_with_cache(
                    frozen_model, retain_inputs, module=frozen_module, no_grad=True
                ).to(updated_model.device)

                retain_loss = torch.nn.functional.mse_loss(
                    updated_retain_activations, frozen_retain_activations
                )
                retain_loss *= args.alpha[topic_idx]

                # Update model
                loss = unlearn_loss + retain_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"loss: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g} | retain_loss: {retain_loss.item():.4g} | param_change: {params[0].grad.abs().mean().item():.4g}")
                
    tokenizer.truncation_side = truncation_side
    # Save model
    if args.output_dir:
        path = args.output_dir
    else:
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = f"models/{args.model_name_or_path}_alpha-{args.alpha}_batches-{num_batches}_layer-{args.layer_id}_{date}"
    updated_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved model to {path}")


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    ### Model arguments
    parser.add_argument(
        "--model_name_or_path", type=str, default="HuggingFaceH4/zephyr-7b-beta"
    )
    parser.add_argument(
        "--module_str", type=str, default="{model_name}.model.layers[{layer_id}]"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None
    )
    ### Data arguments
    parser.add_argument(
        "--dataset_mode",
        type=int,
        help="one of 1, 2, or 3"
    )

    ### rmu hyperparameters
    parser.add_argument("--alpha", type=str, default="100,100", help="retain weight")
    parser.add_argument(
        "--steering_coeffs",
        type=str,
        default="20,20",
        help="Steer vector weight in order of topic",
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--num_examples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_num_batches", type=int, default=80)
    parser.add_argument("--layer_id", type=int, default=7, help="layer to unlearn")
    parser.add_argument("--layer_ids", type=str, default="5,6,7", help="update layers")
    parser.add_argument("--param_ids", type=str, default="6", help="update params")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--verbose", action="store_true", help="Logging the activations norms and cosine at each step")

    args = parser.parse_args()
    args.steering_coeff_list = [float(c) for c in args.steering_coeffs.split(",")]
    args.alpha = [float(c) for c in args.alpha.split(",")]
    args.layer_ids = [int(layer_id) for layer_id in args.layer_ids.split(",")]
    args.param_ids = [int(param_id) for param_id in args.param_ids.split(",")]
    return args 


if __name__ == "__main__":
    args = get_args()

    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    frozen_model, tokenizer = load_model(args.model_name_or_path)
    updated_model, tokenizer = load_model(args.model_name_or_path)
    forget_data_list, retain_data_list = get_data(
        args.dataset_mode,
        args.model_name_or_path,
        args.num_examples,
        args.batch_size,
    )

    run_rmu(
        updated_model,
        frozen_model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        args,
    )
