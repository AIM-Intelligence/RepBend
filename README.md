# Representation Bending

## Overview
Representation Bending is a method for modifying internal representations of LLMs to improve safety while maintaining general capabilities.

This repo is the code and model for the paper "Representation Bending for Large Language Model Safety"

<img src="./github_assets/repbend.png" width="70%"/>

## Paper
### Representation Bending for Large Language Model Safety

Authors: Ashkan Yousefpour, Taeheon Kim, Ryan S Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

Arxiv: [Link](https://arxiv.org/abs/2504.01550)

Idea:  Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses.a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering – simple vector arithmetic for steering model’s behavior during inference – to loss-based fine-tuning. RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks.

## Interesting Results

<img src="./github_assets/bar.png" width="50%"/>

RepBend has the lowest Average Attack Success Rate (ASR) across five black-box and three white-box access attacks on Mistral 7B and Llama3 8B models.

<img src="./github_assets/logitlens.png" width="100%"/>

Heatmaps cells show next token prediction and colors show entropy (blue: high confidence, red: low confidence) across layers (Y-axis) for tokens (X-axis). (a) Original instruction-tuned model LLama 3 8B complies with the request. (b) RepBend refuses the request with high certainty (blue heatmaps at the top). (c) Even when a complying sequence is forced, RepBend's representation diverges to generate random tokens.

**Pretrained Models:**  
- [Mistral 7B](https://huggingface.co/AIM-Intelligence/RepBend_Mistral_7B)  
- [Llama3 8B](https://huggingface.co/AIM-Intelligence/RepBend_Llama3_8B)

## Setup
### Prerequisites
Ensure your NVIDIA driver supports CUDA 11.6 or later by running:

### Installation
Set up the environment with the following commands:
```bash
conda create -n safety python==3.10.14
conda activate safety
pip install transformers==4.40.0
pip install torch==2.3.1 torchvision xformers==0.0.27 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.5.9.post1 --no-build-isolation
pip install peft==0.11.1 bitsandbytes pandas opencv-python timm torch_optimizer easydict pycocoevalcap sentencepiece protobuf trl==0.8.6 deepspeed==0.14.0 numpy==1.26.4 accelerate==0.29.3 jsonlines
```

If `flash-attn` installation fails due to missing CUDA toolkit, install it using:
```bash
conda install -c conda-forge cudatoolkit-dev -y
```

---

## Training Representation Bending
### Dataset
Download the dataset used for training:
[wildjailbreak.jsonl](https://drive.google.com/file/d/1Ht_fifZbw1UoUJtwQY6tEO7lyHo5rHt0/view?usp=drive_link)

### Running the Training Script
Run the training script using:
```bash
sbatch train.sh  # For Slurm
```
OR
```bash
bash train.sh  # For local execution
```

### Training Arguments
| Argument | Description |
|----------|-------------|
| `alpha`, `beta`, `gamma`, `epsilon` | Coefficients for each loss term |
| `target_layer_start_idx` | Start index of target layers for representation modification |
| `layers_window_size` | Number of layers to modify (`target_layer_start_idx` to `target_layer_start_idx + layers_window_size`) |
| `transform_layers` | Layers where LoRA modules are attached (-1 for all layers) |
| `max_step` | Number of training steps |
| `alpha_mode` | `"all"`: Computes safe loss using all layers, `"target"`: Computes safe loss using target layers only |
| `loss_mode` | Determines which token representations contribute to loss calculation:<br> • `prompt_last`: Last token of input prompts<br> • `prompt_all`: All tokens of input prompts<br> • `prompt_response`: All tokens of prompts & responses<br> • `response_all`: All tokens of responses |

### Evaluation
Once training is complete, evaluate the model using the [AI2 Safety Tool](https://github.com/ta3h30nk1m/safety-eval).

---

## Citation
If you use Representation Bending in your work, please cite:
```
@article{repbend,
  title={Representation Bending for Large Language Model Safety},
  author={Yousefpour, Ashkan and Kim, Taeheon and Kwon, Ryan S and Lee, Seungbeen and Jeung, Wonje and Han, Seungju and Wan, Alvin and Ngan, Harrison and Yu, Youngjae and Choi, Jonghyun},
  journal={arXiv preprint arXiv:2504.01550},
  year={2025}
}
```

For any questions or issues, feel free to open an issue on this repository!

