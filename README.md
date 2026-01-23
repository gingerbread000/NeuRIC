# NeuRIC


NeuRIC is built on top of **LlamaFactory** (for SFT / LoRA / QLoRA training) and **Easy R1** (for reasoning- and R1-style training workflows). By primarily modifying **dataset definition files**, NeuRIC allows medical data to be adapted to existing training infrastructures in a clean, reproducible, and maintainable manner.

In parallel, we are actively developing a web-based platform that enables users to directly upload medical data and experiment with model inference and reasoning workflows through an interactive interface (early access: [https://www.neuric.cn](https://www.neuric.cn)).

The repository provides:
- Minimal and reproducible patch files for upstream frameworks  
- Configuration files for training  
- Environment setup and installation guides
- Quick start instructions

---

## Repository Structure (Suggested)

```
NeuRIC/
├── sources/
│   ├── LlamaFactory/
│   └── EASYR1/
├── patches/
│   ├── stage1_2_datasets_patch.py
│   └── stage3_datasets_patch.py
├── configs/
│   ├── stage1.yaml
│   ├── stage2.yaml
│   ├── stage2_merge.yaml
│   └── stage3.yaml
├── requirements/
│   ├── llamafactory_requirements.txt
│   └── easyr1_requirements.txt
└── README.md
```

---

## Requirements

- **Python >= 3.11**
- **PyTorch 2.6.0**
- **CUDA 12.4** (runtime & developer components)

---

## Installation

NeuRIC relies on two upstream training frameworks—**LlamaFactory** and **Easy R1**—which are used in different stages of the training and inference pipeline.
To ensure compatibility with their respective dependencies, two separate Python environments are required and should be installed independently.

| Stage     | Purpose                                  | Framework     | Environment |
| --------- | ---------------------------------------- | ------------- | ----------- |
| Stage 1–2 | Supervised fine-tuning (SFT)             | LlamaFactory | `llama_factory`        |
| Stage 3   | Reasoning optimization (GRPO / R1-style) | Easy R1       | `easyr1`    |


---
### 1. Clone Repositories and Prepare Upstream Frameworks
```bash
git clone <NeuRIC-repo>
cd NeuRIC

git clone <LlamaFactory-repo> ./LlamaFactory
git clone <Easy-R1-repo> ./EasyR1
```

### 2. Environment Setup

**Note**: For reproducing the results reported in the paper, we recommend setting up the environments using the provided requirement files, as described in the instructions below. These requirements reflect our experimental setup.


We provide two requirement files corresponding to the two environments used in our experiments:

- `requirements/llamafactory_requirements.txt`
- `requirements/easyr1_requirements.txt`

#### LlamaFactory Environment

```bash
conda create -n llama_factory python=3.11 -y
conda activate llama_factory
pip install -r requirements/llama_factory_requirements.txt
```

#### EasyR1 Environment

```bash
conda create -n easyr1 python=3.11 -y
conda activate easyr1
pip install -r requirements/easyr1_requirements.txt
```

We strongly recommend consulting the official installation instructions of the upstream frameworks when setting up the environments.
Due to system- and hardware-dependent dependencies, some packages listed in the provided requirement files may require manual installation or adjustment.
This is particularly relevant for CUDA-related dependencies such as FlashAttention or distributed training libraries.

---

### 3. Apply Patches
NeuRIC adapts dataset handling through minimal patch files applied to the upstream frameworks.
```bash
# For Stage 1-2
cp ./patches/stage1_2_datasets_patch.py ./LlamaFactory/src/llamafactory/data/mm_plugin.py

# For Stage 3
cp ./patches/stage3_datasets_patch.py ./EasyR1/verl/utils/dataset.py
```


## Quick Start
This repository provides a minimal example to run NeuRIC after completing the installation and environment setup.

### Training
NeuRIC follows a three-stage training framework:

- Stage 1: Supervised fine-tuning of the image encoder using paired medical images and reports to achieve visual–textual alignment.
- Stage 2: Supervised fine-tuning (referred to as the SFT model in the paper) using multimodal reasoning data to enhance clinical reasoning capability.
- Stage 3: Reinforcement learning for factual-oriented optimization to improve diagnosis accuracy.

---
#### Stage 1: Supervised Fine-Tuning for Visual–Textual Alignment

```bash
cp ./configs/stage1.yaml ./LlamaFactory/examples
cd LlamaFactory
llamafactory-cli train examples/stage1.yaml
```

---

#### Stage 2: Supervised Fine-Tuning for Multimodal Reasoning
Before running Stage 2, set `model_name_or_path` in `stage2.yaml` to the output checkpoint of Stage 1.

Stage 2 is conducted using LoRA-based fine-tuning to efficiently adapt the model for multimodal reasoning.

```bash
cp ./configs/stage2.yaml ./LlamaFactory/examples
cd LlamaFactory
llamafactory-cli train examples/stage2.yaml
```

After Stage 2 training, the LoRA weights need to be merged into the base model to obtain a standalone checkpoint for downstream reasoning optimization.

Required: Set the model path variable `model_name_or_path` and `adapter_name_or_path` in `stage2_merge.yaml` to point to the Stage 1 and Stage 2 checkpoints respectively
```bash
cp ./configs/stage2_merge.yaml ./LlamaFactory/examples
llamafactory-cli export examples/stage2_merge.yaml
```
The merged checkpoint will be used as the initialization model for Stage 3.

---
#### Stage 3: Reinforcement Learning for Factual-oriented Optimization
Before running Stage 3, set `MODEL_PATH` in `stage3.sh` to the merged checkpoint of Stage 2.
```bash
cp ./scripts/stage3.sh ./EasyR1/examples
cp ./configs/stage3.yaml ./EasyR1/examples
cd EasyR1
bash examples/stage3.sh
```

After Stage 3 training, Easy R1 requires manually merging the checkpoint into the base model in Hugging Face format. 

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_x/actor
```

The merged model will be saved under a `huggingface/` directory within the specified `actor` path, which serves as the final NeuRIC model checkpoint for inference and evaluation.

---

### Inference

To ensure a consistent and controlled evaluation setting, all inference is conducted within the **easyr1** environment.

Inference for Stage 2 (SFT model) and Stage 3 (the proposed NeuRIC model) is performed using the same inference script, with only the model checkpoint path changed.

```bash
cp ./scripts/evaluate.sh ./EasyR1/examples
cd EasyR1
# Modify the model path variable in `evaluate.sh` to point to the Stage 2 or Stage 3 checkpoint
bash examples/evaluate.sh
```
---

## Acknowledgements
g't
We would like to thank the authors and contributors of the following open-source projects,
which have greatly inspired and supported the development of **NeuRIC**:

- **EasyR1**  
  https://github.com/hiyouga/EasyR1  
  For its clean and extensible implementation of reinforcement learning and training pipelines.

- **LLaMAFactory**
  https://github.com/hiyouga/LLaMAFactory  
  For providing a comprehensive and practical framework for MLLM fine-tuning and alignment.
