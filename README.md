# NeuRIC


NeuRIC is built on top of **LLaMA-Factory** (for SFT / LoRA / QLoRA training) and **Easy R1** (for reasoning- and R1-style training workflows). By primarily modifying **dataset definition files**, NeuRIC allows medical data to be adapted to existing training infrastructures in a clean, reproducible, and maintainable manner.

In parallel, we are actively developing a web-based platform that enables users to directly upload medical data and experiment with model inference and reasoning workflows through an interactive interface (early access: [https://www.neuric.cn](https://www.neuric.cn:5000/)).

The repository provides:
- Minimal and reproducible **patch files** for upstream frameworks  
- **configuration files** for training  

---

## Repository Structure (Suggested)

```
NeuRIC/
├── patches/
│   ├── stage1_2_datasets_patch.py
│   └── stage3_datasets_patch.py
├── configs/
│   ├── stage1.yaml
│   ├── stage2.yaml
│   └── stage3.yaml
└── README.md
```

---

## Requirements

- **Python >= 3.11**
- **PyTorch 2.6.0**
- **CUDA 12.4** (runtime & developer components)

---

## Quick Start

### 1. Prepare Upstream Frameworks

```bash
git clone <LLaMA-Factory-repo> third_party/LLaMA-Factory
git clone <Easy-R1-repo> third_party/easy-r1
```

### 2. Apply Patches

```bash
cp ./patches/stage1_2_datasets_patch.py third_party/LLaMA-Factory/src/llamafactory/data/mm_plugin.py
cp ./patches/stage3_datasets_patch.py third_party/easy-r1/verl/utils/dataset.py
```

---

## Training

```bash
bash scripts/stage1.sh 
```

```bash
bash scripts/stage2.sh 
```

### Reasoning / R1-style Training

```bash
bash scripts/stage3.sh
```

## Acknowledgements

We would like to thank the authors and contributors of the following open-source projects,
which have greatly inspired and supported the development of **NeuRIC**:

- **EasyR1**  
  https://github.com/hiyouga/EasyR1  
  For its clean and extensible implementation of reinforcement learning and training pipelines.

- **LLaMAFactory**
  https://github.com/hiyouga/LLaMAFactory  
  For providing a comprehensive and practical framework for MLLM fine-tuning and alignment.
