# NeuRIC


NeuRIC is built on top of **LLaMA-Factory** (for SFT / LoRA / QLoRA training) and **Easy R1** (for reasoning- and R1-style training workflows). By primarily modifying **dataset definition files**, NeuRIC allows medical data to be adapted to existing training infrastructures in a clean, reproducible, and maintainable manner.

The repository provides:
- Minimal and reproducible **patch files** for upstream frameworks  
- **configuration files** for training  

---

## Repository Structure (Suggested)

```
NeuRIC/
├── patches/
│   ├── llamafactory_load_data.patch
│   └── easy_r1_load_data.patch
├── configs/
│   ├── stage1.yaml
│   ├── stage2.yaml
│   └── stage3.yaml
└── README.md
```

---

## Requirements

- Python >= 3.11  
- PyTorch (CUDA version matched to your environment)  

---

## Quick Start

### 1. Prepare Upstream Frameworks

```bash
git clone <LLaMA-Factory-repo> third_party/LLaMA-Factory
git clone <Easy-R1-repo> third_party/easy-r1
```

### 2. Apply Patches

```bash
cd third_party/LLaMA-Factory
git apply ../../patches/llamafactory.patch

cd ../easy-r1
git apply ../../patches/easy-r1.patch
```

---

## Training

### Supervised Fine-Tuning

```bash
bash scripts/train_sft.sh configs/sft_med.yaml
```

### Reasoning / R1-style Training

```bash
bash scripts/train_r1.sh configs/r1_med.yaml
```
