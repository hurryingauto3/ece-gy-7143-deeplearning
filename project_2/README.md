# Parameter‑Efficient RoBERTa Fine‑Tuning on AGNews with LoRA

## Overview  
Efficiently adapt RoBERTa‑base (125 M params) for AGNews (120 k train, 7.6 k test, 4 classes) within a 1 M‑parameter budget using Low‑Rank Adaptation (LoRA). Systematic experiments vary adapter **rank** (_r_), **scaling** (α), **modules** (attention QKV, FFN, classifier), **data strategies** (augmentation, filtering) and **regularization** (dropout, label smoothing).  

## Repository structure  
    ├─ code/
    │  ├─ dl-project-2-4-21-1130.ipynb      # end‑to‑end training & evaluation
    │  ├─ results_reproduction.ipynb        # reproduce best result
    │  ├─ requirements.txt                  # Python dependencies
    │  └─ results/                          # per‑experiment logs, checkpoints, plots
    └─ report/
        ├─ main.tex                          # AAAI paper source
        └─ images/                           # figures for report

## Usage:
### Environment setup
Clone and create environment  
```bash
git clone https://github.com/hurryingauto3/ece-gy-7123-project-2.git
conda create -n lora-agnews python=3.10
conda activate lora-agnews
pip install -r code/requirements.txt
```

### Run the code
1. Open and run dl-project-2-4-21-1130.ipynb in Jupyter Notebook or Google Colab for full experiment reproduction.
2. Open and run results_reproduction.ipynb to reproduce the best result.
