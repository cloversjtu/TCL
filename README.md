# Task-wrapped continual learning in task-oriented dialogue systems


Official implementation of the Findings of NAACL 2025 paper:


📄 Paper: [https://aclanthology.org/2025.findings-naacl.174.pdf](https://aclanthology.org/2025.findings-naacl.174.pdf)

---

## Overview

This repository contains the official implementation of our NAACL 2025 Findings paper on continual learning for NLP systems.

Our work extends the continual learning benchmark introduced in:

> Continual Learning for Task-Oriented Dialogue Systems (EMNLP 2021)

and introduces new continual learning strategies, evaluation settings, and experimental analyses.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/cloversjtu/TCL.git
cd TCL
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

The benchmark is based on the original continual learning benchmark for task-oriented dialogue systems.

To download and preprocess the datasets:

```bash
cd data
bash download.sh
```

If you are interested in preprocessing details, please refer to:

```text
utils/preprocess.py
utils/dataloader.py
```

---

## Training

The training and evaluation settings follow the original ToDCL benchmark.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --task_type NLG --CL ADAPTER --bottleneck_size 24 --lr 6.25e-3 --n_epochs 10 --train_batch_size 10
```

---

## Evaluation

```bash
python scorer.py --model_checkpoint runs_NLG/BEST/ --task_type NLG
```

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{zeng2025task,
  title={Task-wrapped continual learning in task-oriented dialogue systems},
  author={Zeng, Min and Yang, Haiqin and Chen, Xi and Guo, Yike},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2025},
  pages={3173--3183},
  year={2025}
}
```

---

## Acknowledgement

This repository is based on the benchmark and codebase from:

> Continual Learning for Task-Oriented Dialogue Systems
> Madotto et al., EMNLP 2021

We thank the original authors for releasing their benchmark and implementation.


 
