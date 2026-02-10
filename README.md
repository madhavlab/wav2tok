## About

This repository contains the updated implementation of wav2tok, reformulated using the BEST-STD backbone with the proposed CTC-based pairwise alignment-based objective.



> Paper: [wav2tok: Deep Sequence Tokenizer for Audio Retrieval](https://openreview.net/forum?id=v8Mi8KU6056)


## Setup

#### Environment Setup (using uv)

This project uses **uv** for fast and reproducible Python environment management.

1. Install uv

**macOS/Linux**
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

**Using pip**

```bash
pip install uv
```


#### Clone the Repository
```sh
git clone https://github.com/madhavlab/wav2tok.git
cd wav2tok
```

### Create Environment


```sh
uv sync
```


## Usage


### Stage-wise Training

Training is performed in **two stages**.

#### Stage 1 — Contrastive-only pretraining

Train the model using only the contrastive objective.

1. Open `configs/main.yaml`.
2. Set:

```yaml
use_ctcloss: false
```

Run training:

```bash
uv run main.py --ckpt_dir /path/to/savedir
```

---

#### Stage 2 — CTC + Contrastive training

Enable the CTC-based pairwise alignment loss and train with both objectives.

1. In `configs/main.yaml`, set:

```yaml
use_ctcloss: true
```

2. Run training (optionally resuming from the Stage 1 checkpoint):

```bash
uv run main.py --ckpt_dir /path/to/savedir
```

In this stage, the model is optimized jointly with the **contrastive loss** and the **CTC-based alignment loss**.

### Search and Demo 

To create the database, build the index, and perform retrieval:
```sh
uv run std_demo.py
```

For a demonstration of word tokenization, check the following Jupyter Notebook:

```sh
demo/word_tokenization.ipynb
```


## Datasets 

- **Dataset**: [LibriSpeech Word Alignments](https://github.com/CorentinJ/librispeech-alignments)

## Citation

If you find our work useful, please cite:
```sh
@inproceedings{banerjee2023wav2tok,
 title={wav2tok: Deep Sequence Tokenizer for Audio Retrieval},
 author={Banerjee, Adhiraj and Arora, Vipul},
 booktitle={The Eleventh International Conference on Learning Representations},
 year={2023}
 }

```


## Acknowledgments
The code in this project is adapted or modifed from the following projects:
- [BEST-STD](https://github.com/anupsingh15/BEST-STD) [MIT License]
- [CTC python](https://github.com/vadimkantorov/ctc) 

