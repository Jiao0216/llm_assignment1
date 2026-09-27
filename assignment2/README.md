# Assignment 2: MiniGPT from Scratch

This project implements a small decoder-only Transformer in PyTorch for next-token prediction. It includes causal self-attention, positional embeddings, layer normalization, cross-entropy training, validation loss/perplexity, checkpoint saving, and plots.

## Setup

Use Python 3.10 or newer, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Put the cleaned Assignment 1 corpus in this directory, or pass its path to `--data`.

## Train on text

```bash
python train.py --data data.txt --format text --epochs 10
```

Text mode uses a character vocabulary built from the training portion only. This keeps the project self-contained; when Assignment 1 provides token IDs, use ID mode instead.

## Train on token IDs

For a JSON array such as `[12, 4, 7, 8, 12]`, or a whitespace-separated text file such as `12 4 7 8 12`:

```bash
python train.py --data token_ids.json --format ids --epochs 10
```

The dataset is split into contiguous train and validation sections before windows are sampled. For ID mode, IDs must be non-negative integers; the vocabulary size is inferred as `max(token_id) + 1`.

For the Assignment 1 PyTorch dataset saved as a list of dictionaries containing `input_ids` and optionally `attention_mask`:

```bash
python train.py --data "/Users/zhangjiao/Desktop/Assignment1_Submission/sample_dataset.pt" --format pt --block-size 64
```

PT mode treats each input row as an independent sequence, splits rows into train and validation sets, and creates non-overlapping next-token windows without joining sequence boundaries. Padded positions from `attention_mask` are removed, and a final incomplete target window is ignored in the loss.

## Hyperparameter experiments

Run separate experiments and output directories so artifacts are preserved:

```bash
python train.py --data data.txt --format text --learning-rate 0.001 --batch-size 32 --layers 2 --embedding-dim 128 --output-dir outputs/lr_1e-3
python train.py --data data.txt --format text --learning-rate 0.0005 --batch-size 32 --layers 2 --embedding-dim 128 --output-dir outputs/lr_5e-4
python train.py --data data.txt --format text --learning-rate 0.001 --batch-size 16 --layers 1 --embedding-dim 64 --output-dir outputs/small_model
```

Default settings are sequence length 64, batch size 32, learning rate 0.001, 2 layers, embedding dimension 128, and 4 attention heads. The device is selected automatically (Apple MPS, CUDA, or CPU).

## Outputs

Each output directory receives `mini_gpt_checkpoint.pt`, `training_history.json`, `loss_curve.png`, and `perplexity_curve.png` (plots require matplotlib). Checkpoints include model and optimizer state, model configuration, vocabulary when applicable, training history, and random seed.

To load a checkpoint in Python:

```python
import torch
from model import GPTConfig, MiniGPT

checkpoint = torch.load("outputs/mini_gpt_checkpoint.pt", map_location="cpu", weights_only=False)
model = MiniGPT(GPTConfig(**checkpoint["config"]))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

## Report

See `Assignment2_Report.md` for the report draft with the measured dataset details, experiment results, and plot links. Verify the referenced artifacts are included when packaging the submission.