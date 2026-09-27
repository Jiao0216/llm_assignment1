# Assignment 1 — Data Collection & Preprocessing for Foundation Model Pre-Training

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python data_collection_preprocessing.py
```

This runs all four stages end-to-end and writes:

```
data/raw/<domain>.jsonl        # raw collected docs, one JSON object per line
data/clean/clean.jsonl         # cleaned + deduplicated docs
data/tokenized/tokenized.jsonl # fixed-length token-id blocks (block_size=1024)
sample_dataset.pt              # first N batches from the DataLoader (submission artifact)
```

## Tuning before you run

Open `Config` at the top of `data_collection_preprocessing.py`:

- **`target_bytes` per source** — the defaults (~450MB Wikipedia + ~450MB
  OpenWebText + ~200MB CC-News ≈ 1.1GB) satisfy the "≥1GB, multi-domain"
  requirement. Lower these first if you just want a fast smoke test end-to-end
  before committing to the full run (streaming ~1GB from OpenWebText alone can
  take a while depending on your connection).
- **`block_size`** — 1024 matches GPT-2's context window. Drop to 256/128 for
  a quicker test run.
- **`num_sample_batches`** — how many batches get dumped into `sample_dataset.pt`
  for submission (assignment asks for 5–10 blocks; default batch_size=8 ×
  8 batches = 64 blocks, comfortably covers that — adjust `batch_size`/
  `num_sample_batches` down if you want to match "5-10" more literally).

## Verifying the sample output

```python
import torch
samples = torch.load("sample_dataset.pt")
print(len(samples), "batches")
print(samples[0]["input_ids"].shape)   # (batch_size, block_size)
print(samples[0]["input_ids"][0][:20]) # first 20 token ids of first example
```

## Notes on design decisions (see report for full discussion)

- **Streaming collection**: uses `datasets.load_dataset(..., streaming=True)`
  so we never materialize the full source dataset on disk — we just pull
  documents until each source's byte budget is hit.
- **Exact-match dedup**: MD5 hash of the normalized text. This catches
  verbatim duplicates cheaply; it will *not* catch near-duplicates (e.g. two
  slightly-edited copies of the same article) — see report for the
  near-dedup extension we discuss but don't implement.
- **Concatenate-then-chunk tokenization**: documents are tokenized, joined
  with an EOS separator, and sliced into fixed-size blocks — the standard
  GPT-style strategy. This avoids wasting an entire block on padding for
  every short document.
- **IterableDataset + shuffle buffer**: because the tokenized corpus doesn't
  fit comfortably in RAM at full scale, the loader streams blocks from disk
  and uses a reservoir-style shuffle buffer instead of `shuffle=True`
  (which requires random access / an in-memory index).
