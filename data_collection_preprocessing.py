"""
Assignment 1: Data Collection and Preprocessing for Foundation Model Pre-Training
===================================================================================

Pipeline stages:
    1. Dataset collection   -> collect_datasets()
    2. Cleaning/dedup       -> clean_and_deduplicate()
    3. Tokenization         -> tokenize_documents()
    4. Custom DataLoader    -> PretrainIterableDataset + build_dataloader()

Run:
    pip install torch transformers datasets tqdm
    python data_collection_preprocessing.py

Notes:
    - Uses HuggingFace `datasets` in streaming mode so we never need to hold the
      full corpus in memory. We stop pulling from each source once we've hit its
      target byte budget.
    - Designed to run on a laptop: raw text is buffered to disk in shards, not
      kept in RAM.
"""

import os
import re
import json
import hashlib
import random
from dataclasses import dataclass, field
from typing import Iterator, List, Dict

# ----------------------------------------------------------------------------
# 0. Config
# ----------------------------------------------------------------------------

@dataclass
class Config:
    # Where to store artifacts
    raw_dir: str = "data/raw"
    clean_dir: str = "data/clean"
    tokenized_dir: str = "data/tokenized"
    sample_out: str = "sample_dataset.pt"

    # Dataset collection targets (bytes). ~1.1 GB total raw text across 3 domains.
    # Tune these down if you just want a quick smoke test.
    sources: Dict[str, Dict] = field(default_factory=lambda: {
        "wikipedia": {
            "hf_name": "wikimedia/wikipedia",
            "hf_config": "20231101.en",
            "text_field": "text",
            "target_bytes": 450 * 1024 * 1024,   # ~450MB
        },
        "openwebtext": {
            "hf_name": "Skylion007/openwebtext",
            "hf_config": None,
            "text_field": "text",
            "target_bytes": 450 * 1024 * 1024,   # ~450MB
        },
        "cc_news": {
            "hf_name": "vblagoje/cc_news",
            "hf_config": None,
            "text_field": "text",
            "target_bytes": 200 * 1024 * 1024,   # ~200MB
        },
    })

    # Cleaning
    min_words: int = 50

    # Tokenization
    tokenizer_name: str = "gpt2"          # BPE tokenizer
    block_size: int = 1024                # max sequence length per training example

    # DataLoader
    batch_size: int = 8
    shuffle_buffer_size: int = 10_000     # for streaming shuffle
    num_sample_batches: int = 8           # how many batches to dump for submission

    seed: int = 42


CFG = Config()
random.seed(CFG.seed)


# ----------------------------------------------------------------------------
# 1. Dataset collection
# ----------------------------------------------------------------------------

def collect_datasets(cfg: Config = CFG) -> Dict[str, str]:
    """
    Stream each configured HF dataset until its byte budget is hit, writing raw
    documents (one JSON object per line: {"text":..., "domain":...}) to
    cfg.raw_dir/<domain>.jsonl.

    Returns a dict mapping domain -> path of the raw jsonl shard written.
    """
    from datasets import load_dataset  # deferred import so pure-logic tests don't need it

    os.makedirs(cfg.raw_dir, exist_ok=True)
    written_paths = {}

    for domain, spec in cfg.sources.items():
        out_path = os.path.join(cfg.raw_dir, f"{domain}.jsonl")
        bytes_written = 0
        n_docs = 0

        print(f"[collect] streaming '{domain}' from {spec['hf_name']} "
              f"(target {spec['target_bytes'] / 1e6:.0f}MB)...")

        ds = load_dataset(
            spec["hf_name"],
            spec["hf_config"],
            split="train",
            streaming=True,
        )

        with open(out_path, "w", encoding="utf-8") as f:
            for example in ds:
                text = example.get(spec["text_field"], "")
                if not text:
                    continue
                line = json.dumps({"text": text, "domain": domain}, ensure_ascii=False)
                f.write(line + "\n")
                bytes_written += len(line.encode("utf-8"))
                n_docs += 1

                if bytes_written >= spec["target_bytes"]:
                    break

        print(f"[collect] '{domain}': {n_docs} docs, {bytes_written / 1e6:.1f}MB -> {out_path}")
        written_paths[domain] = out_path

    return written_paths


# ----------------------------------------------------------------------------
# 2. Cleaning + normalization + deduplication
# ----------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")          # [text](url) -> text
_MD_HEADING_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_REF_MARKER_RE = re.compile(r"\[\d+\]|\[citation needed\]", re.IGNORECASE)
_MULTI_WS_RE = re.compile(r"\s+")
_NON_PRINTABLE_RE = re.compile(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]")


def normalize_text(text: str) -> str:
    """Strip HTML/markdown artifacts, lowercase, collapse whitespace."""
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_HEADING_RE.sub("", text)
    text = _REF_MARKER_RE.sub(" ", text)
    text = _NON_PRINTABLE_RE.sub(" ", text)
    text = text.lower()
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


def doc_hash(text: str) -> str:
    """Content hash used for exact-match deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def clean_and_deduplicate(raw_paths: Dict[str, str], cfg: Config = CFG) -> str:
    """
    Reads each raw jsonl shard, normalizes text, drops short/duplicate docs,
    and writes everything to a single cfg.clean_dir/clean.jsonl file.

    Returns the path to the merged, cleaned jsonl file.
    """
    os.makedirs(cfg.clean_dir, exist_ok=True)
    out_path = os.path.join(cfg.clean_dir, "clean.jsonl")

    seen_hashes = set()
    n_in, n_out, n_dupe, n_short = 0, 0, 0, 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for domain, path in raw_paths.items():
            with open(path, "r", encoding="utf-8") as fin:
                for line in fin:
                    n_in += 1
                    obj = json.loads(line)
                    text = normalize_text(obj["text"])

                    if len(text.split()) < cfg.min_words:
                        n_short += 1
                        continue

                    h = doc_hash(text)
                    if h in seen_hashes:
                        n_dupe += 1
                        continue
                    seen_hashes.add(h)

                    fout.write(json.dumps({"text": text, "domain": domain}, ensure_ascii=False) + "\n")
                    n_out += 1

    print(f"[clean] in={n_in} out={n_out} dropped_short={n_short} dropped_dupe={n_dupe}")
    return out_path


# ----------------------------------------------------------------------------
# 3. Tokenization + chunking into fixed-size blocks
# ----------------------------------------------------------------------------

def tokenize_documents(clean_path: str, cfg: Config = CFG) -> str:
    """
    Tokenizes every cleaned document with a transformer-compatible BPE tokenizer,
    concatenates token ids (separated by EOS), and slices the stream into
    fixed-length blocks of cfg.block_size. Writes one block (list[int]) per
    line as JSON to cfg.tokenized_dir/tokenized.jsonl.

    This "concatenate-then-chunk" strategy (used by GPT-2/GPT-3 style training)
    avoids wasting compute on padding for every short document.
    """
    from transformers import AutoTokenizer

    os.makedirs(cfg.tokenized_dir, exist_ok=True)
    out_path = os.path.join(cfg.tokenized_dir, "tokenized.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})

    buffer: List[int] = []
    n_blocks = 0

    with open(clean_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            text = json.loads(line)["text"]
            ids = tokenizer.encode(text)
            buffer.extend(ids)
            buffer.append(tokenizer.eos_token_id)

            while len(buffer) >= cfg.block_size:
                block, buffer = buffer[: cfg.block_size], buffer[cfg.block_size:]
                fout.write(json.dumps(block) + "\n")
                n_blocks += 1

        # Final partial block: pad instead of discarding remaining tokens.
        if buffer:
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            block = buffer + [pad_id] * (cfg.block_size - len(buffer))
            fout.write(json.dumps(block) + "\n")
            n_blocks += 1

    print(f"[tokenize] wrote {n_blocks} blocks of size {cfg.block_size} -> {out_path}")
    return out_path


# ----------------------------------------------------------------------------
# 4. Custom PyTorch Dataset + DataLoader
# ----------------------------------------------------------------------------

class PretrainIterableDataset:
    """
    Streaming, shuffle-buffered dataset over pre-tokenized fixed-length blocks.

    Subclasses torch.utils.data.IterableDataset (imported lazily so this file
    can be unit-tested without torch installed).
    """

    def __new__(cls, *args, **kwargs):
        import torch

        class _Impl(torch.utils.data.IterableDataset):
            def __init__(self, tokenized_path: str, shuffle_buffer_size: int, seed: int):
                super().__init__()
                self.tokenized_path = tokenized_path
                self.shuffle_buffer_size = shuffle_buffer_size
                self.seed = seed

            def __iter__(self) -> Iterator["torch.Tensor"]:
                import torch as _torch
                rng = random.Random(self.seed)
                buffer: List[List[int]] = []

                def _flush_random():
                    idx = rng.randrange(len(buffer))
                    buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
                    return buffer.pop()

                with open(self.tokenized_path, "r", encoding="utf-8") as f:
                    for line in f:
                        buffer.append(json.loads(line))
                        if len(buffer) >= self.shuffle_buffer_size:
                            yield _torch.tensor(_flush_random(), dtype=_torch.long)
                    while buffer:
                        yield _torch.tensor(_flush_random(), dtype=_torch.long)

        return _Impl(*args, **kwargs)


def collate_batch(examples: List["torch.Tensor"]):
    """
    Blocks are already fixed-length (cfg.block_size), so no padding logic is
    strictly required here -- but we handle variable lengths defensively in
    case block_size changes upstream, e.g. for an eval set with shorter blocks.
    """
    import torch

    max_len = max(ex.size(0) for ex in examples)
    pad_id = 0
    padded = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)

    for i, ex in enumerate(examples):
        padded[i, : ex.size(0)] = ex
        attention_mask[i, : ex.size(0)] = 1

    # Causal LM training target: predict next token, so labels == input_ids
    return {"input_ids": padded, "attention_mask": attention_mask, "labels": padded.clone()}


def build_dataloader(tokenized_path: str, cfg: Config = CFG):
    from torch.utils.data import DataLoader

    dataset = PretrainIterableDataset(tokenized_path, cfg.shuffle_buffer_size, cfg.seed)
    return DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=collate_batch)


# ----------------------------------------------------------------------------
# 5. Orchestration
# ----------------------------------------------------------------------------

def main(cfg: Config = CFG):
    import torch

    raw_paths = collect_datasets(cfg)
    clean_path = clean_and_deduplicate(raw_paths, cfg)
    tokenized_path = tokenize_documents(clean_path, cfg)
    loader = build_dataloader(tokenized_path, cfg)

    samples = []
    for i, batch in enumerate(loader):
        samples.append(batch)
        print(f"[sample] batch {i}: input_ids {tuple(batch['input_ids'].shape)}")
        if i + 1 >= cfg.num_sample_batches:
            break

    torch.save(samples, cfg.sample_out)
    print(f"[done] saved {len(samples)} sample batches -> {cfg.sample_out}")


if __name__ == "__main__":
    main()
