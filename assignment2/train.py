"""Train MiniGPT on a text corpus or a pre-tokenized integer sequence."""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from model import GPTConfig, MiniGPT


class NextTokenDataset(Dataset):
    def __init__(self, token_ids: list[int], block_size: int) -> None:
        if len(token_ids) <= block_size:
            raise ValueError(f"Need more than {block_size} tokens; got {len(token_ids)}")
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.tokens[index : index + self.block_size + 1]
        return window[:-1], window[1:]


class SequenceWindowDataset(Dataset):
    def __init__(self, sequences: list[torch.Tensor], block_size: int) -> None:
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for sequence in sequences:
            for start in range(0, sequence.numel() - 1, block_size):
                inputs = sequence[start : start + block_size]
                targets = sequence[start + 1 : start + block_size + 1]
                if inputs.numel() < block_size:
                    inputs = torch.nn.functional.pad(
                        inputs, (0, block_size - inputs.numel()), value=0
                    )
                if targets.numel() < block_size:
                    targets = torch.nn.functional.pad(
                        targets, (0, block_size - targets.numel()), value=-100
                    )
                self.samples.append((inputs, targets))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index]


def load_pt_sequences(path: Path) -> tuple[list[torch.Tensor], int]:
    batches = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(batches, list) or not batches:
        raise ValueError("PT data must contain a non-empty list of batch dictionaries")

    sequences: list[torch.Tensor] = []
    max_token_id = -1
    for batch in batches:
        if not isinstance(batch, dict) or "input_ids" not in batch:
            raise ValueError("Each PT batch must be a dictionary containing input_ids")
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids must be a 2D integer tensor")
        if attention_mask is not None and (
            not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape
        ):
            raise ValueError("attention_mask must have the same shape as input_ids")

        for row_index, row in enumerate(input_ids):
            if attention_mask is not None:
                row = row[attention_mask[row_index].bool()]
            row = row.to(dtype=torch.long)
            if row.numel() > 1:
                sequences.append(row)
                max_token_id = max(max_token_id, int(row.max()))

    if not sequences:
        raise ValueError("PT data contains no sequences with at least two unmasked tokens")
    return sequences, max_token_id + 1


def load_corpus(
    path: Path, data_format: str, validation_fraction: float
) -> tuple[list[int], dict[str, int] | None]:
    if data_format == "text":
        text = path.read_text(encoding="utf-8")
        split_at = int(len(text) * (1.0 - validation_fraction))
        train_text = text[:split_at]
        vocabulary = {character: index for index, character in enumerate(sorted(set(train_text)))}
        if len(vocabulary) < 2:
            raise ValueError("Training text must contain at least two distinct characters")
        unknown_id = len(vocabulary)
        vocabulary["<unk>"] = unknown_id
        all_ids = [vocabulary.get(character, unknown_id) for character in text]
        return all_ids, vocabulary

    raw = json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower() == ".json" else None
    if raw is None:
        raw = [int(value) for value in path.read_text(encoding="utf-8").split()]
    if not isinstance(raw, list) or not raw or any(not isinstance(value, int) for value in raw):
        raise ValueError("Token-ID input must be a non-empty JSON list or whitespace-separated integers")
    if min(raw) < 0:
        raise ValueError("Token IDs must be non-negative integers")
    return raw, None


def split_ids(token_ids: list[int], validation_fraction: float) -> tuple[list[int], list[int]]:
    split_at = int(len(token_ids) * (1.0 - validation_fraction))
    train_ids, validation_ids = token_ids[:split_at], token_ids[split_at:]
    return train_ids, validation_ids


@torch.no_grad()
def evaluate(model: MiniGPT, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, loss = model(inputs, targets)
        count = int((targets != -100).sum())
        total_loss += float(loss) * count
        total_tokens += count
    model.train()
    if not total_tokens:
        raise ValueError("Validation split has no full sequence windows; provide more data or reduce block size")
    return total_loss / total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="A UTF-8 text corpus, token-ID file, or PT dataset")
    parser.add_argument("--format", choices=("text", "ids", "pt"), default="text")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    vocabulary = None
    if args.format == "pt":
        sequences, vocab_size = load_pt_sequences(args.data)
        random.shuffle(sequences)
        split_at = int(len(sequences) * (1.0 - args.validation_fraction))
        if split_at < 1 or split_at >= len(sequences):
            parser.error("PT data needs enough independent sequences for both train and validation splits")
        train_data = SequenceWindowDataset(sequences[:split_at], args.block_size)
        validation_data = SequenceWindowDataset(sequences[split_at:], args.block_size)
        if not len(train_data) or not len(validation_data):
            parser.error("Each PT split must contain at least one sequence window")
        train_tokens = sum(sequence.numel() for sequence in sequences[:split_at])
        validation_tokens = sum(sequence.numel() for sequence in sequences[split_at:])
    else:
        all_ids, vocabulary = load_corpus(args.data, args.format, args.validation_fraction)
        train_ids, validation_ids = split_ids(all_ids, args.validation_fraction)
        train_data = NextTokenDataset(train_ids, args.block_size)
        validation_data = NextTokenDataset(validation_ids, args.block_size)
        vocab_size = len(vocabulary) if vocabulary is not None else max(all_ids) + 1
        train_tokens, validation_tokens = len(train_ids), len(validation_ids)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_embd=args.embedding_dim,
        n_head=args.heads,
        n_layer=args.layers,
    )
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []

    print(f"device={device} vocab_size={vocab_size} train_tokens={train_tokens} val_tokens={validation_tokens}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int((targets != -100).sum())
            total_loss += float(loss.detach()) * count
            total_tokens += count

        train_loss = total_loss / total_tokens
        validation_loss = evaluate(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_perplexity": math.exp(min(train_loss, 20)),
            "validation_loss": validation_loss,
            "validation_perplexity": math.exp(min(validation_loss, 20)),
        }
        history.append(record)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_ppl={record['train_perplexity']:.2f} "
            f"val_loss={validation_loss:.4f} val_ppl={record['validation_perplexity']:.2f}"
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model.config_dict(),
        "vocabulary": vocabulary,
        "history": history,
        "data_format": args.format,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.output_dir / "mini_gpt_checkpoint.pt")
    history_path = args.output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        epochs = [item["epoch"] for item in history]
        for metric, label, filename in (
            ("loss", "Cross-entropy loss", "loss_curve.png"),
            ("perplexity", "Perplexity", "perplexity_curve.png"),
        ):
            plt.figure(figsize=(7, 4))
            plt.plot(epochs, [item[f"train_{metric}"] for item in history], label="Train")
            plt.plot(epochs, [item[f"validation_{metric}"] for item in history], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel(label)
            plt.title(f"Training and validation {label.lower()}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.output_dir / filename, dpi=160)
            plt.close()
        print(f"Saved plots and checkpoint under {args.output_dir}")
    except ImportError:
        print("matplotlib is not installed; checkpoint and training_history.json were saved without plots")


if __name__ == "__main__":
    main()