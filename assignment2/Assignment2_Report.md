# Assignment 2: Building a Small-Scale Foundation Model

## 1. Model Architecture

The model is a decoder-only Transformer language model implemented in PyTorch. Token embeddings are added to learned positional embeddings, then passed through two pre-norm Transformer blocks. Each block uses four-head causal self-attention and a GELU feed-forward network with residual connections. A final layer normalization and a linear output head produce next-token logits. The output head shares its weights with the token embedding table. A causal mask prevents each position from attending to future tokens.

| Parameter | Value |
| --- | --- |
| Vocabulary size | 50,257 |
| Transformer layers | 2 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Sequence length | 64 tokens |
| Dropout | 0.1 |

## 2. Dataset

The Assignment 1 pipeline is configured to stream three Hugging Face sources: English Wikipedia (`wikimedia/wikipedia`, configuration `20231101.en`), OpenWebText (`Skylion007/openwebtext`), and CC-News (`vblagoje/cc_news`). The configured collection budgets are approximately 450 MB, 450 MB, and 200 MB respectively; these are target budgets in the script, not measurements of the sample file used in this experiment. Cleaning removes HTML tags, Markdown links/headings, citation markers, and non-printable characters; converts text to lowercase; collapses whitespace; drops documents with fewer than 50 words; and deduplicates exact normalized text using an MD5 hash. Tokenization uses the Hugging Face `gpt2` byte-pair encoding tokenizer. Documents are concatenated with an EOS token and chunked into fixed-length 1,024-token blocks.

The input artifact `sample_dataset.pt` contains 8 batches of 8 rows, for 64 sampled blocks total. Each block contains 1,024 token IDs. The observed token ID range is 0 to 50,256, giving a model vocabulary size of 50,257. The file provides `input_ids`, `attention_mask`, and `labels`; all observed attention-mask values are 1, and the stored labels match the input IDs. The training code constructs next-token targets by shifting each block one position, so the stored labels are not used as already-shifted targets.

The 64 sampled blocks were split at the block level into 58 training blocks and 6 validation blocks before each block was windowed into non-overlapping 64-token contexts. Windows do not cross block boundaries; the final incomplete target within each block is excluded from the loss. This produced 58,368 training input tokens and 7,168 validation input tokens. The sample artifact does not preserve source document IDs or block offsets, so possible adjacency between blocks across the train/validation split cannot be ruled out. The validation estimate is also limited by having only six blocks.

## 3. Training Setup

Training used Python 3.14.7 and PyTorch 2.14.0 on Apple MPS. The optimizer was AdamW with batch size 32 and random seed 42. The baseline learning rate was 0.001; the comparison run used 0.0005. The baseline ran for 10 epochs, while the lower-learning-rate comparison also ran for 10 epochs. Both runs used two Transformer layers, embedding dimension 128, four attention heads, sequence length 64, and dropout 0.1. Cross-entropy loss was computed over next-token targets: for a sequence $x_0, x_1, \ldots$, inputs are the window's tokens and targets are the same window shifted left by one token. Perplexity was computed as $\exp(\text{mean cross-entropy loss})$.

## 4. Experiments and Results

| Run | Learning rate | Batch size | Layers | Embedding | Best validation loss | Best validation perplexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (best at epoch 2) | 0.001 | 32 | 2 | 128 | 8.4349 | 4,605.08 |
| Lower learning rate (best at epoch 4) | 0.0005 | 32 | 2 | 128 | 8.4291 | 4,578.36 |

The lower learning rate gave a slightly lower best validation loss (a difference of 0.0058) and perplexity (a difference of about 26.72), with its best epoch occurring later. The improvement is small. In both runs, validation performance improved early and then deteriorated while training loss continued to fall, which is consistent with overfitting. The validation set contains only six original sequences, so the comparison is noisy and should not be treated as strong evidence that one learning rate is generally better.

Full 10-epoch baseline curves: [loss](outputs/loss_curve.png) and [perplexity](outputs/perplexity_curve.png). Full 10-epoch lower-learning-rate curves: [loss](outputs/lr_5e-4/loss_curve.png) and [perplexity](outputs/lr_5e-4/perplexity_curve.png). The best-observed model, from the 0.0005 learning-rate run at epoch 4, is saved at [mini_gpt_checkpoint.pt](outputs/best_lr_5e-4/mini_gpt_checkpoint.pt). The baseline's best metric was observed at epoch 2.

## 5. Observations and Challenges

The main implementation challenge was adapting the Assignment 1 `.pt` file, which stores a list of batch dictionaries, rather than a plain text corpus or flat token list. The loader preserves each row as an independent sequence, respects the attention mask, and avoids creating windows across sequence boundaries. Experiments ran on Apple MPS. Validation loss and perplexity were lowest early in training, so the final-epoch checkpoint was not the best checkpoint; separate short runs were made to save checkpoints at the best observed epochs (epoch 2 for the baseline and epoch 4 for the lower learning rate).

The dataset sample is small relative to the 50,257-token vocabulary, and validation uses only six blocks. This limits generalization and makes perplexity estimates sensitive to the validation sample. The very large perplexity values should therefore be interpreted in the context of this small sample and large vocabulary, not as a robust estimate of performance on a broader corpus. Future experiments could use more tokenized blocks, preserve source/document boundaries for a leakage-resistant split, evaluate on a larger held-out set, and compare additional settings while selecting checkpoints by validation loss.