"""
Induction head detection across BabyLM baseline GPT-2 models and checkpoints.
Computes prefix-matching score (PS) per head and associative recall (AR) per model.

Checkpoint naming convention from model card:
  - Every 1M words for the first 10M words: chck_1M ... chck_10M
  - Every 10M words afterward: chck_20M, chck_30M, ..., chck_100M
  - Final: main

Usage:
    python run_induction_scores_babylm.py
"""

import os
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# ── Configuration ───────────────────────────────────────────────────────

MODELS = [
    'BabyLM-community/babylm-baseline-10m-gpt2',
    'BabyLM-community/babylm-baseline-100m-gpt2',
]

# Checkpoint schedule (in words seen)
# First 10M: every 1M
# Then: every 10M up to 100M
CHECKPOINTS = (
    [f"chck_{n}M" for n in range(1, 11)] +
    [f"chck_{n}M" for n in range(20, 101, 10)] +
    ["main"]  # final checkpoint
)

SEQ_LEN = 50
NUM_SAMPLES = 100
BATCH_SIZE = 5

SAVEPATH = "data/processed/induction_results_babylm"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_words_seen(checkpoint):
    """Convert checkpoint name to number of words seen."""
    if checkpoint == "main":
        return None  # will be set per-model based on total budget
    # chck_5M -> 5_000_000
    return int(checkpoint.replace("chck_", "").replace("M", "")) * 1_000_000


def generate_repeated_random_tokens(vocab_size, seq_len, num_samples):
    """Unique random tokens per sample, repeated twice."""
    tokens = torch.stack([
        torch.randperm(vocab_size)[:seq_len]
        for _ in range(num_samples)
    ])
    return tokens.repeat(1, 2)  # (num_samples, seq_len * 2)


def compute_induction_scores(model, config, seq_len=50, num_samples=100,
                              batch_size=50, device="cpu"):
    """
    Compute PS per head per sample and AR per sample.

    Returns:
        ps_all: np.ndarray (num_samples, n_layers, n_heads)
        ar_acc_all: np.ndarray (num_samples,)
        ar_rank_all: np.ndarray (num_samples,)
    """
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    vocab_size = config.vocab_size

    ps_all = []
    ar_acc_all = []
    ar_rank_all = []

    for start in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - start)
        tokens = generate_repeated_random_tokens(vocab_size, seq_len, bs).to(device)

        with torch.no_grad():
            out = model(tokens, output_attentions=True)

        # ── PS: attention from repeated-half positions to induction targets ──
        src = torch.arange(seq_len, seq_len * 2)
        tgt = src - (seq_len - 1)

        batch_ps = torch.zeros(bs, n_layers, n_heads)
        for layer in range(n_layers):
            attn = out.attentions[layer]  # (bs, heads, seq, seq)
            batch_ps[:, layer, :] = attn[:, :, src, tgt].mean(dim=-1).cpu()

        ps_all.append(batch_ps)

        # ── AR: does the model predict B given [A, B, ..., A]? ──
        logits = out.logits  # (bs, seq_len*2, vocab)
        target_ids = tokens[:, 1:seq_len + 1]
        relevant_logits = logits[:, seq_len:seq_len * 2, :]

        target_logits = torch.gather(
            relevant_logits, dim=-1, index=target_ids.unsqueeze(-1)
        )

        ranks = (relevant_logits > target_logits).sum(dim=-1)
        ar_acc_all.append((ranks == 0).float().mean(dim=1).cpu())
        ar_rank_all.append(ranks.float().mean(dim=1).cpu())

    ps_all = torch.cat(ps_all, dim=0).numpy()
    ar_acc_all = torch.cat(ar_acc_all, dim=0).numpy()
    ar_rank_all = torch.cat(ar_rank_all, dim=0).numpy()

    return ps_all, ar_acc_all, ar_rank_all


def scores_to_dataframe(ps_all, ar_acc_all, ar_rank_all, config,
                         mpath, checkpoint, words_seen, n_params):
    """
    Convert arrays to a trial-level DataFrame.
    One row per (sample, layer, head).
    """
    n_samples, n_layers, n_heads = ps_all.shape

    rows = []
    for sample_idx in range(n_samples):
        for layer in range(n_layers):
            for head in range(n_heads):
                rows.append({
                    'mpath': mpath,
                    'checkpoint': checkpoint,
                    'words_seen': words_seen,
                    'n_params': n_params,
                    'n_layers': n_layers,
                    'n_heads': n_heads,
                    'sample': sample_idx,
                    'layer': layer,
                    'head': head,
                    'ps': float(ps_all[sample_idx, layer, head]),
                    'ar_acc': float(ar_acc_all[sample_idx]),
                    'ar_mean_rank': float(ar_rank_all[sample_idx]),
                })

    return pd.DataFrame(rows)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(SAVEPATH, exist_ok=True)
    print(f"Models: {len(MODELS)}, Checkpoints per model: {len(CHECKPOINTS)}")

    for mpath in MODELS:
        # Total training budget for "main" — 10 epochs of either 10M or 100M
        if "10m" in mpath.lower():
            total_words = 100_000_000  # 10 epochs × 10M
        else:
            total_words = 1_000_000_000  # 10 epochs × 100M

        for checkpoint in tqdm(CHECKPOINTS, desc=mpath):
            # ── Check if already done ──
            filename = (f"induction_scores-{mpath.split('/')[1]}"
                        f"-{checkpoint}.csv")
            filepath = os.path.join(SAVEPATH, filename)

            if os.path.exists(filepath):
                print(f"  Skipping {filename} (exists)")
                continue

            # ── Words seen at this checkpoint ──
            words_seen = parse_words_seen(checkpoint)
            if words_seen is None:
                words_seen = total_words

            # ── Load model ──
            print(f"  {mpath} @ {checkpoint} ({words_seen:,} words)")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    mpath,
                    revision=checkpoint,
                    attn_implementation="eager",
                )
            except Exception as e:
                print(f"  Failed to load: {e}")
                continue

            model.to(device).eval()
            config = model.config
            n_params = count_parameters(model)

            # ── Compute scores ──
            try:
                ps_all, ar_acc_all, ar_rank_all = compute_induction_scores(
                    model, config,
                    seq_len=SEQ_LEN,
                    num_samples=NUM_SAMPLES,
                    batch_size=BATCH_SIZE,
                    device=device,
                )
            except Exception as e:
                print(f"  Failed during inference: {e}")
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            # ── Save ──
            df = scores_to_dataframe(
                ps_all, ar_acc_all, ar_rank_all,
                config, mpath, checkpoint, words_seen, n_params
            )
            df.to_csv(filepath, index=False)
            print(f"  Saved {filename} ({len(df)} rows)")

            # ── Cleanup ──
            del model, ps_all, ar_acc_all, ar_rank_all, df
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()