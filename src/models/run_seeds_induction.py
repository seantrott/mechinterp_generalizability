"""
Induction head detection across Pythia seeds and checkpoints.
Computes prefix-matching score (PS) per head and associative recall (AR) per model.
Saves trial-level data: one row per (head, random_sequence).

Usage:
    python run_induction_scores.py
"""

import os
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoConfig


# ── Configuration ───────────────────────────────────────────────────────

MODELS = [
    # 'EleutherAI/pythia-14m',
   # 'EleutherAI/pythia-70m',
    #'EleutherAI/pythia-160m',
    'EleutherAI/pythia-410m'
]

SEQ_LEN = 50
NUM_SAMPLES = 100
BATCH_SIZE = 50  # adjust down if OOM on larger models

SAVEPATH = "data/processed/induction_results"


def generate_revisions_limited():
    """Checkpoints matching your original script."""
    revisions = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                 1000, 10000, 50000, 100000, 143000]
    return [f"step{step}" for step in revisions]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        ps_all: np.ndarray (num_samples, n_layers, n_heads) — mean PS per sample
        ar_acc_all: np.ndarray (num_samples,) — per-sample accuracy
        ar_rank_all: np.ndarray (num_samples,) — per-sample mean rank
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
            # mean PS across positions within each sample
            batch_ps[:, layer, :] = attn[:, :, src, tgt].mean(dim=-1).cpu()

        ps_all.append(batch_ps)

        # ── AR: does the model predict B given [A, B, ..., A]? ──
        logits = out.logits  # (bs, seq_len*2, vocab)
        target_ids = tokens[:, 1:seq_len + 1]  # the B tokens from first half
        relevant_logits = logits[:, seq_len:seq_len * 2, :]  # logits at repeated-half positions

        target_logits = torch.gather(
            relevant_logits, dim=-1, index=target_ids.unsqueeze(-1)
        )  # (bs, seq_len, 1)

        ranks = (relevant_logits > target_logits).sum(dim=-1)  # (bs, seq_len)
        ar_acc_all.append((ranks == 0).float().mean(dim=1).cpu())  # per-sample accuracy
        ar_rank_all.append(ranks.float().mean(dim=1).cpu())  # per-sample mean rank

    ps_all = torch.cat(ps_all, dim=0).numpy()        # (num_samples, n_layers, n_heads)
    ar_acc_all = torch.cat(ar_acc_all, dim=0).numpy()  # (num_samples,)
    ar_rank_all = torch.cat(ar_rank_all, dim=0).numpy()  # (num_samples,)

    return ps_all, ar_acc_all, ar_rank_all


def scores_to_dataframe(ps_all, ar_acc_all, ar_rank_all, config,
                         mpath, seed, checkpoint, n_params):
    """
    Convert arrays to a trial-level DataFrame.
    One row per (sample, layer, head).
    """
    n_samples, n_layers, n_heads = ps_all.shape
    step = int(checkpoint.replace("step", ""))

    rows = []
    for sample_idx in range(n_samples):
        for layer in range(n_layers):
            for head in range(n_heads):
                rows.append({
                    'mpath': mpath,
                    'seed': seed,
                    'step': step,
                    'revision': checkpoint,
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
    revisions = generate_revisions_limited()
    print(f"Checkpoints: {len(revisions)}")

    for mpath in MODELS:
        for checkpoint in tqdm(revisions, desc=mpath):
            for seed in range(1, 10):
                seed_name = f"seed{seed}"
                model_name = f"{mpath}-{seed_name}"

                # ── Check if already done ──
                filename = (f"induction_scores-{mpath.split('/')[1]}"
                           f"-{checkpoint}-{seed_name}.csv")
                filepath = os.path.join(SAVEPATH, filename)

                if os.path.exists(filepath):
                    print(f"  Skipping {filename} (exists)")
                    continue

                # ── Load model ──
                print(f"  {model_name} @ {checkpoint}")
                try:
                    model = GPTNeoXForCausalLM.from_pretrained(
                        model_name, revision=checkpoint,
                        attn_implementation="eager"
                    )
                except Exception as e:
                    print(f"  Failed to load: {e}")
                    continue

                model.to(device).eval()
                config = model.config
                n_params = count_parameters(model)

                # ── Compute scores ──
                ps_all, ar_acc_all, ar_rank_all = compute_induction_scores(
                    model, config,
                    seq_len=SEQ_LEN,
                    num_samples=NUM_SAMPLES,
                    batch_size=BATCH_SIZE,
                    device=device,
                )

                # ── Save ──
                df = scores_to_dataframe(
                    ps_all, ar_acc_all, ar_rank_all,
                    config, mpath, seed, checkpoint, n_params
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