"""
Induction head detection across the Gemstones model suite (final checkpoint only).
Computes prefix-matching score (PS) per head and associative recall (AR) per model.
Saves trial-level data: one row per (head, random_sequence).

Usage:
    python run_induction_scores_gemstones.py
"""

import os
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# ── Configuration ───────────────────────────────────────────────────────

MODELS = {
    "Gemstone-Models/Gemstone-256x23": "Gemstone 256x23 (48M)",
    "Gemstone-Models/Gemstone-256x27": "Gemstone 256x27 (52M)",
    "Gemstone-Models/Gemstone-256x71": "Gemstone 256x71 (96M)",
    "Gemstone-Models/Gemstone-256x80": "Gemstone 256x80 (107M)",
    "Gemstone-Models/Gemstone-384x13": "Gemstone 384x13 (67M)",
    "Gemstone-Models/Gemstone-384x36": "Gemstone 384x36 (117M)",
    "Gemstone-Models/Gemstone-512x11": "Gemstone 512x11 (95M)",
    "Gemstone-Models/Gemstone-512x12": "Gemstone 512x12 (99M)",
    "Gemstone-Models/Gemstone-512x13": "Gemstone 512x13 (103M)",
    "Gemstone-Models/Gemstone-512x14": "Gemstone 512x14 (107M)",
    "Gemstone-Models/Gemstone-512x16": "Gemstone 512x16 (115M)",
    "Gemstone-Models/Gemstone-768x3": "Gemstone 768x3 (100M)",
    "Gemstone-Models/Gemstone-768x45": "Gemstone 768x45 (500M)",
    "Gemstone-Models/Gemstone-1024x28": "Gemstone 1024x28 (500M)",
    "Gemstone-Models/Gemstone-1280x15": "Gemstone 1280x15 (500M)",
    "Gemstone-Models/Gemstone-1280x36": "Gemstone 1280x36 (1B)",
    "Gemstone-Models/Gemstone-1536x50": "Gemstone 1536x50 (2B)",
    "Gemstone-Models/Gemstone-1792x7": "Gemstone 1792x7 (500M)",
    "Gemstone-Models/Gemstone-1792x18": "Gemstone 1792x18 (1B)",
    "Gemstone-Models/Gemstone-2048x27": "Gemstone 2048x27 (2B)",
    "Gemstone-Models/Gemstone-2560x8": "Gemstone 2560x8 (1B)",
    "Gemstone-Models/Gemstone-3072x12": "Gemstone 3072x12 (2B)",
}

SEQ_LEN = 50
NUM_SAMPLES = 100
BATCH_SIZE = 5

SAVEPATH = "data/processed/induction_results_gemstones"


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
        relevant_logits = logits[:, seq_len:seq_len * 2, :]

        target_logits = torch.gather(
            relevant_logits, dim=-1, index=target_ids.unsqueeze(-1)
        )  # (bs, seq_len, 1)

        ranks = (relevant_logits > target_logits).sum(dim=-1)  # (bs, seq_len)
        ar_acc_all.append((ranks == 0).float().mean(dim=1).cpu())
        ar_rank_all.append(ranks.float().mean(dim=1).cpu())

    ps_all = torch.cat(ps_all, dim=0).numpy()
    ar_acc_all = torch.cat(ar_acc_all, dim=0).numpy()
    ar_rank_all = torch.cat(ar_rank_all, dim=0).numpy()

    return ps_all, ar_acc_all, ar_rank_all


def scores_to_dataframe(ps_all, ar_acc_all, ar_rank_all, config,
                         mpath, display_name, n_params):
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
                    'display_name': display_name,
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
    print(f"Number of models: {len(MODELS)}")

    for mpath, display_name in tqdm(MODELS.items(), desc="Gemstones"):

        # ── Check if already done ──
        filename = f"induction_scores-{mpath.split('/')[1]}.csv"
        filepath = os.path.join(SAVEPATH, filename)

        if os.path.exists(filepath):
            print(f"  Skipping {filename} (exists)")
            continue

        # ── Load model ──
        print(f"  Loading {mpath}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                mpath,
                attn_implementation="eager"
            )
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        model.to(device).eval()
        config = model.config
        n_params = count_parameters(model)

        print(f"  layers={config.num_hidden_layers}, heads={config.num_attention_heads}, "
              f"params={n_params:,}")

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
            config, mpath, display_name, n_params
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