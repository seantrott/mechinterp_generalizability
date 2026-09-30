"""
Composed-QK scores across Pythia-14M seeds at step 143000.
For each would-be induction head (L1, H1), finds the layer-0 head partner
that maximizes the composed-QK eigenvalue score.

Saves trial-level data: one row per (seed, layer, head).
"""

import os
import gc
import torch
import pandas as pd
from tqdm import tqdm
from transformers import GPTNeoXForCausalLM


# ── Configuration ───────────────────────────────────────────────────────

MODELS = [
    # 'EleutherAI/pythia-14m',
    'EleutherAI/pythia-70m',
    # 'EleutherAI/pythia-160m',
    # 'EleutherAI/pythia-410m',
]


def generate_revisions():
    """Manually generate the list of checkpoints available for Pythia modeling suite"""
    
    # Fixed initial steps
    revisions = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                 1000, 10000, 50000, 100000, 143000]

    revisions = [143000]
    
    # Format each step as "stepX"
    return [f"step{step}" for step in revisions]



CHECKPOINTS = generate_revisions() # ['step143000']
SEEDS = list(range(1, 10))

SAVEPATH = "data/processed/composed_qk_frob_results"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_qkv(model, layer_idx, n_heads, d_head, d_model):
    """Return Q, K, V weight tensors, each shape (n_heads, d_head, d_model)."""
    attn = model.gpt_neox.layers[layer_idx].attention
    qkv = attn.query_key_value.weight.view(n_heads, 3, d_head, d_model)
    return qkv[:, 0], qkv[:, 1], qkv[:, 2]


def get_o(model, layer_idx, n_heads, d_head, d_model):
    """Return O weight, shape (n_heads, d_model, d_head)."""
    attn = model.gpt_neox.layers[layer_idx].attention
    o = attn.dense.weight.view(d_model, n_heads, d_head)
    return o.permute(1, 0, 2)  # (n_heads, d_model, d_head)


def eigval_score(M):
    """Fraction of positive-real eigenvalue mass."""
    eigvals = torch.linalg.eigvals(M.float())
    pos_real = eigvals.real.clamp(min=0).sum()
    total_abs = eigvals.abs().sum()
    return (pos_real / total_abs).item()


def frob_norm(X):
    return torch.linalg.matrix_norm(X.float(), ord='fro')


def frobenius_comp_score(qk, ov, composed):
    """Standard K-composition score: ||W_QK W_OV||_F / (||W_QK||_F ||W_OV||_F)."""
    return (frob_norm(composed) / (frob_norm(qk) * frob_norm(ov))).item()


def compute_composed_qk_scores(model, config):
    """
    For each L1 head (L1 >= 1, any H1), search all earlier (L0, H0) partners
    and keep the one maximizing the eigenvalue score of the composed
    W_QK @ W_OV matrix. Records that partner's eigenvalue score and its
    Frobenius K-composition score.

    Returns a list of dicts: one per (L1, H1).
    """
    d_model = config.hidden_size
    n_heads = config.num_attention_heads
    n_layers = config.num_hidden_layers
    d_head = d_model // n_heads

    results = []

    with torch.no_grad():
        for L1 in range(1, n_layers):
            Q1, K1, _ = get_qkv(model, L1, n_heads, d_head, d_model)
            for H1 in range(n_heads):
                qk_attn = Q1[H1].T @ K1[H1]  # (d_model, d_model)

                best_score = -float("inf")
                best_partner = None
                best_frob = None
                for L0 in range(L1):
                    _, _, V0 = get_qkv(model, L0, n_heads, d_head, d_model)
                    O0 = get_o(model, L0, n_heads, d_head, d_model)
                    for H0 in range(n_heads):
                        ov0 = O0[H0] @ V0[H0]            # (d_model, d_model)
                        composed = qk_attn @ ov0
                        score = eigval_score(composed)
                        if score > best_score:
                            best_score = score
                            best_partner = (L0, H0)
                            best_frob = frobenius_comp_score(qk_attn, ov0, composed)

                results.append({
                    'layer': L1,
                    'head': H1,
                    'composed_qk_score': best_score,        # eigenvalue score of best partner
                    'partner_layer': best_partner[0],
                    'partner_head': best_partner[1],
                    'composed_qk_frob': best_frob,          # Frobenius score of SAME partner
                })

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(SAVEPATH, exist_ok=True)

    for mpath in MODELS:
        for checkpoint in tqdm(CHECKPOINTS, desc=mpath):
            for seed in SEEDS:
                seed_name = f"seed{seed}"
                model_name = f"{mpath}-{seed_name}"

                filename = (f"composed_qk-{mpath.split('/')[1]}"
                           f"-{checkpoint}-{seed_name}.csv")
                filepath = os.path.join(SAVEPATH, filename)

                if os.path.exists(filepath):
                    print(f"  Skipping {filename} (exists)")
                    continue

                print(f"  {model_name} @ {checkpoint}")
                try:
                    model = GPTNeoXForCausalLM.from_pretrained(
                        model_name, revision=checkpoint,
                    )
                except Exception as e:
                    print(f"  Failed to load: {e}")
                    continue

                model.to(device).eval()
                config = model.config
                n_params = count_parameters(model)
                step = int(checkpoint.replace("step", ""))

                results = compute_composed_qk_scores(model, config)

                # Add metadata to each row
                for row in results:
                    row['mpath'] = mpath
                    row['seed'] = seed
                    row['step'] = step
                    row['revision'] = checkpoint
                    row['n_params'] = n_params
                    row['n_layers'] = config.num_hidden_layers
                    row['n_heads'] = config.num_attention_heads

                df = pd.DataFrame(results)
                df.to_csv(filepath, index=False)
                print(f"  Saved {filename} ({len(df)} rows)")

                del model, results, df
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()