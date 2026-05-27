"""Run BabyLM baseline GPT-2 models across checkpoints on Natural Stories
for 1-back and first-token attention."""

import pandas as pd
import torch
import os
import gc
import numpy as np

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    ["main"]
)


def run_model(model, tokenizer, sentence, device):
    """Run model, return hidden states and attention"""
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model(**inputs, output_attentions=True)
        hidden_states = output.hidden_states
        attentions = output.attentions

    return {'hidden_states': hidden_states,
            'attentions': attentions,
            'tokens': inputs}


def parse_words_seen(checkpoint):
    """Convert checkpoint name to number of words seen."""
    if checkpoint == "main":
        return None
    return int(checkpoint.replace("chck_", "").replace("M", "")) * 1_000_000


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main(dfs, checkpoints, mpath):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("number of checkpoints:", len(checkpoints))

    # Total training budget — 10 epochs of unique corpus
    if "10m-gpt2" in mpath:
        total_words = 100_000_000   # 10 epochs × 10M
    elif "100m-gpt2" in mpath:
        total_words = 1_000_000_000  # 10 epochs × 100M
    else:
        total_words = None

    for checkpoint in tqdm(checkpoints):

        model_name = mpath
        print(model_name)

        ### Set up savepath
        savepath = "data/processed/attentions_babylm"
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        filename = ("natural_stories-rs_model-" + mpath.split("/")[1]
                    + "-" + checkpoint + ".csv")
        print(filename)

        print("Checking if we've already run this analysis...")
        if os.path.exists(os.path.join(savepath, filename)):
            print("Already run this model for this checkpoint.")
            continue

        ### Words seen
        words_seen = parse_words_seen(checkpoint)
        if words_seen is None:
            words_seen = total_words

        ### Load model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=checkpoint,
                output_hidden_states=True,
                attn_implementation="eager",
            )
        except Exception as e:
            print(f"Failed to load {model_name} @ {checkpoint}: {e}")
            continue

        model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=checkpoint)

        n_layers = model.config.num_hidden_layers
        print("number of layers:", n_layers)
        n_heads = model.config.num_attention_heads
        print("number of heads:", n_heads)

        n_params = count_parameters(model)
        results = []

        for story, df in dfs.items():
            for (ix, row) in tqdm(df.iterrows(), total=df.shape[0]):

                ### Get sentence
                sentence = row['Sentence']

                ### Run model for each sentence
                model_outputs = run_model(model, tokenizer, sentence, device)

                ### Now, for each layer...
                for layer in range(n_layers):

                    for head in range(n_heads):

                        ### Get attention weights for the given head
                        attn_weights = model_outputs['attentions'][layer][0, head]  # (seq_len, seq_len)

                        ### Extract avg. attentions to previous and self
                        prev_token_attention = torch.diagonal(attn_weights, offset=-1).mean().item()
                        avg_self_attn = torch.diagonal(attn_weights, offset=0).mean().item()

                        ### Avg attention overall
                        avg_attn = attn_weights.mean().item()

                        ### Attention to first token
                        first_token_attention = attn_weights[:, 0].mean().item()

                        ### Compute total attention sum
                        total_attention = attn_weights.sum().item()

                        ### Add to results dictionary
                        results.append({
                            'Sentence': row['Sentence'],
                            'Head': head + 1,
                            'Layer': layer + 1,
                            'Dataset': 'Natural Stories',
                            '1-back attention': prev_token_attention,
                            'self-attention': avg_self_attn,
                            'total_attention': total_attention,
                            'prev_to_self_ratio': prev_token_attention / avg_self_attn,
                            'avg_attention': avg_attn,
                            'prev_to_all_ratio': prev_token_attention / avg_attn,
                            'first_token': first_token_attention,
                            'first_to_self_ratio': first_token_attention / avg_self_attn,
                            'prev_token_fraction': prev_token_attention / total_attention,
                            'self_attention_fraction': avg_self_attn / total_attention,
                            'first_attention_fraction': first_token_attention / total_attention,
                            'mpath': mpath,
                            'n_heads': n_heads,
                            'n_layers': n_layers,
                            'Story': story,
                        })

        df_results = pd.DataFrame(results)
        df_results['n_params'] = n_params
        df_results['mpath'] = mpath
        df_results['checkpoint'] = checkpoint
        df_results['words_seen'] = words_seen

        df_results.to_csv(os.path.join(savepath, filename), index=False)

        ### Cleanup
        del model, df_results, results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":

    ### Load datasets
    dfs = pd.read_excel("data/raw/natstories-parsed-natural-stories.xlsx",
                        sheet_name=None, engine="openpyxl")

    ### Main
    for mpath in MODELS:
        main(dfs, CHECKPOINTS, mpath)