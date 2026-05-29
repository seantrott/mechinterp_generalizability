"""Run main Pythia suite (no seeds) on Natural Stories for 1-back and first-token attention."""

import pandas as pd
import torch
import os
import gc
import numpy as np

from tqdm import tqdm
from transformers import GPTNeoXForCausalLM, AutoTokenizer


MODELS = [
    'EleutherAI/pythia-14m',
    'EleutherAI/pythia-70m',
    'EleutherAI/pythia-160m',
    'EleutherAI/pythia-410m',
    'EleutherAI/pythia-1b',
    'EleutherAI/pythia-1.4b',
    # 'EleutherAI/pythia-2.8b',
    'EleutherAI/pythia-6.9b',
    'EleutherAI/pythia-12b',
]


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


def generate_revisions():
    """All checkpoints up to 143000."""
    revisions = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
    revisions.extend(range(2000, 144000, 1000))
    return [f"step{step}" for step in revisions]


def generate_revisions_limited():
    """Manually generate the list of checkpoints to analyze."""
    revisions = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                 1000, 10000, # 50000, 
                 100000, 143000]
    return [f"step{step}" for step in revisions]


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main(dfs, revisions, mpath):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("number of checkpoints:", len(revisions))

    for checkpoint in tqdm(revisions):

        model_name = mpath  # main suite — no seed suffix
        print(model_name)

        ### Set up savepaths
        savepath = "data/processed/attentions_main"
        summary_path = "data/processed/attention_main_summaries"
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        filename = ("natural_stories-rs_model-" + mpath.split("/")[1]
                    + "-" + checkpoint + ".csv")
        summary_filename = filename.replace(".csv", "_summary.csv")
        print(filename)

        print("Checking if we've already run this analysis...")
        if os.path.exists(os.path.join(savepath, filename)):
            print("Already run this model for this checkpoint.")
            continue
        if os.path.exists(os.path.join(summary_path, summary_filename)):
            print("Summary already exists — raw was deleted to save space. Skipping.")
            continue

        ### Load model
        try:
            model = GPTNeoXForCausalLM.from_pretrained(
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

        tokenizer = AutoTokenizer.from_pretrained(model_name, revision="step143000")

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
        df_results['revision'] = checkpoint
        df_results['step'] = int(checkpoint.replace("step", ""))

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

    ### Get revisions
    revisions = generate_revisions_limited()
    len(revisions)

    ### Main
    for mpath in MODELS:
        main(dfs, revisions, mpath)