"""Run the Gemstones model suite (final checkpoint only) on Natural Stories
for 1-back and first-token attention.

Usage:
    python run_1back_attention_gemstones.py
"""

import pandas as pd
import torch
import os
import gc
import numpy as np

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def run_model(model, tokenizer, sentence, device):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model(**inputs, output_attentions=True)
        hidden_states = output.hidden_states
        attentions = output.attentions

    return {'hidden_states': hidden_states,
            'attentions': attentions,
            'tokens': inputs}


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main(dfs, models):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("number of models:", len(models))

    for mpath, display_name in tqdm(models.items(), desc="Gemstones"):

        model_name = mpath
        print(model_name)

        savepath = "data/processed/attentions_gemstones"
        summary_path = "data/processed/attention_gemstones_summaries"   # NEW
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        filename = ("natural_stories-rs_model-" + mpath.split("/")[1]
                    + ".csv")
        summary_filename = filename.replace(".csv", "_summary.csv")     # NEW
        print(filename)

        print("Checking if we've already run this analysis...")
        if os.path.exists(os.path.join(savepath, filename)):
            print("Already run this model.")
            continue
        if os.path.exists(os.path.join(summary_path, summary_filename)): # NEW
            print("Summary already exists — raw was deleted to save space. Skipping.")
            continue

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="eager",
            )
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        n_layers = model.config.num_hidden_layers
        print("number of layers:", n_layers)
        n_heads = model.config.num_attention_heads
        print("number of heads:", n_heads)

        n_params = count_parameters(model)
        results = []

        for story, df in dfs.items():
            for (ix, row) in tqdm(df.iterrows(), total=df.shape[0]):

                sentence = row['Sentence']

                model_outputs = run_model(model, tokenizer, sentence, device)

                for layer in range(n_layers):

                    for head in range(n_heads):

                        attn_weights = model_outputs['attentions'][layer][0, head]

                        prev_token_attention = torch.diagonal(attn_weights, offset=-1).mean().item()
                        avg_self_attn = torch.diagonal(attn_weights, offset=0).mean().item()

                        avg_attn = attn_weights.mean().item()

                        first_token_attention = attn_weights[:, 0].mean().item()

                        total_attention = attn_weights.sum().item()

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
        df_results['display_name'] = display_name

        df_results.to_csv(os.path.join(savepath, filename), index=False)

        del model, df_results, results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":

    dfs = pd.read_excel("data/raw/natstories-parsed-natural-stories.xlsx",
                        sheet_name=None, engine="openpyxl")

    main(dfs, MODELS)