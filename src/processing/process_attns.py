"""Create by-step and final-step summaries of attentions."""


import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm import tqdm

# Directory containing your attention CSVs
input_dir = "data/processed/attentions"
output_dir = "data/processed/attention_summaries"
os.makedirs(output_dir, exist_ok=True)

# Get all CSV file paths
csv_files = glob(os.path.join(input_dir, "*.csv"))

# Columns we need to load (saves memory)
usecols = [
    "mpath", "step", "revision", "Layer", "Head", "seed", "seed_name",
    "prev_to_self_ratio", "prev_to_all_ratio", "prev_token_fraction", 
    "n_params", "n_layers", "1-back attention",
]

# Loop through each file and summarize
for filepath in tqdm(csv_files):
    try:
        df = pd.read_csv(filepath, usecols=usecols)

        # Add step_modded
        df["step_modded"] = df["step"] + 1

        # Group and aggregate
        summary = (
            df.groupby(["mpath", "step_modded", "revision", "Layer", "Head", "seed", "seed_name", "n_params", "n_layers"])
              .agg(
                  mean_prev_self_ratio=("prev_to_self_ratio", "mean"),
                  mean_prev_all_ratio=("prev_to_all_ratio", "mean"),
                  prev_fraction=("prev_token_fraction", "mean"),
                  mean_1back=("1-back attention", "mean")
              )
              .reset_index()
        )

        # Save the summary to disk
        base = os.path.basename(filepath).replace(".csv", "_summary.csv")
        outpath = os.path.join(output_dir, base)
        summary.to_csv(outpath, index=False)

    except Exception as e:
        print(f"Failed on {filepath}: {e}")