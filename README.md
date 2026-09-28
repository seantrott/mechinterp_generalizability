# Mechanistic Convergence and Divergence

Code and data for the paper:

> Trott, S. (To Appear). Mechanistic Convergence and Divergence Across LLM Instances. *Computational Linguistics*.

Includes the necessary files to reproduce the primary analyses of 1-back attention, first-token attention, and induction heads.

## Data files

The `data/processed` directory contains either the raw attention outputs or summaries (i.e., aggregated across individual sentences). 

The `data/raw` directory contains an Excel file for the Natural Stories Corpus (used to elicit attention head behaviors).

## Analysis files

The `src/analysis` directory contains files for key analyses, including the `.Rmd` file and the knit `.html` file: 

- 

## Modeling files


Code and data to reproduce analysis of 1-back attention across random seeds of Pythia. 

- `src/models/run_seeds_attn.py` collects 1-back attention for each head in each layer for each sentence in the Natural Stories Corpus. 
- `process_attns.py` summarizes these scores to produce an average 1-back attention score for each head/layer across sentences. 
- The output of `process_attns.py` is included in `data/processed/attention_summaries`.
- The full analysis can be run in `src/analysis/seed_variability_attention_anon.Rmd`.