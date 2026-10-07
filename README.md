# Mechanistic Convergence and Divergence

Code and data for the paper:

> Trott, S. (To Appear). Mechanistic Convergence and Divergence Across LLM Instances: Axes of Correspondence for Generalizable Interpretability Research. *Computational Linguistics*.

Includes the necessary files to reproduce the primary analyses of 1-back attention, first-token attention, and induction heads.

## Data files

The `data/processed` directory contains either the raw attention outputs or summaries (i.e., aggregated across individual sentences). 

The `data/raw` directory contains an Excel file for the Natural Stories Corpus (used to elicit attention head behaviors).

## Analysis files

The `src/analysis` directory contains files for key analyses, including the `.Rmd` file and the knit `.html` file: 

Original analyses of random seeds:

- `seed_variability_attention`: Variability in 1-back attention across random seeds.
- `seed_variability_first_token`: Variability in first-token attention across random seeds.
- `seed_variability_induction`: Variability in induction heads across random seeds.
- `induction_and_1back`: Identifies induction heads and 1-back attention heads.

Files for replication and generalization study:

- `induction_pythia_replication`: Replication of the induction head analyses in Pythia.
- `babylm_replication`: Replication using BabyLM models.
- `gemstones_replication`: Replication using the Gemstones models. Also includes positional analysis of entire model sample.

Files for appendix analysis:

- `induction_configurational`: analysis of K-composition. 


## Modeling files

The `src/models` directory contains the Python code to actually run each of the scripts to extract attention scores.