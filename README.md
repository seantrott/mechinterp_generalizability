# Generalizability for Mechanistic Interpretability

Code and data to reproduce analysis of 1-back attention across random seeds of Pythia. 

- `src/models/run_seeds_attn.py` collects 1-back attention for each head in each layer for each sentence in the Natural Stories Corpus. 
- `process_attns.py` summarizes these scores to produce an average 1-back attention score for each head/layer across sentences. 
- The output of `process_attns.py` is included in `data/processed/attention_summaries`.
- The full analysis can be run in `src/analysis/seed_variability_attention_anon.Rmd`.