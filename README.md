# LAD
Lipschitz Anomaly Discriminator

Contains all of the code needed to run comparisons of the LAD method against existing baselines present in the paper

https://arxiv.org/abs/1905.10710

uses `make_runs.py` to compile lists of models with parameters to test and uses snakemake to manage jobs.

`Snakefile2` is for running AUROC comparisons, and `Snakefile3` is for the black image robustness tests.
