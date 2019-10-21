import os
import pandas as pd
from pprint import pprint

EMAIL = "alexanderytong@gmail.com"
FILE = 'runs/spectral_runs.csv'

def get_files(wildcards):
    runs = pd.read_csv(FILE)
    return list(runs.path)

def get_score_files(wildcards):
    runs = pd.read_csv(FILE)
    return [os.path.join(os.path.split(path)[0], 'scores.npy') for path in runs.path]

rule:
  # input: "/data/atong/anomaly/mnist/all_scores.pkl"
    #input: "/data/atong/anomaly/mnist2/all_scores.pkl"
    input: get_score_files

rule train:
    output: "{prefix}/{dataset}/{model}/{seed}/{num_sevens}/model.json"
    shell: "python train_all.py {wildcards.prefix} {wildcards.dataset} {wildcards.model} {wildcards.seed} {wildcards.num_sevens} 256 20000"

rule predict_models:
    input: rules.train.output
    output: "{prefix}/{dataset}/{model}/{seed}/{num_sevens}/scores.npy"
    shell: "CUDA_VISIBLE_DEVICES='' python score_all.py {wildcards.prefix} {wildcards.dataset} {wildcards.model} {wildcards.seed} {wildcards.num_sevens}"

rule predict_baseline:
    output: "{prefix}/{dataset}/shallow_{model}/{seed}/{num_sevens}/scores.npy"
    shell: "python train_baseline.py {wildcards.prefix} {wildcards.dataset} {wildcards.model} {wildcards.seed} {wildcards.num_sevens}"

ruleorder: predict_baseline > predict_models
    
rule accumulate_scores:
    input: 
        scores = get_score_files,
        program = 'accumulate_scores.py'
    output: "{prefix}/{dataset}/all_scores.pkl"
    shell: "python {input.program} {input.scores} {output}"

onsuccess:
    shell("mail -s 'Snakemake Completed!' %s < {log}" % EMAIL)

onerror:
    shell("mail -s 'Snakemake Error' %s < {log}" % EMAIL)

