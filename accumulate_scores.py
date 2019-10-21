import click
import numpy as np
from sklearn import metrics
import pandas as pd

import atongtf.dataset as md

@click.command()
@click.argument('input_files', type=click.Path(), nargs=-1)
@click.argument('output_file', type=click.Path())
def accumulate_scores(input_files, output_file):
    to_return = []
    d = md.Mnist_Fives_Small_Sevens_Dataset(num_fives=450, num_sevens=50)
    y_true = (d.get_test_labels().flatten() == 7)

    for path in input_files:
        spath = path.split('/')
        dataset, model, seed, num_sevens = spath[-5:-1]
        seed = int(seed)
        num_sevens = int(num_sevens)
        scores = np.load(path)

        if model.startswith('shallow') or model.startswith('lipschitz') or model.startswith('ALOCC'):
            # Flip score vector for higher score == more normal
            scores = -scores
        
        # Calculate Metrics
        percent_anomaly = num_sevens / (5000) * 100
        ap = metrics.average_precision_score(y_true, scores)
        auc = metrics.roc_auc_score(y_true, scores)
        # Calc P@10 metric, percent anomaly in top ten error scores
        top_anomalies = np.argsort(scores)[::-1][:10]
        p_at_10 = metrics.accuracy_score(y_true[top_anomalies], np.ones(10))
        to_return.append((dataset, model, seed, num_sevens, ap, auc, p_at_10))
    df = pd.DataFrame(to_return, columns = ['dataset', 'model', 'seed', 'num_sevens', 'average_precision', 'auc', 'p_at_10'])
    df.to_pickle(output_file)

if __name__ == '__main__':
    accumulate_scores()
