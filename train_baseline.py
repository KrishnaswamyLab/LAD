"""
Trains baseline models:
    Isolation Forest
    Local Outlier Factor
    One-Class SVM
"""

import os
import click
import numpy as np
from atongtf import dataset
import sklearn
import sklearn.ensemble
import sklearn.svm
import sklearn.neighbors

@click.command()
@click.argument('prefix', type=click.Path())
@click.argument('dataset_name', type=str)
@click.argument('model', type=str)
@click.argument('seed', type=int)
@click.argument('num_sevens', type=int)
def train_baseline(prefix, dataset_name, model, seed, num_sevens):
    path = '%s/%s/shallow_%s/%d/%d/' % (prefix, dataset_name, model, seed, num_sevens)
    d = dataset.Mnist_Fives_Small_Sevens_Dataset(num_fives=5000-num_sevens, num_sevens=num_sevens)
    outliers_fraction = num_sevens / 5000
    eps = 1e-3
    if outliers_fraction < eps:
        outliers_fraction = eps
    if model == 'isolation_forest':
        m = sklearn.ensemble.IsolationForest(contamination=outliers_fraction, behaviour='new',
                                             random_state=seed, n_jobs=-1)
    elif model == 'ocsvm':
        # OC-SVM has no seed
        m = sklearn.svm.OneClassSVM(nu=outliers_fraction, 
                                    kernel='rbf', 
                                    gamma=0.1)
    elif model == 'lof':
        # Local Outlier Factor has no seed
        m = sklearn.neighbors.LocalOutlierFactor(n_jobs=-1, contamination=outliers_fraction, novelty=True)
    else:
        raise ValueError('Unknown Model: %s' % model)
    m.fit(d.get_train().reshape(-1, 784))

    d_test = dataset.Mnist_Fives_Small_Sevens_Dataset(num_fives=450, num_sevens=50)
    scores = m.score_samples(d_test.get_test().reshape(-1, 784))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save('%s/scores.npy' % path, scores)

if __name__ == '__main__':
    train_baseline()
