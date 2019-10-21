"""
Trains baseline models:
    Isolation Forest
    Local Outlier Factor
    One-Class SVM
"""

import functools
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
@click.argument('cls', type=int)
@click.argument('seed', type=int)
@click.argument('frac_corrupt', type=float)
def train_baseline(prefix, dataset_name, model, cls, seed, frac_corrupt):
    path = '%s/%s/shallow_%s/%d/%d/%0.3f' % (prefix, dataset_name, model, cls, seed, frac_corrupt)
    if dataset_name.startswith('mnist'):
        d = dataset.Mnist_Anomaly_Dataset(cls, frac_corrupt)
    elif dataset_name.startswith('cifar'):
        d = dataset.Cifar_Anomaly_Dataset(cls, frac_corrupt)
    elif dataset_name.startswith('vacs'):
        d = dataset.VACS_Dataset(frac_corrupt)
    eps = 1e-3
    if frac_corrupt < eps:
        frac_corrupt = eps
    if model == 'isolation_forest':
        m = sklearn.ensemble.IsolationForest(contamination=frac_corrupt, behaviour='new',
                                             random_state=seed, n_jobs=1)
    elif model == 'ocsvm':
        # OC-SVM has no seed
        m = sklearn.svm.OneClassSVM(nu=frac_corrupt, 
                                    kernel='rbf', 
                                    gamma=0.1)
    elif model == 'lof':
        # Local Outlier Factor has no seed
        m = sklearn.neighbors.LocalOutlierFactor(n_jobs=1, contamination=frac_corrupt, novelty=True)
    else:
        raise ValueError('Unknown Model: %s' % model)
    shape = d.get_shape()
    flat_shape = functools.reduce(lambda x,y: x*y, shape)
    m.fit(d.get_train().reshape(-1, flat_shape))

    scores = m.score_samples(d.get_test().reshape(-1, flat_shape))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save('%s/scores.npy' % path, scores)

if __name__ == '__main__':
    train_baseline()
