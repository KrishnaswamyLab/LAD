"""
Trains many lipschitz anomaly detectors in parallel
from joblib import Parallel, delayed
"""

import click
import numpy as np

import atongtf.util
from atongtf import dataset
import train

@click.command()
@click.argument('prefix', type=click.Path())
@click.argument('dataset_name', type=str)
@click.argument('model', type=str)
@click.argument('cls', type=int)
@click.argument('seed', type=int)
@click.argument('frac_corrupt', type=float)
@click.argument('batch_size', type=int)
@click.argument('num_batches', type=int)
def train_all(prefix, dataset_name, model, cls, seed, frac_corrupt, batch_size, num_batches):
    atongtf.util.set_config(gpu_idx='auto', seed=seed)
    path = '%s/%s/%s/%d/%d/%0.3f' % (prefix, dataset_name, model, cls, seed, frac_corrupt)
    if dataset_name.startswith('mnist'):
        d = dataset.Mnist_Anomaly_Dataset(cls, frac_corrupt)
    elif dataset_name.startswith('cifar'):
        d = dataset.Cifar_Anomaly_Dataset(cls, frac_corrupt)
    elif dataset_name.startswith('vacs'):
        d = dataset.VACS_Dataset(frac_corrupt)
    train.train(path, d, model, batch_size, num_batches)

if __name__ == '__main__':
    train_all()
