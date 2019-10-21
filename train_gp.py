"""
Trains many lipschitz anomaly detectors in parallel
from joblib import Parallel, delayed
"""

import click
import models
import common as mycommon
from common import dataset


@click.command()
@click.argument('prefix', type=click.Path())
@click.argument('dataset_name', type=str)
@click.argument('model', type=str)
@click.argument('seed', type=int)
@click.argument('num_sevens', type=int)
@click.argument('batch_size', type=int)
@click.argument('num_batches', type=int)
def train(prefix, dataset_name, model, seed, num_sevens, batch_size, num_batches):
    mycommon.util.set_config(seed=seed)
    path = '%s/%s/%s/%d/%d/' % (prefix, dataset_name, model, seed, num_sevens)
    d = dataset.Mnist_Fives_Small_Sevens_Dataset(num_fives=5000-num_sevens, num_sevens=num_sevens)
    m = models.Gradient_Penalty_Lipschitz_Network(path, data_shape=d.get_shape(), batch_size=batch_size, conv=True)
    m.train(d.get_train(), num_batches, sample_interval=num_batches // 5)
    m.save()
if __name__ == '__main__':
    train()
