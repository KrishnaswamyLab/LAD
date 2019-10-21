import click
import numpy as np
import atongtf.util
import atongtf.dataset
import models

@click.command()
@click.argument('prefix', type=click.Path())
@click.argument('dataset_name', type=str)
@click.argument('model', type=str)
@click.argument('cls', type=int)
@click.argument('seed', type=int)
@click.argument('frac_corrupt', type=float)
def score_all_black(prefix, dataset_name, model, cls, seed, frac_corrupt):
    atongtf.util.set_config(limit_gpu_fraction=0.0, seed=seed)
    path = '%s/%s/%s/%d/%d/%0.2f/' % (prefix, dataset_name, model, cls, seed, frac_corrupt)
    if dataset_name.startswith('mnist'):
        d_test = atongtf.dataset.Mnist_Plus_Black_Dataset()
    #elif dataset_name.startswith('cifar'):
        #d_test = atongtf.dataset.Cifar_Anomaly_Dataset(cls, frac_corrupt)
    m = models.load_model(path + 'model')
    predictions = m.predict(d_test.get_test())
    is_autoencoder = (not model.startswith('lipschitz') and not model.startswith('ALOCC'))
    if is_autoencoder:
        if model.startswith('dsvdd'):
            # DeepSVDD returns an array, we would like to test the MSE between
            # Test points and the center 
            center = np.ones_like(predictions) * 0.1
            scores = np.mean((predictions - center) ** 2, axis=1)
        else:
            # Autoencoder Reconstruction error
            scores = np.mean((predictions - d_test.get_test()) ** 2, axis=(1,2,3))
    else:
        scores = predictions
    scores = scores.squeeze()
    if len(scores.shape) != 1:
        raise ValueError('Scores should be a 1 dimensional array is %d dimensional' % len(scores.shape))
    print(scores.shape)
    np.save('%s/black_scores.npy' % path, scores)

if __name__ == '__main__':
    score_all_black()
