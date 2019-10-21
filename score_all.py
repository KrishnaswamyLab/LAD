import click
import numpy as np
import atongtf as mycommon
import atongtf.dataset as dataset
import models

@click.command()
@click.argument('prefix', type=click.Path())
@click.argument('dataset_name', type=str)
@click.argument('model', type=str)
@click.argument('seed', type=int)
@click.argument('num_sevens', type=int)
def score_all(prefix, dataset_name, model, seed, num_sevens):
    mycommon.util.set_config(seed=seed)
    path = '%s/%s/%s/%d/%d/' % (prefix, dataset_name, model, seed, num_sevens)
    d_test = dataset.Mnist_Fives_Small_Sevens_Dataset(num_fives=450, num_sevens=50)
    m = models.load_model(path + 'model')
    predictions = m.predict(d_test.get_test())
    is_autoencoder = not model.startswith('lipschitz')
    if is_autoencoder:
        if model.startswith('dsvdd'):
            # DeepSVDD returns an array, we would like to test the MSE between
            # Test points and the center 
            center = np.ones_like(predictions) * 0.1
            scores = np.mean((predictions - center) ** 2, axis=1)
        else:
            # Autoencoder Reconstruction error
            scores = np.mean((predictions - d_test.get_test()) ** 2, axis=(1,2,3))
    elif model.startswith('ALOCC'):
        # lipschitz network predicts directly
        scores = predictions[1]
    else:
        scores = predictions
    scores = scores.squeeze()
    if len(scores.shape) != 1:
        raise ValueError('Scores should be a 1 dimensional array is %d dimensional' % len(scores.shape))
    np.save('%s/scores.npy' % path, scores)

if __name__ == '__main__':
    score_all()
