"""
Trains many lipschitz anomaly detectors in parallel
from joblib import Parallel, delayed
"""

import numpy as np
import alocc
import models
from atongtf.models import *
from dsvdd import DeepSVDD

from tensorflow.keras.layers import GaussianNoise, LeakyReLU, Flatten, Dense, Conv2D
from tensorflow.keras.models import Model


class Denoising_ConvAE_Flat_Embedding(ConvAE_Flat_Embedding):
    def build_encoder(self):
        x = self.input
        x = GaussianNoise(0.1)(x)
        for filters in self.layer_filters:
            x = self.conv_wrapper(Conv2D)(filters)(x)
            x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        x = Dense(10)(x)
        self.embedding = x
        return Model(self.input, self.embedding)


def soft_threshold(thresh, mat):
    """ Computes noise matrix given x_train and x_predict"""
    noise = np.zeros_like(mat)
    k = np.where(mat > thresh)
    noise[k] = mat[k] - thresh

    k = np.where(mat < -thresh)
    noise[k] = mat[k] + thresh

    return noise

def train(path, data, model, batch_size, num_batches):
    d = data
    if model == 'lipschitz_gp':
        m = models.Gradient_Penalty_Lipschitz_Network(path, data_shape=d.get_shape(), batch_size=batch_size, conv=True)
        m.train(d.get_train(), num_batches, sample_interval=num_batches // 5)
    elif model == 'lipschitz_gp_long':
        m = models.Gradient_Penalty_Lipschitz_Network(path, data_shape=d.get_shape(), batch_size=batch_size, conv=True)
        m.train(d.get_train(), 40000, sample_interval=num_batches // 5)
    elif model == 'lipschitz_gp_high_noise':
        m = models.Gradient_Penalty_Lipschitz_Network(path, data_shape=d.get_shape(), batch_size=batch_size, conv=True, noise=1)
        m.train(d.get_train(), 40000, sample_interval=num_batches // 5)
    elif model == 'lipschitz_gp_beta_zero':
        m = models.Gradient_Penalty_Lipschitz_Network(path, data_shape=d.get_shape(), batch_size=batch_size, conv=True, noise=1, beta_1=0.)
        m.train(d.get_train(), num_batches, sample_interval=num_batches // 5)
    elif model == 'lipschitz_gp_beta_zero_long':
        m = models.Gradient_Penalty_Lipschitz_Network(path, data_shape=d.get_shape(), batch_size=batch_size, conv=True, noise=1, beta_1=0.)
        m.train(d.get_train(), 2*num_batches, sample_interval=num_batches // 5)
    elif model == 'lipschitz_gp_higher_noise':
        m = models.Gradient_Penalty_Lipschitz_Network(path, data_shape=d.get_shape(), batch_size=batch_size, conv=True, noise=2)
        m.train(d.get_train(), 40000, sample_interval=num_batches // 5)
    elif model == 'lipschitz_gp_dense':
        m = models.Gradient_Penalty_Lipschitz_Network(path, data_shape=d.get_shape(), batch_size=batch_size, conv=False, noise=0.2)
        m.train(d.get_train(), 40000, sample_interval=num_batches // 5)
    elif model == 'lipschitz_gp_big_high_noise':
        m = models.Gradient_Penalty_Lipschitz_Network(path, data_shape=d.get_shape(), batch_size=batch_size, big=True, noise=1)
        m.train(d.get_train(), 40000, sample_interval=num_batches // 5)
    elif model == 'lipschitz_spectral_dense' or model == 'lipschitz_gp_spectral':
        m = models.Spectral_Normalized_Lipschitz_Network(path, data_shape=d.get_shape(), batch_size=batch_size, conv=False, noise=1, beta_1=0.)
        m.train(d.get_train(), num_batches, sample_interval=num_batches // 5)
    elif model == 'lipschitz_spectral_conv':
        m = models.Spectral_Normalized_Lipschitz_Network(path, data_shape=d.get_shape(), batch_size=batch_size, conv=True, noise=1, beta_1=0.)
        m.train(d.get_train(), num_batches, sample_interval=num_batches // 5)
    elif model == 'lipschitz_spectral_patches':
        m = models.Spectral_Normalized_Lipschitz_Network(path, data_shape=d.get_shape(), batch_size=batch_size, conv=True, noise=0, beta_1=0., corrupt_patches=4)
        m.train(d.get_train(), num_batches, sample_interval=num_batches // 5)
    elif model == 'lipschitz_gp_patches':
        m = models.Gradient_Penalty_Lipschitz_Network(path, data_shape=d.get_shape(), batch_size=batch_size, conv=True, noise=0, beta_1=0., corrupt_patches=4)
        m.train(d.get_train(), num_batches, sample_interval=num_batches // 5)
    elif model == 'lipschitz_gp_patches_small':
        m = models.Gradient_Penalty_Lipschitz_Network(path, data_shape=d.get_shape(), batch_size=batch_size, conv=True, noise=0, beta_1=0., corrupt_patches=2)
        m.train(d.get_train(), num_batches, sample_interval=num_batches // 5)
    elif model == 'lipschitz_gp_patches_noise':
        m = models.Gradient_Penalty_Lipschitz_Network(path, data_shape=d.get_shape(), batch_size=batch_size, conv=True, noise=0.1, beta_1=0., corrupt_patches=4)
        m.train(d.get_train(), num_batches, sample_interval=num_batches // 5)
    elif model == 'ALOCC':
        m = alocc.ALOCC(path, data_shape=d.get_shape(), batch_size=batch_size, conv=True, noise=0.155, verbose=False) # parameter from paper
        m.train(d.get_train(), num_batches, sample_interval=num_batches // 5)
    elif model == 'dsvdd':
        m = DeepSVDD(path, d.get_shape(), batch_size=batch_size)
        m.train(d.get_train(), num_batches, sample_interval = num_batches // 5)
    elif model == 'conv':
        m = ConvAE_Flat_Embedding(path, d.get_shape(), nfilt=16)
        m.train(d.get_train(), num_batches, batch_size)
    elif model == 'dcae':
        m = Denoising_ConvAE_Flat_Embedding(path, d.get_shape(), nfilt=16)
        m.train(d.get_train(), num_batches, batch_size)
    elif model == 'rcae':
        # https://github.com/raghavchalapathy/oc-nn
        # The original code doesn't seem to update the noise matrix after
        # Initialization.
        # https://github.com/raghavchalapathy/rcae/blob/master/section_5.1_anomaly_detection_CIFAR_10.py
        # Line 298 as of April 3, 2019
        # We do 10 updates, as a pretty arbitrary number in lew of a better alternative
        noise_updates = 10
        noise = np.zeros_like(d.get_train())
        m = ConvAE_Flat_Embedding(path, d.get_shape(), nfilt=16)
        x_train = d.get_train()
        for i in range(noise_updates):
            x_denoised = x_train - noise
            m.train(x_denoised, num_batches // noise_updates, batch_size)
            x_output = m.model.predict(x_denoised)
            # Note that our threshold is really lambda / 2 in original paper
            # The original code seems to try lambda [0.0, 0.01, 0.1, 0.5, 1.0]
            # for every experiment.
            # We stick with 0.1 here as it seems like the default value
            noise = soft_threshold(0.1 / 2, x_train - x_output)
        #np.save(path + 'noise.npy', noise)
####
# Models for VACS
####
    elif model == 'lipschitz_gp_dense_vacs':
        m = models.Gradient_Penalty_Lipschitz_Network(path, data_shape=d.get_shape(), batch_size=batch_size, conv=False, noise=0.2)
        m.train(d.get_train(), num_batches, sample_interval=num_batches // 5)
    elif model == 'dense_alocc':
        m = alocc.ALOCC(path, data_shape=d.get_shape(), batch_size=batch_size, conv=False, noise=0.155, verbose=False) # parameter from paper
        m.train(d.get_train(), num_batches, sample_interval=num_batches // 5)
    elif model == 'dsvdd_dense':
        m = DeepSVDD(path, d.get_shape(), batch_size=batch_size, conv=False)
        m.train(d.get_train(), num_batches, sample_interval=num_batches // 5)
    elif model == 'dense_ae':
        m = DenseAE(path, d.get_shape(), layer_widths = [128,96,64], latent_dim=10)
        m.train(d.get_train(), num_batches, batch_size)
    elif model == 'dense_dae':
        m = DenseDenoisingAE(path, d.get_shape(), layer_widths = [128,96,64], latent_dim=10, noise=0.1)
        m.train(d.get_train(), num_batches, batch_size)
    elif model == 'dense_rae':
        # https://github.com/raghavchalapathy/oc-nn
        # The original code doesn't seem to update the noise matrix after
        # Initialization.
        # https://github.com/raghavchalapathy/rcae/blob/master/section_5.1_anomaly_detection_CIFAR_10.py
        # Line 298 as of April 3, 2019
        # We do 10 updates, as a pretty arbitrary number in lew of a better alternative
        noise_updates = 10
        noise = np.zeros_like(d.get_train())
        m = DenseAE(path, d.get_shape(), layer_widths = [128, 96, 64], latent_dim=10)
        x_train = d.get_train()
        for i in range(noise_updates):
            x_denoised = x_train - noise
            m.train(x_denoised, num_batches // noise_updates, batch_size)
            x_output = m.model.predict(x_denoised)
            # Note that our threshold is really lambda / 2 in original paper
            # The original code seems to try lambda [0.0, 0.01, 0.1, 0.5, 1.0]
            # for every experiment.
            # We stick with 0.1 here as it seems like the default value
            noise = soft_threshold(0.1 / 2, x_train - x_output)
        np.save(path + 'noise.npy', noise)
####
# END Models for VACS
####
    else:
        raise ValueError('Unknown Model %s' % model)
    m.save()
