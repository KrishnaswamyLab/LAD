"""
Implements version of DeepSVDD
Ruff et al. 2018

For comparison purposes, does not pretrain as an autoencoder.
Initializes the center as 0.1 ^ n
"""
from functools import partial

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Conv2D, GaussianNoise, Dense, Lambda, Reshape, Layer
from tensorflow.keras.layers import LeakyReLU, Flatten
import numpy as np

import atongtf.dataset
import atongtf.util

def debias(layer):
    return partial(layer, use_bias=False)

def conv_wrapper(layer):
    return partial(layer,
                   kernel_size=4,
                   strides=2,
                   padding='same',
                   # tf.nn.leaky_relu cannot be saved
                   # activation=tf.nn.leaky_relu,
                   kernel_initializer='he_normal',
                   use_bias=False)


class DeepSVDD():
    """ Build SVDD no bias parameters no pretraining """
    def __init__(self, model_dir, data_shape, batch_size=256, verbose=1, conv=True):
        self.filter_widths = [16, 32, 64]
        self.conv = conv
        self.optimizer = Adam()
        self.model_dir = model_dir
        self.data_shape = data_shape
        self.verbose = verbose
        self.batch_size = batch_size
        self.model = self.build_model()
        self.compile()
        if verbose:
            self.model.summary()

    def build_model(self):
        """ Build Model """
        inp = Input(shape=self.data_shape)
        x = inp
        if self.conv:
            for filters in self.filter_widths:
                x = conv_wrapper(Conv2D)(filters)(x)
                x = LeakyReLU(0.2)(x)
            x = Flatten()(x)
            x = Dense(64, use_bias=False)(x)
            x = LeakyReLU(0.2)(x)
            x = Dense(64, use_bias=False)(x)
            return Model(inp, x)
        else:
            x = Dense(256, use_bias=False)(x)
            x = LeakyReLU(0.2)(x)
            x = Dense(128, use_bias=False)(x)
            x = LeakyReLU(0.2)(x)
            x = Dense(64, use_bias=False)(x)
            return Model(inp, x)

    def compile(self):
        self.model.compile(loss='mse',
                           optimizer=self.optimizer)

    def train(self, train_data, num_batches, sample_interval):
        c = np.ones((self.batch_size, 64)) * 0.1

        for batch_idx in range(num_batches+1):
            idx = np.random.randint(0, train_data.shape[0], self.batch_size)
            samples = train_data[idx]
            loss = self.model.train_on_batch(samples, c)

            if batch_idx % sample_interval == 0 and self.verbose:
                self.save(batch_idx)
                print(loss)
        self.save()

    def save(self, suffix=None):
        print('Saving model to: %s' % self.model_dir)
        if suffix is None:
            path = self.model_dir + '/model'
        else:
            path = self.model_dir + '/model_%s' % suffix
        atongtf.util.save(self.model, path)


if __name__ == '__main__':
    atongtf.util.set_config()
    data = atongtf.dataset.Mnist_Anomaly_Dataset(0, frac_out=0, verbose=True)
    model = DeepSVDD('tmp/', data.get_shape())
    model.train(data.get_train(), num_batches = 20000, sample_interval=1000)
