"""
models.py
Author: Alexander Tong

Implements lipshitz conditioned networks with code based on:
https://github.com/eriklindernoren/Keras-GAN
https://github.com/pfnet-research/sngan_projection
"""
import itertools
import functools
from functools import partial
import math
import os

#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, model_from_json, Sequential
from tensorflow.keras.layers import Input, Conv2D, GaussianNoise, Dense, Lambda, Reshape, Layer
from tensorflow.keras.layers import Conv2DTranspose, Cropping2D, LeakyReLU, Flatten, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Average, InputSpec
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow.keras.losses as losses
from tensorflow.keras.utils import CustomObjectScope
from sklearn import datasets

import atongtf.util
import atongtf.image_transforms

def load_model(path):
    with open(path + '.json', 'r') as f:
        json = f.read()
    with CustomObjectScope({'DenseSN' : DenseSN, 'ConvSN2D': ConvSN2D}):
        m = model_from_json(json)
        #import json as js
        #d = js.loads(json)
        #print(js.dumps(d, sort_keys=True, indent=4))
        m.load_weights(path + '.h5')
        return m

def conv_wrapper(layer):
    return partial(layer,
                   kernel_size=4,
                   strides=2,
                   padding='same',
                   # tf.nn.leaky_relu cannot be saved
                   # activation=tf.nn.leaky_relu,
                   kernel_initializer='he_normal')

class Lipschitz_Network():
    """ Implements a network with a bounded lipschitz condition.
    A lipschitz condition can be enforced in a number of ways.

    We implement the following:
        No Bound
        Weight Clipping
        Gradient Penalty
        Spectral Normalization

    We also implement the following networks:
        Dense
        Convolutional

    These networks should be flexible enough to incorporate
    fake data specifically 1d and 2d distributions for visualization purposes,
    but also image shaped data with (height, width, depth) input.

    Creates a network function f: (Data_Shape) --> Real
    """
    def __init__(self, model_dir, data_shape, verbose=1, conv=False, big=False,
                 batch_size=256, noise=0.2, beta_1 = 0.9, beta_2 = 0.99, corrupt_patches=None):
        # Constants for all networks
        self.filter_widths = [16, 32, 64]
        self.layer_widths = [256, 128, 64]
        #self.layer_widths = [1024, 512, 256]
        self.optimizer = Adam(beta_1=beta_1)
        self.model_dir = model_dir
        self.verbose = verbose
        self.conv = conv
        self.big = big
        self.batch_size = batch_size
        self.noise = noise
        self.corrupt_patches = corrupt_patches
        # Make dir if does not exist
        os.makedirs(os.path.dirname(self.model_dir + '/'), exist_ok=True)
        os.makedirs(os.path.dirname(self.model_dir + '/data/'), exist_ok=True)
        self.data_shape = data_shape
        self.build()
        if self.verbose > 0:
            self.discriminator.summary()
        self.compile()

    def build(self):
        corruption_model = self.build_corruption()
        self.discriminator = self.build_discriminator()

        self.input = Input(shape=self.data_shape)
        self.corrupt_input = Input(shape=self.data_shape)
        corrupt = corruption_model(self.corrupt_input)

        real_out = self.discriminator(self.input)
        corrupt_out = self.discriminator(corrupt)

        self.model = Model(inputs=[self.input, self.corrupt_input],
                           outputs=[real_out, corrupt_out])

    def build_discriminator(self):
        if self.big:
            assert len(self.data_shape) == 3
            return self.build_big_conv_discriminator()
        if self.conv:
            assert len(self.data_shape) == 3
            return self.build_conv_discriminator()
        return self.build_dense_discriminator()

    def build_dense_discriminator(self):
        inp = Input(shape=self.data_shape)
        x = inp
        x = Flatten()(x)
        for width in self.layer_widths:
            x = Dense(width)(x)
            x = LeakyReLU()(x)
        x = Dense(1)(x)
        return Model(inp, x)

    def build_big_conv_discriminator(self):
        """ Build Discriminator """
        inp = Input(shape=self.data_shape)
        x = inp
        for filters in self.filter_widths:
            x = conv_wrapper(Conv2D)(filters)(x)
            x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        x = Dense(256)(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(256)(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(1)(x)
        return Model(inp, x)

    def build_conv_discriminator(self):
        """ Build Discriminator """
        inp = Input(shape=self.data_shape)
        x = inp
        for filters in self.filter_widths:
            x = conv_wrapper(Conv2D)(filters)(x)
            x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        x = Dense(64)(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(64)(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(1)(x)
        return Model(inp, x)

    def build_corruption(self):
        inp = Input(shape=self.data_shape)
        x = inp
        x = GaussianNoise(self.noise)(x)
        return Model(inp, x)

    def compile(self):
        self.model.compile(loss=[self.wasserstein_loss, 
                                 self.wasserstein_loss],
                           optimizer=self.optimizer)

    def wasserstein_loss(self, y_true, y_pred):
        """ if y_true is a vector of all -1s for fake and all 1s for real
        then this is equivalent to a wasserstein loss:
            E[f(true)] - E[f(fake)]
        """
        return K.mean(y_true * y_pred)

    def save(self, suffix=None):
        print('Saving model to: %s' % self.model_dir)
        if suffix is None:
            path = self.model_dir + '/model'
        else:
            path = self.model_dir + '/model_%s' % suffix
        atongtf.util.save(self.discriminator, path)

    def train(self, train_data, num_batches, sample_interval):
        raise NotImplementedError

    def plot_discriminator(self, batch_idx):
        xlim = (-3, 3)
        x = np.linspace(xlim[0], xlim[1], 100)
        points = np.array(list(itertools.product(x, repeat=2)))
        z = self.discriminator.predict(points)

        fig, ax = plt.subplots(1, 1)
        z = z.reshape(100, 100)
        z = z.transpose()
        np.save(self.model_dir + '/data/%d.npy' % batch_idx, z)

        plt.imshow(z, cmap=matplotlib.cm.coolwarm,
                   extent=[xlim[0], xlim[1], xlim[0], xlim[1]],
                   # vmin=0,
                   # vmax=1, 
                   origin='lower')
        plt.colorbar()
        plt.savefig(self.model_dir + '/discriminator_%d.png' % batch_idx)
        plt.close()

class Clipped_Lipschitz_Network(Lipschitz_Network):
    """ Implements a network with a lipshitz condition via weight clipping
    """
    def train(self, train_data, num_batches, sample_interval):
        """ Train specific to clipped networks """
        # Adversarial ground truths
        # This gives a large discriminator value on true points and a small
        # value on fake points
        true_gt = -np.ones((self.batch_size, 1))
        fake_gt = np.ones((self.batch_size, 1))

        for batch_idx in range(num_batches+1):
            # Should these be the same index?
            idx = np.random.randint(0, train_data.shape[0], self.batch_size)
            fake_idx = np.random.randint(0, train_data.shape[0], self.batch_size)
            real = train_data[idx]
            fake = train_data[fake_idx]
            loss = self.model.train_on_batch([real, fake],
                                             [true_gt, fake_gt])

            if batch_idx % sample_interval == 0:
                if self.verbose > 0:
                    print('%s %d [D loss: %0.3f]' % (self.model_dir, 
                                                     batch_idx, loss[0]))
                    if self.data_shape[0] == 2 and len(self.data_shape) == 1:
                        self.plot_discriminator(batch_idx)
                self.save(batch_idx)

            for layer in self.discriminator.layers:
                weights = layer.get_weights()
                weights = [np.clip(w, -0.01, 0.01) for w in weights]
                layer.set_weights(weights)

##############################################################################
# Gradient Penalty
##############################################################################

class RandomWeightedAverage(Average):
    """ Provides a (random) weighted average between real and 
    generated image samples
    """
    def __init__(self, data_shape, batch_size):
        self.batch_size = batch_size
        self.data_shape = data_shape
        super().__init__()

    def _merge_function(self, inputs):
        if len(self.data_shape) == 1:
            alpha = K.random_uniform((self.batch_size, 1))
        elif len(self.data_shape) == 3:
            alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        else:
            raise NotImplementedError
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class Gradient_Penalty_Lipschitz_Network(Lipschitz_Network):
    """ Implements gradient penalty enforcement of lipschitz condition"""
    def build(self):
        """ Build self.model with a third output for gradient penalty"""
        corruption_model = self.build_corruption()
        self.discriminator = self.build_discriminator()

        self.input = Input(shape=self.data_shape)
        self.corrupt_input = Input(shape=self.data_shape)
        corrupt = corruption_model(self.corrupt_input)
        rwa = RandomWeightedAverage(self.data_shape, self.batch_size)
        interpolated = rwa([self.input, self.corrupt_input])

        self.partial_gp_loss = partial(self.gradient_penalty_loss,
                                       averaged_samples=interpolated)
        self.partial_gp_loss.__name__ = 'gradient_penalty'

        real_out = self.discriminator(self.input)
        corrupt_out = self.discriminator(corrupt)
        interpolated_out = self.discriminator(interpolated)

        self.model = Model(inputs=[self.input, self.corrupt_input],
                           outputs=[real_out, corrupt_out, interpolated_out])

    def compile(self):
        """ Compile loss with gradient penalty, with default weight
        from paper
        """
        self.model.compile(loss=[self.wasserstein_loss,
                                 self.wasserstein_loss,
                                 self.partial_gp_loss],
                           optimizer=self.optimizer,
                           loss_weights=[1, 1, 10])

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted 
        real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def train(self, train_data, num_batches, sample_interval):
        """ Train specific to clipped networks """
        # Adversarial ground truths
        # This gives a large discriminator value on true points and a small
        # value on fake points
        true_gt = -np.ones((self.batch_size, 1))
        fake_gt = np.ones((self.batch_size, 1))
        dummy_gt = np.zeros((self.batch_size, 1))

        for batch_idx in range(num_batches+1):
            # Should these be the same index?
            idx = np.random.randint(0, train_data.shape[0], self.batch_size)
            fake_idx = np.random.randint(0, train_data.shape[0], self.batch_size)
            real = train_data[idx]
            fake = train_data[fake_idx]
            if self.corrupt_patches is not None:
                fake = atongtf.image_transforms.shuffle_patches(fake, self.corrupt_patches)
            loss = self.model.train_on_batch([real, fake],
                                             [true_gt, fake_gt, dummy_gt])

            if batch_idx % sample_interval == 0:
                if self.verbose > 0:
                    print('%s %d [D loss: %0.3f]' % (self.model_dir, 
                                                     batch_idx, loss[0]))
                    if self.data_shape[0] == 2 and len(self.data_shape) == 1:
                        self.plot_discriminator(batch_idx)
                self.save(batch_idx)

##############################################################################
# Spectral Normalization
##############################################################################

class DenseSN(Dense):
    """ Implements a dense layer with spectral normalization """
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = int(input_shape[-1])
        print(type(input_dim), type(input_shape))
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                 initializer=tf.keras.initializers.RandomNormal(0, 1),
                                 name='sn',
                                 trainable=False)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
        
    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        W_shape = self.kernel.shape.as_list()
        # Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        # Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        # normalize it
        W_bar = W_reshaped / sigma
        # reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                 W_bar = K.reshape(W_bar, W_shape)  
        output = K.dot(inputs, W_bar)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output 


class ConvSN2D(Conv2D):

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
            
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                         initializer=tf.keras.initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)
        
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        #Spectral Normalization
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
                
        outputs = K.conv2d(
                inputs,
                W_bar,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class Spectral_Normalized_Lipschitz_Network(Lipschitz_Network):
    """ Implements spectral normalized discriminator network """
    def build_dense_discriminator(self):
        inp = Input(shape=self.data_shape)
        x = inp
        x = Flatten()(x)
        for width in self.layer_widths:
            x = DenseSN(width)(x)
            x = LeakyReLU()(x)
        x = DenseSN(1)(x)
        return Model(inp, x)

    def build_conv_discriminator(self):
        """ Build Discriminator """
        inp = Input(shape=self.data_shape)
        x = inp
        for filters in self.filter_widths:
            x = conv_wrapper(ConvSN2D)(filters)(x)
            x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        x = DenseSN(64)(x)
        x = LeakyReLU(0.2)(x)
        x = DenseSN(64)(x)
        x = LeakyReLU(0.2)(x)
        x = DenseSN(1)(x)
        return Model(inp, x)

    def train(self, train_data, num_batches, sample_interval):
        """ Train specific to clipped networks """
        # Adversarial ground truths
        # This gives a large discriminator value on true points and a small
        # value on fake points
        true_gt = -np.ones((self.batch_size, 1))
        fake_gt = np.ones((self.batch_size, 1))
        n_samples = train_data.shape[0]

        for batch_idx in range(num_batches+1):
            # Should these be the same index?
            idx = np.random.randint(0, n_samples, self.batch_size)
            fake_idx = np.random.randint(0, n_samples, self.batch_size)
            real = train_data[idx]
            fake = train_data[fake_idx]
            loss = self.model.train_on_batch([real, fake], [true_gt, fake_gt])

            if batch_idx % sample_interval == 0:
                if self.verbose > 0:
                    print('%s %d [D loss: %0.3f]' % (self.model_dir, 
                                                     batch_idx, loss[0]))
                    if self.data_shape[0] == 2 and len(self.data_shape) == 1:
                        self.plot_discriminator(batch_idx)
                self.save(batch_idx)
        self.save()


if __name__ == '__main__':
    atongtf.util.set_config()
    # m = Clipped_Lipschitz_Network('model_dir/clipped_lipschitz', (2,))
    m = Gradient_Penalty_Lipschitz_Network('model_dir/gp_lipschitz_0.2_noise', (2,), batch_size=1024)
    # m = Spectral_Normalized_Lipschitz_Network('model_dir/sn_lipschitz_0.2_noise', (2,), batch_size=1024)
    moons, labels = datasets.make_moons(n_samples=2**16, noise=0.05)
    circles, labels = datasets.make_circles(n_samples=2**16, factor=0.9, noise = 0.05)
    m.train(circles, 10000, 100)
