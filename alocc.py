import functools
import os
import math

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model, model_from_json, Sequential
from tensorflow.keras.layers import Input, Conv2D, GaussianNoise, Dense, Lambda, Reshape, Layer
from tensorflow.keras.layers import Conv2DTranspose, Cropping2D, LeakyReLU, Flatten, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Average, InputSpec
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow.keras.losses as losses

import models
import atongtf.models
import atongtf.dataset
import atongtf.util

class ALOCC():
    def __init__(self, model_dir, data_shape, verbose=1, conv=False,
            batch_size=256, noise=0.2, beta_1 = 0.9, beta_2 = 0.99):
        self.filter_widths = [16, 32, 64]
        self.layer_widths = [256, 128, 64]
        #self.layer_widths = [1024, 512, 256]
        self.optimizer = Adam(beta_1=beta_1)
        self.model_dir = model_dir
        self.verbose = verbose
        self.conv = conv
        self.batch_size = batch_size
        self.noise = noise
        # Make dir if does not exist
        os.makedirs(os.path.dirname(self.model_dir + '/'), exist_ok=True)
        os.makedirs(os.path.dirname(self.model_dir + '/data/'), exist_ok=True)
        self.data_shape = data_shape
        self.build()

    def build(self):
        self.d = self.build_dae()
        self.r = self.build_discriminator()
        if self.verbose:
            self.d.summary()
            self.r.summary()
        self.r.compile(loss='binary_crossentropy',
                       optimizer=self.optimizer,
                       metrics=['accuracy'])
        self.d.compile(loss='mse', optimizer=self.optimizer)
        z = Input(shape=self.data_shape)

        sample = self.d(z)

        self.r.trainable = False
        score = self.r(sample)

        self.combined = Model(z, score)
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=self.optimizer)


    def conv_wrapper(self, layer):
        return functools.partial(layer,
                                 kernel_size=4,
                                 strides=2,
                                 padding='same',
                                 # tf.nn.leaky_relu cannot be saved
                                 # activation=tf.nn.leaky_relu,
                                 kernel_initializer='he_normal')

    def build_dae(self):
        inp = Input(shape=self.data_shape)
        x = inp
        x = GaussianNoise(self.noise)(x)
        layer_widths = [128, 96, 64]
        if self.conv:
            for filters in self.filter_widths:
                x = self.conv_wrapper(Conv2D)(filters)(x)
                x = LeakyReLU(0.2)(x)
            x = Flatten()(x)
            x = Dense(10)(x)
            self.embedding = x
            _, width, depth = self.data_shape
            # ignore batch dimension
            x = Dense(256)(x)
            x = LeakyReLU(0.2)(x)
            x = Reshape((4, 4, 16))(x)
            for filters in self.filter_widths[::-1]:
                x = self.conv_wrapper(Conv2DTranspose)(filters)(x)
                x = LeakyReLU(0.2)(x)
            x = Conv2D(depth, 4, padding='same')(x)
            # TODO: Cropping doesn't seem like the best way to do this
            # Crop back to input dimensions
            self.output = Cropping2D((32 - width) // 2)(x)
        else:
            x = Flatten()(x)
            for w in layer_widths:
                x = Dense(w)(x)
                x = LeakyReLU(0.2)(x)
            x = Dense(10)(x)
            for w in layer_widths[::-1]:
                x = LeakyReLU(0.2)(Dense(w)(x))
            x = Dense(functools.reduce(lambda x, y: x*y, self.data_shape))(x)
            self.output = Reshape(self.data_shape)(x)
        return Model(inp, self.output, name='Decoder')

    
    def build_discriminator(self):
        """ Build Discriminator """
        inp = Input(shape=self.data_shape)
        x = inp
        if self.conv:
            for filters in self.filter_widths:
                x = self.conv_wrapper(Conv2D)(filters)(x)
                x = LeakyReLU(0.2)(x)
            x = Flatten()(x)
            x = Dense(64)(x)
            x = LeakyReLU(0.2)(x)
            x = Dense(64)(x)
            x = LeakyReLU(0.2)(x)
            x = Dense(1, activation='sigmoid')(x)
            return Model(inp, x)
        else:
            layer_widths = [256, 128, 64]
            x = Flatten()(x)
            for width in layer_widths:
                x = Dense(width)(x)
                x = LeakyReLU()(x)
            x = Dense(1, activation='sigmoid')(x)
            return Model(inp, x)

    def train(self, train_data, num_batches, batch_size=128, sample_interval=500):
        # Adversarial ground truths
        true_gt = np.ones((batch_size, 1))
        fake_gt = np.zeros((batch_size, 1))

        for batch_idx in range(num_batches):
            idx = np.random.randint(0, train_data.shape[0], batch_size)
            real = train_data[idx]
            fake = self.d.predict(real)

            # Train dsicriminator
            d_loss_real = self.r.train_on_batch(real, true_gt)
            d_loss_fake = self.r.train_on_batch(fake, fake_gt)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            # TODO Why do we want new noise??? (Alex)
            idx2 = np.random.randint(0, train_data.shape[0], batch_size)
            fake2 = self.d.predict(train_data[idx2])

            # two generator steps
            g_loss = self.combined.train_on_batch(fake2, true_gt) 
            r_loss = 0.2 * self.d.train_on_batch(real, real)

            if self.verbose:
                print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]' %
                      (batch_idx, d_loss[0], 100*d_loss[1], g_loss))
    
    def save(self):
        path = self.model_dir + '/model'
        atongtf.util.save(self.combined, path)

if __name__ == '__main__':
    atongtf.util.set_config(gpu_idx='auto')
    path = '/tmp/alocc'
    batch_size=128
    num_batches=1000
    d = atongtf.dataset.VACS_Dataset(frac_out=0)
    #d = atongtf.dataset.Mnist_Dataset()
    m = ALOCC(path, data_shape=d.get_shape(), batch_size=batch_size, conv=False, noise=0.155) # From paper
    #m = ALOCC(path, data_shape=d.get_shape(), batch_size=batch_size, conv=True, noise=0.155) # From paper
    m.train(d.get_train(), num_batches, sample_interval=num_batches // 5)
    m.save()
    #m = models.load_model('%s/model' % path)
    #scores = m.predict(d.get_test())
    #print(scores, d.get_test_labels())
