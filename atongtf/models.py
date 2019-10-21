"""
models.py

Implements Keras Autoencoder Models:
    BaseAE
    ConvAE
    DenoisingAE
    DenseAE

"""
import functools
import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Conv2D, GaussianNoise, Dense, Lambda, Reshape, Layer
from tensorflow.keras.layers import Conv2DTranspose, Cropping2D, LeakyReLU, Flatten, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
import tensorflow.keras.backend as K
import tensorflow.keras.losses as losses


def load_model(path):
    with open(path + '.json', 'r') as file:
        json = file.read()
    model = model_from_json(json)
    model.load_weights(path + '.h5')
    return model


def save_model(model, path):
    mjson = model.to_json()
    with open(path + '.json', "w") as file:
        file.write(mjson)
    model.save_weights(path+'.h5')


def load_ae(path):
    model = load_model(path + '/model')
    encoder = load_model(path + '/encoder')
    decoder = load_model(path + '/decoder')
    return model, encoder, decoder


class BaseAE():
    def __init__(self, model_dir, data_shape, verbose=1):
        self.model_dir = model_dir
        os.makedirs(os.path.dirname(self.model_dir + '/'), exist_ok=True)
        self.data_shape = data_shape
        self.input = Input(shape=data_shape)
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.model = self.build_autoencoder()
        self.verbose = verbose
        if verbose:
            self.summary()
        self.compile()

    def build_autoencoder(self):
        return Model(self.input, self.decoder(self.encoder(self.input)))

    def build_encoder(self):
        raise NotImplementedError

    def build_decoder(self):
        raise NotImplementedError

    def compile(self):
        self.model.compile(optimizer='adam', loss='mse')

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def save(self):
        save_model(self.model, self.model_dir + '/model')
        save_model(self.encoder, self.model_dir + '/encoder')
        save_model(self.decoder, self.model_dir + '/decoder')

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)

    def train(self, train_data, num_batches, batch_size):
        for batch_idx in range(num_batches+1):
            idx = np.random.randint(0, train_data.shape[0], batch_size)
            sample = train_data[idx]
            self.model.train_on_batch(sample, sample)

class DenseAE(BaseAE):
    def __init__(self, model_dir, data_shape,
                 layer_widths=None, latent_dim=32):
        if layer_widths is None:
            layer_widths = [128, 128, 128]
        self.layer_widths = layer_widths
        self.latent_dim = latent_dim
        super().__init__(model_dir, data_shape)

    def build_encoder(self):
        x = self.input
        x = Flatten()(x)
        for w in self.layer_widths:
            x = Dense(w)(x)
            x = LeakyReLU(0.2)(x)
        self.embedding = Dense(self.latent_dim)(x)
        return Model(self.input, self.embedding)

    def build_decoder(self):
        decoder_input = Input(shape=(self.latent_dim,))
        x = decoder_input
        for w in self.layer_widths[::-1]:
            x = LeakyReLU(0.2)(Dense(w)(x))
        x = Dense(functools.reduce(lambda x, y: x*y, self.data_shape))(x)
        self.output = Reshape(self.data_shape)(x)
        return Model(decoder_input, self.output)

class DenseDenoisingAE(DenseAE):
    def __init__(self, model_dir, data_shape,
                 layer_widths=None, latent_dim=32, noise=0.1):
        self.noise = noise
        super().__init__(model_dir, data_shape, layer_widths, latent_dim)

    def build_encoder(self):
        x = self.input
        x = Flatten()(x)
        x = GaussianNoise(self.noise)(x)
        for w in self.layer_widths:
            x = Dense(w)(x)
            x = LeakyReLU(0.2)(x)
        self.embedding = Dense(self.latent_dim)(x)
        return Model(self.input, self.embedding)



class ConvAE(BaseAE):
    """ Convolutional autoencoder, strided convolutions.
    Data must be in channels_last format. Depth doubles and width halves at
    every encoder layer. Vis versa for the decoder. Image is cropped at the 
    end if not the right shape.
    Args:
        nfilt: (int) number of filters in first layer.
        data_shape: (tuple) should be (height, width, depth)
        model_dir: (str) location to save model data
        strides: (int) Convolutional layer strides
    """
    def __init__(self, model_dir, data_shape, nfilt=16, 
                 strides=2, kernel_size=4):
        assert len(data_shape) == 3
        self.nfilt = nfilt
        self.log_width = int(math.ceil(math.log(data_shape[1], 2)))
        # reduce down to 4x4xD
        self.layer_filters = [nfilt << i for i in range(self.log_width - 2)]
        self.strides = strides
        self.kernel_size = kernel_size
        super().__init__(model_dir, data_shape)

    def conv_wrapper(self, layer):
        return functools.partial(layer,
                                 kernel_size=self.kernel_size,
                                 strides=self.strides,
                                 padding='same',
                                 # tf.nn.leaky_relu cannot be saved
                                 # activation=tf.nn.leaky_relu,
                                 kernel_initializer='he_normal')

    def build_encoder(self):
        x = self.input
        for filters in self.layer_filters:
            x = self.conv_wrapper(Conv2D)(filters)(x)
            x = LeakyReLU(0.2)(x)
        self.embedding = x
        return Model(self.input, self.embedding)

    def build_decoder(self):
        _, width, depth = self.data_shape
        # ignore batch dimension
        decoder_input = Input(shape=self.embedding.get_shape()[1:])
        x = decoder_input
        print(self.layer_filters)
        for filters in self.layer_filters[::-1]:
            x = self.conv_wrapper(Conv2DTranspose)(filters)(x)
            x = LeakyReLU(0.2)(x)
        x = Conv2D(depth, self.kernel_size, padding='same')(x)
        # TODO: Cropping doesn't seem like the best way to do this
        # Crop back to input dimensions
        self.output = Cropping2D((2 ** self.log_width - width) // 2)(x)
        return Model(decoder_input, self.output)

class ConvAE_Flat_Embedding(ConvAE):
    def build_encoder(self):
        x = self.input
        for filters in self.layer_filters:
            x = self.conv_wrapper(Conv2D)(filters)(x)
            x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        x = Dense(10)(x)
        self.embedding = x
        return Model(self.input, self.embedding, name='Encoder')

    def build_decoder(self):
        _, width, depth = self.data_shape
        # ignore batch dimension
        decoder_input = Input(shape=self.embedding.get_shape()[1:])
        x = decoder_input
        x = Dense(256)(x)
        x = LeakyReLU(0.2)(x)
        x = Reshape((4, 4, 16))(x)
        for filters in self.layer_filters[::-1]:
            x = self.conv_wrapper(Conv2DTranspose)(filters)(x)
            x = LeakyReLU(0.2)(x)
        x = Conv2D(depth, self.kernel_size, padding='same')(x)
        # TODO: Cropping doesn't seem like the best way to do this
        # Crop back to input dimensions
        self.output = Cropping2D((2 ** self.log_width - width) // 2)(x)
        return Model(decoder_input, self.output, name='Decoder')


class DenoisingAE(ConvAE):
    def __init__(self, nfilt, data_shape, model_dir, 
                 input_noise=0.0, **kwargs):
        super().__init__(nfilt, data_shape, model_dir, **kwargs)
        self.input_noise = input_noise

    def build_encoder(self):
        x = self.input
        if self.input_noise:
            x = GaussianNoise(self.input_noise)(x)
        for filters in self.layer_filters:
            x = self.conv_wrapper(Conv2D)(filters)(x)
            x = LeakyReLU(0.2)(x)
        self.embedding = x
        return Model(self.input, self.embedding)


class ArchetypeAE(ConvAE):
    def __init__(self, nfilt, data_shape, model_dir, n_latent=3, **kwargs):
        """
        Constructs an archetypal autoencoder (AAE).
        An AAE has the property that all of the points are located
        within the [0,1]^n_latent space.
        Args:
            n_latent (int): number of latent dimensions / archetypes
        """
        self.n_latent = n_latent
        super().__init__(nfilt, data_shape, model_dir, **kwargs)

    def build_encoder(self):
        x = self.input
        for filters in self.layer_filters:
            x = self.conv_wrapper(Conv2D)(filters)(x)
            x = LeakyReLU(0.2)(x)
        x = Conv2D(filters=self.n_latent, kernel_size=4, 
                   activation='sigmoid')(x)
        x = layers.Flatten()(x)
        x = Lambda(lambda x: x / (tf.reduce_sum(tf.abs(x), 
                                                axis=1, keep_dims=True)))(x)
        self.embedding = x
        return Model(self.input, self.embedding)

    def build_decoder(self):
        _, width, depth = self.input
        # ignore batch dimension
        decoder_input = Input(shape=self.embedding.get_shape()[1:])
        x = decoder_input
        x = Dense(4*4*10*self.n_latent)(x)
        x = layers.Reshape((4, 4, 10*self.n_latent))(x)
        for filters in self.layer_filters[::-1]:
            x = self.conv_wrapper(Conv2DTranspose)(filters)(x)
            x = LeakyReLU(0.2)(x)
        x = Conv2D(depth, self.kernel_size, padding='same')(x)
        # TODO: Cropping doesn't seem like the best way to do this
        # Crop back to input dimensions
        self.output = Cropping2D((2 ** self.log_width - width) // 2)(x)
        return Model(decoder_input, self.output)

class RCAE(ConvAE):
    def build_encoder(self):
        return Model(self.input, self.encoder(self.input))

    def encoder(self,input_img):
        # encoder
        # input = 28 x 28 x 1 (wide and thin)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # 28 x 28 x 32
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  # 14 x 14 x 64
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)  # 7 x 7 x 128 (small and thick)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)  # 7 x 7 x 256 (small and thick)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        self.embedding = conv4
        return conv4

    def build_decoder(self):
        decoder_input = Input(shape=self.embedding.get_shape()[1:])
        return Model(decoder_input, self.decoder(decoder_input))

    def decoder(self,conv4):
        # decoder
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)  # 7 x 7 x 128
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)  # 7 x 7 x 64
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)
        up1 = UpSampling2D((2, 2))(conv6)  # 14 x 14 x 64
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)  # 14 x 14 x 32
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)
        up2 = UpSampling2D((2, 2))(conv7)  # 28 x 28 x 32
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)  # 28 x 28 x 1
        return decoded


class CAE(ConvAE):
    def compile(self):
        self.model.compile(optimizer='adam', loss='mse')


class KLDivergenceLayer(Layer):
    def __init__(self, *args, **kwargs):
        self.is_placeholder=True
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs

        kl_batch = -0.5*K.sum(1 + log_var - 
                              K.square(mu) - 
                              K.exp(log_var), axis=-1)
        self.add_loss(K.mean(kl_batch), inputs=inputs)
        return inputs


class VAE(DenseAE):
    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size=batch_size
        super().__init__(*args, **kwargs)
    def build_encoder(self):
        x = self.input
        x = Flatten()(x)
        for w in self.layer_widths:
            x = Dense(w)(x)
            x = LeakyReLU(0.2)(x)
        self.z_mean = Dense(self.latent_dim)(x)
        self.z_log_var = Dense(self.latent_dim)(x)
        self.z_mean, self.z_log_var = KLDivergenceLayer()([self.z_mean, self.z_log_var])

        def sampling(args):
            (z_mean, z_log_sigma) = args
            print(z_mean.shape)
            epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim))
            return z_mean + K.exp(z_log_sigma) * epsilon
        self.embedding = Lambda(sampling, output_shape=(self.latent_dim,), name='embedding')([self.z_mean, self.z_log_var])
        return Model(self.input, self.embedding)

    def build_decoder(self):
        decoder_input = Input(shape=(self.latent_dim,))
        x = decoder_input
        for w in self.layer_widths[::-1]:
            x = LeakyReLU(0.2)(Dense(w)(x))
        x = Dense(functools.reduce(lambda x, y: x*y, self.data_shape))(x)
        self.output = Reshape(self.data_shape)(x)
        return Model(decoder_input, self.output)

    def compile(self):
        self.model.compile(optimizer='adam', loss=losses.binary_crossentropy)
