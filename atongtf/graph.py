"""
graph.py

Implements graph based regularization layer.
"""
import tensorflow as tf

class GraphLayer(tf.keras.layers.Layer):
    """ Outputs the laplacian and applies it to the input layer as a regularizer

    Given activation vector x, and laplacian L call returns x^t*L*x.
    """
    def __init__(self, name='graph_layer', initializer=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.laplacian_initializer = initializer
        if initializer is None:
            self.laplacian_initializer = tf.keras.initializers.Zeros
        self.laplacian = None

    def build(self, input_shape):
        width = input_shape.as_list()[1]
        self.laplacian = self.add_weight(name='laplacian',
                                         shape=[width, width],
                                         dtype=self.dtype,
                                         initializer=self.laplacian_initializer,
                                         trainable=False)

    def call(self, inputs, training=None):
        """ Computes x^t L x for inputs x and laplacian L """
        with tf.device("/device:GPU:0"):
            return tf.reshape(tf.einsum('bj,jk,bk->b', inputs, self.laplacian, inputs), (-1, 1))
