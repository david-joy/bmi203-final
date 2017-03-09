""" Layers for the Neural Network

API Based on the model mini-language from Keras, but with way fewer options.

Each layer maintains a weight and bias matrix, and an activation.
"""

# Imports
import numpy as np

# Constants
WEIGHT_INIT_SCALE = 1e-3


# Functions


def sigmoid(x):
    # Logistic function

    # Need to gracefully handle over/underflow
    x_over = x > 100
    x_under = x < -100
    y = 1.0 / (1.0 + np.exp(-x))
    y[x_over] = 1.0
    y[x_under] = 0.0
    return y


def sigmoid_prime(z):
    # Derivative of the logistic function
    # Note that this is in terms of y, not x
    f = sigmoid(z)
    return f * (1 - f)


def relu(x):
    # Rectified linear unit
    f = x.copy()
    f[f < 0] = 0
    return f


def relu_prime(z):
    return (z >= 0).astype(np.float32)


# Classes


class Layer(object):
    """ Base class for layers """

    @classmethod
    def from_dict(cls, layer_data):
        """ Create a layer from the dictionary spec for it

        :param layer_data:
            A dictionary with 'type': layer.Layer class and keys for the other
            arguments to the layer initailizer
        :returns:
            That layer, initialized with those arguments
        """
        layer_type = layer_data.pop('type')
        # Lookup all subclasses of the Layer base class
        layer_subclasses = {c.__name__: c for c in Layer.__subclasses__()}
        return layer_subclasses[layer_type](**layer_data)

    def to_dict(self):
        raise NotImplementedError("Implement conversion to dictionary")


class FullyConnected(Layer):
    """ Implement a fully connected layer

    This layer represents a fully connected layer with an activation function
    """

    activations = {
        'sigmoid': sigmoid,
        'relu': relu,
    }
    gradients = {
        'sigmoid': sigmoid_prime,
        'relu': relu_prime,
    }

    def __init__(self, size, func):
        self.size = size
        self.func = func

        # Weight and bias matrices
        self.activation = self.activations[func]
        self.gradient = self.gradients[func]

        self.weight = None
        self.bias = None

        # Memorize the last activation
        self.z = None
        self.a = None

    def __eq__(self, other):
        return self.size == other.size

    def to_dict(self):
        """ Convert this layer to a dictionary """
        return {'type': 'FullyConnected',
                'func': self.func,
                'size': self.size}

    def init_weights(self, prev_size):
        """ Initialize the weight matrix and bias vector

        :param prev_size:
            The size of the previous layer
        """

        # Input x will be prev_size
        # Weight will be size x prev_size
        # Output will be size
        self.weight = np.random.uniform(low=-WEIGHT_INIT_SCALE,
                                        high=WEIGHT_INIT_SCALE,
                                        size=(self.size, prev_size))

        self.bias = np.random.uniform(low=-WEIGHT_INIT_SCALE,
                                      high=WEIGHT_INIT_SCALE,
                                      size=(self.size, 1))

    def set_weights(self, weight, bias):
        """ Set the weights to given values

        :param weight:
            A size x prev_size numpy array
        :param bias:
            A size x 1 numpy array
        """
        weight = np.squeeze(weight)
        if weight.ndim != 2:
            err = 'Expected 2D weight matrix, got {}'.format(weight.shape)
            raise ValueError(err)

        bias = np.squeeze(bias)
        if bias.ndim != 1:
            err = 'Expected 1D bias vector, got {}'.format(bias.shape)
            raise ValueError(bias)

        if weight.shape[0] != self.size:
            err = 'Expected weight matrix {} x m, got {}'
            err = err.format(self.size, *weight.shape)
            raise ValueError(err)

        if bias.shape[0] != self.size:
            err = 'Expected bias matrix {} x 1, got {}'
            err = err.format(self.size, *bias.shape)
            raise ValueError(err)

        self.weight = weight
        self.bias = bias[:, np.newaxis]

    def predict(self, x):
        """ Calculate the forward prediction

        :param x:
            A numpy array of prev_size x 1
        :returns:
            A numpy array of size x 1
        """
        if x.ndim == 1:
            x = x[:, np.newaxis]

        self.z = self.weight @ x + self.bias
        self.a = self.activation(self.z)
        return self.a

    def calc_error(self, ytarget):
        if ytarget.ndim == 1:
            ytarget = ytarget[:, np.newaxis]
        return (self.a - ytarget) * self.gradient(self.z)

    def calc_delta(self, delta):
        """ Calculate the update """
        return (self.weight @ delta) * self.gradient(self.z)

    def update_weights(self, delta, learn_rate=1.0, weight_decay=0.0):
        """ Calculate the weight update """

        delta_weight = self.a.T @ delta
        delta_bias = delta

        self.weight -= learn_rate * (delta_weight + weight_decay * self.weight)
        self.bias -= learn_rate * delta_bias


# Functions


def from_dict(layer_data):
    """ Create a layer from the dictionary spec for it

    :param layer_data:
        A dictionary with 'type': layer.Layer class and keys for the other
        arguments to the layer initailizer
    :returns:
        That layer, initialized with those arguments
    """
    return Layer.from_dict(layer_data)
