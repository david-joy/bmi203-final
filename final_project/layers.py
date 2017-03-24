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


def sigmoid_prime(a):
    # Derivative of the logistic function
    # Note that this is in terms of f(x) (aka a, the activation)
    # And not in terms of x
    return a * (1.0 - a)


def relu(x):
    # Rectified linear unit
    f = x.copy()
    f[f < 0] = 0
    return f


def relu_prime(a):
    # Derivative of the ReLU function
    # Note that this is in terms of f(x) (aka a, the activation)
    # And not in terms of x
    return (a > 0).astype(np.float32)


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

    :param size:
        The size of the OUTPUT of this layer
    :param func:
        The name of the activation function for this layer
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
        self.prev_size = None
        self.x = None
        self.z = None
        self.a = None

    def __eq__(self, other):
        return self.size == other.size

    def to_dict(self):
        """ Convert this layer to a dictionary

        :returns:
            A dictionary describing this layer
        """
        return {'type': 'FullyConnected',
                'func': self.func,
                'size': self.size}

    def init_weights(self, prev_size, rng=None):
        """ Initialize the weight matrix and bias vector

        :param prev_size:
            The size of the previous layer
        :param rng:
            If not None, the numpy.random.RandomState object to use for all
            random numbers
        """
        if rng is None:
            rng = np.random

        # Input x will be prev_size x k
        # Weight will be size x prev_size
        # Bias will be size x 1
        # Output will be size x k
        self.weight = rng.uniform(low=-WEIGHT_INIT_SCALE,
                                  high=WEIGHT_INIT_SCALE,
                                  size=(self.size, prev_size))

        self.bias = rng.uniform(low=-WEIGHT_INIT_SCALE,
                                high=WEIGHT_INIT_SCALE,
                                size=(self.size, 1))
        self.prev_size = prev_size

    def set_weights(self, weight, bias):
        """ Set the weights to given values

        :param weight:
            A size x prev_size numpy array
        :param bias:
            A size x 1 numpy array
        """
        weight = np.squeeze(weight)
        if weight.ndim != 2:
            if self.size == 1:
                weight = weight[np.newaxis, :]
            else:
                err = 'Expected 2D weight matrix, got {}'.format(weight.shape)
                raise ValueError(err)

        bias = np.squeeze(bias)
        if bias.ndim != 1:
            if self.size == 1:
                bias = np.array([bias])
            else:
                err = 'Expected 1D bias vector, got {}'.format(bias.shape)
                raise ValueError(err)

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
            A numpy array of prev_size x k
        :returns:
            A numpy array of size x k
        """
        if x.ndim == 1:
            x = x[:, np.newaxis]

        self.x = x
        self.z = self.weight @ x + self.bias
        self.a = self.activation(self.z)
        return self.a

    def calc_error(self, ytarget):
        """ Calculate the error for the OUTPUT layer

        :param ytarget:
            The size x k array of y values to learn
        :returns:
            The error delta for the OUTPUT layer
        """
        if ytarget.ndim == 1:
            ytarget = ytarget[:, np.newaxis]
        assert ytarget.shape == self.a.shape

        delta = self.a - ytarget
        grad = self.gradient(self.a)
        assert delta.shape == grad.shape

        return delta * grad

    def calc_delta(self, delta):
        """ Calculate the error for a hidden layer

        :param delta:
            The size x k error delta from the next layer
        :returns:
            The prev_size x k error delta for this layer
        """
        if delta.ndim == 1:
            delta = delta[:, np.newaxis]

        # Run the previous delta backwards through the weight matrix
        delta = (self.weight.T @ delta)
        grad = self.gradient(self.x)
        assert delta.shape == grad.shape

        return delta * grad

    def calc_grad(self, delta):
        """ Calculate the gradient from the delta

        :param delta:
            The size x k error delta from the next layer
        :returns:
            The size x prev_size weight matrix gradient
            and the size x 1 bias matrix gradient
        """
        if delta.ndim == 1:
            delta = delta[:, np.newaxis]

        assert delta.shape[0] == self.size
        assert delta.shape[1] == self.x.shape[1]

        delta_weight = delta @ self.x.T / delta.shape[1]
        delta_bias = np.mean(delta, axis=1)[:, np.newaxis]

        assert delta_weight.shape == self.weight.shape
        assert delta_bias.shape == self.bias.shape
        return delta_weight, delta_bias

    def update_weights(self, delta, learn_rate=1.0, weight_decay=0.0):
        """ Calculate the weight update

        .. warning:: This is a simple optimizer that doesn't work very well

        :param delta:
            The error for the current layer
        :param learn_rate:
            The learning rate for this update
        :param weight_decay:
            The weight_decay for this update
        """
        delta_weight, delta_bias = self.calc_grad(delta)

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
