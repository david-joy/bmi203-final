""" Master object to combine the layers into a net """

# Imports
import json
import pathlib

import numpy as np

from . import layers

# Classes


class Model(object):
    """ Model object

    To build an 8-3-8 autoencoder::

        auto = Model('8-3-8', input_size=8)
        auto.add_layer(layers.FullyConnected(3, func='sigmoid'))
        auto.add_layer(layers.FullyConnected(8, func='sigmoid'))

    """

    def __init__(self, name=None, input_size=None, rng=None):
        self.name = name
        self.input_size = input_size
        self.layers = []

        if rng is None:
            rng = np.random
        self.rng = rng

        self.opt = None

    def __eq__(self, other):
        if len(self.layers) != len(other.layers):
            return False
        for sl, ol in zip(self.layers, other.layers):
            if type(sl) != type(ol):
                return False
            if sl != ol:
                return False
        return True

    def set_optimizer(self, opt):
        """ Set the optimizer """
        self.opt = opt

    def init_weights(self):
        """ Initialize the layer's weights """

        prev_size = self.input_size

        for layer in self.layers:
            layer.init_weights(prev_size, rng=self.rng)
            prev_size = layer.size

    def predict(self, x):
        """ Predict the output from the input """
        if x.ndim == 1:
            x = x[:, np.newaxis]

        for i, layer in enumerate(self.layers):
            x = layer.predict(x)
        return x

    def calc_error(self, y, yhat):
        """ Calculate the MSE """
        return np.sum((yhat - y)**2, axis=0)[np.newaxis, :]

    def gradient_descent(self, x, y):
        """ Implement gradient descent """
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if y.ndim == 1:
            y = y[:, np.newaxis]

        assert x.shape[1] == y.shape[1]

        yhat = self.predict(x)

        assert yhat.shape == y.shape

        # Backprop the error
        deltas = [self.layers[-1].calc_error(y)]
        for layer in reversed(self.layers[1:]):
            deltas.append(layer.calc_delta(deltas[-1]))
        assert len(deltas) == len(self.layers)

        # Convert the deltas to gradients
        weights = self.get_weight_list()
        grads = self.get_grad_list(list(reversed(deltas)))

        # Use the optimizer to update the weights
        new_weights = self.opt.update(weights, grads)

        # Update the weights
        self.set_weight_list(new_weights)

        # Return the MSE for each prediction
        return self.calc_error(y, yhat)

    def add_layer(self, layer):
        """ Add a layer to the model """
        if len(self.layers) == 0:
            layer.prev_size = self.input_size
        else:
            layer.prev_size = self.layers[-1].size
        self.layers.append(layer)

    def save_model(self, modelfile):
        """ Dump the model specification

        :param modelfile:
            The JSON file to write the specification to
        """
        modelfile = pathlib.Path(modelfile)

        # Convert the model to a JSON compatible dictionary
        data = {'name': self.name,
                'input_size': self.input_size,
                'layers': [l.to_dict() for l in self.layers]}
        with modelfile.open('wt') as fp:
            json.dump(data, fp,
                      sort_keys=True,
                      indent=4,
                      separators=(',', ': '))

    def save_weights(self, weightfile):
        """ Dump the model weights

        :param weightfile:
            The numpy npz file to write to
        """
        weightfile = pathlib.Path(weightfile)
        layer_data = {}
        for i, layer in enumerate(self.layers):
            prefix = '{}-{:02d}-'.format(type(layer).__name__, i)
            layer_data[prefix + 'weight'] = layer.weight
            layer_data[prefix + 'bias'] = layer.bias
        np.savez(str(weightfile), **layer_data)

    def load_weights(self, weightfile):
        """ Load weights from a file

        :param weightfile:
            The numpy npz file to read from
        """
        layer_data = np.load(str(weightfile))
        layer_keys = set(layer_data.keys())

        for i, layer in enumerate(self.layers):
            prefix = '{}-{:02d}-'.format(type(layer).__name__, i)
            weight = layer_data[prefix + 'weight']
            bias = layer_data[prefix + 'bias']

            layer_keys.remove(prefix + 'weight')
            layer_keys.remove(prefix + 'bias')

            layer.set_weights(weight, bias)

        if len(layer_keys) > 0:
            raise ValueError('Got extra layer data: {}'.format(layer_keys))

    def get_weight_list(self):
        """ Get all the weights as a list """
        weights = []
        for layer in self.layers:
            weights.append(layer.weight)
            weights.append(layer.bias)
        return weights

    def get_grad_list(self, deltas):
        """ Get all the gradients as a list

        :param deltas:
            The deltas, IN FORWARD PASS ORDER
        :returns:
            The list of gradients
        """
        assert len(deltas) == len(self.layers)

        grads = []
        for layer, delta in zip(self.layers, deltas):
            d_weight, d_bias = layer.calc_grad(delta)
            grads.append(d_weight)
            grads.append(d_bias)
        return grads

    def set_weight_list(self, weights):
        """ Set all the weights from a list """
        assert len(weights) == len(self.layers) * 2
        for i, w in enumerate(weights):
            li = i // 2
            layer = self.layers[li]
            if i % 2 == 0:
                layer.weight = w
            else:
                layer.bias = w

    @classmethod
    def load_model(cls, modelfile):
        """ Load a model from the specficiation

        :param modelfile:
            The model spec, as written by save model
        :returns:
            A Model instance loaded from the file
        """
        modelfile = pathlib.Path(modelfile)

        with modelfile.open('rt') as fp:
            data = json.load(fp)

        # Build up the model layer by layer, just like normal
        model = cls(name=data.get('name'),
                    input_size=data.get('input_size'))
        for layer in data.get('layers', []):
            model.add_layer(layers.from_dict(layer))
        return model


def load_model(modelfile):
    """ Top level helper constructor """
    return Model.load_model(modelfile)
