""" Master object to combine the layers into a net """

# Imports

import json
import pathlib

from . import layers

# Classes


class Model(object):
    """ Model object

    To build an 8-3-8 autoencoder::

        auto = Model('8-3-8', input_size=8)
        auto.add_layer(layers.FullyConnected(3, func='sigmoid'))
        auto.add_layer(layers.FullyConnected(8, func='sigmoid'))

    """

    def __init__(self, name=None, input_size=None):
        self.name = name
        self.input_size = input_size
        self.layers = []

    def __eq__(self, other):
        if len(self.layers) != len(other.layers):
            return False
        for sl, ol in zip(self.layers, other.layers):
            if type(sl) != type(ol):
                return False
            if sl != ol:
                return False
        return True

    def predict(self, x):
        """ Predict the output from the input """

        for layer in self.layers:
            x = layer.predict(x)
        return x

    def calc_error(self, y, ytarget):
        """ Calculate the error between the prediction and the target """

        # Mean squared error, no weight decay
        return 0.5 * (y - ytarget)**2

    def add_layer(self, layer):
        """ Add a layer to the model """
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
