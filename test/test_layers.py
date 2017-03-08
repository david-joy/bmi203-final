""" Tests for the neural net layers """

# Imports
import numpy as np

import pytest

from final_project import layers


# Constants


LAYERS_FROM_DICT = [
    ({'type': 'FullyConnected', 'size': 8, 'func': 'sigmoid'},
     layers.FullyConnected(size=8, func='sigmoid')),
]


# Tests


@pytest.mark.parametrize('data,layer', LAYERS_FROM_DICT)
def test_from_dict(data, layer):
    res = layers.from_dict(data)
    assert res == layer



def test_fully_connected_forward_activations():
    layer = layers.FullyConnected(size=3, func='sigmoid')

    # Make a 3x2 weight matrix
    weight = np.array([
        [0.1, 0],
        [0, -0.1],
        [0.5, 0.5],
    ])
    # Bias 3x1
    bias = np.array([0.1, -0.2, -0.3])

    # Input 2x1
    x = np.array([0.4, -0.5])

    layer.set_weights(weight, bias)
    y = layer.forward(x)

    assert y.shape == (3, )

    exp_y = np.array([0.5349429, 0.4625702, 0.4133824])

    np.testing.assert_almost_equal(y, exp_y)
