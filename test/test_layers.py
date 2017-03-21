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


def test_fully_connected_activations():
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

    y = layer.predict(x)

    assert y.shape == (3, 1)

    exp_y = np.array([
        [0.5349429, 0.4625702, 0.4133824],
    ]).T

    np.testing.assert_almost_equal(y, exp_y)

    ytarget = np.array([0.4, 0.5, 0.6])

    # Propagate the y-error backwards
    delta1 = layer.calc_error(ytarget)
    exp_delta1 = np.array([
        [0.033571, -0.009305, -0.0452543],
    ]).T

    assert delta1.shape == (3, 1)
    np.testing.assert_almost_equal(delta1, exp_delta1)

    layer.update_weights(delta1, learn_rate=0.5)

    exp_weight = np.array([
        [0.0932858, 0.0083927],
        [0.001861, -0.1023263],
        [0.5090509, 0.4886864],
    ])
    exp_bias = np.array([
        [0.08321452],
        [-0.19534749],
        [-0.27737286],
    ])

    np.testing.assert_almost_equal(layer.weight, exp_weight)
    np.testing.assert_almost_equal(layer.bias, exp_bias)

    # Propagate the delta error backwards
    prev_delta = np.array([0.033571, -0.009305, -0.0452543])

    delta2 = layer.calc_delta(prev_delta)
    exp_delta2 = np.array([
        [-0.0047814, 0.0156609]
    ]).T

    assert delta2.shape == (2, 1)
    np.testing.assert_almost_equal(delta2, exp_delta2)


def test_fully_connected_activations_batch():
    layer = layers.FullyConnected(size=3, func='sigmoid')

    # Make a 3x2 weight matrix
    weight = np.array([
        [0.1, 0],
        [0, -0.1],
        [0.5, 0.5],
    ])
    # Bias 3x1
    bias = np.array([0.1, -0.2, -0.3])

    # Input 2x4
    x = np.array([
        [0.4, -0.5],
        [0.1, -0.5],
        [0.4, 0.5],
        [0.2, 0.2],
    ]).T

    layer.set_weights(weight, bias)

    y = layer.predict(x)

    assert y.shape == (3, 4)

    exp_y = np.array([
        [0.5349429, 0.5274723, 0.5349429, 0.5299641],
        [0.4625702, 0.4625702, 0.4378235, 0.4452208],
        [0.4133824, 0.3775407, 0.5374298, 0.4750208],
    ])
    np.testing.assert_almost_equal(y, exp_y)

    ytarget = np.array([
        [0.4, 0.5, 0.6],
        [0.4, 0.5, 0.6],
        [0.4, 0.5, 0.6],
        [0.4, 0.5, 0.6],
    ]).T

    delta_batch = layer.calc_error(ytarget)
    assert delta_batch.shape == (3, 4)

    for i in range(4):
        # Have to predict each vector to keep the cache consistent
        layer.predict(x[:, i])
        delta_r = layer.calc_error(ytarget[:, i])
        assert delta_r.shape == (3, 1)
        np.testing.assert_almost_equal(delta_batch[:, i], delta_r[:, 0])

    # Have to predict again so our cache isn't invalid
    layer.predict(x)
    layer.update_weights(delta_batch, learn_rate=0.5)

    # Note, these are different then four steps of learning alone
    exp_weight = np.array([
        [0.0954364, 0.0011764],
        [0.001685, -0.0998684],
        [0.5044731, 0.4956555],
    ])
    exp_bias = np.array([
        [0.083589],
        [-0.1940695],
        [-0.2819682],
    ])

    np.testing.assert_almost_equal(layer.weight, exp_weight)
    np.testing.assert_almost_equal(layer.bias, exp_bias)
