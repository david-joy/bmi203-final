""" Tests for the model builder """

# Imports
import pathlib
import tempfile

import numpy as np

from final_project import model, layers

# Tests


def test_create_read_write_model():
    # Make sure we can read and write the model spec

    autoenc = model.Model('3-8-3 Autoencoder', input_size=8)
    autoenc.add_layer(layers.FullyConnected(size=3, func='sigmoid'))
    autoenc.add_layer(layers.FullyConnected(size=8, func='sigmoid'))

    with tempfile.TemporaryDirectory() as tempdir:
        modelfile = pathlib.Path(tempdir) / 'model.json'

        autoenc.save_model(modelfile)
        res_autoenc = model.Model.load_model(modelfile)

        assert res_autoenc == autoenc


def test_one_step_forward_backward():
    # Using values from
    # https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

    net = model.Model('Demo Net', input_size=2)

    # hidden layer
    layer1 = layers.FullyConnected(size=2, func='sigmoid')
    weight = np.array([
        [0.15, 0.2],
        [0.25, 0.3],
    ])
    bias = np.array([0.35, 0.35])
    layer1.set_weights(weight, bias)

    # output layer
    layer2 = layers.FullyConnected(size=2, func='sigmoid')
    weight = np.array([
        [0.4, 0.45],
        [0.5, 0.55],
    ])
    bias = np.array([0.6, 0.6])
    layer2.set_weights(weight, bias)

    # Build out the whole model
    net.add_layer(layer1)
    net.add_layer(layer2)

    ytarget = np.array([0.01, 0.99])

    # Input 2x1
    x = np.array([0.05, 0.1])

    y1 = layer1.predict(x)

    assert y1.shape == (2, 1)

    exp_y1 = np.array([[0.59327], [0.5968844]])

    np.testing.assert_almost_equal(y1, exp_y1)

    y2 = layer2.predict(y1)

    exp_y2 = np.array([[0.75136507], [0.772928465]])

    np.testing.assert_almost_equal(y2, exp_y2)

    yhat = net.predict(x)

    np.testing.assert_almost_equal(yhat, exp_y2)

    # Now do stepwise backprop
    delta2 = layer2.calc_error(ytarget)
    layer2.update_weights(delta2, learn_rate=0.5)

    exp_weight = np.array([
        [0.3626921, 0.4126921],
        [0.4626921, 0.5126921],
    ])

    exp_bias = np.array([[0.53075072], [0.61904912]])

    np.testing.assert_almost_equal(exp_weight, layer2.weight)
    np.testing.assert_almost_equal(exp_bias, layer2.bias)

    delta1 = layer1.calc_delta(delta2)
    layer1.update_weights(delta1, learn_rate=0.5)

    exp_weight = np.array([
        [0.1480264, 0.1980264],
        [0.2480264, 0.2980264],
    ])
    exp_bias = np.array([[0.3486427], [0.3480426]])

    np.testing.assert_almost_equal(exp_weight, layer1.weight)
    np.testing.assert_almost_equal(exp_bias, layer1.bias)
