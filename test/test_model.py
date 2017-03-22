""" Tests for the model builder """

# Imports
import pathlib
import tempfile

import numpy as np

from final_project import model, layers, optimizers

# Tests


def test_create_read_write_model():
    # Make sure we can read and write the model spec

    autoenc = model.Model('3-8-3 Autoencoder', input_size=8)
    autoenc.add_layer(layers.FullyConnected(size=3, func='sigmoid'))
    autoenc.add_layer(layers.FullyConnected(size=8, func='sigmoid'))

    with tempfile.TemporaryDirectory() as tempdir:
        modelfile = pathlib.Path(tempdir) / 'model.json'

        autoenc.save_model(modelfile)

        assert modelfile.is_file()

        res_autoenc = model.Model.load_model(modelfile)

        assert res_autoenc == autoenc


def test_can_read_write_weights():

    autoenc = model.Model('3-8-3 Autoencoder', input_size=8)
    autoenc.add_layer(layers.FullyConnected(size=3, func='sigmoid'))
    autoenc.add_layer(layers.FullyConnected(size=8, func='sigmoid'))
    autoenc.init_weights()
    autoenc.set_optimizer(optimizers.Adam())

    x = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    autoenc.gradient_descent(x, x)

    with tempfile.TemporaryDirectory() as tempdir:
        weightfile = pathlib.Path(tempdir) / 'weights.npz'

        autoenc.save_weights(weightfile)

        assert weightfile.is_file()

        res_autoenc = model.Model('3-8-3 Autoencoder', input_size=8)
        res_autoenc.add_layer(layers.FullyConnected(size=3, func='sigmoid'))
        res_autoenc.add_layer(layers.FullyConnected(size=8, func='sigmoid'))

        res_autoenc.load_weights(weightfile)

        for exp_layer, res_layer in zip(autoenc.layers, res_autoenc.layers):
            np.testing.assert_almost_equal(exp_layer.weight, res_layer.weight)
            np.testing.assert_almost_equal(exp_layer.bias, res_layer.bias)


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
    net.set_optimizer(optimizers.SGD(learn_rate=0.5,
                                     weight_decay=0.0))

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

    # Now do gradient descent
    err = net.gradient_descent(x, ytarget)

    assert np.round(err, 4) == 0.5967

    exp_weight = np.array([
        [0.3589165, 0.4086662],
        [0.5113013, 0.5613701],
    ])
    exp_bias = np.array([[0.53075072], [0.61904912]])

    np.testing.assert_almost_equal(exp_weight, layer2.weight)
    np.testing.assert_almost_equal(exp_bias, layer2.bias)

    exp_weight = np.array([
        [0.1497807, 0.1995614],
        [0.2497511, 0.2995023],
    ])
    exp_bias = np.array([[0.3456143], [0.3450229]])

    np.testing.assert_almost_equal(exp_weight, layer1.weight)
    np.testing.assert_almost_equal(exp_bias, layer1.bias)


def test_batch_prediction_gradient_descent():

    # Hardwire weights so we can get an answer back
    w1 = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [-0.1, 0.2, -0.3, 0.4],
        [-0.1, -0.2, 0.3, 0.4],
    ])
    b1 = np.array([0.1, -0.1, 0.1])
    l1 = layers.FullyConnected(size=3, func='sigmoid')
    l1.set_weights(w1, b1)

    w2 = np.array([
        [0.15, 0.25, -0.3],
        [-0.15, 0.25, 0.3],
    ])
    b2 = np.array([0.15, -0.15])
    l2 = layers.FullyConnected(size=2, func='sigmoid')
    l2.set_weights(w2, b2)

    net = model.Model('4-3-2 Net', input_size=4)
    net.add_layer(l1)
    net.add_layer(l2)
    net.set_optimizer(optimizers.SGD(learn_rate=0.5,
                                     weight_decay=0.0))

    assert net.layers[0].weight.shape == (3, 4)
    assert net.layers[1].weight.shape == (2, 3)

    x = np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]).T
    y = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0],
    ]).T

    # Make sure we can predict a batch
    y_batch = net.predict(x)
    assert y_batch.shape == (2, 7)
    for i, xr in enumerate(x.T):
        yr = net.predict(xr)
        assert yr.shape == (2, 1)
        np.testing.assert_almost_equal(y_batch[:, i], yr[:, 0])

    # Make sure we can learn with a batch
    y_loss = net.gradient_descent(x, y)
    assert y_loss.shape == (1, 7)
