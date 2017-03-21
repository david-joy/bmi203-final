#!/usr/bin/env python

""" Train an 8-3-8 autoencoder """

import secrets
import pathlib
import argparse

import matplotlib.pyplot as plt

import numpy as np

from final_project import model, layers

# Constants
THISDIR = pathlib.Path(__file__).resolve().parent
WEIGHTDIR = THISDIR / 'weights'
PLOTDIR = THISDIR / 'plots'

LEARN_RATE = 0.11
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 10000
NOISE_MAGNITUDE = 0.0


# Main Function


def train_autoenc(activation,
                  learn_rate=LEARN_RATE,
                  weight_decay=WEIGHT_DECAY,
                  num_epochs=NUM_EPOCHS,
                  noise_magnitude=NOISE_MAGNITUDE,
                  seed=None):
    """ Train an 8-3-8 autoencoder """

    # Training data
    train_x = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]

    if seed is None:
        seed = secrets.randbelow(2**32)
    print('Random seed: {}'.format(seed))
    rng = np.random.RandomState(seed)

    # Make a big batch to train on
    x_out = np.array(train_x * 1000).T.astype(np.float64)
    print('Batch shape: {}'.format(x_out.shape))

    # Add some noise
    x_noise = rng.rand(*x_out.shape)
    x_noise = (x_noise - 0.5) * noise_magnitude
    x_in = x_out + x_noise

    # Model
    autoenc = model.Model('3-8-3 Autoencoder',
                          input_size=8,
                          rng=rng)
    autoenc.add_layer(layers.FullyConnected(size=3, func=activation))
    autoenc.add_layer(layers.FullyConnected(size=8, func=activation))
    autoenc.init_weights()

    losses = []
    alpha = learn_rate
    for epoch in range(num_epochs):
        err = autoenc.gradient_descent(x_in, x_out,
                                       learn_rate=alpha,
                                       weight_decay=weight_decay)
        err = np.mean(err)
        losses.append(err)

        if epoch % 100 == 0:
            print('Epoch: {:04d}: Last error: {}'.format(epoch, losses[-1]))

    print('Saving final weights...')
    weightfile = 'autoencoder_838_{}.npz'.format(activation)
    autoenc.save_weights(str(WEIGHTDIR / weightfile))

    losses = np.array(losses)

    print('Plotting losses for {}...'.format(activation))
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    title = 'Loss for 8-3-8 {} Autoencoder ($\\alpha$={}) ($\\lambda={}$)'
    title = title.format(activation, learn_rate, weight_decay)

    ax.plot(np.arange(losses.shape[0]), losses, linewidth=3)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(title)

    plotfile = 'autoencoder_838_{}_loss.png'.format(activation)
    fig.savefig(str(PLOTDIR / plotfile))
    plt.close()

    print('Plotting layerwise encoding for {}...'.format(activation))

    fig, axes = plt.subplots(1, 4, figsize=(16, 8))

    title = 'Encoding for 8-3-8 {} Autoencoder ($\\alpha$={}) ($\\lambda={}$)'
    title = title.format(activation, learn_rate, weight_decay)

    input_layer = []
    code_layer = []
    output_layer = []
    expected_layer = []

    for i in range(8):
        y = autoenc.predict(x_in[:, i])

        input_layer.append(np.squeeze(x_in[:, i]).copy())
        output_layer.append(np.squeeze(y).copy())
        code_layer.append(np.squeeze(autoenc.layers[0].a).copy())
        expected_layer.append(np.squeeze(x_out[:, i].copy()))

    input_layer = np.stack(input_layer, axis=0)
    code_layer = np.stack(code_layer, axis=0)
    output_layer = np.stack(output_layer, axis=0)
    expected_layer = np.stack(expected_layer, axis=0)

    assert input_layer.shape == (8, 8)
    assert code_layer.shape == (8, 3)
    assert output_layer.shape == (8, 8)
    assert expected_layer.shape == (8, 8)

    axes[0].imshow(input_layer, cmap='gray')
    axes[0].set_title('Input One-Hot Encoding')
    axes[0].set_xticks([])

    axes[1].imshow(code_layer, cmap='gray')
    axes[1].set_title('Hidden Encoding')
    axes[1].set_xticks([])

    axes[2].imshow(output_layer, cmap='gray')
    axes[2].set_title('Output One-Hot Encoding')
    axes[2].set_xticks([])

    axes[3].imshow(expected_layer, cmap='gray')
    axes[3].set_title('Expected One-Hot Encoding')
    axes[3].set_xticks([])

    fig.suptitle(title)

    plotfile = 'autoencoder_838_{}_code.png'.format(activation)
    fig.savefig(str(PLOTDIR / plotfile))
    plt.close()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--learn-rate', type=float, default=LEARN_RATE,
                        help='Learning rate to use')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY,
                        help='Weight decay rate to use')
    parser.add_argument('-n', '--num-epochs', type=int, default=NUM_EPOCHS,
                        help='Number of epochs to use')
    parser.add_argument('--seed', type=int,
                        help='Random seed to use for training')
    parser.add_argument('--noise-magnitude', type=float,
                        default=NOISE_MAGNITUDE,
                        help='Noise to add as a fraction of signal')
    parser.add_argument('activation', choices=('relu', 'sigmoid'),
                        help='Activation function')
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)
    train_autoenc(**vars(args))


if __name__ == '__main__':
    main()
