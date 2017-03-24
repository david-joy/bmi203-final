#!/usr/bin/env python

""" Train the Rap1 site detector """

# Imports
import time
import secrets
import pathlib
import argparse

import matplotlib.pyplot as plt

import numpy as np

from final_project import model, layers, optimizers, alignment

# Constants
THISDIR = pathlib.Path(__file__).resolve().parent
WEIGHTDIR = THISDIR / 'weights'
PLOTDIR = THISDIR / 'plots'
DATADIR = THISDIR / 'data'

LEARN_RATE = 0.001
NUM_EPOCHS = 2500

# Functions


def load_training_data(datafile):
    """ Load the training data

    Convert the sequence to a one-hot-encoding

    :param datafile:
        A CSV file of score,sequence records
    :returns:
        one-hot-encoded sequences, scores
    """

    scores = []
    seqs = []

    with datafile.open('rt') as fp:
        for line in fp:
            line = line.split('#', 1)[0].strip()
            if line == '':
                continue
            score, seq = line.split(',')
            scores.append(float(score))
            seqs.append(alignment.recode_as_one_hot(seq).ravel())

    scores = np.array(scores)[:, np.newaxis].T
    return np.array(seqs).T, scores


def train_rap1_det(learn_rate=LEARN_RATE,
                   num_epochs=NUM_EPOCHS,
                   seed=None):
    """ Train a Rap1 site detector """

    # Training data
    train_x, train_y = load_training_data(DATADIR / 'train_final.txt')
    test_x, test_y = load_training_data(DATADIR / 'test_final.txt')

    print('X training data: {}'.format(train_x.shape))
    print('Y training data: {}'.format(train_y.shape))

    print('X test data: {}'.format(test_x.shape))
    print('Y test data: {}'.format(test_y.shape))

    if seed is None:
        seed = secrets.randbelow(2**32)
    print('Random seed: {}'.format(seed))
    rng = np.random.RandomState(seed)

    # Model
    net = model.Model('Rap1 Detector',
                      input_size=68,
                      rng=rng)
    net.add_layer(layers.FullyConnected(size=16, func='relu'))
    net.add_layer(layers.FullyConnected(size=8, func='relu'))
    net.add_layer(layers.FullyConnected(size=1, func='sigmoid'))
    net.init_weights()
    net.set_optimizer(optimizers.Adam(learn_rate=learn_rate))

    t0 = time.time()
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        err = net.gradient_descent(train_x, train_y)
        err = np.mean(err)
        train_losses.append(err)

        test_yhat = net.predict(test_x)
        err = np.mean(net.calc_error(test_y, test_yhat))
        test_losses.append(err)

        if epoch % 100 == 0:
            print('Epoch: {:4d}: Last train error: {}'.format(
                epoch, train_losses[-1]))
            print('           : Last test error:  {}'.format(
                test_losses[-1]))

    print('Training took {} secs'.format(time.time() - t0))

    print('Saving final weights...')
    net.save_weights(str(WEIGHTDIR / 'ann_rap1.npz'))

    train_losses = np.array(train_losses)

    print('Plotting losses...')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    title = 'Loss for Rap1 Detector ($\\alpha$={})'
    title = title.format(learn_rate)

    epochs = np.arange(train_losses.shape[0])

    ax.plot(epochs, train_losses, '-b', linewidth=3, label='training')
    ax.plot(epochs, test_losses, '-r', linewidth=3, label='testing')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(title)
    ax.legend()

    fig.savefig(str(PLOTDIR / 'ann_rap1_loss.png'))
    plt.close()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--learn-rate', type=float, default=LEARN_RATE,
                        help='Learning rate to use')
    parser.add_argument('-n', '--num-epochs', type=int, default=NUM_EPOCHS,
                        help='Number of epochs to use')
    parser.add_argument('--seed', type=int,
                        help='Random seed to use for training')
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)
    train_rap1_det(**vars(args))


if __name__ == '__main__':
    main()
