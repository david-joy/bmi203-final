#!/usr/bin/env python

""" Predict with the Rap1 site detector """

# Imports
import pathlib
import argparse

import numpy as np

from final_project import model, io, alignment

# Constants
THISDIR = pathlib.Path(__file__).resolve().parent
WEIGHTDIR = THISDIR / 'weights'
MODELDIR = THISDIR / 'models'
DATADIR = THISDIR / 'data'

# Functions


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                        help='Threshold to call positive vs negative')
    parser.add_argument('infile', help='Sequences to predict on')
    parser.add_argument('outfile', help='Predictions on the sequence')
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)

    modelfile = MODELDIR / 'ann_rap1.json'
    weightfile = WEIGHTDIR / 'ann_rap1.npz'

    if not modelfile.is_file():
        raise OSError('Train an ANN using `train_ann.py')

    if not weightfile.is_file():
        raise OSError('Train an ANN using `train_ann.py`')

    # Load the model we trained
    net = model.load_model(modelfile)
    net.load_weights(weightfile)

    # Load the test data and encode as one-hot
    recs = io.read_example_file(args.infile)
    x_data = []
    for rec in recs:
        x_data.append(alignment.recode_as_one_hot(rec).ravel())
    x_data = np.array(x_data).T
    print('Got {} samples'.format(x_data.shape[1]))

    # Batch predict on the samples
    y_data = np.squeeze(net.predict(x_data))

    y_pos = y_data > args.threshold
    print('Predicting {} Positives'.format(np.sum(y_pos)))
    print('Predicting {} Negatives'.format(np.sum(~y_pos)))

    # Save the files
    assert len(recs) == y_data.shape[0]
    io.write_score_file(args.outfile, zip(recs, y_data))


if __name__ == '__main__':
    main()
