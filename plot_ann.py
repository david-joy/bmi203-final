#!/usr/bin/env python

""" Make ROC plots for the ANN we trained """

import pathlib

import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

from final_project import model, io, alignment

# Constants
THISDIR = pathlib.Path(__file__).resolve().parent
WEIGHTDIR = THISDIR / 'weights'
MODELDIR = THISDIR / 'models'
DATADIR = THISDIR / 'data'
PLOTDIR = THISDIR / 'plots'

# Functions


def load_rand_data(positive_data, negative_file):
    """ Load a random sample of data

    :param positive_data:
        The numpy array of positive data, or a file containing positive
        examples
    :param negative_file:
        The negative examples as a FASTA file
    """
    if isinstance(positive_data, pathlib.Path):
        pos_recs = io.read_example_file(positive_data)
        kmer = None
        x_data = []
        y_data = []
        for rec in pos_recs:
            if kmer is None:
                kmer = len(rec)
            else:
                assert kmer == len(rec)
            x_data.append(alignment.recode_as_one_hot(rec).ravel())
            y_data.append(1.0)
    else:
        kmer = positive_data.shape[0] // 4
        x_data = [rec for rec in positive_data.T]
        y_data = [1.0 for _ in x_data]

    # Random sample negatives (with replacement)
    print('Sampling {} negatives'.format(len(x_data)))
    neg_recs = io.read_fasta_file(negative_file)
    neg_indicies = np.random.randint(0, len(neg_recs),
                                     size=(len(x_data), ))
    neg_recs = [neg_recs[i] for i in neg_indicies]

    for rec in neg_recs:
        # Random sample a 17-mer
        i = np.random.randint(0, len(rec) - kmer, size=(1, ))
        samp = rec[int(i):int(i)+kmer]
        x_data.append(alignment.recode_as_one_hot(samp).ravel())
        y_data.append(0.0)
    return np.array(x_data).T, np.array(y_data)[:, np.newaxis].T


def calc_roc(positive_scores, negative_scores):
    """ Calculate the reciever operating characteristic

    :param positive_scores:
        The vector of positive scores
    :param negative_scores:
        The vector of negative scores
    :returns:
        the false positive rate, the true positive rate
    """

    # Find the split points for each cutoff
    cutoff_min = np.min([positive_scores, negative_scores])
    cutoff_max = np.max([positive_scores, negative_scores])

    cutoffs = np.linspace(cutoff_min, cutoff_max, 200)

    # Using those cutoffs, calculate the empirical rates
    num_positive = positive_scores.shape[0]
    num_negative = negative_scores.shape[0]

    tp_rate = [1.0]
    fp_rate = [1.0]
    for cutoff in cutoffs:
        tp_rate.append(np.sum(positive_scores >= cutoff) / num_positive)
        fp_rate.append(np.sum(negative_scores >= cutoff) / num_negative)
    tp_rate.append(0.0)
    fp_rate.append(0.0)
    return np.array(fp_rate), np.array(tp_rate)


def plot_roc_curve(x_data, labels, net, plotfile,
                   title=''):
    """ Plot the ROC curve for the net's predictions

    :param x_data:
        The data to predict on
    :param labels:
        A mask of labels where 1 is a positive example and 0 a negative
    :param net:
        The net to predict the labels with
    :param plotfile:
        The filename to save the plot as
    """

    # Have the net predict, then split the scores by ground truth
    scores = net.predict(x_data)

    distfile = PLOTDIR / plotfile.replace('roc', 'dist')

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    df = pd.DataFrame({'Condition': ['Positive' if int(i) == 1 else 'Negative'
                                     for i in labels[0, :]],
                       'Score': scores[0, :]})
    sns.violinplot(x='Condition', y='Score', data=df, ax=ax)
    ax.set_title('{} Dist for Rap1 Identification'.format(title))

    fig.savefig(str(distfile))

    plt.close()

    fp_rate, tp_rate = calc_roc(scores[labels], scores[~labels])

    # Make the plot
    plotfile = PLOTDIR / plotfile

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    ax.plot(fp_rate, tp_rate, '-o', linewidth=3)

    # Plot the line for perfect confusion
    ax.plot([0, 1], [0, 1], '--', linewidth=3)

    ax.set_title('{} ROC for Rap1 Identification'.format(title))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    fig.savefig(str(plotfile))
    plt.close()


def main():

    modelfile = MODELDIR / 'ann_rap1.json'
    weightfile = WEIGHTDIR / 'ann_rap1.npz'

    if not modelfile.is_file():
        raise OSError('Train an ANN using `train_ann.py')

    if not weightfile.is_file():
        raise OSError('Train an ANN using `train_ann.py`')

    # Load the model we trained
    net = model.load_model(modelfile)
    net.load_weights(weightfile)

    # ROC curve for the training data
    x_train, y_train = io.load_training_data(DATADIR / 'train_final.txt')
    plot_roc_curve(x_train, y_train > 0.5, net, 'roc_train_final.png',
                   title='Training Data')

    # ROC curve for the test data
    x_test, y_test = io.load_training_data(DATADIR / 'test_final.txt')
    plot_roc_curve(x_test, y_test > 0.5, net, 'roc_test_final.png',
                   title='Test Data')

    # ROC curve for the original positive examples vs a random negtive set
    x_rand, y_rand = load_rand_data(DATADIR / 'rap1-lieb-positives.txt',
                                    DATADIR / 'yeast-upstream-1k-negative.fa')
    plot_roc_curve(x_rand, y_rand > 0.5, net, 'roc_rand_true.png',
                   title='Random True Sample')

    # ROC curve for our assumed positive examples vs a random negtive set
    x_train_pos = x_train[:, y_train[0, :] > 0.5]
    x_test_pos = x_test[:, y_test[0, :] > 0.5]
    print('Positive training: {}'.format(x_train_pos.shape))
    print('Positive test:     {}'.format(x_test_pos.shape))

    x_pos = np.concatenate([x_train_pos, x_test_pos], axis=1)
    x_rand, y_rand = load_rand_data(x_pos,
                                    DATADIR / 'yeast-upstream-1k-negative.fa')
    plot_roc_curve(x_rand, y_rand > 0.5, net, 'roc_rand_final.png',
                   title='Random Full Sample')

    # Test the two examples given in the HW
    pos_seq = 'ACATCCGTGCACCATTT'
    neg_seq = 'AAAAAAACGCAACTAAT'

    pos_enc = alignment.recode_as_one_hot(pos_seq).ravel()[:, np.newaxis]
    neg_enc = alignment.recode_as_one_hot(neg_seq).ravel()[:, np.newaxis]

    pos_score = net.predict(pos_enc)
    neg_score = net.predict(neg_enc)

    print('Positive example: {}'.format(pos_score))
    print('Negative example: {}'.format(neg_score))


if __name__ == '__main__':
    main()
