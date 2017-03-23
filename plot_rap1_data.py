#!/usr/bin/env python

""" Plot the normalized mututal information distribution """

import pathlib
import random

import numpy as np

from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt

from final_project import alignment

# Constants

THISDIR = pathlib.Path(__file__).resolve().parent
DATADIR = THISDIR / 'data' / 'seq_scores'
TRAIN_DATA = DATADIR.parent / 'train_final.txt'
TEST_DATA = DATADIR.parent / 'test_final.txt'
NEG_OUTLIER_DATA = DATADIR.parent / 'neg_outlier.txt'

PLOTDIR = THISDIR / 'plots'

NMI_THRESHOLD = 0.9  # Threshold for negative scores
POS_SNP_SCORE = 0.8  # Score for a single point mutant
HOLDOUT = 0.1  # How much data to set aside

# Functions


def load_all_scores(datadir):
    """ Load all the score files, splitting into positive and negative """

    neg_scores = []
    neg_seqs = []
    pos_scores = []
    pos_seqs = []

    for datafile in DATADIR.iterdir():
        if not datafile.suffix == '.txt':
            continue
        if not datafile.is_file():
            continue
        print(f'Loading {datafile}')
        with datafile.open('rt') as fp:
            for line in fp:
                line = line.split('#', 1)[0].strip()
                if line == '':
                    continue
                score, seq = line.split(',')
                score = float(score)

                if 'negs-' in datafile.name:
                    neg_scores.append(score)
                    neg_seqs.append(seq)
                elif 'pos-' in datafile.name:
                    pos_scores.append(score)
                    pos_seqs.append(seq)
                else:
                    raise ValueError(f'Unknown score type: {datafile}')
    neg_scores = np.array(neg_scores)
    pos_scores = np.array(pos_scores)

    return neg_scores, neg_seqs, pos_scores, pos_seqs


def select_negs(neg_scores, neg_seqs, num_records):
    """ Select negative examples """
    neg_probs = neg_scores / np.sum(neg_scores)
    selections = np.random.choice(neg_seqs,
                                  size=num_records,
                                  p=neg_probs)
    return [(0.0, s) for s in selections]


def main():

    neg_scores, neg_seqs, pos_scores, pos_seqs = load_all_scores(DATADIR)

    print(f'Got {len(neg_scores)} negative scores')
    print(f'Got {len(pos_scores)} positive scores')

    neg_dist = gaussian_kde(neg_scores)
    pos_dist = gaussian_kde(pos_scores)

    x = np.linspace(0, 1, 500)
    neg_y = neg_dist(x)
    pos_y = pos_dist(x)

    plotfile = PLOTDIR / 'align_score_dist.png'

    print(f'Saving plot to {plotfile}...')

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].plot(x, neg_y, '-b', linewidth=3, label='Negative')
    axes[0].plot(neg_scores, np.zeros_like(neg_scores), 'ob', markersize=6)
    axes[0].plot([NMI_THRESHOLD, NMI_THRESHOLD], [0, np.max(neg_y)], '--b')

    axes[1].plot(x, pos_y, '-r', linewidth=3, label='Positive')
    axes[1].plot(pos_scores, np.zeros_like(pos_scores), 'or', markersize=6)

    axes[0].set_title('Negative Alignment Scores')
    axes[0].set_xlabel('Normalized Mutual Information')
    axes[0].set_ylabel('Score Frequency')

    axes[1].set_title('Positive Alignment Scores')
    axes[1].set_xlabel('Normalized Mutual Information')
    axes[1].set_ylabel('Score Frequency')

    fig.savefig(str(plotfile))

    plt.close()

    # Deduplicate the data
    neg_scores, neg_seqs = alignment.deduplicate_data(
        neg_scores, neg_seqs)
    pos_scores, pos_seqs = alignment.deduplicate_data(
        pos_scores, pos_seqs)

    print(f'Got {len(neg_seqs)} unique negative scores')
    print(f'Got {len(pos_seqs)} unique positive scores')

    # Split the negative examples at the threshold
    neg_indicies = np.arange(neg_scores.shape[0])
    neg_mask = neg_scores >= NMI_THRESHOLD
    neg_mask_indicies = neg_indicies[neg_mask]
    neg_rem_indicies = neg_indicies[~neg_mask]

    neg_scores_outliers = neg_scores[neg_mask]
    neg_seqs_outliers = [neg_seqs[i] for i in neg_mask_indicies]

    neg_seqs_rem = [neg_seqs[i] for i in neg_rem_indicies]
    neg_scores_rem = neg_scores[~neg_mask]

    assert len(neg_seqs_rem) == neg_scores_rem.shape[0]

    print(f'Thresholded Negatives: {len(neg_seqs_outliers)}')

    with NEG_OUTLIER_DATA.open('wt') as fp:
        fp.write('#score,seq\n')
        for score, seq in zip(neg_scores_outliers, neg_seqs_outliers):
            fp.write(f'{score:f},{seq}\n')

    pos_final = alignment.amplify_positive_data(pos_seqs)
    print(f'{len(pos_final)} Positive examples after amplification')

    neg_final = select_negs(neg_scores_rem, neg_seqs_rem,
                            num_records=len(pos_final))
    print(f'{len(neg_final)} Negative examples after filtering')

    assert len(pos_final) == len(neg_final)

    random.shuffle(pos_final)
    random.shuffle(neg_final)

    # And split into training/testing data
    num_test = round(len(pos_final) * HOLDOUT)

    pos_test = pos_final[:num_test]
    pos_train = pos_final[num_test:]

    neg_test = neg_final[:num_test]
    neg_train = neg_final[num_test:]

    with TRAIN_DATA.open('wt') as fp:
        fp.write('#score,seq\n')
        for score, seq in pos_train:
            fp.write(f'{score:f},{seq}\n')
        for score, seq in neg_train:
            fp.write(f'{score:f},{seq}\n')

    with TEST_DATA.open('wt') as fp:
        fp.write('#score,seq\n')
        for score, seq in pos_test:
            fp.write(f'{score:f},{seq}\n')
        for score, seq in neg_test:
            fp.write(f'{score:f},{seq}\n')


if __name__ == '__main__':
    main()
