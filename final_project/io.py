""" Tools for reading and writing files """

# Imports
import pathlib

import numpy as np

from . import alignment

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


def read_fasta_file(filepath):
    """ Read in a fasta file

    Doesn't return the header because we don't need it

    :param filepath:
        The path to the fasta file
    :returns:
        A list of sequences, one for each record
    """

    filepath = pathlib.Path(filepath)

    records = []
    with filepath.open('rt') as fp:
        cur_record = []
        for line in fp:
            line = line.strip()
            if line == '':
                continue
            if line.startswith('>'):
                # Header
                if cur_record != []:
                    records.append(''.join(cur_record))
                    cur_record = []
            else:
                cur_record.append(line.upper())
    if cur_record != []:
        records.append(''.join(cur_record))
    return records


def read_example_file(filepath):
    """ Read in the positive and test example files

    :param filepath:
        The path to the example file
    :returns:
        A list of examples, one for each record
    """
    filepath = pathlib.Path(filepath)

    records = []
    with filepath.open('rt') as fp:
        for line in fp:
            line = line.strip()
            if line == '':
                continue
            records.append(line.upper())
    return records


def read_score_file(filepath):
    """ Read in a score file

    :param filepath:
        The path to the score file
    :returns:
        A tuple of (sequence, score) or (sequence, score, subsequence)
    """
    filepath = pathlib.Path(filepath)

    records = []
    with filepath.open('rt') as fp:
        for line in fp:
            line = line.strip()
            if line == '':
                continue
            rec = line.split('\t')
            if len(rec) == 2:
                records.append((rec[0], float(rec[1])))
            elif len(rec) == 3:
                records.append((rec[0], float(rec[1]), rec[2]))
            else:
                raise ValueError('Malformed line: {}'.format(line))
    return records


def write_score_file(filepath, records):
    """ Write out the score file

    Score files are tab separated files::

        sequence score (subsequence)

    Where ``sequence`` is the original sequence, ``score`` is the probability
    that the algorithm assigns to this sequence being a binding site, and
    ``subsequence``, if present, is the subsequence the algorithm used to make
    that descision (otherwise, using the whole sequence is assumed).

    :param filepath:
        The path to the score file to write
    :param records:
        A list of (sequence, score) or (sequence, score, subsequence) tuples
    """
    filepath = pathlib.Path(filepath)

    with filepath.open('wt') as fp:
        for rec in records:
            if len(rec) == 2:
                fp.write('{}\t{:0.4f}\n'.format(*rec))
            elif len(rec) == 3:
                fp.write('{}\t{:0.4f}\t{}\n'.format(*rec))
            else:
                raise ValueError('Malformed record: {}'.format(rec))
