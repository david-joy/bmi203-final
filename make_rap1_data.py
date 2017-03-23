#!/usr/bin/env python

""" Generate the training data for Rap1 sites """

# Imports

# Standard lib
import pathlib
import multiprocessing

# 3rd party
import numpy as np

# Our own imports
from final_project import io, alignment

# Constants

THISDIR = pathlib.Path(__file__).resolve().parent
DATADIR = THISDIR / 'data'

NEGATIVE_FA_FILE = DATADIR / 'yeast-upstream-1k-negative.fa'
POSITIVE_FILE = DATADIR / 'rap1-lieb-positives.txt'
NEGATIVE_CANDIDATE_DIR = DATADIR / 'negative_seqs'

PROCESSES = 8

TOPK = 5


def find_top_alignments(pos_rec, neg_rec):
    """ Find the top alignments """

    score = alignment.score_alignments(pos_rec, neg_rec)
    top_locs = np.argpartition(-score, TOPK)[:TOPK]

    window = len(pos_rec)
    indicies = np.arange(len(neg_rec) - window + 1)

    res = []
    for s, i in zip(score[top_locs], indicies[top_locs]):
        res.append((s, neg_rec[i:i+window]))
    return res


def maybe_find_top_alignments(item):
    pos_rec, neg_rec = item
    try:
        return find_top_alignments(pos_rec, neg_rec)
    except Exception:
        print('Error aligning things')
        return []


def find_all_candidates(positive_recs, neg_rec, neg_candidate_file):

    items = [(p, neg_rec) for p in positive_recs]

    print(f'Finding negative candidates: {neg_candidate_file}')

    with neg_candidate_file.open('wt') as fp:
        fp.write('#score,sequence\n')
        with multiprocessing.Pool(PROCESSES) as pool:
            results = pool.imap(maybe_find_top_alignments, items)
            for i, res in enumerate(results):
                print('Writing result for {}'.format(i))
                for score, line in res:
                    if score is None:
                        fp.write(f',{line}\n')
                    else:
                        fp.write(f'{score},{line}\n')


def main():
    positive_recs = io.read_example_file(POSITIVE_FILE)
    negative_recs = io.read_fasta_file(NEGATIVE_FA_FILE)

    NEGATIVE_CANDIDATE_DIR.mkdir(exist_ok=True, parents=True)

    for i, neg_rec in enumerate(negative_recs):
        neg_candidate_file = f'rap1-candidate-negs-{i:04d}.txt'
        neg_candidate_file = NEGATIVE_CANDIDATE_DIR / neg_candidate_file

        if neg_candidate_file.is_file():
            print(f'Already found all negatives for {neg_candidate_file}')
            continue

        try:
            find_all_candidates(positive_recs, neg_rec, neg_candidate_file)
        except Exception:
            if neg_candidate_file.is_file():
                neg_candidate_file.unlink()
            raise


if __name__ == '__main__':
    main()
