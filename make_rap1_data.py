#!/usr/bin/env python

""" Generate the training data for Rap1 sites """

# Imports

# Standard lib
import argparse
import time
import pathlib
import multiprocessing
import traceback

# 3rd party
import numpy as np

# Our own imports
from final_project import io, alignment

# Constants

THISDIR = pathlib.Path(__file__).resolve().parent
DATADIR = THISDIR / 'data'

NEGATIVE_FA_FILE = DATADIR / 'yeast-upstream-1k-negative.fa'
POSITIVE_FILE = DATADIR / 'rap1-lieb-positives.txt'
CANDIDATE_DIR = DATADIR / 'seq_scores'

PROCESSES = 1

TOPK = 5


def find_top_alignments(pos_rec, test_rec):
    """ Find the top alignments """

    score = alignment.score_alignments(pos_rec, test_rec)
    topk = min([TOPK, score.shape[0]])
    if topk > 1:
        top_locs = np.argpartition(-score, topk)[:topk]
    elif topk == 1:
        top_locs = np.array([0])
    else:
        top_locs = np.array([])

    window = len(pos_rec)
    indicies = np.arange(len(test_rec) - window + 1)

    res = []
    for s, i in zip(score[top_locs], indicies[top_locs]):
        res.append((s, test_rec[i:i+window]))
    return res


def maybe_find_top_alignments(item):
    pos_rec, test_rec = item
    try:
        return find_top_alignments(pos_rec, test_rec)
    except Exception:
        print('Error aligning things')
        traceback.print_exc()
        return []


def find_all_candidates(positive_recs, test_rec, candidate_file,
                        processes=PROCESSES):

    items = [(p, test_rec) for p in positive_recs]

    print(f'Finding candidates: {candidate_file}')

    with candidate_file.open('wt') as fp:
        fp.write('#score,sequence\n')
        with multiprocessing.Pool(processes) as pool:
            results = pool.imap(maybe_find_top_alignments, items)
            for i, res in enumerate(results):
                print('Writing result for {}'.format(i))
                for score, line in res:
                    if score is None:
                        fp.write(f',{line}\n')
                    else:
                        fp.write(f'{score},{line}\n')


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, default=PROCESSES)
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)

    positive_recs = io.read_example_file(POSITIVE_FILE)
    negative_recs = io.read_fasta_file(NEGATIVE_FA_FILE)

    CANDIDATE_DIR.mkdir(exist_ok=True, parents=True)

    # Pairwise align positive and negative recs
    for i, neg_rec in enumerate(negative_recs):
        neg_candidate_file = f'rap1-candidate-negs-{i:04d}.txt'
        neg_candidate_file = CANDIDATE_DIR / neg_candidate_file

        if neg_candidate_file.is_file():
            print(f'Already found all negatives for {neg_candidate_file}')
            continue
        t0 = time.time()
        try:
            find_all_candidates(positive_recs, neg_rec, neg_candidate_file,
                                processes=args.processes)
        except Exception:
            if neg_candidate_file.is_file():
                neg_candidate_file.unlink()
            raise
        print('Finished in {} secs'.format(time.time() - t0))

    # Pairwise align positive and positive recs
    for i, pos_rec in enumerate(positive_recs):
        pos_candidate_file = f'rap1-candidate-pos-{i:04d}.txt'
        pos_candidate_file = CANDIDATE_DIR / pos_candidate_file

        if pos_candidate_file.is_file():
            print(f'Already found all positives for {pos_candidate_file}')
            continue
        t0 = time.time()
        try:
            find_all_candidates(positive_recs, pos_rec, pos_candidate_file,
                                processes=args.processes)
        except Exception:
            if pos_candidate_file.is_file():
                pos_candidate_file.unlink()
            raise
        print('Finished in {} secs'.format(time.time() - t0))


if __name__ == '__main__':
    main()
