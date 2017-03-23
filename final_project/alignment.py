from collections import OrderedDict

import numpy as np

from ._alignment import (
    recode_as_int, recode_as_one_hot, score_alignments
)
__all__ = [
    'recode_as_int', 'recode_as_one_hot', 'score_alignments'
]

# Constants

SNPS = {
    'A': ['C', 'G', 'T'],
    'C': ['A', 'G', 'T'],
    'G': ['A', 'C', 'T'],
    'T': ['A', 'C', 'G'],
}

# Functions


def amplify_positive_data(seqs, snp_score=0.8):
    """ amplify the positive sequences """

    out_seqs = []
    for seq in seqs:
        out_seqs.append((1.0, seq))
        for i, c in enumerate(seq):
            pre = seq[:i]
            post = seq[i+1:]

            for sub in SNPS[c]:
                out_seqs.append((snp_score, pre+sub+post))
    return out_seqs


def deduplicate_data(scores, seqs):
    """ Deduplicate the sequences """
    final_seqs = OrderedDict()

    assert scores.shape[0] == len(seqs)
    for score, seq in zip(scores, seqs):
        final_seqs.setdefault(seq, []).append(score)

    out_seqs = [seq for seq in final_seqs.keys()]
    out_scores = [max(score) for score in final_seqs.values()]
    return np.array(out_scores), out_seqs
