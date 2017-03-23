# Imports
import numpy as np

from final_project import alignment

# Tests


def test_recode_as_int():

    str1 = 'AAAA'
    res = alignment.recode_as_int(str1)
    exp = np.array([0, 0, 0, 0])

    np.testing.assert_equal(res, exp)

    str1 = 'GATTACA'
    res = alignment.recode_as_int(str1)
    exp = np.array([3, 0, 1, 1, 0, 2, 0])

    np.testing.assert_equal(res, exp)


def test_recode_as_one_hot():

    str1 = 'AAAA'
    res = alignment.recode_as_one_hot(str1)
    exp = np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ], dtype=np.int)

    np.testing.assert_equal(res, exp)

    str1 = 'GATTACA'
    res = alignment.recode_as_one_hot(str1)
    exp = np.array([
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
    ], dtype=np.int)

    np.testing.assert_equal(res, exp)


def test_score_alignments():

    str1 = 'AAAAA'
    str2 = 'GATTAAAAACA'

    res = alignment.score_alignments(str1, str2)
    exp = np.array([-2.22e-06,  -1.11e-06,  -1.11e-06, 0, 1, 0, 0])

    np.testing.assert_almost_equal(res, exp, decimal=7)

    str1 = 'AAAAA'
    str2 = 'AAAAA'

    res = alignment.score_alignments(str1, str2)
    exp = np.array([1.0])

    np.testing.assert_almost_equal(res, exp, decimal=7)

    str1 = 'AAAAA'
    str2 = 'GGGGGC'

    res = alignment.score_alignments(str1, str2)
    exp = np.array([1.0, 0.0])

    np.testing.assert_almost_equal(res, exp, decimal=7)


def test_amplify_positive_data():

    seqs = ['AA', 'T', 'AAA']

    res = alignment.amplify_positive_data(seqs)
    exp = [
        (1.0, 'AA'),
        (0.8, 'CA'),
        (0.8, 'GA'),
        (0.8, 'TA'),
        (0.8, 'AC'),
        (0.8, 'AG'),
        (0.8, 'AT'),
        (1.0, 'T'),
        (0.8, 'A'),
        (0.8, 'C'),
        (0.8, 'G'),
        (1.0, 'AAA'),
        (0.8, 'CAA'),
        (0.8, 'GAA'),
        (0.8, 'TAA'),
        (0.8, 'ACA'),
        (0.8, 'AGA'),
        (0.8, 'ATA'),
        (0.8, 'AAC'),
        (0.8, 'AAG'),
        (0.8, 'AAT'),
    ]
    assert res == exp


def test_deduplicate_data():

    scores = np.array([0.0, 0.5, 0.4, 0.6])
    seqs = ['AA', 'AT', 'AA', 'TA']

    res = alignment.deduplicate_data(scores, seqs)
    exp_scores = np.array([0.4, 0.5, 0.6])
    exp_seqs = ['AA', 'AT', 'TA']

    np.testing.assert_almost_equal(exp_scores, res[0])
    assert exp_seqs == res[1]
