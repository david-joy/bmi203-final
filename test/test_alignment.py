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
