import numpy as np
cimport numpy as np

cimport cython

from sklearn.metrics import normalized_mutual_info_score


cdef dict INT_CODE = {
    'A': 0,
    'T': 1,
    'C': 2,
    'G': 3,
}

cdef dict ONE_HOT_CODE = {
    'A': np.array([1, 0, 0, 0], dtype=np.int),
    'T': np.array([0, 1, 0, 0], dtype=np.int),
    'C': np.array([0, 0, 1, 0], dtype=np.int),
    'G': np.array([0, 0, 0, 1], dtype=np.int),
}

# Encode functions


def recode_as_int(str rec):
    """ Recode the record as an int numpy array

    :param rec:
        The string to recode
    :returns:
        The string coded as an integer numpy array
    """

    cdef np.ndarray[dtype=np.int_t, ndim=1] out_rec
    cdef str c
    cdef int i
    out_rec = np.zeros((len(rec), ), dtype=np.int)

    for i, c in enumerate(rec):
        out_rec[i] = INT_CODE[c]
    return out_rec


def recode_as_one_hot(str rec):
    """ Recode the record as a one hot numpy array

    :param rec:
        The string to recode
    :returns:
        The string coded as a one-hot numpy array
    """

    cdef np.ndarray[dtype=np.int_t, ndim=2] out_rec
    cdef str c
    cdef int i
    out_rec = np.zeros((len(rec), 4), dtype=np.int)

    for i, c in enumerate(rec):
        out_rec[i, :] = ONE_HOT_CODE[c]
    return out_rec


# Brute force alignment functions


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
def score_alignments(str rec1, str rec2):
    """ Score the alignment between rec1 and rec2 """

    cdef np.ndarray[dtype=np.int_t, ndim=1] irec1, irec2
    cdef np.ndarray[dtype=np.float_t, ndim=1] score
    cdef int window, extent, i

    irec1 = recode_as_int(rec1)
    irec2 = recode_as_int(rec2)

    window = irec1.shape[0]
    extent = irec2.shape[0]

    score = np.empty((extent - window + 1), dtype=np.float)

    for i in range(0, extent - window + 1):
        score[i] = normalized_mutual_info_score(irec1, irec2[i:i+window])
    return score
