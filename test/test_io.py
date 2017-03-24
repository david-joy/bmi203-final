""" Tests for the file IO tools """

# Imports
import pathlib
import tempfile

import pytest

from final_project import io

# Constants
THISDIR = pathlib.Path(__file__).resolve().parent
DATADIR = THISDIR.parent / 'data'
NEGATIVE_FA_FILE = DATADIR / 'yeast-upstream-1k-negative.fa'

# Filepath, number of records
EXAMPLE_FILES = [(DATADIR / 'rap1-lieb-positives.txt', 137),
                 (DATADIR / 'rap1-lieb-test.txt', 3195)]
FINAL_DATA = [(DATADIR / 'train_final.txt', 12824),
              (DATADIR / 'test_final.txt', 1424)]

# Tests


def test_read_fasta_file():
    # Read the file in
    res = io.read_fasta_file(NEGATIVE_FA_FILE)
    assert len(res) == 3164

    # Sanity check each record
    for rec in res:
        assert set(rec) <= {'A', 'T', 'C', 'G'}
        # Should all be 1000 bp long, but there're a few weird records
        assert len(rec) in (1000, 334, 52, 490, 629, 792)


@pytest.mark.parametrize('filepath,exp_len', EXAMPLE_FILES)
def test_read_example_file(filepath, exp_len):
    # Read in the example file
    res = io.read_example_file(filepath)
    assert len(res) == exp_len

    # Sanity check each record
    for rec in res:
        assert set(rec) <= {'A', 'T', 'C', 'G'}
        assert len(rec) == 17


def test_write_score_file():

    # Records without truncation
    records = [
        ('AAAAAAACGCAACTAAT', 0.9),
        ('AAAAACACACATCTGGC', 0.1),
        ('AAAACCAAACACCTGAA', 0.5555),
    ]

    # Write out a tab-separated score file
    with tempfile.TemporaryDirectory() as tempdir:
        scorefile = pathlib.Path(tempdir) / 'scores.txt'

        # Make sure we can read our own writes
        io.write_score_file(scorefile, records)
        res_records = io.read_score_file(scorefile)

        assert res_records == records

    # Records with truncation
    records = [
        ('AAAAAAACGCAACTAAT', 0.9, 'AAA'),
        ('AAAAACACACATCTGGC', 0.1, 'ATA'),
        ('AAAACCAAACACCTGAA', 0.5555, 'ACC'),
    ]

    # Write out a tab-separated score file
    with tempfile.TemporaryDirectory() as tempdir:
        scorefile = pathlib.Path(tempdir) / 'scores.txt'

        # And that we can read them when they have a fragment
        io.write_score_file(scorefile, records)
        res_records = io.read_score_file(scorefile)

        assert res_records == records


@pytest.mark.parametrize('filepath,num_recs', FINAL_DATA)
def test_load_training_data(filepath, num_recs):

    res_x, res_y = io.load_training_data(filepath)

    assert res_x.shape == (68, num_recs)
    assert res_y.shape == (1, num_recs)
