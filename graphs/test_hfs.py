import numpy as np
import pytest

from graphs.generate_data import blobs
from graphs.hard_hfs import hfs

modes = ['simple', 'iterative', 'online']

# http://remusao.github.io/pytest-paramaterize-tests-with-external-data.html

@pytest.mark.parametrize("mode", modes)
def test_hfs(mode):
    n_samples = 200
    n_kept = 150
    r = n_kept / n_samples
    X, Y = blobs(n_samples, 3, 2)
    Y += 1
    Y_masked = np.append(Y[:n_kept], np.zeros(50), axis=0)
    labs = hfs(X, Y_masked, mode=mode)
    assert (labs == Y).mean() > r
