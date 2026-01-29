"""Test utility functions.

TODO: there should be tests for the other utility functions.
Not high priority but good for long term.
"""

import numpy as np

from atomate2.common.utils import _recursive_to_list

try:
    import torch
except ImportError:
    torch = None


def test_to_list():
    as_list = [[1, 2, 3, 4, 5], [10, 9, 8, 7, 6], [3, 5, 7, 9, 11]]
    arr = np.array(as_list)

    assert _recursive_to_list(arr) == as_list
    if torch is not None:
        assert _recursive_to_list(torch.from_numpy(arr)) == as_list
    for obj in (None, 1, 1.5):
        assert _recursive_to_list(obj) == obj
