import random

import numpy as np

from atomate2.vasp.schemas.defect import sort_pos_dist


def test_sort_pos_dist():
    def abs_d(s1, s2):
        return np.linalg.norm(np.array(s1) - np.array(s2))

    input = [(0, 0), (1, 1), (-1, -1), (2, 2), (-2, -2)]
    random.shuffle(input)
    r, d = sort_pos_dist(input, s1=(0, 0), s2=(1.5, 1.5), dist=abs_d)
    assert r == [(-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2)]

    r, d = sort_pos_dist(input, s1=(0, 0), s2=(-2.5, -2.5), dist=abs_d)
    assert r == [(2, 2), (1, 1), (0, 0), (-1, -1), (-2, -2)]
