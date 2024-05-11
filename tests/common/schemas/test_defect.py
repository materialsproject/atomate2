import json

import numpy as np
import pytest
from monty.json import MontyEncoder

from atomate2.common.schemas.defects import (
    CCDDocument,
    FormationEnergyDiagramDocument,
    sort_pos_dist,
)


def test_sort_pos_dist():
    """
    Test the sorting algorithm with a list of 2D positions.
    The algorithm should sort the list into a straight line depending on
    the direction of s1 and s2
    """

    def abs_d(s1, s2):
        return np.linalg.norm(np.array(s1) - np.array(s2))

    points_on_line_2d = [(1, 1), (-2, -2), (0, 0), (2, 2), (-1, -1)]
    sorted_pts, dists = sort_pos_dist(
        points_on_line_2d, s1=(0, 0), s2=(1.5, 1.5), dist=abs_d
    )
    assert sorted_pts == [(-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2)]
    assert dists == pytest.approx([-2.8284271, -1.4142135, 0, 1.41421356, 2.82842712])

    sorted_pts, dists = sort_pos_dist(
        points_on_line_2d, s1=(0, 0), s2=(-2.5, -2.5), dist=abs_d
    )
    assert sorted_pts == [(2, 2), (1, 1), (0, 0), (-1, -1), (-2, -2)]
    assert dists == pytest.approx([-2.8284271, -1.4142135, 0, 1.41421356, 2.82842712])


# schemas where all fields have default values
@pytest.mark.parametrize(
    "model_cls",
    [FormationEnergyDiagramDocument, CCDDocument],
)
def test_model_validate(model_cls):
    model_cls.model_validate_json(json.dumps(model_cls(), cls=MontyEncoder))
