"""Test utility functions.

TODO: there should be tests for the other utility functions.
Not high priority but good for long term.
"""

from enum import StrEnum

import numpy as np
import pytest

from atomate2.common.utils import _recursive_to_list, to_enum_member

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


class Color(StrEnum):
    red = "red"
    dark_blue = "dark_blue"
    Mixed = "Mixed"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("red", Color.red),
        ("DARK_BLUE", Color.dark_blue),
        ("Mixed", Color.Mixed),
        ("mixed", Color.Mixed),
        (Color.red, Color.red),
    ],
)
def test_to_enum_member(value, expected) -> None:
    assert to_enum_member(Color, value) is expected


def test_to_enum_member_invalid() -> None:
    match = r"Unknown Color 'green'\. Valid values: \['red', 'dark_blue', 'Mixed'\]"
    with pytest.raises(ValueError, match=match):
        to_enum_member(Color, "green")
