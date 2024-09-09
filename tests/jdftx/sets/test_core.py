import pytest
import os

@pytest.mark.parametrize("deleteme", [
    ("delete this"),
])
def test_input_generators(deleteme):
    # from atomate2.jdftx.sets.core import (
    #     RelaxSetGenerator,
    # )
    # from atomate2.jdftx.sets.base import JdftxInputSet

    # gen = RelaxSetGenerator()

    print(deleteme)
    assert len(deleteme)


assert True