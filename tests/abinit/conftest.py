import logging

import pytest

logger = logging.getLogger("atomate2")


@pytest.fixture(scope="session")
def abinit_test_dir(test_dir):
    return test_dir / "abinit"
