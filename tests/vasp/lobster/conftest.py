from atomate2.utils.testing.lobster import *
from pytest import MonkeyPatch

from typing import TYPE_CHECKING, Any, Final

from collections.abc import Generator, Callable



@pytest.fixture
def mock_lobster(
    monkeypatch: MonkeyPatch, lobster_test_dir: Path
) -> Generator[Callable[[Any, Any], Any], None, None]:
    """

    """
    yield from monkeypatch_lobster(monkeypatch, lobster_test_dir)