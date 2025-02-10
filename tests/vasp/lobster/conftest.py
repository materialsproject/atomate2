from collections.abc import Callable, Generator
from typing import Any

from pytest import MonkeyPatch

from atomate2.utils.testing.lobster import *


@pytest.fixture
def mock_lobster(
    monkeypatch: MonkeyPatch, lobster_test_dir: Path
) -> Generator[Callable[[Any, Any], Any], None, None]:
    """ """
    yield from monkeypatch_lobster(monkeypatch, lobster_test_dir)
