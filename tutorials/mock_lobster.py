"""Mock LOBSTER functions for executing tutorials."""

import contextlib
import os
import tempfile
from collections.abc import Generator
from pathlib import Path

from pytest import MonkeyPatch

from atomate2.utils.testing.lobster import monkeypatch_lobster

TEST_ROOT = Path(__file__).parent.parent / "tests"
TEST_DIR = TEST_ROOT / "test_data"


@contextlib.contextmanager
def mock_lobster(ref_paths: dict) -> Generator:
    """Mock LOBSTER functions.

    Parameters
    ----------
    ref_paths (dict): A dictionary of reference paths to the test data.

    Yields
    ------
        function: A function that mocks calls to VASP.
    """
    for mf in monkeypatch_lobster(MonkeyPatch(), TEST_DIR / "lobster"):
        fake_run_lobster_kwargs = {k: {"check_lobster_inputs": ()} for k in ref_paths}

        old_cwd = os.getcwd()
        new_path = tempfile.mkdtemp()
        os.chdir(new_path)
        try:
            yield mf(ref_paths, fake_run_lobster_kwargs=fake_run_lobster_kwargs)
        finally:
            os.chdir(old_cwd)
            # shutil.rmtree(new_path)
