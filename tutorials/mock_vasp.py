"""Mock VASP functions for executing tutorials."""

import contextlib
import os
import shutil
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path

from pytest import MonkeyPatch

# load the vasp conftest
TEST_ROOT = Path(__file__).parent.parent / "tests"
TEST_DIR = TEST_ROOT / "test_data"
VASP_TEST_DATA = TEST_ROOT / "test_data/vasp"
sys.path.insert(0, str(TEST_ROOT / "vasp"))
from conftest import _mock_vasp  # noqa: E402


@contextlib.contextmanager
def mock_vasp(ref_paths: dict) -> Generator:
    """Mock VASP functions.

    Parameters
    ----------
    ref_paths (dict): A dictionary of reference paths to the test data.

    Yields
    ------
        function: A function that mocks calls to VASP.
    """
    for mf in _mock_vasp(MonkeyPatch(), TEST_ROOT / "test_data/vasp"):
        fake_run_vasp_kwargs = {k: {"check_inputs": ()} for k in ref_paths}
        old_cwd = os.getcwd()
        new_path = tempfile.mkdtemp()
        os.chdir(new_path)
        try:
            yield mf(ref_paths, fake_run_vasp_kwargs)
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(new_path)
