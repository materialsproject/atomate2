import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def test_dir():
    module_dir = Path(__file__).resolve().parent
    test_dir = module_dir / "test_data"
    return test_dir.resolve()


@pytest.fixture
def vasp_test_dir(test_dir):
    return test_dir / "vasp"


@pytest.fixture
def log_to_stdout():
    # Set Logging
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    root.addHandler(ch)


@pytest.fixture
def clean_dir():
    old_cwd = os.getcwd()
    newpath = tempfile.mkdtemp()
    os.chdir(newpath)
    yield
    os.chdir(old_cwd)
    shutil.rmtree(newpath)
