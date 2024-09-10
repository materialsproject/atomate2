"""Tests for file manipulation"""

from pathlib import Path

import pytest

TEST_DIR = Path(__file__).parent


@pytest.fixture
def tmp_dir():
    """Same as clean_dir but is fresh for every test"""
    import os
    import shutil
    import tempfile

    old_cwd = os.getcwd()
    newpath = tempfile.mkdtemp()
    os.chdir(newpath)
    yield
    os.chdir(old_cwd)
    shutil.rmtree(newpath)


def test_copy_aims_outputs(tmp_dir):
    from atomate2.aims.files import copy_aims_outputs

    files = ["aims.out"]
    restart_files = ["geometry.in.next_step", "D_spin_01_kpt_000001.csc"]

    path = TEST_DIR / "outputs"
    copy_aims_outputs(src_dir=path, restart_to_input=True, additional_aims_files=files)

    for f in files + restart_files:
        assert Path(f).exists()
