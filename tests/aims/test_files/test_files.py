"""Tests for file manipulation"""

import warnings
from pathlib import Path

import pytest

from atomate2.aims.files import get_aims_zip_files_setting
from atomate2.settings import Atomate2Settings

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


@pytest.mark.parametrize(
    ("settings_kwargs", "expected", "deprecated"),
    [
        ({}, "atomate", False),
        ({"AIMS_ZIP_FILES": False}, False, False),
        ({"AIMS_ZIP_FILES": True, "VASP_ZIP_FILES": False}, True, False),
        ({"VASP_ZIP_FILES": False}, False, True),
        ({"VASP_ZIP_FILES": "atomate"}, "atomate", True),
    ],
)
def test_get_aims_zip_files_setting(monkeypatch, settings_kwargs, expected, deprecated):
    monkeypatch.setattr("atomate2.SETTINGS", Atomate2Settings(**settings_kwargs))

    if deprecated:
        with pytest.warns(DeprecationWarning, match="AIMS_ZIP_FILES"):
            assert get_aims_zip_files_setting() == expected
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert get_aims_zip_files_setting() == expected


def test_get_aims_zip_files_setting_from_env(monkeypatch):
    monkeypatch.setenv("ATOMATE2_VASP_ZIP_FILES", "false")
    monkeypatch.setattr("atomate2.SETTINGS", Atomate2Settings())

    with pytest.warns(DeprecationWarning, match="AIMS_ZIP_FILES"):
        assert get_aims_zip_files_setting() is False
