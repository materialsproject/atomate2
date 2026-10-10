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


@pytest.mark.parametrize(
    ("src_dir_name", "src_files", "kwargs", "expected"),
    [
        (
            "prev",
            ["hessian.aims.gz", "geometry.in.next_step.gz", "rs.csc.gz", "aims.out"],
            {"restart_to_input": True},
            ["geometry.in.next_step", "hessian.aims", "rs.csc"],
        ),
        (
            "prev",
            ["a.cube", "b.cube.gz", "c.cube", "c.cube.gz", "aims.out"],
            {"additional_aims_files": ["*.cube"]},
            ["a.cube", "b.cube", "c.cube"],
        ),
        (
            "prev[1]",
            ["a.cube", "aims.out"],
            {"additional_aims_files": ["*.cube"]},
            ["a.cube"],
        ),
    ],
)
def test_copy_aims_outputs_patterns(
    tmp_path, monkeypatch, src_dir_name, src_files, kwargs, expected
):
    from atomate2.aims.files import copy_aims_outputs

    src_dir = tmp_path / src_dir_name
    src_dir.mkdir()
    for name in src_files:
        (src_dir / name).touch()
    dst_dir = tmp_path / "dst"
    dst_dir.mkdir()
    monkeypatch.chdir(dst_dir)

    copy_aims_outputs(src_dir=src_dir, **kwargs)

    assert sorted(f.name for f in dst_dir.iterdir()) == expected


def test_copy_aims_outputs_keeps_plain_content(tmp_path, monkeypatch):
    from atomate2.aims.files import copy_aims_outputs

    src_dir = tmp_path / "prev"
    src_dir.mkdir()
    (src_dir / "hessian.aims").write_text("hessian")
    dst_dir = tmp_path / "dst"
    dst_dir.mkdir()
    monkeypatch.chdir(dst_dir)

    copy_aims_outputs(src_dir=src_dir, restart_to_input=True)

    assert sorted(f.name for f in dst_dir.iterdir()) == ["hessian.aims"]
    assert (dst_dir / "hessian.aims").read_text() == "hessian"
