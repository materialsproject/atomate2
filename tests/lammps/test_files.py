import gzip

import pytest

from atomate2.lammps.files import copy_lammps_restart_file


@pytest.mark.parametrize(
    ("prev_dir_name", "gzipped"),
    [("prev", False), ("prev", True), ("prev[1]", True)],
)
def test_copy_lammps_restart_file(tmp_path, monkeypatch, prev_dir_name, gzipped):
    prev_dir = tmp_path / prev_dir_name
    prev_dir.mkdir()
    if gzipped:
        with gzip.open(prev_dir / "md.restart.gz", "wt") as file:
            file.write("restart")
    else:
        (prev_dir / "md.restart").write_text("restart")
    prev_files = sorted(prev_dir.iterdir())

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(run_dir)

    assert copy_lammps_restart_file(prev_dir) == "md.restart"
    assert sorted(f.name for f in run_dir.iterdir()) == ["md.restart"]
    assert (run_dir / "md.restart").read_text() == "restart"
    # the previous directory is left untouched
    assert sorted(prev_dir.iterdir()) == prev_files


def test_copy_lammps_restart_file_plain_and_gzipped(tmp_path, monkeypatch):
    prev_dir = tmp_path / "prev"
    prev_dir.mkdir()
    (prev_dir / "md.restart").write_text("restart")
    with gzip.open(prev_dir / "md.restart.gz", "wt") as file:
        file.write("restart")
    monkeypatch.chdir(tmp_path)

    assert copy_lammps_restart_file(prev_dir) == "md.restart"
    assert (tmp_path / "md.restart").read_text() == "restart"


@pytest.mark.parametrize("restart_files", [[], ["a.restart", "b.restart"]])
def test_copy_lammps_restart_file_not_one(tmp_path, monkeypatch, restart_files):
    prev_dir = tmp_path / "prev"
    prev_dir.mkdir()
    for name in restart_files:
        (prev_dir / name).touch()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="Expected exactly one restart file"):
        copy_lammps_restart_file(prev_dir)
