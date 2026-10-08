from pathlib import Path

from atomate2.common.files import (
    copy_files,
    gunzip_files,
    gzip_files,
    gzip_output_folder,
)


def test_gunzip_force_overwrites(tmp_path):
    files = ["file1", "file2", "file3"]
    for fname in files:
        f = tmp_path / fname
        f.write_text(fname)
    gzip_files(tmp_path)

    for fname in files:
        f = tmp_path / fname
        f.write_text(f"{fname} overwritten")
    # "file1" in the zipped files and "file1 overwritten" in the unzipped files
    gunzip_files(tmp_path, force=True)

    for fname in files:
        f = tmp_path / fname
        assert f.read_text() == fname

    gzip_files(tmp_path)

    for fname in files:
        f = tmp_path / fname
        f.write_text(f"{fname} overwritten")

    # "file1" in the zipped files and "file1 overwritten" in the unzipped files
    gunzip_files(tmp_path, force="skip")
    for fname in files:
        f = tmp_path / fname
        assert f.read_text() == f"{fname} overwritten"


def test_gunzip_missing_file_keeps_existing(tmp_path):
    existing = tmp_path / "file1"
    existing.write_text("file1")

    gunzip_files(
        tmp_path, include_files=["file1.gz", "file2.gz"], allow_missing=True, force=True
    )

    assert existing.read_text() == "file1"
    assert not (tmp_path / "file2").exists()


def test_copy_files_glob_chars_in_directory(tmp_path, monkeypatch):
    src_dir = tmp_path / "prev[1]"
    src_dir.mkdir()
    (src_dir / "a.dat").write_text("a")
    dst_dir = tmp_path / "dst"
    dst_dir.mkdir()
    monkeypatch.chdir(dst_dir)

    copy_files(src_dir, include_files=["*.dat"])

    assert (dst_dir / "a.dat").read_text() == "a"


def test_zip_outputs(tmp_dir):
    for file_name in ("a", "b"):
        (Path.cwd() / file_name).touch()

    gzip_output_folder(directory=Path.cwd(), setting=False, files_list=["a"])

    assert (Path.cwd() / "a").exists()
    assert not (Path.cwd() / "a.gz").exists()
    assert (Path.cwd() / "b").exists()
    assert not (Path.cwd() / "b.gz").exists()

    gzip_output_folder(directory=Path.cwd(), setting="atomate", files_list=["a"])

    assert not (Path.cwd() / "a").exists()
    assert (Path.cwd() / "a.gz").exists()
    assert (Path.cwd() / "b").exists()
    assert not (Path.cwd() / "b.gz").exists()

    gzip_output_folder(directory=Path.cwd(), setting=True, files_list=["a"])

    assert not (Path.cwd() / "a").exists()
    assert (Path.cwd() / "a.gz").exists()
    assert not (Path.cwd() / "b").exists()
    assert (Path.cwd() / "b.gz").exists()
