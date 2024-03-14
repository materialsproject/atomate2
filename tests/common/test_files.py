from pathlib import Path

from atomate2.common.files import gunzip_files, gzip_files, gzip_output_folder


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
