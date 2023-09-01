def test_gunzip_force_overwrites(tmp_path):
    from atomate2.common.files import gunzip_files, gzip_files

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
