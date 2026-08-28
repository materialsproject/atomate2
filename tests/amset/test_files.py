import gzip

import pytest

from atomate2.amset.files import copy_amset_files


@pytest.mark.parametrize(
    "transport_filename,expect_prev_file",
    [
        ("transport_55x55x83.json.gz", True),
        (None, False),
    ],
)
# testing if a transport file can be extracted from the prev_dir and renamed
# testing that first run case where no prev_dir exists behaves
def test_copy_amset_files_transport_handling(
    tmp_path, monkeypatch, transport_filename, expect_prev_file
):
    prev_dir = tmp_path / "prev_job"
    prev_dir.mkdir()
    if transport_filename:
        with gzip.open(prev_dir / transport_filename, "wt") as f:
            f.write("{}")  # write the file to the prev_dir

    new_dir = tmp_path / "new_job"
    new_dir.mkdir()
    monkeypatch.chdir(new_dir)
    # copy file from prev_dir to new_dir, changing the name to transport.prev.json
    copy_amset_files(src_dir=str(prev_dir))

    assert (new_dir / "transport.prev.json").exists() == expect_prev_file
