import pytest
from atomate2.amset.files import copy_amset_files


@pytest.mark.parametrize(
    "transport_filename,expect_prev_file",
    [
        ("transport_55x55x83.json", True),
        (None, False),
    ],
)
def test_copy_amset_files_transport_handling(
    tmp_path, monkeypatch, transport_filename, expect_prev_file
):
    prev_dir = tmp_path / "prev_job"
    prev_dir.mkdir()
    if transport_filename:
        (prev_dir / transport_filename).write_text("{}")

    new_dir = tmp_path / "new_job"
    new_dir.mkdir()
    monkeypatch.chdir(new_dir)

    copy_amset_files(src_dir=str(prev_dir))  

    assert (new_dir / "transport.prev.json").exists() == expect_prev_file
