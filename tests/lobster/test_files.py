from pathlib import Path

from atomate2.common.files import copy_files, gunzip_files, gzip_output_folder
from atomate2.lobster.files import LOBSTEROUTPUT_FILES


def test_gzip_lobster_output_files(tmp_path):
    lobster_test_dir = Path(__file__).parent / ".." / "test_data" / "lobster"

    files_to_zip = [*LOBSTEROUTPUT_FILES, "lobsterin"]

    # copy and unzip test files
    copy_files(
        src_dir=lobster_test_dir / "lobsteroutputs" / "AlN_LCFO",
        dest_dir=tmp_path,
        allow_missing=True,
    )
    gunzip_files(tmp_path)

    # gzip folder
    gzip_output_folder(
        directory=tmp_path,
        setting=True,
        files_list=files_to_zip,
    )

    # verify all gzipped files are expected output files
    for file in files_to_zip:
        if file in tmp_path.iterdir():
            gz_file = tmp_path / f"{file}.gz"
            assert gz_file.exists()
