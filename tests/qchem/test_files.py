from pathlib import Path

import pytest

from atomate2.qchem.files import copy_qchem_outputs, get_largest_opt_extension


@pytest.mark.parametrize(
    "files",
    [("custodian.json.gz", "FW.json.gz")],
)
def test_copy_qchem_outputs_sp(qchem_test_dir, tmp_dir, files):
    path = qchem_test_dir / "water_single_point" / "outputs"
    copy_qchem_outputs(src_dir=path, additional_qchem_files=files)

    for file in files:
        assert Path(path / file).exists()


@pytest.mark.parametrize(
    "files",
    [("custodian.json.gz", "FW.json.gz")],
)
def test_copy_qchem_outputs_freq(qchem_test_dir, tmp_dir, files):
    path = qchem_test_dir / "water_frequency" / "outputs"
    copy_qchem_outputs(src_dir=path, additional_qchem_files=files)

    for file in files:
        assert Path(path / file).exists()


def test_get_largest_opt_extension(qchem_test_dir):
    path = qchem_test_dir / "double_opt_test" / "outputs"
    extension = get_largest_opt_extension(directory=path)
    assert extension == ".opt_2"

    path = qchem_test_dir / "water_single_point" / "static" / "outputs"
    extension = get_largest_opt_extension(directory=path)
    assert extension == ""
