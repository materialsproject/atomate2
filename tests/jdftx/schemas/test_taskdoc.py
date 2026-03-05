# test that TaskDoc is loaded with the right attributes
from pathlib import Path

import pytest
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from pymatgen.io.jdftx.sets import FILE_NAMES

from atomate2.jdftx.schemas.task import TaskDoc

from ..conftest import copy_jdftx_outputs  # noqa: TID252


@pytest.mark.parametrize("task_name", ["sp_test"], indirect=True)
@pytest.mark.parametrize("task_dir_name", ["sp_test"], indirect=False)
def test_taskdoc(task_name, task_dir_name, mock_filenames, jdftx_test_dir, tmp_dir):
    """
    Test the JDFTx TaskDoc to verify that attributes are created properly.
    """
    for subdir in ("inputs", "outputs"):
        copy_jdftx_outputs(jdftx_test_dir / Path(task_dir_name), suffix=subdir)
    taskdoc = TaskDoc.from_directory(dir_name=".")
    jdftxoutfile = JDFTXOutfile.from_file(Path(FILE_NAMES["out"]))
    # check that the taskdoc attributes correspond to the expected values.
    # currently checking task_type and energy
    assert taskdoc.task_type == task_name
    assert taskdoc.calc_outputs.energy == jdftxoutfile.e
