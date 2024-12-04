# test that TaskDoc is loaded with the right attributes
import os

import pytest
from pymatgen.io.jdftx.outputs import JDFTXOutfile

from atomate2.jdftx.schemas.task import TaskDoc
from atomate2.jdftx.sets.base import FILE_NAMES


@pytest.mark.parametrize("task_name", ["sp_test"], indirect=True)
@pytest.mark.parametrize("mock_cwd", ["sp_test"], indirect=True)
def test_taskdoc(mock_cwd, task_name, mock_filenames):
    """
    Test the JDFTx TaskDoc to verify that attributes are created properly.
    """
    cwd = os.getcwd()
    taskdoc = TaskDoc.from_directory(dir_name=cwd, filenames=FILE_NAMES)
    jdftxoutfile = JDFTXOutfile.from_file(os.path.join(cwd, FILE_NAMES["out"]))
    # check that the taskdoc attributes correspond to the expected values.
    # currently checking task_type, dir_name, and energy
    assert taskdoc.task_type == task_name
    assert str(taskdoc.dir_name) == str(cwd)
    assert taskdoc.calc_outputs.energy == jdftxoutfile.e
