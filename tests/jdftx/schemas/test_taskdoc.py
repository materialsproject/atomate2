# test that TaskDoc is loaded with the right attributes
from pathlib import Path

import pytest
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from pymatgen.io.jdftx.sets import FILE_NAMES

from atomate2.jdftx.schemas.task import TaskDoc


@pytest.mark.parametrize("task_name", ["sp_test"], indirect=True)
@pytest.mark.parametrize("task_dir_name", ["sp_test"], indirect=False)
def test_taskdoc(task_name, task_dir_name, mock_filenames, jdftx_test_dir):
    """
    Test the JDFTx TaskDoc to verify that attributes are created properly.
    """
    FILE_NAMES["in"] = "inputs/" + FILE_NAMES["in"]
    FILE_NAMES["out"] = "outputs/" + FILE_NAMES["out"]
    dir_name = jdftx_test_dir / Path(task_dir_name)
    taskdoc = TaskDoc.from_directory(dir_name=dir_name)
    jdftxoutfile = JDFTXOutfile.from_file(dir_name / Path(FILE_NAMES["out"]))
    # check that the taskdoc attributes correspond to the expected values.
    # currently checking task_type and energy
    assert taskdoc.task_type == task_name
    assert taskdoc.calc_outputs.energy == jdftxoutfile.e
