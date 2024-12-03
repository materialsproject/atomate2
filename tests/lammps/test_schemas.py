from atomate2.lammps.schemas.task import LammpsTaskDocument
import os

test_data_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test_data', 'lammps'))

def test_task_doc():
    return