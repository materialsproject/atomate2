from pymatgen.io.openmm.sets import OpenMMSet
from atomate2.openmm.constants import OpenMMConstants
from pathlib import Path
import jobflow
import pytest
import os


@pytest.fixture(scope="function")
def job_store():
    job_store = jobflow.SETTINGS.JOB_STORE
    return job_store


@pytest.fixture(scope="session")
def platform():
    return "CPU"


@pytest.fixture(scope="session")
def platform_properties():
    return None


@pytest.fixture(scope="session")
def test_data_dir():
    module_dir = Path(__file__).resolve().parent.parent
    test_data_dir = Path(os.path.join(module_dir, "./test_data")).resolve()
    return test_data_dir


@pytest.fixture(scope="session")
def alchemy_input_set_dir(test_data_dir):
    alchemy_input_set_dir = Path(os.path.join(test_data_dir, "./minimized_alchemy_input_set")).resolve()
    return alchemy_input_set_dir


@pytest.fixture(scope="session")
def alchemy_input_set(alchemy_input_set_dir):
    input_set = OpenMMSet.from_directory(
        directory=alchemy_input_set_dir,
        topology_file=OpenMMConstants.TOPOLOGY_PDD_FILE_NAME.value,
        state_file=OpenMMConstants.STATE_XML_FILE_NAME.value,
        system_file=OpenMMConstants.SYSTEM_XML_FILE_NAME.value,
        integrator_file=OpenMMConstants.INTEGRATOR_XML_FILE_NAME.value,
        contents_file=OpenMMConstants.CONTENTS_JOSN_FILE_NAME.value,
    )
    return input_set


@pytest.fixture
def test_state_report_file(test_data_dir):
    state_file = Path(os.path.join(test_data_dir, "./reporters/state.txt"))
    return state_file.resolve()


@pytest.fixture(scope="function")
def task_details():
    from atomate2.openmm.schemas.task_details import TaskDetails

    return TaskDetails(
        task_name="fixture",
        task_kwargs={"fixture": "fixture"},
        platform_kwargs={"platform": "CPU"},
        total_steps=0,
    )



@pytest.fixture(scope="function")
def test_output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp(basename="output")
