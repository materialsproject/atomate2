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
def openmm_data(test_dir):
    return test_dir / "openmm"


@pytest.fixture
def test_state_report_file(openmm_data):
    state_file = Path(os.path.join(openmm_data, "./reporters/state.txt"))
    return state_file.resolve()


@pytest.fixture(scope="function")
def test_output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp(basename="output")
