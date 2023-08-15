import pytest
from monty.serialization import loadfn


@pytest.fixture(scope="session")
def test_dir():
    from pathlib import Path

    module_dir = Path(__file__).resolve().parent
    test_dir = module_dir / "test_data"
    return test_dir.resolve()


@pytest.fixture(scope="session")
def log_to_stdout():
    import logging
    import sys

    # Set Logging
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    root.addHandler(ch)


@pytest.fixture(scope="session")
def clean_dir(debug_mode):
    import os
    import shutil
    import tempfile

    old_cwd = os.getcwd()
    new_path = tempfile.mkdtemp()
    os.chdir(new_path)
    yield
    if debug_mode:
        print(f"Tests ran in {new_path}")
    else:
        os.chdir(old_cwd)
        shutil.rmtree(new_path)


@pytest.fixture()
def tmp_dir():
    """Same as clean_dir but is fresh for every test"""
    import os
    import shutil
    import tempfile

    old_cwd = os.getcwd()
    newpath = tempfile.mkdtemp()
    os.chdir(newpath)
    yield
    os.chdir(old_cwd)
    shutil.rmtree(newpath)


@pytest.fixture(scope="session")
def debug_mode():
    return False


@pytest.fixture(scope="session")
def lpad(database, debug_mode):
    from fireworks import LaunchPad

    lpad = LaunchPad(name=database)
    lpad.reset("", require_password=False)
    yield lpad

    if not debug_mode:
        lpad.reset("", require_password=False)
        for coll in lpad.db.list_collection_names():
            lpad.db[coll].drop()


@pytest.fixture()
def memory_jobstore():
    from jobflow import JobStore
    from maggma.stores import MemoryStore

    store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
    store.connect()

    return store


@pytest.fixture(scope="session", autouse=True)
def log_to_stdout_auto_use():
    from atomate2.utils.log import initialize_logger

    initialize_logger()


@pytest.fixture()
def si_structure(test_dir):
    from pymatgen.core import Structure

    return Structure.from_file(test_dir / "structures" / "Si.cif")


@pytest.fixture(autouse=True)
def mock_jobflow_settings(memory_jobstore):
    """Mock the jobflow settings to use our specific jobstore (with data store)."""
    from unittest import mock

    from jobflow.settings import JobflowSettings

    settings = JobflowSettings(JOB_STORE=memory_jobstore)

    with mock.patch("jobflow.SETTINGS", settings):
        yield


@pytest.fixture(
    params=[
        "Si_227",
        "Fe_229",
        "S_58",
        "Rb2P3_69",
        "K2Au3_71",
        "LaI3_63",
        "KCeF4_123",
        "RbO2_129",
        "BaN2_15",
        "TiNi_11",
        "CaC2_2",
        "KNO3_160",
        "ZnO_186",
    ],
    ids=[
        "F cubic",
        "I cubic",
        "P orth",
        "I orth",
        "F orth",
        "C orth",
        "P tet",
        "I tet",
        "C mono",
        "P mono",
        "tri",
        "rhom",
        "hex",
    ],
    scope="session",
)
def symmetry_structure(test_dir, request):
    """The structures are copied from amset.
    See https://github.com/hackingmaterials/amset/blob/main/tests/conftest.py
    """
    return loadfn(test_dir / "symmetry_structures" / f"{request.param}.json.gz")
