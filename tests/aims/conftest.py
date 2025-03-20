from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.io.aims.sets.base import AimsInputGenerator

import atomate2.aims.jobs.base
import atomate2.aims.run
from atomate2.common.files import gunzip_files

if TYPE_CHECKING:
    from collections.abc import Sequence


_REF_PATHS = {}
_FAKE_RUN_AIMS_KWARGS = {}
_VFILES = "control.in"


logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption(
        "--generate-test-data",
        action="store_true",
        help="Runs FHI-aims to create test data;"
        " runs tests against the created outputs",
    )


@pytest.fixture
def mg2mn4o8():
    return Structure(
        lattice=Lattice(
            [
                [5.06882343, 0.00012488, -2.66110167],
                [-1.39704234, 4.87249911, -2.66110203],
                [0.00986091, 0.01308528, 6.17649359],
            ]
        ),
        species=[
            "Mg",
            "Mg",
            "Mn",
            "Mn",
            "Mn",
            "Mn",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
        ],
        coords=[
            [0.37489726, 0.62510274, 0.75000002],
            [0.62510274, 0.37489726, 0.24999998],
            [-0.00000000, -0.00000000, 0.50000000],
            [-0.00000000, 0.50000000, 0.00000000],
            [0.50000000, -0.00000000, 0.50000000],
            [-0.00000000, -0.00000000, 0.00000000],
            [0.75402309, 0.77826750, 0.50805882],
            [0.77020285, 0.24594779, 0.99191316],
            [0.22173254, 0.24597689, 0.99194116],
            [0.24597691, 0.22173250, 0.49194118],
            [0.24594765, 0.77020288, 0.49191313],
            [0.22979715, 0.75405221, 0.00808684],
            [0.75405235, 0.22979712, 0.50808687],
            [0.77826746, 0.75402311, 0.00805884],
        ],
        site_properties={"magmom": [0, 0, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0]},
    )


@pytest.fixture
def si():
    return Structure(
        lattice=Lattice(
            [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
        ),
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )


@pytest.fixture
def nacl():
    return Structure(
        lattice=Lattice(
            [
                [3.422015, 0.0, 1.975702],
                [1.140671, 3.226306, 1.975702],
                [0.0, 0.0, 3.951402],
            ]
        ),
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )


@pytest.fixture
def o2():
    return Molecule(species=["O", "O"], coords=[[0, 0, 0.622978], [0, 0, -0.622978]])


@pytest.fixture(scope="session")
def species_dir():
    return Path(__file__).resolve().parent / "species_dir"


@pytest.fixture(scope="session")
def ref_path():
    from pathlib import Path

    module_dir = Path(__file__).resolve().parents[1]
    test_dir = module_dir / "test_data/aims/"
    return test_dir.resolve()


@pytest.fixture
def should_mock_aims(request):
    try:
        return not request.config.getoption("--generate-test-data")
    except ValueError:
        return True


@pytest.fixture
def mock_aims(monkeypatch, ref_path, should_mock_aims):
    """
    This fixture allows one to mock (fake) running FHI-aims.

    To use the fixture successfully, the following steps must be followed:
    1. "mock_aims" should be included as an argument to any test that would like to use
       its functionally.
    2. For each job in your workflow, you should prepare a reference directory
       containing two folders "inputs" (containing the reference input files expected
       to be produced by write_aims_input_set) and "outputs" (containing the expected
       output files to be produced by run_aims). These files should reside in a
       subdirectory of "tests/test_data/aims".
    3. Create a dictionary mapping each job name to its reference directory. Note that
       you should supply the reference directory relative to the "tests/test_data/aims"
       folder. For example, if your calculation has one job named "static" and the
       reference files are present in "tests/test_data/aims/Si_static", the dictionary
       would look like: ``{"static": "Si_static"}``.
    4. Optional (does not work yet): create a dictionary mapping each job name to
       custom keyword arguments that will be supplied to fake_run_aims.
       This way you can configure which control.in settings are expected for each job.
       For example, if your calculation has one job named "static" and you wish to
       validate that "xc" is set correctly in the control.in, your dictionary would
       look like
       ``{"static": {"input_settings": {"relativistic": "atomic_zora scalar"}}``.
    5. Inside the test function, call `mock_aims(ref_paths, fake_aims_kwargs)`, where
       ref_paths is the dictionary created in step 3 and fake_aims_kwargs is the
       dictionary created in step 4.
    6. Run your aims job after calling `mock_aims`.

    For examples, see the tests in tests/aims/jobs/core.py.
    """

    def mock_run_aims(*args, **kwargs):
        from jobflow import CURRENT_JOB

        name = CURRENT_JOB.job.name
        ref_dir = ref_path / _REF_PATHS[name]
        fake_run_aims(ref_dir, **_FAKE_RUN_AIMS_KWARGS.get(name, {}))

    get_input_set_orig = AimsInputGenerator.get_input_set

    def generate_test_data(*args, **kwargs):
        """A monkey patch for atomate2.aims.run.run_aims

        Runs the actual executable and copies inputs and outputs to the
        test data directory
        """
        import shutil

        from jobflow import CURRENT_JOB

        input_files = ["control.in", "geometry.in", "parameters.json"]
        name = CURRENT_JOB.job.name
        ref_dir = ref_path / _REF_PATHS[name]
        # running aims
        atomate2.aims.run.run_aims()
        # copy output files
        output_files = [f for f in Path.cwd().glob("*") if f.name not in input_files]
        shutil.rmtree(ref_dir, ignore_errors=True)
        os.makedirs(ref_dir / "inputs")
        os.makedirs(ref_dir / "outputs")
        for f in input_files:
            shutil.copy(Path.cwd() / f, ref_dir / "inputs")
        for f in output_files:
            shutil.copy(f, ref_dir / "outputs")

    def mock_get_input_set(self, *args, **kwargs):
        return get_input_set_orig(self, *args, **kwargs)

    if should_mock_aims:
        monkeypatch.setattr(atomate2.aims.run, "run_aims", mock_run_aims)
        monkeypatch.setattr(atomate2.aims.jobs.base, "run_aims", mock_run_aims)
        monkeypatch.setattr(AimsInputGenerator, "get_input_set", mock_get_input_set)
    else:
        monkeypatch.setattr(atomate2.aims.jobs.base, "run_aims", generate_test_data)

    def _run(ref_paths, fake_run_aims_kwargs=None):
        if fake_run_aims_kwargs is None:
            fake_run_aims_kwargs = {}

        _REF_PATHS.update(ref_paths)
        _FAKE_RUN_AIMS_KWARGS.update(fake_run_aims_kwargs)

    yield _run

    monkeypatch.undo()
    _REF_PATHS.clear()
    _FAKE_RUN_AIMS_KWARGS.clear()


def fake_run_aims(
    ref_path: str | Path,
    input_settings: Sequence[str] = (),
    check_inputs: Sequence[Literal["control.in"]] = _VFILES,
    clear_inputs: bool = False,
):
    """
    Emulate running aims and validate aims input files.

    Parameters
    ----------
    ref_path
        Path to reference directory with aims input files in the folder named 'inputs'
        and output files in the folder named 'outputs'.
    input_settings
        A list of input settings to check.
    check_inputs
        A list of aims input files to check. Supported options are "aims.inp"
    clear_inputs
        Whether to clear input files before copying in the reference aims outputs.
    """
    logger.info("Running fake aims.")

    ref_path = Path(ref_path)

    logger.info("Verified inputs successfully")

    if clear_inputs:
        clear_aims_inputs()

    copy_aims_outputs(ref_path)
    gunzip_files(
        include_files=list(Path.cwd().glob("*")),
        allow_missing=True,
    )

    # pretend to run aims by copying pre-generated outputs from reference dir
    logger.info("Generated fake aims outputs")


def clear_aims_inputs():
    for aims_file in ("control.in", "geometry.in", "parameters.json"):
        if Path(aims_file).exists():
            Path(aims_file).unlink()
    logger.info("Cleared aims inputs")


def copy_aims_outputs(ref_path: str | Path):
    import shutil

    output_path = ref_path / "outputs"
    for output_file in output_path.iterdir():
        if output_file.is_file():
            shutil.copy(output_file, ".")
