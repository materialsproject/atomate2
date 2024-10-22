from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest
from pymatgen.io.lobster import Lobsterin

import atomate2.lobster.jobs
import atomate2.lobster.run

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("atomate2")

_LFILES = "lobsterin"
_DFT_FILES = ("WAVECAR", "POSCAR", "INCAR", "KPOINTS", "POTCAR")
_LOBS_REF_PATHS = {}
_FAKE_RUN_LOBSTER_KWARGS = {}


@pytest.fixture(scope="session")
def lobster_test_dir(test_dir):
    return test_dir / "lobster"


@pytest.fixture
def mock_lobster(monkeypatch, lobster_test_dir):
    """
    This fixture allows one to mock (fake) running LOBSTER.
    It works by monkeypatching (replacing) calls to run_lobster that will
    work when the lobster executables
    are not present.
    The primary idea is that instead of running LOBSTER to generate the output files,
    reference files will be copied into the directory instead. As we do not want to
    test whether LOBSTER is giving the correct output rather that the calculation inputs
    are generated correctly and that the outputs are parsed properly, this should be
    sufficient for our needs.
    To use the fixture successfully, the following steps must be followed:
    1. "mock_lobster" should be included as an argument to any test that would
        like to use its functionally.
    2. For each job in your workflow, you should prepare a reference directory
       containing two folders "inputs" (containing the reference input files
       expected to be produced by Lobsterin.standard_calculations_from_vasp_files
       and "outputs" (containing the expected
       output files to be produced by run_lobster). These files should reside in a
       subdirectory of "tests/test_data/lobster".
    3. Create a dictionary mapping each job name to its reference directory.
       Note that you should supply the reference directory relative to the
       "tests/test_data/lobster" folder. For example, if your calculation
       has one job named "lobster_run_0" and the reference files are present in
       "tests/test_data/lobster/Si_lobster_run_0", the dictionary
       would look like: ``{"lobster_run_0": "Si_lobster_run_0"}``.
    4. Optional: create a dictionary mapping each job name to custom
       keyword arguments that will be supplied to fake_run_lobster.
       This way you can configure which lobsterin settings are expected for each job.
       For example, if your calculation has one job named "lobster_run_0"
       and you wish to validate that "basisfunctions" is set correctly
       in the lobsterin, your dictionary would look like
       ``{"lobster_run_0": {"lobsterin_settings": {"basisfunctions": Ba 5p 5s 6s}}``.
    5. Inside the test function, call `mock_lobster(ref_paths, fake_lobster_kwargs)`,
       where ref_paths is the dictionary created in step 3
       and fake_lobster_kwargs is the
       dictionary created in step 4.
    6. Run your lobster job after calling `mock_lobster`.
    """

    def mock_run_lobster(*args, **kwargs):
        from jobflow import CURRENT_JOB

        name = CURRENT_JOB.job.name
        ref_path = lobster_test_dir / _LOBS_REF_PATHS[name]
        fake_run_lobster(ref_path, **_FAKE_RUN_LOBSTER_KWARGS.get(name, {}))

    monkeypatch.setattr(atomate2.lobster.run, "run_lobster", mock_run_lobster)
    monkeypatch.setattr(atomate2.lobster.jobs, "run_lobster", mock_run_lobster)

    def _run(ref_paths, fake_run_lobster_kwargs):
        _LOBS_REF_PATHS.update(ref_paths)
        _FAKE_RUN_LOBSTER_KWARGS.update(fake_run_lobster_kwargs)

    yield _run

    monkeypatch.undo()
    _LOBS_REF_PATHS.clear()


def fake_run_lobster(
    ref_path: str | Path,
    check_lobster_inputs: Sequence[Literal["lobsterin"]] = _LFILES,
    check_dft_inputs: Sequence[Literal["WAVECAR", "POSCAR"]] = _DFT_FILES,
    lobsterin_settings: Sequence[str] = (),
):
    """
    Emulate running LOBSTER and validate LOBSTER input files.
    Parameters
    ----------
    ref_path
        Path to reference directory with VASP input files in the folder named 'inputs'
        and output files in the folder named 'outputs'.
    check_lobster_inputs
        A list of lobster input files to check. Supported options are "lobsterin.gz".
    lobsterin_settings
        A list of LOBSTER settings to check.
    """
    logger.info("Running fake LOBSTER.")
    ref_path = Path(ref_path)

    # Checks if DFT files have been copied
    for file in check_dft_inputs:
        Path(file).exists()
    logger.info("Verified copying of VASP files successfully")
    # zipped or not zipped?
    if "lobsterin" in check_lobster_inputs:
        verify_inputs(ref_path, lobsterin_settings)

    logger.info("Verified LOBSTER inputs successfully")

    copy_lobster_outputs(ref_path)

    # pretend to run LOBSTER by copying pre-generated outputs from reference dir
    logger.info("ran fake LOBSTER, generated outputs")


def verify_inputs(ref_path: str | Path, lobsterin_settings: Sequence[str]):
    user = Lobsterin.from_file("lobsterin")

    # Check lobsterin
    ref = Lobsterin.from_file(ref_path / "inputs" / "lobsterin")

    for key in lobsterin_settings:
        if user.get(key) != ref.get(key):
            raise ValueError(f"lobsterin value of {key} is inconsistent!")


def copy_lobster_outputs(ref_path: str | Path):
    output_path = ref_path / "outputs"
    for output_file in output_path.iterdir():
        if output_file.is_file():
            shutil.copy(output_file, ".")
