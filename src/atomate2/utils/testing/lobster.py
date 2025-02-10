"""Utilities for testing LOBSTER calculations."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pymatgen.io.lobster import Lobsterin

import atomate2.lobster.jobs
import atomate2.lobster.run

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("atomate2")

_LFILES = "lobsterin"
_DFT_FILES = ("WAVECAR", "POSCAR", "INCAR", "KPOINTS", "POTCAR")
_LOBS_REF_PATHS: dict[str, str | Path] = {}
_FAKE_RUN_LOBSTER_KWARGS: dict[str, dict[str, Sequence]] = {}


@pytest.fixture(scope="session")
def lobster_test_dir(test_dir):
    return test_dir / "lobster"


def monkeypatch_lobster(monkeypatch: pytest.MonkeyPatch, lobster_test_dir: Path):
    """Monkeypatch LOBSTER run calls for testing purposes.

    This generator can be used as a context manager or pytest fixture ("mock_lobster").
    It replaces calls to run_lobster with a mock function that copies reference files
    instead of running LOBSTER.

    The primary idea is that instead of running LOBSTER to generate the output files,
    reference files will be copied into the directory instead. This ensures that the
    calculation inputs are generated correctly and that the outputs are parsed properly.

    To use the fixture successfully, follow these steps:
    1. Include "mock_lobster" as an argument to any test that would like to use its functionality.
    2. For each job in your workflow, prepare a reference directory containing two folders:
       "inputs" (containing the reference input files expected to be produced by
       Lobsterin.standard_calculations_from_vasp_files) and "outputs" (containing the expected
       output files to be produced by run_lobster). These files should reside in a subdirectory
       of "tests/test_data/lobster".
    3. Create a dictionary mapping each job name to its reference directory. Note that you should
       supply the reference directory relative to the "tests/test_data/lobster" folder. For example,
       if your calculation has one job named "lobster_run_0" and the reference files are present in
       "tests/test_data/lobster/Si_lobster_run_0", the dictionary would look like:
       {"lobster_run_0": "Si_lobster_run_0"}.
    4. Optionally, create a dictionary mapping each job name to custom keyword arguments that will be
       supplied to fake_run_lobster. This way you can configure which lobsterin settings are expected
       for each job. For example, if your calculation has one job named "lobster_run_0" and you wish
       to validate that "basisfunctions" is set correctly in the lobsterin, your dictionary would look like:
       {"lobster_run_0": {"lobsterin_settings": {"basisfunctions": Ba 5p 5s 6s}}.
    5. Inside the test function, call `mock_lobster(ref_paths, fake_lobster_kwargs)`, where ref_paths is the
       dictionary created in step 3 and fake_lobster_kwargs is the dictionary created in step 4.
    6. Run your LOBSTER job after calling `mock_lobster`.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
        lobster_test_dir (Path): The directory containing reference files for LOBSTER tests.
    """

    def mock_run_lobster(*args, **kwargs) -> None:
        from jobflow import CURRENT_JOB

        name = CURRENT_JOB.job.name
        ref_path = lobster_test_dir / _LOBS_REF_PATHS[name]
        fake_run_lobster(ref_path, **_FAKE_RUN_LOBSTER_KWARGS.get(name, {}))

    monkeypatch.setattr(atomate2.lobster.run, "run_lobster", mock_run_lobster)
    monkeypatch.setattr(atomate2.lobster.jobs, "run_lobster", mock_run_lobster)

    def _run(
        ref_paths: dict[str, str | Path],
        fake_run_lobster_kwargs: dict[str, dict[str, Sequence]],
    ) -> None:
        _LOBS_REF_PATHS.update(ref_paths)
        _FAKE_RUN_LOBSTER_KWARGS.update(fake_run_lobster_kwargs)

    yield _run

    monkeypatch.undo()
    _LOBS_REF_PATHS.clear()


def fake_run_lobster(
    ref_path: str | Path,
    check_lobster_inputs: Sequence[str] = _LFILES,
    check_dft_inputs: Sequence[str] = _DFT_FILES,
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
    check_dft_inputs
        A list of VASP files that need to be copied to start the LOBSTER runs.
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


def verify_inputs(ref_path: str | Path, lobsterin_settings: Sequence[str]) -> None:
    """Verify LOBSTER input files against reference settings.

    Args:
        ref_path (str | Path): Path to the reference directory containing input files.
        lobsterin_settings (Sequence[str]): A list of LOBSTER settings to check.
    """
    user = Lobsterin.from_file("lobsterin")

    # Check lobsterin
    ref = Lobsterin.from_file(Path(ref_path) / "inputs" / "lobsterin")

    for key in lobsterin_settings:
        if user.get(key) != ref.get(key):
            raise ValueError(f"lobsterin value of {key} is inconsistent!")


def copy_lobster_outputs(ref_path: str | Path) -> None:
    """Copy LOBSTER output files from the reference directory to the current directory.

    Args:
        ref_path (str | Path): Path to the reference directory containing output files.
    """
    output_path = Path(ref_path) / "outputs"
    for output_file in output_path.iterdir():
        if output_file.is_file():
            shutil.copy(output_file, ".")
