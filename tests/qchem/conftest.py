from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import pytest
from jobflow import CURRENT_JOB
from pymatgen.core import Molecule
from pymatgen.io.qchem.inputs import QCInput
from pytest import MonkeyPatch

import atomate2.qchem.jobs.base
import atomate2.qchem.jobs.core
import atomate2.qchem.run
from atomate2.qchem.sets.base import QCInputGenerator

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence


logger = logging.getLogger("atomate2")

_QFILES: Final = "mol.qin.gz"
_REF_PATHS: dict[str, str | Path] = {}
_FAKE_RUN_QCHEM_KWARGS: dict[str, dict] = {}


@pytest.fixture
def h2o_molecule():
    return Molecule(
        coords=[
            [0.0000, 0.0000, 0.12124],
            [-0.78304, -0.00000, -0.48495],
            [0.78304, -0.00000, -0.48495],
        ],
        species=["O", "H", "H"],
    )


@pytest.fixture(scope="session")
def qchem_test_dir(test_dir):
    return test_dir / "qchem"


@pytest.fixture
def mock_qchem(
    monkeypatch: MonkeyPatch, qchem_test_dir: Path
) -> Generator[Callable[[Any, Any], Any], None, None]:
    """
    This fixture allows one to mock (fake) running qchem.

    It works by monkeypatching (replacing) calls to run_qchem and
    QCInputSet.write_inputs with versions that will work when the
    Qchem executables are not present.

    The primary idea is that instead of running QChem to generate the output files,
    reference files will be copied into the directory instead. As we do not want to
    test whether QChem is giving the correct output rather that the calculation inputs
    are generated correctly and that the outputs are parsed properly, this should be
    sufficient for our needs.

    To use the fixture successfully, the following steps must be followed:
    1. "mock_qchem" should be included as an argument to any test that would like to use
       its functionally.
    2. For each job in your workflow, you should prepare a reference directory
       containing two folders "inputs" (containing the reference input files expected
       to be produced by write_qchem_input_set) and "outputs" (containing the expected
       output files to be produced by run_qchem). These files should reside in a
       subdirectory of "tests/test_data/qchem".
    3. Create a dictionary mapping each job name to its reference directory. Note that
       you should supply the reference directory relative to the "tests/test_data/qchem"
       folder. For example, if your calculation has one job named "single_point"
       and the reference files are present in
       "tests/test_data/qchem/single_point", the dictionary
       would look like: ``{"single_point": "single_point"}``.
    4. Optional: create a dictionary mapping each job name to
       custom keyword arguments that will be supplied to fake_run_qchem.
       This way you can configure which rem settings are expected for each job.
       For example, if your calculation has one job named "single_point" and
       you wish to validate that "BASIS" is set correctly in the qin,
       your dictionary would look like ``{"single_point": {"rem": {"BASIS": "6-31G"}}``.
    5. Inside the test function, call `mock_qchem(ref_paths, fake_qchem_kwargs)`, where
       ref_paths is the dictionary created in step 3 and fake_qchem_kwargs is the
       dictionary created in step 4.
    6. Run your qchem job after calling `mock_qchem`.

    For examples, see the tests in tests/qchem/makers/core.py.
    """

    # print(f"qchem_test directory is {qchem_test_dir}")
    def mock_run_qchem(*args, **kwargs):
        name = CURRENT_JOB.job.name
        try:
            ref_path = qchem_test_dir / _REF_PATHS[name]
        except KeyError:
            raise ValueError(
                f"no reference directory found for job {name!r}; "
                f"reference paths received={_REF_PATHS}"
            ) from None

        fake_run_qchem(ref_path, **_FAKE_RUN_QCHEM_KWARGS.get(name, {}))

    get_input_set_orig = QCInputGenerator.get_input_set

    def mock_get_input_set(self, *args, **kwargs):
        return get_input_set_orig(self, *args, **kwargs)

    monkeypatch.setattr(atomate2.qchem.run, "run_qchem", mock_run_qchem)
    monkeypatch.setattr(atomate2.qchem.jobs.base, "run_qchem", mock_run_qchem)
    monkeypatch.setattr(QCInputGenerator, "get_input_set", mock_get_input_set)
    # monkeypatch.setattr(QCInputGenerator, "get_nelect", mock_get_nelect)

    def _run(ref_paths, fake_run_qchem_kwargs=None):
        _REF_PATHS.update(ref_paths)
        _FAKE_RUN_QCHEM_KWARGS.update(fake_run_qchem_kwargs or {})

    yield _run

    monkeypatch.undo()
    _REF_PATHS.clear()
    _FAKE_RUN_QCHEM_KWARGS.clear()


def fake_run_qchem(
    ref_path: Path,
    input_settings: Sequence[str] = None,
    input_exclude: Sequence[str] = None,
    check_inputs: Sequence[Literal["qin"]] = _QFILES,
    clear_inputs: bool = True,
):
    """
    Emulate running QChem and validate QChem input files.

    Parameters
    ----------
    ref_path
        Path to reference directory with QChem input files in the folder named 'inputs'
        and output files in the folder named 'outputs'.
    input_settings
        A list of input settings to check. Defaults to None which checks all settings.
        Empty list or tuple means no settings will be checked.
    input_exclude
        A list of input settings to exclude from checking. Defaults to None, meaning
        no settings will be excluded.
    check_inputs
        A list of qchem input files to check. In case of qchem, it is "qin".
    clear_inputs
        Whether to clear input files before copying in the reference QChem outputs.
    """
    logger.info("Running fake QChem.")

    if "mol.qin.gz" in check_inputs:
        check_qin(ref_path, input_settings, input_exclude)

    # This is useful to check if the WAVECAR has been copied
    logger.info("Verified inputs successfully")

    if clear_inputs:
        clear_qchem_inputs()

    copy_qchem_outputs(ref_path)

    # pretend to run VASP by copying pre-generated outputs from reference dir
    logger.info("Generated fake qchem outputs")


def check_qin(
    ref_path: Path, qin_settings: Sequence[str], qin_exclude: Sequence[str]
) -> None:
    # user_qin = QCInput.from_file("mol.qin.gz")
    ref_qin_path = ref_path / "inputs" / "mol.qin.gz"
    ref_qin = QCInput.from_file(ref_qin_path)
    script_directory = Path(__file__).resolve().parent
    # print(f"The job name is {job_name}")
    # defaults = {"sym_ignore": True, "symmetry": False, "xc_grid": 3}
    job_name = ref_path.stem
    if job_name == "water_single_point":
        user_qin_path = script_directory / "sp.qin.gz"
    elif job_name == "water_optimization":
        user_qin_path = script_directory / "opt.qin.gz"
    elif job_name == "water_frequency":
        user_qin_path = script_directory / "freq.qin.gz"
    else:
        user_qin_path = Path("mol.qin")

    user_qin = QCInput.from_file(user_qin_path)

    keys_to_check = (
        set(user_qin.as_dict()) if qin_settings is None else set(qin_settings)
    ) - set(qin_exclude or [])
    user_dict = user_qin.as_dict()
    ref_dict = ref_qin.as_dict()
    for key in keys_to_check:
        user_val = user_dict[key]
        ref_val = ref_dict[key]
        if user_val != ref_val:
            raise ValueError(
                f"\n\nQCInput value of {key} is inconsistent: expected {ref_val}, "
                f"got {user_val} \nin ref file {ref_qin_path}"
            )


def clear_qchem_inputs():
    for qchem_file in ("mol.qin.gz", "mol.qin.orig.gz"):
        if Path(qchem_file).exists():
            Path(qchem_file).unlink()
    logger.info("Cleared qchem inputs")


def copy_qchem_outputs(ref_path: Path):
    output_path = ref_path / "outputs"
    for output_file in output_path.iterdir():
        if output_file.is_file():
            shutil.copy(output_file, ".")
