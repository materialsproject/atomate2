from __future__ import annotations

import logging
from hashlib import md5
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("atomate2")

_VFILES = "cp2k.inp"
_REF_PATHS = {}
_FAKE_RUN_CP2K_KWARGS = {}


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch, test_dir):
    settings = {
        "PMG_CP2K_DATA_DIR": Path(test_dir / "cp2k/data"),
        "PMG_DEFAULT_CP2K_FUNCTIONAL": "PBE",
        "PMG_DEFAULT_CP2K_BASIS_TYPE": "DZVP-MOLOPT",
        "PMG_DEFAULT_CP2K_AUX_BASIS_TYPE": "pFIT",
    }
    monkeypatch.setattr("pymatgen.core.SETTINGS", settings)


@pytest.fixture(scope="session")
def basis_and_potential():
    return {
        "basis_and_potential": {
            "Si": {
                "basis": "DZVP-MOLOPT-SR-GTH",
                "potential": "GTH-PBE-q4",
                "aux_basis": "pFIT3",
            }
        }
    }


@pytest.fixture(scope="session")
def cp2k_test_dir(test_dir):
    return test_dir / "cp2k"


@pytest.fixture(scope="session")
def cp2k_test_inputs(test_dir):
    return Path(test_dir / "cp2k").glob("*/inputs")


@pytest.fixture(scope="session")
def cp2k_test_outputs(test_dir):
    return Path(test_dir / "cp2k").glob("*/outputs")


@pytest.fixture
def mock_cp2k(monkeypatch, cp2k_test_dir):
    """
    This fixture allows one to mock (fake) running CP2K.

    It works by monkeypatching (replacing) calls to run_cp2k and
    Cp2kInputSet.write_inputs with versions that will work when the cp2k executables or
    POTCAR files are not present.

    The primary idea is that instead of running CP2K to generate the output files,
    reference files will be copied into the directory instead. As we do not want to
    test whether CP2K is giving the correct output rather that the calculation inputs
    are generated correctly and that the outputs are parsed properly, this should be
    sufficient for our needs.

    To use the fixture successfully, the following steps must be followed:
    1. "mock_cp2k" should be included as an argument to any test that would like to use
       its functionally.
    2. For each job in your workflow, you should prepare a reference directory
       containing two folders "inputs" (containing the reference input files expected
       to be produced by write_cp2k_input_set) and "outputs" (containing the expected
       output files to be produced by run_cp2k). These files should reside in a
       subdirectory of "tests/test_data/cp2k".
    3. Create a dictionary mapping each job name to its reference directory. Note that
       you should supply the reference directory relative to the "tests/test_data/cp2k"
       folder. For example, if your calculation has one job named "static" and the
       reference files are present in "tests/test_data/cp2k/Si_static", the dictionary
       would look like: ``{"static": "Si_static"}``.
    4. Optional: create a dictionary mapping each job name to custom keyword arguments
       that will be supplied to fake_run_cp2k. This way you can configure which incar
       settings are expected for each job. For example, if your calculation has one job
       named "static" and you wish to validate that "NSW" is set correctly in the INCAR,
       your dictionary would look like ``{"static": {"incar_settings": {"NSW": 0}}``.
    5. Inside the test function, call `mock_cp2k(ref_paths, fake_cp2k_kwargs)`, where
       ref_paths is the dictionary created in step 3 and fake_cp2k_kwargs is the
       dictionary created in step 4.
    6. Run your cp2k job after calling `mock_cp2k`.

    For examples, see the tests in tests/cp2k/jobs/core.py.
    """
    import atomate2.cp2k.jobs.base
    import atomate2.cp2k.run
    from atomate2.cp2k.sets.base import Cp2kInputGenerator

    def mock_run_cp2k(*args, **kwargs):
        from jobflow import CURRENT_JOB

        name = CURRENT_JOB.job.name
        ref_path = cp2k_test_dir / _REF_PATHS[name]
        fake_run_cp2k(ref_path, **_FAKE_RUN_CP2K_KWARGS.get(name, {}))

    get_input_set_orig = Cp2kInputGenerator.get_input_set

    def mock_get_input_set(self, *args, **kwargs):
        return get_input_set_orig(self, *args, **kwargs)

    monkeypatch.setattr(atomate2.cp2k.run, "run_cp2k", mock_run_cp2k)
    monkeypatch.setattr(atomate2.cp2k.jobs.base, "run_cp2k", mock_run_cp2k)
    monkeypatch.setattr(Cp2kInputGenerator, "get_input_set", mock_get_input_set)

    def _run(ref_paths, fake_run_cp2k_kwargs=None):
        if fake_run_cp2k_kwargs is None:
            fake_run_cp2k_kwargs = {}

        _REF_PATHS.update(ref_paths)
        _FAKE_RUN_CP2K_KWARGS.update(fake_run_cp2k_kwargs)

    yield _run

    monkeypatch.undo()
    _REF_PATHS.clear()
    _FAKE_RUN_CP2K_KWARGS.clear()


def fake_run_cp2k(
    ref_path: str | Path,
    input_settings: Sequence[str] = (),
    check_inputs: Sequence[Literal["cp2k.inp"]] = _VFILES,
    clear_inputs: bool = True,
) -> None:
    """
    Emulate running CP2K and validate CP2K input files.

    Parameters
    ----------
    ref_path
        Path to reference directory with CP2K input files in the folder named 'inputs'
        and output files in the folder named 'outputs'.
    input_settings
        A list of input settings to check.
    check_inputs
        A list of cp2k input files to check. Supported options are "cp2k.inp"
    clear_inputs
        Whether to clear input files before copying in the reference CP2K outputs.
    """
    logger.info("Running fake CP2K.")

    ref_path = Path(ref_path)

    if "incar" in check_inputs:
        check_input(ref_path, input_settings)

    logger.info("Verified inputs successfully")

    if clear_inputs:
        clear_cp2k_inputs()

    copy_cp2k_outputs(ref_path)

    # pretend to run cp2k by copying pre-generated outputs from reference dir
    logger.info("Generated fake cp2k outputs")


@pytest.fixture
def check_input():
    from pymatgen.io.cp2k.inputs import Cp2kInput

    def _check_input(ref_path, user_input: Cp2kInput):
        ref_input = Cp2kInput.from_file(ref_path / "inputs" / "cp2k.inp")
        user_input.verbosity(verbosity=False)
        ref_input.verbosity(verbosity=False)
        user_string = " ".join(user_input.get_str().lower().split())
        user_hash = md5(user_string.encode("utf-8")).hexdigest()

        ref_string = " ".join(ref_input.get_str().lower().split())
        ref_hash = md5(ref_string.encode("utf-8")).hexdigest()

        if ref_hash != user_hash:
            raise ValueError("Cp2k Inputs do not match!")

    return _check_input


def clear_cp2k_inputs():
    for cp2k_file in ("cp2k.inp", "cp2k.out"):
        if Path(cp2k_file).exists():
            Path(cp2k_file).unlink()
    logger.info("Cleared cp2k inputs")


def copy_cp2k_outputs(ref_path: str | Path):
    import shutil

    output_path = ref_path / "outputs"
    for output_file in output_path.iterdir():
        if output_file.is_file():
            shutil.copy(output_file, ".")
