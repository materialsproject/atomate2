import pytest
from pymatgen.core import Molecule, Structure
import os
import atomate2.lammps.run
from atomate2.lammps.sets.base import BaseLammpsSet
from pathlib import Path
import logging
from typing import TYPE_CHECKING, Literal
from collections.abc import Sequence
from pymatgen.io.lammps.inputs import LammpsInputFile
import jobflow

logger = logging.getLogger(__name__)
_VAL_SETTINGS = ('units', 'atom_style', 'dimension', 'boundary', 'pair_style', 'thermo', 'dump', 'timestep', 'run', 'minimize', 'fix')
_REF_PATHS = {}
_FAKE_RUN_LAMMPS_KWARGS = {}

@pytest.fixture(scope="session")
def ref_path():
    from pathlib import Path

    module_dir = Path(__file__).resolve().parents[1]
    test_dir = module_dir / "test_data/lammps/"
    return test_dir.resolve()
    

@pytest.fixture
def test_si_structure() -> Structure:
    return Structure(
    lattice=[[0, 0, 2.73], [2.73, 0, 0], [0, 2.73, 0]],
    species=["Si", "Si"],
    coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
)
    
@pytest.fixture
def test_si_force_field(ref_path) -> dict:
    return {'pair_style': 'tersoff',
            'pair_coeff': f'* * {os.path.normpath(os.path.join(ref_path, "Si.tersoff"))}',
            'species': ['Si']}

@pytest.fixture
def test_h2o_molecule() -> Molecule:
    return Molecule(
    species=["H", "O", "H"],
    coords=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
)


@pytest.fixture
def mock_lammps(monkeypatch, ref_path):
    """
    This fixture allows one to mock (fake) running lammps.

    To use the fixture successfully, the following steps must be followed:
    1. "mock_lammps" should be included as an argument to any test that would like to use
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

    def mock_run_lammps(*args, **kwargs):
        from jobflow import CURRENT_JOB

        name = CURRENT_JOB.job.name
        ref_dir = ref_path / _REF_PATHS[name]
        fake_run_lammps(ref_dir, **_FAKE_RUN_LAMMPS_KWARGS.get(name, {}))

    get_input_set_orig = BaseLammpsSet.get_input_set

    def mock_get_input_set(self, *args, **kwargs):
        return get_input_set_orig(self, *args, **kwargs)

    monkeypatch.setattr(atomate2.lammps.run, "run_lammps", mock_run_lammps)
    monkeypatch.setattr(atomate2.lammps.jobs.base, "run_lammps", mock_run_lammps)
    monkeypatch.setattr(BaseLammpsSet, "get_input_set", mock_get_input_set)

    def _run(ref_paths, fake_run_lammps_kwargs=None):
        if fake_run_lammps_kwargs is None:
            fake_run_lammps_kwargs = {}

        _REF_PATHS.update(ref_paths)
        _FAKE_RUN_LAMMPS_KWARGS.update(fake_run_lammps_kwargs)

    yield _run

    monkeypatch.undo()
    _REF_PATHS.clear()
    _FAKE_RUN_LAMMPS_KWARGS.clear()


def fake_run_lammps(
    ref_path: str | Path,
    input_settings: Sequence[str] = _VAL_SETTINGS,
    check_inputs: Sequence[Literal["in.lammps"]] = ("in.lammps",),
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
        A list of aims input files to check. Supported options are "in.lammps"
    clear_inputs
        Whether to clear input files before copying in the reference lammps outputs.
    """
    logger.info("Running fake lammps.")

    ref_path = Path(ref_path)
    
    if "in.lammps" in check_inputs:
        check_lammps_in(ref_path, input_settings=input_settings)

    logger.info("Verified inputs successfully")

    if clear_inputs:
        clear_lammps_inputs()

    copy_lammps_outputs(ref_path)

    # pretend to run aims by copying pre-generated outputs from reference dir
    logger.info("Generated fake lammps outputs")


def check_lammps_in(ref_path: Path, 
                    input_settings: Sequence[str] = None,
                    ):
    ref_input_path = ref_path / "inputs/in.lammps"
    ref_input = LammpsInputFile.from_file(ref_input_path, ignore_comments=True)
    user_input = LammpsInputFile.from_file("in.lammps", ignore_comments=True)
    
    if input_settings:
        for setting in input_settings:
            if ref_input.contains_command(setting) and user_input.contains_command(setting):
                assert ref_input.get_args(setting) == user_input.get_args(setting), f"{user_input.get_args(setting)} != {ref_input.get_args(setting)}"
    
def clear_lammps_inputs():
    for file in ("in.lammps", "system.data"):
        if Path(file).exists():
            Path(file).unlink()
    logger.info("Cleared lammps inputs")


def copy_lammps_outputs(ref_path: str | Path):
    import shutil

    output_path = ref_path / "outputs"
    for output_file in output_path.iterdir():
        if output_file.is_file():
            shutil.copy(output_file, ".")
