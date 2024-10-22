from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import pytest
from jobflow import CURRENT_JOB
from monty.io import zopen
from monty.os.path import zpath as monty_zpath
from pymatgen.io.vasp import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.vasp.sets import VaspInputSet
from pymatgen.util.coord import find_in_coord_list_pbc
from pytest import MonkeyPatch

import atomate2.vasp.jobs.base
import atomate2.vasp.jobs.defect
import atomate2.vasp.run
from atomate2.vasp.sets.base import VaspInputGenerator

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence


logger = logging.getLogger("atomate2")

_VFILES: Final = ("incar", "kpoints", "potcar", "poscar")
_REF_PATHS: dict[str, str | Path] = {}
_FAKE_RUN_VASP_KWARGS: dict[str, dict] = {}


def zpath(path: str | Path) -> Path:
    return Path(monty_zpath(str(path)))


@pytest.fixture(scope="session")
def vasp_test_dir(test_dir):
    return test_dir / "vasp"


@pytest.fixture(scope="session")
def lobster_test_dir(test_dir):
    return test_dir / "lobster"


@pytest.fixture
def mock_vasp(
    monkeypatch: MonkeyPatch, vasp_test_dir: Path
) -> Generator[Callable[[Any, Any], Any], None, None]:
    """
    This fixture allows one to mock (fake) running VASP.

    It works by monkeypatching (replacing) calls to run_vasp and
    VaspInputSet.write_inputs with versions that will work when the vasp executables or
    POTCAR files are not present.

    The primary idea is that instead of running VASP to generate the output files,
    reference files will be copied into the directory instead. As we do not want to
    test whether VASP is giving the correct output rather that the calculation inputs
    are generated correctly and that the outputs are parsed properly, this should be
    sufficient for our needs. Another potential issue is that the POTCAR files
    distributed with VASP are not present on the testing server due to licensing
    constraints. Accordingly, VaspInputSet.write_inputs will fail unless the
    "potcar_spec" option is set to True, in which case a POTCAR.spec file will be
    written instead. This fixture solves both of these issues.

    To use the fixture successfully, the following steps must be followed:
    1. "mock_vasp" should be included as an argument to any test that would like to use
       its functionally.
    2. For each job in your workflow, you should prepare a reference directory
       containing two folders "inputs" (containing the reference input files expected
       to be produced by write_vasp_input_set) and "outputs" (containing the expected
       output files to be produced by run_vasp). These files should reside in a
       subdirectory of "tests/test_data/vasp".
    3. Create a dictionary mapping each job name to its reference directory. Note that
       you should supply the reference directory relative to the "tests/test_data/vasp"
       folder. For example, if your calculation has one job named "static" and the
       reference files are present in "tests/test_data/vasp/Si_static", the dictionary
       would look like: ``{"static": "Si_static"}``.
    4. Optional: create a dictionary mapping each job name to custom keyword arguments
       that will be supplied to fake_run_vasp. This way you can configure which incar
       settings are expected for each job. For example, if your calculation has one job
       named "static" and you wish to validate that "NSW" is set correctly in the INCAR,
       your dictionary would look like ``{"static": {"incar_settings": {"NSW": 0}}``.
    5. Inside the test function, call `mock_vasp(ref_paths, fake_vasp_kwargs)`, where
       ref_paths is the dictionary created in step 3 and fake_vasp_kwargs is the
       dictionary created in step 4.
    6. Run your vasp job after calling `mock_vasp`.

    For examples, see the tests in tests/vasp/makers/core.py.
    """
    yield from _mock_vasp(monkeypatch, vasp_test_dir)


def _mock_vasp(
    monkeypatch: MonkeyPatch, vasp_test_dir: Path
) -> Generator[Callable[[Any, Any], Any], None, None]:
    """
    Isolated version of the mock_vasp fixture that can be used in other contexts.
    """

    def mock_run_vasp(*args, **kwargs):
        name = CURRENT_JOB.job.name
        try:
            ref_path = vasp_test_dir / _REF_PATHS[name]
        except KeyError:
            raise ValueError(
                f"no reference directory found for job {name!r}; "
                f"reference paths received={_REF_PATHS}"
            ) from None
        fake_run_vasp(ref_path, **_FAKE_RUN_VASP_KWARGS.get(name, {}))

    get_input_set_orig = VaspInputGenerator.get_input_set

    def mock_get_input_set(self, *args, **kwargs):
        kwargs["potcar_spec"] = True
        return get_input_set_orig(self, *args, **kwargs)

    def mock_nelect(*_, **__):
        return 12

    monkeypatch.setattr(atomate2.vasp.run, "run_vasp", mock_run_vasp)
    monkeypatch.setattr(atomate2.vasp.jobs.base, "run_vasp", mock_run_vasp)
    monkeypatch.setattr(atomate2.vasp.jobs.defect, "run_vasp", mock_run_vasp)
    monkeypatch.setattr(VaspInputSet, "get_input_set", mock_get_input_set)
    monkeypatch.setattr(VaspInputSet, "nelect", mock_nelect)

    def _run(ref_paths, fake_run_vasp_kwargs=None):
        _REF_PATHS.update(ref_paths)
        _FAKE_RUN_VASP_KWARGS.update(fake_run_vasp_kwargs or {})

    yield _run

    monkeypatch.undo()
    _REF_PATHS.clear()
    _FAKE_RUN_VASP_KWARGS.clear()


def fake_run_vasp(
    ref_path: Path,
    incar_settings: Sequence[str] = None,
    incar_exclude: Sequence[str] = None,
    check_inputs: Sequence[Literal["incar", "kpoints", "poscar", "potcar"]] = _VFILES,
    clear_inputs: bool = True,
):
    """
    Emulate running VASP and validate VASP input files.

    Parameters
    ----------
    ref_path
        Path to reference directory with VASP input files in the folder named 'inputs'
        and output files in the folder named 'outputs'.
    incar_settings
        A list of INCAR settings to check. Defaults to None which checks all settings.
        Empty list or tuple means no settings will be checked.
    incar_exclude
        A list of INCAR settings to exclude from checking. Defaults to None, meaning
        no settings will be excluded.
    check_inputs
        A list of vasp input files to check. Supported options are "incar", "kpoints",
        "poscar", "potcar", "wavecar".
    clear_inputs
        Whether to clear input files before copying in the reference VASP outputs.
    """
    logger.info("Running fake VASP.")

    if "incar" in check_inputs:
        check_incar(ref_path, incar_settings, incar_exclude)

    if "kpoints" in check_inputs:
        check_kpoints(ref_path)

    if "poscar" in check_inputs:
        check_poscar(ref_path)

    if "potcar" in check_inputs:
        check_potcar(ref_path)

    # This is useful to check if the WAVECAR has been copied
    if "wavecar" in check_inputs and not Path("WAVECAR").exists():
        raise ValueError("WAVECAR was not correctly copied")

    logger.info("Verified inputs successfully")

    if clear_inputs:
        clear_vasp_inputs()

    copy_vasp_outputs(ref_path)

    # pretend to run VASP by copying pre-generated outputs from reference dir
    logger.info("Generated fake vasp outputs")


def check_incar(
    ref_path: Path, incar_settings: Sequence[str], incar_exclude: Sequence[str]
) -> None:
    user_incar = Incar.from_file(zpath("INCAR"))
    ref_incar_path = zpath(ref_path / "inputs" / "INCAR")
    ref_incar = Incar.from_file(ref_incar_path)
    defaults = {"ISPIN": 1, "ISMEAR": 1, "SIGMA": 0.2}

    keys_to_check = (
        set(user_incar) if incar_settings is None else set(incar_settings)
    ) - set(incar_exclude or [])
    for key in keys_to_check:
        user_val = user_incar.get(key, defaults.get(key))
        ref_val = ref_incar.get(key, defaults.get(key))
        if user_val != ref_val:
            raise ValueError(
                f"\n\nINCAR value of {key} is inconsistent: expected {ref_val}, "
                f"got {user_val} \nin ref file {ref_incar_path}"
            )


def check_kpoints(ref_path: Path):
    user_kpoints_exists = (user_kpt_path := zpath("KPOINTS")).exists()
    ref_kpoints_exists = (
        ref_kpt_path := zpath(ref_path / "inputs" / "KPOINTS")
    ).exists()

    if user_kpoints_exists and not ref_kpoints_exists:
        raise ValueError(
            "atomate2 generated a KPOINTS file but the reference calculation is using "
            "KSPACING"
        )
    if not user_kpoints_exists and ref_kpoints_exists:
        raise ValueError(
            "atomate2 is using KSPACING but the reference calculation is using "
            "a KPOINTS file"
        )
    if user_kpoints_exists and ref_kpoints_exists:
        user_kpts = Kpoints.from_file(user_kpt_path)
        ref_kpts = Kpoints.from_file(ref_kpt_path)
        if user_kpts.style != ref_kpts.style or user_kpts.num_kpts != ref_kpts.num_kpts:
            raise ValueError(
                f"\n\nKPOINTS files are inconsistent: {user_kpts.style} != "
                f"{ref_kpts.style} or {user_kpts.num_kpts} != {ref_kpts.num_kpts}\nin "
                f"ref file {ref_kpt_path}"
            )
    else:
        # check k-spacing
        user_incar = Incar.from_file(zpath("INCAR"))
        ref_incar_path = zpath(ref_path / "inputs" / "INCAR")
        ref_incar = Incar.from_file(ref_incar_path)

        user_ksp, ref_ksp = user_incar.get("KSPACING"), ref_incar.get("KSPACING")
        if user_ksp != ref_ksp:
            raise ValueError(
                f"\n\nKSPACING is inconsistent: expected {ref_ksp}, got {user_ksp} "
                f"\nin ref file {ref_incar_path}"
            )


def check_poscar(ref_path: Path):
    user_poscar_path = zpath("POSCAR")
    ref_poscar_path = zpath(ref_path / "inputs" / "POSCAR")

    user_poscar = Poscar.from_file(user_poscar_path)
    ref_poscar = Poscar.from_file(ref_poscar_path)

    user_frac_coords = user_poscar.structure.frac_coords
    ref_frac_coords = ref_poscar.structure.frac_coords

    # In some cases, the ordering of sites can change when copying input files.
    # To account for this, we check that the sites are the same, within a tolerance,
    # while accounting for PBC.
    coord_match = [
        len(find_in_coord_list_pbc(ref_frac_coords, coord, atol=1e-3)) > 0
        for coord in user_frac_coords
    ]
    if (
        user_poscar.natoms != ref_poscar.natoms
        or user_poscar.site_symbols != ref_poscar.site_symbols
        or not all(coord_match)
    ):
        raise ValueError(
            f"POSCAR files are inconsistent\n\n{ref_poscar_path!s}\n{ref_poscar}"
            f"\n\n{user_poscar_path!s}\n{user_poscar}"
        )


def check_potcar(ref_path: Path):
    potcars = {"reference": None, "user": None}
    paths = {"reference": ref_path / "inputs", "user": Path(".")}
    for mode, path in paths.items():
        if (potcar_path := zpath(path / "POTCAR")).exists():
            potcars[mode] = Potcar.from_file(potcar_path).symbols
        elif (potcar_path := zpath(path / "POTCAR.spec")).exists():
            with zopen(potcar_path, "rt") as f:
                potcars[mode] = f.read().strip().split("\n")
        else:
            raise FileNotFoundError(f"no {mode} POTCAR or POTCAR.spec file found")

    if potcars["reference"] != potcars["user"]:
        raise ValueError(
            "POTCAR files are inconsistent: "
            f"{potcars['reference']} != {potcars['user']}"
        )


def clear_vasp_inputs():
    for vasp_file in (
        "INCAR",
        "KPOINTS",
        "POSCAR",
        "POTCAR",
        "CHGCAR",
        "OUTCAR",
        "vasprun.xml",
        "CONTCAR",
    ):
        if (file_path := zpath(vasp_file)).exists():
            file_path.unlink()
    logger.info("Cleared vasp inputs")


def copy_vasp_outputs(ref_path: Path):
    output_path = ref_path / "outputs"
    for output_file in output_path.iterdir():
        if output_file.is_file():
            shutil.copy(output_file, ".")
