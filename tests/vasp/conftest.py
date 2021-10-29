import logging
from pathlib import Path
from typing import Literal, Sequence, Union

import pytest

vfiles = ("incar", "kpoints", "potcar", "poscar")

logger = logging.getLogger("atomate2")


@pytest.fixture(scope="session")
def vasp_test_dir(test_dir):
    return test_dir / "vasp"


_REF_PATHS = {}
_FAKE_RUN_VASP_KWARGS = {}


@pytest.fixture(scope="function")
def mock_vasp(monkeypatch, vasp_test_dir):
    """
    This fixture allows one to mock (fake) running VASP.

    It works by monkeypatching (replacing) calls to run_vasp and
    VaspInputSet.write_inputs with versions that will work when the vasp executables or
    POTCAR files are not present.

    The primary idea is that instead of running VASP to generate the output files,
    reference files will be copied into the directory instead. As we do not want to
    test whether VASP is giving the correct output rather that the calculation inputs
    are generated correctly and that the ouputs are parsed properly, this should be
    sufficient for our needs. An other potential issue is that the POTCAR files
    distribute with VASP are not present on the testing server due to licensing
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
    import atomate2.vasp.run
    from atomate2.vasp.sets.base import VaspInputSet

    def mock_run_vasp():
        from jobflow import CURRENT_JOB

        name = CURRENT_JOB.job.name
        ref_path = vasp_test_dir / _REF_PATHS[name]
        fake_run_vasp(ref_path, **_FAKE_RUN_VASP_KWARGS.get(name, {}))

    write_input_orig = VaspInputSet.write_input

    def mock_write_input(self, *args, **kwargs):
        kwargs["potcar_spec"] = True
        write_input_orig(self, *args, **kwargs)

    monkeypatch.setattr(atomate2.vasp.run, "run_vasp", mock_run_vasp)
    monkeypatch.setattr(VaspInputSet, "write_input", mock_write_input)

    def _run(ref_paths, fake_run_vasp_kwargs=None):
        if fake_run_vasp_kwargs is None:
            fake_run_vasp_kwargs = {}

        _REF_PATHS.update(ref_paths)
        _FAKE_RUN_VASP_KWARGS.update(fake_run_vasp_kwargs)
        print(ref_paths)

    yield _run

    monkeypatch.undo()
    _REF_PATHS.clear()
    _FAKE_RUN_VASP_KWARGS.clear()


def fake_run_vasp(
    ref_path: Union[str, Path],
    incar_settings: Sequence[str] = tuple(),
    check_inputs: Sequence[Literal["incar", "kpoints", "poscar", "potcar"]] = vfiles,
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
        A list of INCAR settings to check.
    check_inputs
        A list of vasp input files to check. Supported options are "incar", "kpoints",
        "poscar", "potcar".
    clear_inputs
        Whether to clear input files before copying in the reference VASP outputs.
    """
    logger.info("Running fake VASP.")

    ref_path = Path(ref_path)

    if "incar" in check_inputs:
        check_incar(ref_path, incar_settings)

    if "kpoints" in check_inputs:
        check_kpoints(ref_path)

    if "poscar" in check_inputs:
        check_poscar(ref_path)

    if "potcar" in check_inputs:
        check_potcar(ref_path)

    logger.info("Verified inputs successfully")

    if clear_inputs:
        clear_vasp_inputs()

    copy_vasp_outputs(ref_path)

    # pretend to run VASP by copying pre-generated outputs from reference dir
    logger.info("Generated fake vasp outputs")


def check_incar(ref_path: Union[str, Path], incar_settings: Sequence[str]):
    from pymatgen.io.vasp import Incar

    user = Incar.from_file("INCAR")
    ref = Incar.from_file(ref_path / "inputs" / "INCAR")
    defaults = {"ISPIN": 1, "ISMEAR": 1, "SIGMA": 0.2}
    for p in incar_settings:
        if user.get(p, defaults.get(p)) != ref.get(p, defaults.get(p)):
            raise ValueError(
                f"INCAR value of {p} is inconsistent: "
                f"{user.get(p, defaults.get(p))} != {ref.get(p, defaults.get(p))}"
            )


def check_kpoints(ref_path: Union[str, Path]):
    from pymatgen.io.vasp import Incar, Kpoints

    user_kpoints_exists = Path("KPOINTS").exists()
    ref_kpoints_exists = Path(ref_path / "inputs" / "KPOINTS").exists()

    if user_kpoints_exists and not ref_kpoints_exists:
        raise ValueError(
            "atomate2 generated a KPOINTS file but the reference calculation is using "
            "KSPACING"
        )
    elif not user_kpoints_exists and ref_kpoints_exists:
        raise ValueError(
            "atomate2 is using KSPACING but the reference calculation is using "
            "a KPOINTS file"
        )
    elif user_kpoints_exists and ref_kpoints_exists:
        user = Kpoints.from_file("KPOINTS")
        ref = Kpoints.from_file(ref_path / "inputs" / "KPOINTS")
        if user.style != ref.style or user.num_kpts != ref.num_kpts:
            raise ValueError("KPOINTS files are inconsistent")
    else:
        # check k-spacing
        user = Incar.from_file("INCAR")
        ref = Incar.from_file(ref_path / "inputs" / "INCAR")

        if user.get("KSPACING", None) != ref.get("KSPACING", None):
            raise ValueError(
                "KSPACING is not inconsistent: "
                f"{user.get('KSPACING', None)} != {ref.get('KSPACING', None)}"
            )


def check_poscar(ref_path: Union[str, Path]):
    import numpy as np
    from pymatgen.io.vasp import Poscar
    from pymatgen.util.coord import pbc_diff

    user = Poscar.from_file("POSCAR")
    ref = Poscar.from_file(ref_path / "inputs" / "POSCAR")

    if (
        user.natoms != ref.natoms
        or user.site_symbols != ref.site_symbols
        or np.any(
            np.abs(pbc_diff(user.structure.frac_coords, ref.structure.frac_coords))
            > 1e-3
        )
    ):
        raise ValueError("POSCAR files are inconsistent")


def check_potcar(ref_path: Union[str, Path]):
    from pymatgen.io.vasp import Potcar

    if Path(ref_path / "inputs" / "POTCAR").exists():
        ref = Potcar.from_file(ref_path / "inputs" / "POTCAR").symbols
    elif Path(ref_path / "inputs" / "POTCAR.spec").exists():
        ref = Path(ref_path / "inputs" / "POTCAR.spec").read_text().strip().split("\n")
    else:
        raise FileNotFoundError("no reference POTCAR or POTCAR.spec file found")

    if Path("POTCAR").exists():
        user = Potcar.from_file("POTCAR").symbols
    elif Path("POTCAR.spec").exists():
        user = Path("POTCAR.spec").read_text().strip().split("\n")
    else:
        raise FileNotFoundError("no POTCAR or POTCAR.spec file found")

    if user != ref:
        raise ValueError(f"POTCAR files are inconsistent: {user} != {ref}")


def clear_vasp_inputs():
    for vasp_file in (
        "INCAR",
        "KPOINTS",
        "POSCAR",
        "POTCAR",
        "CHGCAR",
        "OUTCAR",
        "vasprun.xml",
    ):
        if Path(vasp_file).exists():
            Path(vasp_file).unlink()
    logger.info("Cleared vasp inputs")


def copy_vasp_outputs(ref_path: Union[str, Path]):
    import shutil

    output_path = ref_path / "outputs"
    for output_file in output_path.iterdir():
        if output_file.is_file():
            shutil.copy(output_file, ".")
