from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from monty.os.path import zpath as monty_zpath
from pytest import MonkeyPatch

from atomate2.utils.testing.vasp import monkeypatch_vasp

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


logger = logging.getLogger("atomate2")


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
) -> Generator[Callable[[Any, Any], Any]]:
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
    yield from monkeypatch_vasp(monkeypatch, vasp_test_dir)
