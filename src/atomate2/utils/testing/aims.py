from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pymatgen.io.aims.sets.base import AimsInputGenerator
from pymatgen.io.aims.inputs import AimsControlIn

import atomate2.aims.jobs.base
import atomate2.aims.run
from atomate2.common.files import gunzip_files

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Any, Final

from jobflow import CURRENT_JOB
from monty.io import zopen
from monty.os.path import zpath as monty_zpath
from pymatgen.io.aims.sets import AimsInputSet
from pymatgen.io.aims.sets.base import AimsInputGenerator
from pymatgen.util.coord import find_in_coord_list_pbc

import atomate2.aims.jobs.base
import atomate2.aims.run

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from pytest import MonkeyPatch

logger = logging.getLogger("atomate2")


_VFILES: Final = ("control.in")
_REF_PATHS: dict[str, str | Path] = {}
_FAKE_RUN_AIMS_KWARGS: dict[str, dict] = {}

def zpath(path: str | Path) -> Path:
    """Return the path of a zip file.

    Returns an existing (zipped or unzipped) file path given the unzipped
    version. If no path exists, returns the unmodified path.
    """
    return Path(monty_zpath(str(path)))


def monkeypatch_aims(
    monkeypatch: MonkeyPatch, ref_path: Path
) -> Generator[Callable[[Any, Any], Any], None, None]:
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

    def mock_run_aims(*args, **kwargs) -> None:
        name = CURRENT_JOB.job.name
        try:
            ref_dir = ref_path / _REF_PATHS[name]
        except KeyError:
            raise ValueError(
                f"no reference directory found for job {name!r}; "
                f"reference paths received={_REF_PATHS}"
            ) from None
        fake_run_aims(ref_dir, **_FAKE_RUN_AIMS_KWARGS.get(name, {}))

    get_input_set_orig = AimsInputGenerator.get_input_set

    def mock_get_input_set(self, *args, **kwargs) -> AimsInputSet:
        return get_input_set_orig(self, *args, **kwargs)

    monkeypatch.setattr(atomate2.aims.run, "run_aims", mock_run_aims)
    monkeypatch.setattr(atomate2.aims.jobs.base, "run_aims", mock_run_aims)
    monkeypatch.setattr(AimsInputGenerator, "get_input_set", mock_get_input_set)

    def _run(ref_paths: dict, fake_run_aims_kwargs: dict | None = None) -> None:
        _REF_PATHS.update(ref_paths)
        _FAKE_RUN_AIMS_KWARGS.update(fake_run_aims_kwargs or {})

    yield _run

    monkeypatch.undo()
    _REF_PATHS.clear()
    _FAKE_RUN_AIMS_KWARGS.clear()


def fake_run_aims(
    ref_path: str | Path,
    input_settings: Sequence[str] | None = None,
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
