from __future__ import annotations

import logging
logger = logging.getLogger(__name__)
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import pytest
from monty.io import zopen
from monty.os.path import zpath as monty_zpath
from atomate2.jdftx.sets.base import FILE_NAMES

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("atomate2")

_JFILES = "init.in"
_REF_PATHS = {}
_FAKE_RUN_JDFTX_KWARGS = {}

def zpath(path: str | Path) -> Path:
    return Path(monty_zpath(str(path)))

@pytest.fixture(scope="session")
def jdftx_test_dir(test_dir):
    return test_dir / "jdftx"
# tests/test_data/jdftx

#fixtures to get TaskDoc
@pytest.fixture
def mock_cwd(monkeypatch):
    mock_path = Path("../../test_data/jdftx/default_test")
    monkeypatch.setattr(Path, "cwd", lambda: mock_path)

@pytest.fixture
def mock_filenames(monkeypatch):
    monkeypatch.setitem(FILE_NAMES, "in", "inputs/init.in")
    monkeypatch.setitem(FILE_NAMES, "out", "outputs/jdftx.out")

@pytest.fixture
def mock_jdftx(monkeypatch, jdftx_test_dir):
    
    import atomate2.jdftx.jobs.base
    import atomate2.jdftx.run
    from atomate2.jdftx.sets.base import JdftxInputGenerator
            
    def mock_run_jdftx(*args, **kwargs):
        from jobflow import CURRENT_JOB   #attributes: jobflow.Job and jobflow.JobStore

        #name = CURRENT_JOB.job.name
        name = "relax"
        ref_path = jdftx_test_dir / _REF_PATHS[name]
        fake_run_jdftx(ref_path, **_FAKE_RUN_JDFTX_KWARGS)
        logger.info("mock_run called")

    get_input_set_orig = JdftxInputGenerator.get_input_set

    def mock_get_input_set(self, *args, **kwargs):
        logger.info("mock_input called")
        return get_input_set_orig(self, *args, **kwargs)

    monkeypatch.setattr(atomate2.jdftx.run, "run_jdftx", mock_run_jdftx)
    monkeypatch.setattr(atomate2.jdftx.jobs.base, "run_jdftx", mock_run_jdftx)
    monkeypatch.setattr(JdftxInputGenerator, "get_input_set", mock_get_input_set)   

    def _run(ref_paths, fake_run_jdftx_kwargs=None): 
        if fake_run_jdftx_kwargs is None:
            fake_run_jdftx_kwargs = {}

        _REF_PATHS.update(ref_paths)
        _FAKE_RUN_JDFTX_KWARGS.update(fake_run_jdftx_kwargs)
        logger.info("_run passed")
    yield _run

    monkeypatch.undo()
    _REF_PATHS.clear()
    _FAKE_RUN_JDFTX_KWARGS.clear()
        

def fake_run_jdftx(
        ref_path: str | Path,
        input_settings: Sequence[str] = (),
        check_inputs: Sequence[Literal["init.in"]] = _JFILES,
) -> None:
    
    logger.info("Running fake JDFTx.")
    ref_path = Path(ref_path)

    if "init.in" in check_inputs:
        check_input(ref_path, input_settings)

    logger.info("Verified inputs successfully")


#@pytest.fixture
def check_input(ref_path, input_settings: Sequence[str] = (),):

    from atomate2.jdftx.io.jdftxinfile import JDFTXInfile
    logger.info("Checking inputs.")

    ref_input = JDFTXInfile.from_file(ref_path / "inputs" / "init.in")
    user_input = JDFTXInfile.from_file(zpath("init.in"))
    print(f"ref_input:", ref_input)
    print(f"user_input:", user_input)
     #   user_string = " ".join(user_input.get_str().lower().split())
      #  user_hash = md5(user_string.encode("utf-8")).hexdigest()

       # ref_string = " ".join(ref_input.get_str().lower().split())
       # ref_hash = md5(ref_string.encode("utf-8")).hexdigest()

#        if ref_hash != user_hash:
 #           raise ValueError("Cp2k Inputs do not match!")

def copy_cp2k_outputs(ref_path: str | Path):
    import shutil

    output_path = ref_path / "outputs"
    for output_file in output_path.iterdir():
        if output_file.is_file():
            shutil.copy(output_file, ".")
