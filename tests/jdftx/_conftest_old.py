from __future__ import annotations

import logging
logger = logging.getLogger(__name__)
from pathlib import Path
from typing import TYPE_CHECKING
import pytest
from monty.os.path import zpath as monty_zpath
from atomate2.jdftx.sets.base import FILE_NAMES
from atomate2.jdftx.io.jdftxinfile import JDFTXInfile

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
def run_function_jdftx(jdftx_test_dir):
    def _run(ref_paths, fake_run_jdftx_kwargs=None): 
        _REF_PATHS.update(ref_paths)
        _FAKE_RUN_JDFTX_KWARGS.update(fake_run_jdftx_kwargs or {})
        logger.info("_run passed")

    yield _run

    _REF_PATHS.clear()
    _FAKE_RUN_JDFTX_KWARGS.clear()

#@pytest.fixture
#def mock_jdftx(monkeypatch, jdftx_test_dir):
    
    # import atomate2.jdftx.jobs.base
    # import atomate2.jdftx.run
    # from atomate2.jdftx.sets.base import JdftxInputGenerator

    # monkeypatch.setattr(atomate2.jdftx.run, "run_jdftx", mock_run_jdftx)
    # monkeypatch.setattr(atomate2.jdftx.jobs.base, "run_jdftx", mock_run_jdftx)
    # monkeypatch.setattr(JdftxInputGenerator, "get_input_set", mock_get_input_set)   
@pytest.fixture
def mock_run_jdftx(monkeypatch, *args, **kwargs):
    from jobflow import CURRENT_JOB
    import atomate2.jdftx.jobs.base
    import atomate2.jdftx.run
    monkeypatch.setattr(atomate2.jdftx.run, "run_jdftx", mock_run_jdftx)
    
    logger.info("mock_run called")
    #name = CURRENT_JOB.job.name
    name = "relax"
    yield name
        # ref_path = jdftx_test_dir / _REF_PATHS[name]
        # fake_run_jdftx(ref_path, **_FAKE_RUN_JDFTX_KWARGS)
    monkeypatch.undo()


@pytest.fixture
def mock_get_input_set(monkeypatch, *args, **kwargs):
    from atomate2.jdftx.sets.base import JdftxInputGenerator
    monkeypatch.setattr(JdftxInputGenerator, "get_input_set", mock_get_input_set) 

    logger.info("mock_input called")
    get_input_set_orig = JdftxInputGenerator.get_input_set
    yield get_input_set_orig(*args, **kwargs)

    monkeypatch.undo()
        

@pytest.fixture
def check_input():
    def _check_input(ref_path, input_settings: Sequence[str] = None):
        logger.info("Checking inputs.")

        ref_input = JDFTXInfile.from_file(ref_path / "inputs" / "init.in")
        user_input = JDFTXInfile.from_file(zpath("init.in"))

        keys_to_check = (
            set(user_input) if input_settings is None else set(input_settings)
        )

        results = {}
        for key in keys_to_check:
            user_val = user_input.get(key)
            ref_val = ref_input.get(key)
            results[key] = (user_val, ref_val)
        logger.info("Checked inputs successfully.")
        return results
    return _check_input


def copy_cp2k_outputs(ref_path: str | Path):
    import shutil

    output_path = ref_path / "outputs"
    for output_file in output_path.iterdir():
        if output_file.is_file():
            shutil.copy(output_file, ".")

# def fake_run_jdftx(
#         ref_path: str | Path,
#         input_settings: Sequence[str] = None,
#         check_inputs: Sequence[Literal["init.in"]] = _JFILES,
# ) -> None:
#     logger.info("Running fake JDFTx.")
    # ref_path = Path(ref_path)

    # if "init.in" in check_inputs:
    #     results = check_input(ref_path, input_settings)
    #     for key, (user_val, ref_val) in results.items():
    #         assert user_val == ref_val, f"Mismatch for {key}: user_val={user_val}, ref_val={ref_val}"


    # logger.info("Verified inputs successfully")