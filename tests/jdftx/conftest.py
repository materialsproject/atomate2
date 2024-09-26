from __future__ import annotations

import logging
import os
logger = logging.getLogger(__name__)
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import pytest
from monty.io import zopen
from monty.os.path import zpath as monty_zpath
from contextlib import contextmanager
from atomate2.jdftx.sets.base import FILE_NAMES
from atomate2.jdftx.io.jdftxinfile import JDFTXInfile
from jobflow import CURRENT_JOB
import shutil

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("atomate2")

_JFILES = "init.in"
_REF_PATHS: dict[str, str | Path] = {}
_FAKE_RUN_JDFTX_KWARGS: dict[str, dict]  = {}

def zpath(path: str | Path) -> Path:
    return Path(monty_zpath(str(path)))

@pytest.fixture(scope="session")
def jdftx_test_dir(test_dir):
    return test_dir / "jdftx"

@pytest.fixture
def mock_cwd(monkeypatch, request):
    test_name = request.param
    print(f"test_name: {test_name}")
    mock_path = (Path(__file__).resolve().parent / f"../test_data/jdftx/{test_name}").resolve()
    monkeypatch.setattr(os, "getcwd", lambda: mock_path)


@pytest.fixture
def mock_filenames(monkeypatch):
    monkeypatch.setitem(FILE_NAMES, "in", "inputs/init.in")
    monkeypatch.setitem(FILE_NAMES, "out", "outputs/jdftx.out")

@pytest.fixture
def mock_jdftx(monkeypatch, jdftx_test_dir: Path):
    import atomate2.jdftx.jobs.base
    import atomate2.jdftx.run
    from atomate2.jdftx.sets.base import JdftxInputGenerator

    def mock_run_jdftx(*args, **kwargs):

        name = CURRENT_JOB.job.name
        print(f"name:", name)
        ref_path = jdftx_test_dir / _REF_PATHS[name]
        logger.info("mock_run called")
        fake_run_jdftx(ref_path, **_FAKE_RUN_JDFTX_KWARGS, clear_inputs=False)
        

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
    input_settings: Sequence[str] = None,
    check_inputs: Sequence[Literal["init.in"]] = _JFILES,
    clear_inputs: bool = True,
):
    logger.info("Running fake JDFTx.")
    ref_path = Path(ref_path)

    if "init.in" in check_inputs:
        results = check_input(ref_path, input_settings)
        for key, (user_val, ref_val) in results.items():
            assert user_val == ref_val, f"Mismatch for {key}: user_val={user_val}, ref_val={ref_val}"

    logger.info("Verified inputs successfully")
 
    if clear_inputs:
        clear_jdftx_inputs()

    copy_jdftx_outputs(ref_path)

def check_input(ref_path, input_settings: Sequence[str] = None):
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
    
    return results

def clear_jdftx_inputs():
    if (file_path:= zpath("init.in")).exists():
        file_path.unlink()
        print(f"Deleting file: {file_path.resolve()}") 
    logger.info("Cleared jdftx inputs")

def copy_jdftx_outputs(ref_path: Path):
    import os
    output_path = ref_path / "outputs"
    for output_file in output_path.iterdir():
        if output_file.is_file():
            shutil.copy(output_file, ".")