from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger("atomate2")

_REF_PATHS = {}
_ABINIT_FILES = ("run.abi", "abinit_input.json")
_FAKE_RUN_ABINIT_KWARGS = {}
_MRGDDB_FILES = "mrgddb.in"
_FAKE_RUN_MRGDDB_KWARGS = {}
_ANADDB_FILES = ("anaddb.in", "anaddb_input.json")
_FAKE_RUN_ANADDB_KWARGS = {}


@pytest.fixture(scope="session")
def abinit_test_dir(test_dir):
    return test_dir / "abinit"


@pytest.fixture(scope="session", autouse=True)
def load_pseudos(abinit_test_dir):
    import abipy.flowtk.psrepos

    abipy.flowtk.psrepos.REPOS_ROOT = str(abinit_test_dir / "pseudos")


@pytest.fixture(scope="session")
def abinit_integration_tests(pytestconfig):
    return pytestconfig.getoption("abinit_integration")


@pytest.fixture
def mock_abinit(mocker, abinit_test_dir, abinit_integration_tests):
    """
    This fixture allows one to mock running ABINIT.

    It works by monkeypatching (replacing) calls to run_abinit.

    The primary idea is that instead of running ABINIT to generate the output files,
    reference files will be copied into the directory instead.
    """
    import atomate2.abinit.files
    import atomate2.abinit.jobs.base
    import atomate2.abinit.run

    # Wrap the write_abinit_input_set so that we can check inputs after calling it
    def wrapped_write_abinit_input_set(*args, **kwargs):
        from jobflow import CURRENT_JOB

        name = CURRENT_JOB.job.name
        index = CURRENT_JOB.job.index
        ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]

        atomate2.abinit.files.write_abinit_input_set(*args, **kwargs)
        check_abinit_inputs(ref_path)

    mocker.patch.object(
        atomate2.abinit.jobs.base,
        "write_abinit_input_set",
        wrapped_write_abinit_input_set,
    )

    if not abinit_integration_tests:
        # Mock abinit run (i.e. this will copy reference files)
        def mock_run_abinit(wall_time=None, start_time=None):
            from jobflow import CURRENT_JOB

            name = CURRENT_JOB.job.name
            index = CURRENT_JOB.job.index
            ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]
            check_abinit_inputs(ref_path)
            fake_run_abinit(ref_path)

        mocker.patch.object(atomate2.abinit.run, "run_abinit", mock_run_abinit)
        mocker.patch.object(atomate2.abinit.jobs.base, "run_abinit", mock_run_abinit)

        def _run(ref_paths, fake_run_abinit_kwargs=None):
            if fake_run_abinit_kwargs is None:
                fake_run_abinit_kwargs = {}
            _REF_PATHS.update(ref_paths)
            _FAKE_RUN_ABINIT_KWARGS.update(fake_run_abinit_kwargs)

        yield _run

    mocker.stopall()
    _REF_PATHS.clear()
    _FAKE_RUN_ABINIT_KWARGS.clear()


def fake_run_abinit(ref_path: str | Path):
    """
    Emulate running ABINIT.

    Parameters
    ----------
    ref_path
        Path to reference directory with ABINIT input files in the folder named 'inputs'
        and output files in the folder named 'outputs'.
    """
    logger.info("Running fake ABINIT.")

    ref_path = Path(ref_path)

    copy_abinit_outputs(ref_path)

    # pretend to run ABINIT by copying pre-generated outputs from reference dir
    logger.info("Generated fake ABINIT outputs")


def check_abinit_inputs(
    ref_path: str | Path,
    check_inputs: Sequence[Literal["run.abi"]] = _ABINIT_FILES,
):
    ref_path = Path(ref_path)

    if "run.abi" in check_inputs:
        check_run_abi(ref_path)

    if "abinit_input.json" in check_inputs:
        check_abinit_input_json(ref_path)

    logger.info("Verified inputs successfully")


def check_run_abi(ref_path: str | Path):
    from abipy.abio.abivars import AbinitInputFile
    from monty.io import zopen

    user = AbinitInputFile.from_file("run.abi")
    assert user.ndtset == 1, f"'run.abi' has multiple datasets (ndtset={user.ndtset})."
    with zopen(ref_path / "inputs" / "run.abi.gz") as file:
        ref_str = file.read()
    ref = AbinitInputFile.from_string(ref_str.decode("utf-8"))
    # Ignore the pseudos as the directory depends on the pseudo root directory
    diffs = user.get_differences(ref, ignore_vars=["pseudos", "pp_dirpath"])
    # TODO: should we still add some check on the pseudos here ?
    assert diffs == [], "'run.abi' is different from reference."


def check_abinit_input_json(ref_path: str | Path):
    from abipy.abio.inputs import AbinitInput
    from monty.serialization import loadfn

    user = loadfn("abinit_input.json")
    assert isinstance(user, AbinitInput)
    ref = loadfn(ref_path / "inputs" / "abinit_input.json.gz")
    assert user.structure == ref.structure
    assert user.runlevel == ref.runlevel


def clear_abinit_files():
    for abinit_file in ("run.abo",):
        if Path(abinit_file).exists():
            Path(abinit_file).unlink()
    logger.info("Cleared abinit files.")


@pytest.fixture
def mock_mrgddb(mocker, abinit_test_dir, abinit_integration_tests):
    """
    This fixture allows one to mock running Mrgddb.

    It works by monkeypatching (replacing) calls to run_mrgddb.

    The primary idea is that instead of running Mrgddb to generate the output files,
    reference files will be copied into the directory instead.
    """
    import atomate2.abinit.files
    import atomate2.abinit.jobs.mrgddb
    import atomate2.abinit.run

    # Wrap the write_mrgddb_input_set so that we can check inputs after calling it
    def wrapped_write_mrgddb_input_set(*args, **kwargs):
        from jobflow import CURRENT_JOB

        name = CURRENT_JOB.job.name
        index = CURRENT_JOB.job.index
        ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]

        atomate2.abinit.files.write_mrgddb_input_set(*args, **kwargs)
        check_mrgddb_inputs(ref_path)

    mocker.patch.object(
        atomate2.abinit.jobs.mrgddb,
        "write_mrgddb_input_set",
        wrapped_write_mrgddb_input_set,
    )

    if not abinit_integration_tests:
        # Mock abinit run (i.e. this will copy reference files)
        def mock_run_mrgddb(wall_time=None, start_time=None):
            from jobflow import CURRENT_JOB

            name = CURRENT_JOB.job.name
            index = CURRENT_JOB.job.index
            ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]
            check_mrgddb_inputs(ref_path)
            fake_run_abinit(ref_path)

        mocker.patch.object(atomate2.abinit.run, "run_mrgddb", mock_run_mrgddb)
        mocker.patch.object(atomate2.abinit.jobs.mrgddb, "run_mrgddb", mock_run_mrgddb)

        def _run(ref_paths, fake_run_mrgddb_kwargs=None):
            if fake_run_mrgddb_kwargs is None:
                fake_run_mrgddb_kwargs = {}
            _REF_PATHS.update(ref_paths)
            _FAKE_RUN_MRGDDB_KWARGS.update(fake_run_mrgddb_kwargs)

        yield _run

    mocker.stopall()
    _REF_PATHS.clear()
    _FAKE_RUN_MRGDDB_KWARGS.clear()


def check_mrgddb_inputs(
    ref_path: str | Path,
    check_inputs: Sequence[Literal["mrgddb.in"]] = _MRGDDB_FILES,
):
    ref_path = Path(ref_path)

    if "mrgddb.in" in check_inputs:
        from monty.io import zopen

        with open("mrgddb.in") as file:
            str_in = file.readlines()
        str_in.pop(1)

        with zopen(ref_path / "inputs" / "mrgddb.in.gz", "r") as file:
            ref_str = file.readlines()
        ref_str.pop(1)

        assert (
            str_in[1] == ref_str[1].decode()
        ), "'mrgddb.in' is different from reference."

        str_in.pop(1)
        ref_str.pop(1)

        for i, _ in enumerate(str_in):
            str_in[i] = str_in[i][
                -16:
            ]  # Only keep the "outdata/out_DDB\n" from the path
            ref_str[i] = ref_str[i][
                -16:
            ].decode()  # Only keep the "outdata/out_DDB\n" from the path

        assert str_in == ref_str, "'mrgddb.in' is different from reference."

    logger.info("Verified inputs successfully")


@pytest.fixture
def mock_anaddb(mocker, abinit_test_dir, abinit_integration_tests):
    """
    This fixture allows one to mock running Anaddb.

    It works by monkeypatching (replacing) calls to run_anaddb.

    The primary idea is that instead of running Anaddb to generate the output files,
    reference files will be copied into the directory instead.
    """
    import atomate2.abinit.files
    import atomate2.abinit.jobs.anaddb
    import atomate2.abinit.run

    # Wrap the write_anaddb_input_set so that we can check inputs after calling it
    def wrapped_write_anaddb_input_set(*args, **kwargs):
        from jobflow import CURRENT_JOB

        name = CURRENT_JOB.job.name
        index = CURRENT_JOB.job.index
        ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]

        atomate2.abinit.files.write_anaddb_input_set(*args, **kwargs)
        check_anaddb_inputs(ref_path)

    mocker.patch.object(
        atomate2.abinit.jobs.anaddb,
        "write_anaddb_input_set",
        wrapped_write_anaddb_input_set,
    )

    if not abinit_integration_tests:
        # Mock anaddb run (i.e. this will copy reference files)
        def mock_run_anaddb(wall_time=None, start_time=None):
            from jobflow import CURRENT_JOB

            name = CURRENT_JOB.job.name
            index = CURRENT_JOB.job.index
            ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]
            check_anaddb_inputs(ref_path)
            fake_run_abinit(ref_path)

        mocker.patch.object(atomate2.abinit.run, "run_anaddb", mock_run_anaddb)
        mocker.patch.object(atomate2.abinit.jobs.anaddb, "run_anaddb", mock_run_anaddb)

        def _run(ref_paths, fake_run_anaddb_kwargs=None):
            if fake_run_anaddb_kwargs is None:
                fake_run_anaddb_kwargs = {}
            _REF_PATHS.update(ref_paths)
            _FAKE_RUN_ANADDB_KWARGS.update(fake_run_anaddb_kwargs)

        yield _run

    mocker.stopall()
    _REF_PATHS.clear()
    _FAKE_RUN_ANADDB_KWARGS.clear()


def check_anaddb_inputs(
    ref_path: str | Path,
    check_inputs: Sequence[Literal["anaddb.in"]] = _ANADDB_FILES,
):
    ref_path = Path(ref_path)

    if "anaddb.in" in check_inputs:
        check_anaddb_in(ref_path)

    if "anaddb_input.json" in check_inputs:
        check_anaddb_input_json(ref_path)

    logger.info("Verified inputs successfully")


def convert_file_to_dict(file_path):
    import gzip

    result_dict = {}

    if file_path.endswith(".gz"):
        file_opener = gzip.open
        mode = "rt"  # read text mode for gzip
    else:
        file_opener = open
        mode = "r"

    with file_opener(file_path, mode) as file:
        for line in file:
            key, value = line.split()
            try:
                result_dict[key] = int(value)  # Assuming values are integers
            except ValueError:
                result_dict[key] = str(value)  # Fall back to string if not an integer
    return result_dict


def check_anaddb_in(ref_path: str | Path):
    user = convert_file_to_dict("anaddb.in")
    ref = convert_file_to_dict(str(ref_path / "inputs" / "anaddb.in.gz"))
    assert user == ref, "'anaddb.in' is different from reference."


def check_anaddb_input_json(ref_path: str | Path):
    from abipy.abio.inputs import AnaddbInput
    from monty.serialization import loadfn

    user = loadfn("anaddb_input.json")
    assert isinstance(user, AnaddbInput)
    ref = loadfn(ref_path / "inputs" / "anaddb_input.json.gz")
    assert user.structure == ref.structure
    assert user == ref


def copy_abinit_outputs(ref_path: str | Path):
    import shutil

    from monty.shutil import decompress_file

    ref_path = Path(ref_path)
    output_path = ref_path / "outputs"
    for output_file in output_path.iterdir():
        if output_file.is_file():
            shutil.copy(output_file, ".")
            decompress_file(output_file.name)
    for data_dir in ("indata", "outdata", "tmpdata"):
        ref_data_dir = output_path / data_dir
        for file in ref_data_dir.iterdir():
            if file.is_file():
                shutil.copy(file, data_dir)
                decompress_file(str(Path(data_dir, file.name)))
