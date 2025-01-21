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


@pytest.fixture(scope="session")
def abinit_test_dir(test_dir):
    return test_dir / "abinit"


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
    diffs = _get_differences_tol(user, ref, ignore_vars=["pseudos"])
    # TODO: should we still add some check on the pseudos here ?
    assert diffs == [], f"'run.abi' is different from reference.\n{diffs}"


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


def _get_differences_tol(
    abi1, abi2, ignore_vars=None, rtol=1e-5, atol=1e-12
) -> list[str]:
    """
    Get the differences between this AbinitInputFile and another.
    Allow tolerance for floats.
    """
    diffs = []
    to_ignore = {
        "acell",
        "angdeg",
        "rprim",
        "ntypat",
        "natom",
        "znucl",
        "typat",
        "xred",
        "xcart",
        "xangst",
    }
    if ignore_vars is not None:
        to_ignore.update(ignore_vars)
    if abi1.ndtset != abi2.ndtset:
        diffs.append(
            f"Number of datasets in this file is {abi1.ndtset} "
            f"while other file has {abi2.ndtset} datasets."
        )
        return diffs
    for idataset, self_dataset in enumerate(abi1.datasets):
        other_dataset = abi2.datasets[idataset]
        if self_dataset.structure != other_dataset.structure:
            diffs.append("Structures are different.")
        self_dataset_dict = dict(self_dataset)
        other_dataset_dict = dict(other_dataset)
        for k in to_ignore:
            if k in self_dataset_dict:
                del self_dataset_dict[k]
            if k in other_dataset_dict:
                del other_dataset_dict[k]
        common_keys = set(self_dataset_dict.keys()).intersection(
            other_dataset_dict.keys()
        )
        self_only_keys = set(self_dataset_dict.keys()).difference(
            other_dataset_dict.keys()
        )
        other_only_keys = set(other_dataset_dict.keys()).difference(
            self_dataset_dict.keys()
        )
        if self_only_keys:
            diffs.append(
                f"The following variables are in this file but not in other: "
                f"{', '.join([str(k) for k in self_only_keys])}"
            )
        if other_only_keys:
            diffs.append(
                f"The following variables are in other file but not in this one: "
                f"{', '.join([str(k) for k in other_only_keys])}"
            )
        for k in common_keys:
            matched = False
            if isinstance(self_dataset_dict[k], float):
                matched = (
                    pytest.approx(self_dataset_dict[k], rel=rtol, abs=atol)
                    == other_dataset_dict[k]
                )
            else:
                matched = self_dataset_dict[k] == other_dataset_dict[k]

            if not matched:
                diffs.append(
                    f"The variable '{k}' is different in the two files:\n"
                    f" - this file:  '{self_dataset_dict[k]}'\n"
                    f" - other file: '{other_dataset_dict[k]}'"
                )
    return diffs
