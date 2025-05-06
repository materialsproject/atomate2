from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from pymatgen.util.coord import find_in_coord_list_pbc

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core.structure import Structure


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
    with zopen(ref_path / "inputs" / "run.abi.gz", "rt", encoding="utf-8") as file:
        ref_str = file.read()
    ref = AbinitInputFile.from_string(ref_str)
    # Ignore the pseudos as the directory depends on the pseudo root directory
    # diffs = user.get_differences(ref, ignore_vars=["pseudos"])
    diffs = _get_differences_tol(user, ref, ignore_vars=["pseudos"])
    # TODO: should we still add some check on the pseudos here ?
    assert diffs == [], f"'run.abi' is different from reference: \n{diffs}"


# Adapted from check_poscar in atomate2.utils.testing.vasp.py
def check_equivalent_frac_coords(
    struct: Structure,
    struct_ref: Structure,
    atol=1e-3,
) -> None:
    """Check that the frac. coords. of two structures are equivalent (includes pbc)."""

    user_frac_coords = struct.frac_coords
    ref_frac_coords = struct_ref.frac_coords

    # In some cases, the ordering of sites can change when copying input files.
    # To account for this, we check that the sites are the same, within a tolerance,
    # while accounting for PBC.
    coord_match = [
        len(find_in_coord_list_pbc(ref_frac_coords, coord, atol=atol)) > 0
        for coord in user_frac_coords
    ]
    assert all(coord_match), (
        f"The two structures have different frac. coords: \
        {user_frac_coords} vs. {ref_frac_coords}."
    )


def check_equivalent_znucl_typat(
    znucl_a: list | np.ndarray,
    znucl_b: list | np.ndarray,
    typat_a: list | np.ndarray,
    typat_b: list | np.ndarray,
) -> None:
    """Check that the elements and their number of atoms are equivalent."""

    sorted_znucl_a = sorted(znucl_a, reverse=True)
    sorted_znucl_b = sorted(znucl_b, reverse=True)
    assert sorted_znucl_a == sorted_znucl_b, (
        f"The elements are different: {znucl_a} vs. {znucl_b}"
    )

    count_sorted_znucl_a = [
        list(typat_a).count(list(znucl_a).index(s) + 1) for s in sorted_znucl_a
    ]
    count_sorted_znucl_b = [
        list(typat_b).count(list(znucl_b).index(s) + 1) for s in sorted_znucl_b
    ]
    assert count_sorted_znucl_a == count_sorted_znucl_b, (
        f"The number of same elements is different: \
        {count_sorted_znucl_a} vs. {count_sorted_znucl_b}"
    )


def check_abinit_input_json(ref_path: str | Path):
    from abipy.abio.inputs import AbinitInput
    from monty.serialization import loadfn

    user = loadfn("abinit_input.json")
    assert isinstance(user, AbinitInput)
    user_abivars = user.structure.to_abivars()

    ref = loadfn(ref_path / "inputs" / "abinit_input.json.gz")
    ref_abivars = ref.structure.to_abivars()

    check_equivalent_frac_coords(user.structure, ref.structure)
    check_equivalent_znucl_typat(
        user_abivars["znucl"],
        ref_abivars["znucl"],
        user_abivars["typat"],
        ref_abivars["typat"],
    )

    for k, user_v in user_abivars.items():
        if k in ["xred", "znucl", "typat"]:
            continue
        assert k in ref_abivars, f"{k = } is not a key of the reference input."
        ref_v = ref_abivars[k]
        if isinstance(user_v, str):
            assert user_v == ref_v, f"{k = }-->{user_v = } versus {ref_v = }"
        else:
            assert np.allclose(user_v, ref_v), f"{k = }-->{user_v = } versus {ref_v = }"
    assert user.runlevel == ref.runlevel, f"{user.runlevel = } versus {ref.runlevel = }"


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


# Patch to allow for a tolerance in the comparison of the ABINIT input variables
# TODO: remove once new version of Abipy is released
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
            val1 = self_dataset_dict[k]
            val2 = other_dataset_dict[k]
            matched = False
            if isinstance(val1, str):
                if val1.endswith(" Ha"):
                    val1 = val1.replace(" Ha", "")
                if val1.count(".") <= 1 and val1.replace(".", "").isdecimal():
                    val1 = float(val1)

            if isinstance(val2, str):
                if val2.endswith(" Ha"):
                    val2 = val2.replace(" Ha", "")
                if val2.count(".") <= 1 and val2.replace(".", "").isdecimal():
                    val2 = float(val2)

            if isinstance(val1, float):
                matched = pytest.approx(val1, rel=rtol, abs=atol) == val2
            else:
                matched = self_dataset_dict[k] == other_dataset_dict[k]

            if not matched:
                diffs.append(
                    f"The variable '{k}' is different in the two files:\n"
                    f" - this file:  '{self_dataset_dict[k]}'\n"
                    f" - other file: '{other_dataset_dict[k]}'"
                )
    return diffs
