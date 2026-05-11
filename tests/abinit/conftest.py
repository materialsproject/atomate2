from __future__ import annotations

import gzip
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from abipy.abio.abivars import is_anaddb_var
from pymatgen.util.coord import find_in_coord_list_pbc

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core.structure import Structure
    from pytest import TempPathFactory


logger = logging.getLogger("atomate2")

_REF_PATHS = {}
_ABINIT_FILES = ("run.abi", "abinit_input.json")
_FAKE_RUN_ABINIT_KWARGS = {}
_MRGDDB_FILES = "mrgddb.in"
_FAKE_RUN_MRGDDB_KWARGS = {}
_MRGDV_FILES = "mrgdv.in"
_FAKE_RUN_MRGDV_KWARGS = {}
_ANADDB_FILES = ("anaddb.in", "anaddb_input.json")
_FAKE_RUN_ANADDB_KWARGS = {}

# Do this here to prevent issues with threaded CI runners
# In abipy, it's possible to have thread collisions in
# making this directory because `exist_ok = False` there
_ABINIT_PATH = Path("~/.abinit/abipy").expanduser()
if not _ABINIT_PATH.is_dir():
    _ABINIT_PATH.mkdir(exist_ok=True, parents=True)


@pytest.fixture(scope="session")
def abinit_test_dir(test_dir: Path) -> Path:
    """
    Get the ABINIT test directory.

    Parameters
    ----------
    test_dir
        The root test directory.

    Returns
    -------
    Path
        The ABINIT test directory path.
    """
    return test_dir / "abinit"


@pytest.fixture(scope="session", autouse=True)
def load_pseudos(abinit_test_dir: Path, tmp_path_factory: TempPathFactory):
    """
    Configure the pseudopotential repository root directory for tests.

    Parameters
    ----------
    abinit_test_dir
        The ABINIT test directory containing the pseudos subdirectory.
    """
    # Create a session-scoped temp directory
    temp_dir = tmp_path_factory.mktemp("pseudos")
    temp_dir = Path(temp_dir)

    # Path to the compressed pseudos
    compressed_pseudos_dir = abinit_test_dir / "pseudos"

    # Iterate over all .gz files in subdirectories
    for gz_file in compressed_pseudos_dir.glob("**/*.gz"):
        # Recreate the relative path in the temp directory
        relative_path = gz_file.relative_to(compressed_pseudos_dir)
        target_dir = temp_dir / relative_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        # Uncompress the file to the target directory
        target_file = target_dir / relative_path.stem
        with gzip.open(gz_file, "rb") as f_in, open(target_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Set the REPOS_ROOT to the temp directory
    import abipy.flowtk.psrepos

    abipy.flowtk.psrepos.REPOS_ROOT = str(temp_dir)
    # Cleanup is automatic for tmp_path_factory


@pytest.fixture(scope="session", autouse=True)
def load_manager(abinit_test_dir: Path) -> None:
    """
    Configure the ABINIT task manager configuration directory for tests.

    Parameters
    ----------
    abinit_test_dir
        The ABINIT test directory containing the abipy subdirectory.
    """
    import abipy.flowtk.tasks

    abipy.flowtk.tasks.TaskManager.USER_CONFIG_DIR = str(abinit_test_dir / "abipy")


@pytest.fixture(scope="session")
def abinit_integration_tests(pytestconfig: pytest.Config) -> bool:
    """
    Get the ABINIT integration test flag from pytest configuration.

    Parameters
    ----------
    pytestconfig
        The pytest configuration object.

    Returns
    -------
    bool
        True if ABINIT integration tests are enabled, False otherwise.
    """
    return pytestconfig.getoption("abinit_integration")


@pytest.fixture
def mock_abinit(monkeypatch, abinit_test_dir, abinit_integration_tests):
    """
    This fixture allows one to mock running ABINIT.

    It works by monkeypatching (replacing) calls to run_abinit.

    The primary idea is that instead of running ABINIT to generate the output files,
    reference files will be copied into the directory instead.
    """
    import atomate2.abinit.files
    import atomate2.abinit.jobs.base
    import atomate2.abinit.run

    # Wrap write_abinit_input_set to check inputs after writing
    def wrapped_write_abinit_input_set(*args, **kwargs) -> None:
        from jobflow import CURRENT_JOB

        name = CURRENT_JOB.job.name
        index = CURRENT_JOB.job.index
        ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]

        atomate2.abinit.files.write_abinit_input_set(*args, **kwargs)
        check_abinit_inputs(ref_path)

    monkeypatch.setattr(
        atomate2.abinit.jobs.base,
        "write_abinit_input_set",
        wrapped_write_abinit_input_set,
    )

    if not abinit_integration_tests:
        # Mock ABINIT run by copying reference output files
        def mock_run_abinit(
            wall_time: float | None = None, start_time: float | None = None
        ) -> None:
            from jobflow import CURRENT_JOB

            name = CURRENT_JOB.job.name
            index = CURRENT_JOB.job.index
            ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]
            check_abinit_inputs(ref_path)
            fake_run_abinit(ref_path)

        monkeypatch.setattr(atomate2.abinit.run, "run_abinit", mock_run_abinit)
        monkeypatch.setattr(atomate2.abinit.jobs.base, "run_abinit", mock_run_abinit)

        def _run(ref_paths: dict, fake_run_abinit_kwargs: dict | None = None) -> None:
            if fake_run_abinit_kwargs is None:
                fake_run_abinit_kwargs = {}
            _REF_PATHS.update(ref_paths)
            _FAKE_RUN_ABINIT_KWARGS.update(fake_run_abinit_kwargs)

        yield _run

    monkeypatch.undo()
    _REF_PATHS.clear()
    _FAKE_RUN_ABINIT_KWARGS.clear()


def fake_run_abinit(ref_path: str | Path) -> None:
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

    # Pretend to run ABINIT by copying pre-generated outputs from reference directory
    logger.info("Generated fake ABINIT outputs")


def check_abinit_inputs(
    ref_path: str | Path,
    check_inputs: Sequence[Literal["run.abi"]] = _ABINIT_FILES,
) -> None:
    """
    Verify ABINIT input files against reference inputs.

    Parameters
    ----------
    ref_path
        Path to reference directory containing input files.
    check_inputs
        Sequence of input file names to check.
    """
    ref_path = Path(ref_path)

    if "run.abi" in check_inputs:
        check_run_abi(ref_path)

    if "abinit_input.json" in check_inputs:
        check_abinit_input_json(ref_path)

    logger.info("Verified inputs successfully")


def check_run_abi(ref_path: str | Path) -> None:
    """
    Verify run.abi input file against reference.

    Parameters
    ----------
    ref_path
        Path to reference directory containing the reference run.abi file.
    """
    from abipy.abio.abivars import AbinitInputFile
    from monty.io import zopen

    user = AbinitInputFile.from_file("run.abi")
    assert user.ndtset == 1, f"'run.abi' has multiple datasets (ndtset={user.ndtset})."
    with zopen(ref_path / "inputs" / "run.abi.gz", "rt", encoding="utf-8") as file:
        ref_str = file.read()
    ref = AbinitInputFile.from_string(ref_str)
    # Ignore pseudos and pp_dirpath as they depend on the pseudo root directory
    diffs = _get_differences_tol(user, ref, ignore_vars=["pseudos", "pp_dirpath"])
    # TODO: Should we still add some check on the pseudos here?
    assert diffs == [], f"'run.abi' is different from reference: \n{diffs}"


# Adapted from check_poscar in atomate2.utils.testing.vasp.py
def check_equivalent_frac_coords(
    struct: Structure,
    struct_ref: Structure,
    atol: float = 1e-3,
) -> None:
    """
    Check that the fractional coordinates of two structures are equivalent.

    This function verifies that all sites in the structure match the reference
    structure's sites within a tolerance, accounting for periodic boundary conditions
    (PBC). Site ordering is not required to match.

    Parameters
    ----------
    struct
        The structure to check.
    struct_ref
        The reference structure.
    atol
        Absolute tolerance for coordinate comparison.
    """
    user_frac_coords = struct.frac_coords
    ref_frac_coords = struct_ref.frac_coords

    # In some cases, the ordering of sites can change when copying input files.
    # To account for this, we check that the sites are the same, within a tolerance,
    # while accounting for periodic boundary conditions.
    coord_match = [
        len(find_in_coord_list_pbc(ref_frac_coords, coord, atol=atol)) > 0
        for coord in user_frac_coords
    ]
    assert all(coord_match), (
        f"The two structures have different frac. coords: "
        f"{user_frac_coords} vs. {ref_frac_coords}."
    )


def check_equivalent_znucl_typat(
    znucl_a: list | np.ndarray,
    znucl_b: list | np.ndarray,
    typat_a: list | np.ndarray,
    typat_b: list | np.ndarray,
) -> None:
    """
    Check that the elements and their counts are equivalent between two structures.

    Parameters
    ----------
    znucl_a
        Atomic numbers from the first structure.
    znucl_b
        Atomic numbers from the second structure.
    typat_a
        Atom types from the first structure.
    typat_b
        Atom types from the second structure.
    """
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
        f"The number of same elements is different: "
        f"{count_sorted_znucl_a} vs. {count_sorted_znucl_b}"
    )


def check_abinit_input_json(ref_path: str | Path) -> None:
    """
    Verify abinit_input.json against reference.

    Parameters
    ----------
    ref_path
        Path to reference directory containing the reference abinit_input.json file.
    """
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


def clear_abinit_files() -> None:
    """Remove ABINIT output files from the current directory."""
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

    # Wrap write_mrgddb_input_set to check inputs after writing
    def wrapped_write_mrgddb_input_set(*args, **kwargs) -> None:
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
        # Mock mrgddb run by copying reference output files
        def mock_run_mrgddb(
            wall_time: float | None = None, start_time: float | None = None
        ) -> None:
            from jobflow import CURRENT_JOB

            name = CURRENT_JOB.job.name
            index = CURRENT_JOB.job.index
            ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]
            check_mrgddb_inputs(ref_path)
            fake_run_abinit(ref_path)

        mocker.patch.object(atomate2.abinit.run, "run_mrgddb", mock_run_mrgddb)
        mocker.patch.object(atomate2.abinit.jobs.mrgddb, "run_mrgddb", mock_run_mrgddb)

        def _run(ref_paths: dict, fake_run_mrgddb_kwargs: dict | None = None) -> None:
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
) -> None:
    """
    Verify mrgddb input files against reference inputs.

    Parameters
    ----------
    ref_path
        Path to reference directory containing input files.
    check_inputs
        Sequence of input file names to check.
    """
    ref_path = Path(ref_path)

    if "mrgddb.in" in check_inputs:
        from monty.io import zopen

        with open("mrgddb.in") as file:
            str_in = file.readlines()
        str_in.pop(1)  # Remove second line (DDB output path)

        with zopen(
            ref_path / "inputs" / "mrgddb.in.gz", "rt", encoding="utf-8"
        ) as file:
            ref_str = file.readlines()
        ref_str.pop(1)  # Remove second line (DDB output path)

        assert str_in[1] == ref_str[1], "'mrgddb.in' is different from reference."

        str_in.pop(1)
        ref_str.pop(1)

        # Only keep the "outdata/out_DDB\n" from the path for comparison
        for i, _ in enumerate(str_in):
            str_in[i] = str_in[i][-16:]
            ref_str[i] = ref_str[i][-16:]

        assert str_in == ref_str, "'mrgddb.in' is different from reference."

    logger.info("Verified inputs successfully")


@pytest.fixture
def mock_mrgdvdb(mocker, abinit_test_dir, abinit_integration_tests):
    """
    This fixture allows one to mock running Mrgdvdb.

    It works by monkeypatching (replacing) calls to run_mrgdv.

    The primary idea is that instead of running Mrgdvdb to generate the output files,
    reference files will be copied into the directory instead.
    """
    import atomate2.abinit.files
    import atomate2.abinit.jobs.mrgdv
    import atomate2.abinit.run

    # Wrap write_mrgdv_input_set to check inputs after writing
    def wrapped_write_mrgdv_input_set(*args, **kwargs) -> None:
        from jobflow import CURRENT_JOB

        name = CURRENT_JOB.job.name
        index = CURRENT_JOB.job.index
        ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]

        atomate2.abinit.files.write_mrgdv_input_set(*args, **kwargs)
        check_mrgdv_inputs(ref_path)

    mocker.patch.object(
        atomate2.abinit.jobs.mrgdv,
        "write_mrgdv_input_set",
        wrapped_write_mrgdv_input_set,
    )

    if not abinit_integration_tests:
        # Mock mrgdv run by copying reference output files
        def mock_run_mrgdv(
            wall_time: float | None = None, start_time: float | None = None
        ) -> None:
            from jobflow import CURRENT_JOB

            name = CURRENT_JOB.job.name
            index = CURRENT_JOB.job.index
            ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]
            check_mrgdv_inputs(ref_path)
            fake_run_abinit(ref_path)

        mocker.patch.object(atomate2.abinit.run, "run_mrgdv", mock_run_mrgdv)
        mocker.patch.object(atomate2.abinit.jobs.mrgdv, "run_mrgdv", mock_run_mrgdv)

        def _run(ref_paths: dict, fake_run_mrgdv_kwargs: dict | None = None) -> None:
            if fake_run_mrgdv_kwargs is None:
                fake_run_mrgdv_kwargs = {}
            _REF_PATHS.update(ref_paths)
            _FAKE_RUN_MRGDV_KWARGS.update(fake_run_mrgdv_kwargs)

        yield _run

    mocker.stopall()
    _REF_PATHS.clear()
    _FAKE_RUN_MRGDV_KWARGS.clear()


def check_mrgdv_inputs(
    ref_path: str | Path,
    check_inputs: Sequence[Literal["mrgdv.in"]] = _MRGDV_FILES,
) -> None:
    """
    Verify mrgdv input files against reference inputs.

    Parameters
    ----------
    ref_path
        Path to reference directory containing input files.
    check_inputs
        Sequence of input file names to check.
    """
    ref_path = Path(ref_path)

    if "mrgdv.in" in check_inputs:
        from monty.io import zopen

        with open("mrgdv.in") as file:
            str_in = file.readlines()
        str_in.pop(1)  # Remove second line (POT output path)

        with zopen(ref_path / "inputs" / "mrgdv.in.gz", "rt", encoding="utf-8") as file:
            ref_str = file.readlines()
        ref_str.pop(1)  # Remove second line (POT output path)

        assert str_in[1] == ref_str[1], "'mrgdv.in' is different from reference."

        str_in.pop(1)
        ref_str.pop(1)

        # Only keep the "outdata/out_POT*\n" from the path for comparison
        for i, _ in enumerate(str_in):
            str_in[i] = "/".join(str_in[i].split("/")[-2:])
            ref_str[i] = "/".join(ref_str[i].split("/")[-2:])

        assert str_in == ref_str, "'mrgdv.in' is different from reference."

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

    # Wrap write_anaddb_input_set to check inputs after writing
    def wrapped_write_anaddb_input_set(*args, **kwargs) -> None:
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
        # Mock anaddb run by copying reference output files
        def mock_run_anaddb(
            wall_time: float | None = None, start_time: float | None = None
        ) -> None:
            from jobflow import CURRENT_JOB

            name = CURRENT_JOB.job.name
            index = CURRENT_JOB.job.index
            ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]
            check_anaddb_inputs(ref_path)
            fake_run_abinit(ref_path)

        mocker.patch.object(atomate2.abinit.run, "run_anaddb", mock_run_anaddb)
        mocker.patch.object(atomate2.abinit.jobs.anaddb, "run_anaddb", mock_run_anaddb)

        def _run(ref_paths: dict, fake_run_anaddb_kwargs: dict | None = None) -> None:
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
) -> None:
    """
    Verify anaddb input files against reference inputs.

    Parameters
    ----------
    ref_path
        Path to reference directory containing input files.
    check_inputs
        Sequence of input file names to check.
    """
    ref_path = Path(ref_path)

    if "anaddb.in" in check_inputs:
        check_anaddb_in(ref_path)

    if "anaddb_input.json" in check_inputs:
        check_anaddb_input_json(ref_path)

    logger.info("Verified inputs successfully")


def convert_file_to_dict(file_path: str) -> dict:
    """
    Convert an anaddb input file to a dictionary.

    Parameters
    ----------
    file_path
        Path to the anaddb input file (can be gzipped).

    Returns
    -------
    dict
        Dictionary representation of the anaddb input file.
    """
    from monty.io import zopen

    result_dict: dict = {}

    if file_path.endswith(".gz"):
        file_opener = zopen
        mode = "rt"  # Read text mode for gzip
    else:
        file_opener = open
        mode = "r"

    with file_opener(file_path, mode) as file:
        current_key = None
        for line in file:
            if "#" in line or len(line) == 1:
                continue
            sl = line.strip().split(" ", 1)
            if is_anaddb_var(sl[0]) and len(sl) > 1:
                try:
                    result_dict[sl[0]] = int(sl[1])
                except ValueError:
                    result_dict[sl[0]] = sl[1]
            elif is_anaddb_var(sl[0]) and len(sl) == 1:
                current_key = sl[0]
                result_dict[current_key] = []
            elif current_key is not None:
                result_dict[current_key].append([float(t) for t in line.split()])
    return result_dict


def check_anaddb_in(ref_path: str | Path) -> None:
    """
    Verify anaddb.in input file against reference.

    Parameters
    ----------
    ref_path
        Path to reference directory containing the reference anaddb.in file.
    """
    user = convert_file_to_dict("anaddb.in")
    ref = convert_file_to_dict(str(ref_path / "inputs" / "anaddb.in.gz"))
    assert user == ref, "'anaddb.in' is different from reference."


def check_anaddb_input_json(ref_path: str | Path) -> None:
    """
    Verify anaddb_input.json against reference.

    Parameters
    ----------
    ref_path
        Path to reference directory containing the reference anaddb_input.json file.
    """
    from abipy.abio.inputs import AnaddbInput
    from monty.serialization import loadfn

    user = loadfn("anaddb_input.json")
    assert isinstance(user, AnaddbInput)
    user_abivars = user.structure.to_abivars()

    ref = loadfn(ref_path / "inputs" / "anaddb_input.json.gz")
    ref_abivars = ref.structure.to_abivars()

    # Check structure
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

    # Check anaddb input
    user_args = dict(user.as_dict()["anaddb_args"])
    ref_args = dict(ref.as_dict()["anaddb_args"])
    for k, user_v in user_args.items():
        assert k in ref_args, f"{k = } is not a key of the reference input."
        ref_v = ref_args[k]
        if isinstance(user_v, str):
            assert user_v == ref_v, f"{k = }-->{user_v = } versus {ref_v = }"
        elif user_v is None and ref_v is None:
            continue
        else:
            assert np.allclose(user_v, ref_v), f"{k = }-->{user_v = } versus {ref_v = }"


def copy_abinit_outputs(ref_path: str | Path) -> None:
    """
    Copy ABINIT output files from reference directory to current directory.

    Parameters
    ----------
    ref_path
        Path to reference directory containing output files in the outputs subdirectory.
    """
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
        if not ref_data_dir.exists():
            # means this ref dir was empty and thus removed
            continue
        for file in ref_data_dir.iterdir():
            if file.is_file():
                shutil.copy(file, data_dir)
                decompress_file(str(Path(data_dir, file.name)))


# Patch to allow for a tolerance in the comparison of the ABINIT input variables
# TODO: Remove once new version of Abipy is released
def _get_differences_tol(
    abi1,
    abi2,
    ignore_vars: set[str] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-12,
) -> list[str]:
    """
    Get the differences between two AbinitInputFile objects with tolerance.

    This is a patched version that allows tolerance when comparing floating-point
    ABINIT input variables. This function compares two ABINIT input files and
    returns a list of differences, ignoring structure-related variables and
    optionally other specified variables.

    Parameters
    ----------
    abi1
        First AbinitInputFile object.
    abi2
        Second AbinitInputFile object to compare against.
    ignore_vars
        Set of variable names to ignore during comparison.
    rtol
        Relative tolerance for floating-point comparison.
    atol
        Absolute tolerance for floating-point comparison.

    Returns
    -------
    list[str]
        List of difference descriptions. Empty list if files are identical.
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
            self_dataset_dict.pop(k, None)
            other_dataset_dict.pop(k, None)
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
