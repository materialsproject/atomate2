import gzip
import shutil
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "copy_kwargs,files",
    [
        (
            {"additional_cp2k_files": ("electron_density",)},
            ("CP2K-ELECTRON_DENSITY-1_0.cube", "cp2k.out", "cp2k.inp"),
        ),
        (
            {"additional_cp2k_files": ("wfn", "restart")},
            ("CP2K-WFN-1_0.wfn", "cp2k.out", "cp2k.inp"),
        ),
        (
            {"additional_cp2k_files": None},
            ("cp2k.out", "cp2k.inp"),
        ),
    ],
)
def test_copy_cp2k_outputs_static(
    tmp_dir: Path, copy_kwargs, files: tuple[str, ...]
) -> None:
    from atomate2.cp2k.files import copy_cp2k_outputs

    # Create test directory with necessary files
    test_dir = Path("test_outputs")
    test_dir.mkdir()

    # Create minimal cp2k.out file with required content
    with open(test_dir / "cp2k.out", "w") as file:
        file.write("CP2K| Output file names:\n")
        file.write(" CP2K-ELECTRON_DENSITY-1_0.cube\n")
        file.write(" CP2K-WFN-1_0.wfn\n")

    # Create other necessary files
    (test_dir / "cp2k.inp").touch()
    (test_dir / "CP2K-ELECTRON_DENSITY-1_0.cube").touch()
    (test_dir / "CP2K-WFN-1_0.wfn").touch()

    # Test copying with the parametrized arguments
    copy_cp2k_outputs(src_dir=test_dir, **copy_kwargs)

    # Check that expected files exist
    for file in files:
        assert Path(file).exists()


@pytest.mark.parametrize(
    "directory, expected_extension",
    [
        ("Si_band_structure/static/outputs", ""),
        ("Si_double_relax/relax_1/outputs", ""),
    ],
)
def test_get_largest_relax_extension(
    tmp_path: Path, cp2k_test_dir: Path, directory: str, expected_extension: str
) -> None:
    from atomate2.cp2k.files import get_largest_relax_extension

    path = cp2k_test_dir / directory
    extension = get_largest_relax_extension(directory=path)
    assert extension == expected_extension

    # Create test files with relax extensions
    Path(f"{tmp_path}/test.relax1").touch()
    Path(f"{tmp_path}/test.relax2").touch()
    Path(f"{tmp_path}/test.relax3").touch()

    extension = get_largest_relax_extension(directory=tmp_path)
    assert extension == ".relax3"


def test_write_cp2k_input_set(tmp_dir: Path) -> None:
    from pymatgen.core import Lattice, Structure

    from atomate2.cp2k.files import write_cp2k_input_set
    from atomate2.cp2k.sets.base import Cp2kInputGenerator

    # Create a simple structure for testing
    lattice = Lattice.cubic(5.43)  # Create a cubic lattice with a=5.43
    structure = Structure(lattice, ["Si"], [[0, 0, 0]])

    class DummyInputGenerator(Cp2kInputGenerator):
        def get_input_set(self, structure, prev_dir=None, optional_files=None):
            class DummyInputSet:
                def __init__(self):
                    self.cp2k_input = {"GLOBAL": {"PROJECT": "test"}}

                def write_input(self, directory, **kwargs):
                    with open(Path(directory) / "cp2k.inp", "w") as f:
                        f.write("test")

            return DummyInputSet()

    # Test basic writing
    write_cp2k_input_set(structure, DummyInputGenerator())
    assert Path("cp2k.inp").exists()
    with open("cp2k.inp") as f:
        assert f.read() == "test"


def test_cleanup_cp2k_outputs(tmp_dir: Path) -> None:
    from atomate2.cp2k.files import cleanup_cp2k_outputs

    # Create some test files
    files = ["test.bak", "test.wfn", "test.bak2", "keep.txt"]
    for file in files:
        Path(file).touch()

    # Test cleanup
    cleanup_cp2k_outputs(".", file_patterns=("*.bak*",))

    # Check that only non-bak files remain
    remaining_files = list(Path(".").glob("*"))
    assert len(remaining_files) == 2
    assert Path("test.wfn") in remaining_files
    assert Path("keep.txt") in remaining_files


def test_copy_cp2k_outputs_with_gzipped_files(tmp_dir: Path) -> None:
    from atomate2.cp2k.files import copy_cp2k_outputs

    # Create test directory structure
    test_dir = Path("test_outputs")
    test_dir.mkdir()

    # Create and gzip some test files
    files = ["cp2k.out", "cp2k.inp", "CP2K-WFN-1_0.wfn"]
    for filename in files:
        with open(test_dir / filename, "w") as file:
            file.write("test content")
        with (
            open(test_dir / filename, "rb") as f_in,
            gzip.open(test_dir / f"{filename}.gz", "wb") as f_out,
        ):
            shutil.copyfileobj(f_in, f_out)
        Path(test_dir / filename).unlink()  # Remove original file

    # Test copying and unzipping
    copy_cp2k_outputs(src_dir=test_dir, additional_cp2k_files=["wfn"])

    # Check that files were copied and unzipped
    for filename in files:
        assert Path(filename).exists()
        assert not Path(f"{filename}.gz").exists()
        with open(filename) as file:
            assert file.read() == "test content"
