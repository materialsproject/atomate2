import gzip
import shutil

import pytest
from pymatgen.core import Lattice, Structure

import atomate2.espresso.jobs
from atomate2 import SETTINGS


@pytest.fixture
def espresso_test_dir(test_dir):
    return test_dir / "espresso"


@pytest.fixture
def mock_espresso(monkeypatch, espresso_test_dir):
    """Replace pw.x and ph.x by a copy of the outputs of real runs.

    The reference runs used Quantum ESPRESSO 7.6 with the PBE pseudopotentials
    Si.pbe-rrkj.UPF and C.pbe-rrkjus.UPF of the Quantum ESPRESSO distribution.
    Each written input file must equal the reference one, apart from the
    directory of the pseudopotentials.
    """
    monkeypatch.setattr(
        SETTINGS, "ESPRESSO_PSEUDO_DIR", str(espresso_test_dir / "pseudo")
    )

    def _mock(name: str) -> None:
        ref_dir = espresso_test_dir / name

        def fake_run_espresso(command: str, input_file: str, output_file: str):
            commands = {
                "pw.in": SETTINGS.ESPRESSO_PW_CMD,
                "ph.in": SETTINGS.ESPRESSO_PH_CMD,
            }
            assert command == commands[input_file]
            with (
                open(input_file) as file,
                gzip.open(ref_dir / f"{input_file}.gz", "rt") as ref_file,
            ):
                lines = [line for line in file if "pseudo_dir" not in line]
                ref_lines = [line for line in ref_file if "pseudo_dir" not in line]
            assert lines == ref_lines
            with (
                gzip.open(ref_dir / f"{output_file}.gz", "rb") as ref_file,
                open(output_file, "wb") as file,
            ):
                shutil.copyfileobj(ref_file, file)

        monkeypatch.setattr(atomate2.espresso.jobs, "run_espresso", fake_run_espresso)

    return _mock


@pytest.fixture
def si_primitive():
    return Structure(
        Lattice.from_parameters(3.867, 3.867, 3.867, 60, 60, 60),
        ["Si", "Si"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )
