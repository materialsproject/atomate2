"""Shared pytest fixtures for atomate2siesta tests."""

import pytest
from pathlib import Path
from pymatgen.core import Structure, Lattice


@pytest.fixture
def si_structure():
    """
    Standard Si structure for testing (diamond structure).

    Returns
    -------
    Structure
        Si structure with 2 atoms
    """
    lattice = Lattice.cubic(5.43)
    return Structure(
        lattice,
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


@pytest.fixture
def al_structure():
    """
    Standard Al structure for testing (FCC).

    Returns
    -------
    Structure
        Al structure with 1 atom
    """
    lattice = Lattice.cubic(4.05)
    return Structure(
        lattice,
        ["Al"],
        [[0.0, 0.0, 0.0]],
    )


@pytest.fixture
def graphene_structure():
    """
    Graphene monolayer structure for testing.

    Returns
    -------
    Structure
        Graphene structure with 2 atoms
    """
    a = 2.46
    c = 20.0  # Large vacuum in z
    lattice = Lattice([[a, 0, 0], [-a / 2, a * 3**0.5 / 2, 0], [0, 0, c]])
    return Structure(
        lattice,
        ["C", "C"],
        [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]],
    )


@pytest.fixture
def h2o_structure():
    """
    Water molecule for testing molecular calculations.

    Returns
    -------
    Structure
        H2O molecule in a box
    """
    from pymatgen.core import Molecule

    mol = Molecule(
        ["O", "H", "H"], [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
    )

    # Put in a box for periodic boundary conditions
    lattice = Lattice.cubic(15.0)
    coords = [
        [atom.coords[0] + 7.5, atom.coords[1] + 7.5, atom.coords[2] + 7.5]
        for atom in mol
    ]

    return Structure(lattice, ["O", "H", "H"], coords, coords_are_cartesian=True)


@pytest.fixture
def tmp_siesta_dir(tmp_path):
    """
    Create a temporary directory for SIESTA calculations.

    Parameters
    ----------
    tmp_path : Path
        pytest tmp_path fixture

    Returns
    -------
    Path
        Path to temporary SIESTA directory
    """
    siesta_dir = tmp_path / "siesta_calc"
    siesta_dir.mkdir()
    return siesta_dir


@pytest.fixture
def mock_siesta_output(tmp_path):
    """
    Create mock SIESTA output files for testing.

    Parameters
    ----------
    tmp_path : Path
        pytest tmp_path fixture

    Returns
    -------
    dict
        Dictionary with paths to mock output files
    """
    # Create mock siesta.out file
    out_file = tmp_path / "siesta.out"
    out_file.write_text(
        """
        siesta: Program's energy decomposition (eV):
        siesta: Ebs     =      -123.456789
        siesta: Eions   =       234.567890
        siesta: Ena     =        45.678901
        siesta: Ekin    =       111.111111
        siesta: Enl     =        -9.999999
        siesta: DEna    =         1.111111
        siesta: DUscf   =         0.123456
        siesta: DUext   =         0.000000
        siesta: Exc     =       -87.654321
        siesta: eta*DQ  =         0.000000
        siesta: Emadel  =         0.000000
        siesta: Emeta   =         0.000000
        siesta: Emolmec =         0.000000
        siesta: Ekinion =         0.000000
        siesta: Eharris =      -123.456789
        siesta: Etot    =      -123.456789
        siesta: FreeEng =      -123.456789

        SCF converged in 15 iterations
    """
    )

    # Create mock .XV file
    xv_file = tmp_path / "siesta.XV"
    xv_file.write_text(
        """
        5.430000000000   0.000000000000   0.000000000000
        0.000000000000   5.430000000000   0.000000000000
        0.000000000000   0.000000000000   5.430000000000
        2
        14  0.000000000000   0.000000000000   0.000000000000
        14  1.357500000000   1.357500000000   1.357500000000
    """
    )

    return {
        "out": out_file,
        "xv": xv_file,
        "energy": -123.456789,
    }


def assert_fdf_contains(fdf_path: Path, expected_params: dict) -> None:
    """
    Assert that an FDF file contains the expected parameters.

    Parameters
    ----------
    fdf_path : Path
        Path to the FDF file
    expected_params : dict
        Dictionary of expected parameter key-value pairs

    Raises
    ------
    AssertionError
        If expected parameters are not found
    """
    if not fdf_path.exists():
        raise AssertionError(f"FDF file not found: {fdf_path}")

    content = fdf_path.read_text()

    for key, expected_value in expected_params.items():
        # Normalize key for comparison
        key_normalized = key.replace(".", "").replace("-", "").lower()

        # Check if parameter appears in file
        found = False
        for line in content.split("\n"):
            line_normalized = line.replace(".", "").replace("-", "").lower()
            if key_normalized in line_normalized:
                found = True
                # Optionally check value if it's a string/number
                if isinstance(expected_value, (str, int, float)):
                    if str(expected_value).lower() not in line.lower():
                        raise AssertionError(
                            f"Parameter {key} found but value doesn't match. "
                            f"Expected: {expected_value}, Line: {line}"
                        )
                break

        if not found:
            raise AssertionError(f"Parameter {key} not found in FDF file: {fdf_path}")


def assert_maker_valid(maker) -> None:
    """
    Assert that a Maker has valid structure.

    Parameters
    ----------
    maker : BaseSiestaMaker
        The maker to validate

    Raises
    ------
    AssertionError
        If maker structure is invalid
    """
    # Check required attributes
    assert hasattr(maker, "name"), "Maker must have a name"
    assert hasattr(maker, "input_set_generator"), "Maker must have input_set_generator"
    assert hasattr(maker, "make"), "Maker must have make method"
    assert callable(maker.make), "make must be callable"

    # Check input set generator
    generator = maker.input_set_generator
    assert hasattr(generator, "get_input_set"), "InputGenerator must have get_input_set"


def assert_job_valid(job) -> None:
    """
    Assert that a Job has valid structure.

    Parameters
    ----------
    job : Job
        The job to validate

    Raises
    ------
    AssertionError
        If job structure is invalid
    """
    # Check required attributes
    assert hasattr(job, "name"), "Job must have a name"
    assert hasattr(job, "function"), "Job must have a function"
    assert callable(job.function), "Job function must be callable"
