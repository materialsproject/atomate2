"""Shared pytest fixtures for atomate2siesta tests."""

import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure


@pytest.fixture(scope="session", autouse=True)
def packaged_pseudos():
    """Provide the packaged pseudopotentials when none are configured.

    Several tests generate real SIESTA inputs (dry-run mode) and need a valid
    pseudopotential directory; without one, input generation aborts and the
    tests fail (or pass vacuously). If the environment does not already set
    SIESTA_PP_PATH, extract the standard scalar-relativistic PBE PseudoDojo
    set shipped with the package into a session-scoped temporary directory
    and point SIESTA_PP_PATH at it. A user-configured pseudo_path still takes
    precedence in the lookup chain, so this only fills the gap.
    """
    if os.environ.get("SIESTA_PP_PATH"):
        yield
        return

    try:
        import atomate2.siesta.pseudos as pseudos_pkg

        tarball = Path(pseudos_pkg.__file__).parent / "nc-sr-04_pbe_standard_psml.tgz"
    except ImportError:
        tarball = None

    if tarball is None or not tarball.exists():
        # Package does not ship pseudos in this installation; leave the
        # environment untouched (affected tests keep their current behavior).
        yield
        return

    tmpdir = tempfile.mkdtemp(prefix="siesta_pp_")
    with tarfile.open(tarball) as tf:
        try:
            tf.extractall(tmpdir, filter="data")
        except TypeError:  # Python without the tarfile filter parameter
            tf.extractall(tmpdir)
    os.environ["SIESTA_PP_PATH"] = tmpdir
    try:
        yield
    finally:
        os.environ.pop("SIESTA_PP_PATH", None)
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def flos_neb_template():
    """Provide a stub flos NEB template when FLOS_PATH is not configured.

    generate_neb_band() copies <FLOS_PATH>/examples/neb.lua and rewrites its
    n_images value; the NEB tests exercise that image-generation and
    rewriting logic, not flos itself. When no flos installation is
    configured, point SETTINGS.FLOS_PATH at a temporary directory containing
    a minimal template that provides the rewritten pattern. A real
    FLOS_PATH from the user's config or environment always takes precedence.
    """
    from atomate2.siesta import SETTINGS

    if SETTINGS.FLOS_PATH:
        yield
        return

    tmpdir = tempfile.mkdtemp(prefix="flos_stub_")
    examples = Path(tmpdir) / "examples"
    examples.mkdir()
    (examples / "neb.lua").write_text(
        "-- Test stub for the flos examples/neb.lua template.\n"
        "-- Only the pattern rewritten by atomate2siesta is required here.\n"
        "local n_images = 6\n"
    )
    SETTINGS.FLOS_PATH = tmpdir
    try:
        yield
    finally:
        SETTINGS.FLOS_PATH = None
        shutil.rmtree(tmpdir, ignore_errors=True)


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
