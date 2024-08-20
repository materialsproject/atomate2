from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Molecule, Structure


@pytest.fixture
def fcc_ne_structure(a0: float = 4.6) -> Structure:
    """Generic fcc Ne structure for testing LJ jobs."""
    return Structure(
        Lattice([[0.0 if i == j else 0.5 * a0 for j in range(3)] for i in range(3)]),
        ["Ne"],
        [[0.0, 0.0, 0.0]],
    )


@pytest.fixture
def lj_fcc_ne_pars() -> dict[str, float]:
    """
    LJ parameters optimized to reproduce experimentally reported
    fcc Ne:
        - lattice constant a0 = 4.4644 angstrom,
        - bulk modulus b0 = 1.102 GPa
    See Table I of 10.1103/PhysRevB.80.064106
    """
    return {
        "sigma": 2.887,
        "epsilon": 2.330e-03,
    }


@pytest.fixture
def h2o_3uud_trimer() -> Molecule:
    return Molecule.from_str(
        """9
# 3UUD water trimer from BEGDB, http://www.begdb.org/index.php?action=oneMolecule&state=show&id=4180
O  -1.38183  -0.79188  -0.17297
H  -0.45433  -1.10048  -0.23187
H  -1.81183  -1.39808   0.44213
O   1.41257  -0.77648  -0.31737
H   1.22807   0.17372  -0.17317
H   1.93037  -1.04028   0.45363
O   0.01887   1.60382   0.21583
H  -0.73003   0.98002   0.13523
H  -0.21183   2.34962  -0.35147""",
        fmt="xyz",
    )
