from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure


@pytest.fixture()
def fcc_ne_structure(a0: float = 4.6) -> Structure:
    """Generic fcc Ne structure for testing LJ jobs."""
    return Structure(
        Lattice([[0.0 if i == j else 0.5 * a0 for j in range(3)] for i in range(3)]),
        ["Ne"],
        [[0.0, 0.0, 0.0]],
    )


@pytest.fixture()
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
