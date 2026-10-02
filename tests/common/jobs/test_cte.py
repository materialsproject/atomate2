"""Tests for get_cte and the Born charge mapping of the CTE document.

The force field workflow is tested in tests/forcefields/flows/test_cte.py.
"""

import numpy as np
import pytest
from ase.build import bulk
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.analysis.elasticity import ElasticTensor
from scipy.constants import Boltzmann, Planck
from scipy.spatial.transform import Rotation

from atomate2.common.schemas.cte import _expand_born_to_unitcell, get_cte


def _heat_capacity(frequency: float, temperature: float) -> float:
    x = Planck * frequency * 1e12 / (Boltzmann * temperature)
    return Boltzmann * x**2 * np.exp(x) / (np.exp(x) - 1) ** 2


def test_get_cte_cubic():
    """For a cubic crystal with gamma = g * I, alpha = g * Cv / (3 * B * V)."""
    c11, c12, c44 = 170.0, 120.0, 75.0
    elastic = np.zeros((6, 6))
    elastic[:3, :3] = c12
    np.fill_diagonal(elastic[:3, :3], c11)
    elastic[3:, 3:] = np.eye(3) * c44
    bulk_modulus = (c11 + 2 * c12) / 3 * 1e9  # Pa

    frequencies = np.array([[2.0, 5.0, 7.0], [3.0, 4.0, 6.0]])
    weights = np.array([1, 3])
    gruneisen = 1.7 * np.broadcast_to(np.eye(3), (2, 3, 3, 3))
    volume, temperature = 45.0, 300.0

    alpha, mean_gruneisen = get_cte(
        frequencies, gruneisen, weights, elastic, volume, [temperature]
    )

    heat_capacity = (
        sum(
            w * _heat_capacity(f, temperature)
            for w, row in zip(weights, frequencies, strict=True)
            for f in row
        )
        / weights.sum()
    )
    expected = 1.7 * heat_capacity / (3 * bulk_modulus * volume * 1e-30)
    assert alpha[0] == pytest.approx(expected * np.eye(3), rel=1e-10, abs=1e-20)
    assert mean_gruneisen[0] == pytest.approx(1.7 * np.eye(3))


def test_get_cte_rotation():
    """Rotating the inputs must rotate alpha, which checks the shear terms.

    The mode Grueneisen tensors from phono3py are not symmetric, so the input
    tensors here are not symmetric either.
    """
    # hexagonal elastic tensor, with C66 = (C11 - C12) / 2
    c11, c12, c13, c33, c44 = 350.0, 120.0, 90.0, 400.0, 110.0
    elastic = np.array(
        [
            [c11, c12, c13, 0, 0, 0],
            [c12, c11, c13, 0, 0, 0],
            [c13, c13, c33, 0, 0, 0],
            [0, 0, 0, c44, 0, 0],
            [0, 0, 0, 0, c44, 0],
            [0, 0, 0, 0, 0, (c11 - c12) / 2],
        ]
    )
    rng = np.random.default_rng(7)
    frequencies = rng.uniform(1.0, 10.0, size=(4, 6))
    gruneisen = rng.normal(1.0, 0.5, size=(4, 6, 3, 3))
    weights = np.ones(4)
    temperatures = [100.0, 500.0]

    alpha, _ = get_cte(frequencies, gruneisen, weights, elastic, 30.0, temperatures)
    rotation = Rotation.from_euler("zxz", [30, 50, 70], degrees=True).as_matrix()
    rotated_elastic = ElasticTensor.from_voigt(elastic).rotate(rotation).voigt
    rotated_gruneisen = np.einsum("ik,qbkl,jl->qbij", rotation, gruneisen, rotation)
    rotated_alpha, _ = get_cte(
        frequencies, rotated_gruneisen, weights, rotated_elastic, 30.0, temperatures
    )

    expected = np.einsum("ik,tkl,jl->tij", rotation, alpha, rotation)
    assert np.abs(expected[:, 0, 1]).max() > 1e-7  # the shear terms are tested
    assert rotated_alpha == pytest.approx(expected, rel=1e-8, abs=1e-15)


def test_get_cte_left_out_modes():
    """Modes below min_frequency are left out, and T = 0 gives zero."""
    elastic = np.diag([200.0, 200.0, 200.0, 80.0, 80.0, 80.0])
    frequencies = np.array([[0.0, 0.0, 0.0, 4.0], [-0.5, 3.0, 5.0, 6.0]])
    gruneisen = np.ones((2, 4, 3, 3))
    gruneisen[0, :3] = np.nan  # NaN checks that the left-out modes are ignored

    alpha, mean_gruneisen = get_cte(
        frequencies, gruneisen, np.ones(2), elastic, 40.0, [0.0, 300.0]
    )
    assert np.all(alpha[0] == 0.0)
    assert mean_gruneisen[0] is None
    assert mean_gruneisen[1] == pytest.approx(np.ones((3, 3)))

    # only the modes at 4, 3, 5 and 6 THz count, over two q-points
    heat_capacity = sum(_heat_capacity(f, 300.0) for f in (4.0, 3.0, 5.0, 6.0)) / 2
    thermal_stress = heat_capacity / (40.0 * 1e-30)
    expected = np.full((3, 3), thermal_stress / 80e9 / 2)
    np.fill_diagonal(expected, thermal_stress / 200e9)
    assert alpha[1] == pytest.approx(expected, rel=1e-10)


def test_expand_born_to_unitcell():
    """Born charges of the primitive cell are mapped onto the conventional cell."""
    atoms = bulk("MgO", "rocksalt", a=4.2, cubic=True)
    unitcell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell[:],
        scaled_positions=atoms.get_scaled_positions(),
    )
    phonon = Phonopy(unitcell, supercell_matrix=np.eye(3), primitive_matrix="auto")
    assert len(phonon.primitive) == 2
    born_primitive = {"Mg": 1.9, "O": -1.9}
    phonon.nac_params = {
        "born": [np.eye(3) * born_primitive[s] for s in phonon.primitive.symbols],
        "dielectric": np.eye(3) * 3.0,
    }

    born = _expand_born_to_unitcell(phonon)
    assert born.shape == (8, 3, 3)
    for symbol, charges in zip(unitcell.symbols, born, strict=True):
        assert charges == pytest.approx(np.eye(3) * born_primitive[symbol])
