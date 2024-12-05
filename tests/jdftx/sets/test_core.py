import numpy as np
import pytest

from atomate2.jdftx.sets.base import JdftxInputGenerator
from atomate2.jdftx.sets.core import (
    IonicMinSetGenerator,
    LatticeMinSetGenerator,
    SinglePointSetGenerator,
)


@pytest.fixture
def basis_and_potential():
    return {
        "fluid-cation": {"name": "Na+", "concentration": 1.0},
        "fluid-anion": {"name": "F-", "concentration": 1.0},
    }


def test_singlepoint_generator(si_structure, basis_and_potential):
    gen = SinglePointSetGenerator(user_settings=basis_and_potential)
    input_set = gen.get_input_set(si_structure)
    jdftx_input = input_set.jdftxinput
    assert jdftx_input["fluid-cation"]["concentration"] == 1.0
    assert jdftx_input["lattice-minimize"]["nIterations"] == 0


def test_default_generator(si_structure, basis_and_potential):
    gen = JdftxInputGenerator(user_settings=basis_and_potential)
    input_set = gen.get_input_set(si_structure)
    jdftx_input = input_set.jdftxinput
    assert jdftx_input["fluid-cation"]["concentration"] == 1.0


def test_ionicmin_generator(si_structure, basis_and_potential):
    gen = IonicMinSetGenerator(user_settings=basis_and_potential)
    input_set = gen.get_input_set(si_structure)
    jdftx_input = input_set.jdftxinput
    assert jdftx_input["ionic-minimize"]["nIterations"] == 100


def test_latticemin_generator(si_structure, basis_and_potential):
    gen = LatticeMinSetGenerator(user_settings=basis_and_potential)
    input_set = gen.get_input_set(si_structure)
    jdftx_input = input_set.jdftxinput
    assert jdftx_input["lattice-minimize"]["nIterations"] == 100


def test_coulomb_truncation(si_structure):
    cart_gen = JdftxInputGenerator(
        calc_type="surface", user_settings={"coords-type": "Cartesian"}
    )
    frac_gen = JdftxInputGenerator(
        calc_type="surface", user_settings={"coords-type": "Lattice"}
    )
    cart_input_set = cart_gen.get_input_set(si_structure)
    frac_input_set = frac_gen.get_input_set(si_structure)
    cart_jdftx_input = cart_input_set.jdftxinput
    frac_jdftx_input = frac_input_set.jdftxinput

    cart_center_of_mass = np.array(
        list(cart_jdftx_input["coulomb-truncation-embed"].values())
    )
    frac_center_of_mass = np.array(
        list(frac_jdftx_input["coulomb-truncation-embed"].values())
    )
    assert any(cart_center_of_mass > 1)
    assert all(frac_center_of_mass < 1)
    assert np.allclose(
        cart_center_of_mass, frac_center_of_mass @ si_structure.lattice.matrix
    )
