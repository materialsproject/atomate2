import pytest
from atomate2.jdftx.sets.core import SinglePointSetGenerator
from atomate2.jdftx.sets.base import JdftxInputGenerator
from atomate2.jdftx.sets.core import IonicMinSetGenerator
from atomate2.jdftx.sets.core import LatticeMinSetGenerator

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