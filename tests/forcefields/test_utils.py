import os

import pytest
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
from ase.spacegroup.symmetrize import check_symmetry
from numpy.testing import assert_allclose
from pymatgen.core import Structure

from atomate2.forcefields import MLFF
from atomate2.forcefields.utils import (
    FrechetCellFilter,
    Relaxer,
    TrajectoryObserver,
    ase_calculator,
)


def test_safe_import():
    assert FrechetCellFilter is None or FrechetCellFilter.__module__ == "ase.filters"


def test_trajectory_observer(si_structure: Structure, test_dir, tmp_dir):
    atoms = si_structure.to_ase_atoms()
    atoms.set_calculator(LennardJones())

    traj = TrajectoryObserver(atoms)

    expected_energy = -0.06830751105
    assert traj.compute_energy() == pytest.approx(expected_energy)

    traj()
    # NB: always 3 Cartesian components to each interatomic force,
    # and only 6 unique elements of the flattened stress tensor
    assert traj.energies[0] == pytest.approx(expected_energy)

    expected_forces = [
        [8.32667268e-17, 4.16333634e-17, 7.31069641e-17],
        [-8.32667268e-17, -4.16333634e-17, -7.31069641e-17],
    ]
    assert_allclose(traj.forces[0], expected_forces, atol=1e-8)
    expected_stresses = [
        4.38808e-03,
        4.38808e-03,
        4.38808e-03,
        -9.47784e-19,
        -1.24675e-18,
        -1.76448e-18,
    ]
    assert_allclose(traj.stresses[0], expected_stresses, atol=1e-8)

    save_file_name = "log_file.traj"
    traj.save(save_file_name)
    assert os.path.isfile(save_file_name)


@pytest.mark.parametrize(
    ("optimizer", "traj_file"),
    [("BFGS", None), (None, None), (BFGS, "log_file.traj")],
)
def test_relaxer(si_structure, test_dir, tmp_dir, optimizer, traj_file):
    if FrechetCellFilter:
        expected_lattice = {
            "a": 3.866974,
            "b": 3.866974,
            "c": 3.866974,
            "volume": 40.888292,
        }
        expected_forces = [
            [8.32667268e-17, 4.16333634e-17, 7.31069641e-17],
            [-8.32667268e-17, -4.16333634e-17, -7.31069641e-17],
        ]
        expected_energy = -0.0683075110
        expected_stresses = [
            4.38808588e-03,
            4.38808588e-03,
            4.38808588e-03,
            -9.74728670e-19,
            -1.31340626e-18,
            -1.60482883e-18,
        ]
    else:
        expected_lattice = {
            "a": 1.77102507,
            "b": 1.77102507,
            "c": 1.77102507,
            "volume": 3.927888,
        }
        expected_forces = [
            [-5.95083358e-12, -1.65202964e-12, 2.84683735e-13],
            [5.92662724e-12, 1.65667133e-12, -2.77979812e-13],
        ]
        expected_energy = -5.846762493
        expected_stresses = [
            -1.27190530e-03,
            -1.27190530e-03,
            -1.27190530e-03,
            -2.31413557e-14,
            -3.26060788e-14,
            5.09222979e-13,
        ]

    if optimizer is None:
        with pytest.raises(ValueError, match="Optimizer cannot be None"):
            Relaxer(calculator=LennardJones(), optimizer=optimizer)
        return

    relaxer = Relaxer(calculator=LennardJones(), optimizer=optimizer)

    try:
        relax_output = relaxer.relax(atoms=si_structure, traj_file=traj_file)
    except TypeError:
        return

    assert {
        key: getattr(relax_output["final_structure"].lattice, key)
        for key in expected_lattice
    } == pytest.approx(expected_lattice)

    assert relax_output["trajectory"].frame_properties[-1]["energy"] == pytest.approx(
        expected_energy
    )

    assert_allclose(
        relax_output["trajectory"].frame_properties[-1]["forces"], expected_forces
    )

    assert_allclose(
        relax_output["trajectory"].frame_properties[-1]["stress"], expected_stresses
    )

    if traj_file:
        assert os.path.isfile(traj_file)


@pytest.mark.parametrize(("force_field"), ["CHGNet", "MACE"])
def test_ext_load(force_field: str):
    decode_dict = {
        "CHGNet": {"@module": "chgnet.model.dynamics", "@callable": "CHGNetCalculator"},
        "MACE": {"@module": "mace.calculators", "@callable": "mace_mp"},
    }[force_field]
    calc_from_decode = ase_calculator(decode_dict)
    calc_from_preset = ase_calculator(str(MLFF(force_field)))
    assert type(calc_from_decode) is type(calc_from_preset)
    assert calc_from_decode.name == calc_from_preset.name
    assert calc_from_decode.parameters == calc_from_preset.parameters == {}


@pytest.mark.parametrize(("fix_symmetry"), [True, False])
def test_fix_symmetry(fix_symmetry):
    # adapted from the example at https://wiki.fysik.dtu.dk/ase/ase/constraints.html#the-fixsymmetry-class
    relaxer = Relaxer(
        calculator=LennardJones(), relax_cell=True, fix_symmetry=fix_symmetry
    )
    atoms_al = bulk("Al", "bcc", a=2 / 3**0.5, cubic=True)
    atoms_al *= 2, 2, 2
    atoms_al.positions[0, 0] += 1e-7
    symmetry_init = check_symmetry(atoms_al, 1e-6)
    final_struct: Structure = relaxer.relax(atoms=atoms_al, steps=1)["final_structure"]
    symmetry_final = check_symmetry(final_struct.to_ase_atoms(), 1e-6)
    if fix_symmetry:
        assert symmetry_init["number"] == symmetry_final["number"] == 229
    else:
        assert symmetry_init["number"] != symmetry_final["number"] == 99


def test_m3gnet_pot():
    import matgl
    from matgl.ext.ase import PESCalculator

    kwargs_calc = {"path": "M3GNet-MP-2021.2.8-DIRECT-PES", "stress_weight": 2.0}
    kwargs_default = {"stress_weight": 2.0}

    m3gnet_calculator = ase_calculator(calculator_meta="MLFF.M3GNet", **kwargs_calc)

    # uses "M3GNet-MP-2021.2.8-PES" per default
    m3gnet_default = ase_calculator(calculator_meta="MLFF.M3GNet", **kwargs_default)

    potential = matgl.load_model("M3GNet-MP-2021.2.8-DIRECT-PES")
    m3gnet_pes_calc = PESCalculator(potential=potential, stress_weight=2.0)

    assert str(m3gnet_pes_calc.potential) == str(m3gnet_calculator.potential)
    # casting necessary because <class 'matgl.apps.pes.Potential'> can't be compared
    assert str(m3gnet_pes_calc.potential) != str(m3gnet_default.potential)
    assert m3gnet_pes_calc.stress_weight == m3gnet_calculator.stress_weight
    assert m3gnet_pes_calc.stress_weight == m3gnet_default.stress_weight
