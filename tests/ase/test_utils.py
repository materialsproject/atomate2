from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
from ase.spacegroup.symmetrize import check_symmetry
from numpy.testing import assert_allclose

if TYPE_CHECKING:
    from pymatgen.core import Structure


from atomate2.ase.utils import AseRelaxer, TrajectoryObserver


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

    if optimizer is None:
        with pytest.raises(ValueError, match="Optimizer cannot be None"):
            AseRelaxer(calculator=LennardJones(), optimizer=optimizer)
        return

    relaxer = AseRelaxer(calculator=LennardJones(), optimizer=optimizer)

    try:
        relax_output = relaxer.relax(atoms=si_structure, traj_file=traj_file)
    except TypeError:
        return

    assert {
        key: getattr(relax_output.final_mol_or_struct.lattice, key)
        for key in expected_lattice
    } == pytest.approx(expected_lattice)

    assert relax_output.trajectory.frame_properties[-1]["energy"] == pytest.approx(
        expected_energy
    )

    assert_allclose(
        relax_output["trajectory"].frame_properties[-1]["forces"],
        expected_forces,
        atol=1e-11,
    )

    assert_allclose(
        relax_output["trajectory"].frame_properties[-1]["stress"],
        expected_stresses,
        atol=1e-11,
    )

    if traj_file:
        assert os.path.isfile(traj_file)


@pytest.mark.parametrize(("fix_symmetry"), [True, False])
def test_fix_symmetry(fix_symmetry):
    # adapted from the example at https://wiki.fysik.dtu.dk/ase/ase/constraints.html#the-fixsymmetry-class
    relaxer = AseRelaxer(
        calculator=LennardJones(), relax_cell=True, fix_symmetry=fix_symmetry
    )
    atoms_al = bulk("Al", "bcc", a=2 / 3**0.5, cubic=True)
    atoms_al = atoms_al * (2, 2, 2)
    atoms_al.positions[0, 0] += 1e-7
    symmetry_init = check_symmetry(atoms_al, 1e-6)
    final_struct: Structure = relaxer.relax(atoms=atoms_al, steps=1).final_mol_or_struct
    symmetry_final = check_symmetry(final_struct.to_ase_atoms(), 1e-6)
    if fix_symmetry:
        assert symmetry_init["number"] == symmetry_final["number"] == 229
    else:
        assert symmetry_init["number"] != symmetry_final["number"] == 99
