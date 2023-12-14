import numpy as np
import pytest
from ase.calculators.lj import LennardJones
from pymatgen.io.ase import AseAtomsAdaptor

from atomate2.forcefields.utils import (
    Relaxer,
    TrajectoryObserver,
)

_rel_tol = 1.0e-6


def test_TrajectoryObserver(si_structure):
    output = {
        "energy": -0.06830751105098437,
        "forces": np.array(
            [
                [8.32667268e-17, 4.16333634e-17, 7.31069641e-17],
                [-8.32667268e-17, -4.16333634e-17, -7.31069641e-17],
            ]
        ),
        "stresses": np.array(
            [
                [
                    4.38808588e-03,
                    4.38808588e-03,
                    4.38808588e-03,
                    -9.47784334e-19,
                    -1.24675748e-18,
                    -1.76448007e-18,
                ]
            ]
        ),
    }

    atoms = AseAtomsAdaptor.get_atoms(structure=si_structure, calculator=LennardJones())

    traj = TrajectoryObserver(atoms)

    assert traj.compute_energy() == pytest.approx(output["energy"], rel=_rel_tol)

    traj()
    assert traj.energies[0] == pytest.approx(output["energy"], rel=_rel_tol)
    assert np.all(np.abs(traj.forces[0] - output["forces"]) < _rel_tol)
    assert np.all(np.abs(traj.stresses[0] - output["stresses"]) < _rel_tol)


def test_Relaxer(si_structure):
    final_lattice = {
        "a": 1.7710250785292723,
        "b": 1.7710250785292574,
        "c": 1.771025078529287,
        "volume": 3.927888357259462,
    }

    final_traj = {
        "energy": -5.846762493093085,
        "forces": np.array(
            [
                [-5.95083358e-12, -1.65202964e-12, 2.84683735e-13],
                [5.92662724e-12, 1.65667133e-12, -2.77979812e-13],
            ]
        ),
        "stresses": np.array(
            [
                -1.27190530e-03,
                -1.27190530e-03,
                -1.27190530e-03,
                -2.31413557e-14,
                -3.26060788e-14,
                5.09222979e-13,
            ]
        ),
    }

    atoms = AseAtomsAdaptor.get_atoms(structure=si_structure, calculator=LennardJones())

    relaxer = Relaxer(calculator=LennardJones(), optimizer="BFGS")

    relax_output = relaxer.relax(atoms=atoms)

    for key in final_lattice:
        assert relax_output["final_structure"].lattice.__getattribute__(
            key
        ) == pytest.approx(final_lattice[key], rel=_rel_tol)

    assert relax_output["trajectory"].energies[-1] == pytest.approx(
        final_traj["energy"], rel=_rel_tol
    )
    assert np.all(
        np.abs(relax_output["trajectory"].forces[-1] - final_traj["forces"]) < _rel_tol
    )
    assert np.all(
        np.abs(relax_output["trajectory"].stresses[-1] - final_traj["stresses"])
        < _rel_tol
    )
