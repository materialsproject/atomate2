import logging
from os import path

import pytest
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
from monty.serialization import loadfn
from pymatgen.io.ase import AseAtomsAdaptor

from atomate2.forcefields.utils import (
    FrechetCellFilter,
    Relaxer,
    TrajectoryObserver,
)

logger = logging.getLogger(__name__)

_rel_tol = 1.0e-6


def approx(val):
    return pytest.approx(val, rel=_rel_tol)


def test_safe_import():
    assert FrechetCellFilter is None or FrechetCellFilter.__module__ == "ase.filters"
    if FrechetCellFilter is None:
        with pytest.raises(ImportError):
            pass


def test_TrajectoryObserver(si_structure, test_dir, tmp_dir):
    output = loadfn(f"{test_dir}/forcefields/utils_test_data.json")[
        "TrajectoryObserver"
    ]
    atoms = AseAtomsAdaptor.get_atoms(structure=si_structure, calculator=LennardJones())

    traj = TrajectoryObserver(atoms)

    assert traj.compute_energy() == approx(output["energy"])

    traj()
    # NB: always 3 Cartesian components to each interatomic force,
    # and only 6 unique elements of the flattened stress tensor
    assert traj.energies[0] == approx(output["energy"])
    assert all(
        traj.forces[0][i][j] == approx(output["forces"][i][j])
        for i in range(len(si_structure))
        for j in range(3)
    )
    assert all(traj.stresses[0][i] == approx(output["stresses"][i]) for i in range(6))

    save_file_name = "log_file.json.gz"
    traj.save(save_file_name)
    assert path.isfile(save_file_name)


@pytest.mark.parametrize(
    ("optimizer", "traj_file"),
    [("BFGS", None), (None, None), (BFGS, "log_file.json.gz")],
)
def test_Relaxer(si_structure, test_dir, tmp_dir, optimizer, traj_file):
    data_key = "pypi" if FrechetCellFilter is None else "git"
    test_data = loadfn(f"{test_dir}/forcefields/utils_test_data.json")["Relaxer"][
        data_key
    ]

    if optimizer is None:
        # None is invalid, should raise ValueError
        with pytest.raises(ValueError):
            Relaxer(calculator=LennardJones(), optimizer=optimizer)
        return

    relaxer = Relaxer(calculator=LennardJones(), optimizer=optimizer)

    try:
        relax_output = relaxer.relax(atoms=si_structure, traj_file=traj_file)
    except TypeError:
        # if using PyPI ASE, FrechetCellFilter is not callable, set to None
        assert FrechetCellFilter is None
        return

    assert all(
        (
            relax_output["final_structure"].lattice.__getattribute__(key)
            == approx(test_data["relaxer_final_lattice"][key])
        )
        for key in test_data["relaxer_final_lattice"]
    )

    assert relax_output["trajectory"].energies[-1] == approx(
        test_data["relaxer_final_traj"]["energy"]
    )

    assert all(
        relax_output["trajectory"].forces[-1][i][j]
        == approx(test_data["relaxer_final_traj"]["forces"][i][j])
        for i in range(len(si_structure))
        for j in range(3)
    )

    assert all(
        relax_output["trajectory"].stresses[-1][i]
        == approx(test_data["relaxer_final_traj"]["stresses"][i])
        for i in range(6)
    )

    if traj_file:
        print(traj_file)
        assert path.isfile(traj_file)
