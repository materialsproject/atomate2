"""Utils for using a force field (aka an interatomic potential).

The following code has been taken and modified from
https://github.com/materialsvirtuallab/m3gnet
The code has been released under BSD 3-Clause License
and the following copyright applies:
Copyright (c) 2022, Materials Virtual Lab.
"""

from __future__ import annotations

import contextlib
import io
import pickle
import sys
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

from ase.optimize import BFGS, FIRE, LBFGS, BFGSLineSearch, LBFGSLineSearch, MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.spacegroup.symmetrize import FixSymmetry
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from ase.filters import FrechetCellFilter
except ImportError:
    FrechetCellFilter = None
    warnings.warn(
        "Due to errors in the implementation of gradients in the ASE"
        " ExpCellFilter, we recommend installing ASE from gitlab\n"
        "    pip install git+https://gitlab.com/ase/ase.git\n"
        "rather than PyPi to access FrechetCellFilter. See\n"
        "    https://wiki.fysik.dtu.dk/ase/ase/filters.html#the-frechetcellfilter-class\n"
        "for more details. Otherwise, you must specify an alternate ASE Filter.",
        stacklevel=2,
    )

if TYPE_CHECKING:
    from collections.abc import Generator
    from os import PathLike
    from typing import Any

    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.filters import Filter
    from ase.optimize.optimize import Optimizer

OPTIMIZERS = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "MDMin": MDMin,
    "SciPyFminCG": SciPyFminCG,
    "SciPyFminBFGS": SciPyFminBFGS,
    "BFGSLineSearch": BFGSLineSearch,
}


class TrajectoryObserver:
    """Trajectory observer.

    This is a hook in the relaxation process that saves the intermediate structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """
        Initialize the Observer.

        Parameters
        ----------
        atoms (Atoms): the structure to observe.

        Returns
        -------
            None
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """Save the properties of an Atoms during the relaxation."""
        # TODO: maybe include magnetic moments
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def compute_energy(self) -> float:
        """
        Calculate the energy, here we just use the potential energy.

        Returns
        -------
            energy (float)
        """
        return self.atoms.get_potential_energy()

    def save(self, filename: str | PathLike) -> None:
        """
        Save the trajectory file.

        Parameters
        ----------
        filename (str): filename to save the trajectory.

        Returns
        -------
            None
        """
        traj_dict = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(traj_dict, file)


class Relaxer:
    """Relaxer is a class for structural relaxation."""

    def __init__(
        self,
        calculator: Calculator,
        optimizer: Optimizer | str = "FIRE",
        relax_cell: bool = True,
        fix_symmetry: bool = False,
    ) -> None:
        """
        Initialize the Relaxer.

        Parameters
        ----------
        calculator (ase Calculator): an ase calculator
        optimizer (str or ase Optimizer): the optimization algorithm.
        relax_cell (bool): if True, cell parameters will be optimized.
        fix_symmetry (bool): if True, symmetry will be fixed during relaxation.
        """
        self.calculator = calculator

        if isinstance(optimizer, str):
            optimizer_obj = OPTIMIZERS.get(optimizer)
        elif optimizer is None:
            raise ValueError("Optimizer cannot be None")
        else:
            optimizer_obj = optimizer

        self.opt_class: Optimizer = optimizer_obj
        self.relax_cell = relax_cell
        self.ase_adaptor = AseAtomsAdaptor()
        self.fix_symmetry = fix_symmetry

    def relax(
        self,
        atoms: Atoms,
        fmax: float = 0.1,
        steps: int = 500,
        traj_file: str = None,
        interval: int = 1,
        verbose: bool = False,
        cell_filter: Filter = FrechetCellFilter,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Relax the structure.

        Parameters
        ----------
        atoms : Atoms
            The atoms for relaxation.
        fmax : float
            Total force tolerance for relaxation convergence.
        steps : int
            Max number of steps for relaxation.
        traj_file : str
            The trajectory file for saving.
        interval : int
            The step interval for saving the trajectories.
        verbose : bool
            If True, screen output will be shown.
        **kwargs
            Further kwargs.

        Returns
        -------
            dict including optimized structure and the trajectory
        """
        if isinstance(atoms, (Structure, Molecule)):
            atoms = self.ase_adaptor.get_atoms(atoms)
        if self.fix_symmetry:
            atoms.set_constraint(FixSymmetry(atoms))
        atoms.set_calculator(self.calculator)
        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if self.relax_cell:
                atoms = cell_filter(atoms)
            optimizer = self.opt_class(atoms, **kwargs)
            optimizer.attach(obs, interval=interval)
            optimizer.run(fmax=fmax, steps=steps)
            obs()
        if traj_file is not None:
            obs.save(traj_file)
        if isinstance(atoms, cell_filter):
            atoms = atoms.atoms

        struct = self.ase_adaptor.get_structure(atoms)
        return {"final_structure": struct, "trajectory": obs}


@contextmanager
def revert_default_dtype() -> Generator[None, None, None]:
    """Context manager for torch.default_dtype.

    Reverts it to whatever torch.get_default_dtype() was when entering the context.

    Originally added for use with MACE(Relax|Static)Maker.
    https://github.com/ACEsuit/mace/issues/328
    """
    import torch

    orig = torch.get_default_dtype()
    yield
    torch.set_default_dtype(orig)
