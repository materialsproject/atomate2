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
import json
import sys
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixSymmetry
from ase.io import Trajectory as AseTrajectory
from ase.optimize import BFGS, FIRE, LBFGS, BFGSLineSearch, LBFGSLineSearch, MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from monty.json import MontyDecoder
from monty.serialization import dumpfn
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.trajectory import Trajectory as PmgTrajectory
from pymatgen.io.ase import AseAtomsAdaptor

from atomate2.forcefields import MLFF
from atomate2.forcefields.schemas import ForcefieldResult

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
    from typing import Any, Literal

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.filters import Filter
    from ase.io.trajectory import TrajectoryReader
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


def _get_pymatgen_trajectory_from_observer(
    trajectory_observer: Any, frame_property_keys: list[str]
) -> PmgTrajectory:
    to_singular = {"energies": "energy", "stresses": "stress"}

    if hasattr(trajectory_observer, "as_dict"):
        traj = trajectory_observer.as_dict()
    else:
        traj = trajectory_observer.__dict__

    n_md_steps = len(traj["cells"])
    species = AseAtomsAdaptor.get_structure(traj["atoms"]).species

    structures = [
        Structure(
            lattice=traj["cells"][idx],
            coords=traj["atom_positions"][idx],
            species=species,
            coords_are_cartesian=True,
        )
        for idx in range(n_md_steps)
    ]

    frame_properties = [
        {
            to_singular.get(key, key): traj[key][idx]
            for key in frame_property_keys
            if key in traj
        }
        for idx in range(n_md_steps)
    ]

    return PmgTrajectory.from_structures(
        structures,
        frame_properties=frame_properties,
        constant_lattice=False,
    )


class TrajectoryObserver:
    """Trajectory observer.

    This is a hook in the relaxation process that saves the intermediate structures.
    """

    def __init__(self, atoms: Atoms, store_md_outputs: bool = False) -> None:
        """Initialize the Observer.

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

        self._store_magmoms = True
        self.magmoms: list[np.ndarray] = []

        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

        self._store_md_outputs = store_md_outputs
        # `self.{velocities,temperatures}` always initialized,
        # but data is only stored / saved to trajectory for MD runs
        self.velocities: list[np.ndarray] = []
        self.temperatures: list[float] = []

    def __call__(self) -> None:
        """Save the properties of an Atoms during the relaxation."""
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        # MD needs kinetic energy parts of stress, relaxations do not
        # When _store_md_outputs is True, ideal gas contribution to
        # stress is included.
        self.stresses.append(
            self.atoms.get_stress(include_ideal_gas=self._store_md_outputs)
        )

        if self._store_magmoms:
            try:
                self.magmoms.append(self.atoms.get_magnetic_moments())
            except PropertyNotImplementedError:
                self._store_magmoms = False

        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

        if self._store_md_outputs:
            self.velocities.append(self.atoms.get_velocities())
            self.temperatures.append(self.atoms.get_temperature())

    def compute_energy(self) -> float:
        """
        Calculate the energy, here we just use the potential energy.

        Returns
        -------
            energy (float)
        """
        return self.atoms.get_potential_energy()

    def save(
        self, filename: str | PathLike | None, fmt: Literal["pmg", "ase"] = "ase"
    ) -> None:
        """
        Save the trajectory file using monty.serialization.

        Parameters
        ----------
        filename (str): filename to save the trajectory.

        Returns
        -------
            None
        """
        filename = str(filename) if filename is not None else None
        if fmt == "pmg":
            self.to_pymatgen_trajectory(filename=filename)
        elif fmt == "ase":
            self.to_ase_trajectory(filename=filename)

    def to_ase_trajectory(
        self, filename: str | None = "atoms.traj"
    ) -> TrajectoryReader:
        """
        Convert to an ASE .Trajectory.

        Parameters
        ----------
        filename : str | None
            Name of the file to write the ASE trajectory to.
            If None, no file is written.
        """
        for idx in range(len(self.cells)):
            atoms = self.atoms.copy()
            atoms.set_positions(self.atom_positions[idx])
            atoms.set_cell(self.cells[idx])

            if self._store_md_outputs:
                atoms.set_velocities(self.velocities[idx])

            kwargs = {
                "energy": self.energies[idx],
                "forces": self.forces[idx],
                "stress": self.stresses[idx],
            }
            if self._store_magmoms:
                kwargs["magmom"] = self.magmoms[idx]

            atoms.calc = SinglePointCalculator(atoms=atoms, **kwargs)
            with AseTrajectory(filename, "a" if idx > 0 else "w", atoms=atoms) as file:
                file.write()

        return AseTrajectory(filename, "r")

    def to_pymatgen_trajectory(
        self, filename: str | None = "trajectory.json.gz"
    ) -> PmgTrajectory:
        """
        Convert the trajectory to a pymatgen .Trajectory object.

        Parameters
        ----------
        filename : str or None
            Name of the file to write the pymatgen trajectory to.
            If None, no file is written.
        """
        frame_property_keys = ["energy", "forces", "stress"]
        if self._store_magmoms:
            frame_property_keys += ["magmoms"]
        if self._store_md_outputs:
            frame_property_keys += ["velocities", "temperature"]

        traj = _get_pymatgen_trajectory_from_observer(
            self, frame_property_keys=frame_property_keys
        )

        if filename:
            dumpfn(traj, filename)

        return traj

    def as_dict(self) -> dict:
        """Make JSONable dict representation of the Trajectory."""
        traj_dict = {
            "energy": self.energies,
            "forces": self.forces,
            "stress": self.stresses,
            "atom_positions": self.atom_positions,
            "cells": self.cells,
            "atoms": self.atoms,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }

        if self._store_magmoms:
            traj_dict["magmoms"] = self.magmoms

        if self._store_md_outputs:
            traj_dict.update(velocities=self.velocities, temperature=self.temperatures)
        # sanitize dict
        for key in traj_dict:
            if all(isinstance(val, np.ndarray) for val in traj_dict[key]):
                traj_dict[key] = [val.tolist() for val in traj_dict[key]]
            elif isinstance(traj_dict[key], np.ndarray):
                traj_dict[key] = traj_dict[key].tolist()
        return traj_dict


class Relaxer:
    """Relaxer is a class for structural relaxation."""

    def __init__(
        self,
        calculator: Calculator,
        optimizer: Optimizer | str = "FIRE",
        relax_cell: bool = True,
        fix_symmetry: bool = False,
        symprec: float = 1e-2,
    ) -> None:
        """Initialize the Relaxer.

        Parameters
        ----------
        calculator (ase Calculator): an ase calculator
        optimizer (str or ase Optimizer): the optimization algorithm.
        relax_cell (bool): if True, cell parameters will be optimized.
        fix_symmetry (bool): if True, symmetry will be fixed during relaxation.
        symprec (float): Tolerance for symmetry finding in case of fix_symmetry.
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
        self.symprec = symprec

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
    ) -> ForcefieldResult:
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
            atoms.set_constraint(FixSymmetry(atoms, symprec=self.symprec))
        atoms.set_calculator(self.calculator)
        with contextlib.redirect_stdout(sys.stdout if verbose else io.StringIO()):
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
        traj = obs.to_pymatgen_trajectory(None)
        is_force_conv = all(
            np.linalg.norm(traj.frame_properties[-1]["forces"][idx]) < abs(fmax)
            for idx in range(len(struct))
        )
        return ForcefieldResult(
            final_structure=struct, trajectory=traj, is_force_converged=is_force_conv
        )


def ase_calculator(calculator_meta: str | dict, **kwargs: Any) -> Calculator | None:
    """
    Create an ASE calculator from a given set of metadata.

    Parameters
    ----------
    calculator_meta : str or dict
        If a str, should be one of `atomate2.forcefields.MLFF`.
        If a dict, should be decodable by `monty.json.MontyDecoder`.
        For example, one can also call the CHGNet calculator as follows
        ```
            calculator_meta = {
                "@module": "chgnet.model.dynamics",
                "@callable": "CHGNetCalculator"
            }
        ```
    args : optional args to pass to a calculator
    kwargs : optional kwargs to pass to a calculator

    Returns
    -------
    ASE .Calculator
    """
    calculator = None

    if isinstance(calculator_meta, str) and calculator_meta in map(str, MLFF):
        calculator_name = MLFF(calculator_meta.split("MLFF.")[-1])

        if calculator_name == MLFF.CHGNet:
            from chgnet.model.dynamics import CHGNetCalculator

            calculator = CHGNetCalculator(**kwargs)

        elif calculator_name == MLFF.M3GNet:
            import matgl
            from matgl.ext.ase import PESCalculator

            path = kwargs.get("path", "M3GNet-MP-2021.2.8-PES")
            potential = matgl.load_model(path)
            calculator = PESCalculator(potential, **kwargs)

        elif calculator_name == MLFF.MACE:
            from mace.calculators import mace_mp

            calculator = mace_mp(**kwargs)

        elif calculator_name == MLFF.GAP:
            from quippy.potential import Potential

            calculator = Potential(**kwargs)

        elif calculator_name == MLFF.NEP:
            from calorine.calculators import CPUNEP

            calculator = CPUNEP(**kwargs)

        elif calculator_name == MLFF.Nequip:
            from nequip.ase import NequIPCalculator

            calculator = NequIPCalculator.from_deployed_model(**kwargs)

        elif calculator_name == MLFF.SevenNet:
            from sevenn.sevennet_calculator import SevenNetCalculator

            calculator = SevenNetCalculator(**{"model": "7net-0"} | kwargs)

    elif isinstance(calculator_meta, dict):
        calc_cls = MontyDecoder().decode(json.dumps(calculator_meta))
        calculator = calc_cls(**kwargs)

    return calculator


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
