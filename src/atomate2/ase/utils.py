"""Utils for accessing Atomic Simulation Environment calculators."""

from __future__ import annotations

import contextlib
import io
import sys
import warnings
from typing import TYPE_CHECKING

import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixSymmetry
from ase.io import Trajectory as AseTrajectory
from ase.optimize import BFGS, FIRE, LBFGS, BFGSLineSearch, LBFGSLineSearch, MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from monty.serialization import dumpfn
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.trajectory import Trajectory as PmgTrajectory
from pymatgen.io.ase import AseAtomsAdaptor

from atomate2.ase.schemas import AseResult

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
    from os import PathLike
    from typing import Literal

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

        to_singular = {"energies": "energy", "stresses": "stress"}

        traj = self.as_dict() if hasattr(self, "as_dict") else self.__dict__

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

        pmg_traj = PmgTrajectory.from_structures(
            structures,
            frame_properties=frame_properties,
            constant_lattice=False,
        )

        if filename:
            dumpfn(pmg_traj, filename)

        return pmg_traj

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


class AseRelaxer:
    """Relax a structure using the Atomic Simulation Environment."""

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
    ) -> AseResult:
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
        return AseResult(
            final_structure=struct,
            trajectory=traj,
            is_force_converged=is_force_conv,
            energy_downhill=traj.frame_properties[-1]["energy"]
            < traj.frame_properties[0]["energy"],
        )
