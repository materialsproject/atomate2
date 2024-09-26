"""Makers to perform MD with the Atomic Simulation Environment (ASE)."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np
from ase import units
from ase.md.md import MolecularDynamics
from ase.md.npt import NPT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from emmet.core.vasp.calculation import StoreTrajectoryOption
from jobflow import job
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.interpolate import interp1d
from scipy.linalg import schur

from atomate2.ase.jobs import _ASE_DATA_OBJECTS, AseMaker
from atomate2.ase.schemas import AseResult, AseTaskDoc
from atomate2.ase.utils import TrajectoryObserver

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from ase.calculators.calculator import Calculator

    from atomate2.ase.schemas import AseMoleculeTaskDoc, AseStructureTaskDoc


class MDEnsemble(Enum):
    """Define known MD ensembles."""

    nve = "nve"
    nvt = "nvt"
    npt = "npt"


class DynamicsPresets(Enum):
    """Define mappings between MD ensembles and thermo-/baro-stats."""

    nve_velocityverlet = "ase.md.verlet.VelocityVerlet"
    nvt_andersen = "ase.md.andersen.Andersen"
    nvt_berendsen = "ase.md.nvtberendsen.NVTBerendsen"
    nvt_langevin = "ase.md.langevin.Langevin"
    nvt_nose_hoover = "ase.md.npt.NPT"
    npt_berendsen = "ase.md.nptberendsen.NPTBerendsen"
    npt_nose_hoover = "ase.md.npt.NPT"  # noqa: PIE796


default_dynamics = {
    MDEnsemble.nve: "velocityverlet",
    MDEnsemble.nvt: "langevin",
    MDEnsemble.npt: "nose-hoover",
}

_valid_dynamics: dict[MDEnsemble, set[str]] = {}
for preset in DynamicsPresets.__members__:
    ensemble = MDEnsemble(preset.split("_")[0])
    thermostat = "-".join(preset.split("_")[1:])
    if ensemble not in _valid_dynamics:
        _valid_dynamics[ensemble] = set()
    _valid_dynamics[ensemble].add(thermostat)


@dataclass
class AseMDMaker(AseMaker):
    """
    Perform MD with the Atomic Simulation Environment (ASE).

    Note the the following units are consistent with the VASP MD implementation:
    - `temperature` in Kelvin (TEBEG and TEEND)
    - `time_step` in femtoseconds (POTIM)
    - `pressure` in kB (PSTRESS)

    The default dynamics is Langevin NVT consistent with VASP MD, with the friction
    coefficient set to 10 ps^-1 (LANGEVIN_GAMMA).

    For the rest of preset dynamics (`_valid_dynamics`) and custom dynamics inherited
    from ASE (`MolecularDynamics`), the user can specify the dynamics as a string or an
    ASE class into the `dynamics` attribute. In this case, please consult the ASE
    documentation for the parameters and units to pass into the ASE .MolecularDynamics
    function through `ase_md_kwargs`.

    Parameters
    ----------
    name : str
        The name of the MD Maker
    time_step : float | None = None.
        The timestep of the MD run in fs.
        If `None`, defaults to 0.5 fs if a structure/molecule contains an isotope of
        hydrogen and 2 fs otherwise.
    n_steps : int = 1000
        The number of MD steps to run
    ensemble : MDEnsemble | str = MDEnsemble.nvt
        The ensemble to use. Valid ensembles are nve, nvt, or npt
    temperature: float | Sequence | np.ndarray | None.
        The temperature in Kelvin. If a sequence or 1D array, the temperature
        schedule will be interpolated linearly between the given values. If a
        float, the temperature will be constant throughout the run.
    pressure: float | Sequence | None = None
        The pressure in kilobar. If a sequence or 1D array, the pressure
        schedule will be interpolated linearly between the given values. If a
        float, the pressure will be constant throughout the run.
    dynamics : str | ASE .MolecularDynamics | None = None
        The dynamical thermostat to use. If dynamics is an ASE .MolecularDynamics
        object, this uses the option specified explicitly by the user.
        See _valid_dynamics for a list of pre-defined options when
        specifying dynamics as a string.
    ase_md_kwargs : dict | None = None
        Options except for temperature and pressure to pass into the ASE
        .MolecularDynamics function
    calculator_kwargs : dict
        kwargs to pass to the ASE calculator class
    ionic_step_data : tuple[str,...] or None
        Quantities to store in the TaskDocument ionic_steps.
        Possible options are "struct_or_mol", "energy",
        "forces", "stress", and "magmoms".
        "structure" and "molecule" are aliases for "struct_or_mol".
    store_trajectory : emmet .StoreTrajectoryOption = "partial"
        Whether to store trajectory information ("no") or complete trajectories
        ("partial" or "full", which are identical).
    tags : list[str] or None
        A list of tags for the task.
    traj_file : str | Path | None = None
        If a str or Path, the name of the file to save the MD trajectory to.
        If None, the trajectory is not written to disk
    traj_file_fmt : Literal["ase","pmg","xdatcar"]
        The format of the trajectory file to write.
        If "ase", writes an ASE .Trajectory.
        If "pmg", writes a Pymatgen .Trajectory.
        If "xdatcar, writes a VASP-style XDATCAR
    traj_interval : int
        The step interval for saving the trajectories.
    mb_velocity_seed : int or None
        If an int, a random number seed for generating initial velocities
        from a Maxwell-Boltzmann distribution.
    zero_linear_momentum : bool = False
        Whether to initialize the atomic velocities with zero linear momentum
    zero_angular_momentum : bool = False
        Whether to initialize the atomic velocities with zero angular momentum
    verbose : bool = False
        Whether to print stdout to screen during the MD run.
    """

    name: str = "ASE MD"
    time_step: float | None = None
    n_steps: int = 1000
    ensemble: MDEnsemble = MDEnsemble.nvt
    dynamics: str | MolecularDynamics | None = None
    temperature: float | Sequence | np.ndarray | None = 300.0
    pressure: float | Sequence | np.ndarray | None = None
    ase_md_kwargs: dict | None = None
    calculator_kwargs: dict = field(default_factory=dict)
    ionic_step_data: tuple[str, ...] | None = None
    store_trajectory: StoreTrajectoryOption = StoreTrajectoryOption.PARTIAL
    traj_file: str | Path | None = None
    traj_file_fmt: Literal["pmg", "ase"] = "ase"
    traj_interval: int = 1
    mb_velocity_seed: int | None = None
    zero_linear_momentum: bool = False
    zero_angular_momentum: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        """Ensure that ensemble is an enum."""
        if isinstance(self.ensemble, str):
            self.ensemble = MDEnsemble(self.ensemble)

    @staticmethod
    def _interpolate_quantity(values: Sequence | np.ndarray, n_pts: int) -> np.ndarray:
        """Interpolate temperature / pressure on a schedule."""
        n_vals = len(values)
        return np.interp(
            np.linspace(0, n_vals - 1, n_pts + 1),
            np.linspace(0, n_vals - 1, n_vals),
            values,
        )

    def _get_ensemble_schedule(self) -> None:
        if self.ensemble == MDEnsemble.nve:
            # Disable thermostat and barostat
            self.temperature = np.nan
            self.pressure = np.nan
            self.t_schedule = np.full(self.n_steps + 1, self.temperature)
            self.p_schedule = np.full(self.n_steps + 1, self.pressure)
            return

        if isinstance(self.temperature, Sequence) or (
            isinstance(self.temperature, np.ndarray) and self.temperature.ndim == 1
        ):
            self.t_schedule = self._interpolate_quantity(self.temperature, self.n_steps)
        # NOTE: In ASE Langevin dynamics, the temperature are normally
        # scalars, but in principle one quantity per atom could be specified by giving
        # an array. This is not implemented yet here.
        else:
            self.t_schedule = np.full(self.n_steps + 1, self.temperature)

        if self.ensemble == MDEnsemble.nvt:
            self.pressure = np.nan
            self.p_schedule = np.full(self.n_steps + 1, self.pressure)
            return

        if isinstance(self.pressure, Sequence) or (
            isinstance(self.pressure, np.ndarray) and self.pressure.ndim == 1
        ):
            self.p_schedule = self._interpolate_quantity(self.pressure, self.n_steps)
        elif isinstance(self.pressure, np.ndarray) and self.pressure.ndim == 4:
            self.p_schedule = interp1d(
                np.arange(self.n_steps + 1), self.pressure, kind="linear"
            )
        else:
            self.p_schedule = np.full(self.n_steps + 1, self.pressure)

    def _get_ensemble_defaults(self) -> None:
        """Update ASE MD kwargs with defaults consistent with VASP MD."""
        self.ase_md_kwargs = self.ase_md_kwargs or {}
        self.dynamics = self.dynamics or default_dynamics[self.ensemble]

        if self.ensemble == MDEnsemble.nve:
            self.ase_md_kwargs.pop("temperature", None)
            self.ase_md_kwargs.pop("temperature_K", None)
            self.ase_md_kwargs.pop("externalstress", None)
        elif self.ensemble == MDEnsemble.nvt:
            self.ase_md_kwargs["temperature_K"] = self.t_schedule[0]
            self.ase_md_kwargs.pop("externalstress", None)
        elif self.ensemble == MDEnsemble.npt:
            self.ase_md_kwargs["temperature_K"] = self.t_schedule[0]
            self.ase_md_kwargs["externalstress"] = self.p_schedule[0] * 1e3 * units.bar

        if isinstance(self.dynamics, str) and self.dynamics.lower() == "langevin":
            self.ase_md_kwargs["friction"] = self.ase_md_kwargs.get(
                "friction",
                10.0 * 1e-3 / units.fs,  # Same default as in VASP: 10 ps^-1
            )

    @job(data=[*_ASE_DATA_OBJECTS, "ionic_steps"])
    def make(
        self,
        mol_or_struct: Molecule | Structure,
        prev_dir: str | Path | None = None,
    ) -> AseStructureTaskDoc | AseMoleculeTaskDoc:
        """
        Perform MD on a structure using ASE and jobflow.

        Parameters
        ----------
        mol_or_struct: .Molecule or .Structure
            pymatgen molecule or structure
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from. Unused, just
            added to match the method signature of other makers.
        """
        return AseTaskDoc.to_mol_or_struct_metadata_doc(
            getattr(self.calculator, "name", type(self.calculator).__name__),
            self.run_ase(mol_or_struct, prev_dir=prev_dir),
            steps=self.n_steps,
            relax_kwargs=None,
            optimizer_kwargs=None,
            fix_symmetry=False,
            symprec=None,
            ionic_step_data=self.ionic_step_data,
            store_trajectory=self.store_trajectory,
            tags=self.tags,
        )

    def run_ase(
        self,
        mol_or_struct: Molecule | Structure,
        prev_dir: str | Path | None = None,
    ) -> AseResult:
        """
        Perform MD on a structure using ASE without jobflow.

        This function is implemented to permit different schemas on output.
        See for example, the forcefield MD jobs.

        Parameters
        ----------
        mol_or_struct: .Molecule or .Structure
            pymatgen molecule or structure
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from. Unused, just
            added to match the method signature of other makers.
        """
        self._get_ensemble_schedule()
        self._get_ensemble_defaults()

        if self.time_step is None:
            # If a mol_or_struct contains an isotope of hydrogen,
            # set default `time_step` to 0.5 fs, and 2 fs otherwise.
            has_h_isotope = any(element.Z == 1 for element in mol_or_struct.composition)
            self.time_step = 0.5 if has_h_isotope else 2.0

        initial_velocities = mol_or_struct.site_properties.get("velocities")

        if isinstance(self.dynamics, str):
            # Use known dynamics if `self.dynamics` is a str
            self.dynamics = self.dynamics.lower()
            if self.dynamics not in _valid_dynamics[self.ensemble]:
                raise ValueError(
                    f"{self.dynamics} thermostat not available for "
                    f"{self.ensemble.value}."
                    f"Available {self.ensemble.value} thermostats are:"
                    " ".join(_valid_dynamics[self.ensemble])
                )

            _dyn_mod_path = DynamicsPresets[
                f"{self.ensemble.value}_{self.dynamics.replace('-', '_')}"
            ].value.split(".")
            dynamics = getattr(
                import_module(".".join(_dyn_mod_path[:-1])), _dyn_mod_path[-1]
            )

        elif issubclass(self.dynamics, MolecularDynamics):
            # Allow user to explicitly run ASE Dynamics class
            dynamics = self.dynamics

        atoms = mol_or_struct.to_ase_atoms()

        if dynamics is NPT:
            # Note that until dynamics is instantiated,
            # `isinstance(dynamics,NPT)` is False

            # ASE NPT implementation requires upper triangular cell
            schur_decomp, _ = schur(atoms.get_cell(complete=True), output="complex")
            atoms.set_cell(schur_decomp.real, scale_atoms=True)

        if initial_velocities:
            atoms.set_velocities(initial_velocities)
        elif not np.isnan(self.t_schedule).any():
            MaxwellBoltzmannDistribution(
                atoms=atoms,
                temperature_K=self.t_schedule[0],
                rng=np.random.default_rng(seed=self.mb_velocity_seed),
            )
            if self.zero_linear_momentum:
                Stationary(atoms)
            if self.zero_angular_momentum:
                ZeroRotation(atoms)

        atoms.calc = self.calculator

        md_observer = TrajectoryObserver(atoms, store_md_outputs=True)

        md_runner = dynamics(
            atoms=atoms, timestep=self.time_step * units.fs, **self.ase_md_kwargs
        )

        md_runner.attach(md_observer, interval=self.traj_interval)

        def _callback(dyn: MolecularDynamics = md_runner) -> None:
            if self.ensemble == MDEnsemble.nve:
                return
            dyn.set_temperature(temperature_K=self.t_schedule[dyn.nsteps])
            if self.ensemble == MDEnsemble.nvt:
                return
            dyn.set_stress(self.p_schedule[dyn.nsteps] * 1e3 * units.bar)

        md_runner.attach(_callback, interval=1)
        with contextlib.redirect_stdout(sys.stdout if self.verbose else io.StringIO()):
            t_i = time.perf_counter()
            md_runner.run(steps=self.n_steps)
            t_f = time.perf_counter()

        if self.traj_file is not None:
            md_observer.save(filename=self.traj_file, fmt=self.traj_file_fmt)

        mol_or_struct = AseAtomsAdaptor.get_structure(
            atoms,
            cls=Structure if isinstance(mol_or_struct, Structure) else Molecule,
        )

        return AseResult(
            final_mol_or_struct=mol_or_struct,
            trajectory=md_observer.to_pymatgen_trajectory(filename=None),
            dir_name=os.getcwd(),
            elapsed_time=t_f - t_i,
        )

    @property
    def calculator(self) -> Calculator:
        """ASE calculator, to be overwritten by user."""
        raise NotImplementedError


@dataclass
class LennardJonesMDMaker(AseMDMaker):
    """
    Perform an MD run with a Lennard-Jones 6-12 potential.

    See `atomate2.ase.md.AseMDMaker` for full documentation.
    """

    name: str = "Lennard-Jones 6-12 MD"

    @property
    def calculator(self) -> Calculator:
        """Lennard-Jones calculator."""
        from ase.calculators.lj import LennardJones

        return LennardJones(**self.calculator_kwargs)


@dataclass
class GFNxTBMDMaker(AseMDMaker):
    """
    Perform an MD run with GFNn-xTB.

    See `atomate2.ase.md.AseMDMaker` for full documentation.
    """

    name: str = "GFNn-xTB MD"
    calculator_kwargs: dict = field(
        default_factory=lambda: {
            "method": "GFN1-xTB",
            "charge": None,
            "multiplicity": None,
            "accuracy": 1.0,
            "guess": "sad",
            "max_iterations": 250,
            "mixer_damping": 0.4,
            "electric_field": None,
            "spin_polarization": None,
            "electronic_temperature": 300.0,
            "cache_api": True,
            "verbosity": 1,
        }
    )

    @property
    def calculator(self) -> Calculator:
        """GFN-xTB / TBLite calculator."""
        try:
            from tblite.ase import TBLite
        except ImportError:
            raise ImportError(
                "TBLite must be installed; please install TBLite using\n"
                "`pip install -c conda-forge tblite-python`"
            ) from None

        return TBLite(atoms=None, **self.calculator_kwargs)
