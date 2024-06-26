"""Makers to perform MD with forcefields."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from ase import units
from ase.md.andersen import Andersen
from ase.md.langevin import Langevin
from ase.md.md import MolecularDynamics
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet
from jobflow import Maker
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.interpolate import interp1d
from scipy.linalg import schur

from atomate2.forcefields import MLFF
from atomate2.forcefields.jobs import forcefield_job
from atomate2.forcefields.schemas import ForcefieldResult, ForceFieldTaskDocument
from atomate2.forcefields.utils import (
    TrajectoryObserver,
    ase_calculator,
    revert_default_dtype,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from ase.calculators.calculator import Calculator
    from pymatgen.core.structure import Structure

_valid_dynamics: dict[str, tuple[str, ...]] = {
    "nve": ("velocityverlet",),
    "nvt": ("nose-hoover", "langevin", "andersen", "berendsen"),
    "npt": ("nose-hoover", "berendsen"),
}

_preset_dynamics: dict = {
    "nve_velocityverlet": VelocityVerlet,
    "nvt_andersen": Andersen,
    "nvt_berendsen": NVTBerendsen,
    "nvt_langevin": Langevin,
    "nvt_nose-hoover": NPT,
    "npt_berendsen": NPTBerendsen,
    "npt_nose-hoover": NPT,
}


@dataclass
class ForceFieldMDMaker(Maker):
    """
    Perform MD with a force field.

    Note the the following units are consistent with the VASP MD implementation:
    - `temperature` in Kelvin (TEBEG and TEEND)
    - `time_step` in femtoseconds (POTIM)
    - `pressure` in kB (PSTRESS)

    The default dynamics is Langevin NVT consistent with VASP MD, with the friction
    coefficient set to 10 ps^-1 (LANGEVIN_GAMMA).

    For the rest of preset dynamics (`_valid_dynamics`) and custom dynamics inherited
    from ASE (`MolecularDynamics`), the user can specify the dynamics as a string or an
    ASE class into the `dynamics` attribute. In this case, please consult the ASE
    documentation for the parameters and units to pass into the ASE MD function through
    `ase_md_kwargs`.

    Parameters
    ----------
    name : str
        The name of the MD Maker
    force_field_name : str
        The name of the forcefield (for provenance)
    time_step : float | None = None.
        The timestep of the MD run in fs.
        If `None`, defaults to 0.5 fs if a structure contains an isotope of
        hydrogen and 2 fs otherwise.
    n_steps : int = 1000
        The number of MD steps to run
    ensemble : str = "nvt"
        The ensemble to use. Valid ensembles are nve, nvt, or npt
    temperature: float | Sequence | np.ndarray | None.
        The temperature in Kelvin. If a sequence or 1D array, the temperature
        schedule will be interpolated linearly between the given values. If a
        float, the temperature will be constant throughout the run.
    pressure: float | Sequence | None = None
        The pressure in kilobar. If a sequence or 1D array, the pressure
        schedule will be interpolated linearly between the given values. If a
        float, the pressure will be constant throughout the run.
    dynamics : str | ASE .MolecularDynamics = "langevin"
        The dynamical thermostat to use. If dynamics is an ASE .MolecularDynamics
        object, this uses the option specified explicitly by the user.
        See _valid_dynamics for a list of pre-defined options when
        specifying dynamics as a string.
    ase_md_kwargs : dict | None = None
        Options except for temperature and pressure to pass into the ASE MD function
    calculator_kwargs : dict
        kwargs to pass to the ASE calculator class
    traj_file : str | Path | None = None
        If a str or Path, the name of the file to save the MD trajectory to.
        If None, the trajectory is not written to disk
    traj_file_fmt : Literal["ase","pmg"]
        The format of the trajectory file to write. If "ase", writes an
        ASE trajectory, if "pmg", writes a Pymatgen trajectory.
    traj_interval : int
        The step interval for saving the trajectories.
    mb_velocity_seed : int | None = None
        If an int, a random number seed for generating initial velocities
        from a Maxwell-Boltzmann distribution.
    zero_linear_momentum : bool = False
        Whether to initialize the atomic velocities with zero linear momentum
    zero_angular_momentum : bool = False
        Whether to initialize the atomic velocities with zero angular momentum
    task_document_kwargs: dict
        Options to pass to the TaskDoc. Default choice
            {"store_trajectory": "partial", "ionic_step_data": ("energy",),}
        is consistent with atomate2.vasp.md.MDMaker
    """

    name: str = "Forcefield MD"
    force_field_name: str = f"{MLFF.Forcefield}"
    time_step: float | None = None
    n_steps: int = 1000
    ensemble: Literal["nve", "nvt", "npt"] = "nvt"
    dynamics: str | MolecularDynamics = "langevin"
    temperature: float | Sequence | np.ndarray | None = 300.0
    pressure: float | Sequence | np.ndarray | None = None
    ase_md_kwargs: dict | None = None
    calculator_kwargs: dict = field(default_factory=dict)
    traj_file: str | Path | None = None
    traj_file_fmt: Literal["pmg", "ase"] = "ase"
    traj_interval: int = 1
    mb_velocity_seed: int | None = None
    zero_linear_momentum: bool = False
    zero_angular_momentum: bool = False
    task_document_kwargs: dict = field(
        default_factory=lambda: {
            "store_trajectory": "partial",
            "ionic_step_data": ("energy",),  # energy is required in ionic steps
        }
    )

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
        if self.ensemble == "nve":
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

        if self.ensemble == "nvt":
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

        if self.ensemble == "nve":
            self.ase_md_kwargs.pop("temperature", None)
            self.ase_md_kwargs.pop("temperature_K", None)
            self.ase_md_kwargs.pop("externalstress", None)
        elif self.ensemble == "nvt":
            self.ase_md_kwargs["temperature_K"] = self.t_schedule[0]
            self.ase_md_kwargs.pop("externalstress", None)
        elif self.ensemble == "npt":
            self.ase_md_kwargs["temperature_K"] = self.t_schedule[0]
            self.ase_md_kwargs["externalstress"] = self.p_schedule[0] * 1e3 * units.bar

        if isinstance(self.dynamics, str) and self.dynamics.lower() == "langevin":
            self.ase_md_kwargs["friction"] = self.ase_md_kwargs.get(
                "friction",
                10.0 * 1e-3 / units.fs,  # Same default as in VASP: 10 ps^-1
            )

    @forcefield_job
    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ) -> ForceFieldTaskDocument:
        """
        Perform MD on a structure using a force field.

        Parameters
        ----------
        structure: .Structure
            pymatgen structure.
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from. Unused, just
            added to match the method signature of other makers.
        """
        self._get_ensemble_schedule()
        self._get_ensemble_defaults()

        if self.time_step is None:
            # If a structure contains an isotope of hydrogen, set default `time_step`
            # to 0.5 fs, and 2 fs otherwise.
            has_h_isotope = any(element.Z == 1 for element in structure.composition)
            self.time_step = 0.5 if has_h_isotope else 2.0

        initial_velocities = structure.site_properties.get("velocities")

        if isinstance(self.dynamics, str):
            # Use known dynamics if `self.dynamics` is a str
            self.dynamics = self.dynamics.lower()
            if self.dynamics not in _valid_dynamics[self.ensemble]:
                raise ValueError(
                    f"{self.dynamics} thermostat not available for {self.ensemble}."
                    f"Available {self.ensemble} thermostats are:"
                    " ".join(_valid_dynamics[self.ensemble])
                )

            if self.ensemble == "nve" and self.dynamics is None:
                self.dynamics = "velocityverlet"
            md_func = _preset_dynamics[f"{self.ensemble}_{self.dynamics}"]

        elif issubclass(self.dynamics, MolecularDynamics):
            # Allow user to explicitly run ASE Dynamics class
            md_func = self.dynamics

        atoms = structure.to_ase_atoms()

        if md_func is NPT:
            # Note that until md_func is instantiated, isinstance(md_func,NPT) is False
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

        with revert_default_dtype():
            atoms.calc = self.calculator

            md_observer = TrajectoryObserver(atoms, store_md_outputs=True)

            md_runner = md_func(
                atoms=atoms, timestep=self.time_step * units.fs, **self.ase_md_kwargs
            )

            md_runner.attach(md_observer, interval=self.traj_interval)

            def _callback(dyn: MolecularDynamics = md_runner) -> None:
                if self.ensemble == "nve":
                    return
                dyn.set_temperature(temperature_K=self.t_schedule[dyn.nsteps])
                if self.ensemble == "nvt":
                    return
                dyn.set_stress(self.p_schedule[dyn.nsteps] * 1e3 * units.bar)

            md_runner.attach(_callback, interval=1)
            md_runner.run(steps=self.n_steps)

            if self.traj_file is not None:
                md_observer.save(filename=self.traj_file, fmt=self.traj_file_fmt)

            structure = AseAtomsAdaptor.get_structure(atoms)

        self.task_document_kwargs = self.task_document_kwargs or {}

        return ForceFieldTaskDocument.from_ase_compatible_result(
            self.force_field_name,
            ForcefieldResult(
                final_structure=structure,
                trajectory=md_observer.to_pymatgen_trajectory(None),
            ),
            relax_cell=(self.ensemble == "npt"),
            steps=self.n_steps,
            relax_kwargs=None,
            optimizer_kwargs=None,
            **self.task_document_kwargs,
        )

    @property
    def calculator(self) -> Calculator:
        """ASE calculator, can be overwritten by user."""
        return ase_calculator(self.force_field_name, **self.calculator_kwargs)


@dataclass
class NEPMDMaker(ForceFieldMDMaker):
    """Perform an MD run with NEP."""

    name: str = f"{MLFF.NEP} MD"
    force_field_name: str = f"{MLFF.NEP}"
    calculator_kwargs: dict = field(
        default_factory=lambda: {"model_filename": "nep.txt"}
    )


@dataclass
class MACEMDMaker(ForceFieldMDMaker):
    """Perform an MD run with MACE."""

    name: str = f"{MLFF.MACE} MD"
    force_field_name: str = f"{MLFF.MACE}"
    calculator_kwargs: dict = field(
        default_factory=lambda: {"default_dtype": "float32"}
    )


@dataclass
class M3GNetMDMaker(ForceFieldMDMaker):
    """Perform an MD run with M3GNet."""

    name: str = f"{MLFF.M3GNet} MD"
    force_field_name: str = f"{MLFF.M3GNet}"


@dataclass
class CHGNetMDMaker(ForceFieldMDMaker):
    """Perform an MD run with CHGNet."""

    name: str = f"{MLFF.CHGNet} MD"
    force_field_name: str = f"{MLFF.CHGNet}"


@dataclass
class GAPMDMaker(ForceFieldMDMaker):
    """Perform an MD run with GAP."""

    name: str = f"{MLFF.GAP} MD"
    force_field_name: str = f"{MLFF.GAP}"
    calculator_kwargs: dict = field(
        default_factory=lambda: {"args_str": "IP GAP", "param_filename": "gap.xml"}
    )


@dataclass
class NequipMDMaker(ForceFieldMDMaker):
    """Perform an MD run with nequip."""

    name: str = f"{MLFF.Nequip} MD"
    force_field_name: str = f"{MLFF.Nequip}"
