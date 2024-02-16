"""Makers to perform MD with forcefields."""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ase.md.andersen import Andersen
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet
from ase.units import bar, fs
from jobflow import Maker, job
from pymatgen.io.ase import AseAtomsAdaptor

from atomate2.forcefields.schemas import ForceFieldTaskDocument
from atomate2.forcefields.utils import TrajectoryObserver

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from ase.calculators.calculator import Calculator
    from pymatgen.core.structure import Structure


_valid_thermostats: dict[str, tuple[str, ...]] = {
    "nve": ("velocityverlet",),
    "nvt": ("nose-hoover", "langevin", "andersen", "berendsen"),
    "npt": ("nose-hoover", "berendsen"),
}

_thermostats: dict = {
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
    Perform MD with a forcefield.

    Note that units are consistent with the VASP MD implementation.
    - Temperature is in Kelvin (TEBEG and TEEND)
    - Time steps are in femtoseconds (POTIM)
    - Langevin NVT friction coefficients are in picoseconds^-1 (LANGEVIN_GAMMA)
    - Pressure in kB (PSTRESS)

    Parameters
    ----------
    name : str
        The name of the MD Maker
    force_field_name : str
        The name of the forcefield (for provenance)
    timestep : float | None = 2.
        The timestep of the MD run in femtoseconds
    md_steps : int = 1000
        The number of MD steps to run
    ensemble : str = "nvt"
        The ensemble to use. Valid ensembles are nve, nvt, or npt
    start_temperature : float | None = 300.
        The temperature to initialize the system, in Kelvin.
    end_temperature : float | None = 300.
        The temperature to equilibrate towards, in Kelvin.
        If start_temperature is a float and end_temperature is None,
        the system will be initialized at and equilibrated towards
        the start_temperature.
    thermostat : str = "langevin"
        The thermostat to use. See _valid_thermostats for a list of options
    ase_md_kwargs : dict | None = None
        Options to pass to the ASE MD function
    calculator_args : Sequence | None = None
        args to pass to the ASE calculator class
    calculator_kwargs : dict | None = None
        kwargs to pass to the ASE calculator class
    traj_file : str | Path | None = None
        If a str or Path, the name of the file to save the MD trajectory to.
        If None, the trajectory is not written to disk
    traj_interval : int
        The step interval for saving the trajectories.
    zero_linear_momentum : bool = False
        Whether to initialize the atomic velocities with zero linear momentum
    zero_angular_momentum : bool = False
        Whether to initialize the atomic velocities with zero angular momentum
    task_document_kwargs: dict
        Options to pass to the TaskDoc. Default choice
        {"store_trajectory": "partial"}
        is consistent with atomate2.vasp.md.MDMaker
    """

    name: str = "Forcefield MD"
    force_field_name: str = "Forcefield"
    timestep: float | None = 2.0
    md_steps: int = 1000
    ensemble: Literal["nve", "nvt", "npt"] = "nvt"
    start_temperature: float | None = 300.0
    end_temperature: float | None = 300.0
    pressure: float | None = None
    thermostat: str = "langevin"
    ase_md_kwargs: dict | None = None
    calculator_args: list | tuple | None = None
    calculator_kwargs: dict | None = None
    traj_file: str | Path | None = None
    traj_interval: int = 1
    zero_linear_momentum: bool = False
    zero_angular_momentum: bool = False
    task_document_kwargs: dict = field(
        default_factory=lambda: {"store_trajectory": "partial"}
    )

    def _get_ensemble_defaults(self, structure: Structure) -> None:
        """Update ASE MD kwargs with defaults consistent with VASP MD."""
        self.ase_md_kwargs = self.ase_md_kwargs or {}
        if self.ensemble in ("nvt", "npt") and all(
            self.ase_md_kwargs.get(key) is None
            for key in ("temperature_K", "temperature")
        ):
            self.ase_md_kwargs["temperature_K"] = (
                self.end_temperature if self.end_temperature else self.start_temperature
            )

        if self.ensemble == "npt":
            # convert from kilobar to eV/Ang**3
            self.ase_md_kwargs["pressure_au"] = (
                self.pressure * 1.0e-3 / bar if self.pressure else 0.0
            )

        if self.thermostat == "langevin":
            # Same default as in VASP
            self.ase_md_kwargs["friction"] = self.ase_md_kwargs.get("friction", 10.0)
            # friction coefficient(s) specified in ev^-1, convert from picoseconds
            if isinstance(self.ase_md_kwargs["friction"], (list, tuple)):
                self.ase_md_kwargs["friction"] = [
                    coeff * 1.0e-3 / fs for coeff in self.ase_md_kwargs["friction"]
                ]
            else:
                self.ase_md_kwargs["friction"] *= 1.0e-3 / fs

    @job(output_schema=ForceFieldTaskDocument)
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
        self._get_ensemble_defaults(structure=structure)

        initial_velocities = structure.site_properties.get("velocities")

        self.thermostat = self.thermostat.lower()
        if self.thermostat not in _valid_thermostats[self.ensemble]:
            raise ValueError(
                f"{self.thermostat} thermostat not available for {self.ensemble}."
                f"Available {self.ensemble} thermostats are:"
                " ".join(_valid_thermostats[self.ensemble])
            )

        if self.ensemble == "nve" and self.thermostat is None:
            self.thermostat = "velocityverlet"
        md_func = _thermostats[f"{self.ensemble}_{self.thermostat}"]

        atoms = AseAtomsAdaptor.get_atoms(structure)
        if initial_velocities:
            atoms.set_velocities(initial_velocities)

        elif self.start_temperature:
            MaxwellBoltzmannDistribution(
                atoms=atoms, temperature_K=self.start_temperature
            )

            if self.zero_linear_momentum:
                Stationary(atoms)

            if self.zero_angular_momentum:
                ZeroRotation(atoms)

        else:
            atoms.set_velocities(
                [[0.0 for _ in range(3)] for _ in range(len(structure))]
            )

        self.calculator_args = self.calculator_args or []
        self.calculator_kwargs = self.calculator_kwargs or {}
        atoms.set_calculator(self._calculator())

        with contextlib.redirect_stdout(io.StringIO()):
            md_observer = TrajectoryObserver(atoms, store_md_outputs=True)

            md_runner = md_func(
                atoms=atoms, timestep=self.timestep * fs, **self.ase_md_kwargs
            )
            md_runner.attach(md_observer, interval=self.traj_interval)

            md_runner.run(steps=self.md_steps)
            md_observer()

            if self.traj_file is not None:
                md_observer.save(self.traj_file)

        structure = AseAtomsAdaptor.get_structure(atoms)

        self.task_document_kwargs = self.task_document_kwargs or {}
        return ForceFieldTaskDocument.from_ase_compatible_result(
            self.force_field_name,
            {"final_structure": structure, "trajectory": md_observer},
            relax_cell=(self.ensemble == "npt"),
            steps=self.md_steps,
            relax_kwargs=None,
            optimizer_kwargs=None,
            **self.task_document_kwargs,
        )

    def _calculator(self) -> Calculator:
        """To be implemented by the user."""
        return NotImplementedError


class MACEMDMaker(ForceFieldMDMaker):
    """Perform an MD run with MACE."""

    name: str = "MACE MD"
    force_field_name: str = "MACE"
    calculator_kwargs: dict = field(default_factory=lambda: {"dtype": "float32"})

    def _calculator(self) -> Calculator:
        from mace.calculators import mace_mp

        return mace_mp(*self.calculator_args, **self.calculator_kwargs)


class M3GNetMDMaker(ForceFieldMDMaker):
    """Perform an MD run with M3GNet."""

    name: str = "M3GNet MD"
    force_field_name: str = "M3GNet"

    def _calculator(self) -> Calculator:
        import matgl
        from matgl.ext.ase import PESCalculator

        potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        return PESCalculator(potential, **self.calculator_kwargs)


class CHGNetMDMaker(ForceFieldMDMaker):
    """Perform an MD run with CHGNet."""

    name: str = "CHGNet MD"
    force_field_name: str = "CHGNet"

    def _calculator(self) -> Calculator:
        from chgnet.model.dynamics import CHGNetCalculator

        return CHGNetCalculator(*self.calculator_args, **self.calculator_kwargs)
