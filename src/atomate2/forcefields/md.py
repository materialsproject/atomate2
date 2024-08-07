"""Makers to perform MD with forcefields."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.ase.md import AseMDMaker
from atomate2.forcefields import MLFF
from atomate2.forcefields.jobs import forcefield_job
from atomate2.forcefields.schemas import ForceFieldTaskDocument
from atomate2.forcefields.utils import ase_calculator, revert_default_dtype

if TYPE_CHECKING:
    from pathlib import Path

    from ase.calculators.calculator import Calculator
    from pymatgen.core.structure import Structure

@dataclass
class ForceFieldMDMaker(AseMDMaker):
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

    @forcefield_job
    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ) -> ForceFieldTaskDocument:
        """
        Perform MD on a structure using forcefields and jobflow.

        Parameters
        ----------
        structure: .Structure
            pymatgen structure.
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from. Unused, just
            added to match the method signature of other makers.
        """
        with revert_default_dtype():
            md_result = self._make(structure,prev_dir=prev_dir)

        self.task_document_kwargs = self.task_document_kwargs or {}

        return ForceFieldTaskDocument.from_ase_compatible_result(
            self.force_field_name,
            md_result,
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
