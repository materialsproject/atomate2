"""Makers to perform MD with forcefields."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import job
from monty.dev import deprecated

from atomate2.ase.md import AseMDMaker, MDEnsemble
from atomate2.forcefields import MLFF, _get_formatted_ff_name
from atomate2.forcefields.jobs import (
    _DEFAULT_CALCULATOR_KWARGS,
    _FORCEFIELD_DATA_OBJECTS,
)
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
    documentation for the parameters and units to pass into the ASE .MolecularDynamics
    function through `ase_md_kwargs`.

    Parameters
    ----------
    name : str
        The name of the MD Maker
    force_field_name : str or .MLFF
        The name of the forcefield (for provenance)
    time_step : float | None = None.
        The timestep of the MD run in fs.
        If `None`, defaults to 0.5 fs if a structure contains an isotope of
        hydrogen and 2 fs otherwise.
    n_steps : int = 1000
        The number of MD steps to run
    ensemble : MDEnsemble = "nvt"
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
    task_document_kwargs: dict or None (deprecated)
        Options to pass to the TaskDoc.
    """

    name: str = "Forcefield MD"
    force_field_name: str | MLFF = MLFF.Forcefield
    task_document_kwargs: dict = None

    def __post_init__(self) -> None:
        """Ensure that force_field_name is correctly assigned."""
        super().__post_init__()
        self.force_field_name = _get_formatted_ff_name(self.force_field_name)

        # Pad calculator_kwargs with default values, but permit user to override them
        self.calculator_kwargs = {
            **_DEFAULT_CALCULATOR_KWARGS.get(
                MLFF(self.force_field_name.split("MLFF.")[-1]), {}
            ),
            **self.calculator_kwargs,
        }

    @job(
        data=[*_FORCEFIELD_DATA_OBJECTS, "ionic_steps"],
        output_schema=ForceFieldTaskDocument,
    )
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
            md_result = self.run_ase(structure, prev_dir=prev_dir)

        self.task_document_kwargs = self.task_document_kwargs or {}
        if len(self.task_document_kwargs) > 0:
            warnings.warn(
                "`task_document_kwargs` is now deprecated, please use the top-level "
                "attributes `ionic_step_data` and `store_trajectory`",
                category=DeprecationWarning,
                stacklevel=1,
            )

        return ForceFieldTaskDocument.from_ase_compatible_result(
            str(self.force_field_name),  # make mypy happy
            md_result,
            relax_cell=(self.ensemble == MDEnsemble.npt),
            steps=self.n_steps,
            relax_kwargs=None,
            optimizer_kwargs=None,
            fix_symmetry=False,
            symprec=None,
            ionic_step_data=self.ionic_step_data,
            store_trajectory=self.store_trajectory,
            tags=self.tags,
            **self.task_document_kwargs,
        )

    @property
    def calculator(self) -> Calculator:
        """ASE calculator, can be overwritten by user."""
        return ase_calculator(
            str(self.force_field_name),  # make mypy happy
            **self.calculator_kwargs,
        )


@deprecated(
    replacement=ForceFieldMDMaker,
    deadline=(2025, 1, 1),
    message="To use NEP, set `force_field_name = 'NEP'` in ForceFieldMDMaker.",
)
@dataclass
class NEPMDMaker(ForceFieldMDMaker):
    """Perform an MD run with NEP."""

    name: str = f"{MLFF.NEP} MD"
    force_field_name: str | MLFF = MLFF.NEP
    calculator_kwargs: dict = field(
        default_factory=lambda: _DEFAULT_CALCULATOR_KWARGS[MLFF.NEP]
    )


@deprecated(
    replacement=ForceFieldMDMaker,
    deadline=(2025, 1, 1),
    message="To use MACE, set `force_field_name = 'MACE'` in ForceFieldMDMaker.",
)
@dataclass
class MACEMDMaker(ForceFieldMDMaker):
    """Perform an MD run with MACE."""

    name: str = f"{MLFF.MACE} MD"
    force_field_name: str | MLFF = MLFF.MACE
    calculator_kwargs: dict = field(
        default_factory=lambda: {"default_dtype": "float32"}
    )


@deprecated(
    replacement=ForceFieldMDMaker,
    deadline=(2025, 1, 1),
    message="To use M3GNet, set `force_field_name = 'M3GNet'` in ForceFieldMDMaker.",
)
@dataclass
class M3GNetMDMaker(ForceFieldMDMaker):
    """Perform an MD run with M3GNet."""

    name: str = f"{MLFF.M3GNet} MD"
    force_field_name: str | MLFF = MLFF.M3GNet


@deprecated(
    replacement=ForceFieldMDMaker,
    deadline=(2025, 1, 1),
    message="To use CHGNet, set `force_field_name = 'CHGNet'` in ForceFieldMDMaker.",
)
@dataclass
class CHGNetMDMaker(ForceFieldMDMaker):
    """Perform an MD run with CHGNet."""

    name: str = f"{MLFF.CHGNet} MD"
    force_field_name: str | MLFF = MLFF.CHGNet


@deprecated(
    replacement=ForceFieldMDMaker,
    deadline=(2025, 1, 1),
    message="To use GAP, set `force_field_name = 'GAP'` in ForceFieldMDMaker.",
)
@dataclass
class GAPMDMaker(ForceFieldMDMaker):
    """Perform an MD run with GAP."""

    name: str = f"{MLFF.GAP} MD"
    force_field_name: str | MLFF = MLFF.GAP
    calculator_kwargs: dict = field(
        default_factory=lambda: _DEFAULT_CALCULATOR_KWARGS[MLFF.GAP]
    )


@deprecated(
    replacement=ForceFieldMDMaker,
    deadline=(2025, 1, 1),
    message="To use Nequip, set `force_field_name = 'Nequip'` in ForceFieldMDMaker.",
)
@dataclass
class NequipMDMaker(ForceFieldMDMaker):
    """Perform an MD run with nequip."""

    name: str = f"{MLFF.Nequip} MD"
    force_field_name: str | MLFF = MLFF.Nequip
