"""Define the force field makers for finite-temperature phonons."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from ase import units

from atomate2.ase.md import MDEnsemble
from atomate2.common.flows.finite_temperature_phonons import (
    BaseFiniteTemperaturePhononMaker,
)
from atomate2.common.jobs.finite_temperature_phonons import ASE_TRAJECTORY_FILE
from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker
from atomate2.forcefields.md import ForceFieldMDMaker
from atomate2.vasp.flows.finite_temperature_phonons import FiniteTemperaturePhononMaker

if TYPE_CHECKING:
    from jobflow import Maker
    from typing_extensions import Self

    from atomate2.forcefields import MLFF

_DEFAULT_FORCE_FIELD = "MACE-MP-0"

# VASP sets the Nose mass for SMASS = 0 so that the temperature oscillates with a
# period of about 40 time steps. With ASE's thermostat mass Q = 3 N k_B T tdamp**2, the
# linearised Nose-Hoover period is close to pi * sqrt(2) * tdamp.
_NOSE_HOOVER_TDAMP_STEPS = 40 / (math.pi * math.sqrt(2))
# time constant of the barostat of the NPT MD
_BAROSTAT_PDAMP_STEPS = 1000


def _get_md_maker(
    force_field_name: str | MLFF | dict, calculator_kwargs: dict | None = None
) -> ForceFieldMDMaker:
    """Get the force field MD maker, whose settings the flow sets."""
    return ForceFieldMDMaker(
        force_field_name=force_field_name,
        calculator_kwargs=calculator_kwargs or {},
        store_trajectory="no",
        ionic_step_data=("energy",),
    )


def _get_force_field_md_maker(
    maker: BaseFiniteTemperaturePhononMaker,
    n_steps: int,
    index: int = 0,
    npt: bool = False,
) -> ForceFieldMDMaker:
    """
    Get the force field MD maker of one MD job of a finite-temperature flow.

    See :obj:`ForceFieldFiniteTemperaturePhononMaker` for the MD settings.

    Parameters
    ----------
    maker: .BaseFiniteTemperaturePhononMaker
        The finite-temperature phonon maker.
    n_steps: int
        Number of MD steps of the job.
    index: int
        Position of the job in the MD, counted from 0. It is added to
        maker.random_seed. The NPT MD uses maker.md_runs instead.
    npt: bool
        If True, get the NPT MD maker from maker.npt_maker. If False, get the
        NVT MD maker from maker.md_maker.

    Returns
    -------
    ForceFieldMDMaker
    """
    md_maker = maker.npt_maker if npt else maker.md_maker
    name = "NPT maker" if npt else "MD maker"
    if not isinstance(md_maker, ForceFieldMDMaker):
        raise TypeError(
            f"The {name} must be a ForceFieldMDMaker, not {type(md_maker).__name__}."
        )
    if md_maker.dynamics is not None or md_maker.ase_md_kwargs:
        raise ValueError(
            f"The flow sets the thermostat. The {name} must not set dynamics or "
            "ase_md_kwargs."
        )
    if npt or maker.thermostat == "nose-hoover":
        dynamics = "nose-hoover-chain"
        md_kwargs = {
            "tdamp": _NOSE_HOOVER_TDAMP_STEPS * maker.md_time_step * units.fs,
            "tchain": 1,
        }
        if npt:
            md_kwargs["pdamp"] = _BAROSTAT_PDAMP_STEPS * maker.md_time_step * units.fs
    else:
        # AseMDMaker sets its default friction of 10 ps^-1
        dynamics = "langevin"
        md_kwargs = {}
    return replace(
        md_maker,
        ensemble=MDEnsemble.npt if npt else MDEnsemble.nvt,
        temperature=maker.temperature,
        pressure=maker.pressure if npt else md_maker.pressure,
        n_steps=n_steps,
        time_step=maker.md_time_step,
        dynamics=dynamics,
        ase_md_kwargs=md_kwargs,
        traj_file=ASE_TRAJECTORY_FILE,
        traj_file_fmt="ase",
        traj_interval=1,
        store_trajectory="no",
        mb_velocity_seed=(
            None
            if maker.random_seed is None
            else maker.random_seed + (maker.md_runs if npt else index)
        ),
        zero_linear_momentum=True,
    )


@dataclass
class ForceFieldFiniteTemperaturePhononMaker(BaseFiniteTemperaturePhononMaker):
    """
    Maker for effective harmonic phonons at a finite temperature with a force field.

    The relaxation, the MD and the phonon displacement calculations all use one
    force field, MACE-MP-0 by default. Use :obj:`from_force_field_name` to use
    another one. The MD writes an ASE trajectory file with a frame at every
    time step. The snapshots are read from this file, and the trajectory is not
    stored in the output document of the MD job. The Langevin thermostat has
    the default friction of :obj:`.AseMDMaker`, 10 ps^-1. The Nose-Hoover
    thermostat is ASE's NoseHooverChainNVT with one thermostat variable, like
    MDALGO = 2 in VASP. Its time constant gives a period of about 40 time
    steps, like SMASS = 0 in VASP. The initial velocities of the first MD job
    follow the Maxwell-Boltzmann distribution, with zero total momentum. The
    NPT MD, if any, uses ASE's MTKNPT, a Nose-Hoover thermostat and barostat
    that change the whole cell, whatever the thermostat setting (Martyna et
    al., J. Chem. Phys. 101, 4177 (1994)). Its thermostat time constant is
    that of the Nose-Hoover thermostat above, and its barostat time constant
    is 1000 time steps. MD job i, counted from 0, seeds its initial velocities
    and its Langevin random forces with random_seed + i. The NPT MD uses
    random_seed + md_runs.

    See :obj:`.BaseFiniteTemperaturePhononMaker` for the workflow.

    This workflow is new and has not been tested widely. It might still change
    in future versions.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    bulk_relax_maker: .ForceFieldRelaxMaker | None
        Maker for the relaxation of the unit cell. It keeps the symmetry of
        the structure. None skips the relaxation.
    born_maker: .Maker | None
        Maker for the Born effective charges and the dielectric tensor, for
        example a ForceFieldDielectricMaker with MACE-Field or a VASP
        DielectricMaker. It is None by default, as in the force field pheasy
        phonon workflow.
    npt_maker: .ForceFieldMDMaker | None
        Maker for the NPT MD. The flow sets the same settings as for md_maker,
        and the pressure. It must not set dynamics or ase_md_kwargs. None skips
        the NPT MD.
    fixed_cell_relax_maker: .ForceFieldRelaxMaker | None
        Maker for the relaxation of the atoms in the cell from the NPT MD. It
        must keep the cell.
    md_maker: .ForceFieldMDMaker
        Maker for the MD. The flow sets its ensemble, temperature, n_steps,
        time_step, dynamics, ase_md_kwargs, traj_file, traj_file_fmt,
        traj_interval, store_trajectory, mb_velocity_seed and
        zero_linear_momentum. It must not set dynamics or ase_md_kwargs.
    phonon_displacement_maker: .ForceFieldStaticMaker
        Maker for the static calculations on the snapshots and the undisplaced
        supercell.
    code: str
        Code of the phonon displacement calculations.
    md_code: str
        Code of the MD.
    """

    name: str = "force field finite temperature phonon"
    bulk_relax_maker: ForceFieldRelaxMaker | None = field(
        default_factory=lambda: ForceFieldRelaxMaker(
            force_field_name=_DEFAULT_FORCE_FIELD,
            relax_kwargs={"fmax": 1e-5},
            fix_symmetry=True,
        )
    )
    md_maker: ForceFieldMDMaker = field(
        default_factory=lambda: _get_md_maker(_DEFAULT_FORCE_FIELD)
    )
    phonon_displacement_maker: ForceFieldStaticMaker = field(
        default_factory=lambda: ForceFieldStaticMaker(
            force_field_name=_DEFAULT_FORCE_FIELD
        )
    )
    code: str = "forcefields"
    md_code: str = "forcefields"

    @property
    def prev_calc_dir_argname(self) -> None:
        """Name of the prev_dir argument of the phonon displacement and Born makers.

        Force field makers take no previous directory, so it is None. A VASP
        born_maker then runs without a previous directory.
        """
        return

    def get_md_maker(self, n_steps: int, index: int = 0) -> ForceFieldMDMaker:
        """
        Get the force field MD maker of one MD job.

        Parameters
        ----------
        n_steps: int
            Number of MD steps of the job.
        index: int
            Position of the job in the MD, counted from 0.

        Returns
        -------
        ForceFieldMDMaker
        """
        return _get_force_field_md_maker(self, n_steps, index)

    def get_npt_maker(self, n_steps: int) -> ForceFieldMDMaker:
        """
        Get the force field maker of the NPT MD job.

        Parameters
        ----------
        n_steps: int
            Number of MD steps.

        Returns
        -------
        ForceFieldMDMaker
        """
        return _get_force_field_md_maker(self, n_steps, npt=True)

    @classmethod
    def from_force_field_name(
        cls,
        force_field_name: str | MLFF | dict,
        calculator_kwargs: dict | None = None,
        relax_initial_structure: bool = True,
        run_npt: bool = False,
        **kwargs,
    ) -> Self:
        """
        Create a finite-temperature phonon flow from a force field name.

        The relaxation, the MD and the phonon displacement calculations use the
        force field, and born_maker is None. These makers replace any given in
        kwargs.

        Parameters
        ----------
        force_field_name : str or .MLFF or dict
            The name of the force field.
        calculator_kwargs : dict | None
            The keyword arguments to pass to the calculator.
        relax_initial_structure: bool = True
            Whether to relax the initial structure.
        run_npt: bool = False
            Whether to run an NPT MD with the force field first. The NVT MD then
            uses its average cell, with the atoms relaxed in that cell.
        **kwargs
            Additional kwargs to pass to ForceFieldFiniteTemperaturePhononMaker.

        Returns
        -------
        ForceFieldFiniteTemperaturePhononMaker
        """
        calculator_kwargs = calculator_kwargs or {}
        phonon_displacement_maker = ForceFieldStaticMaker(
            force_field_name=force_field_name,
            calculator_kwargs=calculator_kwargs,
        )
        npt_maker: ForceFieldMDMaker | None = None
        fixed_cell_relax_maker: ForceFieldRelaxMaker | None = None
        if run_npt:
            npt_maker = _get_md_maker(force_field_name, calculator_kwargs)
            fixed_cell_relax_maker = ForceFieldRelaxMaker(
                force_field_name=force_field_name,
                calculator_kwargs=calculator_kwargs,
                relax_cell=False,
                relax_kwargs={"fmax": 1e-5},
                fix_symmetry=True,
            )
        kwargs.update(
            npt_maker=npt_maker,
            fixed_cell_relax_maker=fixed_cell_relax_maker,
            bulk_relax_maker=(
                ForceFieldRelaxMaker(
                    force_field_name=force_field_name,
                    calculator_kwargs=calculator_kwargs,
                    relax_kwargs={"fmax": 1e-5},
                    fix_symmetry=True,
                )
                if relax_initial_structure
                else None
            ),
            md_maker=_get_md_maker(force_field_name, calculator_kwargs),
            phonon_displacement_maker=phonon_displacement_maker,
            born_maker=None,
        )
        return cls(
            name=(
                f"{phonon_displacement_maker.mlff.name} Finite Temperature Phonon Maker"
            ),
            **kwargs,
        )


@dataclass
class VaspMDMLFFStaticFiniteTemperaturePhononMaker(FiniteTemperaturePhononMaker):
    """
    Maker for finite-temperature phonons from VASP MD and force field statics.

    The relaxation and the MD are those of
    :obj:`.FiniteTemperaturePhononMaker`. The forces on the snapshots and on
    the undisplaced supercell come from a force field, MACE-MP-0 by default.
    Use :obj:`from_force_field_name` to use another one.

    This workflow is new and has not been tested widely. It might still change
    in future versions.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    bulk_relax_maker: .Maker | None
        Maker for the relaxation of the unit cell. None skips the relaxation.
    born_maker: .Maker | None
        Maker for the Born effective charges and the dielectric tensor. It is
        None by default, as in the force field pheasy phonon workflow, since
        the force constants come from the force field.
    md_maker: .MDMaker
        Maker for the MD, see :obj:`.FiniteTemperaturePhononMaker`.
    phonon_displacement_maker: .ForceFieldStaticMaker
        Maker for the static calculations on the snapshots and the undisplaced
        supercell.
    code: str
        Code of the phonon displacement calculations.
    """

    name: str = "vasp md mlff static finite temperature phonon"
    born_maker: Maker | None = None
    phonon_displacement_maker: Maker = field(
        default_factory=lambda: ForceFieldStaticMaker(
            force_field_name=_DEFAULT_FORCE_FIELD
        )
    )
    code: str = "forcefields"

    @property
    def prev_calc_dir_argname(self) -> None:
        """Name of the prev_dir argument of the phonon displacement and Born makers.

        Force field makers take no previous directory, so it is None. A VASP
        born_maker then runs without a previous directory.
        """
        return

    @classmethod
    def from_force_field_name(
        cls,
        force_field_name: str | MLFF | dict,
        calculator_kwargs: dict | None = None,
        **kwargs,
    ) -> Self:
        """
        Create a finite-temperature phonon flow whose statics use a force field.

        The phonon displacement calculations use the force field, and born_maker
        is None. These makers replace any given in kwargs.

        Parameters
        ----------
        force_field_name : str or .MLFF or dict
            The name of the force field.
        calculator_kwargs : dict | None
            The keyword arguments to pass to the calculator.
        **kwargs
            Additional kwargs to pass to
            VaspMDMLFFStaticFiniteTemperaturePhononMaker.

        Returns
        -------
        VaspMDMLFFStaticFiniteTemperaturePhononMaker
        """
        phonon_displacement_maker = ForceFieldStaticMaker(
            force_field_name=force_field_name,
            calculator_kwargs=calculator_kwargs or {},
        )
        kwargs.update(
            phonon_displacement_maker=phonon_displacement_maker, born_maker=None
        )
        return cls(
            name=(
                f"VASP MD {phonon_displacement_maker.mlff.name} Static Finite "
                "Temperature Phonon Maker"
            ),
            **kwargs,
        )


@dataclass
class MLFFMDVaspStaticFiniteTemperaturePhononMaker(FiniteTemperaturePhononMaker):
    """
    Maker for finite-temperature phonons from force field MD and VASP statics.

    The relaxation, the Born charges and the phonon displacement calculations
    are those of :obj:`.FiniteTemperaturePhononMaker`. The MD uses a force
    field, MACE-MP-0 by default, and starts from the supercell of the
    VASP-relaxed structure. Use :obj:`from_force_field_name` to use another
    force field. The MD settings are those of
    :obj:`ForceFieldFiniteTemperaturePhononMaker`.

    This workflow is new and has not been tested widely. It might still change
    in future versions.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    bulk_relax_maker: .Maker | None
        Maker for the relaxation of the unit cell. None skips the relaxation.
    born_maker: .BaseVaspMaker | None
        Maker for the Born effective charges and the dielectric tensor, used
        for the non-analytical correction. It is a DielectricMaker by default,
        as in the VASP pheasy phonon workflow. None skips it.
    md_maker: .ForceFieldMDMaker
        Maker for the MD. The flow sets its ensemble, temperature, n_steps,
        time_step, dynamics, ase_md_kwargs, traj_file, traj_file_fmt,
        traj_interval, store_trajectory, mb_velocity_seed and
        zero_linear_momentum. It must not set dynamics or ase_md_kwargs.
    npt_maker: .ForceFieldMDMaker | None
        Maker for the NPT MD, under the same conditions as md_maker. None skips
        the NPT MD.
    phonon_displacement_maker: .BaseVaspMaker
        Maker for the static calculations on the snapshots and the undisplaced
        supercell.
    md_code: str
        Code of the MD.
    """

    name: str = "mlff md vasp static finite temperature phonon"
    md_maker: Maker = field(default_factory=lambda: _get_md_maker(_DEFAULT_FORCE_FIELD))
    md_code: str = "forcefields"

    def get_md_maker(self, n_steps: int, index: int = 0) -> ForceFieldMDMaker:
        """
        Get the force field MD maker of one MD job.

        Parameters
        ----------
        n_steps: int
            Number of MD steps of the job.
        index: int
            Position of the job in the MD, counted from 0.

        Returns
        -------
        ForceFieldMDMaker
        """
        return _get_force_field_md_maker(self, n_steps, index)

    def get_npt_maker(self, n_steps: int) -> ForceFieldMDMaker:
        """
        Get the force field maker of the NPT MD job.

        Parameters
        ----------
        n_steps: int
            Number of MD steps.

        Returns
        -------
        ForceFieldMDMaker
        """
        return _get_force_field_md_maker(self, n_steps, npt=True)

    @classmethod
    def from_force_field_name(
        cls,
        force_field_name: str | MLFF | dict,
        calculator_kwargs: dict | None = None,
        **kwargs,
    ) -> Self:
        """
        Create a finite-temperature phonon flow whose MD uses a force field.

        The MD maker built for the force field replaces any given in kwargs.

        Parameters
        ----------
        force_field_name : str or .MLFF or dict
            The name of the force field.
        calculator_kwargs : dict | None
            The keyword arguments to pass to the calculator.
        **kwargs
            Additional kwargs to pass to
            MLFFMDVaspStaticFiniteTemperaturePhononMaker.

        Returns
        -------
        MLFFMDVaspStaticFiniteTemperaturePhononMaker
        """
        md_maker = _get_md_maker(force_field_name, calculator_kwargs)
        kwargs.update(md_maker=md_maker)
        return cls(
            name=(
                f"{md_maker.mlff.name} MD VASP Static Finite Temperature Phonon Maker"
            ),
            **kwargs,
        )
