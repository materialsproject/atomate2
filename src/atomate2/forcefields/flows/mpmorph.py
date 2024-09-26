"""Define MPMorph flows for interatomic forcefields.

For information about the current flows, contact:
- Bryant Li (@BryantLi-BLI)
- Aaron Kaplan (@esoteric-ephemera)
- Max Gallant (@mcgalcode)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.common.flows.mpmorph import (
    EquilibriumVolumeMaker,
    FastQuenchMaker,
    MPMorphMDMaker,
    SlowQuenchMaker,
)
from atomate2.forcefields import MLFF
from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker
from atomate2.forcefields.md import ForceFieldMDMaker

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Flow, Job
    from pymatgen.core import Structure
    from typing_extensions import Self


@dataclass
class MPMorphMLFFMDMaker(MPMorphMDMaker):
    """
    Define a ML ForceField MPMorph flow.

    This flow uses NVT molecular dynamics to:
    (1 - optional) Determine the equilibrium volume of an amorphous
        structure via EOS fit.
    (2 - optional) Quench the equilibrium volume structure from a higher
        temperature down to a lower desired "production" temperature.
    (3) Run a production, longer-time MD run in NVT.
        The production run can be broken up into smaller steps to
        ensure the simulation does not hit wall time limits.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker.

    Unlike the VASP base MPMorph flows, this class will not run
    calculations by default. The user needs to specify a forcefield.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher;
        uses EquilibriumVolumeMaker with a ForceFieldMDMaker (MLFF)
    production_md_maker : ForceFieldMDMaker
        MDMaker to generate the production run(s);
        inherits from ForceFieldMDMaker (MLFF)
    quench_maker :  SlowQuenchMaker or FastQuenchMaker or None
        SlowQuenchMaker - MLFFMDMaker that quenches structure from
        high to low temperature
        FastQuenchMaker - DoubleRelaxMaker + Static that "quenches"
        structure to 0K
    """

    name: str = "MP Morph MLFF MD Maker"
    convergence_md_maker: EquilibriumVolumeMaker | None = None
    production_md_maker: ForceFieldMDMaker = field(default_factory=ForceFieldMDMaker)
    quench_maker: FastQuenchMaker | SlowQuenchMaker | None = None

    @classmethod
    def from_temperature_and_steps(
        cls,
        temperature: float,
        n_steps_convergence: int = 5000,
        n_steps_production: int = 1000,
        end_temp: float | None = None,
        md_maker: ForceFieldMDMaker = None,
        quench_maker: FastQuenchMaker | SlowQuenchMaker | None = None,
    ) -> Self:
        """
        Create a MPMorphMLFFMDMaker from temperature and steps.

        Recommended for friendly user experience.

        Parameters
        ----------
        temperature : float
            Temperature of the equilibrium volume search and production run in Kelvin
        n_steps_convergence : int
            Number of steps for the convergence run(s)
        n_steps_production : int
            Total number of steps for the production run(s)
        end_temp : float or None
            If a float, the temperature to ramp down to in the production run.
            If None (default), set to `temperature`.
        md_maker : ForceFieldMDMaker
            MDMaker to generate the molecular dynamics jobs specifically for MLFF MDs.
            This is a generalization to any MLFF MD Maker, e.g., CHGNetMDMaker
        quench_maker :  SlowQuenchMaker or FastQuenchMaker or None
            SlowQuenchMaker - MLFFMDMaker that quenches structure from
            high to low temperature
            FastQuenchMaker - DoubleRelaxMaker + Static that "quenches"
            structure to 0K
        """
        conv_md_maker = md_maker.update_kwargs(
            update={
                "temperature": temperature,
                "n_steps": n_steps_convergence,
                "name": "Convergence MPMorph MLFF MD Maker",
            },
            class_filter=ForceFieldMDMaker,
        )

        convergence_md_maker = EquilibriumVolumeMaker(
            name="MP Morph MLFF Equilibrium Volume Maker",
            md_maker=conv_md_maker,
        )

        production_md_maker = md_maker.update_kwargs(
            update={
                "name": "Production Run MLFF MD Maker",
                "temperature": temperature
                if end_temp is None
                else [temperature, end_temp],
                "n_steps": n_steps_production,
            }
        )

        return cls(
            name="MP Morph MLFF MD Maker",
            convergence_md_maker=convergence_md_maker,
            production_md_maker=production_md_maker,
            quench_maker=quench_maker,
        )


@dataclass
class SlowQuenchMLFFMDMaker(SlowQuenchMaker):
    """Slow quench from high to low temperature using ForceFieldMDMaker.

    Quenches a provided structure with a molecular dynamics run
    from a desired high temperature to a desired low temperature.
    Flow creates a series of MD runs that holds at a certain temperature
    and initiates the following MD run at a lower temperature (step-wise
    temperature MD runs).

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    md_maker :  ForceFieldMDMaker
        MDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    quench_start_temperature : float = 3000
        Starting temperature for quench; default 3000K
    quench_end_temperature : float = 500
        Ending temperature for quench; default 500K
    quench_temperature_step : float = 500
        Temperature step for quench; default 500K drop
    quench_n_steps : int = 1000
        Number of steps for quench; default 1000 steps
    """

    name: str = "ForceField slow quench"
    md_maker: ForceFieldMDMaker = field(default_factory=ForceFieldMDMaker)

    def call_md_maker(
        self,
        structure: Structure,
        temp: float | tuple[float, float],
        prev_dir: str | Path | None = None,
    ) -> Flow | Job:
        """Call the MD maker to create the MD jobs for MLFF Only."""
        self.md_maker = self.md_maker.update_kwargs(
            update={
                "name": f"Slow quench MLFF MD Maker {temp}K",
                "temperature": temp,
                "n_steps": self.quench_n_steps,
            }
        )
        return self.md_maker.make(structure=structure, prev_dir=prev_dir)


@dataclass
class FastQuenchMLFFMDMaker(FastQuenchMaker):
    """Fast quench from high temperature to 0K structures with forcefields.

    Quenches a provided structure with a single (or double)
    relaxation and a static calculation at 0K.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker :  ForceFieldRelaxMaker
        Relax Maker
    relax_maker2 :  ForceFieldRelaxMaker
        Relax Maker for a second relaxation; useful for tighter convergence
    static_maker : ForceFieldStaticMaker
        Static Maker

    """

    name: str = "ForceField fast quench"
    relax_maker: ForceFieldRelaxMaker = field(default_factory=ForceFieldRelaxMaker)
    relax_maker2: ForceFieldRelaxMaker = field(default_factory=ForceFieldRelaxMaker)
    static_maker: ForceFieldStaticMaker = field(default_factory=ForceFieldStaticMaker)

    @classmethod
    def from_force_field_name(cls, force_field_name: str | MLFF) -> Self:
        """
        Create a fast quench maker from the force field name.

        Parameters
        ----------
        force_field_name : str or .MLFF
            The name of the forcefield or its enum value

        Returns
        -------
        FastQuenchMaker
            A fast quench maker that consists of a double relax + static using
            the specified MLFF.
        """
        if isinstance(force_field_name, str) and force_field_name in MLFF.__members__:
            # ensure `force_field_name` uses enum format
            force_field_name = MLFF(force_field_name)
        force_field_name = str(force_field_name)

        return cls(
            name=f"{force_field_name} fast quench maker",
            relax_maker=ForceFieldRelaxMaker(force_field_name=force_field_name),
            relax_maker2=ForceFieldRelaxMaker(force_field_name=force_field_name),
            static_maker=ForceFieldStaticMaker(force_field_name=force_field_name),
        )
