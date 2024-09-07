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
from atomate2.common.jobs.eos import MPMorphEVPostProcess
from atomate2.forcefields.jobs import (
    CHGNetRelaxMaker,
    CHGNetStaticMaker,
    ForceFieldRelaxMaker,
    ForceFieldStaticMaker,
    LJRelaxMaker,
    LJStaticMaker,
    M3GNetRelaxMaker,
    M3GNetStaticMaker,
    MACERelaxMaker,
    MACEStaticMaker,
)
from atomate2.forcefields.md import (
    CHGNetMDMaker,
    ForceFieldMDMaker,
    LJMDMaker,
    M3GNetMDMaker,
    MACEMDMaker,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any
    from typing_extensions import Self

    from jobflow import Flow, Job, Maker
    from pymatgen.core import Structure


@dataclass
class MLFFMPMorphMDMaker(MPMorphMDMaker):
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
        MDMaker to generate the equilibrium volumer searcher
    production_md_maker : Maker
        MDMaker to generate the production run(s)
    quench_maker :  SlowQuenchMaker or FastQuenchMaker or None
        SlowQuenchMaker - MDMaker that quenchs structure from high to low temperature
        FastQuenchMaker - DoubleRelaxMaker + Static that "quenchs" structure at 0K
    """

    name: str = "Forcefield MPMorph MD"
    convergence_md_maker: Maker | None = field(
        default_factory = lambda : EquilibriumVolumeMaker(
            name="MP Morph MLFF Equilibrium Volume Maker",
            md_maker = ForceFieldMDMaker,
            postprocessor=MPMorphEVPostProcess(),
        )
    )
    production_md_maker: Maker | None = field(default_factory=ForceFieldMDMaker)
    quench_maker: FastQuenchMaker | SlowQuenchMaker | None = None

    @classmethod
    def from_temperature_and_nsteps(
        cls,
        temperature: float,
        n_steps_convergence : int = 5000,
        n_steps_production : int = 10000,
        end_temp : float | None = None,
        md_maker : Maker = ForceFieldMDMaker,
    ) -> Self:
        """
        Create an MPMorph flow from a temperature and number of steps.

        This is a convenience class constructor. The user need only
        input the desired temperature and steps for convergence / production
        MD runs.
        
        Parameters
        -----------
        temperature : float
            The (starting) temperature
        n_steps_convergence : int = 5000
            The number of steps used in MD runs for equilibrating structures.
        n_steps_production : int =
            The number of steps used in MD production runs. Default, 10000.
        end_temp : float or None
            If a float, the temperature to ramp down to in the production run.
            If None (default), set to `temperature`.
        base_md_maker : Maker
            The Maker used to start MD runs.
        """

        conv_md_maker = md_maker.update_kwargs(
            update={
                "temperature": temperature,
                "n_steps": n_steps_convergence,
            },
            class_filter=ForceFieldMDMaker,
        )
        convergence_md_maker = EquilibriumVolumeMaker(
            name="MP Morph MLFF Equilibrium Volume Maker",
            md_maker=conv_md_maker,
            postprocessor=MPMorphEVPostProcess(),
        )

        production_md_maker = md_maker.update_kwargs(
            update=dict(
                name="Production Run MLFF MD Maker",
                temperature=temperature if end_temp is None else [temperature,end_temp],
                n_steps=n_steps_production,
            )
        )

        return cls(
            convergence_md_maker = convergence_md_maker,
            production_md_maker = production_md_maker,
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


@dataclass
class MPMorphLJMDMaker(MLFFMPMorphMDMaker):
    """Lennard-Jones MPMorph flow for volume equilibration and production.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and finally a production run at
    a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : float = 300
        Temperature of the equilibrium volume search and production run
        in Kelvin, default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    md_maker : LJMDMaker
        LJMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    production_md_maker : LJMDMaker
        LJMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    """

    name: str = "MP Morph LJ MD Maker"
    temperature: float = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: LJMDMaker = field(default_factory=lambda: LJMDMaker(name="LJ MD Maker"))
    production_md_maker: LJMDMaker = field(
        default_factory=lambda: LJMDMaker(name="Production Run LJ MD Maker")
    )


@dataclass
class MPMorphSlowQuenchLJMDMaker(MLFFMPMorphMDMaker):
    """Lennard Jones ForceField MPMorph flow plus slow quench.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and a production run at a given temperature.
    Then proceed with a slow quench from high temperature to low temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : float = 300
        Temperature of the equilibrium volume search and production run in Kelvin,
        default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : LJMDMaker
        LJMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    production_md_maker : LJMDMaker
        LJMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    quench_maker : SlowQuenchMLFFMDMaker
        Using the LJMDMaker to perform SlowQuenchMLFFMDMaker with the default settings
        Check SlowQuenchMLFFMDMaker for more information.
    """

    name: str = "MP Morph LJ MD Maker Slow Quench"
    temperature: float = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: LJMDMaker = field(default_factory=lambda: LJMDMaker(name="LJ MD Maker"))
    production_md_maker: LJMDMaker = field(
        default_factory=lambda: LJMDMaker(name="Production Run LJ MD Maker")
    )
    quench_maker: SlowQuenchMLFFMDMaker = field(
        default_factory=lambda: SlowQuenchMLFFMDMaker(
            md_maker=LJMDMaker(name="LJ MD Maker"),
            quench_n_steps=1000,
            quench_temperature_step=500,
            quench_end_temperature=500,
            quench_start_temperature=3000,
        )
    )


@dataclass
class MPMorphFastQuenchLJMDMaker(MLFFMPMorphMDMaker):
    """Lennard Jones ForceField MPMorph flow plus fast quench.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and finally a production run at
    a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : float = 300
        Temperature of the equilibrium volume search and production run in Kelvin,
        default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : LJMDMaker
        LJMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    production_md_maker : LJMDMaker
        LJMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    quench_maker : FastQuenchMLFFMDMaker
        Using the LJMDMaker to perform FastQuenchMLFFMDMaker with the default settings
        Check FastQuenchMLFFMDMaker for more information.
    """

    name: str = "MP Morph LJ MD Maker Fast Quench"
    temperature: float = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: LJMDMaker = field(default_factory=lambda: LJMDMaker(name="LJ MD Maker"))
    production_md_maker: LJMDMaker = field(
        default_factory=lambda: LJMDMaker(name="Production Run LJ MD Maker")
    )
    quench_maker: FastQuenchMLFFMDMaker = field(
        default_factory=lambda: FastQuenchMLFFMDMaker(
            relax_maker=LJRelaxMaker(),
            relax_maker2=LJRelaxMaker(),
            static_maker=LJStaticMaker(),
        )
    )


@dataclass
class MPMorphCHGNetMDMaker(MLFFMPMorphMDMaker):
    """CHGNet ML ForceField MPMorph flow for volume equilibration and production.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and finally a production run at a
    given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : float = 300
        Temperature of the equilibrium volume search and production run in Kelvin,
        default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    md_maker : CHGNetMDMaker
        CHGNetMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    production_md_maker : CHGNetMDMaker
        CHGNetMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    """

    name: str = "MP Morph CHGNet MD Maker"
    temperature: float = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: CHGNetMDMaker = field(
        default_factory=lambda: CHGNetMDMaker(name="CHGNet MD Maker")
    )
    production_md_maker: CHGNetMDMaker = field(
        default_factory=lambda: CHGNetMDMaker(name="Production Run CHGNet MD Maker")
    )


@dataclass
class MPMorphSlowQuenchCHGNetMDMaker(MLFFMPMorphMDMaker):
    """CHGNet ML ForceField MPMorph flow plus slow quench..

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and a production run at a given temperature.
    Then proceed with a slow quench from high temperature to low temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : float = 300
        Temperature of the equilibrium volume search and production run in Kelvin,
        default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : CHGNetMDMaker
        CHGNetMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    production_md_maker : CHGNetMDMaker
        CHGNetMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    quench_maker : SlowQuenchMLFFMDMaker
        Using the CHGNetMDMaker to perform SlowQuenchMLFFMDMaker with the default settings
        Check SlowQuenchMLFFMDMaker for more information.
    """

    name: str = "MP Morph CHGNet MD Maker Slow Quench"
    temperature: float = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: CHGNetMDMaker = field(
        default_factory=lambda: CHGNetMDMaker(name="CHGNet MD Maker")
    )
    production_md_maker: CHGNetMDMaker = field(
        default_factory=lambda: CHGNetMDMaker(name="Production Run CHGNet MD Maker")
    )
    quench_maker: SlowQuenchMLFFMDMaker = field(
        default_factory=lambda: SlowQuenchMLFFMDMaker(
            md_maker=CHGNetMDMaker(name="CHGNet MD Maker"),
            quench_n_steps=1000,
            quench_temperature_step=500,
            quench_end_temperature=500,
            quench_start_temperature=3000,
        )
    )


@dataclass
class MPMorphFastQuenchCHGNetMDMaker(MLFFMPMorphMDMaker):
    """CHGNet ML ForceField MPMorph flow plus fast quench.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and finally a production run at
    a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : float = 300
        Temperature of the equilibrium volume search and production run in Kelvin,
        default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : CHGNetMDMaker
        CHGNetMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    production_md_maker : CHGNetMDMaker
        CHGNetMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    quench_maker : FastQuenchMLFFMDMaker
        Using the CHGNetMDMaker to perform FastQuenchMLFFMDMaker with the default settings
        Check FastQuenchMLFFMDMaker for more information.
    """

    name: str = "MP Morph CHGNet MD Maker Fast Quench"
    temperature: float = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: CHGNetMDMaker = field(
        default_factory=lambda: CHGNetMDMaker(name="CHGNet MD Maker")
    )
    production_md_maker: CHGNetMDMaker = field(
        default_factory=lambda: CHGNetMDMaker(name="Production Run CHGNet MD Maker")
    )
    quench_maker: FastQuenchMLFFMDMaker = field(
        default_factory=lambda: FastQuenchMLFFMDMaker(
            relax_maker=CHGNetRelaxMaker(),
            relax_maker2=CHGNetRelaxMaker(),
            static_maker=CHGNetStaticMaker(),
        )
    )


@dataclass
class MPMorphM3GNetMDMaker(MLFFMPMorphMDMaker):
    """M3GNet ML ForceField MPMorph flow for volume equilibration and production.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and finally a production run at a
    given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : float = 300
        Temperature of the equilibrium volume search and production run in Kelvin,
        default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    md_maker : M3GNetMDMaker
        M3GNetMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    production_md_maker : M3GNetMDMaker
        M3GNetMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    """

    name: str = "MP Morph M3GNet MD Maker"
    temperature: float = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: M3GNetMDMaker = field(
        default_factory=lambda: M3GNetMDMaker(name="M3GNet MD Maker")
    )
    production_md_maker: M3GNetMDMaker = field(
        default_factory=lambda: M3GNetMDMaker(name="Production Run M3GNet MD Maker")
    )


@dataclass
class MPMorphSlowQuenchM3GNetMDMaker(MLFFMPMorphMDMaker):
    """M3GNet ML ForceField MPMorph flow plus slow quench.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and a production run at a given temperature.
    Then proceed with a slow quench from high temperature to low temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : float = 300
        Temperature of the equilibrium volume search and production run in Kelvin,
        default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : M3GNetMDMaker
        M3GNetMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    production_md_maker : M3GNetMDMaker
        M3GNetMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    quench_maker : SlowQuenchMLFFMDMaker
        Using the M3GNetMDMaker to perform SlowQuenchMLFFMDMaker with the default settings
        Check SlowQuenchMLFFMDMaker for more information.
    """

    name: str = "MP Morph M3GNet MD Maker Slow Quench"
    temperature: float = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: M3GNetMDMaker = field(
        default_factory=lambda: M3GNetMDMaker(name="M3GNet MD Maker")
    )
    production_md_maker: M3GNetMDMaker = field(
        default_factory=lambda: M3GNetMDMaker(name="Production Run M3GNet MD Maker")
    )
    quench_maker: SlowQuenchMLFFMDMaker = field(
        default_factory=lambda: SlowQuenchMLFFMDMaker(
            md_maker=M3GNetMDMaker(name="M3GNet MD Maker"),
            quench_n_steps=1000,
            quench_temperature_step=500,
            quench_end_temperature=500,
            quench_start_temperature=3000,
        )
    )


@dataclass
class MPMorphFastQuenchM3GNetMDMaker(MLFFMPMorphMDMaker):
    """M3GNet ML ForceField MPMorph flow plus fast quench.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and finally a production run at a
    given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : float = 300
        Temperature of the equilibrium volume search and production run in Kelvin,
        default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : M3GNetMDMaker
        M3GNetMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    production_md_maker : M3GNetMDMaker
        M3GNetMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    quench_maker : FastQuenchMLFFMDMaker
        Using the M3GNetMDMaker to perform FastQuenchMLFFMDMaker with the default settings
        Check FastQuenchMLFFMDMaker for more information.
    """

    name: str = "MP Morph M3GNet MD Maker Fast Quench"
    temperature: float = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: M3GNetMDMaker = field(
        default_factory=lambda: M3GNetMDMaker(name="M3GNet MD Maker")
    )
    production_md_maker: M3GNetMDMaker = field(
        default_factory=lambda: M3GNetMDMaker(name="Production Run CHGNet MD Maker")
    )
    quench_maker: FastQuenchMLFFMDMaker = field(
        default_factory=lambda: FastQuenchMLFFMDMaker(
            relax_maker=M3GNetRelaxMaker(),
            relax_maker2=M3GNetRelaxMaker(),
            static_maker=M3GNetStaticMaker(),
        )
    )


@dataclass
class MPMorphMACEMDMaker(MLFFMPMorphMDMaker):
    """MACE ML ForceField MPMorph flow for volume equilibration and production runs.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and finally a production run at a
    given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : float = 300
        Temperature of the equilibrium volume search and production run in Kelvin,
        default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    md_maker : MACEMDMaker
        MACEMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    production_md_maker : MACEMDMaker
        MACEMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    """

    name: str = "MP Morph MACE MD Maker"
    temperature: float = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: MACEMDMaker = field(
        default_factory=lambda: MACEMDMaker(name="MACE MD Maker")
    )
    production_md_maker: MACEMDMaker = field(
        default_factory=lambda: MACEMDMaker(name="Production Run MACE MD Maker")
    )


@dataclass
class MPMorphSlowQuenchMACEMDMaker(MLFFMPMorphMDMaker):
    """MACE ML ForceField MPMorph flow plus slow quench.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and a production run at a given temperature.
    Then proceed with a slow quench from high temperature to low temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : float = 300
        Temperature of the equilibrium volume search and production run in Kelvin,
        default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : MACEMDMaker
        MACEMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    production_md_maker : MACEMDMaker
        MACEMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    quench_maker : SlowQuenchMLFFMDMaker
        Using the MACEMDMaker to perform SlowQuenchMLFFMDMaker with the default settings
        Check SlowQuenchMLFFMDMaker for more information.
    """

    name: str = "MP Morph MACE MD Maker Slow Quench"
    temperature: float = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: MACEMDMaker = field(
        default_factory=lambda: MACEMDMaker(name="MACE MD Maker")
    )
    production_md_maker: MACEMDMaker = field(
        default_factory=lambda: MACEMDMaker(name="Production Run MACE MD Maker")
    )
    quench_maker: SlowQuenchMLFFMDMaker = field(
        default_factory=lambda: SlowQuenchMLFFMDMaker(
            md_maker=MACEMDMaker(name="MACE MD Maker"),
            quench_n_steps=1000,
            quench_temperature_step=500,
            quench_end_temperature=500,
            quench_start_temperature=3000,
        )
    )


@dataclass
class MPMorphFastQuenchMACEMDMaker(MLFFMPMorphMDMaker):
    """MACE ML ForceField MPMorph flow plus fast quench.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and finally a production run
    at a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : float = 300
        Temperature of the equilibrium volume search and production run in Kelvin,
        default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : MACEMDMaker
        MACEMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    production_md_maker : MACEMDMaker
        MACEMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    quench_maker : FastQuenchMLFFMDMaker
        Using the MACEMDMaker to perform FastQuenchMLFFMDMaker with the default settings
        Check FastQuenchMLFFMDMaker for more information.
    """

    name: str = "MP Morph MACE MD Maker Fast Quench"
    temperature: float = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: MACEMDMaker = field(
        default_factory=lambda: MACEMDMaker(name="MACE MD Maker")
    )
    production_md_maker: MACEMDMaker = field(
        default_factory=lambda: MACEMDMaker(name="Production Run MACE MD Maker")
    )
    quench_maker: FastQuenchMLFFMDMaker = field(
        default_factory=lambda: FastQuenchMLFFMDMaker(
            relax_maker=MACERelaxMaker(),
            relax_maker2=MACERelaxMaker(),
            static_maker=MACEStaticMaker(),
        )
    )
