"""Flows adapted from MPMorph *link to origin github repo*"""  # TODO: Add link to origin github repo

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.common.flows.mpmorph import (
    FastQuenchMaker,
    SlowQuenchMaker,
    MPMorphMDMaker,
    EquilibriumVolumeMaker,
)

from atomate2.forcefields.md import ForceFieldMDMaker
from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker

from atomate2.common.jobs.eos import MPMorphEVPostProcess, MPMorphPVPostProcess

from atomate2.forcefields.jobs import (
    ForceFieldRelaxMaker,
    ForceFieldStaticMaker,
    CHGNetRelaxMaker,
    CHGNetStaticMaker,
    M3GNetRelaxMaker,
    M3GNetStaticMaker,
    MACERelaxMaker,
    MACEStaticMaker,
    LJRelaxMaker,
    LJStaticMaker,
)
from atomate2.forcefields.md import (
    ForceFieldMDMaker,
    CHGNetMDMaker,
    M3GNetMDMaker,
    MACEMDMaker,
    LJMDMaker,
)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pathlib import Path
    from jobflow import Flow, Job


@dataclass
class MPMorphMLFFMDMaker(MPMorphMDMaker):
    """ML ForceField MPMorph flow for volume equilibration, quench, and production runs via molecular dynamics

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    (optional) for the volume followed by a production run(s) at a given temperature
    and finally quench (optional) from high temperature to low temperature

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : int = 300
        Temperature of the equilibrium volume search and production run in Kelvin, default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : ForceFieldMDMaker
        MDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher; inherits from EquilibriumVolumeMaker and ForceFieldMDMaker (MLFF)
    production_md_maker : ForceFieldMDMaker
        MDMaker to generate the production run(s); inherits from ForceFieldMDMaker (MLFF)
    quench_maker :  SlowQuenchMaker or FastQuenchMaker or None
        SlowQuenchMaker - MLFFMDMaker that quenchs structure from high temperature to low temperature
        FastQuenchMaker - DoubleRelaxMaker + Static that "quenchs" structure at 0K
    """

    name: str = "MP Morph MLFF MD Maker"
    temperature: int = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: ForceFieldMDMaker | None = field(default_factory=ForceFieldMDMaker)
    convergence_md_maker: EquilibriumVolumeMaker | None = None
    production_md_maker: ForceFieldMDMaker = field(default_factory=ForceFieldMDMaker)

    quench_maker: FastQuenchMLFFMDMaker | SlowQuenchMLFFMDMaker | None = None

    def _post_init_update(self) -> None:
        """Ensure that forcefield makers correctly set temperature."""

        self.md_maker = self.md_maker.update_kwargs(
            update={
                "temperature": self.temperature,
                "n_steps": self.steps_convergence,
            },
            class_filter=ForceFieldMDMaker,
        )
        self.convergence_md_maker = EquilibriumVolumeMaker(
            name="MP Morph MLFF Equilibrium Volume Maker",
            md_maker=self.md_maker,
            postprocessor=MPMorphEVPostProcess(),
        )  # TODO: check EV versus PV

        self.production_md_maker = self.md_maker.update_kwargs(
            update=dict(
                name="Production Run MLFF MD Maker",
                temperature=self.temperature,
                n_steps=self.steps_total_production,
            )
        )


@dataclass
class SlowQuenchMLFFMDMaker(SlowQuenchMaker):
    """Slow quench flow for quenching high temperature structures to low temperature using ForceFieldMDMaker.

    Quench's a provided structure with a molecular dyanmics run from a desired high temperature to
    a desired low temperature. Flow creates a series of MD runs that holds at a certain temperature
    and initiates the following MD run at a lower temperature (step-wise temperature MD runs).
    Adapted from MPMorph Workflow.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    md_maker :  ForceFieldMDMaker
        MDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    quench_start_temperature : int = 3000
        Starting temperature for quench; default 3000K
    quench_end_temperature : int = 500
        Ending temperature for quench; default 500K
    quench_temperature_step : int = 500
        Temperature step for quench; default 500K drop
    quench_n_steps : int = 1000
        Number of steps for quench; default 1000 steps
    """

    name: str = "ForceField slow quench"
    md_maker: ForceFieldMDMaker = field(default_factory=ForceFieldMDMaker)

    def call_md_maker(
        self,
        structure: Structure,
        temp: float,
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
    """Fast quench flow for quenching high temperature structures to 0K with MLFF.

    Quench's a provided structure with a single (or double) relaxation and a static calculation at 0K.
    Adapted from MPMorph Workflow.

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
class MPMorphLJMDMaker(MPMorphMLFFMDMaker):
    """Lennard Jones ForceField MPMorph flow for volume equilibration and production runs via molecular dynamics only.

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    for the volume and finally a production run at a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : int = 300
        Temperature of the equilibrium volume search and production run in Kelvin, default 300K
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
    temperature: int = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: LJMDMaker = field(default_factory=lambda: LJMDMaker(name="LJ MD Maker"))
    production_md_maker: LJMDMaker = field(
        default_factory=lambda: LJMDMaker(name="Production Run LJ MD Maker")
    )


@dataclass
class MPMorphSlowQuenchLJMDMaker(MPMorphMLFFMDMaker):
    """Lennard Jones ForceField MPMorph flow for volume equilibration, production run, and slow quench via molecular dynamics.

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    for the volume and a production run at a given temperature. Then proceed with a slow quench from high
    temeprature to low temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : int = 300
        Temperature of the equilibrium volume search and production run in Kelvin, default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : LJMDMaker
        LJMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    production_md_maker : LJMDMaker
        LJMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    """

    name: str = "MP Morph LJ MD Maker Slow Quench"
    temperature: int = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: LJMDMaker = field(default_factory=lambda: LJMDMaker(name="LJ MD Maker"))
    production_md_maker: LJMDMaker = field(
        default_factory=lambda: LJMDMaker(name="Production Run LJ MD Maker")
    )
    quench_maker: SlowQuenchMLFFMDMaker = field(
        default_factory=lambda: SlowQuenchMLFFMDMaker(md_maker=LJMDMaker())
    )


@dataclass
class MPMorphFastQuenchLJMDMaker(MPMorphMLFFMDMaker):
    """Lennard Jones ForceField MPMorph flow for volume equilibration, production runs, and fast quench via molecular dynamics with relaxation and statics.

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    for the volume and finally a production run at a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : int = 300
        Temperature of the equilibrium volume search and production run in Kelvin, default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : LJMDMaker
        LJMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    production_md_maker : LJMDMaker
        LJMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    """

    name: str = "MP Morph LJ MD Maker Fast Quench"
    temperature: int = 300
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
class MPMorphCHGNetMDMaker(MPMorphMLFFMDMaker):
    """CHGNet ML ForceField MPMorph flow for volume equilibration and production runs via molecular dynamics only.

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    for the volume and finally a production run at a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : int = 300
        Temperature of the equilibrium volume search and production run in Kelvin, default 300K
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
    temperature: int = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: CHGNetMDMaker = field(
        default_factory=lambda: CHGNetMDMaker(name="CHGNet MD Maker")
    )
    production_md_maker: CHGNetMDMaker = field(
        default_factory=lambda: CHGNetMDMaker(name="Production Run CHGNet MD Maker")
    )


@dataclass
class MPMorphSlowQuenchCHGNetMDMaker(MPMorphMLFFMDMaker):
    """CHGNet ML ForceField MPMorph flow for volume equilibration, production run, and slow quench via molecular dynamics.

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    for the volume and a production run at a given temperature. Then proceed with a slow quench from high
    temeprature to low temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : int = 300
        Temperature of the equilibrium volume search and production run in Kelvin, default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : CHGNetMDMaker
        CHGNetMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    production_md_maker : CHGNetMDMaker
        CHGNetMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    """

    name: str = "MP Morph CHGNet MD Maker Slow Quench"
    temperature: int = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: CHGNetMDMaker = field(
        default_factory=lambda: CHGNetMDMaker(name="CHGNet MD Maker")
    )
    production_md_maker: CHGNetMDMaker = field(
        default_factory=lambda: CHGNetMDMaker(name="Production Run CHGNet MD Maker")
    )
    quench_maker: SlowQuenchMLFFMDMaker = field(
        default_factory=lambda: SlowQuenchMLFFMDMaker(md_maker=CHGNetMDMaker())
    )


@dataclass
class MPMorphFastQuenchCHGNetMDMaker(MPMorphMLFFMDMaker):
    """CHGNet ML ForceField MPMorph flow for volume equilibration, production runs, and fast quench via molecular dynamics with relaxation and statics.

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    for the volume and finally a production run at a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : int = 300
        Temperature of the equilibrium volume search and production run in Kelvin, default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : CHGNetMDMaker
        CHGNetMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    production_md_maker : CHGNetMDMaker
        CHGNetMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    """

    name: str = "MP Morph CHGNet MD Maker Fast Quench"
    temperature: int = 300
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
class MPMorphM3GNetMDMaker(MPMorphMLFFMDMaker):
    """M3GNet ML ForceField MPMorph flow for volume equilibration and production runs via molecular dynamics only.

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    for the volume and finally a production run at a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : int = 300
        Temperature of the equilibrium volume search and production run in Kelvin, default 300K
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
    temperature: int = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: M3GNetMDMaker = field(
        default_factory=lambda: M3GNetMDMaker(name="M3GNet MD Maker")
    )
    production_md_maker: M3GNetMDMaker = field(
        default_factory=lambda: M3GNetMDMaker(name="Production Run M3GNet MD Maker")
    )


@dataclass
class MPMorphSlowQuenchM3GNetMDMaker(MPMorphMLFFMDMaker):
    """M3GNet ML ForceField MPMorph flow for volume equilibration, production run, and slow quench via molecular dynamics.

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    for the volume and a production run at a given temperature. Then proceed with a slow quench from high
    temeprature to low temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : int = 300
        Temperature of the equilibrium volume search and production run in Kelvin, default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : M3GNetMDMaker
        M3GNetMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    production_md_maker : M3GNetMDMaker
        M3GNetMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    """

    name: str = "MP Morph M3GNet MD Maker Slow Quench"
    temperature: int = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: M3GNetMDMaker = field(
        default_factory=lambda: M3GNetMDMaker(name="M3GNet MD Maker")
    )
    production_md_maker: M3GNetMDMaker = field(
        default_factory=lambda: M3GNetMDMaker(name="Production Run M3GNet MD Maker")
    )
    quench_maker: SlowQuenchMLFFMDMaker = field(
        default_factory=lambda: SlowQuenchMLFFMDMaker(md_maker=M3GNetMDMaker())
    )


@dataclass
class MPMorphFastQuenchM3GNetMDMaker(MPMorphMLFFMDMaker):
    """M3GNet ML ForceField MPMorph flow for volume equilibration, production runs, and fast quench via molecular dynamics with relaxation and statics.

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    for the volume and finally a production run at a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : int = 300
        Temperature of the equilibrium volume search and production run in Kelvin, default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : M3GNetMDMaker
        M3GNetMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    production_md_maker : M3GNetMDMaker
        M3GNetMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    """

    name: str = "MP Morph M3GNet MD Maker Fast Quench"
    temperature: int = 300
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
class MPMorphMACEMDMaker(MPMorphMLFFMDMaker):
    """MACE ML ForceField MPMorph flow for volume equilibration and production runs via molecular dynamics only.

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    for the volume and finally a production run at a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : int = 300
        Temperature of the equilibrium volume search and production run in Kelvin, default 300K
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
    temperature: int = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: MACEMDMaker = field(
        default_factory=lambda: MACEMDMaker(name="MACE MD Maker")
    )
    production_md_maker: MACEMDMaker = field(
        default_factory=lambda: MACEMDMaker(name="Production Run MACE MD Maker")
    )


@dataclass
class MPMorphSlowQuenchMACEMDMaker(MPMorphMLFFMDMaker):
    """MACE ML ForceField MPMorph flow for volume equilibration, production run, and slow quench via molecular dynamics.

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    for the volume and a production run at a given temperature. Then proceed with a slow quench from high
    temeprature to low temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : int = 300
        Temperature of the equilibrium volume search and production run in Kelvin, default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : MACEMDMaker
        MACEMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    production_md_maker : MACEMDMaker
        MACEMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    """

    name: str = "MP Morph MACE MD Maker Slow Quench"
    temperature: int = 300
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    md_maker: MACEMDMaker = field(
        default_factory=lambda: MACEMDMaker(name="MACE MD Maker")
    )
    production_md_maker: MACEMDMaker = field(
        default_factory=lambda: MACEMDMaker(name="Production Run MACE MD Maker")
    )
    quench_maker: SlowQuenchMLFFMDMaker = field(
        default_factory=lambda: SlowQuenchMLFFMDMaker(md_maker=MACEMDMaker())
    )


@dataclass
class MPMorphFastQuenchMACEMDMaker(MPMorphMLFFMDMaker):
    """MACE ML ForceField MPMorph flow for volume equilibration, production runs, and fast quench via molecular dynamics with relaxation and statics.

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    for the volume and finally a production run at a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : int = 300
        Temperature of the equilibrium volume search and production run in Kelvin, default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : MACEMDMaker
        MACEMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    production_md_maker : MACEMDMaker
        MACEMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    """

    name: str = "MP Morph MACE MD Maker Fast Quench"
    temperature: int = 300
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
