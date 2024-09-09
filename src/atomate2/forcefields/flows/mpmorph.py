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

    from jobflow import Flow, Job
    from pymatgen.core import Structure


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

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

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
        SlowQuenchMaker - MLFFMDMaker that quenchs structure from
        high to low temperature
        FastQuenchMaker - DoubleRelaxMaker + Static that "quenches"
        structure to 0K
    """

    name: str = "MP Morph MLFF MD Maker"
    convergence_md_maker: EquilibriumVolumeMaker | None = None
    production_md_maker: ForceFieldMDMaker = field(default_factory=ForceFieldMDMaker)
    quench_maker: FastQuenchMLFFMDMaker | SlowQuenchMLFFMDMaker | None = None

    @classmethod
    def from_temperature_and_steps(
        cls,
        temperature: float,
        n_steps_convergence: int,
        n_steps_production: int,
        mlff_maker: ForceFieldMDMaker = ForceFieldMDMaker(),
    ) -> MPMorphMLFFMDMaker:
        """Create a MPMorphMLFFMDMaker from temperature and steps.
        Recommended for friendly user experience.

        Parameters
        ----------
        temperature : float
            Temperature of the equilibrium volume search and production run in Kelvin
        n_steps_convergence : int
            Number of steps for the convergence run(s)
        n_steps_production : int
            Total number of steps for the production run(s)
        mlff_maker : ForceFieldMDMaker
            MDMaker to generate the molecular dynamics jobs specifically for MLFF MDs.
            This is generalization to any MLFF MD Maker. E.g. LJMDMaker, CHGNetMDMaker, etc.
        """
        base_md_maker = mlff_maker.update_kwargs(
            update={
                "temperature": temperature,
                "n_steps": n_steps_convergence,
                "name": "Convergence MPMorph MLFF MD Maker",
            },
            class_filter=ForceFieldMDMaker,
        )

        updated_convergence_md_maker = EquilibriumVolumeMaker(
            name="MP Morph MLFF Equilibrium Volume Maker",
            md_maker=base_md_maker,
        )

        updated_production_md_maker = base_md_maker.update_kwargs(
            update=dict(
                name="Production Run MLFF MD Maker",
                temperature=temperature,
                n_steps=n_steps_production,
            )
        )
        return cls(
            name="MP Morph MLFF MD Maker",
            convergence_md_maker=updated_convergence_md_maker,
            production_md_maker=updated_production_md_maker,
        )


@dataclass
class OldMPMorphMLFFMDMaker(MPMorphMDMaker):
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
    md_maker : ForceFieldMDMaker
        MDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher;
        inherits from EquilibriumVolumeMaker and ForceFieldMDMaker (MLFF)
    quench_maker :  SlowQuenchMaker or FastQuenchMaker or None
        SlowQuenchMaker - MLFFMDMaker that quenchs structure from
        high to low temperature
        FastQuenchMaker - DoubleRelaxMaker + Static that "quenches"
        structure to 0K
    production_md_maker : ForceFieldMDMaker
        MDMaker to generate the production run(s);
        inherits from ForceFieldMDMaker (MLFF)
    """

    name: str = "MP Morph MLFF MD Maker"
    temperature: float = 300
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
        if self.convergence_md_maker is None:
            # Default to equilibrium volume maker
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
class MPMorphLJMDMaker(MPMorphMLFFMDMaker):
    """Lennard-Jones MPMorph flow for volume equilibration and production.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and finally a production run at
    a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    convergence_md_maker : EquilibriumVolumeMaker
        MDMaker to generate the equilibrium volumer searcher;
        uses EquilibriumVolumeMaker with LJMDMaker (MLFF)
    production_md_maker : LJMDMaker
        LJMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    """

    name: str = "MP Morph LJ MD Maker"
    convergence_md_maker: EquilibriumVolumeMaker = field(
        default_factory=lambda: EquilibriumVolumeMaker(
            name="MP Morph LJ Equilibrium Volume Maker",
            md_maker=LJMDMaker(),
        )
    )
    production_md_maker: LJMDMaker = field(
        default_factory=lambda: LJMDMaker(name="Production Run LJ MD Maker")
    )


@dataclass
class MPMorphSlowQuenchLJMDMaker(MPMorphMLFFMDMaker):
    """Lennard Jones ForceField MPMorph flow plus slow quench.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and a production run at a given temperature.
    Then proceed with a slow quench from high temperature to low temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher;
        uses EquilibriumVolumeMaker with LJMDMaker (MLFF)
    production_md_maker : LJMDMaker
        LJMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    quench_maker : SlowQuenchMLFFMDMaker
        Using the LJMDMaker to perform SlowQuenchMLFFMDMaker with the default settings
        Check SlowQuenchMLFFMDMaker for more information.
    """

    name: str = "MP Morph LJ MD Maker Slow Quench"
    convergence_md_maker: EquilibriumVolumeMaker = field(
        default_factory=lambda: EquilibriumVolumeMaker(
            name="MP Morph LJ Equilibrium Volume Maker",
            md_maker=LJMDMaker(),
        )
    )
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
class MPMorphFastQuenchLJMDMaker(MPMorphMLFFMDMaker):
    """Lennard Jones ForceField MPMorph flow plus fast quench.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and finally a production run at
    a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher;
        uses EquilibriumVolumeMaker with LJMDMaker (MLFF)
    production_md_maker : LJMDMaker
        LJMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    quench_maker : FastQuenchMLFFMDMaker
        Using the LJMDMaker to perform FastQuenchMLFFMDMaker with the default settings
        Check FastQuenchMLFFMDMaker for more information.
    """

    name: str = "MP Morph LJ MD Maker Fast Quench"
    convergence_md_maker: EquilibriumVolumeMaker = field(
        default_factory=lambda: EquilibriumVolumeMaker(
            name="MP Morph LJ Equilibrium Volume Maker",
            md_maker=LJMDMaker(),
        )
    )
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
    """CHGNet ML ForceField MPMorph flow for volume equilibration and production.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and finally a production run at a
    given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher;
        uses EquilibriumVolumeMaker with CHGNetMDMaker (MLFF)
    production_md_maker : CHGNetMDMaker
        CHGNetMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    """

    name: str = "MP Morph CHGNet MD Maker"
    convergence_md_maker: EquilibriumVolumeMaker = field(
        default_factory=lambda: EquilibriumVolumeMaker(
            name="MP Morph LJ Equilibrium Volume Maker",
            md_maker=CHGNetMDMaker(),
        )
    )
    production_md_maker: CHGNetMDMaker = field(
        default_factory=lambda: CHGNetMDMaker(name="Production Run CHGNet MD Maker")
    )


@dataclass
class MPMorphSlowQuenchCHGNetMDMaker(MPMorphMLFFMDMaker):
    """CHGNet ML ForceField MPMorph flow plus slow quench..

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and a production run at a given temperature.
    Then proceed with a slow quench from high temperature to low temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher;
        uses EquilibriumVolumeMaker with CHGNetMDMaker (MLFF)
    production_md_maker : CHGNetMDMaker
        CHGNetMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    quench_maker : SlowQuenchMLFFMDMaker
        Using the CHGNetMDMaker to perform SlowQuenchMLFFMDMaker with the default settings
        Check SlowQuenchMLFFMDMaker for more information.
    """

    name: str = "MP Morph CHGNet MD Maker Slow Quench"
    convergence_md_maker: EquilibriumVolumeMaker = field(
        default_factory=lambda: EquilibriumVolumeMaker(
            name="MP Morph LJ Equilibrium Volume Maker",
            md_maker=CHGNetMDMaker(),
        )
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
class MPMorphFastQuenchCHGNetMDMaker(MPMorphMLFFMDMaker):
    """CHGNet ML ForceField MPMorph flow plus fast quench.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and finally a production run at
    a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher;
        uses EquilibriumVolumeMaker with CHGNetMDMaker (MLFF)
    production_md_maker : CHGNetMDMaker
        CHGNetMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    quench_maker : FastQuenchMLFFMDMaker
        Using the CHGNetMDMaker to perform FastQuenchMLFFMDMaker with the default settings
        Check FastQuenchMLFFMDMaker for more information.
    """

    name: str = "MP Morph CHGNet MD Maker Fast Quench"
    convergence_md_maker: EquilibriumVolumeMaker = field(
        default_factory=lambda: EquilibriumVolumeMaker(
            name="MP Morph LJ Equilibrium Volume Maker",
            md_maker=CHGNetMDMaker(),
        )
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
    """M3GNet ML ForceField MPMorph flow for volume equilibration and production.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and finally a production run at a
    given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher;
        uses EquilibriumVolumeMaker with M3GNetMDMaker (MLFF)
    production_md_maker : M3GNetMDMaker
        M3GNetMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    """

    name: str = "MP Morph M3GNet MD Maker"
    convergence_md_maker: EquilibriumVolumeMaker = field(
        default_factory=lambda: EquilibriumVolumeMaker(
            name="MP Morph LJ Equilibrium Volume Maker",
            md_maker=M3GNetMDMaker(),
        )
    )
    production_md_maker: M3GNetMDMaker = field(
        default_factory=lambda: M3GNetMDMaker(name="Production Run M3GNet MD Maker")
    )


@dataclass
class MPMorphSlowQuenchM3GNetMDMaker(MPMorphMLFFMDMaker):
    """M3GNet ML ForceField MPMorph flow plus slow quench.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and a production run at a given temperature.
    Then proceed with a slow quench from high temperature to low temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher;
        uses EquilibriumVolumeMaker with M3GNetMDMaker (MLFF)
    production_md_maker : M3GNetMDMaker
        M3GNetMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    quench_maker : SlowQuenchMLFFMDMaker
        Using the M3GNetMDMaker to perform SlowQuenchMLFFMDMaker with the default settings
        Check SlowQuenchMLFFMDMaker for more information.
    """

    name: str = "MP Morph M3GNet MD Maker Slow Quench"
    convergence_md_maker: EquilibriumVolumeMaker = field(
        default_factory=lambda: EquilibriumVolumeMaker(
            name="MP Morph LJ Equilibrium Volume Maker",
            md_maker=M3GNetMDMaker(),
        )
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
class MPMorphFastQuenchM3GNetMDMaker(MPMorphMLFFMDMaker):
    """M3GNet ML ForceField MPMorph flow plus fast quench.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and finally a production run at a
    given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher;
        uses EquilibriumVolumeMaker with M3GNetMDMaker (MLFF)
    production_md_maker : M3GNetMDMaker
        M3GNetMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    quench_maker : FastQuenchMLFFMDMaker
        Using the M3GNetMDMaker to perform FastQuenchMLFFMDMaker with the default settings
        Check FastQuenchMLFFMDMaker for more information.
    """

    name: str = "MP Morph M3GNet MD Maker Fast Quench"
    convergence_md_maker: EquilibriumVolumeMaker = field(
        default_factory=lambda: EquilibriumVolumeMaker(
            name="MP Morph LJ Equilibrium Volume Maker",
            md_maker=M3GNetMDMaker(),
        )
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
    """MACE ML ForceField MPMorph flow for volume equilibration and production runs.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and finally a production run at a
    given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher;
        uses EquilibriumVolumeMaker with MACE MD Maker (MLFF)
    production_md_maker : MACEMDMaker
        MACEMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    """

    name: str = "MP Morph MACE MD Maker"
    convergence_md_maker: EquilibriumVolumeMaker = field(
        default_factory=lambda: EquilibriumVolumeMaker(
            name="MP Morph LJ Equilibrium Volume Maker",
            md_maker=MACEMDMaker(),
        )
    )
    production_md_maker: MACEMDMaker = field(
        default_factory=lambda: MACEMDMaker(name="Production Run MACE MD Maker")
    )


@dataclass
class MPMorphSlowQuenchMACEMDMaker(MPMorphMLFFMDMaker):
    """MACE ML ForceField MPMorph flow plus slow quench.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and a production run at a given temperature.
    Then proceed with a slow quench from high temperature to low temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher;
        uses EquilibriumVolumeMaker with MACE MD Maker (MLFF)
    production_md_maker : MACEMDMaker
        MACEMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    quench_maker : SlowQuenchMLFFMDMaker
        Using the MACEMDMaker to perform SlowQuenchMLFFMDMaker with the default settings
        Check SlowQuenchMLFFMDMaker for more information.
    """

    name: str = "MP Morph MACE MD Maker Slow Quench"
    convergence_md_maker: EquilibriumVolumeMaker = field(
        default_factory=lambda: EquilibriumVolumeMaker(
            name="MP Morph LJ Equilibrium Volume Maker",
            md_maker=MACEMDMaker(),
        )
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
class MPMorphFastQuenchMACEMDMaker(MPMorphMLFFMDMaker):
    """MACE ML ForceField MPMorph flow plus fast quench.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and finally a production run
    at a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher;
        uses EquilibriumVolumeMaker with MACE MD Maker (MLFF)
    production_md_maker : MACEMDMaker
        MACEMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    quench_maker : FastQuenchMLFFMDMaker
        Using the MACEMDMaker to perform FastQuenchMLFFMDMaker with the default settings
        Check FastQuenchMLFFMDMaker for more information.
    """

    name: str = "MP Morph MACE MD Maker Fast Quench"
    convergence_md_maker: EquilibriumVolumeMaker = field(
        default_factory=lambda: EquilibriumVolumeMaker(
            name="MP Morph LJ Equilibrium Volume Maker",
            md_maker=MACEMDMaker(),
        )
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
