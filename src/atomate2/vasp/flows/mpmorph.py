"""Define VASP implementations of MPMorph flows.

For information about the current flows, contact:
- Bryant Li (@BryantLi-BLI)
- Aaron Kaplan (@esoteric-ephemera)
- Max Gallant (@mcgalcode)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Response

from atomate2.common.flows.mpmorph import (
    EquilibriumVolumeMaker,
    FastQuenchMaker,
    MPMorphMDMaker,
    SlowQuenchMaker,
)
from atomate2.vasp.flows.md import MultiMDMaker
from atomate2.vasp.jobs.mp import (
    MPMetaGGARelaxMaker,
    MPMetaGGAStaticMaker,
    MPPreRelaxMaker,
)
from atomate2.vasp.jobs.mpmorph import (
    BaseMPMorphMDMaker,
    FastQuenchVaspMaker,
    SlowQuenchVaspMaker,
)
from atomate2.vasp.powerups import update_user_incar_settings

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from jobflow import Flow
    from pymatgen.core import Structure

    from atomate2.vasp.jobs.md import MDMaker


@dataclass
class BaseMPMorphVaspMDMaker(MPMorphMDMaker):
    """Base MPMorph flow for amorphous solid equilibration using VASP.

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
    end_temp : float = 300
        End temperature of the equilibrium volume search and production run in Kelvin,
        default 300K.
        Use only for lowering temperarture for production run
    md_maker : BaseMPMorphMDMaker
        MDMaker to generate the molecular dynamics jobs specifically for MPMorph AIMD;
        inherits from MDMaker (VASP)
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    steps_single_production_run: int | None = 5000
        Number of steps for a single production run; default 5000 steps.
        If set and steps_total_production > steps_single_production_run,
        multiple production runs (MultiMDMaker) will be generated.
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher;
        inherits from EquilibriumVolumeMaker and MDMaker (VASP)
    quench_maker :  SlowQuenchMaker or FastQuenchMaker or None
        SlowQuenchMaker - MDMaker that quenches structure from high to low temperature
        FastQuenchMaker - DoubleRelaxMaker + Static that "quenches" structure at 0K
    production_md_maker : BaseMPMorphMDMaker
        MDMaker to generate the production run(s);
        inherits from MDMaker (VASP) or MultiMDMaker
    """

    name: str = "MP Morph VASP Skeleton MD Maker"
    temperature: float = 300
    end_temp: float = 300
    steps_convergence: int | None = None
    steps_single_production_run: int | None = 5000
    steps_total_production: int = 10000

    md_maker: MDMaker = field(default_factory=BaseMPMorphMDMaker)
    convergence_md_maker: EquilibriumVolumeMaker | None = None
    production_md_maker: MDMaker = field(default_factory=BaseMPMorphMDMaker)

    quench_maker: FastQuenchVaspMaker | SlowQuenchVaspMaker | None = None

    def _post_init_update(self) -> None:
        """Ensure that VASP input sets correctly set temperature."""
        # TODO: check if this properly updates INCAR for all MD runs
        if self.steps_convergence is None:
            # Equilibrium volume search is only at single temperature
            # (temperature sweep not allowed)
            self.md_maker = update_user_incar_settings(
                flow=self.md_maker,
                incar_updates={
                    "TEBEG": self.temperature,
                    "TEEND": self.temperature,
                },
            )
        elif (
            self.steps_convergence is not None
        ):  # TODO: make this elif statement more efficient
            self.md_maker = update_user_incar_settings(
                flow=self.md_maker,
                incar_updates={
                    "TEBEG": self.temperature,
                    "TEEND": self.end_temp,
                    "NSW": self.steps_convergence,
                },
            )
        if self.convergence_md_maker is None:
            self.convergence_md_maker = EquilibriumVolumeMaker(
                name="MP Morph VASP Equilibrium Volume Maker", md_maker=self.md_maker
            )

        if self.steps_single_production_run is None:
            self.steps_single_production_run = self.steps_total_production

        self.production_md_maker = update_user_incar_settings(
            flow=self.production_md_maker,
            incar_updates={
                "TEBEG": self.temperature,
                "TEEND": self.end_temp,
                "NSW": self.steps_single_production_run,
            },
        )

        if self.steps_total_production > self.steps_single_production_run:
            n_prod_md_steps = math.ceil(
                self.steps_total_production / self.steps_single_production_run
            )
            self.production_md_maker = Response(
                replace=MultiMDMaker(
                    md_makers=[self.production_md_maker for _ in range(n_prod_md_steps)]
                )
            )


@dataclass
class MPMorphVaspMDMaker(BaseMPMorphVaspMDMaker):
    """VASP MPMorph flow for volume equilibration and single production run.

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
    end_temp : float = 300
        End temperature of the equilibrium volume search and production run in Kelvin,
        default 300K.
        Use only for lowering temperarture for production run.
    md_maker : BaseMPMorphMDMaker
        MDMaker to generate the molecular dynamics jobs specifically for MPMorph AIMD;
        inherits from MDMaker (VASP)
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    steps_single_production_run: int | None = None
        This maker only generates a single production run;
        check base or MPMorphVASPMultiMDMaker for multiple production runs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    production_md_maker : BaseMPMorphMDMaker
        MDMaker to generate the production run(s);
        inherits from MDMaker (VASP) or MultiMDMaker
    """

    name: str = "MP Morph VASP MD Maker"
    temperature: float = 300
    end_temp: float = 300
    steps_convergence: int | None = None
    steps_single_production_run: int | None = None
    steps_total_production: int = 10000

    md_maker: MDMaker = field(default_factory=BaseMPMorphMDMaker)
    production_md_maker: MDMaker = field(default_factory=BaseMPMorphMDMaker)


@dataclass
class MPMorphVaspMultiMDMaker(BaseMPMorphVaspMDMaker):
    """VASP MPMorph flow for volume equilibration and multiple production runs.

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
    end_temp : float = 300
        End temperature of the equilibrium volume search and production run in Kelvin,
        default 300K. Use only for lowering temperarture for production run.
    md_maker : BaseMPMorphMDMaker
        MDMaker to generate the molecular dynamics jobs specifically for MPMorph AIMD;
        inherits from MDMaker (VASP)
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    steps_single_production_run: int | None = 5000
        Number of steps for a single production run; default 5000 steps.
        If set and steps_total_production > steps_single_production_run,
        multiple production runs (MultiMDMaker) will be generated.
    steps_total_production: int = 10000
        Total number of steps for the production run(s);
        default 10000 steps (in this default 10000/5000 = 2 individual production runs).
    production_md_maker : BaseMPMorphMDMaker
        MDMaker to generate the production run(s);
        inherits from MDMaker (VASP) or MultiMDMaker
    """

    name: str = "MP Morph VASP Multi MD Maker"
    temperature: float = 300
    end_temp: float = 300
    steps_convergence: int | None = None
    steps_single_production_run: int | None = 5000
    steps_total_production: int = 10000

    md_maker: MDMaker = field(default_factory=BaseMPMorphMDMaker)
    production_md_maker: MDMaker = field(default_factory=BaseMPMorphMDMaker)
    # TODO: change this into:
    # production_md_maker: MultiMDMaker = field(
    #   default_factory= lambda: MultiMDMaker(
    #       md_makers=[
    #           BaseMPMorphMDMaker for _ in range(
    #               steps_total_production/steps_single_production_run
    #           )
    #       ]
    #   )
    # )


@dataclass
class MPMorphVaspMDSlowQuenchMaker(BaseMPMorphVaspMDMaker):
    """VASP MPMorph flow plus slow quench.

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
    end_temp : float = 300
        End temperature of the equilibrium volume search and production run in Kelvin,
        default 300K. Use only for lowering temperarture for production run.
    md_maker : BaseMPMorphMDMaker
        MDMaker to generate the molecular dynamics jobs specifically for MPMorph AIMD;
        inherits from MDMaker (VASP)
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    steps_single_production_run: int | None = 5000
        Number of steps for a single production run; default 5000 steps.
        If set and steps_total_production > steps_single_production_run,
        multiple production runs (MultiMDMaker) will be generated.
    steps_total_production: int = 10000
        Total number of steps for the production run(s);
        default 10000 steps (in this default 10000/5000 = 2individual production runs).
    production_md_maker : BaseMPMorphMDMaker
        MDMaker to generate the production run(s); inherits from MDMaker
        (VASP) or MultiMDMaker.
    quench_maker :  SlowQuenchVaspMaker
        SlowQuenchVaspMaker - MDMaker that quenches structure from high
        to low temperature in piece-wise AIMD runs.
        Check atomate2.vasp.jobs.mpmorph for SlowQuenchVaspMaker.
    """

    name: str = "MP Morph VASP MD Maker Slow Quench"
    temperature: float = 300
    end_temp: float = 300
    steps_convergence: int | None = None
    steps_single_production_run: int | None = None
    steps_total_production: int = 10000

    md_maker: MDMaker = field(default_factory=BaseMPMorphMDMaker)
    production_md_maker: MDMaker = field(default_factory=BaseMPMorphMDMaker)
    quench_maker: SlowQuenchVaspMaker = field(
        default_factory=lambda: SlowQuenchVaspMaker(
            BaseMPMorphMDMaker(name="Slow Quench VASP Maker"),
            quench_n_steps=1000,
            quench_temperature_step=500,
            quench_end_temperature=500,
            quench_start_temperature=3000,
        )
    )


@dataclass
class MPMorphVaspMDFastQuenchMaker(BaseMPMorphVaspMDMaker):
    """
    VASP MPMorph flow including multiple production runs and slow quench.

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
    end_temp : float = 300
        End temperature of the equilibrium volume search and production run in Kelvin,
        default 300K. Use only for lowering temperarture for production run.
    md_maker : BaseMPMorphMDMaker
        MDMaker to generate the molecular dynamics jobs specifically for MPMorph AIMD;
        inherits from MDMaker (VASP).
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    steps_single_production_run: int | None = 5000
        Number of steps for a single production run; default 5000 steps.
        If set and steps_total_production > steps_single_production_run,
        multiple production runs (MultiMDMaker) will be generated.
    steps_total_production: int = 10000
        Total number of steps for the production run(s);
        default 10000 steps (in this default 10000/5000 = 2
        individual production runs).
    production_md_maker : BaseMPMorphMDMaker
        MDMaker to generate the production run(s); inherits from
        MDMaker (VASP) or MultiMDMaker.
    quench_maker :  FastQuenchVaspMaker
        FastQuenchVaspMaker - MDMaker that quenches structure from
        high temperature to 0K.
        Check atomate2.vasp.jobs.mpmorph for FastQuenchVaspMaker.
    """

    name: str = "MP Morph VASP MD Maker Fast Quench"
    temperature: float = 300
    end_temp: float = 300
    steps_convergence: int | None = None
    steps_single_production_run: int | None = None
    steps_total_production: int = 10000

    md_maker: MDMaker = field(default_factory=BaseMPMorphMDMaker)
    production_md_maker: MDMaker = field(default_factory=BaseMPMorphMDMaker)
    quench_maker: FastQuenchVaspMaker = field(default_factory=FastQuenchVaspMaker)


# TODO: Below is first version of MPMorphVaspMDMaker;
# remove once all other MPMorphVaspMDMaker are updated and tested
@dataclass
class MPMorphVaspOldMDMaker(MPMorphMDMaker):
    """
    Skeleton VASP MPMorph flow for volume equilibration, quench, and production runs.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting (optional) for the volume followed by quench (optional)
    from high temperature to low temperature
    and finally a production run(s) at a given temperature.
    Production run is broken up into multiple
    smaller steps to ensure simulation does not hit wall time limits.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : float = 300
        Temperature of the equilibrium volume search and production run in Kelvin,
        default 300K
    end_temp : float = 300
        End temperature of the equilibrium volume search and production run in Kelvin,
        default 300K. Use only for lowering temperarture for production run.
    md_maker : BaseMPMorphMDMaker
        MDMaker to generate the molecular dynamics jobs specifically for MPMorph AIMD;
        inherits from MDMaker (VASP)
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    steps_single_production_run: int | None = 5000
        Number of steps for a single production run; default 5000 steps.
        If set and steps_total_production > steps_single_production_run,
        multiple production runs (MultiMDMaker) will be generated
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    quench_tempature_setup: dict = None
        Only needed for SlowQuenchMaker.
        Setup for slow quenching the structure from high to low temperature.
        Example:
        quench_tempature_setup ={
            "start_temp": 3000, # Starting temperature for quench
            "end_temp": 500, # Ending temperature for quench
            "temp_step": 500, # Temperature step for quench
            "n_steps": 1000, # Number of steps for quench
        }
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher;
        inherits from EquilibriumVolumeMaker and MDMaker (VASP).
    production_md_maker : BaseMPMorphMDMaker
        MDMaker to generate the production run(s); inherits from
        MDMaker (VASP) or MultiMDMaker/
    quench_maker :  SlowQuenchMaker or FastQuenchMaker or None
        SlowQuenchMaker - MDMaker that quenches structure from high to low temperature.
        FastQuenchMaker - DoubleRelaxMaker + Static that "quenches" structure at 0K.
    """

    name: str = "MP Morph VASP Skeleton MD Maker"
    temperature: float = 300
    end_temp: float = 300
    steps_convergence: int | None = None
    steps_single_production_run: int | None = 5000
    steps_total_production: int = 10000

    quench_tempature_setup: dict | None = None

    md_maker: MDMaker | None = field(default_factory=BaseMPMorphMDMaker)
    convergence_md_maker: EquilibriumVolumeMaker | None = None
    production_md_maker: MDMaker = field(default_factory=BaseMPMorphMDMaker)

    quench_maker: FastQuenchVaspMaker | SlowQuenchVaspMaker | None = None

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """Run a VASP Skeleton VASP MPMorph flow."""
        # TODO: check if this properly updates INCAR for all MD runs
        if self.steps_convergence is None:
            # Equilibrium volume search is only at single temperature
            # (temperature sweep not allowed)
            self.md_maker = update_user_incar_settings(
                flow=self.md_maker,
                incar_updates={
                    "TEBEG": self.temperature,
                    "TEEND": self.temperature,
                },
            )
        elif (
            self.steps_convergence is not None
        ):  # TODO: make this elif statement more efficient
            self.md_maker = update_user_incar_settings(
                flow=self.md_maker,
                incar_updates={
                    "TEBEG": self.temperature,
                    "TEEND": self.end_temp,
                    "NSW": self.steps_convergence,
                },
            )
        self.convergence_md_maker = EquilibriumVolumeMaker(
            name="MP Morph VASP Equilibrium Volume Maker", md_maker=self.md_maker
        )

        if self.steps_single_production_run is None:
            self.steps_single_production_run = self.steps_total_production

        self.production_md_maker = update_user_incar_settings(
            flow=self.production_md_maker,
            incar_updates={
                "TEBEG": self.temperature,
                "TEEND": self.end_temp,
                "NSW": self.steps_single_production_run,
            },
        )

        if self.steps_total_production > self.steps_single_production_run:
            n_prod_md_steps = math.ceil(
                self.steps_total_production / self.steps_single_production_run
            )
            self.production_md_maker = Response(
                replace=MultiMDMaker(
                    md_makers=[self.production_md_maker for _ in range(n_prod_md_steps)]
                )
            )

        if self.quench_maker is not None:
            if isinstance(self.quench_maker, SlowQuenchMaker):
                self.quench_maker = SlowQuenchMaker(
                    md_maker=self.md_maker,
                    quench_tempature_setup=self.quench_tempature_setup,
                )
            elif isinstance(self.quench_maker, FastQuenchMaker):
                self.quench_maker = FastQuenchMaker(
                    relax_maker=MPPreRelaxMaker,
                    relax_maker2=MPMetaGGARelaxMaker(
                        copy_vasp_kwargs={
                            "additional_vasp_files": ("WAVECAR", "CHGCAR")
                        }
                    ),
                    static_maker=MPMetaGGAStaticMaker(
                        copy_vasp_kwargs={
                            "additional_vasp_files": ("WAVECAR", "CHGCAR")
                        }
                    ),
                )

        return super().make(structure=structure, prev_dir=prev_dir)
