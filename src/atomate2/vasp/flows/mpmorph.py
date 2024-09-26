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
from atomate2.vasp.jobs.mpmorph import (
    BaseMPMorphMDMaker,
    FastQuenchVaspMaker,
    SlowQuenchVaspMaker,
)
from atomate2.vasp.powerups import update_user_incar_settings

if TYPE_CHECKING:
    from typing import Self

    from jobflow import Maker

    from atomate2.vasp.jobs.base import BaseVaspMaker
    from atomate2.vasp.jobs.md import MDMaker


@dataclass
class MPMorphVaspMDMaker(MPMorphMDMaker):
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

    name: str = "MP Morph VASP MD Maker"
    convergence_md_maker: EquilibriumVolumeMaker = field(
        default_factory=lambda: EquilibriumVolumeMaker(md_maker=BaseMPMorphMDMaker())
    )
    production_md_maker: MDMaker | MultiMDMaker = field(
        default_factory=BaseMPMorphMDMaker
    )

    @classmethod
    def from_temperature_and_steps(  # type: ignore[override]
        cls,
        temperature: float,
        n_steps_convergence: int = 5000,
        n_steps_production: int = 10000,
        end_temp: float | None = None,
        md_maker: Maker = BaseMPMorphMDMaker,
        n_steps_per_production_run: int | None = None,
        quench_maker: FastQuenchMaker | SlowQuenchMaker | None = None,
    ) -> Self:
        """
        Create VASP MPMorph flow from a temperature and number of steps.

        Parameters
        ----------
        temperature : float
            The (starting) temperature
        n_steps_convergence : int = 5000
            The number of steps used in MD runs for equilibrating structures.
        n_steps_production : int = 10000
            The number of steps used in MD production runs.
        end_temp : float or None
            If a float, the temperature to ramp down to in the production run.
            If None (default), set to `temperature`.
        base_md_maker : Maker
            The Maker used to start MD runs.
        n_steps_per_production_run : int or None (default)
            If an int, the number of steps to use per production run,
            using MultiMDMaker to orchestrate chained production runs.
        quench_maker :  SlowQuenchMaker or FastQuenchMaker or None
            SlowQuenchMaker - MDMaker that quenches structure from
                high to low temperature
            FastQuenchMaker - DoubleRelaxMaker + Static that "quenches"
                structure at 0K
        """
        end_temp = end_temp or temperature

        conv_md_maker = update_user_incar_settings(
            flow=md_maker(),
            incar_updates={
                "TEBEG": temperature,
                "TEEND": temperature,
                "NSW": n_steps_convergence,
            },
        )
        conv_md_maker = conv_md_maker.update_kwargs(
            update={"name": "Convergence MPMorph VASP MD Maker"}
        )

        convergence_md_maker = EquilibriumVolumeMaker(
            name="MP Morph VASP Equilibrium Volume Maker", md_maker=conv_md_maker
        )

        if n_steps_per_production_run is None:
            n_steps_per_production_run = n_steps_production
            n_production_runs = 1
        else:
            n_production_runs = math.ceil(
                n_steps_production / n_steps_per_production_run
            )

        production_md_maker = update_user_incar_settings(
            flow=md_maker(),
            incar_updates={
                "TEBEG": temperature,
                "TEEND": temperature,
                "NSW": n_steps_convergence,
            },
        )

        production_md_maker = production_md_maker.update_kwargs(
            update={"name": "Production MPMorph VASP MD Maker"}
        )

        if n_production_runs > 1:
            production_md_maker = Response(
                replace=MultiMDMaker(
                    name="Production MPMorph VASP MultiMD Maker",
                    md_makers=[production_md_maker for _ in range(n_production_runs)],
                )
            )

        return cls(
            name="MP Morph VASP MD Maker",
            convergence_md_maker=convergence_md_maker,
            production_md_maker=production_md_maker,
            quench_maker=quench_maker,
        )


@dataclass
class MPMorphSlowQuenchVaspMDMaker(MPMorphVaspMDMaker):
    """VASP MPMorph flow plus slow quench.

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
        inherits from EquilibriumVolumeMaker and MDMaker (VASP)
    production_md_maker : BaseMPMorphMDMaker
        MDMaker to generate the production run(s); inherits from MDMaker
        (VASP) or MultiMDMaker.
    quench_maker :  SlowQuenchVaspMaker
        SlowQuenchVaspMaker - MDMaker that quenches structure from high
        to low temperature in piece-wise ('stepwise') AIMD runs.
        Check atomate2.vasp.jobs.mpmorph for SlowQuenchVaspMaker.
    """

    name: str = "MP Morph VASP MD Maker Slow Quench"
    convergence_md_maker: EquilibriumVolumeMaker = field(
        default_factory=lambda: EquilibriumVolumeMaker(md_maker=BaseMPMorphMDMaker())
    )
    production_md_maker: MDMaker = field(default_factory=BaseMPMorphMDMaker)
    quench_maker: SlowQuenchVaspMaker = field(
        default_factory=lambda: SlowQuenchVaspMaker(
            BaseMPMorphMDMaker(name="Slow Quench VASP Maker"),
            quench_n_steps=1000,
            quench_temperature_step=500,
            quench_end_temperature=500,
            quench_start_temperature=3000,
            descent_method="stepwise",
        )
    )


@dataclass
class MPMorphFastQuenchVaspMDMaker(MPMorphVaspMDMaker):
    """
    VASP MPMorph flow including multiple production runs and slow quench.

    Calculates the equilibrium volume of a structure at a given temperature.
    A convergence fitting for the volume and finally a production run at a
    given temperature. Runs a "Fast Quench" at 0K using a double relaxation
    plus static.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher;
        inherits from EquilibriumVolumeMaker and MDMaker (VASP)
    production_md_maker : BaseMPMorphMDMaker
        MDMaker to generate the production run(s); inherits from
        MDMaker (VASP) or MultiMDMaker.
    quench_maker :  FastQuenchVaspMaker
        FastQuenchVaspMaker - MDMaker that quenches structure from
        high temperature to 0K.
        Check atomate2.vasp.jobs.mpmorph for FastQuenchVaspMaker.
    """

    name: str = "MP Morph VASP MD Maker Fast Quench"
    convergence_md_maker: EquilibriumVolumeMaker = field(
        default_factory=lambda: EquilibriumVolumeMaker(md_maker=BaseMPMorphMDMaker())
    )
    production_md_maker: MDMaker = field(default_factory=BaseMPMorphMDMaker)
    quench_maker: BaseVaspMaker = field(default_factory=FastQuenchVaspMaker)
