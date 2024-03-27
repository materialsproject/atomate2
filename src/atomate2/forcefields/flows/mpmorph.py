"""Flows adapted from MPMorph *link to origin github repo*"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Response

from atomate2.common.flows.mpmorph import (
    FastQuenchMaker,
    SlowQuenchMaker,
    MPMorphMDMaker,
    EquilibriumVolumeMaker,
)

from atomate2.forcefields.md import ForceFieldMDMaker

import math

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pathlib import Path
    from jobflow import Flow, Maker


@dataclass
class MPMorphMLFFMDMaker(MPMorphMDMaker):
    """ML ForceField MPMorph flow for volume equilibration, quench, and production runs via molecular dynamics

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    (optional) for the volume followed by quench (optional) from high temperature to low temperature
    and finally a production run(s) at a given temperature. Production run is broken up into multiple
    smaller steps to ensure simulation does not hit wall time limits.

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
    quench_tempature_setup: dict = None
        Only needed for SlowQuenchMaker. Setup for slow quenching the structure from high temperature to low temperature
        Example:
        quench_tempature_setup ={
            "start_temp": 3000, # Starting temperature for quench
            "end_temp": 500, # Ending temperature for quench
            "temp_step": 500, # Temperature step for quench
            "nsteps": 1000, # Number of steps for quench
        }
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher; inherits from EquilibriumVolumeMaker and ForceFieldMDMaker (MLFF)
    production_md_maker : ForceFieldMDMaker
        MDMaker to generate the production run(s); inherits from ForceFieldMDMaker (MLFF)
    quench_maker :  SlowQuenchMaker or FastQuenchMaker or None
        SlowQuenchMaker - MDMaker that quenchs structure from high temperature to low temperature
        FastQuenchMaker - DoubleRelaxMaker + Static that "quenchs" structure at 0K
    """

    name: str = "MP Morph VASP MD Maker"
    temperature: int = 300
    end_temp: int = 300  # DEPRECATED: Unusable for MLFF
    steps_convergence: int | None = None
    steps_total_production: int = 10000

    quench_tempature_setup: dict | None = None

    md_maker: ForceFieldMDMaker | None = ForceFieldMDMaker
    convergence_md_maker: EquilibriumVolumeMaker | None = None
    production_md_maker: ForceFieldMDMaker = ForceFieldMDMaker

    quench_maker: FastQuenchMaker | SlowQuenchMaker | None = None

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:

        # TODO: check if this properly updates INCAR for all MD runs
        if self.steps_convergence is None:
            self.md_maker = update_user_incar_settings(
                flow=self.md_maker,
                incar_updates={
                    "TEBEG": self.temperature,
                    "TEEND": self.temperature,  # Equilibrium volume search is only at single temperature (temperature sweep not allowed)
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
                    quench_tempature_setup=self.quench_tempature_setup
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

        return super().make.original(self, structure=structure, prev_dir=prev_dir)
