"""Flows adapted from MPMorph *link to origin github repo*"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Response

from atomate2.vasp.flows.md import MultiMDMaker
from atomate2.vasp.jobs.md import MDMaker
from atoamte2.vasp.jobs.mpmorph import BaseMPMorphMDMaker
from atomate2.vasp.sets.core import MDSetGenerator

from pymatgen.io.vasp import Kpoints

from atomate2.common.flows.amorphous import (
    FastQuenchMaker,
    SlowQuenchMaker,
    MPMorphMDMaker,
    EquilibriumVolumeMaker,
)
from atomate2.vasp.powerups import update_user_incar_settings

import math

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pathlib import Path
    from jobflow import Flow, Maker

"""
@dataclass
class MPMorphVaspEquilibriumVolumeMaker(EquilibriumVolumeMaker):
    # TODO: docstr, do we need this???
    name: str = "MP Morph VASP Equilibrium Volume Maker"
    md_maker: Maker = BaseMPMorphMDMaker
"""

@dataclass
class MPMorphVaspMDMaker(MPMorphMDMaker):
    """ TODO: docstr """
    name: str = "MP Morph VASP MD Maker"
    temperature: int = 300
    end_temp: int = 300
    #steps_convergence: int = 2000
    steps_quench: int = 500  # check old MPMorph code
    quench_temps: dict = {
        "start_temp": 3000,
        "end_temp": 500,
        "temp_step": 500,
    }
    steps_single_production_run : int | None = 5000
    steps_total_production: int = 10000

    md_maker : MDMaker | None = BaseMPMorphMDMaker
    convergence_md_maker: EquilibriumVolumeMaker | None = None
    quench_maker: FastQuenchMaker | SlowQuenchMaker | None = field(
        default_factory=SlowQuenchMaker(
            md_maker=MDMaker(
                input_set_generator=MDSetGenerator(
                    ensemble="nvt",
                    start_temp=temperature,
                    end_temp=temperature,
                    nsteps=steps_quench,
                    time_step=2,
                    # adapted from MPMorph settings
                    user_incar_settings={
                        "ISPIN": 1,  # Do not consider magnetism in AIMD simulations
                        "LREAL": "Auto",  # Peform calculation in real space for AIMD due to large unit cell size
                        "LAECHG": False,  # Don't need AECCAR for AIMD
                        "EDIFFG": None,  # Does not apply to MD simulations, see: https://www.vasp.at/wiki/index.php/EDIFFG
                        "GGA": "PS",  # Just let VASP decide based on POTCAR - the default, PS yields the error below
                        "LPLANE": False,  # LPLANE is recommended to be False on Cray machines (https://www.vasp.at/wiki/index.php/LPLANE)
                        "LDAUPRINT": 0,
                    },
                    user_kpoints_settings=Kpoints(
                        comment="Gamma only",
                        num_kpts=1,
                        kpts=[[0, 0, 0]],
                        kpts_weights=[1.0],
                    ),
                )
            ),
            quench_tempature_setup=quench_temps,
        )
    )
    production_md_maker : Maker = BaseMPMorphMDMaker

    def make(
        self,
        structure : Structure,
        prev_dir : str | Path | None = None
    ) -> Flow:
        
        # TODO: check if this properly updates INCAR for all MD runs
        self.md_maker = update_user_incar_settings(
            flow = self.md_maker,
            incar_updates = {
                "TEBEG": self.temperature,
                "TEEND": self.end_temp,
            }
        )
        self.convergence_md_maker = EquilibriumVolumeMaker(
            name = "MP Morph VASP Equilibrium Volume Maker",
            md_maker = self.md_maker
        )

        if self.steps_single_production_run is None:
            self.steps_single_production_run = self.steps_total_production

        self.production_md_maker = update_user_incar_settings(
            flow = self.production_md_maker,
            incar_updates = {
                "TEBEG": self.temperature,
                "TEEND": self.end_temp,
                "NSW": self.steps_single_production_run
            }
        )

        if self.steps_total_production > self.steps_single_production_run:
            n_prod_md_steps = math.ceil(
                self.steps_total_production/self.steps_single_production_run
            )
            self.production_md_maker = Response(
                replace = MultiMDMaker(
                    md_makers = [self.production_md_maker for _ in range(n_prod_md_steps)]
                )
            )

        return super().make.original(self, structure = structure, prev_dir = prev_dir)