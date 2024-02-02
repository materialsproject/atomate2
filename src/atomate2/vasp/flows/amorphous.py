"""Flows adapted from MPMorph *link to origin github repo*"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.vasp.flows.md import MultiMDMaker
from atomate2.vasp.jobs.md import MDMaker
from atomate2.vasp.sets.core import MDSetGenerator

from pymatgen.io.vasp import Kpoints

import math

if TYPE_CHECKING:

    from atomate2.common.jobs.equilibrate import EquilibrateVolumeMaker
    from atomate2.common.flows.amorphous import (
        FastQuenchMaker,
        SlowQuenchMaker,
        MPMorphMDMaker,
    )


@dataclass
class MPMorphVaspMDMaker(MPMorphMDMaker):

    name: str = "MP Morph vasp md"
    temperature: int = 300
    end_temp: int | None = None
    steps_convergence: int = 2000
    steps_quench: int = 500  # check old MPMorph code
    quench_temps: dict = {
        "start_temp": 3000,
        "end_temp": 500,
        "temp_step": 500,
    }
    steps_production: int = 10000

    convergence_md_maker: EquilibrateVolumeMaker | None = field(
        default_factory=EquilibrateVolumeMaker(
            convergence_md_maker=field(
                default_factory=MDMaker(
                    input_set_generator=MDSetGenerator(
                        ensemble="nvt",
                        start_temp=temperature,
                        end_temp=end_temp,
                        nsteps=steps_convergence,
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
                    ),
                )
            )
        )
    )  # check logic on this line

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
    base_production_md_maker: MDMaker = field(
        MDMaker(
            input_set_generator=MDSetGenerator(
                ensemble="nvt",
                start_temp=temperature,
                end_temp=end_temp,
                nsteps=5000,
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
        )
    )
    production_md_maker: MultiMDMaker = field(
        default_factory=MultiMDMaker(
            md_makers=[MDMaker for _ in range(math.ceil(steps_production / 5000))]
        )
    )
