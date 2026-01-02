"""Define the VASP HiphiveMaker.

Uses hiPhive, phono3py, phonopy & alma/shengbte for calculating harmonic & anharmonic
props of phonon.
"""

# Basic Python packages
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.common.flows.hiphive import BaseHiphiveMaker
from atomate2.vasp.flows.core import DoubleRelaxMaker

# Atomate2 packages
from atomate2.vasp.jobs.core import TightRelaxMaker
from atomate2.vasp.jobs.phonons import PhononDisplacementMaker
from atomate2.vasp.sets.core import StaticSetGenerator, TightRelaxSetGenerator

if TYPE_CHECKING:
    from atomate2.vasp.jobs.base import BaseVaspMaker

logger = logging.getLogger(__name__)

__all__ = ["HiphiveMaker"]

__author__ = "Alex Ganose, Junsoo Park, Zhuoying Zhu, Hrushikesh Sahasrabuddhe"
__email__ = "aganose@lbl.gov, jsyony37@lbl.gov, zyzhu@lbl.gov, hpsahasrabuddhe@lbl.gov"


@dataclass
class HiphiveMaker(BaseHiphiveMaker):
    """
    Workflow to calc. interatomic force constants and vibrational props using hiPhive.

    A summary of the workflow is as follows:
    1. Structure relaxtion
    2. Calculate a supercell transformation matrix that brings the
       structure as close as cubic as possible, with all lattice lengths
       greater than 5 nearest neighbor distances.
    3. Perturb the atomic sites for each supercell using a Monte Carlo
       rattle procedure. The atoms are perturbed roughly according to a
       normal deviation. A number of standard deviation perturbation distances
       are included. Multiple supercells may be generated for each perturbation
       distance
    4. Run static VASP calculations on each perturbed supercell to calculate
       atomic forces.
    5. Aggregate the forces and conduct the fit atomic force constants using
       the minimization schemes in hiPhive.
    6. Output the interatomic force constants, and phonon band structure and
       density of states to the database.
    7. Optional: Perform phonon renormalization at finite temperature - useful
       when unstable modes exist
    8. Optional: Solve the lattice thermal conductivity using ShengBTE and
       output to the database.

    Args
    ----------
    name : str
        Name of the flows produced by this maker.
    static_energy_maker (BaseVaspMaker):
        The VASP input generator for static calculations, default is StaticMaker.
    bulk_relax_maker (BaseVaspMaker | None):
        The VASP input generator for bulk relaxation,
        default is DoubleRelaxMaker using TightRelaxMaker.
    phonon_displacement_maker (BaseVaspMaker | None):
        The VASP input generator for phonon displacement calculations,
        default is PhononDisplacementMaker.
    """

    name: str = "Lattice-Dynamics-VASP"
    bulk_relax_maker: DoubleRelaxMaker = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(
            TightRelaxMaker(
                input_set_generator=TightRelaxSetGenerator(
                    user_incar_settings={
                        "PREC": "Accurate",
                        "GGA": "PS",
                        "IBRION": 2,
                        "NSW": 20,
                        "NELMIN": 5,
                        "ISIF": 3,
                        # "ENCUT": 648.744200,
                        "EDIFF": 1.000000e-06,
                        # "EDIFFG": -1.000000e-08,
                        "ISMEAR": 0,
                        "SIGMA": 1.000000e-02,
                        "IALGO": 38,
                        "LREAL": ".FALSE.",
                        "ADDGRID": ".TRUE.",
                        "LWAVE": ".FALSE.",
                        "LCHARG": ".FALSE.",
                        "NPAR": 4,
                    }
                )
            )
        )
    )
    phonon_displacement_maker: BaseVaspMaker | None = field(
        default_factory=lambda: PhononDisplacementMaker(
            input_set_generator=StaticSetGenerator(
                user_kpoints_settings={"reciprocal_density": 100},
                user_incar_settings={
                    "ADDGRID": True,
                    "ALGO": "Normal",
                    "EDIFF": 1e-06,
                    # "EDIFFG": -1.000000e-08,
                    "ENCUT": 600,
                    "GGA": "PS",
                    "IBRION": -1,
                    "ISIF": 3,
                    "ISMEAR": 0,
                    "ISPIN": 2,
                    "LAECHG": False,
                    "LASPH": True,
                    "LCHARG": False,
                    "LORBIT": 11,
                    "LREAL": "Auto",
                    "LVHAR": False,
                    "LVTOT": False,
                    "LWAVE": False,
                    # "MAGMOM": 250*0.6,
                    "NCORE": 6,
                    "NELM": 100,
                    "NSW": 0,
                    "PREC": "Accurate",
                    "SIGMA": 0.1,  # changed from 0.1
                },
                # auto_ispin=True,
            )
        )
    )

    @property
    def prev_calc_dir_argname(self) -> str:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
        return "prev_dir"
