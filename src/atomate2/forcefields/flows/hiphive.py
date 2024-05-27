"""Define the ForceField HiphiveMaker.

Uses hiPhive, phono3py, phonopy & alma/shengbte for calculating harmonic & anharmonic
props of phonon.
"""

# Basic Python packages
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import ClassVar

from atomate2.common.flows.hiphive import BaseHiphiveMaker
from atomate2.forcefields.jobs import (
    CHGNetRelaxMaker,
    CHGNetStaticMaker,
    ForceFieldRelaxMaker,
    ForceFieldStaticMaker,
)

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
       distance.
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

    name: str = "Lattice-Dynamics-FORCE_FIELD"
    static_energy_maker: ForceFieldStaticMaker | None = field(
        default_factory=CHGNetStaticMaker
    )
    bulk_relax_maker: ForceFieldRelaxMaker = field(
        default_factory=lambda: CHGNetRelaxMaker(relax_kwargs={"fmax": 0.00001})
    )
    phonon_displacement_maker: ForceFieldStaticMaker = field(
        default_factory=CHGNetStaticMaker
    )
    # ff_displacement_maker: ForceFieldStaticMaker = field(
    #     default_factory=CHGNetStaticMaker
    # )
    # phonon_displacement_maker: ForceFieldStaticMaker = field(
    #     default_factory=lambda: CHGNetRelaxMaker(relax_kwargs={"fmax": 2})
    # )

    @property
    def prev_calc_dir_argname(self) -> None:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
        return

