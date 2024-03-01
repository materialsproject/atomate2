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
    task_document_kwargs (dict):
        Keyword arguments for task document, default is {"task_label": "dummy_label"}.
    static_energy_maker (BaseVaspMaker):
        The VASP input generator for static calculations, default is StaticMaker.
    bulk_relax_maker (BaseVaspMaker | None):
        The VASP input generator for bulk relaxation,
        default is DoubleRelaxMaker using TightRelaxMaker.
    phonon_displacement_maker (BaseVaspMaker | None):
        The VASP input generator for phonon displacement calculations,
        default is PhononDisplacementMaker.
    min_length (float):
        Minimum length of supercell lattice vectors in Angstroms, default is 13.0.
    prefer_90_degrees (bool):
        Whether to prefer 90 degree angles in supercell matrix,
        default is True.
    supercell_matrix_kwargs (dict):
        Keyword arguments for supercell matrix calculation, default is {}.
    IMAGINARY_TOL (float):
        Imaginary frequency tolerance in THz, default is 0.025.
    MESH_DENSITY (float):
        Mesh density for phonon calculations, default is 100.0.
    T_QHA (list):
        Temperatures for phonopy thermodynamic calculations,
        default is [0, 100, 200, ..., 2000].
    T_RENORM (list):
        Temperatures for renormalization calculations, default is [1500].
    T_KLAT (int):
        Temperature for lattice thermal conductivity calculation, default is 300.
    T_THERMAL_CONDUCTIVITY (list):
        Temperatures for thermal conductivity calculations,
        default is [0, 100, 200, 300].
    FIT_METHOD (str):
        Method for fitting force constants, default is "rfe".
    RENORM_METHOD (str):
        Method for renormalization, default is 'pseudoinverse'.
    RENORM_NCONFIG (int):
        Number of configurations for renormalization, default is 5.
    RENORM_CONV_THRESH (float):
        Convergence threshold for renormalization in meV/atom, default is 0.1.
    RENORM_MAX_ITER (int):
        Maximum iterations for renormalization, default is 30.
    SHENGBTE_CMD (str):
        Command for executing ShengBTE,
        default is "srun -n 32 ./ShengBTE 2>BTE.err >BTE.out".
    PHONO3PY_CMD (str):
        Command for executing Phono3py, default is
        "phono3py --mesh 19 19 19 --fc3 --fc2 --br --dim 5 5 5".
    """

    name: str = "Lattice-Dynamics"
    task_document_kwargs: dict = field(
        default_factory=lambda: {"task_label": "dummy_label"}
    )
    static_energy_maker: ForceFieldStaticMaker | None = field(
        default_factory=CHGNetStaticMaker
    )
    bulk_relax_maker: ForceFieldRelaxMaker = field(
        default_factory=lambda: CHGNetRelaxMaker(relax_kwargs={"fmax": 0.00001})
    )
    phonon_displacement_maker: ForceFieldStaticMaker = field(
        default_factory=CHGNetStaticMaker
    )
    min_length: float | None = 13.0
    prefer_90_degrees: bool = True
    supercell_matrix_kwargs: dict = field(default_factory=dict)
    IMAGINARY_TOL = 0.025  # in THz
    MESH_DENSITY = 100.0  # should always be a float
    T_QHA: ClassVar[list[int]] = [
        i * 100 for i in range(21)
    ]  # Temp. for phonopy calc. of thermo. properties (free energy etc.)
    T_RENORM: ClassVar[list[int]] = [
        400 # 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
    ]  # [i*100 for i in range(0,16)] # Temp. at which renorm. is to be performed
    # Temperature at which lattice thermal conductivity is calculated
    # If renormalization is performed,
    # T_RENORM overrides T_KLAT for lattice thermal conductivity
    T_KLAT: ClassVar[list[int]] = [100, 200, 300, 400]  # [i*100 for i in range(0,11)]
    T_THERMAL_CONDUCTIVITY: ClassVar[list[int]] = [
        0,
        100,
        200,
        300,
    ]  # [i*100 for i in range(0,16)]
    FIT_METHOD = "rfe"
    RENORM_METHOD = "least_squares"
    RENORM_NCONFIG = 5  # Changed from 50
    RENORM_CONV_THRESH = 0.1  # meV/atom
    RENORM_MAX_ITER = 30  # Changed from 20
    # SHENGBTE_CMD = "srun -n 16 -c 32 --cpu_bind=cores -G 16 --gpu-bind=none /global/homes/h/hrushi99/code/FourPhonon/ShengBTE"
    PHONO3PY_CMD = "phono3py --mesh 19 19 19 --fc3 --fc2 --br --dim 5 5 5"

    @property
    def prev_calc_dir_argname(self) -> None:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
        return

