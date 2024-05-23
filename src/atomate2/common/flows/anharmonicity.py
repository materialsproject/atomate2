"""A workflow to evaluate the anharmonicity of a material with sigma^A.

For details see: doi.org/10.1103/PhysRevMaterials.4.083809
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jobflow import Flow, Maker
from pymatgen.core.structure import Structure

from atomate2.common.jobs.anharmonicity import (
    displace_structure,
    get_phonon_supercell,
    get_sigma_a,
    run_displacements,
)

if TYPE_CHECKING:
    from pathlib import Path

    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

    from atomate2.common.flows.phonons import BasePhononMaker

SUPPORTED_CODES = ["vasp", "aims", "forcefields"]


@dataclass
class BaseAnharmonicityMaker(Maker):
    """
    Maker to calculate the anharmonicity score of a material.

    Calculate sigma^A as defined in doi.org/10.1103/PhysRevMaterials.4.083809, by
    first calculating the phonons for a material and then generating the one-shot
    sample and calculating the DFT and harmonic forces.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    phonon_maker: BasePhononMaker
        The maker to generate the phonon model
    """

    name: str = "anharmonicity"
    phonon_maker: BasePhononMaker = None

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
        born: list[Matrix3D] | None = None,
        epsilon_static: Matrix3D | None = None,
        total_dft_energy_per_formula_unit: float | None = None,
        supercell_matrix: Matrix3D | None = None,
        temperature: float = 300,
    ) -> Flow:
        """Make the anharmonicity calculation flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object. Please start with a structure
            that is nearly fully optimized as the internal optimizers
            have very strict settings!
        prev_dir : str or Path or None
            A previous calculation directory to use for copying outputs.
        born: list[Matrix3D]
            Instead of recomputing born charges and epsilon, these values can also be
            provided manually. If born and epsilon_static are provided, the born run
            will be skipped it can be provided in the VASP convention with information
            for every atom in unit cell. Please be careful when converting structures
            within in this workflow as this could lead to errors
        epsilon_static: Matrix3D
            The high-frequency dielectric constant to use instead of recomputing born
            charges and epsilon. If born, epsilon_static are provided, the born run
            will be skipped
        total_dft_energy_per_formula_unit: float
            It has to be given per formula unit (as a result in corresponding Doc).
            Instead of recomputing the energy of the bulk structure every time, this
            value can also be provided in eV. If it is provided, the static run will be
            skipped. This energy is the typical output dft energy of the dft workflow.
            No conversion needed.
        supercell_matrix: Matrix3D | None
            Instead of min_length, also a supercell_matrix can be given, e.g.
            [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]
        temperature: float
            The temperature for the anharmonicity calculation

        Returns
        -------
        Flow
            The workflow for the anharmonicity calculations
        """
        phonon_flow = self.phonon_maker.make(
            structure,
            prev_dir,
            born,
            epsilon_static,
            total_dft_energy_per_formula_unit,
            supercell_matrix,
        )
        jobs = [phonon_flow]

        phonon_supercell_job = get_phonon_supercell(phonon_flow.output)
        jobs.append(phonon_supercell_job)

        phonon_supercell = phonon_supercell_job.output

        displace_supercell = displace_structure(
            phonon_supercell=phonon_supercell,
            force_constants=phonon_flow.output.force_constants,
            temp=temperature,
        )
        jobs.append(displace_supercell)
        self.displaced_supercell = displace_supercell.output

        displacement_calcs = run_displacements(
            displacements=[displace_supercell.output],
            phonon_supercell=phonon_supercell,
            force_eval_maker=self.phonon_maker.phonon_displacement_maker,
            socket=self.phonon_maker.socket,
            prev_dir_argname=self.prev_calc_dir_argname,
            prev_dir=prev_dir,
        )
        jobs.append(displacement_calcs)

        # Calculate oneshot approximation of sigma_A
        calc_sigma_a_os = get_sigma_a(
            phonon_flow.output.force_constants,
            phonon_supercell,
            displacement_calcs.output,
        )
        jobs.append(calc_sigma_a_os)
        self.sigma_A_oneshot = calc_sigma_a_os.output

        return Flow(jobs, calc_sigma_a_os.output)

    # Note: Came from aims/flows/phonons.py
    # (Might be different for different DFT codes, but this might work as a
    # first-pass solution)
    @property
    def prev_calc_dir_argname(self) -> str:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
        return "prev_dir"
