"""A workflow to evaluate the anharmonicity of a material with sigma^A.

For details see: doi.org/10.1103/PhysRevMaterials.4.083809
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from warnings import warn

from jobflow import Flow, Maker

from atomate2.common.jobs.anharmonicity import (
    displace_structure,
    get_forces,
    get_phonon_supercell,
    get_sigmas,
    run_displacements,
    store_results,
)

if TYPE_CHECKING:
    from pathlib import Path

    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

    from atomate2.common.flows.phonons import BasePhononMaker
    from atomate2.common.schemas.phonons import PhononBSDOSDoc

SUPPORTED_CODES = ["aims"]


@dataclass
class BaseAnharmonicityMaker(Maker, ABC):
    """
    Maker to calculate the anharmonicity score of a material.

    Calculate sigma^A as defined in doi.org/10.1103/PhysRevMaterials.4.083809, by
    first calculating the phonons for a material and then generating the one-shot
    sample and calculating the DFT and harmonic forces.

    Parameters
    ----------
    name: str
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
        one_shot_approx: bool = True,
        seed: int | None = None,
        element_resolved: bool = False,
        mode_resolved: bool = False,
        site_resolved: bool = False,
        n_samples: int = 1,
    ) -> Flow:
        """Make the anharmonicity calculation flow.

        Parameters
        ----------
        structure: Structure
            A pymatgen structure object. Please start with a structure
            that is nearly fully optimized as the internal optimizers
            have very strict settings!
        prev_dir: Optional[str | Path]
            A previous calculation directory to use for copying outputs.
            Default is None.
        born: Optional[list[Matrix3D]]
            Instead of recomputing born charges and epsilon, these values can also be
            provided manually. If born and epsilon_static are provided, the born run
            will be skipped it can be provided in the VASP convention with information
            for every atom in unit cell. Please be careful when converting structures
            within in this workflow as this could lead to errors. The default for this
            field is None.
        epsilon_static: Optional[Matrix3D]
            The high-frequency dielectric constant to use instead of recomputing born
            charges and epsilon. If born, epsilon_static are provided, the born run
            will be skipped. The default for this field is None.
        total_dft_energy_per_formula_unit: Optional[float]
            It has to be given per formula unit (as a result in corresponding Doc).
            Instead of recomputing the energy of the bulk structure every time, this
            value can also be provided in eV. If it is provided, the static run will be
            skipped. This energy is the typical output dft energy of the dft workflow.
            No conversion needed. It is set to 0 by default.
        supercell_matrix: Optional[Matrix3D]
            Instead of min_length, also a supercell_matrix can be given, e.g.
            [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]. By default, this matrix is
            set to None.
        temperature: float
            The temperature for the anharmonicity calculation. The default temp is 300.
        one_shot_approx: bool
            If true, finds the one shot approximation of sigma^A and if false, finds
            the full sigma^A. The default is True.
        seed: Optional[int]
            Seed to use for the random number generator
            (only used if one_shot_approx == False). Set to None by default.
        element_resolved: bool
            If true, calculate the atom-resolved sigma^A. This is false by default.
        mode_resolved: bool
            If true, calculate the mode-resolved sigma^A. This is false by default.
        site_resolved: bool
            If true, resolve sigma^A to the different sites.
            Default is false.
        n_samples: int
            Number of times displaced structures are sampled.
            Must be >= 1 and cannot be used when one_shot_approx == True.
            It is set to 1 by default.

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

        phonon_doc = phonon_flow.output
        anharmon_flow = self.make_from_phonon_doc(
            phonon_doc,
            prev_dir,
            temperature,
            one_shot_approx,
            seed,
            element_resolved,
            mode_resolved,
            site_resolved,
            n_samples,
        )

        results = store_results(
            sigma_dict=anharmon_flow.output,
            phonon_doc=phonon_flow.output,
            one_shot=one_shot_approx,
            temp=temperature,
            n_samples=n_samples,
            seed=seed,
        )

        jobs = [phonon_flow, anharmon_flow, results]
        return Flow(jobs, results.output)

    def make_from_phonon_doc(
        self,
        phonon_doc: PhononBSDOSDoc,
        prev_dir: str | Path | None = None,
        temperature: float = 300,
        one_shot_approx: bool = True,
        seed: int | None = None,
        element_resolved: bool = False,
        mode_resolved: bool = False,
        site_resolved: bool = False,
        n_samples: int = 1,
    ) -> Flow:
        """Create an anharmonicity workflow from a phonon calculation.

        Parameters
        ----------
        phonon_doc: PhononBSDOSDoc
            The document to get the anharmonicity for
        prev_dir: Optional[str | Path]
            A previous calculation directory to use for copying outputs.
            Default is None.
        temperature: float
            The temperature for the anharmonicity calculation. Default is 300.
        one_shot_approx: bool
            If true, finds the one shot approximation of sigma^A and if false,
            finds the full sigma^A. The default is True.
        seed: Optional[int]
            Seed to use for the random number generator
            (only used if one_shot_approx == False). Default is None.
        element_resolved: bool
            If true, calculate the atom-resolved sigma^A. This is false by default.
        mode_resolved: bool
            If true, calculate the mode-resolved sigma^A. This is false by default.
        site_resolved: bool
            If true, resolve sigma^A to the different sites.
            Default is false.
        n_samples: int
            Number of times displaced structures are sampled.
            Must be >= 1 and cannot be used when one_shot_approx == True.
            It is set to 1 by default.

        Returns
        -------
        Flow
            The anharmonicity quantification workflow
        """
        # KB: Not always an error. There could be negative acoustic modes from
        # unconverged phonon calculations.
        if phonon_doc.has_imaginary_modes:
            warn(
                "The phonon model has imaginary modes, sampling maybe incorrect.",
                stacklevel=1,
            )

        jobs = []
        phonon_supercell_job = get_phonon_supercell(
            phonon_doc.structure,
            phonon_doc.supercell_matrix,
        )
        jobs.append(phonon_supercell_job)

        phonon_supercell = phonon_supercell_job.output

        displace_supercell = displace_structure(
            phonon_supercell=phonon_supercell,
            force_constants=phonon_doc.force_constants,
            temp=temperature,
            one_shot=one_shot_approx,
            seed=seed,
            n_samples=n_samples,
        )
        jobs.append(displace_supercell)

        force_eval_maker = self.phonon_maker.phonon_displacement_maker
        displacement_calcs = run_displacements(
            displacements=displace_supercell.output,
            phonon_supercell=phonon_supercell,
            force_eval_maker=force_eval_maker,
            socket=self.phonon_maker.socket,
            prev_dir_argname=self.prev_calc_dir_argname,
            prev_dir=prev_dir,
        )
        jobs.append(displacement_calcs)

        # Get DFT and harmonic forces
        force_calcs = get_forces(
            phonon_doc.force_constants,
            phonon_supercell,
            displacement_calcs.output,
        )
        jobs.append(force_calcs)

        # Calculate all desired sigma^A types
        sigma_calcs = get_sigmas(
            force_calcs.output[0],
            force_calcs.output[1],
            phonon_doc.force_constants,
            phonon_supercell,
            one_shot_approx,
            element_resolved,
            mode_resolved,
            site_resolved,
        )
        jobs.append(sigma_calcs)
        sigma_a_vals = sigma_calcs.output

        return Flow(jobs, sigma_a_vals)

    @property
    @abstractmethod
    def prev_calc_dir_argname(self) -> str | None:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
