"""Jobs for calculating harmonic & anharmonic props of phonon using hiPhive."""

# Basic Python packages
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

# Jobflow packages
from jobflow import job
from monty.serialization import dumpfn

# Phonopy & Phono3py
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.transformations.standard_transformations import SupercellTransformation

from atomate2.common.schemas.hiphive import ForceConstants, PhononBSDOSDoc
from atomate2.utils.log import initialize_logger

warnings.filterwarnings("ignore", module="pymatgen")
warnings.filterwarnings("ignore", module="ase")

if TYPE_CHECKING:
    from emmet.core.math import Matrix3D


logger = initialize_logger(level=3)

IMAGINARY_TOL = 0.1  # in THz # changed from 0.025
FIT_METHOD = "rfe"

ev2j = sp.constants.elementary_charge
hbar = sp.constants.hbar  # J-s
kb = sp.constants.Boltzmann  # J/K

__all__ = [
    "generate_phonon_displacements",
    "generate_frequencies_eigenvectors",
]

__author__ = "Alex Ganose, Junsoo Park, Zhuoying Zhu, Hrushikesh Sahasrabuddhe"
__email__ = "aganose@lbl.gov, jsyony37@lbl.gov, zyzhu@lbl.gov, hpsahasrabuddhe@lbl.gov"


@job(data=[Structure])
def generate_phonon_displacements(
    structure: Structure,
    supercell_matrix: np.array,
    fixed_displs: list[float],
    nconfigs: int = 1,
) -> list[Structure]:
    """
    Generate displaced structures with phonopy.

    Parameters
    ----------
    structure: Structure object
        Fully optimized input structure for phonon run
    supercell_matrix: np.array
        array to describe supercell matrix
    displacement: float
        displacement in Angstrom
    sym_reduce: bool
        if True, symmetry will be used to generate displacements
    symprec: float
        precision to determine symmetry
    use_symmetrized_structure: str or None
        primitive, conventional or None
    kpath_scheme: str
        scheme to generate kpath
    code:
        code to perform the computations
    """
    logger.info(f"supercell_matrix = {supercell_matrix}")
    supercell_structure = SupercellTransformation(
        scaling_matrix=supercell_matrix
    ).apply_transformation(structure)
    logger.info(f"supercell_structure = {supercell_structure}")
    structure_data = {
        "structure": structure,
        "supercell_structure": supercell_structure,
        "supercell_matrix": supercell_matrix,
    }

    dumpfn(structure_data, "structure_data.json")

    rattled_structures = [
        supercell_structure for _ in range(nconfigs * len(fixed_displs))
    ]

    atoms_list = []
    for idx, rattled_structure in enumerate(rattled_structures):
        logger.info(f"iter number = {idx}")
        atoms = AseAtomsAdaptor.get_atoms(rattled_structure)
        atoms_tmp = atoms.copy()

        # number of unique displs
        n_displs = len(fixed_displs)

        # map the index to distance from fixed_displs list
        distance_mapping = {i: fixed_displs[i] for i in range(n_displs)}

        # Calculate the distance based on the index pattern
        distance = distance_mapping[idx % n_displs]
        logger.info(f"distance_{idx % n_displs} = {distance}")

        # set the random seed for reproducibility
        # 6 is the number of fixed_displs
        rng = np.random.default_rng(seed=(idx // n_displs))  # idx // 6 % 6

        total_inds = list(enumerate(atoms))
        logger.info(f"total_inds = {total_inds}")
        logger.info(f"len(total_inds) = {len(total_inds)}")

        def generate_normal_displacement(
            distance: float, n: int, rng: np.random.Generator
        ) -> np.ndarray:
            directions = rng.normal(size=(n, 3))
            normalizer = np.linalg.norm(directions, axis=1, keepdims=True)
            distance_normal_distribution = rng.normal(
                distance, distance / 5, size=(n, 1)
            )
            displacements = distance_normal_distribution * directions / normalizer
            logger.info(f"displacements = {displacements}")
            return displacements

        # Generate displacements
        disp_normal = generate_normal_displacement(
            distance, len(total_inds), rng
        )  # Uncomment this later
        mean_displacements = np.linalg.norm(
            disp_normal, axis=1
        ).mean()  # Uncomment this later

        logger.info(f"mean_displacements = {mean_displacements}")

        atoms_tmp = atoms.copy()

        # add the disp_normal to the all the atoms in the structure
        for i in range(len(total_inds)):
            atoms_tmp.positions[i] += disp_normal[i]

        atoms_list.append(atoms_tmp)

    # Convert back to pymatgen structure
    structures_pymatgen = []
    for atoms_ase in range(len(atoms_list)):
        logger.info(f"atoms: {atoms_ase}")
        logger.info(f"structures_ase_all[atoms]: {atoms_list[atoms_ase]}")
        structure_i = AseAtomsAdaptor.get_structure(atoms_list[atoms_ase])
        structures_pymatgen.append(structure_i)

    for i in range(len(structures_pymatgen)):
        structures_pymatgen[i].to(f"POSCAR_{i}", "poscar")

    dumpfn(structures_pymatgen, "perturbed_structures.json")

    return structures_pymatgen


@job(
    output_schema=PhononBSDOSDoc,
    data=[PhononDos, PhononBandStructureSymmLine, ForceConstants],
)
def generate_frequencies_eigenvectors(
    structure: Structure,
    supercell_matrix: np.array,
    displacement: float,
    sym_reduce: bool,
    symprec: float,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    code: str,
    displacement_data: dict[str, list],
    total_dft_energy: float,
    epsilon_static: Matrix3D = None,
    born: Matrix3D = None,
    bulk_modulus: float = None,
    disp_cut: float = 0.05,
    **kwargs,
) -> PhononBSDOSDoc:
    """
    Analyze the phonon runs and summarize the results.

    Parameters
    ----------
    structure: Structure object
        Fully optimized structure used for phonon runs
    supercell_matrix: np.array
        array to describe supercell
    displacement: float
        displacement in Angstrom used for supercell computation
    sym_reduce: bool
        if True, symmetry will be used in phonopy
    symprec: float
        precision to determine symmetry
    use_symmetrized_structure: str
        primitive, conventional, None are allowed
    kpath_scheme: str
        kpath scheme for phonon band structure computation
    code: str
        code to run computations
    displacement_data: dict
        outputs from displacements
    total_dft_energy: float
        total DFT energy in eV per cell
    epsilon_static: Matrix3D
        The high-frequency dielectric constant
    born: Matrix3D
        Born charges
    bulk_modulus: float
        Bulk modulus in GPa
    disp_cut: float
        Displacement cut-off in Angstrom
    kwargs: dict
        Additional parameters that are passed to PhononBSDOSDoc.from_forces_born
    """
    logger.info("Starting generate_frequencies_eigenvectors()")
    return PhononBSDOSDoc.from_forces_born(
        structure=structure.remove_site_property(property_name="magmom")
        if "magmom" in structure.site_properties
        else structure,
        supercell_matrix=supercell_matrix,
        displacement=displacement,
        sym_reduce=sym_reduce,
        symprec=symprec,
        use_symmetrized_structure=use_symmetrized_structure,
        kpath_scheme=kpath_scheme,
        code=code,
        displacement_data=displacement_data,
        total_dft_energy=total_dft_energy,
        epsilon_static=epsilon_static,
        born=born,
        bulk_modulus=bulk_modulus,
        disp_cut=disp_cut,
        **kwargs,
    )
