"""Jobs for calculating harmonic & anharmonic props of phonon using hiPhive."""

# Basic Python packages
# Joblib parallelization
# ASE packages
from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy as sp

# Hiphive packages
from hiphive import ForceConstants

# Jobflow packages
from jobflow import Response, job

# Pymatgen packages
from monty.serialization import dumpfn

# Phonopy & Phono3py
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.transformations.standard_transformations import SupercellTransformation

from atomate2.common.jobs.phonons import get_supercell_size, run_phonon_displacements
from atomate2.common.schemas.hiphive import ForceConstants, PhononBSDOSDoc
from atomate2.utils.log import initialize_logger

warnings.filterwarnings("ignore", module="pymatgen")
warnings.filterwarnings("ignore", module="ase")

if TYPE_CHECKING:
    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

    from atomate2.vasp.jobs.base import BaseVaspMaker

logger = initialize_logger(level=3)

T_QHA = [
    i * 100 for i in range(31)
]  # Temp. for straight-up phonopy calculation of thermo. props. (free energy etc.)
# Temperature at which lattice thermal conductivity is calculated
# If renorm. is performed, T_RENORM overrides T_KLAT for lattice thermal conductivity
# T_KLAT = [300]  # [i*100 for i in range(0,11)]
T_KLAT = {"min":100,"max":1000,"step":100} #[i*100 for i in range(0,11)]
T_THERMAL_CONDUCTIVITY = [0, 100, 200, 300]  # [i*100 for i in range(0,16)]
IMAGINARY_TOL = 0.1  # in THz # changed from 0.025
FIT_METHOD = "rfe"

ev2j = sp.constants.elementary_charge
hbar = sp.constants.hbar # J-s
kb = sp.constants.Boltzmann # J/K

__all__ = [
    "hiphive_static_calcs",
    "run_hiphive",
    "run_thermal_cond_solver",
    "run_fc_to_pdos",
    "run_hiphive_renormalization",
    "run_lattice_thermal_conductivity",
]

__author__ = "Alex Ganose, Junsoo Park, Zhuoying Zhu, Hrushikesh Sahasrabuddhe"
__email__ = "aganose@lbl.gov, jsyony37@lbl.gov, zyzhu@lbl.gov, hpsahasrabuddhe@lbl.gov"

@job
def hiphive_static_calcs(
                structure: Structure,
                supercell_matrix: list[list[int]] | None = None,
                min_length: float | None = None,
                prefer_90_degrees: bool = True,
                n_structures: int | None = None,
                fixed_displs: list[float] = None,
                prev_dir: str | None = None,
                phonon_displacement_maker: BaseVaspMaker | None = None,
                supercell_matrix_kwargs: dict[str, Any] | None = None,
                mpid: str | None = None,
        ) -> Response:
    """Run the static calculations for hiPhive fitting."""
    if fixed_displs is None:
        fixed_displs = [0.01, 0.03, 0.08, 0.1]
        logger.info(f"fixed_displs inside hiphive_static_calcs = {fixed_displs}")

    jobs = []
    outputs: dict[str, list] = {
        "supercell_matrix": [],
        "forces": [],
        "structure": [],
        "all_structures": [],
        "all_forces": [],
        "structure_data": [],
        "current_dir": [],
    }

    if supercell_matrix is None:
        supercell_job = get_supercell_size(
            structure,
            min_length,
            prefer_90_degrees,
            **supercell_matrix_kwargs,
        )

        jobs.append(supercell_job)
        supercell_matrix = supercell_job.output
        outputs["supercell_matrix"] = supercell_job.output

    displacement_job = get_displaced_structures(
        structure=structure,
        supercell_matrix=supercell_matrix,
        fixed_displs=fixed_displs,
        mpid=mpid,
        nconfigs=n_structures
    )
    jobs.append(displacement_job)

    static_calcs_job = run_phonon_displacements(
            displacements=displacement_job.output,
            structure=structure,
            supercell_matrix=supercell_matrix,
            phonon_maker=phonon_displacement_maker,
            prev_dir=prev_dir,
        )
    jobs.append(static_calcs_job)
    outputs["forces"] = static_calcs_job.output["forces"]
    outputs["structure"] = static_calcs_job.output["structure"]

    dump_static_calc_job = get_static_results(
            structure=structure,
            supercell_matrix=supercell_matrix,
            perturbed_structures=outputs["structure"],
            perturbed_forces=outputs["forces"],
        )
    jobs.append(dump_static_calc_job)
    outputs["perturbed_structures"] = dump_static_calc_job.output[0]
    outputs["perturbed_forces"] = dump_static_calc_job.output[1]
    outputs["structure_data"] = dump_static_calc_job.output[2]
    outputs["current_dir"] = dump_static_calc_job.output[3]

    return Response(replace=jobs, output=outputs)

@job(data=[Structure])
def generate_phonon_displacements(
    structure: Structure,
    supercell_matrix: np.array,
    fixed_displs: list[float],
    sym_reduce: bool,
    symprec: float,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    code: str,
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

    rattled_structures = [supercell_structure for _ in range(nconfigs*len(fixed_displs))]

    atoms_list = []
    for idx, structure in enumerate(rattled_structures):
        logger.info(f"iter number = {idx}")
        atoms = AseAtomsAdaptor.get_atoms(structure)
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
        rng = np.random.default_rng(seed=(idx // n_displs)) # idx // 6 % 6

        total_inds = list(enumerate(atoms))
        logger.info(f"total_inds = {total_inds}")
        logger.info(f"len(total_inds) = {len(total_inds)}")
        # if you want to select specific species instead, then use the following code
        # Li_inds = [i for i, a in enumerate(atoms) if a.symbol == 'Li']

        # Uncomment this later
        def generate_normal_displacement(
                distance: float, n: int, rng: np.random.Generator
            ) -> np.ndarray:
            directions = rng.normal(size=(n, 3))
            normalizer = np.linalg.norm(directions, axis=1, keepdims=True)
            distance_normal_distribution = rng.normal(
                distance, distance/5, size=(n, 1)
            )
            displacements = distance_normal_distribution * directions / normalizer
            logger.info(f"displacements = {displacements}")
            return displacements

        # Generate displacements
        disp_normal = generate_normal_displacement(distance, len(total_inds), rng) # Uncomment this later
        mean_displacements = np.linalg.norm(disp_normal, axis=1).mean() # Uncomment this later

        logger.info(f"mean_displacements = {mean_displacements}")

        atoms_tmp = atoms.copy()


        # add the disp_normal to the all the atoms in the structrue
        for i in range(len(total_inds)): # uncomment this later to perturb all the atoms
            atoms_tmp.positions[i] += disp_normal[i] # Uncomment this later

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


@job
def get_displaced_structures(
    structure: Structure,
    supercell_matrix: list[list[int]] | None = None,
    fixed_displs: list[float] | None = None,
    mpid: str | None = None,
    nconfigs: int = 1,
) -> list[Structure]:

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

    rattled_structures = [supercell_structure for _ in range(nconfigs*len(fixed_displs))]

    atoms_list = []
    for idx, structure in enumerate(rattled_structures):
        logger.info(f"iter number = {idx}")
        atoms = AseAtomsAdaptor.get_atoms(structure)
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
        rng = np.random.default_rng(seed=(idx // n_displs)) # idx // 6 % 6

        total_inds = list(enumerate(atoms))
        logger.info(f"total_inds = {total_inds}")
        logger.info(f"len(total_inds) = {len(total_inds)}")
        # if you want to select specific species instead, then use the following code
        # Li_inds = [i for i, a in enumerate(atoms) if a.symbol == 'Li']

        # Uncomment this later
        def generate_normal_displacement(
                distance: float, n: int, rng: np.random.Generator
            ) -> np.ndarray:
            directions = rng.normal(size=(n, 3))
            normalizer = np.linalg.norm(directions, axis=1, keepdims=True)
            distance_normal_distribution = rng.normal(
                distance, distance/5, size=(n, 1)
            )
            displacements = distance_normal_distribution * directions / normalizer
            logger.info(f"displacements = {displacements}")
            return displacements

        # Generate displacements
        disp_normal = generate_normal_displacement(distance, len(total_inds), rng) # Uncomment this later
        mean_displacements = np.linalg.norm(disp_normal, axis=1).mean() # Uncomment this later

        logger.info(f"mean_displacements = {mean_displacements}")

        atoms_tmp = atoms.copy()


        # add the disp_normal to the all the atoms in the structrue
        for i in range(len(total_inds)): # uncomment this later to perturb all the atoms
            atoms_tmp.positions[i] += disp_normal[i] # Uncomment this later

        atoms_list.append(atoms_tmp)

    # Convert back to pymatgen structure
    structures_pymatgen = []
    for atoms_ase in range(len(atoms_list)):
        logger.info(f"atoms: {atoms_ase}")
        logger.info(f"structures_ase_all[atoms]: {atoms_list[atoms_ase]}")
        structure_i = AseAtomsAdaptor.get_structure(atoms_list[atoms_ase])
        structures_pymatgen.append(structure_i)

    dumpfn(structures_pymatgen, f"perturbed_structures_{mpid}.json")

    return structures_pymatgen

@job
def get_static_results(
    structure: Structure,
    supercell_matrix: list[list[int]] | None = None,
    perturbed_structures: list[Structure] | None = None,
    perturbed_forces: Any | None = None,
) -> list:
    """
    Aggregate the structures and forces of perturbed supercells.

    Args:
        structure (Structure): input structure
        supercell (Structure): supercell structure
        supercell_matrix (list[list[int]]): supercell matrix
        rattled_structures (list[Structure]): list of Structures
        forces (list[list[int]]): forces
        perturbed_tasks (int): number of perturbed tasks
    Returns:
        None.
    """
    logger.info(f"scaling_matrix = {supercell_matrix}")
    logger.info(f"structure = {structure}")
    supercell = SupercellTransformation(
        scaling_matrix=supercell_matrix
    ).apply_transformation(structure)
    structure_data = {
        "structure": structure,
        "supercell_structure": supercell,
        "supercell_matrix": supercell_matrix,
    }

    dumpfn(perturbed_structures, "perturbed_structures.json")
    dumpfn(perturbed_forces, "perturbed_forces.json")
    dumpfn(structure_data, "structure_data.json")
    dumpfn(structure, "relaxed_structure.json")

    # Convert list of lists to numpy arrayx
    perturbed_forces = np.array(perturbed_forces)

    perturbed_forces_new = [
        {
            "@module": "numpy",
            "@class": "array",
            "dtype": str(perturbed_forces.dtype),
            "data": sublist.tolist(),
        }
        for sublist in perturbed_forces
    ]

    # Save the data as a JSON file
    dumpfn(perturbed_forces_new, "perturbed_forces_new.json")

    current_dir = os.getcwd()

    return [perturbed_structures, perturbed_forces, structure_data, current_dir]

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
        **kwargs,
    )