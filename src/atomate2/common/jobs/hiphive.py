"""Jobs for calculating harmonic & anharmonic props of phonon using hiPhive."""

# Basic Python packages
# Joblib parallelization
# ASE packages
from __future__ import annotations

import contextlib
import json
import os
import shlex
import subprocess
import warnings
from copy import copy
from itertools import product
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import phonopy as phpy
import psutil
import scipy as sp
from ase.cell import Cell

# Hiphive packages
from hiphive import (
    ClusterSpace,
    ForceConstantPotential,
    ForceConstants,
    StructureContainer,
    enforce_rotational_sum_rules,
)
from hiphive.cutoffs import estimate_maximum_cutoff, is_cutoff_allowed
from hiphive.fitting import Optimizer
from hiphive.renormalization import Renormalization
from hiphive.run_tools import free_energy_correction
from hiphive.utilities import get_displacements

# Jobflow packages
from jobflow import Flow, Response, job

# Pymatgen packages
from monty.serialization import dumpfn, loadfn
from phono3py.phonon3.gruneisen import Gruneisen

# Phonopy & Phono3py
from phonopy import Phonopy
from phonopy.interface.hiphive_interface import phonopy_atoms_to_ase
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import (
    get_phonon_band_structure_from_fc,
    get_phonon_band_structure_symm_line_from_fc,
    get_phonon_dos_from_fc,
    get_phonopy_structure,
)
from pymatgen.io.shengbte import Control
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import SupercellTransformation

from atomate2.common.jobs.phonons import get_supercell_size, run_phonon_displacements
from atomate2.forcefields.md import MACEMDMaker
from atomate2.settings import Atomate2Settings
from atomate2.utils.log import initialize_logger
from atomate2.vasp.files import copy_hiphive_outputs

warnings.filterwarnings("ignore", module="pymatgen")
warnings.filterwarnings("ignore", module="ase")

if TYPE_CHECKING:
    from ase.atoms import Atoms
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
        # fixed_displs = [0.01, 0.03, 0.08, 0.1, 0.15, 0.16]
        fixed_displs = [0.01, 0.03, 0.08, 0.1]
        # fixed_displs = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21] # 11 displ
        # fixed_displs = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11] # 6 displ
        # fixed_displs = [0.01, 0.015, 0.02, 0.025, 0.03, 0.045, 0.051, 0.07, 0.09, 0.11] # 10 displ
        # fixed_displs = [0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] # 10 displ V2
        # fixed_displs = [0.01, 0.02, 0.03, 0.04, 0.049, 0.051, 0.06, 0.07, 0.08, 0.09, 0.1] # 11 displ V2
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

    # displacement_job_md = get_displaced_structures_md(
    #     structure=structure,
    #     supercell_matrix=supercell_matrix,
    #     temp=[1, 3, 5, 40, 50, 60],
    #     mpid=mpid,
    #     nconfigs=n_structures
    # )
    # jobs.append(displacement_job_md)

    static_calcs_job = run_phonon_displacements(
            displacements=displacement_job.output,
            # displacements=displacement_job_md.output["structure"],
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

@job
def get_displaced_structures_md(
    structure: Structure,
    supercell_matrix: list[list[int]] | None = None,
    temp: list[int] | None = None,
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

    phonon_md_jobs = []
    outputs_md: dict[str, list] = {
        "forces": [],
        "structure": []
    }
    for t in temp:
        mace_md_job = MACEMDMaker(
            n_steps=25*nconfigs,
            temperature=t,
            traj_file="mace.traj",
            traj_interval=10,
            task_document_kwargs={"store_trajectory": "full"},
        ).make(
            structure=supercell_structure,
        )
        phonon_md_jobs.append(mace_md_job)
        for i in range(nconfigs):
            outputs_md["forces"].append(mace_md_job.output.output.ionic_steps[-10*(i) - 1].forces)
            outputs_md["structure"].append(mace_md_job.output.output.ionic_steps[-10*(i) - 1].structure)

    displacement_md_flow = Flow(phonon_md_jobs, outputs_md)
    return Response(replace=displacement_md_flow)

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

    rattled_structures = [supercell_structure for _ in range(nconfigs*4)]

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
        disp_normal = generate_normal_displacement(distance, len(total_inds), rng)
        mean_displacements = np.linalg.norm(disp_normal, axis=1).mean()
        logger.info(f"mean_displacements = {mean_displacements}")

        atoms_tmp = atoms.copy()

        # Na_inds = [i for i, a in enumerate(atoms_tmp) if a.symbol == 'Na']
        # Zr_inds = [i for i, a in enumerate(atoms_tmp) if a.symbol == 'Zr']
        # P_inds = [i for i, a in enumerate(atoms_tmp) if a.symbol == 'P']
        # O_inds = [i for i, a in enumerate(atoms_tmp) if a.symbol == 'O']

        # amp_Na = 1
        # amp_Zr = 1
        # amp_P = 1
        # amp_O = 1


        # add the disp_normal to the all the atoms in the structrue
        for i in range(len(total_inds)):
            atoms_tmp.positions[i] += disp_normal[i]
            # atoms_tmp.positions[Na_inds] += amp * disp_normal[i]

        # # if you want to add the displacements to specific species, then use the
        # # following code
        # # if displ_number is 0, 1, 2, 3 -- then add the disp_normal to the Na atoms
        # if idx == 0 or idx == 1 or idx == 2 or idx == 3:
        #     # similar to atoms_normal, but only for Li atoms, add the disp_normal to
        #     # the Na atoms.
        #     for i in Li_inds:
        #         atoms_tmp.positions[i] += disp_normal[i]

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

@job
def run_hiphive(
    cutoffs: list[list] | None = None,
    fit_method: str | None = None,
    disp_cut: float | None = None,
    bulk_modulus: float | None = None,
    temperature_qha: float | None = None,
    imaginary_tol: float | None = None,
    prev_dir_json_saver: str | None = None,
) -> list:
    """
    Fit force constants using hiPhive.

    Requires "perturbed_structures.json", "perturbed_forces.json", and
    "structure_data.json" files to be present in the current working directory.

    Args:
        cutoffs (Optional[list[list]]): A list of cutoffs to trial. If None,
            a set of trial cutoffs will be generated based on the structure
            (default).
        separate_fit: If True, harmonic and anharmonic force constants are fit
            separately and sequentially, harmonic first then anharmonic. If
            False, then they are all fit in one go. Default is False.
        disp_cut: if separate_fit=True, determines the mean displacement of perturbed
            structure to be included in harmonic (<) or anharmonic (>) fitting
        imaginary_tol (float): Tolerance used to decide if a phonon mode
            is imaginary, in THz.
        fit_method (str): Method used for fitting force constants. This can
            be any of the values allowed by the hiphive ``Optimizer`` class.
    """
    logger.info(f"cutoffs = {cutoffs}")
    logger.info(f"disp_cut is {disp_cut}")
    logger.info(f"fit_method is {fit_method}")

    copy_hiphive_outputs(prev_dir_json_saver)

    perturbed_structures = loadfn("perturbed_structures.json")
    perturbed_forces = loadfn("perturbed_forces_new.json")
    structure_data = loadfn("structure_data.json")

    parent_structure = structure_data["structure"]
    supercell_structure = structure_data["supercell_structure"]
    supercell_matrix = np.array(structure_data["supercell_matrix"])

    if cutoffs is None:
        # cutoffs = get_cutoffs(supercell_structure)
        # cutoffs = [[9.599999999999996, 6.7771008403361375, 	3.931250000000001]]
        # cutoffs = [[9.131249999999996, 7.9140625000000036, 5.231250000000002]]
        # cutoffs = [[12, 9.5, 5.5]]
        # cutoffs = [[4.3, 4.2, 3.6]]
        # cutoffs = [[4.27, 4.214062500000001, 3.2687500000000003], [4.27, 4.214062500000001, 3.601041666666667],
        #            [4.27, 4.214062500000001, 4.151500000000001]]
        # cutoffs = [[4.3, 4.2, 3.3], [4.3, 4.2, 3.6],
        #            [4.3, 4.2, 4.2]]
        # cutoffs = [[4.6, 5.3, 4.2]]
        # cutoffs = [[4.6, 4, 3]]
        # cutoffs = [[3.8, 4.1, 3.0]]
        # cutoffs = [[5.8, 4.8, 3.8]]
        # cutoffs = [[6, 4.8, 4], [7, 4.8, 4], [8, 4.8, 4], [9, 4.8, 4], [10, 4.8, 4]]
        # cutoffs = [[6, 4.8, 2.5], [7, 4.8, 2.5], [8, 4.8, 2.5], [9, 4.8, 2.5], [10, 4.8, 2.5]]
        # cutoffs = [[6.4, 5.5, 4]] # BaO2
        # cutoffs = [[6.5, 5.1, 4]] # SrO2
        # cutoffs = [[8, 5.7, 4.7]] # TlCl
        # cutoffs = [[6.9, 5.7, 4.4]] # InBr
        # cutoffs = [[4, 5, 4.3]] # CsBr2F
        # cutoffs = [[5.6, 4.6, 4]] # CsBr2F
        # cutoffs = [[4, 4.6, 4]] # CsBr2F
        # cutoffs = [[7, 4.6, 4]] # CsBr2F
        # cutoffs = [[5.2, 4.8, 4.1]] # AgCl
        # cutoffs = [[10, 8, 6]] # AgCl
        # cutoffs = [[4, 4, 3.5]] # NaBrO3 [5, 3, 2.5]
        # cutoffs = [[7, 4, 3.5]] # AuSe
        # cutoffs = [[4, 4, 3.5]] # NaAgO2 [7, 4, 3.5]
        # cutoffs = [[5, 4, 3]] # Al4C3
        # cutoffs = [[4, 4, 3]] # Al4C3
        # cutoffs = [[10, 7.5, 5]] # GeTe
        # cutoffs = [[7, 6.3, 4.4]] # Sb2Te3
        # cutoffs = [[4, 3.5, 2.5]] # NaZr2(PO4)3
        # cutoffs = [[9, 3.5, 2.5]] # NaZr2(PO4)3
        # cutoffs = [[7, 5, 4]] # CsAlO2
        # cutoffs = [[10, 4.6, 3.3]] # Ga2Te5
        # cutoffs = [[10, 6.65, 4.1]] # Bi2Se3
        # cutoffs = [[8, 4, 3]] # Bi2Se3
        # cutoffs = [[11, 5.7, 3.6]] # BP [[8, 6.3, 4.5]] [[8.6, 5.8, 3.8]]
        # cutoffs = [[7, 5.0625, 3.95]] # NaCl
        # cutoffs = [[10, 7.3, 4.5]] # GaAs [[10, 5.5, 4]] [[10, 7.3, 4.5]]
        # cutoffs = [[10, 7, 4]] # BaO
        # cutoffs = [[11.5, 7.625, 5.9]] # Ba
        # cutoffs = [[8, 5.5, 4]] # GaAs [[12, 5.5, 4]]
        # cutoffs = [[11, 6.3, 4.3]] # GaP
        # cutoffs = [[7, 6, 3]] # MgO [[7.5, 6.6, 3.7]]
        # cutoffs = [[12, 7.5, 4.5]] # NaCl
        cutoffs = [[7, 3.5, 3]] # BP
        logger.info(f"cutoffs is {cutoffs}")
    else:
        pass

    t_qha = temperature_qha if temperature_qha is not None else T_QHA
    if isinstance(t_qha, list):
        t_qha.sort()
    else:
        # Handle the case where T_qha is not a list
        # You can choose to raise an exception or handle it differently
        # For example, if T_qha is a single temperature value, you can
        # convert it to a list
        t_qha = [t_qha]

    logger.info(f"t_qha is {t_qha}")

    structures = []
    logger.info(f"supercell_structure is {supercell_structure}")

    supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)
    for structure, forces in zip(perturbed_structures, perturbed_forces):
        logger.info(f"structure is {structure}")
        atoms = AseAtomsAdaptor.get_atoms(structure)
        displacements = get_displacements(atoms, supercell_atoms)
        atoms.new_array("displacements", displacements)
        atoms.new_array("forces", forces)
        atoms.positions = supercell_atoms.get_positions()
        structures.append(atoms)

        # Calculate mean displacements
        mean_displacements = np.linalg.norm(displacements, axis=1).mean()
        logger.info(f"Mean displacements while reading individual displacements: "
                    f"{mean_displacements}")
        # Calculate standard deviation of displacements
        std_displacements = np.linalg.norm(displacements, axis=1).std()
        logger.info("Standard deviation of displacements while reading individual"
                    "displacements: "
                    f"{std_displacements}")

    all_cutoffs = cutoffs
    logger.info(f"all_cutoffs is {all_cutoffs}")

    fcs, param, cs, fitting_data, fcp = fit_force_constants(
        parent_structure=parent_structure,
        supercell_matrix=supercell_matrix,
        supercell_structure=supercell_structure,
        structures=structures,
        all_cutoffs=all_cutoffs,
        disp_cut=disp_cut,
        imaginary_tol=imaginary_tol,
        fit_method=fit_method,
    )

    logger.info("Saving Harmonic props")
    thermal_data, phonopy = harmonic_properties(
        parent_structure, supercell_matrix, fcs, t_qha, imaginary_tol
    )

    anharmonic_data = anharmonic_properties(
        phonopy,
        fcs,
        t_qha,
        thermal_data["heat_capacity"],
        thermal_data["n_imaginary"],
        bulk_modulus,
    )

    if fcs is None:
        raise RuntimeError("Could not find a force constant solution")

    if isinstance(fcs, ForceConstants):
        logger.info("Writing force_constants")
        fcs.write("force_constants.fcs")
    else:
        logger.info("fcs is not an instance of ForceConstants")

    if isinstance(fcp, ForceConstantPotential):
        logger.info("Writing force_constants_potential")
        fcp.write("force_constants_potential.fcp")

    logger.info("Saving parameters")
    np.savetxt("parameters.txt", param)

    if isinstance(cs, ClusterSpace):
        logger.info("Writing cluster_space")
        cs.write("cluster_space.cs")
        logger.info("cluster_space writing is complete")
    else:
        logger.info("cs is not an instance of ClusterSpace")

    logger.info("Saving phonopy_params")
    phonopy.save("phonopy_params.yaml")
    fitting_data["best_n_imaginary"] = thermal_data.pop("n_imaginary")
    thermal_data.update(anharmonic_data)
    logger.info("Saving fitting_data")
    dumpfn(fitting_data, "fitting_data.json")
    logger.info("Saving thermal_data")
    dumpfn(thermal_data, "thermal_data.json")

    logger.info("Writing cluster space and force_constants")
    logger.info(f"{type(fcs)}")

    logger.info(f"phonopy_atoms_to_ase(phonopy.supercell) = {phonopy_atoms_to_ase(phonopy.supercell)}")
    logger.info(f"supercell_atoms = {supercell_atoms}")
    from pymatgen.io.ase import MSONAtoms
    structure_data_phonopy_atoms = {
        "supercell_structure_phonopy_atoms": MSONAtoms(phonopy_atoms_to_ase(phonopy.supercell)),
    }
    structure_data_pymatgen_atoms = {
        "supercell_structure_pymatgen_atoms": supercell_atoms,
    }

    dumpfn(structure_data_phonopy_atoms, "structure_data_phonopy_atoms.json")
    dumpfn(structure_data_pymatgen_atoms, "structure_data_pymatgen_atoms.json")

    primitive_atoms_phonopy = phonopy_atoms_to_ase(phonopy.primitive)
    primitive_atoms_pymatgen = AseAtomsAdaptor.get_atoms(parent_structure)

    prim_structure_data_phonopy_atoms = {
        "primitive_structure_phonopy_atoms": MSONAtoms(primitive_atoms_phonopy),
    }
    prim_structure_data_pymatgen_atoms = {
        "primitive_structure_pymatgen_atoms": primitive_atoms_pymatgen,
    }
    # dumpfn primitve and supercell atoms
    dumpfn(prim_structure_data_phonopy_atoms, "prim_structure_data_phonopy_atoms.json")
    dumpfn(prim_structure_data_pymatgen_atoms, "prim_structure_data_pymatgen_atoms.json")


    primitive_structure_phonopy = AseAtomsAdaptor.get_structure(primitive_atoms_phonopy)
    primitive_structure_pymatgen = AseAtomsAdaptor.get_structure(primitive_atoms_pymatgen)

    # dumpfn primitve and supercell structures
    dumpfn(primitive_structure_phonopy, "primitive_structure_phonopy.json")
    dumpfn(primitive_structure_pymatgen, "primitive_structure_pymatgen.json")

    primitive_structure_phonopy_new = SpacegroupAnalyzer(primitive_structure_phonopy).find_primitive()
    primitive_structure_pymatgen_new = SpacegroupAnalyzer(primitive_structure_pymatgen).find_primitive()

    # dumpfn primitve and supercell structures
    dumpfn(primitive_structure_phonopy_new, "primitive_structure_phonopy_new.json")
    dumpfn(primitive_structure_pymatgen_new, "primitive_structure_pymatgen_new.json")

    if fitting_data["best_n_imaginary"] == 0:
        logger.info("No imaginary modes! Writing ShengBTE files")
        atoms = AseAtomsAdaptor.get_atoms(parent_structure)
        # fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD_1", atoms, order=3)
        fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD_from_fcs_pymatgen_struct", atoms)
        fcs.write_to_phonopy("FORCE_CONSTANTS_2ND", format="text")

        primitive_atoms_phonopy = phonopy_atoms_to_ase(phonopy.primitive)
        fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD_from_fcs_phonopy_struct", primitive_atoms_phonopy)

        ForceConstants.write_to_phonopy(fcs, "fc2.hdf5", "hdf5")
        # ForceConstants.write_to_phono3py(fcs, "fc3.hdf5", "hdf5")
        ForceConstants.write_to_phono3py(fcs, "fc3.hdf5")

        ### detour from hdf5
        supercell_atoms_phonopy = phonopy_atoms_to_ase(phonopy.supercell)
        supercell_atoms_pymatgen = AseAtomsAdaptor.get_atoms(supercell_structure)

        # check if the supercell_atoms are the same
        if supercell_atoms_phonopy == supercell_atoms_pymatgen:
            logger.info("supercell_atoms are the same")
        else:
            logger.info("supercell_atoms are different")
        supercell_atoms = supercell_atoms_phonopy
        # supercell_atoms = supercell_atoms_pymatgen
        # fcs = ForceConstants.read_phono3py(supercell_atoms, "fc3.hdf5", order=3)
        fcs = ForceConstants.read_phono3py(supercell_atoms, "fc3.hdf5")
        # fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD", atoms, order=3, fc_tol=1e-4)
        # fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD", atoms, fc_tol=1e-4)
        fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD_from_hdf5_phonopy_struct", atoms)

        supercell_atoms = supercell_atoms_pymatgen
        fcs = ForceConstants.read_phono3py(supercell_atoms, "fc3.hdf5")
        fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD_from_hdf5_pymatgen_struct", atoms)

    else:
        logger.info(f"best_n_imaginary = {fitting_data['best_n_imaginary']}")
        logger.info("ShengBTE files not written due to imaginary modes.")
        logger.info("You may want to perform phonon renormalization.")

    current_dir = os.getcwd()

    outputs: dict[str, list] = {
        "thermal_data": thermal_data,
        # "anharmonic_data": anharmonic_data,
        "fitting_data": fitting_data,
        "param": param,
        "current_dir": current_dir
    }

    return outputs


def get_cutoffs(supercell_structure: Structure) -> list[list[float]]:
    """
    Determine the trial cutoffs based on a supercell structure.

    This function calculates and returns the best cutoffs for 2nd, 3rd, and 4th order
    interactions for a given supercell structure. It performs linear interpolation to
    find cutoffs for specific target degrees of freedom (DOFs), generates combinations
    of cutoffs, and filters them to retain unique DOFs.

    Args:
        supercell_structure: A structure.

    Returns
    -------
        list[list[float]]: A list of lists where each sublist contains the best cutoffs
                           for 2nd, 3rd, and 4th order interactions.
    """
    # Initialize lists and variables
    cutoffs_2nd_list = []
    cutoffs_3rd_list = []
    cutoffs_4th_list = []
    n_doffs_2nd_list = []
    n_doffs_3rd_list = []
    n_doffs_4th_list = []
    # create a list of cutoffs to check starting from 2 to 11, with a step size of 0.1
    supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)
    max_cutoff = estimate_maximum_cutoff(supercell_atoms)
    cutoffs_2nd_to_check = np.arange(3, max_cutoff, 0.3)
    cutoffs_3rd_to_check = np.arange(3, max_cutoff, 0.1)
    cutoffs_4th_to_check = np.arange(3, max_cutoff, 0.1)

    # Assume that supercell_atoms is defined before
    supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)

    def get_best_cutoff(
        cutoffs_to_check: np.ndarray,
        target_n_doff: int,
        order: int
    ) -> tuple[float, list[int], np.ndarray]:
        """
        Find the best cutoff value for a given order of interaction.

        This function iterates over a range of cutoff values, evaluates the DOFs, and
        finds the best cutoff that meets the target number of DOFs. It performs
        iterative refinement to narrow down the best cutoff.

        Args:
        cutoffs_to_check (numpy.ndarray): An array of cutoff values to check.
        target_n_doff (int): The target number of degrees of freedom (DOFs).
        order (int): The order of interaction (2 for 2nd order, 3 for 3rd order,
                    4 for 4th order).

        Returns
        -------
            tuple: A tuple containing:
                - best_cutoff (float): The best cutoff value.
                - n_doffs_list (list[int]): A list of DOFs corresponding to the cutoffs
                                            checked.
                - cutoffs_to_check (numpy.ndarray): The array of cutoff values checked.

        """
        n_doffs_list = []
        best_cutoff = np.inf

        if order == 2:
            increment_size = 0.2
        elif order == 3:
            increment_size = 0.25
        elif order == 4:
            increment_size = 0.05

        for cutoff in cutoffs_to_check:
            # check if order == 2 and if no element in n_doffs_list is greater than 1000
            if order == 2 and all(n_doff < 1000 for n_doff in n_doffs_list):
                with contextlib.suppress(ValueError):
                    cs = ClusterSpace(
                        supercell_atoms,
                        [cutoff],
                        symprec=1e-3,
                        acoustic_sum_rules=True
                        )
            elif order == 3 and all(n_doff < 2200 for n_doff in n_doffs_list):
                with contextlib.suppress(ValueError):
                    cs = ClusterSpace(
                        supercell_atoms,
                        [2, cutoff],
                        symprec=1e-3,
                        acoustic_sum_rules=True
                        )
            elif order == 4 and all(n_doff < 2200 for n_doff in n_doffs_list):
                with contextlib.suppress(ValueError):
                    cs = ClusterSpace(
                        supercell_atoms,
                        [2, 3, cutoff],
                        symprec=1e-3,
                        acoustic_sum_rules=True
                        )

            try:
                n_doff = cs.get_n_dofs_by_order(order)
                if (
                    (order == 2 and n_doff < 1000) or
                    (order == 3 and n_doff < 2200) or
                    (order == 4 and n_doff < 2200)
                    ):
                    logger.info(f"adding n_doff = {n_doff} to the n_doff list")
                    n_doffs_list.append(n_doff)
                elif (
                    (order == 2 and n_doff > 1000) or
                    (order == 3 and n_doff > 2200) or
                    (order == 4 and n_doff > 2200)
                    ):
                    # remove all the cutoffs from cutoffs_to_check from the current
                    # cutoff once we are inside this block
                    cutoff_index = np.where(cutoffs_to_check == cutoff)[0][0]
                    cutoffs_to_check = cutoffs_to_check[:cutoff_index]
                    # do the same for n_doffs_list
                    n_doffs_list = n_doffs_list[:cutoff_index + 1]
                    break
            except UnboundLocalError:
                logger.info(f"UnboundLocalError for cutoff = {cutoff}")
                # find the index of the cutoff in the cutoffs_to_check list
                cutoff_index = np.where(cutoffs_to_check == cutoff)[0][0]
                logger.info(cutoff_index)
                # remove only the cutoff corresponding to the index from the list
                cutoffs_to_check = np.delete(cutoffs_to_check, cutoff_index)
                continue

        # Find the closest cutoff to the target
        closest_index = np.argmin(np.abs(np.array(n_doffs_list) - target_n_doff))
        best_cutoff = cutoffs_to_check[closest_index]

        while increment_size >= 0.01:
            if order == 2:
                with contextlib.suppress(ValueError):
                    cs = ClusterSpace(
                        supercell_atoms,
                        [best_cutoff],
                        symprec=1e-3,
                        acoustic_sum_rules=True
                        )
            elif order == 3:
                with contextlib.suppress(ValueError):
                    cs = ClusterSpace(
                        supercell_atoms,
                        [3, best_cutoff],
                        symprec=1e-3,
                        acoustic_sum_rules=True
                        )
            elif order == 4:
                with contextlib.suppress(ValueError):
                    cs = ClusterSpace(
                        supercell_atoms,
                        [3, 3, best_cutoff],
                        symprec=1e-3,
                        acoustic_sum_rules=True
                        )

            n_doff = cs.get_n_dofs_by_order(order)

            if n_doff > target_n_doff:
                best_cutoff -= increment_size / 2
            else:
                best_cutoff += increment_size / 2

            increment_size /= 4

        return best_cutoff, n_doffs_list, cutoffs_to_check

    # Get best cutoffs for 2nd, 3rd, and 4th order
    logger.info("Getting best cutoffs for 2nd order")
    best_2nd_order_cutoff, n_doffs_2nd_list, cutoffs_2nd_to_check = get_best_cutoff(
        cutoffs_2nd_to_check, 100, 2)
    logger.info("Getting best cutoffs for 3rd order")
    best_3rd_order_cutoff, n_doffs_3rd_list, cutoffs_3rd_to_check = get_best_cutoff(
        cutoffs_3rd_to_check, 1000, 3)
    logger.info("Getting best cutoffs for 4th order")
    best_4th_order_cutoff, n_doffs_4th_list, cutoffs_4th_to_check = get_best_cutoff(
        cutoffs_4th_to_check, 1000, 4)

    cutoffs_2nd_list.append(best_2nd_order_cutoff)
    cutoffs_3rd_list.append(best_3rd_order_cutoff)
    cutoffs_4th_list.append(best_4th_order_cutoff)

    best_cutoff_list = [[best_2nd_order_cutoff,
                         best_3rd_order_cutoff,
                         best_4th_order_cutoff]]
    logger.info(f"best_cutoff_list = {best_cutoff_list}")

    # Linear interpolation to find cutoffs for targets
    def interpolate_and_find_cutoffs(
            cutoffs_to_check: np.ndarray,
            n_doffs_list: list[int],
            targets: list[int]) -> np.ndarray:
        """
        Perform linear interpolation to find cutoff values for specific target DOFs.

        This function interpolates the given cutoff values and their corresponding DOFs
        to find the cutoff values that achieve the specified target DOFs.

        Args:
        cutoffs_to_check (numpy.ndarray): An array of cutoff values to check.
        n_doffs_list (list[int]): A list of DOFs corresponding to the cutoffs checked.
        targets (list[int]): A list of target DOFs for which to find the cutoff values.

        Returns
        -------
            numpy.ndarray: An array of interpolated cutoff values corresponding to the
            target DOFs.
        """
        logger.info(f"cutoffs_to_check = {cutoffs_to_check}")
        logger.info(f"n_doffs_list = {n_doffs_list}")
        logger.info(f"targets = {targets}")
        logger.info(f"len(cutoffs_to_check) = {len(cutoffs_to_check)}")
        logger.info(f"len(n_doffs_list) = {len(n_doffs_list)}")
        cutoffs_for_targets = np.interp(targets, n_doffs_list, cutoffs_to_check)
        logger.info(f"cutoffs_for_targets = {cutoffs_for_targets}")
        return cutoffs_for_targets


    cutoffs_2nd_list.extend(interpolate_and_find_cutoffs(
        cutoffs_2nd_to_check, n_doffs_2nd_list, [70, 220]))
    cutoffs_3rd_list.extend(interpolate_and_find_cutoffs(
        cutoffs_3rd_to_check, n_doffs_3rd_list, [1500, 2000]))
    cutoffs_4th_list.extend(interpolate_and_find_cutoffs(
        cutoffs_4th_to_check, n_doffs_4th_list, [1500, 2000]))

    # Sort the lists
    cutoffs_2nd_list = sorted(cutoffs_2nd_list)
    cutoffs_3rd_list = sorted(cutoffs_3rd_list)
    cutoffs_4th_list = sorted(cutoffs_4th_list)

    # Generate combinations of cutoffs
    cutoffs = np.array(list(map(list, product(
        cutoffs_2nd_list, cutoffs_3rd_list, cutoffs_4th_list))))
    logger.info(f"cutoffs = {cutoffs}")
    logger.info(f"len(cutoffs) = {len(cutoffs)}")
    logger.info(f"cutoffs[0] = {cutoffs[0]}")
    dofs_list = []
    for cutoff_value in cutoffs:
        logger.info(f"cutoff inside the for loop = {cutoff_value}")
        cutoff = cutoff_value.tolist()
        logger.info(f"cutoff.tolist() = {cutoff}")
        with contextlib.suppress(ValueError):
            cs = ClusterSpace(
                supercell_atoms,
                cutoff,
                symprec=1e-3,
                acoustic_sum_rules=True
                )
            n_doff_2 = cs.get_n_dofs_by_order(2)
            n_doff_3 = cs.get_n_dofs_by_order(3)
            n_doff_4 = cs.get_n_dofs_by_order(4)
            dofs_list.append([n_doff_2, n_doff_3, n_doff_4])
            logger.info(f"dofs_list = {dofs_list}")

    # Save the plots for cutoffs vs n_doffs
    def save_plot(
            cutoffs_to_check: np.ndarray,
            n_doffs_list: list[int],
            order: int) -> None:
        """Save the plot for cutoffs vs n_doffs."""
        plt.figure()
        plt.scatter(cutoffs_to_check, n_doffs_list, color="blue", label="Data Points")
        plt.plot(cutoffs_to_check,
                 n_doffs_list, color="red", label="Linear Interpolation")
        plt.xlabel(f"Cutoffs {order} Order")
        plt.ylabel(f"n_doffs_{order}")
        plt.title(f"Linear Interpolation for n_doffs_{order} vs Cutoffs {order} Order")
        plt.legend()
        plt.savefig(f"cutoffs_{order}_vs_n_doffs_{order}.png")

    save_plot(cutoffs_2nd_to_check, n_doffs_2nd_list, 2)
    save_plot(cutoffs_3rd_to_check, n_doffs_3rd_list, 3)
    save_plot(cutoffs_4th_to_check, n_doffs_4th_list, 4)

    logger.info("We have completed the cutoffs calculation.")
    logger.info(f"cutoffs = {cutoffs}")

    max_cutoff = estimate_maximum_cutoff(AseAtomsAdaptor.get_atoms(supercell_structure))
    cutoffs[cutoffs > max_cutoff] = max_cutoff
    logger.info(f"CUTOFFS \n {cutoffs}")
    logger.info(f"MAX_CUTOFF \n {max_cutoff}")
    good_cutoffs = np.all(cutoffs < max_cutoff - 0.1, axis=1)
    logger.info(f"GOOD CUTOFFS \n{good_cutoffs}")

    cutoffs = cutoffs[good_cutoffs].tolist()
    logger.info(f"cutoffs_used = {cutoffs}")

    def filter_cutoffs(
            cutoffs: list[list[int]],
            dofs_list: list[list[int]]) -> list[list[int]]:
        """Filter cutoffs based on unique DOFs."""
        # Map cutoffs to dofs_list
        cutoffs_to_dofs = {tuple(c): tuple(d) for c, d in zip(cutoffs, dofs_list)}
        logger.info(f"Cutoffs to DOFs mapping: {cutoffs_to_dofs}")

        # Track seen dofs and keep only the first occurrence of each unique dofs
        seen_dofs = set()
        new_cutoffs = []

        for c, d in zip(cutoffs, dofs_list):
            d_tuple = tuple(d)
            if d_tuple not in seen_dofs:
                seen_dofs.add(d_tuple)
                new_cutoffs.append(c)

        logger.info(f"Unique DOFs: {seen_dofs}")
        logger.info(f"New cutoffs: {new_cutoffs}")

        return new_cutoffs

    cutoffs = filter_cutoffs(cutoffs, dofs_list)
    logger.info(f"Filtered cutoffs: {cutoffs}")

    # Round each order of the cutoff to the first decimal place
    rounded_cutoffs = [[round(order, 1) for order in cutoff] for cutoff in cutoffs]
    logger.info(f"Rounded cutoffs: {rounded_cutoffs}")

    return rounded_cutoffs

def fit_force_constants(
    parent_structure: Structure,
    supercell_matrix: np.ndarray,
    supercell_structure: Structure,
    structures: list[Atoms],
    all_cutoffs: list[list[float]],
    disp_cut: float | None = 0.055,
    imaginary_tol: float | None = IMAGINARY_TOL,
    fit_method: str | None = FIT_METHOD,
    n_jobs: int | None = -1,
    fit_kwargs: dict | None = None,
) -> tuple:
    """
    Fit FC using hiphive.

    Fit force constants using hiphive and select the optimum cutoff values.
    The optimum cutoffs will be determined according to:
    1. Number imaginary modes < ``max_n_imaginary``.
    2. Most negative imaginary frequency < ``max_imaginary_freq``.
    3. Least number of imaginary modes.
    4. Lowest free energy at 300 K.
    If criteria 1 and 2 are not satisfied, None will be returned as the
    force constants.
    Args:
        parent_structure: Initial input structure.
        supercell_matrix: Supercell transformation matrix.
        structures: A list of ase atoms objects with "forces" and
            "displacements" arrays added, as required by hiPhive.
        all_cutoffs: A nested list of cutoff values to trial. Each set of
            cutoffs contains the radii for different orders starting with second
            order.
        disp_cut: determines the mean displacement of perturbed
            structure to be included in harmonic (<) or anharmonic (>) fitting
        imaginary_tol: Tolerance used to decide if a phonon mode is imaginary,
            in THz.
        max_n_imaginary: Maximum number of imaginary modes allowed in the
            the final fitted force constant solution. If this criteria is not
            reached by any cutoff combination this FireTask will fizzle.
        max_imaginary_freq: Maximum allowed imaginary frequency in the
            final fitted force constant solution. If this criteria is not
            reached by any cutoff combination this FireTask will fizzle.
        fit_method: Method used for fitting force constants. This can be
            any of the values allowed by the hiphive ``Optimizer`` class.
        n_jobs: Number of processors to use for fitting coefficients. -1 means use all
            processors.
        fit_kwargs: Additional arguments passed to the hiphive force constant
            optimizer.

    Returns
    -------
        A tuple of the best fitted force constants as a hiphive
        ``SortedForceConstants`` object, array of parameters, cluster space,
        and a dictionary of information on the fitting results.
    """
    logger.info("Starting force constant fitting.")
    logger.info(f"disp_cut is {disp_cut}")
    logger.info(f"fit_method is {fit_method}")

    fitting_data: dict[str, Any] = {
        "cutoffs": [],
        "rmse_test": [],
        "imaginary": [],
        "cs_dofs": [],
        "n_imaginary": [],
        "parent_structure": parent_structure,
        "supercell_structure": supercell_structure,
        "supercell_matrix": supercell_matrix,
        "fit_method": fit_method,
        "disp_cut": disp_cut,
        "imaginary_tol": imaginary_tol,
        "best_cutoff": None,
        "best_rmse": np.inf
    }

    best_fit = {
        "n_imaginary": np.inf,
        "rmse_test": np.inf,
        "imaginary": None,
        "cs_dofs": [None, None, None],
        "cluster_space": None,
        "force_constants": None,
        "parameters": None,
        "cutoffs": None,
        "force_constants_potential": None,
    }
    # all_cutoffs = all_cutoffs[0] #later change it back to all_cutoffs
    n_cutoffs = len(all_cutoffs)
    logger.info(f"len_cutoffs={n_cutoffs}")

    fit_kwargs = fit_kwargs if fit_kwargs else {}
    if fit_method == "rfe" and n_jobs == -1:
        fit_kwargs["n_jobs"] = 1
    elif fit_method == "lasso":
        fit_kwargs["lasso"] = dict(max_iter=1000)
    elif fit_method == "elasticnet":
        fit_kwargs = {"max_iter": 100000}

    logger.info(f"CPU COUNT: {os.cpu_count()}")

    logger.info("We are starting Joblib_s parallellized jobs")

    cutoff_results = []
    for i, cutoffs in enumerate(all_cutoffs):
        result = _run_cutoffs(i, cutoffs,
                              n_cutoffs,
                              parent_structure,
                              supercell_structure,
                              structures,
                              supercell_matrix,
                              fit_method,
                              disp_cut,
                              imaginary_tol,
                              fit_kwargs
                              )
        cutoff_results.append(result)

    for result in cutoff_results:
        if result is None:
            logger.info("result is None")
            continue
        logger.info(f"result = {result}")
        if result != {}:
            if "cutoffs" in result:
                fitting_data["cutoffs"].append(result["cutoffs"])
            else:
                logger.info("Cutoffs not found in result")
                continue
            if "rmse_test" in result:
                fitting_data["rmse_test"].append(result["rmse_test"])
            if "n_imaginary" in result:
                fitting_data["n_imaginary"].append(result["n_imaginary"])
            if "cs_dofs" in result:
                fitting_data["cs_dofs"].append(result["cs_dofs"])
            if "imaginary" in result:
                fitting_data["imaginary"].append(result["imaginary"])
            if "rmse_test" in result and (result["rmse_test"] < best_fit["rmse_test"]):
                best_fit.update(result)
                fitting_data["best_cutoff"] = result["cutoffs"]
                fitting_data["best_rmse"] = result["rmse_test"]
        else:
            logger.info("result is an empty dictionary")

    logger.info(f"best_fit = {best_fit}")
    logger.info(f"fitting_data = {fitting_data}")
    logger.info("Finished fitting force constants.")

    return (
        best_fit["force_constants"],
        best_fit["parameters"],
        best_fit["cluster_space"],
        fitting_data,
        best_fit["force_constants_potential"]
        )

def _run_cutoffs(
    i: int,
    cutoffs: list[float],
    n_cutoffs: int,
    parent_structure: Structure,
    supercell_structure: Structure,
    structures: list[Atoms],
    supercell_matrix: np.ndarray,
    fit_method: str,
    disp_cut: float,
    imaginary_tol: float,
    fit_kwargs: dict[str, Any],
) -> dict[str, Any]:

    logger.info(f"Testing cutoffs {i+1} out of {n_cutoffs}: {cutoffs}")
    logger.info(f"fit_method is {fit_method}")
    supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)
    supercell_atoms = structures[0]
    logger.info(f"supercell_atoms = {supercell_atoms}")

    if not is_cutoff_allowed(supercell_atoms, max(cutoffs)):
        logger.info("Skipping cutoff due as it is not commensurate with supercell size")
        return {}

    # # if you want to select specific clusters then use the following prototype code:
    # from hiphive.cutoffs import BeClusterFilter
    # be_cf = BeClusterFilter()
    # cs = ClusterSpace(
    #     supercell_atoms,
    #     cutoffs,
    #     cluster_filter=be_cf,
    #     symprec=1e-3,
    #     acoustic_sum_rules=True
    #     )
    # logger.info(f"cs orbit data = {cs.orbit_data}")

    try:
        cs = ClusterSpace(supercell_atoms, cutoffs, symprec=1e-3, acoustic_sum_rules=True)
    except ValueError:
        logger.info("ValueError encountered, moving to the next cutoff")
        return {}

    logger.debug(cs.__repr__())
    logger.info(cs)
    cs_2_dofs = cs.get_n_dofs_by_order(2)
    cs_3_dofs = cs.get_n_dofs_by_order(3)
    cs_4_dofs = cs.get_n_dofs_by_order(4)
    cs_dofs = [cs_2_dofs, cs_3_dofs, cs_4_dofs]
    logger.info(cs_dofs)
    n2nd = cs.get_n_dofs_by_order(2)
    nall = cs.n_dofs

    logger.info("Fitting harmonic force constants separately")
    separate_fit = True
    logger.info(f"disp_cut = {disp_cut}")

    sc = get_structure_container(
        cs, structures, separate_fit, disp_cut, ncut=n2nd, param2=None
    )
    opt = Optimizer(sc.get_fit_data(), fit_method, [0, n2nd], **fit_kwargs)
    try:
        opt.train()
    except Exception:
        logger.exception(f"Error occurred in opt.train() for cutoff: {i}, {cutoffs}")

        return {}
    param_harmonic = opt.parameters  # harmonic force constant parameters
    param_tmp = np.concatenate(
        (param_harmonic, np.zeros(cs.n_dofs - len(param_harmonic)))
    )

    parent_phonopy = get_phonopy_structure(parent_structure)
    phonopy = Phonopy(parent_phonopy, supercell_matrix=supercell_matrix)
    phonopy.primitive.get_number_of_atoms()
    mesh = supercell_matrix.diagonal() * 2

    fcp = ForceConstantPotential(cs, param_tmp)
    # supercell_atoms = phonopy_atoms_to_ase(phonopy.supercell)
    logger.info(f"supercell atoms = {supercell_atoms}")
    fcs = fcp.get_force_constants(supercell_atoms)
    logger.info("Did you get the large Condition number error?")

    phonopy.set_force_constants(fcs.get_fc_array(2))
    # phonopy.set_mesh(
    #     mesh, is_eigenvectors=False, is_mesh_symmetry=False
    # )  # run_mesh(is_gamma_center=True)
    phonopy.set_mesh(
        mesh, is_eigenvectors=True, is_mesh_symmetry=True
    )  # run_mesh(is_gamma_center=True)
    # phonopy.run_mesh(mesh, with_eigenvectors=False, is_mesh_symmetry=False)
    phonopy.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=True)
    omega = phonopy.mesh.frequencies  # THz
    omega = np.sort(omega.flatten())
    logger.info(f"omega_one_shot_fit = {omega}")
    imaginary = np.any(omega < -1e-3)
    logger.info(f"imaginary_one_shot_fit = {imaginary}")

    # # Phonopy's way of calculating phonon frequencies
    # structure_phonopy = get_phonopy_structure(parent_structure)
    # phonon = Phonopy(structure_phonopy, supercell_matrix=supercell_matrix)
    # phonon.set_force_constants(fcs.get_fc_array(2))
    # phonon.run_mesh(mesh=100.0, is_mesh_symmetry=False, is_gamma_center=True)
    # mesh = phonon.get_mesh_dict()
    # omega = mesh["frequencies"]
    # omega = np.sort(omega.flatten())
    # logger.info(f"omega_phonopy_one_shot_fitting = {omega}")
    # # imaginary = np.any(omega < -1e-3)
    # imaginary = np.any(omega < -imaginary_tol)
    # logger.info(f"imaginary_phonopy_one_shot_fitting = {imaginary}")
    n_imaginary = int(np.sum(omega < -np.abs(imaginary_tol)))

    if imaginary:
        logger.info(
            "Imaginary modes found! Fitting anharmonic force constants separately"
        )
        sc = get_structure_container(
            cs, structures, separate_fit, disp_cut, ncut=n2nd, param2=param_harmonic
        )
        opt = Optimizer(sc.get_fit_data(), fit_method, [n2nd, nall], **fit_kwargs)

        try:
            opt.train()
        except Exception:
            logger.exception(f"Error occured in opt.train() for cutoff: {i}, {cutoffs}")
            return {}
        param_anharmonic = opt.parameters  # anharmonic force constant parameters

        parameters = np.concatenate((param_harmonic, param_anharmonic))  # combine

        if nall != len(parameters):
            raise ValueError("nall is not equal to the length of parameters.")
        logger.info(f"Training complete for cutoff: {i}, {cutoffs}")


        parent_phonopy = get_phonopy_structure(parent_structure)
        phonopy = Phonopy(parent_phonopy, supercell_matrix=supercell_matrix)
        phonopy.primitive.get_number_of_atoms()
        mesh = supercell_matrix.diagonal() * 2

        fcp = ForceConstantPotential(cs, parameters)
        # supercell_atoms = phonopy_atoms_to_ase(phonopy.supercell)
        logger.info(f"supercell atoms = {supercell_atoms}")
        fcs = fcp.get_force_constants(supercell_atoms)
        logger.info("Did you get the large Condition number error?")

        phonopy.set_force_constants(fcs.get_fc_array(2))
        phonopy.set_mesh(
            mesh, is_eigenvectors=False, is_mesh_symmetry=False
        )  # run_mesh(is_gamma_center=True)
        phonopy.run_mesh(mesh=100.0, with_eigenvectors=False, is_mesh_symmetry=False)
        omega = phonopy.mesh.frequencies  # THz
        omega = np.sort(omega.flatten())
        logger.info(f"omega_seperate_fit = {omega}")
        imaginary = np.any(omega < -1e-3)
        logger.info(f"imaginary_seperate_fit = {imaginary}")

        # Phonopy's way of calculating phonon frequencies
        structure_phonopy = get_phonopy_structure(parent_structure)
        phonon = Phonopy(structure_phonopy, supercell_matrix=supercell_matrix)
        phonon.set_force_constants(fcs.get_fc_array(2))
        phonon.run_mesh(mesh, is_mesh_symmetry=False, is_gamma_center=True)
        mesh = phonon.get_mesh_dict()
        omega = mesh["frequencies"]
        omega = np.sort(omega.flatten())
        logger.info(f"omega_phonopy_seperate_fit = {omega}")
        # imaginary = np.any(omega < -1e-3)
        # logger.info(f"imaginary_phonopy_seperate_fit = {imaginary}")
        imaginary = np.any(omega < -imaginary_tol)
        logger.info(f"imaginary_phonopy_seperate_fit = {imaginary}")
        n_imaginary = int(np.sum(omega < -np.abs(imaginary_tol)))

    else:
        logger.info("No imaginary modes! Fitting all force constants in one shot")
        separate_fit = False
        sc = get_structure_container(
            cs, structures, separate_fit, disp_cut=None, ncut=None, param2=None
        )
        opt = Optimizer(sc.get_fit_data(), fit_method, [0, nall], **fit_kwargs)

        try:
            opt.train()
        except Exception:
            logger.exception(f"Error occured in opt.train() for cutoff: {i}, {cutoffs}")
            return {}
        parameters = opt.parameters
        logger.info(f"Training complete for cutoff: {i}, {cutoffs}")


    logger.info(f"parameters before enforcing sum rules {parameters}")
    logger.info(f"Memory use: {psutil.virtual_memory().percent} %")
    parameters = enforce_rotational_sum_rules(cs, parameters, ["Huang", "Born-Huang"])
    fcp = ForceConstantPotential(cs, parameters)
    # supercell_atoms = phonopy_atoms_to_ase(phonopy.supercell)
    fcs = fcp.get_force_constants(supercell_atoms)
    logger.info(f"FCS generated for cutoff {i}, {cutoffs}")

    return {
        "cutoffs": cutoffs,
        "rmse_test": opt.rmse_test,
        "cluster_space": sc.cluster_space,
        "parameters": parameters,
        "force_constants": fcs,
        "force_constants_potential": fcp,
        "imaginary": imaginary,
        "cs_dofs": cs_dofs,
        "n_imaginary": n_imaginary,
    }


def get_structure_container(
    cs: ClusterSpace,
    structures: list[Atoms],
    separate_fit: bool,
    disp_cut: float,
    ncut: int,
    param2: np.ndarray,
) -> StructureContainer:
    """Get a hiPhive StructureContainer from cutoffs and a list of atoms objects.

    Args:
        cs: ClusterSpace
        structures: A list of ase atoms objects with the "forces" and
            "displacements" arrays included.
        separate_fit: Boolean to determine whether harmonic and anharmonic fitting
            are to be done separately (True) or in one shot (False)
        disp_cut: if separate_fit true, determines the mean displacement of perturbed
            structure to be included in harmonic (<) or anharmonic (>) fitting
        ncut: the parameter index where fitting separation occurs
        param2: previously fit parameter array (harmonic only for now, hence 2).

    Returns
    -------
        A hiPhive StructureContainer.
    """
    sc = StructureContainer(cs)
    logger.info(f"sc = {sc}")
    logger.info(f"initial shape of fit matrix = {sc.data_shape}")
    saved_structures = []
    for _, structure in enumerate(structures):
        displacements = structure.get_array("displacements")
        forces = structure.get_array("forces")
        # Calculate mean displacements
        mean_displacements = np.linalg.norm(displacements, axis=1).mean()
        logger.info(f"Mean displacements: {mean_displacements}")
        # Calculate mean forces
        mean_forces = np.linalg.norm(forces, axis=1).mean()
        logger.info(f"Mean Forces: {mean_forces}")
        # Calculate standard deviation of displacements
        std_displacements = np.linalg.norm(displacements, axis=1).std()
        logger.info(f"Standard deviation of displacements: "
                    f"{std_displacements}")
        # Calculate standard deviation of forces
        std_forces = np.linalg.norm(forces, axis=1).std()
        logger.info(f"Standard deviation of forces: "
                    f"{std_forces}")
        if not separate_fit:  # fit all
            sc.add_structure(structure)
        # for harmonic fitting
        elif separate_fit and param2 is None and mean_displacements < disp_cut:
            logger.info("We are in harmonic fitting if statement")
            # logger.info(f"mean_disp = {mean_displacements}")
            logger.info(f"mean_forces = {mean_forces}")
            sc.add_structure(structure)
        # for anharmonic fitting
        elif separate_fit and param2 is not None and mean_displacements >= disp_cut:
            logger.info("We are in anharmonic fitting if statement")
            # logger.info(f"mean_disp = {mean_displacements}")
            logger.info(f"mean_forces = {mean_forces}")
            sc.add_structure(structure)
            saved_structures.append(structure)

    logger.info("final shape of fit matrix (total # of atoms in all added"
                f"supercells, n_dofs) = (rows, columns) = {sc.data_shape}")
    logger.info("We have completed adding structures")
    logger.info(f"sc.get_fit_data() = {sc.get_fit_data()}")

    if separate_fit and param2 is not None:  # do after anharmonic fitting
        a_mat = sc.get_fit_data()[0]  # displacement matrix
        f_vec = sc.get_fit_data()[1]  # force vector
        f_vec -= np.dot(a_mat[:, :ncut], param2)  # subtract harmonic forces
        sc.delete_all_structures()
        for i, structure in enumerate(saved_structures):
            natoms = structure.get_global_number_of_atoms()
            ndisp = natoms * 3
            structure.set_array(
                "forces", f_vec[i * ndisp : (i + 1) * ndisp].reshape(natoms, 3)
            )
            sc.add_structure(structure)

    logger.debug(sc.__repr__())

    return sc

def harmonic_properties(
    structure: Structure,
    supercell_matrix: np.ndarray,
    fcs: ForceConstants,
    temperature: list,
    imaginary_tol: float = IMAGINARY_TOL,
    mesh: list = None
) -> tuple[dict,Phonopy]:
    """
    Thermodynamic (harmonic) phonon props calculated using the force constants.

    Args:
        structure: The parent structure.
        supercell_matrix: The supercell transformation matrix.
        force_constants: The force constants in numpy format.
        imaginary_tol: Tolerance used to decide if a phonon mode is imaginary,
            in THz.

    Returns
    -------
        A tuple of the number of imaginary modes at Gamma, the minimum phonon
        frequency at Gamma, and the free energy, entropy, and heat capacity
    """
    logger.info("Evaluating harmonic properties...")
    fcs2 = fcs.get_fc_array(2)
    logger.info("fcs2 & fcs3 read...")
    logger.info(f"fcs2 = {fcs2}")
    parent_phonopy = get_phonopy_structure(structure)
    phonopy = Phonopy(parent_phonopy, supercell_matrix=supercell_matrix)
    natom = phonopy.primitive.get_number_of_atoms()
    mesh = supercell_matrix.diagonal()*2
    logger.info(f"Mesh: {mesh}")

    phonopy.set_force_constants(fcs2)
    phonopy.set_mesh(mesh,is_eigenvectors=True,is_mesh_symmetry=False)
    phonopy.run_thermal_properties(temperatures=temperature)
    logger.info("Thermal properties successfully run!")

    _, free_energy, entropy, heat_capacity = phonopy.get_thermal_properties()

    ## Use the following lines to convert the units to eV/atom
    # free_energy *= 1000/sp.constants.Avogadro/eV2J/natom # kJ/mol to eV/atom
    # entropy *= 1/sp.constants.Avogadro/eV2J/natom # J/K/mol to eV/K/atom
    # heat_capacity *= 1/sp.constants.Avogadro/ev2j/natom # J/K/mol to eV/K/atom

    freq = phonopy.mesh.frequencies # in THz
    logger.info(f"Frequencies: {freq}")
    logger.info(f"freq_flatten = {np.sort(freq.flatten())}")

    n_imaginary = int(np.sum(freq < -np.abs(imaginary_tol)))

    if n_imaginary == 0:
        logger.info("No imaginary modes!")
    else:
        logger.warning("Imaginary modes found!")

    return {
        "temperature": temperature,
        "free_energy": free_energy,
        "entropy": entropy,
        "heat_capacity": heat_capacity,
        "n_imaginary": n_imaginary
        }, phonopy


def anharmonic_properties(
    phonopy: Phonopy,
    fcs: ForceConstants,
    temperature: list,
    heat_capacity: np.ndarray,
    n_imaginary: float,
    bulk_modulus: float = None
) -> dict:

    if n_imaginary == 0:
        logger.info("Evaluating anharmonic properties...")
        fcs2 = fcs.get_fc_array(2)
        fcs3 = fcs.get_fc_array(3)
        grun, cte, dlfrac = gruneisen(phonopy,fcs2,fcs3,temperature,heat_capacity,
                                      bulk_modulus=bulk_modulus)
    else: # do not calculate these if imaginary modes exist
        logger.warning("Gruneisen and thermal expansion cannot be calculated with"
                       "imaginary modes. All set to 0.")
        grun = np.zeros((len(temperature),3))
        cte = np.zeros((len(temperature),3))
        dlfrac = np.zeros((len(temperature),3))

    return {
        "gruneisen": grun,
        "thermal_expansion": cte,
        "expansion_fraction": dlfrac,
        }


def get_total_grun(
        omega: np.ndarray,
        grun: np.ndarray,
        kweight: np.ndarray,
        t: float
) -> np.ndarray:
    total = 0
    weight = 0
    nptk = omega.shape[0]
    nbands = omega.shape[1]
    omega = abs(omega)*1e12*2*np.pi
    if t==0:
        total = np.zeros((3,3))
        grun_total_diag = np.zeros(3)
    else:
        for i in range(nptk):
            for j in range(nbands):
                x = hbar*omega[i,j]/(2.0*kb*t)
                dbe = (x/np.sinh(x))**2
                weight += dbe*kweight[i]
                total += dbe*kweight[i]*grun[i,j]
        total = total/weight
        grun_total_diag = np.array([total[0,2],total[1,1],total[2,0]])

        def percent_diff(a: float, b: float) -> float:
            return abs((a-b)/b)
        # This process preserves cell symmetry upon thermal expansion, i.e., it prevents
        # symmetry-identical directions from inadvertently expanding by different ratios
        # when/if the Gruneisen routine returns slightly different ratios for those
        # directions
        avg012 = np.mean((grun_total_diag[0],grun_total_diag[1],grun_total_diag[2]))
        avg01 = np.mean((grun_total_diag[0],grun_total_diag[1]))
        avg02 = np.mean((grun_total_diag[0],grun_total_diag[2]))
        avg12 = np.mean((grun_total_diag[1],grun_total_diag[2]))
        if percent_diff(grun_total_diag[0],avg012) < 0.1:
            if percent_diff(grun_total_diag[1],avg012) < 0.1:
                if percent_diff(grun_total_diag[2],avg012) < 0.1: # all similar
                    grun_total_diag[0] = avg012
                    grun_total_diag[1] = avg012
                    grun_total_diag[2] = avg012
                elif percent_diff(grun_total_diag[2],avg02) < 0.1: # 0 and 2 similar
                    grun_total_diag[0] = avg02
                    grun_total_diag[2] = avg02
                elif percent_diff(grun_total_diag[2],avg12) < 0.1: # 1 and 2 similar
                    grun_total_diag[1] = avg12
                    grun_total_diag[2] = avg12
                else:
                    pass
            elif percent_diff(grun_total_diag[1],avg01) < 0.1: # 0 and 1 similar
                grun_total_diag[0] = avg01
                grun_total_diag[1] = avg01
            elif percent_diff(grun_total_diag[1],avg12) < 0.1: # 1 and 2 similar
                grun_total_diag[1] = avg12
                grun_total_diag[2] = avg12
            else:
                pass
        elif percent_diff(grun_total_diag[0],avg01) < 0.1: # 0 and 1 similar
            grun_total_diag[0] = avg01
            grun_total_diag[1] = avg01
        elif percent_diff(grun_total_diag[0],avg02) < 0.1: # 0 and 2 similar
            grun_total_diag[0] = avg02
            grun_total_diag[2] = avg02
        else: # nothing similar
            pass

    return grun_total_diag


def gruneisen(
        phonopy: Phonopy,
        fcs2: np.ndarray,
        fcs3: np.ndarray,
        temperature: list,
        heat_capacity: np.ndarray, # in J/K/mol
        bulk_modulus: float = None # in GPa
) -> tuple[list,list]:

    gruneisen = Gruneisen(fcs2,fcs3,phonopy.supercell,phonopy.primitive)
    gruneisen.set_sampling_mesh(phonopy.mesh_numbers,is_gamma_center=True)
    gruneisen.run()
    grun = gruneisen.get_gruneisen_parameters() # (nptk,nmode,3,3)
    omega = gruneisen._frequencies  # noqa: SLF001
    # qp = gruneisen._qpoints
    kweight = gruneisen._weights  # noqa: SLF001

    grun_tot = [get_total_grun(omega, grun, kweight, temp) for temp in temperature]
    grun_tot = np.nan_to_num(np.array(grun_tot))

    # linear thermal expansion coefficeint and fraction
    if bulk_modulus is None:
        cte = None
        dlfrac = None
    else:
        # heat_capacity *= eV2J*phonopy.primitive.get_number_of_atoms()
        # # eV/K/atom to J/K
        heat_capacity *= 1/sp.constants.Avogadro # J/K/mol to J/K
        # to convert from J/K/atom multiply by phonopy.primitive.get_number_of_atoms()
        # Convert heat_capacity to an array if it's a scalar
        # heat_capacity = np.array([heat_capacity])
        logger.info(f"heat capacity = {heat_capacity}")
        vol = phonopy.primitive.get_volume()

        logger.info(f"grun_tot: {grun_tot}")
        logger.info(f"grun_tot shape: {grun_tot.shape}")
        logger.info(f"heat_capacity shape: {heat_capacity.shape}")
        logger.info(f"heat_capacity: {heat_capacity}")
        logger.info(f"vol: {vol}")
        logger.info(f"bulk_modulus: {bulk_modulus}")
#        cte = grun_tot*heat_capacity.repeat(3)/(vol/10**30)/(bulk_modulus*10**9)/3
        cte = grun_tot*heat_capacity.repeat(3).reshape(
            len(heat_capacity),3)/(vol/10**30)/(bulk_modulus*10**9)/3
        cte = np.nan_to_num(cte)
        if len(temperature)==1:
            dlfrac = cte*temperature
        else:
            dlfrac = thermal_expansion(temperature, cte)
        logger.info(f"Gruneisen: \n {grun_tot}")
        logger.info(f"Coefficient of Thermal Expansion: \n {cte}")
        logger.info(f"Linear Expansion Fraction: \n {dlfrac}")

    return grun_tot, cte, dlfrac


def thermal_expansion(
        temperature: list,
        cte: np.array,
) -> np.ndarray:
    if len(temperature) != len(cte):
        raise ValueError("Length of temperature and cte lists must be equal.")
    if 0 not in temperature:
        temperature = [0, *temperature]
        cte = np.array([np.array([0, 0, 0]), *list(cte)])
    temperature = np.array(temperature)
    ind = np.argsort(temperature)
    temperature = temperature[ind]
    cte = np.array(cte)[ind]
    # linear expansion fraction
    dlfrac = copy(cte)
    for t in range(len(temperature)):
        dlfrac[t,:] = np.trapz(cte[:t+1,:],temperature[:t+1],axis=0)
    dlfrac = np.nan_to_num(dlfrac)
    logger.info(f"Linear Expansion Fraction: \n {dlfrac}")
    return dlfrac

@job
def run_thermal_cond_solver(
    renormalized: bool | None = False,
    temperature: list[int] | None = None,
    control_kwargs: dict | None = None,
    prev_dir_hiphive: str | None = None,
    therm_cond_solver: str | None = "almabte"
) -> None:
    """
    Thermal conductivity calculation using the specified solver.

    Run the specified solver to calculate lattice thermal conductivity.
    Presumes the FORCE_CONSTANTS_3RD and FORCE_CONSTANTS_2ND files, and a
    "structure_data.json" file with the keys "structure" and "supercell_matrix"
    are in the current directory.

    Parameters
    ----------
        renormalized (bool): Whether force constants are from phonon renormalization
            (True) or directly from fitting (False). Defaults to False.
        temperature (list[int]): The temperature to calculate the lattice thermal
            conductivity for. Can be given as a single float, or a dictionary with the
            keys "min", "max", "step".
        control_kwargs (dict): Options to be included in the solver control file.
        prev_dir_hiphive (str): Directory containing previous HiPhive outputs.
        therm_cond_solver (str): The name of the solver executable to run.
            Defaults to "almabte".

    Raises
    ------
        RuntimeError: If the solver command returns a non-zero exit code.
        TypeError: If temperature type is not int, float, or dict when
            renormalized is True.
        ValueError: If temperature is not a valid type.
    """
    # Select the appropriate solver command
    solver_cmd_mapping = {
        "almabte": Atomate2Settings().ALMABTE_CMD,
        "shengbte": Atomate2Settings().SHENGBTE_CMD,
        "phono3py": Atomate2Settings().PHONO3PY_CMD
    }
    if therm_cond_solver not in solver_cmd_mapping:
        raise ValueError(f"Unsupported solver: {therm_cond_solver}")

    therm_cond_solver_cmd = solver_cmd_mapping[therm_cond_solver]

    logger.info(f"therm_cond_solver_cmd = {therm_cond_solver_cmd}")
    therm_cond_solver_cmd = os.path.expandvars(therm_cond_solver_cmd)

    logger.info(f"Running {therm_cond_solver_cmd} command")

    copy_hiphive_outputs(prev_dir_hiphive)

    structure_data = loadfn("structure_data.json")
    structure = structure_data["structure"]
    supercell_matrix = structure_data["supercell_matrix"]

    # structure = SpacegroupAnalyzer(structure).find_primitive() #TODO refactor this later

    logger.info(f"Temperature = {temperature}")

    temperature = temperature if temperature is not None else T_KLAT
    logger.info(f"Temperature = {temperature}")
    logger.info(f"type of temperature = {type(temperature)}")

    if renormalized:
        if not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be an int or float")
    elif isinstance(temperature, dict):
        if not all(k in temperature for k in ["min", "max", "step"]):
            raise ValueError("Temperature dict must contain 'min', 'max',"
                            "and 'step'")
    elif not isinstance(temperature, (int, float)):
        raise ValueError("Unsupported temperature type, must be int, float,"
                        "or dict")


    logger.info("Creating control dict")
    logger.info(f"temperature = {temperature}")

    control_dict = {
        "scalebroad": 0.5, # 1.1
        "nonanalytic": False,
        "isotopes": False,
        "temperature": temperature,
        "scell": np.diag(supercell_matrix).tolist(),
    }
    control_kwargs = control_kwargs or {}
    control_dict.update(control_kwargs)
    control = Control().from_structure(structure, **control_dict)
    control.to_file()

    if isinstance(therm_cond_solver_cmd, str):
        therm_cond_solver_cmd = shlex.split(therm_cond_solver_cmd)

    therm_cond_solver_cmd = list(therm_cond_solver_cmd)

    with open("shengbte.out", "w") as f_std, open(
        "shengbte_err.txt", "w", buffering=1
    ) as f_err:
        # use line buffering for stderr
        return_code = subprocess.call(therm_cond_solver_cmd, stdout=f_std, stderr=f_err) # noqa: S603

    logger.info(
        f"Command {therm_cond_solver_cmd}"
         "finished running with returncode: {return_code}"
    )

    if return_code == 1:
        raise RuntimeError(
            f"Running the solver failed. Check '{os.getcwd()}/shengbte_err.txt' for "
            "details."
        )



@job
def run_fc_to_pdos(
    renormalized: bool | None = None,
    mesh_density: float | None = None,
    prev_dir_json_saver: str | None = None,
) -> tuple:
    """
    FC to PDOS.

    Add force constants, phonon band structure and density of states
    to the database.

    Assumes you are in a directory with the force constants, fitting
    data, and structure data written to files.

    Required parameters:
        db_file (str): Path to DB file for the database that contains the
            perturbed structure calculations.

    Optional parameters:
        renormalized (bool): Whether FC resulted from original fitting (False)
            or renormalization process (True) determines how data are stored.
            Default is False.
        mesh_density (float): The density of the q-point mesh used to calculate
            the phonon density of states. See the docstring for the ``mesh``
            argument in Phonopy.init_mesh() for more details.
        additional_fields (dict): Additional fields added to the document, such
            as user-defined tags, name, ids, etc.
    """
    copy_hiphive_outputs(prev_dir_json_saver)

    renormalized = renormalized if renormalized else False
    mesh_density = mesh_density if mesh_density else 100.0

    structure_data = loadfn("structure_data.json")
    structure = structure_data["structure"]
    structure_data["supercell_structure"]
    supercell_matrix = structure_data["supercell_matrix"]

    if not renormalized:
        loadfn("perturbed_structures.json")
        loadfn("perturbed_forces_new.json")
        fcs = ForceConstants.read("force_constants.fcs")

        uniform_bs, lm_bs, dos = _get_fc_fsid(
            structure, supercell_matrix, fcs, mesh_density
        )

        logger.info("Finished inserting force constants and phonon data")

    else:
        renorm_thermal_data = loadfn("thermal_data.json") # renorm_thermal_data.json
        fcs = ForceConstants.read("force_constants.fcs")
        t = renorm_thermal_data["temperature"]

        uniform_bs, lm_bs, dos = _get_fc_fsid(
            structure, supercell_matrix, fcs, mesh_density
        )

        logger.info(
            f"Finished inserting renormalized force constants and phonon data at {t} K"
        )

    # convert uniform_bs to dict
    uniform_bs_dict = uniform_bs.as_dict()

    # dump uniform_bs_dict to file
    dumpfn(uniform_bs_dict, "uniform_bs.json")

    return uniform_bs, lm_bs, dos, prev_dir_json_saver


def _get_fc_fsid(structure: Structure,
                 supercell_matrix: list[list[int]] | None = None,
                 fcs: ForceConstants | None = None,
                 mesh_density: float | None = None) -> tuple:
    phonopy_fc = fcs.get_fc_array(order=2)

    logger.info("Getting uniform phonon band structure.")
    uniform_bs = get_phonon_band_structure_from_fc(
        structure, supercell_matrix, phonopy_fc
    )

    logger.info("Getting line mode phonon band structure.")
    lm_bs = get_phonon_band_structure_symm_line_from_fc(
        structure, supercell_matrix, phonopy_fc
    )

    logger.info("Getting phonon density of states.")
    dos = get_phonon_dos_from_fc(
        structure, supercell_matrix, phonopy_fc, mesh_density=mesh_density
    )
    return uniform_bs, lm_bs, dos


@job
def run_hiphive_renormalization(
    temperature: float,
    renorm_method: str,
    nconfig: int,
    renorm_te_iter: bool,
    bulk_modulus: float,
    prev_dir_hiphive: str,
    perform_ti_flag: bool = False,
) -> list[str, dict[str, Any]]:
    """
    Phonon renormalization using hiPhive.

    Perform phonon renormalization to obtain temperature-dependent force constants
    using hiPhive. Requires "structure_data.json" to be present in the current working
    directory.
    Required parameters:

    Optional parameter:
        renorm_temp (List): list of temperatures to perform renormalization -
        defaults to T_RENORM
        renorm_with_te (bool): if True, perform outer iteration over thermally
        expanded volumes
        bulk_modulus (float): input bulk modulus - required for thermal expansion
        iterations
    """
    copy_hiphive_outputs(prev_dir_hiphive)

    cs = ClusterSpace.read("cluster_space.cs")
    fcs = ForceConstants.read("force_constants.fcs")
    param = np.loadtxt("parameters.txt")
    fitting_data = loadfn("fitting_data.json")
    structure_data = loadfn("structure_data.json")
    phonopy_orig = phpy.load("phonopy_params.yaml")

    n_imaginary_orig = fitting_data["n_imaginary"]
    imag_modes_bool = n_imaginary_orig > 0 # True if imaginary modes exist

    cutoffs = fitting_data["cutoffs"]
    fit_method = fitting_data["fit_method"]

    parent_structure = structure_data["structure"]
    supercell_structure = structure_data["supercell_structure"]
    supercell_matrix = np.array(structure_data["supercell_matrix"])

    parent_atoms = AseAtomsAdaptor.get_atoms(parent_structure)
    supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)


    # Renormalization with DFT lattice
    td_data = _run_renormalization(parent_structure, supercell_structure,
                                   supercell_matrix, cs, fcs, param, temperature,
                                   nconfig, renorm_method, fit_method, bulk_modulus,
                                   phonopy_orig)
    td_structure_data = copy(structure_data)
    td_structure_data["structure"] = parent_structure
    td_structure_data["supercell_structure"] = supercell_structure

    logger.info("Renormalization is now completeed")
    # Additional renormalization with thermal expansion -
    # optional - just single "iteration" for now
    if renorm_te_iter:
        n_te_iter = 1
        for i in range(n_te_iter):
            if td_data is None or td_data["n_imaginary"] < 0:
                # failed, incomplete, or still imaginary
                break
            logger.info(
                f"Renormalizing with thermally expanded lattice - iteration {i}"
            )

            dlfrac = td_data["expansion_fraction"]
            param_td = td_data["param"]

            a, b, c, d, e, failed = setup_te_renorm(
                    cs,cutoffs,parent_atoms,supercell_atoms,param_td,dlfrac,supercell_matrix
                )
            if not failed:
                (parent_structure_td,
                 supercell_structure_td,
                 cs_td,
                 phonopy_td,
                 fcs_td) = a, b, c, d, e
                td_data = _run_renormalization(parent_structure_td,
                                               supercell_structure_td, supercell_matrix,
                                                cs_td, fcs, param, temperature, nconfig,
                                                renorm_method, fit_method, bulk_modulus,
                                                phonopy_td, param_td, fcs_td
                                                )
                td_structure_data["structure"] = parent_structure_td
                td_structure_data["supercell_structure"] = supercell_structure_td

    # Thermodynamic integration for anharmonic free energy
    if perform_ti_flag:
        td_data = thermodynamic_integration_ifc(
            td_data, # everything TD
            fcs, # original
            param, # original
            imag_modes_bool, # if False, only uses lambda=0
            )

    # write results
    logger.info("Writing renormalized results")
    fcs_td = td_data["fcs"]
    fcs_td.write("force_constants.fcs")
    if td_data["n_imaginary"] != 0:
        if perform_ti_flag:
            thermal_keys = ["temperature","free_energy","entropy","heat_capacity",
                        "free_energy_correction_S","free_energy_correction_SC",
                        "free_energy_correction_TI"]
        else:
            thermal_keys = ["temperature","free_energy","entropy","heat_capacity",
                        "free_energy_correction_S","free_energy_correction_SC"]
    elif perform_ti_flag:
        thermal_keys = ["temperature","free_energy","entropy","heat_capacity",
                    "gruneisen","thermal_expansion","expansion_fraction",
                    "free_energy_correction_S","free_energy_correction_SC",
                    "free_energy_correction_TI"]
    else:
        thermal_keys = ["temperature","free_energy","entropy","heat_capacity",
                    "gruneisen","thermal_expansion","expansion_fraction",
                    "free_energy_correction_S","free_energy_correction_SC"]

    td_thermal_data = {key: [] for key in thermal_keys}

    for key in thermal_keys:
        td_thermal_data[key].append(td_data[key])

    logger.info(f"DEBUG: {td_data}")
    if td_data["n_imaginary"] > 0:
        logger.warning("Imaginary modes remain still exist")
        logger.warning("ShengBTE FORCE_CONSTANTS_2ND & FORCE_CONSTANTS_3RD not written")
    else:
        logger.info("No imaginary modes! Writing ShengBTE files")

        parent_atoms_td = copy(parent_atoms)
        logger.info(f"TD_data exp frac: {td_data['expansion_fraction']}")
        logger.info(f"TD_data exp frac 0: {td_data['expansion_fraction'][0,2]}")
        new_cell = Cell(
            np.transpose([parent_atoms_td.get_cell()[:,i]*(
                1+td_data["expansion_fraction"][0,i]
                ) for i in range(3)])
            )
        parent_atoms_td.set_cell(new_cell,scale_atoms=True)

        prim_td_phonopy = PhonopyAtoms(symbols=parent_atoms_td.get_chemical_symbols(),
                                        scaled_positions=parent_atoms_td.get_scaled_positions(),
                                        cell=parent_atoms_td.cell)
        phonopy_td = Phonopy(prim_td_phonopy, supercell_matrix=supercell_matrix,
                             primitive_matrix=None)

        atoms = AseAtomsAdaptor.get_atoms(parent_structure_td)
        # fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD", atoms, order=3)
        fcs_td.write_to_phonopy(f"FORCE_CONSTANTS_2ND_{temperature}", format="text")
        ForceConstants.write_to_phonopy(fcs_td, "fc2.hdf5", "hdf5")
        ForceConstants.write_to_phono3py(fcs_td, "fc3.hdf5", "hdf5")
        ### detour from hdf5
        supercell_atoms = phonopy_atoms_to_ase(phonopy_td.supercell)
        fcs = ForceConstants.read_phono3py(supercell_atoms, "fc3.hdf5", order=3)
        fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD_{temperature}", atoms,
                              order=3, fc_tol=1e-4)

        atoms = AseAtomsAdaptor.get_atoms(parent_structure)
        supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)
        # fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD", atoms, order=3)
        fcs_td.write_to_phonopy("FORCE_CONSTANTS_2ND", format="text")
        ForceConstants.write_to_phonopy(fcs_td, "fc2.hdf5", "hdf5")
        ForceConstants.write_to_phono3py(fcs_td, "fc3.hdf5", "hdf5")
        ### detour from hdf5
        # supercell_atoms = phonopy_atoms_to_ase(phonopy.supercell)
        fcs = ForceConstants.read_phono3py(supercell_atoms, "fc3.hdf5", order=3)
        fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD", atoms, order=3, fc_tol=1e-4)

    dumpfn(td_structure_data, "structure_data.json")
    dumpfn(td_thermal_data, "thermal_data.json")

    current_dir = os.getcwd()

    return [current_dir, td_thermal_data]


def _run_renormalization(
    structure: Structure,
    supercell_structure: Structure,
    supercell_matrix: np.ndarray,
    cs: ClusterSpace,
    fcs: ForceConstants,
    param: np.ndarray,
    t: float,
    nconfig: int,
    renorm_method: str,
    fit_method: str,
    bulk_modulus: float = None,
    phonopy_orig: Phonopy = None,
    param_td: np.ndarray = None,
    fcs_td: ForceConstants = None,
    imaginary_tol: float = IMAGINARY_TOL,
) -> dict:
    """
    Run phonon renormalization.

    Uses the force constants to extract phonon properties. Used for comparing
    the accuracy of force constant fits.
    Args:
        structure: pymatgen Structure
            The parent structure.
        supercell : ase Atoms
            Original supercell object
        supercell_matrix: The supercell transformation matrix.
        fcs: ForceConstants from previous fitting or renormalization
        imaginary_tol: Tolerance used to decide if a phonon mode is imaginary,
            in THz.

    Returns
    -------
        A tuple of the number of imaginary modes at Gamma, the minimum phonon
        frequency at Gamma, and the free energy, entropy, and heat capacity.
    """
    logger.info(f"renorm_method = {renorm_method}")
    nconfig = int(nconfig)
    supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)
    renorm = Renormalization(cs,supercell_atoms,param,fcs,t,renorm_method,fit_method,
                             param_TD=param_td,fcs_TD=fcs_td)
    fcp_td, fcs_td, param_td = renorm.renormalize(nconfig)#,conv_tresh)

    td_data, phonopy_td = harmonic_properties(
        structure, supercell_matrix, fcs_td, [t], imaginary_tol
    )
    logger.info(f"Heat capacity_TD_DATA: {td_data['heat_capacity']}")
    if td_data["n_imaginary"] == 0:
        logger.info(f"Renormalized phonon is completely real at T = {t} K!")
        anharmonic_data = anharmonic_properties(
            phonopy_td, fcs_td, [t], td_data["heat_capacity"], td_data["n_imaginary"],
            bulk_modulus=bulk_modulus
        )
        td_data.update(anharmonic_data)

    # phonopy_orig.run_mesh()
    # phonopy_TD.run_mesh()
    mesh = supercell_matrix.diagonal()*2
    phonopy_orig.set_mesh(mesh,is_eigenvectors=True,is_mesh_symmetry=False)
    phonopy_td.set_mesh(mesh,is_eigenvectors=True,is_mesh_symmetry=False)
    omega_h = phonopy_orig.mesh.frequencies # THz
    evec_h = phonopy_orig.mesh.eigenvectors
    omega_td = phonopy_td.mesh.frequencies # THz
    evec_td = phonopy_td.mesh.eigenvectors
    logger.info(f"TD_data = {td_data}")
    logger.info(f"omega_h = {omega_h}")
    logger.info(f"omega_TD = {omega_td}")
    logger.info(f"shape of omega_h = {omega_h.shape}")
    logger.info(f"shape of omega_TD = {omega_td.shape}")
    logger.info(f"evec_h = {evec_h}")
    logger.info(f"evec_td = {evec_td}")
    logger.info(f"phonopy_orig.mesh = {phonopy_orig.mesh}")
    logger.info(f"phonopy_td.mesh = {phonopy_td.mesh}")
    correction_s, correction_sc = free_energy_correction(omega_h,omega_td,evec_h,
                                                         evec_td,[t]) # eV/atom

    td_data["supercell_structure"] = supercell_structure
    td_data["free_energy_correction_S"] = correction_s
    # S = -(dF/dT)_V quasiparticle correction
    td_data["free_energy_correction_SC"] = correction_sc
    # SCPH 4th-order correction (perturbation theory)
    td_data["fcp"] = fcp_td
    td_data["fcs"] = fcs_td
    td_data["param"] = param_td
    td_data["cs"] = cs

    return td_data

def thermodynamic_integration_ifc(
    td_data: dict,
    fcs: ForceConstants,
    param: np.ndarray,
    imag_modes_bool: bool = True,
    lambda_array: np.ndarray = None,
    ti_nconfig: int = 3,
) -> dict:
    if lambda_array is None:
        lambda_array = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1])
    supercell_structure = td_data["supercell_structure"]
    cs = td_data["cs"]
    fcs_td = td_data["fcs"]
    param_td = td_data["param"]
    t = td_data["temperature"][0]
    logger.info(f"Temperature = {t}")
    supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)
    renorm = Renormalization(cs, supercell_atoms, param, fcs, t, "least_squares",
                             "rfe", param_td, fcs_td)
    matcov_td, matcov_bo, matcov_tdbo = renorm.born_oppenheimer_qcv(ti_nconfig)
    if not imag_modes_bool:
        lambda_array = np.array([0])
    correction_ti = renorm.thermodynamic_integration(lambda_array, matcov_td, matcov_bo,
                                                     matcov_tdbo, ti_nconfig)
    td_data["free_energy_correction_TI"] = correction_ti
    return td_data

def setup_te_renorm(cs: ClusterSpace,
                    cutoffs: list[float],
                    parent_atoms: Atoms,
                    supercell_atoms: Atoms,
                    param: np.ndarray,
                    dlfrac: np.ndarray,
                    supercell_matrix: list[list[int]]) -> tuple[Atoms, Atoms,
                                                                ClusterSpace, Phonopy,
                                                                ForceConstants, bool]:
    parent_atoms_te = copy(parent_atoms)
    new_cell = Cell(
            np.transpose([parent_atoms_te.get_cell()[:,i]*(
                1+dlfrac[0,i]
                ) for i in range(3)])
            )
    parent_atoms_te.set_cell(new_cell,scale_atoms=True)
    parent_structure_te = AseAtomsAdaptor.get_structure(parent_atoms_te)
    supercell_atoms_te = copy(supercell_atoms)
    new_supercell = Cell(
            np.transpose([supercell_atoms_te.get_cell()[:,i]*(
                1+dlfrac[0,i]
                ) for i in range(3)])
            )
    supercell_atoms_te.set_cell(new_supercell,scale_atoms=True)
    supercell_structure_te = AseAtomsAdaptor.get_structure(supercell_atoms_te)
    count = 0
    failed = False
    cs_te = ClusterSpace(parent_atoms_te,cutoffs,symprec=1e-2,acoustic_sum_rules=True)
    while True:
        count += 1
        if cs_te.n_dofs == cs.n_dofs:
            logger.info("Matching ClusterSpace found.")
            break

        if count>10:
            logger.warning("Could not find ClusterSpace for expanded cell identical"
                           "to the original cluster space!")
            failed = True
            break

        if count==1:
            cutoffs_te = [i*(1+np.linalg.norm(dlfrac)) for i in cutoffs]
        elif cs_te.n_dofs > cs.n_dofs:
            cutoffs_te = [i*0.999 for i in cutoffs_te]
        elif cs_te.n_dofs < cs.n_dofs:
            cutoffs_te = [i*1.001 for i in cutoffs_te]
        cs_te = ClusterSpace(parent_atoms_te,cutoffs_te,symprec=1e-2,
                             acoustic_sum_rules=True)
    if failed:
        return None, None, None, None, None, failed

    fcp_te = ForceConstantPotential(cs_te, param)
    fcs_te = fcp_te.get_force_constants(supercell_atoms_te)
    prim_te_phonopy = PhonopyAtoms(symbols=parent_atoms_te.get_chemical_symbols(),
                                    scaled_positions=parent_atoms_te.get_scaled_positions(),
                                    cell=parent_atoms_te.cell)
    phonopy_te = Phonopy(prim_te_phonopy, supercell_matrix=supercell_matrix,
                            primitive_matrix=None)
    return (parent_structure_te,
            supercell_structure_te,
            cs_te,
            phonopy_te,
            fcs_te,
            failed)

@job
def run_lattice_thermal_conductivity(
    prev_dir_hiphive: str,
    temperature: float | dict,
    renormalized: bool,
    name: str = "Lattice Thermal Conductivity",
    shengbte_control_kwargs: dict | None = None,
    therm_cond_solver: str | None = "almabte"
) -> Response:
    """
    Calculate the lattice thermal conductivity using ShengBTE.

    Args:
        name: Name of this FW.
        prev_calc_dir: Path to a directory containing the force constant
            information. Will override ``parents`` when collecting the force
            constants to run ShengBTE.
        db_file: Path to a db file.
        shengbte_cmd: The name of the shengbte executable to run. Supports
            env_chk.
        renormalized: boolean to denote whether force constants are from
            phonon renormalization (True) or directly from fitting (False)
        temperature: The temperature to calculate the lattice thermal
            conductivity for. Can be given as a single float, or a
            dictionary with the keys "min", "max", "step".
        shengbte_control_kwargs: Options to be included in the ShengBTE
            control file.
        **kwargs: Other kwargs that are passed to Firework.__init__.
    """
    logger.info("We are in Lattice Thermal Conductivity")

    if renormalized:
        if not isinstance(temperature, (float, int)):
            raise TypeError("Temperature must be a float or an integer.")
        name = f"{name} at {temperature}K"

        copy_hiphive_outputs(prev_dir_hiphive)
        with open("structure_data.json") as file:
            structure_data = json.load(file)
            dumpfn(structure_data, "structure_data.json")

    else:
        # Change this later when the renormalization is implemented
        copy_hiphive_outputs(prev_dir_hiphive)
        with open("structure_data.json") as file:
            structure_data = json.load(file)
            dumpfn(structure_data, "structure_data.json")

    logger.info("Hiphive outputs copied to current directory")

    shengbte = run_thermal_cond_solver(
        renormalized=renormalized,
        temperature=temperature,
        control_kwargs=shengbte_control_kwargs,
        prev_dir_hiphive=prev_dir_hiphive,
        therm_cond_solver=therm_cond_solver
    )
    shengbte.update_config({"manager_config": {"_fworker": "gpu_reg_fworker"}})
    shengbte.name += f" {temperature}"
    shengbte.metadata.update(
        {
            "tag": [
                "run_thermal_cond_solver",
            ]
        }
    )

    return Response(addition=shengbte)
