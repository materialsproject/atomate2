"""Jobs for calculating harmonic & anharmonic props of phonon using hiPhive."""

# Basic Python packages
# Joblib parallelization
# ASE packages
from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
from copy import copy
from itertools import product
from os.path import expandvars
from typing import TYPE_CHECKING, Any

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
from hiphive.structure_generation.random_displacement import (
    generate_displaced_structures,
)
from hiphive.structure_generation.rattle import (
    generate_mc_rattled_structures,
    generate_rattled_structures,
)
from hiphive.utilities import get_displacements

# Jobflow packages
from jobflow import Flow, Response, job
from joblib import Parallel, delayed

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
from atomate2.settings import Atomate2Settings
from atomate2.utils.log import initialize_logger
from atomate2.vasp.files import copy_hiphive_outputs
from atomate2.forcefields.jobs import (
    CHGNetStaticMaker,
    ForceFieldStaticMaker,
)
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
# T_KLAT = {"t_min":100,"t_max":1500,"t_step":100} #[i*100 for i in range(0,11)]
# T_KLAT = [300]  # [i*100 for i in range(0,11)]
T_KLAT = {"min":100,"max":1000,"step":100} #[i*100 for i in range(0,11)]
T_THERMAL_CONDUCTIVITY = [0, 100, 200, 300]  # [i*100 for i in range(0,16)]
IMAGINARY_TOL = 0.1  # in THz # changed from 0.025
FIT_METHOD = "rfe"

eV2J = sp.constants.elementary_charge
hbar = sp.constants.hbar # J-s
kB = sp.constants.Boltzmann # J/K

__all__ = [
    "hiphive_static_calcs",
    "generate_hiphive_displacements",
    "quality_control",
    "run_hiphive_individually"
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
                # fixed_displs: list[float] = [0.01, 0.03, 0.08, 0.1],
                loops: int | None = None,
                prev_dir: str | None = None,
                phonon_displacement_maker: BaseVaspMaker | None = None,
                ff_displacement_maker: ForceFieldStaticMaker | None = None,
                supercell_matrix_kwargs: dict[str, Any] | None = None
        ) -> Response:
    """Run the static calculations for hiPhive fitting."""
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
    
    # displacements_md = force_field_md(
    #     supercell_structure,
    #     bulk_mod,
    #     temperature_md,
    #     mpids
    # )
    # md = MolecularDynamics(
    #     atoms=supercell_structure,
    #     model=chgnet,
    #     ensemble="nvt", #nvt
    #     temperature=1000,  # in K
    #     timestep=2,  # in femto-seconds
    #     trajectory=f"md_out_nvt_1000_{mpids[0]}.traj",
    #     logfile=f"md_out_nvt_1000_{mpids[0]}.log",
    #     loginterval=100,
    #     bulk_modulus=bulk_mod,
    # )
    # md.run(1000)  # run a 0.1 ps MD simulation

    displacements = generate_hiphive_displacements(
            structure=structure,
            supercell_matrix=supercell_matrix,
            n_structures=n_structures,
            # fixed_displs=fixed_displs,
            prev_dir=prev_dir,
            loop=loops,
        )
    jobs.append(displacements)

    chgnet_displacement_calcs = run_phonon_displacements(
            displacements=displacements.output,
            structure=structure,
            supercell_matrix=supercell_matrix,
            phonon_maker=ff_displacement_maker,
            prev_dir=prev_dir,
        )
    jobs.append(chgnet_displacement_calcs)
    forces_array = chgnet_displacement_calcs.output["forces"]
    structures_array = chgnet_displacement_calcs.output["structure"]

    json_saver = collect_perturbed_structures(
            structure=structure,
            supercell_matrix=supercell_matrix,
            rattled_structures=structures_array,
            forces=forces_array,
            loop=loops,
        )
    jobs.append(json_saver)

    force_analyzer = high_force_analyzer(
        structure=structure,
        supercell_matrix=supercell_matrix,
        forces_array=forces_array,
        displacements_list=displacements.output
        )
    jobs.append(force_analyzer)

    displacement_updater = quality_control_forces(
        force_analyzer.output[0],
        force_analyzer.output[1],
        structure=structure,
        supercell_matrix=supercell_matrix,
        n_structures=n_structures,
        phonon_maker=ff_displacement_maker,
        prev_dir=prev_dir,
        loop=loops,
        displacements_pymatgen_list=displacements.output,
    )
    jobs.append(displacement_updater)

    vasp_displacement_calcs = run_phonon_displacements(
            displacements=displacement_updater.output,
            structure=structure,
            supercell_matrix=supercell_matrix,
            phonon_maker=phonon_displacement_maker,
            prev_dir=prev_dir,
        )
    jobs.append(vasp_displacement_calcs)
    outputs["forces"] = vasp_displacement_calcs.output["forces"]
    outputs["structure"] = vasp_displacement_calcs.output["structure"]

    json_saver = collect_perturbed_structures(
            structure=structure,
            supercell_matrix=supercell_matrix,
            rattled_structures=outputs["structure"],
            forces=outputs["forces"],
            loop=loops,
        )
    jobs.append(json_saver)
    outputs["all_structures"] = json_saver.output[0]
    outputs["all_forces"] = json_saver.output[1]
    outputs["structure_data"] = json_saver.output[2]
    outputs["current_dir"] = json_saver.output[3]

    return Response(replace=jobs, output=outputs)


def find_smallest_x(n_structures: int) -> int:
    # Starting from x = 3 because x - 2 in the denominator implies x > 2
    x = 3
    while True:
        # Calculate the value of the expression for the current x
        expression_value = (n_structures - 4) / (x - 2)

        # Check if the expression value satisfies the inequality
        if 5 <= expression_value <= 6:
            return math.floor(x)  # Explicitly return the floor of x
        x += 0.1

@job
def generate_hiphive_displacements(
    structure: Structure,
    displacements: list[Structure] | None = None,
    supercell_matrix: list[list[int]] | None = None,
    n_structures: int | None = None,
    fixed_displs: list[float] | None = None,
    prev_dir: str | None = None,
    loop: int | None = None,
    high_force_atoms: list | None = None,
) -> list[Structure]:
    """Job generates the perturbed structures for hiPhive fitting."""
    if fixed_displs is None:
        # # fixed_displs = [0.01, 0.03, 0.08, 0.1] #TODO, this list is used in
        #                                          # the paper
        # smallest_disp = 0.01

        # # dictionary of largest displacements for each period
        # largest_displacement = {1: 0.1, 2: 0.2, 3: 0.2, 4: 0.3, 5: 0.4, 6: 0.5, 7: 0.5}

        # row = int(
        #     np.around(np.array([s.row for s in structure.species]).mean(), 0)
        # )

        # largest_disp = largest_displacement[row]

        # ratio = int(largest_disp/smallest_disp)
        # if 60 >= n_structures >= 4:
        #     logger.info(f"n_structures inside if statement >= 4 is {n_structures}")
        #     factors = np.sqrt(np.linspace(1,ratio**2,n_structures))
        # elif n_structures < 4   :
        #     factors = np.sqrt(np.linspace(1,ratio**2,4))
        # else:
        #     factors = np.sqrt(np.linspace(1,ratio**2,60))
        # amplitudes = (smallest_disp*factors).round(3)
        # logger.info(f"amplitudes = {amplitudes}")
        # fixed_displs = amplitudes.tolist()
        # logger.info(f"list_amplitudes = {fixed_displs}")

        if n_structures < 4:
            # n_structures = 4
            n_structures = n_structures
        else:
            pass
        logger.info(f"n_structures = {n_structures}")
        # # Ask the user for n_structures
        # n_structures = int(input("Enter the value of n_structures: "))

        # # Find the smallest integer x that satisfies the inequality
        # smallest_x = find_smallest_x(n_structures)

        smallest_x = 2

        # Output the result
        logger.info(f"The smallest integer x that satisfies the inequality is: {smallest_x}")

        # Create structures at equal interval between disp = 0.01 & 0.055
        # equal_interval_structures = np.linspace(0.01, 0.055, smallest_x)
        equal_interval_structures = np.linspace(0.001, 0.005, smallest_x)
        # create a new np array with 3 elements equaling 0.03 and 0.08 respectively
        equal_interval_structures = np.array([0.03, 0.08])

        # # Remaining number of structures
        # remaining_structures = n_structures - smallest_x


        # # structure_data = loadfn(f"/Users/HPSahasrabuddhe/Desktop/Acads/3rd_sem/MSE 299/Hiphive_Atomate2_integration/hiphive_4622_VASP_96/launcher_2024-04-01-08-09-15-480067/structure_data_1.json")
        # # structure = structure_data["structure"]

        # largest_period_default = {1: 0.1, 2: 0.2, 3: 0.2, 4: 0.3, 5: 0.4, 6: 0.5, 7: 0.5}

        # row = int(
        #     np.around(np.array([s.row for s in structure.species]).mean(), 0)
        # )
        # logger.info(f"row = {row}")

        # largest = largest_period_default[row]
        # logger.info(f"largest = {largest}")
        # smallest = 0.056

        # ratio = largest/smallest
        # logger.info(f"ratio = {ratio}")
        # logger.info(f"smallest = {smallest}")
        # logger.info(f"largest = {largest}")
        # logger.info(f"remaining_structures = {remaining_structures}")
        # factors = np.sqrt(np.linspace(1,ratio**2,remaining_structures))
        # logger.info(f"factors = {factors}")
        # amplitudes = (smallest*factors).round(3)
        # logger.info(f"amplitudes = {amplitudes}")

        # # Combine the two sets of structures
        # all_amplitudes = np.concatenate((equal_interval_structures, amplitudes))

        # fixed_displs = all_amplitudes.tolist()
        # logger.info(f"all_amplitudes = {all_amplitudes}")

        fixed_displs = equal_interval_structures.tolist()
        logger.info(f"all_amplitudes = {fixed_displs}")

        fixed_displs = [0.01, 0.03, 0.08, 0.1] #TODO, this list is used in the paper
        fixed_displs = [0.01, 0.03, 0.4, 0.6] #TODO, this list is used in the paper
        fixed_displs = [0.2, 0.4, 0.6, 0.8] #TODO, this list is used in the paper
        # fixed_displs = [0.1, 0.3, 0.5, 0.8] #TODO, for mp-23339 NaBrO3
        # fixed_displs = [0.1, 1] #TODO, for mp-23339 NaBrO3
        # fixed_displs = [1] #TODO, for mp-23339 NaBrO3
        # fixed_displs = [0.002, 0.006, 0.016, 0.02] #TODO, this list is used in the paper
        # fixed_displs = [0.02, 0.04, 0.07, 0.09] # adding more displ to see if it helps
        logger.info(f"default fixed_displs = {fixed_displs}")


    logger.info(f"supercell_matrix = {supercell_matrix}")
    # supercell_matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    supercell_structure = SupercellTransformation(
            scaling_matrix=supercell_matrix
            ).apply_transformation(structure)
    logger.info(f"supercell_structure = {supercell_structure}")
    structure_data = {
        "structure": structure,
        "supercell_structure": supercell_structure,
        "supercell_matrix": supercell_matrix,
    }

    dumpfn(structure_data, f"structure_data_{loop}.json")

    # Generate the rattled structures ####
    structures_ase_all = []
    logger.info(f"n_structures = {n_structures}")
    structures_ase_all_prev = []
    # Convert to ASE atoms
    if high_force_atoms is not None:
            logger.info(f"len_high_force_atoms = {len(high_force_atoms)}")
            logger.info(f"high_force_atoms = {high_force_atoms}")
            for pymatgen_structure in displacements:
                atom = AseAtomsAdaptor.get_atoms(pymatgen_structure)
                structures_ase_all_prev.append(atom)
    for i in range(len(fixed_displs)):
        supercell_ase = AseAtomsAdaptor.get_atoms(supercell_structure)
        structures_ase = generate_displaced_structures(
            atoms=supercell_ase, n_structures=n_structures, distance=fixed_displs[i],
            loop=loop, high_force_atoms=high_force_atoms, displ_number=i,
            structures_ase_all_prev=structures_ase_all_prev
        )
        # structures_ase = generate_rattled_structures(
        #     atoms=supercell_ase, n_structures=n_structures, rattle_std=fixed_displs[i]
        # )
        # structures_ase = generate_mc_rattled_structures(
        #     atoms=supercell_ase, n_configs=n_structures, rattle_std=fixed_displs[i], d_min=1.6,
        #     max_disp=0.1
        # )
        structures_ase_all.extend(structures_ase)

    logger.info(f"structures_ase_all: {structures_ase_all}")
    logger.info(f"len(structures_ase_all): {len(structures_ase_all)}")

    # Convert back to pymatgen structure
    structures_pymatgen = []
    for atoms_ase in range(len(structures_ase_all)):
        logger.info(f"atoms: {atoms_ase}")
        logger.info(f"structures_ase_all[atoms]: {structures_ase_all[atoms_ase]}")
        structure_i = AseAtomsAdaptor.get_structure(structures_ase_all[atoms_ase])
        structures_pymatgen.append(structure_i)
    logger.info(f"len(structures_pymatgen): {len(structures_pymatgen)}")
    for i in range(len(structures_pymatgen)):
        structures_pymatgen[i].to(f"POSCAR_{i}", "poscar")

    dumpfn(structures_pymatgen, f"perturbed_structures_{loop}.json")
    return structures_pymatgen

@job
def high_force_analyzer(
    structure,
    supercell_matrix,
    forces_array,
    displacements_list
) -> (list, int):

    # supercell_matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    print(f"supercell_matrix = {supercell_matrix}")
    supercell_structure = SupercellTransformation(
            scaling_matrix=supercell_matrix
            ).apply_transformation(structure)
    logger.info(f"supercell_structure = {supercell_structure}")
    supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)

    # Initialize a list to keep track of atoms with net norm force > 4
    high_force_atoms = []

    # create a for loop to iterate over forces_array and displacements by zip
    for i, (forces, displacements) in enumerate(zip(forces_array, displacements_list)):
    # for i, (forces, displacements) in enumerate(forces_array):
        # Extract the forces data
        # forces = data[0]['data']
        # forces = structure['data']
        logger.info(f"forces = {forces}")

        # displacements_list = []
        atoms = AseAtomsAdaptor.get_atoms(displacements)
        logger.info(f"atoms = {atoms}")
        logger.info(f"supercell_atoms = {supercell_atoms}")
        displacement = get_displacements(atoms, supercell_atoms)
        logger.info(f"displacement inside 1st for loop = {displacement}")
        # displacements_list.append(displacement)
        # dx, dy, dz = displacement

        # Calculate the net norm force for each atom in each structure
        net_norm_forces = []
        for atom_index, (force, d) in enumerate(zip(forces, displacement)):
            fx, fy, fz = force
            # logger.info(f"displacement inside 2nd for loop = {d}")
            dx, dy, dz = d
            net_norm_force = np.sqrt(fx**2 + fy**2 + fz**2)
            net_norm_forces.append(net_norm_force)

            # # Check if the net norm force is greater than 3
            # if net_norm_force > 3:
            #     high_force_atoms.append((i, atom_index, net_norm_force, [dx, dy, dz]))
            # if net_norm_force <= 0.5:
            #     high_force_atoms.append((i, atom_index, net_norm_force, [dx, dy, dz]))
            high_force_atoms.append((i, atom_index, net_norm_force, [dx, dy, dz]))

    dumpfn(high_force_atoms, "high_force_atoms.json")
    return high_force_atoms, len(high_force_atoms)

@job
def quality_control_forces(
    high_force_atoms: list,
    len_high_force_atoms: int,
    structure: Structure,
    supercell_matrix: list[list[int]],
    n_structures: int,
    phonon_maker: BaseVaspMaker,
    prev_dir: str,
    loop: int,
    displacements_pymatgen_list: list[Structure],
    prev_len_high_force_atoms: int = 0,
    iter: int = 0
):
    """
    Check if the desired Test RMSE is achieved.

    If not, then increase the number of structures
    """
    logger.info(f"len_high_force_atoms = {len_high_force_atoms}")
    logger.info(f"prev_len_high_force_atoms = {prev_len_high_force_atoms}")
    logger.info(f"iter = {iter}")

    if (
        (len_high_force_atoms == prev_len_high_force_atoms and
        iter > 5 and # 5
        prev_len_high_force_atoms != 0)
        or len_high_force_atoms == 0
    ):
        return displacements_pymatgen_list

    return Response(
            replace=update_displ_based_on_forces(
                high_force_atoms,
                len_high_force_atoms,
                structure,
                supercell_matrix,
                n_structures,
                phonon_maker,
                prev_dir,
                loop,
                displacements_pymatgen_list,
                prev_len_high_force_atoms,
                iter
            )
        )

@job
def update_displ_based_on_forces(
    high_force_atoms,
    len_high_force_atoms,
    structure,
    supercell_matrix,
    n_structures,
    phonon_maker,
    prev_dir,
    loops,
    displacements_pymatgen_list,
    prev_len_high_force_atoms: int = 0,
    iter: int = 0
):
    prev_len_high_force_atoms = len_high_force_atoms
    iter += 1
    print(f"prev_len_high_force_atoms = {prev_len_high_force_atoms}")
    print(f"iter = {iter}")
    jobs = []
    displacements = generate_hiphive_displacements(
            high_force_atoms=high_force_atoms,
            displacements=displacements_pymatgen_list,
            structure=structure,
            supercell_matrix=supercell_matrix,
            n_structures=n_structures,
            # fixed_displs=fixed_displs,
            prev_dir=prev_dir,
            loop=loops,
        )
    jobs.append(displacements)

    chgnet_displacement_calcs = run_phonon_displacements(
            displacements=displacements.output,
            structure=structure,
            supercell_matrix=supercell_matrix,
            phonon_maker=phonon_maker,
            prev_dir=prev_dir,
        )
    jobs.append(chgnet_displacement_calcs)
    forces_array = chgnet_displacement_calcs.output["forces"]
    structures_array = chgnet_displacement_calcs.output["structure"]

    json_saver = collect_perturbed_structures(
            structure=structure,
            supercell_matrix=supercell_matrix,
            rattled_structures=structures_array,
            forces=forces_array,
            loop=loops,
        )
    jobs.append(json_saver)

    force_analyzer = high_force_analyzer(
        structure=structure,
        supercell_matrix=supercell_matrix,
        forces_array=forces_array,
        displacements_list=displacements.output
        )
    jobs.append(force_analyzer)

    displacement_updater = quality_control_forces(
        force_analyzer.output[0],
        force_analyzer.output[1],
        structure=structure,
        supercell_matrix=supercell_matrix,
        n_structures=n_structures,
        phonon_maker=phonon_maker,
        prev_dir=prev_dir,
        loop=loops,
        displacements_pymatgen_list=displacements.output,
        prev_len_high_force_atoms=prev_len_high_force_atoms,
        iter=iter
    )
    jobs.append(displacement_updater)

    return Response(replace=jobs, output=displacement_updater.output)

@job
def collect_perturbed_structures(
    structure: Structure,
    supercell_matrix: list[list[int]] | None = None,
    rattled_structures: list[Structure] | None = None,
    forces: Any | None = None,
    loop: int | None = None,
    prev_dir_json_saver: str | None = None,
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
    # supercell_matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 3]] # TODO: remove this line later
    supercell = SupercellTransformation(scaling_matrix=supercell_matrix).apply_transformation(structure)
    structure_data = {
        "structure": structure,
        "supercell_structure": supercell,
        "supercell_matrix": supercell_matrix,
    }

    dumpfn(rattled_structures, f"perturbed_structures_{loop}.json")
    dumpfn(forces, f"perturbed_forces_{loop}.json")
    dumpfn(structure_data, f"structure_data_{loop}.json")
    dumpfn(structure, "relaxed_structure.json")

    with open(f"perturbed_structures_{loop}.json") as file:
        all_structures_loop = json.load(file)
        # all_structures_loop = all_structures_loop["structures"]
        dumpfn(all_structures_loop, f"perturbed_structures_{loop}.json")

    with open(f"perturbed_forces_{loop}.json") as file:
        all_forces_loop = json.load(file)

    # Convert list of lists to numpy arrayx
    all_forces_loop = np.array(all_forces_loop)

    output = [
        {
            "@module": "numpy",
            "@class": "array",
            "dtype": str(all_forces_loop.dtype),
            "data": sublist.tolist(),
        }
        for sublist in all_forces_loop
    ]

    # Save the data as a JSON file
    with open(f"perturbed_forces_{loop}_new.json", "w") as f:
        json.dump(output, f)

    with open(f"perturbed_forces_{loop}_new.json") as file:
        all_forces_loop = json.load(file)

    if prev_dir_json_saver is not None:
        copy_hiphive_outputs(prev_dir_json_saver)
        with open(f"perturbed_structures_{loop-1}.json") as file:
            all_structures = json.load(file)

        with open(f"perturbed_forces_{loop-1}_new.json") as file:
            all_forces = json.load(file)

        all_structures.extend(all_structures_loop)

        for sublist in all_forces_loop:
            all_forces.append(sublist)

        dumpfn(all_structures, f"perturbed_structures_{loop}.json")
        dumpfn(all_forces, f"perturbed_forces_{loop}_new.json")
    else:
        all_structures = all_structures_loop
        all_forces = all_forces_loop

    current_dir = os.getcwd()

    return [all_structures, all_forces, structure_data, current_dir]


# @job
# def quality_control(
#     rmse_test: float,
#     n_structures: int,
#     fixed_displs: list[float],
#     loop: int,
#     fit_method: str,
#     disp_cut: float,
#     bulk_modulus: float,
#     temperature_qha: float,
#     mesh_density: float,
#     imaginary_tol: float,
#     prev_dir_json_saver: str,
#     prev_dir: str,
#     supercell_matrix_kwargs: list[list[int]],
# ):
#     """
#     Check if the desired Test RMSE is achieved.

#     If not, then increase the number of structures
#     """
#     if rmse_test > 0.010:
#         return Response(
#             addition=quality_control_job(
#                 rmse_test,
#                 n_structures,
#                 fixed_displs,
#                 loop,
#                 fit_method,
#                 disp_cut,
#                 bulk_modulus,
#                 temperature_qha,
#                 mesh_density,
#                 imaginary_tol,
#                 prev_dir_json_saver,
#                 prev_vasp_dir,
#                 supercell_matrix_kwargs,
#             )
#         )
#     return None


# @job
# def quality_control_job(
#     rmse_test,
#     n_structures: int,
#     fixed_displs: List[float],
#     loop: int,
#     fit_method: str,
#     disp_cut: float,
#     bulk_modulus: float,
#     temperature_qha: float,
#     mesh_density: float,
#     imaginary_tol: float,
#     prev_dir_json_saver: str,
#     prev_vasp_dir: str,
#     supercell_matrix_kwargs: List[List[int]],
# ):
#     """Increases the number of structures if the desired Test RMSE is not achieved."""
#     jobs = []
#     outputs = []

#     # 4. Quality Control Job to check if the desired Test RMSE is achieved,
#     # if not, then increase the number of structures --
#     # Using "addition" feature of jobflow
#     loop += 1
#     n_structures += 1
#     # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/InAs/
#     # block_2023-06-16-04-09-51-792824/launcher_2023-06-23-23-58-57-102993/
#     # launcher_2023-06-23-23-59-34-157381"
#     error_check_job = quality_control(
#         rmse_test=fw_fit_force_constant.output[5],
#         n_structures=n_structures,
#         fixedDispls=fixed_displs,
#         loop=loop,
#         fit_method=fit_method,
#         disp_cut=disp_cut,
#         bulk_modulus=bulk_modulus,
#         temperature_qha=temperature_qha,
#         mesh_density=mesh_density,
#         imaginary_tol=imaginary_tol,
#         prev_dir_json_saver=json_saver.output[3],
#         # prev_dir_json_saver = prev_dir_json_saver,
#         prev_vasp_dir=prev_vasp_dir,
#         supercell_matrix_kwargs=supercell_matrix_kwargs,
#     )
#     error_check_job.name += f" {loop}"
#     jobs.append(error_check_job)
#     outputs.append(error_check_job.output)
#     error_check_job.metadata.update(
#         {
#             "tag": [
#                 f"error_check_job_{loop}",
#                 f"nConfigsPerStd={n_structures}",
#                 f"fixedDispls={fixed_displs}",
#                 f"dispCut={disp_cut}",
#                 f"supercell_matrix_kwargs={supercell_matrix_kwargs}",
#                 f"loop={loop}",
#             ]
#         }
#     )

#     flow = Flow(jobs=jobs, output=outputs)

#     quality_control_job.name = f"quality_control_job {loop}"

#     return Response(addition=flow)
@job
def run_hiphive_individually(
    mpid: str = None,
    cutoffs: list[list] | None = None,
    fit_method: str | None = None,
    disp_cut: float | None = None,
    bulk_modulus: float | None = None,
    temperature_qha: float | None = None,
    imaginary_tol: float | None = None,
    prev_dir_json_saver: str | None = None,
    loop: int | None = None,
) -> None:
    """Run hiPhive with different cutoffs."""
    copy_hiphive_outputs(prev_dir_json_saver)

    structure_data = loadfn(f"structure_data_{loop}.json")

    supercell_structure = structure_data["supercell_structure"]

    if cutoffs is None:
        # cutoffs = get_cutoffs(supercell_structure)
        # cutoffs = [[8.0]]
        cutoffs = [[4, 3, 2.66]]
        cutoffs = [[5, 3.5]]
        cutoffs = [[3, 2.5]]
        cutoffs = [[5, 2.5]]
        # cutoffs = [[5, 2.5]]
        logger.info(f"cutoffs is {cutoffs}")
    else:
        pass

    jobs = []
    outputs: dict[str, list] = {
        "thermal_data": [],
        "anharmonic_data": [],
        "fitting_data": [],
        "param": [],
        "current_dir": []
    }
    outputs_hiphive = []
    for _, i in enumerate(cutoffs):
        logger.info(f"cutoffs is {i}")
        run_hiphive_cutoffs = run_hiphive(
            cutoffs=[i],
            fit_method=fit_method,
            disp_cut=disp_cut,
            bulk_modulus=bulk_modulus,
            temperature_qha=temperature_qha,
            imaginary_tol=imaginary_tol,
            prev_dir_json_saver=prev_dir_json_saver,
            loop=loop,
        )
        run_hiphive_cutoffs.name += f" {loop} {i}"
        run_hiphive_cutoffs.update_config({"manager_config": {"_fworker": "cpu_reg_fworker"}})
        jobs.append(run_hiphive_cutoffs)
        outputs_hiphive.append(run_hiphive_cutoffs.output)
        run_hiphive_cutoffs.metadata.update(
            {
                "tag": [
                    f"mp_id={mpid}",
                    f"cutoffs={i}",
                    f"fit_method={fit_method}",
                    f"disp_cut={disp_cut}",
                    f"bulk_modulus={bulk_modulus}",
                    f"imaginary_tol={imaginary_tol}",
                    f"prev_dir_json_saver={prev_dir_json_saver}",
                ]
            }
        )

    job_collect_hiphive_outputs = collect_hiphive_outputs(
        fit_method=fit_method,
        disp_cut=disp_cut,
        imaginary_tol=imaginary_tol,
        outputs=outputs_hiphive
    )
    job_collect_hiphive_outputs.name += f" {loop} {i}"
    job_collect_hiphive_outputs.update_config({"manager_config": {"_fworker": "gpu_fworker"}})
    jobs.append(job_collect_hiphive_outputs)

    outputs["thermal_data"] = job_collect_hiphive_outputs.output[0]
    outputs["anharmonic_data"] = job_collect_hiphive_outputs.output[1]
    outputs["fitting_data"] = job_collect_hiphive_outputs.output[2]
    outputs["param"] = job_collect_hiphive_outputs.output[3]
    outputs["current_dir"] = job_collect_hiphive_outputs.output[4]

    job_collect_hiphive_outputs.metadata.update(
        {
            "tag": [
                f"mp_id={mpid}",
                f"cutoffs={i}",
                f"fit_method={fit_method}",
                f"disp_cut={disp_cut}",
                f"bulk_modulus={bulk_modulus}",
                f"imaginary_tol={imaginary_tol}",
                f"prev_dir_json_saver={prev_dir_json_saver}",
            ]
        }
    )

    return Response(replace=jobs, output=outputs)

@job
def collect_hiphive_outputs(
    fit_method: str | None = None,
    disp_cut: float | None = None,
    imaginary_tol: float | None = None,
    outputs: list[dict] | None = None,
) -> list :
    logger.info("We are in collect_hiphive_outputs")

    # Initialize best_fit with high initial values for comparison
    fitting_data: dict[str, Any] = {
        "cutoffs": [],
        "rmse_test": [],
        "fit_method": fit_method,
        "disp_cut": disp_cut,
        "imaginary_tol": imaginary_tol,
        "best_cutoff": None,
        "best_rmse": np.inf,
        "n_imaginary": None
    }

    best_fit = {
        "rmse_test": np.inf,
        "directory": None,
    }
    # Assuming outputs_hiphive is a list of dictionaries with the results
    for result in outputs:
        if result is None:
            continue

        # Assuming result is a dictionary with keys: 'cutoffs', 'rmse_test', etc.
        fitting_data["cutoffs"].append(result["fitting_data"]["cutoffs"][0])
        fitting_data["rmse_test"].append(result["fitting_data"]["rmse_test"][0])

        # Update best_fit if the current result has a lower rmse_test
        # Add additional conditions as needed
        if (
            result["fitting_data"]["rmse_test"][0] < best_fit["rmse_test"]
        ):
            best_fit["directory"] = result["current_dir"]
            fitting_data["best_cutoff"] = result["fitting_data"]["cutoffs"][0]
            fitting_data["best_rmse"] = result["fitting_data"]["rmse_test"][0]
            best_fit["rmse_test"] = result["fitting_data"]["rmse_test"][0]
            # following line is commented out only for testing purposes
            fitting_data["n_imaginary"] = result["fitting_data"]["n_imaginary"]
            # following line is commented out only for testing purposes ends

    copy_hiphive_outputs(best_fit["directory"])
    # following line is commented out only for testing purposes
    thermal_data = loadfn("thermal_data.json")
    # thermal_data = {}
    # following line is commented out only for testing purposes ends
    dumpfn(fitting_data, "fitting_data.json")
    param = np.loadtxt("parameters.txt")

    current_dir = os.getcwd()
    logger.info(f"current_dir = {current_dir}")
    return [thermal_data, thermal_data, fitting_data, param, current_dir]

@job
def run_hiphive(
    cutoffs: list[list] | None = None,
    fit_method: str | None = None,
    disp_cut: float | None = None,
    bulk_modulus: float | None = None,
    temperature_qha: float | None = None,
    imaginary_tol: float | None = None,
    prev_dir_json_saver: str | None = None,
    loop: int | None = None,
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
    logger.info(f"prev_dir_json_saver is {prev_dir_json_saver} this is for def run_hiphive")

    copy_hiphive_outputs(prev_dir_json_saver)

    all_structures = loadfn(f"perturbed_structures_{loop}.json")
    # all_structures = loadfn(f"perturbed_structures.json")
    all_forces = loadfn(f"perturbed_forces_{loop}_new.json")
    # all_forces = loadfn(f"perturbed_forces.json")
    structure_data = loadfn(f"structure_data_{loop}.json")
    # structure_data = loadfn(f"structure_data.json")

    parent_structure = structure_data["structure"]
    supercell_structure = structure_data["supercell_structure"]
    supercell_matrix = np.array(structure_data["supercell_matrix"])

    # parent_structure = SpacegroupAnalyzer(parent_structure).find_primitive() #TODO refactor this later

    if cutoffs is None:
        cutoffs = get_cutoffs(supercell_structure)
        logger.info(f"cutoffs is {cutoffs}")
        # # cutoffs = [[6, 3.5, 3]]
        # # cutoffs = [[9.0, 5.5, 4.25]]
        # # cutoffs = [[8.0]]
        # # cutoffs = [[9.0, 6.5, 5]]
        # # cutoffs = [[9.0, 5.5, 3]]
        # # cutoffs = [[9, 9.25, 5]]
        # # cutoffs = [[9, 8.3125, 5]]
        # # cutoffs = [[4, 4, 4]]
        # # cutoffs = [[3, 2.5]]
        # # cutoffs = [[2, 2.7]]
        # # cutoffs = [[6, 4, 2]]
        # # cutoffs = [[1.8635, 4]]
        # # cutoffs = [[2.559, 4]] # results in 2nd, 3rd order -- 124, 7379 DOFs -- imaginary modes = 8
        # # cutoffs = [[2.54, 4]] # results in 2nd, 3rd order -- 106, 7379 DOFs -- imaginary modes = 2
        # # cutoffs = [[2.5, 4]] # results in 2nd, 3rd order -- 88, 7379 DOFs -- imaginary modes = 0
        # # cutoffs = [[2.52, 4]] # results in 2nd, 3rd order -- 97, 7379 DOFs -- imaginary modes = 2
        # # cutoffs = [[2.52, 3]] # results in 2nd, 3rd order -- 97, 857 DOFs -- imaginary modes = 0
        # # cutoffs = [[2.52, 4.6]] # results in 2nd, 3rd order -- 97, 20804 DOFs -- imaginary modes = 2
        # # cutoffs = [[2.5, 4.6]] # results in 2nd, 3rd order -- 88, 20804 DOFs -- imaginary modes = 0
        # # cutoffs = [[2.5, 5.6]] # results in 2nd, 3rd order -- 88, 67511 DOFs -- imaginary modes = ?
        # # cutoffs = [[9.0, 5.5, 3.5]]
        # # cutoffs = [[9, 8.3125, 5]]
        # # cutoffs = [[6, 5, 4]] # results in {2: 36, 3: 263, 4: 327} DOFs -- imaginary modes = 18, rmse = 132meV/atom
        # # cutoffs = [[8, 5, 4]] # results in {2: 79, 3: 263, 4: 327} DOFs -- imaginary modes = 174, rmse = 68.8meV/atom
        # # cutoffs = [[10, 5, 4]] # results in {2: 151, 3: 263, 4: 327} DOFs -- imaginary modes = 150, rmse = 40.5meV/atom
        # # cutoffs = [[10, 7, 4]] # results in {2: 151, 3: 2149, 4: 327} DOFs -- imaginary modes = 150, rmse = 40.1meV/atom
        # # cutoffs = [[10, 7, 5]] # results in {2: 151, 3: 2149, 4: 2876} DOFs -- imaginary modes = 150, rmse = 38.03meV/atom
        # # cutoffs = [[9, 6, 4]] # {2: 102, 3: 775, 4: 428} -- imaginary modes = 114, rmse = 28.9meV/atom -- MgO 1265
        # # cutoffs = [[9, 7, 4.3]] # {2: 102, 3: 2593, 4: 806} -- imaginary modes = 114, rmse = 29.2meV/atom -- MgO 1265
        # # cutoffs = [[9, 6, 4.5]] # {2: 74, 3: 597, 4: 428} -- imaginary modes = 0, rmse = 13.3meV/atom -- CaO 2605
        # # cutoffs = [[10, 6, 4.5]] # {2: 88, 3: 597, 4: 428} -- imaginary modes = 0, rmse = 9.6meV/atom -- CaO 2605
        # # cutoffs = [[10, 7, 5]] # {2: 88, 3: 775, 4: 806} -- imaginary modes = 0, rmse = 9.6meV/atom -- CaO 2605
        # # cutoffs = [[8.0, 6.5, 4.2]] # {2: 201, 3: 2311, 4: 1642}
        # cutoffs = [[8.0, 6, 3]] # Michael's Bi2SeO2 {2: 201, 3: 2311, 4: 1642}
        # cutoffs = [[8.0, 6, 3]] # Leah's Yb10.5MgSb9 {2: 1732, 3: 8119, 4: 27}
        # cutoffs = [[7, 4.5, 3.5]] # Leah's Yb10.5MgSb9 {2: 1298, 3: 1894, 4: 272}
        # cutoffs = [[7, 4.5, 4]] # Leah's Yb10.5MgSb9 {2: 1298, 3: 1894, 4: 272}
        # cutoffs = [[7, 4.5, 3.7]] # Leah's Yb10.5MgSb9 {2: 1298, 3: 1894, 4: 1258}
        # cutoffs = [[5.5, 4, 3.7]] # Leah's Yb10.5MgSb9 {2: 485, 3: 1136, 4: 1258}
        # cutoffs = [[5.5, 4, 3.8]] # Leah's Yb10.5MgSb9 {2: 485, 3: 1136, 4: 1826}
        # cutoffs = [[4.5, 4, 3.7]] # Leah's Yb10.5MgSb9 {2: 324, 3: 1136, 4: 1258}
        # cutoffs = [[3, 2, 1]] # Leah's Yb10MnSb9 
        # cutoffs = [[8, 5, 3]] # Michael's Bi2SeO2 VASP {2: 197, 3: 433, 4: 60}
        # cutoffs = [[9, 6, 4]] # Michael's Bi2SeO2 VASP {2: 308, 3: 1400, 4: 911}
        # cutoffs = [[9, 7, 4.3]] # Michael's Bi2SeO2 VASP {2: 308, 3: 4478, 4: 911}
        # cutoffs = [[9, 6.5, 5]] # Michael's Bi2SeO2 VASP {2: 308, 3: 2570, 4: 3210}
        # cutoffs = [[7, 5, 3]] # Michael's Bi2SeO2 VASP {2: 148, 3: 433, 4: 60}
        # cutoffs = [[7, 6, 4]] # Michael's Bi2SeO2 VASP {2: 148, 3: 1400, 4: 911}
        # cutoffs = [[6, 5, 3]] # Michael's Bi2SeO2 VASP {2: 86, 3: 433, 4: 60}
        # cutoffs = [[4, 5, 3]] # Michael's Bi2SeO2 VASP {2: 36, 3: 433, 4: 60}
        # cutoffs = [[4, 6, 4]] # Michael's Bi2SeO2 VASP {2: 36, 3: 1400, 4: 911}
        # cutoffs = [[9.5, 5, 3]] # Michael's Bi2SeO2 VASP {2: 337, 3: 433, 4: 60}
        # cutoffs = [[9.7646386289637, 5, 3]] # Michael's Bi2SeO2 VASP {2: 351, 3: 433, 4: 60}
        # cutoffs = [[9.7646386289637, 3]] # Michael's Bi2SeO2 VASP {2: 337, 3: 24}
        # # cutoffs = [[10.430239361718353, 3]] # Michael's Bi2SeO2 VASP {2: 436, 3: 24}
        # # cutoffs = [[9.346003080051858, 5, 3]] # Michael's ZrBeSi VASP {2: 307, 3: 42}
        # cutoffs = [[10.16481312891284, 3]] # Michael's Bi2SeO2 VASP {2: 392, 3: 24}
        # cutoffs = [[10, 6, 4]] # mp-1265
        # # cutoffs = [[10, 7, 4]] # mp-1342
        # cutoffs = [[10, 7, 5]] # mp-1342
        # # cutoffs = [[10.16481312891284, 4, 3]] # Michael's Bi2SeO2 VASP {2: 392, 3: 202, 4: 60}
        # # cutoffs = [[10.16481312891284, 5, 4]] # Michael's Bi2SeO2 VASP {2: 392, 3: 433, 4: 911}
        # # cutoffs = [[10.16481312891284, 6, 4.3]] # Michael's Bi2SeO2 VASP {2: 392, 3: 1400, 4: 911}
        # # cutoffs = [[10.16481312891284, 6, 5]] # Michael's Bi2SeO2 VASP {2: 392, 3: 1400, 4: 3210}
        # # cutoffs = [[10.16481312891284, 3]] # Michael's Bi2SeO2 VASP
        # # cutoffs = [[7, 3]] # Michael's Bi2SeO2 VASP {2: 148, 3: 24}
        # # cutoffs = [[6, 3]] # Michael's Bi2SeO2 VASP
        # # cutoffs = [[10, 4]] # BP VASP {2: 159, 3: 155}
        # # cutoffs = [[10, 7, 5]] # GaP VASP
        # # # cutoffs = [[9, 5.5, 4]] #
        # # cutoffs = [[10, 6, 4]] # MgO VASP
        # # cutoffs = [[10, 6, 4]] # MgO VASP
        # # cutoffs = [[6, 3, 2]] # Bi2SeO2 VASP
        # # cutoffs = [[4, 3, 2]] # Bi2SeO2 VASP
        # cutoffs = [[10, 6, 3]] # mp-6270 {2: 361, 3: 1786, 4: 133}
        # cutoffs = [[11, 6, 3.3]] # mp-6270 {2: 474, 3: 1786, 4: 249}
        # # cutoffs = [[11.4, 6, 4]] # mp-6270 {2: 506, 3: 1786, 4: 517} -- leads to imag modes
        # # cutoffs = [[11, 6, 4]] # mp-6270 {2: 474, 3: 1786, 4: 517} -- leads to imag modes
        # cutoffs = [[11, 6, 3.6]] # mp-6270 {2: 474, 3: 1786, 4: 427}
        # # cutoffs = [[11, 6, 3.9]] # mp-6270 {2: 474, 3: 1786, 4: 427}
        # # cutoffs = [[11, 7, 3.6]] # mp-6270 {2: 474, 3: 1786, 4: 427}
        # # cutoffs = [[7, 6, 3.6]] # mp-6270 {2: 129, 3: 1786, 4: 427}
        # # cutoffs = [[11, 5, 3.6]] # mp-6270 {2: 129, 3: 1786, 4: 427}
        # cutoffs = [[10, 5, 3.6]] # mp-23339 {2: 2143, 3: 4247, 4: 1419}
        cutoffs = [[9, 4, 3]] # mp-23339 {2: 1513, 3: 787, 4: 210}
        # cutoffs = [[9, 5, 4]] # mp-23339 {2: 1513, 3: 787, 4: 210}
        # cutoffs = [[7, 4.3, 3.3]] # mp-23339 {2: 745, 3: 1781, 4: 699}
        # cutoffs = [[9, 3, 2.5]] # mp-23339 {2: 1513, 3: 787, 4: 210}
        # cutoffs = [[10, 3]] # mp-23339
        # cutoffs = [[7, 3, 2]] # mp-23339
        # cutoffs = [[9, 3]] # mp-23339
        # cutoffs = [[7, 2]] # mp-23339
        # cutoffs = [[10, 4]] # Michael's Bi2SeO2
        # cutoffs = [[6, 2.3]] # Keerthi's PbZrO3
        # cutoffs = [[10, 7, 5]] # mp-30530 {2: 218, 3: 3769, 4: 3331}
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
    for structure, forces in zip(all_structures, all_forces):
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
        logger.info(f"Standard deviation of displacements while reading individual displacements: "
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
    fitting_data["n_imaginary"] = thermal_data.pop("n_imaginary")
    thermal_data.update(anharmonic_data)
    logger.info("Saving fitting_data")
    dumpfn(fitting_data, "fitting_data.json")
    logger.info("Saving thermal_data")
    dumpfn(thermal_data, "thermal_data.json")

    logger.info("Writing cluster space and force_constants")
    logger.info(f"{type(fcs)}")

    # # following code is commented only for testing puropose
    # if fitting_data["n_imaginary"] == 0:
    # # following code is commented only for testing puropose ends
    # # if True: # change this back to above if statement
    #     logger.info("No imaginary modes! Writing ShengBTE files")
    #     atoms = AseAtomsAdaptor.get_atoms(parent_structure)
    #     ### fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD", atoms, order=3)
    #     fcs.write_to_phonopy("FORCE_CONSTANTS_2ND", format="text")
    #     # following code is commented only for testing puropose
    #     ForceConstants.write_to_phonopy(fcs, "fc2.hdf5", "hdf5")
    #     ForceConstants.write_to_phono3py(fcs, "fc3.hdf5", "hdf5")
    #     ### detour from hdf5
    #     supercell_atoms = phonopy_atoms_to_ase(phonopy.supercell)
    #     FCS = ForceConstants.read_phono3py(supercell_atoms, "fc3.hdf5", order=3)
    #     FCS.write_to_shengBTE("FORCE_CONSTANTS_3RD", atoms, order=3, fc_tol=1e-4)
    #     # following code is commented only for testing puropose ends
    # else:
    #     logger.info("ShengBTE files not written due to imaginary modes.")
    #     logger.info("You may want to perform phonon renormalization.")

    current_dir = os.getcwd()

    outputs: dict[str, list] = {
        "thermal_data": thermal_data, # replace with "thermal_data"
        "anharmonic_data": thermal_data, # replace with "anharmonic_data"
        "fitting_data": fitting_data,
        "param": param,
        "current_dir": current_dir
    }

    return outputs


def get_cutoffs(supercell_structure: Structure) -> list[list[float]]:
    """
    Trial cutoffs based on supercell structure.

    Get a list of trial cutoffs based on a supercell structure for grid search.
    An initial guess for the lower bound of the cutoffs is made based on the
    average period (row) of the elements in the structure, according to:
    ====== === === ===
    .        Cutoff
    ------ -----------
    Period 2ND 3RD 4TH
    ====== === === ===
     1     5.0 3.0 2.5
     2     6.0 3.5 3.0
     3     7.0 4.5 3.5
     4     8.0 5.5 4.0
     5     9.0 6.0 4.5
     6     10.0 6.5 5.0
     7     11.0 7.0 5.5
    ====== === === ===
    The maximum cutoff for each order is determined by the minimum cutoff and
    the following table. A full grid of all possible cutoff combinations is
    generated based on the step size in the table below times a row factor
    ====== ==== =====
    Cutoff Max  Step
    ====== ==== =====
    2ND    +2.0 1.0
    3RD    +1.5 0.75
    4TH    +0.6 0.6
    ====== ==== =====
    Finally, the max cutoff size is determined by the supercell lattice dimensions.
    Cutoffs which result in multiple of the same orbits being populated will be
    discounted.
    Args:
        supercell_structure: A structure.

    Returns
    -------
        A list of trial cutoffs.
    """
    # # indexed as min_cutoffs[order][period]
    # # DO NOT CHANGE unless you know what you are doing
    # min_cutoffs = {
    #     2: {1: 5.0, 2: 6.0, 3: 7.0, 4: 8.0, 5: 9.0, 6: 10.0, 7: 11.0},
    #     3: {1: 3.0, 2: 3.5, 3: 4.5, 4: 5.5, 5: 6.0, 6: 6.5, 7: 7.0},
    #     4: {1: 2.5, 2: 3.0, 3: 3.5, 4: 4.0, 5: 4.5, 6: 5.0, 7: 5.5},
    # }
    # inc = {2: 2, 3: 1.5, 4: 0.6}
    # steps = {2: 1, 3: 0.75, 4: 0.6}

    # row = int(
    #     np.around(np.array([s.row for s in supercell_structure.species]).mean(), 0)
    # )
    # factor = row / 4
    # mins = {2: min_cutoffs[2][row], 3: min_cutoffs[3][row], 4: min_cutoffs[4][row]}

    # range_two = np.arange(
    #     mins[2], mins[2] + factor * (inc[2] + steps[2]), factor * steps[2]
    # )
    # range_three = np.arange(
    #     mins[3], mins[3] + factor * (inc[3] + steps[3]), factor * steps[3]
    # )
    # range_four = np.arange(
    #     mins[4], mins[4] + factor * (inc[4] + steps[4]), factor * steps[4]
    # )

    # cutoffs = np.array(list(map(list, product(range_two, range_three, range_four))))
    # max_cutoff = estimate_maximum_cutoff(AseAtomsAdaptor.get_atoms(supercell_structure))
    # cutoffs[cutoffs > max_cutoff] = max_cutoff
    # logger.info(f"CUTOFFS \n {cutoffs}")
    # logger.info(f"MAX_CUTOFF \n {max_cutoff}")
    # good_cutoffs = np.all(cutoffs < max_cutoff - 0.1, axis=1)
    # logger.info(f"GOOD CUTOFFS \n{good_cutoffs}")



    min_cutoffs = {
        2: {1: 7.0, 2: 8.0, 3: 9.0, 4: 10.0, 5: 11.0, 6: 12.0, 7: 13.0},
        # 2: {1: 5.0, 2: 6.0, 3: 7.0, 4: 8.0, 5: 9.0, 6: 10.0, 7: 11.0},
        # 2: {1: 6.0, 2: 7.0, 3: 8.0, 4: 9.0, 5: 10.0, 6: 11.0, 7: 12.0},
        3: {1: 2.5, 2: 3.0, 3: 4.0, 4: 5.0, 5: 5.5, 6: 6.0, 7: 6.5},
        # 3: {1: 3.0, 2: 3.5, 3: 4.5, 4: 5.5, 5: 6.0, 6: 6.5, 7: 7.0},
        4: {1: 1.5, 2: 2.0, 3: 2.5, 4: 3.0, 5: 3.5, 6: 4.0, 7: 4.5},
        # 4: {1: 2.5, 2: 3.0, 3: 3.5, 4: 4.0, 5: 4.5, 6: 5.0, 7: 5.5},
    }
    inc = {2: 2, 3: 2.0, 4: 1.2}
    steps = {2: 1, 3: 0.75, 4: 0.6}

    row = int(
        np.around(np.array([s.row for s in supercell_structure.species]).mean(), 0)
    )
    logger.info(f"row = {row}")
    factor = row / 4
    logger.info(f"factor = {factor}")

    mins = {2: min_cutoffs[2][row], 3: min_cutoffs[3][row], 4: min_cutoffs[4][row]}

    # create an NDArray of 2nd order cutofss with only one entry -> mins[2], and 3rd and 4th order cutoffs with a range
    # range_two = np.array([mins[2]])
    range_two = np.arange(
        mins[2], mins[2] + factor * (inc[2] + steps[2]), factor * steps[2]
    )
    range_three = np.arange(
        mins[3], mins[3] + factor * (inc[3] + steps[3]), factor * steps[3]
    )
    range_four = np.arange(
        mins[4], mins[4] + factor * (inc[4] + steps[4]), factor * steps[4]
    )

    cutoffs = np.array(list(map(list, product(range_two, range_three, range_four))))
    logger.info(f"cutoffs = {cutoffs}")

    max_cutoff = estimate_maximum_cutoff(AseAtomsAdaptor.get_atoms(supercell_structure))
    cutoffs[cutoffs > max_cutoff] = max_cutoff
    logger.info(f"CUTOFFS \n {cutoffs}")
    logger.info(f"MAX_CUTOFF \n {max_cutoff}")
    good_cutoffs = np.all(cutoffs < max_cutoff - 0.1, axis=1)
    logger.info(f"GOOD CUTOFFS \n{good_cutoffs}")

    cutoffs_used = cutoffs[good_cutoffs].tolist()
    logger.info(f"cutoffs_used = {cutoffs_used}")

    return cutoffs[good_cutoffs].tolist()


def fit_force_constants(
    parent_structure: Structure,
    supercell_matrix: np.ndarray,
    supercell_structure: Structure,
    structures: list[Atoms],
    all_cutoffs: list[list[float]],
    # separate_fit: bool,
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
        disp_cut: if separate_fit true, determines the mean displacement of perturbed
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
        "fit_method": fit_method,
        "disp_cut": disp_cut,
        "imaginary_tol": imaginary_tol,
        #        "max_n_imaginary": max_n_imaginary,
        "best_cutoff": None,
        "best_rmse": np.inf
    }

    best_fit = {
        "n_imaginary": np.inf,
        "rmse_test": np.inf,
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
    # elif fit_method == "lasso":
    #     fit_kwargs['lasso'] = dict(max_iter=1000)
    elif fit_method == "elasticnet":
        # if fit_method in ['elasticnet', 'lasso']:
        fit_kwargs = {"max_iter": 100000}
        # fit_kwargs = {"max_iter": 10000}
        # fit_kwargs = {"tol": 0.001, "max_iter": 200}

    logger.info(f"CPU COUNT: {os.cpu_count()}")

    logger.info("We are starting Joblib_s parallellized jobs")

    cutoff_results = Parallel(n_jobs=min(os.cpu_count(),len(all_cutoffs)),
                              backend="multiprocessing")(delayed(_run_cutoffs)(
        i, cutoffs, n_cutoffs, parent_structure, supercell_structure, structures,
        supercell_matrix, fit_method, disp_cut, fit_kwargs) for i, cutoffs in enumerate(all_cutoffs))

    for result in cutoff_results:
        if result is None:
            print("result is None")
            continue
        print(f"result = {result}")
        fitting_data["cutoffs"].append(result["cutoffs"])
        fitting_data["rmse_test"].append(result["rmse_test"])
        #        fitting_data["n_imaginary"].append(result["n_imaginary"])
        #        fitting_data["min_frequency"].append(result["min_frequency"])

        if (
            result["rmse_test"]
            < best_fit["rmse_test"]
            #            and result["min_frequency"] > -np.abs(max_imaginary_freq)
            #            and result["n_imaginary"] <= max_n_imaginary
            #            and result["n_imaginary"] < best_fit["n_imaginary"]
        ):
            best_fit.update(result)
            fitting_data["best_cutoff"] = result["cutoffs"]
            fitting_data["best_rmse"] = result["rmse_test"]

    logger.info("Finished fitting force constants.")

    return best_fit["force_constants"], best_fit["parameters"], best_fit["cluster_space"], fitting_data, best_fit["force_constants_potential"]

def _run_cutoffs(
    i: int,
    cutoffs: list[float],
    n_cutoffs: int,
    parent_structure: Structure,
    supercell_structure: Structure,
    structures: list[Atoms],
    supercell_matrix: np.ndarray, # shape=(3, 3), dtype='intc', order='C'.,
    fit_method: str,
    disp_cut: float,
    fit_kwargs: dict[str, Any],
) -> dict[str, Any]:
    logger.info(f"Testing cutoffs {i+1} out of {n_cutoffs}: {cutoffs}")
    supercell_atoms = structures[0] #TODO: only for testing purposes
    # supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)
    logger.info(f"supercell_atoms = {supercell_atoms}")

    if not is_cutoff_allowed(supercell_atoms, max(cutoffs)):
        logger.info("Skipping cutoff due as it is not commensurate with supercell size")
        return {}

    cs = ClusterSpace(supercell_atoms, cutoffs, symprec=1e-3, acoustic_sum_rules=True)
    # cs = ClusterSpace(supercell_atoms, cutoffs, symprec=1e-1, acoustic_sum_rules=True)
    logger.debug(cs.__repr__())
    n2nd = cs.get_n_dofs_by_order(2) # change it back to cs.get_n_dofs_by_order(2)
    nall = cs.n_dofs

    logger.info("Fitting harmonic force constants separately")
    separate_fit = True
    logger.info(f"disp_cut = {disp_cut}")
    # commenting it out only for testing purposes
    sc = get_structure_container(
        cs, structures, separate_fit, disp_cut, ncut=n2nd, param2=None
    )
    opt = Optimizer(sc.get_fit_data(), fit_method, [0, n2nd], **fit_kwargs)
    opt.train()
    param_harmonic = opt.parameters  # harmonic force constant parameters
    param_tmp = np.concatenate(
        (param_harmonic, np.zeros(cs.n_dofs - len(param_harmonic)))
    )
    fcp = ForceConstantPotential(cs, param_tmp)
    logger.info(f"supercell atoms = {supercell_atoms}")
    fcs = fcp.get_force_constants(supercell_atoms)
    logger.info("Did you get the large Condition number error?")

    parent_phonopy = get_phonopy_structure(parent_structure)
    phonopy = Phonopy(parent_phonopy, supercell_matrix=supercell_matrix)
    phonopy.primitive.get_number_of_atoms()
    mesh = supercell_matrix.diagonal() * 2
    phonopy.set_force_constants(fcs.get_fc_array(2))
    phonopy.set_mesh(
        mesh, is_eigenvectors=False, is_mesh_symmetry=False
    )  # run_mesh(is_gamma_center=True)
    phonopy.run_mesh(mesh, with_eigenvectors=False, is_mesh_symmetry=False)
    omega = phonopy.mesh.frequencies  # THz
    omega = np.sort(omega.flatten())
    logger.info(f"omega_one_shot_fit = {omega}")
    imaginary = np.any(omega < -1e-3)
    logger.info(f"imaginary_one_shot_fit = {imaginary}")
    # commenting it out only for testing purposes

    # Phonopy's way of calculating phonon frequencies
    structure_phonopy = get_phonopy_structure(parent_structure)
    phonon = Phonopy(structure_phonopy, supercell_matrix=supercell_matrix)
    phonon.set_force_constants(fcs.get_fc_array(2))
    phonon.run_mesh(mesh=100.0, is_mesh_symmetry=False, is_gamma_center=True)
    mesh = phonon.get_mesh_dict()
    omega = mesh["frequencies"]
    omega = np.sort(omega.flatten())
    logger.info(f"omega_phonopy_one_shot_fitting = {omega}")
    imaginary = np.any(omega < -1e-3)
    logger.info(f"imaginary_phonopy_one_shot_fitting = {imaginary}")

    if imaginary:
    # if False:
    # if True: # only for testing purposes
        logger.info(
            "Imaginary modes found! Fitting anharmonic force constants separately"
        )
        sc = get_structure_container(
            cs, structures, separate_fit, disp_cut, ncut=n2nd, param2=param_harmonic
        )
        opt = Optimizer(sc.get_fit_data(), fit_method, [n2nd, nall], **fit_kwargs)
        opt.train()
        param_anharmonic = opt.parameters  # anharmonic force constant parameters

        parameters = np.concatenate((param_harmonic, param_anharmonic))  # combine
        # if nall != len(parameters):
        #     raise ValueError("Length of parameters does not match nall.")
        assert(nall==len(parameters))
        logger.info(f"Training complete for cutoff: {i}, {cutoffs}")

        fcp = ForceConstantPotential(cs, parameters)
        logger.info(f"supercell atoms = {supercell_atoms}")
        fcs = fcp.get_force_constants(supercell_atoms)
        logger.info("Did you get the large Condition number error?")

        parent_phonopy = get_phonopy_structure(parent_structure)
        phonopy = Phonopy(parent_phonopy, supercell_matrix=supercell_matrix)
        phonopy.primitive.get_number_of_atoms()
        mesh = supercell_matrix.diagonal() * 2
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
        imaginary = np.any(omega < -1e-3)
        logger.info(f"imaginary_phonopy_seperate_fit = {imaginary}")

    else:
    # if True:
        logger.info("No imaginary modes! Fitting all force constants in one shot")
        separate_fit = False
        sc = get_structure_container(
            cs, structures, separate_fit, disp_cut=None, ncut=None, param2=None
        )
        opt = Optimizer(sc.get_fit_data(), fit_method, [0, nall], **fit_kwargs)
        opt.train()
        parameters = opt.parameters
        logger.info(f"Training complete for cutoff: {i}, {cutoffs}")

    # # only for test purpose. Remove this later
    # logger.info("No imaginary modes! Fitting all force constants in one shot")
    # separate_fit = False
    # sc = get_structure_container(
    #     cs, structures, separate_fit, disp_cut=None, ncut=None, param2=None
    # )
    # opt = Optimizer(sc.get_fit_data(), fit_method, [0, nall], **fit_kwargs)
    # opt.train()
    # parameters = opt.parameters
    # logger.info(f"Training complete for cutoff: {i}, {cutoffs}")
    # # only for test purpose. Remove this later

    logger.info(f"parameters before enforcing sum rules {parameters}")
    logger.info(f"Memory use: {psutil.virtual_memory().percent} %")
    parameters = enforce_rotational_sum_rules(cs, parameters, ["Huang", "Born-Huang"])
    fcp = ForceConstantPotential(cs, parameters)
    fcs = fcp.get_force_constants(supercell_atoms)
    logger.info(f"FCS generated for cutoff {i}, {cutoffs}")

    try:
        return {
            "cutoffs": cutoffs,
            "rmse_test": opt.rmse_test,
            "cluster_space": sc.cluster_space,
            "parameters": parameters,
            "force_constants": fcs,
            "force_constants_potential": fcp,
        }
    except Exception:
        return {}


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
        # Calculate mean displacements
        mean_displacements = np.linalg.norm(displacements, axis=1).mean()
        logger.info(f"Mean displacements: {mean_displacements}")
        # Calculate standard deviation of displacements
        std_displacements = np.linalg.norm(displacements, axis=1).std()
        logger.info(f"Standard deviation of displacements: "
                    f"{std_displacements}")
        if not separate_fit:  # fit all
            sc.add_structure(structure)
        else:  # fit separately
            if param2 is None:  # for harmonic fitting
                if mean_displacements < disp_cut:
                    logger.info("We are in harmonic fitting if statement")
                    logger.info(f"mean_disp = {mean_displacements}")
                    sc.add_structure(structure)
            else:  # for anharmonic fitting
                if mean_displacements >= disp_cut:
                    logger.info("We are in anharmonic fitting if statement")
                    logger.info(f"mean_disp = {mean_displacements}")
                    sc.add_structure(structure)
                    saved_structures.append(structure)

    logger.info(f"final shape of fit matrix (total # of atoms in all added supercells, n_dofs) = (rows, columns) = {sc.data_shape}")
    logger.info("We have completed adding structures")
    logger.info(sc.get_fit_data())

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
    imaginary_tol: float = IMAGINARY_TOL
) -> tuple[dict,Phonopy]:
    """
    Uses the force constants to extract phonon properties. Used for comparing
    the accuracy of force constant fits.

    Args:
        structure: The parent structure.
        supercell_matrix: The supercell transformation matrix.
        force_constants: The force constants in numpy format.
        imaginary_tol: Tolerance used to decide if a phonon mode is imaginary,
            in THz.

    Returns:
        A tuple of the number of imaginary modes at Gamma, the minimum phonon
        frequency at Gamma, and the free energy, entropy, and heat capacity
    """

    # fcs = ForceConstants.read("force_constants.fcs")

    logger.info('Evaluating harmonic properties...')
    fcs2 = fcs.get_fc_array(2)
    fcs3 = fcs.get_fc_array(3)
    logger.info('fcs2 & fcs3 read...')
    parent_phonopy = get_phonopy_structure(structure)
    phonopy = Phonopy(parent_phonopy, supercell_matrix=supercell_matrix)
    natom = phonopy.primitive.get_number_of_atoms()
    mesh = supercell_matrix.diagonal()*2
    logger.info(f'Mesh: {mesh}')

    phonopy.set_force_constants(fcs2)
    phonopy.set_mesh(mesh,is_eigenvectors=True,is_mesh_symmetry=False) #run_mesh(is_gamma_center=True)
    phonopy.run_thermal_properties(temperatures=temperature)
    logger.info('Thermal properties successfully run!')

    _, free_energy, entropy, heat_capacity = phonopy.get_thermal_properties()
    # free_energy *= 1000/sp.constants.Avogadro/eV2J/natom # kJ/mol to eV/atom
    # entropy *= 1/sp.constants.Avogadro/eV2J/natom # J/K/mol to eV/K/atom
    # heat_capacity *= 1/sp.constants.Avogadro/eV2J/natom # J/K/mol to eV/K/atom
    logger.info(f"Heat_capacity_harmonic_property: {heat_capacity}")
    freq = phonopy.mesh.frequencies # in THz
    logger.info(f'Frequencies: {freq}')
    logger.info(f"freq_flatten = {np.sort(freq.flatten())}")
    # find imaginary modes at gamma
#    phonopy.run_qpoints([0, 0, 0])
#    gamma_eigs = phonopy.get_qpoints_dict()["frequencies"]
    n_imaginary = int(np.sum(freq < -np.abs(imaginary_tol)))
    min_freq = np.min(freq)


    # # Phonopy's way of calculating phonon frequencies
    # # delete this block of code later. Only for testing purposes
    # logger.info(f"Mesh for calculating phonon frequencies using Phonopy's way = {mesh}")
    # mesh = 100
    # logger.info(f"Mesh for calculating phonon frequencies using Phonopy's way = {mesh}")
    # # fcs = ForceConstants.read("force_constants.fcs")
    # fcs2 = fcs.get_fc_array(2)
    # structure_phonopy = get_phonopy_structure(structure)
    # phonon = Phonopy(structure_phonopy, supercell_matrix=supercell_matrix)
    # phonon.set_force_constants(fcs2)
    # phonon.run_mesh(mesh, is_mesh_symmetry=False, is_gamma_center=True)
    # mesh = phonon.get_mesh_dict()

    # omega = mesh["frequencies"]
    # omega = np.sort(omega.flatten())
    # logger.info(f"omega_phonopy_while_harmonic_prop = {omega}")
    # imaginary = np.any(omega < -1e-3)
    # logger.info(f"imaginary_phonopy_while_harmonic_prop = {imaginary}")
    # n_imaginary = int(np.sum(omega < -np.abs(imaginary_tol)))
    # logger.info(f"n_imaginary_phonopy_while_harmonic_prop = {n_imaginary}")

    if n_imaginary == 0:
        logger.info('No imaginary modes!')
    else: # do not calculate these if imaginary modes exist
        logger.warning('Imaginary modes found!')

    # if len(temperature)==1:
    #     temperature = temperature[0]
    #     free_energy = free_energy[0]
    #     entropy = entropy[0]
    #     heat_capacity = heat_capacity[0]
    logger.info(f"Heat_capacity_harmonic_property[0]: {heat_capacity}")
    # temperature = temperature[0]
    return {
        "temperature": temperature,
        "free_energy": free_energy,
        "entropy": entropy,
        "heat_capacity": heat_capacity,
        "n_imaginary": n_imaginary
        }, phonopy #phonon


def anharmonic_properties(
    phonopy: Phonopy,
    fcs: ForceConstants,
    temperature: List,
    heat_capacity: np.ndarray,
    n_imaginary: float,
    bulk_modulus: float = None
) -> Dict:

    if n_imaginary == 0:
        logger.info('Evaluating anharmonic properties...')
        fcs2 = fcs.get_fc_array(2)
        fcs3 = fcs.get_fc_array(3)
        grun, cte, dLfrac = gruneisen(phonopy,fcs2,fcs3,temperature,heat_capacity,bulk_modulus=bulk_modulus)
    else: # do not calculate these if imaginary modes exist
        logger.warning('Gruneisen and thermal expansion cannot be calculated with imaginary modes. All set to 0.')
        grun = np.zeros((len(temperature),3))
        cte = np.zeros((len(temperature),3))
        dLfrac = np.zeros((len(temperature),3))

    return {
        "gruneisen": grun,
        "thermal_expansion": cte,
        "expansion_fraction": dLfrac,
        }


def get_total_grun(
        omega: np.ndarray,
        grun: np.ndarray,
        kweight: np.ndarray,
        T: float
) -> np.ndarray:
    total = 0
    weight = 0
    nptk = omega.shape[0]
    nbands = omega.shape[1]
    omega = abs(omega)*1e12*2*np.pi
    if T==0:
        total = np.zeros((3,3))
        grun_total_diag = np.zeros(3)
    else:
        for i in range(nptk):
            for j in range(nbands):
                x = hbar*omega[i,j]/(2.0*kB*T)
                dBE = (x/np.sinh(x))**2
                weight += dBE*kweight[i]
                total += dBE*kweight[i]*grun[i,j]
        total = total/weight
        grun_total_diag = np.array([total[0,2],total[1,1],total[2,0]])

        def percent_diff(a,b):
            return abs((a-b)/b)
        # This process preserves cell symmetry upon thermal expansion, i.e., it prevents
        # symmetry-identical directions from inadvertently expanding by different ratios
        # when/if the Gruneisen routine returns slightly different ratios for those directions
        avg012 = np.mean((grun_total_diag[0],grun_total_diag[1],grun_total_diag[2]))
        avg01 = np.mean((grun_total_diag[0],grun_total_diag[1]))
        avg02 = np.mean((grun_total_diag[0],grun_total_diag[2]))
        avg12 = np.mean((grun_total_diag[1],grun_total_diag[2]))
        if percent_diff(grun_total_diag[0],avg012) < 0.1:
            if percent_diff(grun_total_diag[1],avg012) < 0.1:
                if percent_diff(grun_total_diag[2],avg012) < 0.1: # all siilar
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
        temperature: List,
        heat_capacity: np.ndarray, # in J/K/mol
        bulk_modulus: float = None # in GPa
) -> Tuple[List,List]:

    gruneisen = Gruneisen(fcs2,fcs3,phonopy.supercell,phonopy.primitive)
    gruneisen.set_sampling_mesh(phonopy.mesh_numbers,is_gamma_center=True)
    gruneisen.run()
    grun = gruneisen.get_gruneisen_parameters() # (nptk,nmode,3,3)
    omega = gruneisen._frequencies
    qp = gruneisen._qpoints
    kweight = gruneisen._weights
    grun_tot = list()
    for temp in temperature:
        grun_tot.append(get_total_grun(omega,grun,kweight,temp))
    grun_tot = np.nan_to_num(np.array(grun_tot))

    # linear thermal expansion coefficeint and fraction
    if bulk_modulus is None:
        cte = None
        dLfrac = None
    else:
        # heat_capacity *= eV2J*phonopy.primitive.get_number_of_atoms() # eV/K/atom to J/K
        heat_capacity *= 1/sp.constants.Avogadro # J/K/mol to J/K  # to convert from J/K/atom multiply by phonopy.primitive.get_number_of_atoms()
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
        cte = grun_tot*heat_capacity.repeat(3).reshape(len(heat_capacity),3)/(vol/10**30)/(bulk_modulus*10**9)/3
        cte = np.nan_to_num(cte)
        if len(temperature)==1:
            dLfrac = cte*temperature
        else:
            dLfrac = thermal_expansion(temperature, cte)
        logger.info(f"Gruneisen: \n {grun_tot}")
        logger.info(f"Coefficient of Thermal Expansion: \n {cte}")
        logger.info(f"Linear Expansion Fraction: \n {dLfrac}")

    return grun_tot, cte, dLfrac


def thermal_expansion(
        temperature: list,
        cte: np.array,
) -> np.ndarray:
    assert len(temperature)==len(cte)
    if 0 not in temperature:
        temperature = [0] + temperature
        cte = np.array([np.array([0,0,0])] + list(cte))
    temperature = np.array(temperature)
    ind = np.argsort(temperature)
    temperature = temperature[ind]
    cte = np.array(cte)[ind]
    # linear expansion fraction
    dLfrac = copy(cte)
    for t in range(len(temperature)):
        dLfrac[t,:] = np.trapz(cte[:t+1,:],temperature[:t+1],axis=0)
    dLfrac = np.nan_to_num(dLfrac)
    return dLfrac

@job
def run_thermal_cond_solver(
    renormalized: bool | None = None,
    temperature: list[int] | None = None,
    control_kwargs: dict | None = None,
    prev_dir_hiphive: str | None = None,
    loop: int | None = None,
    therm_cond_solver: str | None = "almabte"
) -> None:
    """
    Thermal conductivity calculation using ShengBTE.

    Run ShengBTE to calculate lattice thermal conductivity. Presumes
    the FORCE_CONSTANTS_3RD and FORCE_CONSTANTS_2ND, and a "structure_data.json"
    file, with the keys "structure", " and "supercell_matrix" is in the current
    directory.
    Required parameters:
        shengbte_cmd (str): The name of the shengbte executable to run. Supports
            env_chk.
    Optional parameters:
        renormalized: boolean to denote whether force constants are from
            phonon renormalization (True) or directly from fitting (False)
        temperature (float or dict): The temperature to calculate the lattice
            thermal conductivity for. Can be given as a single float, or a
            dictionary with the keys "t_min", "t_max", "t_step".
        control_kwargs (dict): Options to be included in the ShengBTE control
            file.
    """
    if therm_cond_solver == "almabte":
        therm_cond_solver_cmd = Atomate2Settings().ALMABTE_CMD
    elif therm_cond_solver == "shengbte":
        therm_cond_solver_cmd = Atomate2Settings().SHENGBTE_CMD
    elif therm_cond_solver == "phono3py":
        therm_cond_solver_cmd = Atomate2Settings().PHONO3PY_CMD

    logger.info(f"therm_cond_solver_cmd = {therm_cond_solver_cmd}")
    therm_cond_solver_cmd = expandvars(therm_cond_solver_cmd)

    logger.info(f"Running {therm_cond_solver_cmd} command")

    copy_hiphive_outputs(prev_dir_hiphive)
    with open(f"structure_data_{loop}.json") as file:
        structure_data = json.load(file)
        dumpfn(structure_data, "structure_data.json")

    structure_data = loadfn("structure_data.json")
    structure = structure_data["structure"]
    supercell_matrix = structure_data["supercell_matrix"]

    structure = SpacegroupAnalyzer(structure).find_primitive() #TODO refactor this later

    logger.info(f"Temperature = {temperature}")

    temperature = temperature if temperature is not None else T_KLAT
    logger.info(f"Temperature = {temperature}")
    logger.info(f"type of temperature = {type(temperature)}")

    renormalized = renormalized if renormalized is not None else False

    if renormalized:
        assert isinstance(temperature, (int, float))
    else:
        if isinstance(temperature, (int, float)):
            pass
        elif isinstance(temperature, dict):
            temperature["min"]
            temperature["max"]
            temperature["step"]
        else:
            raise ValueError("Unsupported temperature type, must be float or dict")

    logger.info("Creating control dict")

    control_dict = {
        "scalebroad": 0.5,
        # "scalebroad": 1.1,
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
        therm_cond_solver_cmd = os.path.expandvars(therm_cond_solver_cmd)
        therm_cond_solver_cmd = shlex.split(therm_cond_solver_cmd)

    therm_cond_solver_cmd = list(therm_cond_solver_cmd)

    with open("shengbte.out", "w") as f_std, open(
        "shengbte_err.txt", "w", buffering=1
    ) as f_err:
        # use line buffering for stderr
        return_code = subprocess.call(therm_cond_solver_cmd, stdout=f_std, stderr=f_err)
    logger.info(
        f"Command {therm_cond_solver_cmd} finished running with returncode: {return_code}"
    )

    # logger.info(f"Running command: {shengbte_cmd}")
    # return_code = subprocess.call(shengbte_cmd, shell=True)  # noqa: S602
    # logger.info(f"{shengbte_cmd} finished running with returncode: {return_code}")

    if return_code == 1:
        raise RuntimeError(
            f"Running ShengBTE failed. Check '{os.getcwd()}/shengbte_err.txt' for "
            "details."
        )


@job
def run_fc_to_pdos(
    renormalized: bool | None = None,
    # renorm_temperature: str | None = None,
    mesh_density: float | None = None,
    prev_dir_json_saver: str | None = None,
    loop: int | None = None,
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
    logger.info(f"loop = {loop}")

    renormalized = renormalized if renormalized else False
    # renorm_temperature = renorm_temperature if renorm_temperature else None
    mesh_density = mesh_density if mesh_density else 100.0

    structure_data = loadfn(f"structure_data_{loop}.json")
    # structure_data = loadfn(f"structure_data.json")
    structure = structure_data["structure"]
    structure_data["supercell_structure"]
    supercell_matrix = structure_data["supercell_matrix"]

    if not renormalized:
        loadfn(f"perturbed_structures_{loop}.json")
        loadfn(f"perturbed_forces_{loop}_new.json")
        # fitting_data = loadfn("fitting_data.json")
        # thermal_data = loadfn("thermal_data.json")
        fcs = ForceConstants.read("force_constants.fcs")

        uniform_bs, lm_bs, dos = _get_fc_fsid(
            structure, supercell_matrix, fcs, mesh_density
        )

        logger.info("Finished inserting force constants and phonon data")

    else:
        # following line is commented out only for testing purposes
        renorm_thermal_data = loadfn("thermal_data.json") # renorm_thermal_data.json
        fcs = ForceConstants.read("force_constants.fcs")
        T = renorm_thermal_data["temperature"]

        # dos_fsid, uniform_bs_fsid, lm_bs_fsid, fc_fsid = _get_fc_fsid(
        #     structure, supercell_matrix, fcs, mesh_density, mmdb
        # )

        uniform_bs, lm_bs, dos = _get_fc_fsid(
            structure, supercell_matrix, fcs, mesh_density
        )

        logger.info(
            f"Finished inserting renormalized force constants and phonon data at {T} K"
        )
        # following line is commented out only for testing purposes ends
        # pass

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
    renorm_TE_iter: bool,
    bulk_modulus: float,
    prev_dir_hiphive: str,
    loop: int,
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
    structure_data = loadfn(f"structure_data_{loop}.json")
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
    TD_data = run_renormalization(parent_structure, supercell_structure, supercell_matrix,
                                      cs, fcs, param, temperature, nconfig, renorm_method,
                                      fit_method, bulk_modulus, phonopy_orig)
    TD_structure_data = copy(structure_data)
    TD_structure_data["structure"] = parent_structure
    TD_structure_data["supercell_structure"] = supercell_structure

    logger.info("Renormalization is now completeed")
    # Additional renormalization with thermal expansion -
    # optional - just single "iteration" for now
    if renorm_TE_iter:
        n_TE_iter = 1
        for i in range(n_TE_iter):
            if TD_data is None or TD_data["n_imaginary"] < 0:
                # failed, incomplete, or still imaginary
                break
            logger.info(
                f"Renormalizing with thermally expanded lattice - iteration {i}"
            )

            dLfrac = TD_data["expansion_fraction"]
            param_TD = TD_data["param"]

            a, b, c, d, e, failed = setup_TE_renorm(
                    cs,cutoffs,parent_atoms,supercell_atoms,param_TD,dLfrac,supercell_matrix
                )
            if not failed:
                parent_structure_TD, supercell_structure_TD, cs_TD, phonopy_TD, fcs_TD  = a, b, c, d, e
                TD_data = run_renormalization(parent_structure_TD, supercell_structure_TD, supercell_matrix,
                                                cs_TD, fcs, param, temperature, nconfig,
                                                renorm_method, fit_method, bulk_modulus,
                                                phonopy_TD, param_TD, fcs_TD
                                                )
                TD_structure_data["structure"] = parent_structure_TD
                TD_structure_data["supercell_structure"] = supercell_structure_TD

    # # Thermodynamic integration for anharmonic free energy
    # TD_data = thermodynamic_integration_ifc(
    #     TD_data, # everything TD
    #     fcs, # original
    #     param, # original
    #     imag_modes_bool, # if False, only uses lambda=0
    #     )
    # write results
    logger.info("Writing renormalized results")
    fcs_TD = TD_data['fcs']
    fcs_TD.write("force_constants.fcs")
    # if "n_imaginary" in TD_data:
    if TD_data["n_imaginary"] != 0:
        # thermal_keys = ["temperature","free_energy","entropy","heat_capacity",
        #             "free_energy_correction_S","free_energy_correction_SC",
        #             "free_energy_correction_TI"]
        thermal_keys = ["temperature","free_energy","entropy","heat_capacity",
                    "free_energy_correction_S","free_energy_correction_SC"]
    else:
        # thermal_keys = ["temperature","free_energy","entropy","heat_capacity",
        #             "gruneisen","thermal_expansion","expansion_fraction",
        #             "free_energy_correction_S","free_energy_correction_SC",
        #             "free_energy_correction_TI"]
        thermal_keys = ["temperature","free_energy","entropy","heat_capacity",
                    "gruneisen","thermal_expansion","expansion_fraction",
                    "free_energy_correction_S","free_energy_correction_SC"]
    TD_thermal_data = {key: [] for key in thermal_keys}
    for key in thermal_keys:
        TD_thermal_data[key].append(TD_data[key])

    logger.info("DEBUG: ",TD_data)
    if TD_data["n_imaginary"] > 0:
        logger.warning('Imaginary modes remain still exist')
        logger.warning('ShengBTE FORCE_CONSTANTS_2ND & FORCE_CONSTANTS_3RD not written')
    else:
        logger.info("No imaginary modes! Writing ShengBTE files")

        # parent_atoms_TD = copy(parent_atoms)
        # logger.info(f"TD_data exp frac: {TD_data['expansion_fraction']}")
        # logger.info(f"TD_data exp frac 0: {TD_data['expansion_fraction'][0,0]}")
        # logger.info(f"TD_data exp frac 0: {TD_data['expansion_fraction'][0,1]}")
        # logger.info(f"TD_data exp frac 0: {TD_data['expansion_fraction'][0,2]}")
        # new_cell = Cell(np.transpose([parent_atoms_TD.get_cell()[:,i]*(1+TD_data["expansion_fraction"][0,i]) for i in range(3)]))
        # parent_atoms_TD.set_cell(new_cell,scale_atoms=True)

        # prim_TD_phonopy = PhonopyAtoms(symbols=parent_atoms_TD.get_chemical_symbols(),
        #                                 scaled_positions=parent_atoms_TD.get_scaled_positions(),
        #                                 cell=parent_atoms_TD.cell)
        # phonopy_TD = Phonopy(prim_TD_phonopy, supercell_matrix=supercell_matrix, primitive_matrix=None)

        # atoms = AseAtomsAdaptor.get_atoms(parent_structure_TD)
        # # fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD", atoms, order=3)
        # fcs_TD.write_to_phonopy(f"FORCE_CONSTANTS_2ND_{temperature}", format="text")
        # ForceConstants.write_to_phonopy(fcs_TD, "fc2.hdf5", "hdf5")
        # ForceConstants.write_to_phono3py(fcs_TD, "fc3.hdf5", "hdf5")
        # ### detour from hdf5
        # supercell_atoms = phonopy_atoms_to_ase(phonopy_TD.supercell)
        # FCS = ForceConstants.read_phono3py(supercell_atoms, "fc3.hdf5", order=3)
        # FCS.write_to_shengBTE("FORCE_CONSTANTS_3RD_{temperature}", atoms, order=3, fc_tol=1e-4)


        atoms = AseAtomsAdaptor.get_atoms(parent_structure)
        supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)
        # fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD", atoms, order=3)
        fcs_TD.write_to_phonopy("FORCE_CONSTANTS_2ND", format="text")
        ForceConstants.write_to_phonopy(fcs_TD, "fc2.hdf5", "hdf5")
        ForceConstants.write_to_phono3py(fcs_TD, "fc3.hdf5", "hdf5")
        ### detour from hdf5
        # supercell_atoms = phonopy_atoms_to_ase(phonopy.supercell)
        FCS = ForceConstants.read_phono3py(supercell_atoms, "fc3.hdf5", order=3)
        FCS.write_to_shengBTE("FORCE_CONSTANTS_3RD", atoms, order=3, fc_tol=1e-4)

    dumpfn(TD_structure_data, "structure_data.json")
    dumpfn(TD_thermal_data, "thermal_data.json")

    current_dir = os.getcwd()

    return [current_dir, TD_thermal_data]


def run_renormalization(
    structure: Structure,
    supercell_structure: Structure,
    supercell_matrix: np.ndarray,
    cs: ClusterSpace,
    fcs: ForceConstants,
    param: np.ndarray,
    T: float,
    nconfig: int,
    renorm_method: str,
    fit_method: str,
    bulk_modulus: float = None,
    phonopy_orig: Phonopy = None,
    param_TD: np.ndarray = None,
    fcs_TD: ForceConstants = None,
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
    nconfig = int(nconfig)
    supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)
    renorm = Renormalization(cs,supercell_atoms,param,fcs,T,renorm_method,fit_method,param_TD=param_TD,fcs_TD=fcs_TD)
    fcp_TD, fcs_TD, param_TD = renorm.renormalize(nconfig)#,conv_tresh)

    TD_data, phonopy_TD = harmonic_properties(
        structure, supercell_matrix, fcs_TD, [T], imaginary_tol
    )
    logger.info(f"Heat capacity_TD_DATA: {TD_data['heat_capacity']}")
    if TD_data["n_imaginary"] == 0:
        logger.info(f'Renormalized phonon is completely real at T = {T} K!')
        anharmonic_data = anharmonic_properties(
            phonopy_TD, fcs_TD, [T], TD_data["heat_capacity"], TD_data["n_imaginary"], bulk_modulus=bulk_modulus
        )
        TD_data.update(anharmonic_data)

    # phonopy_orig.run_mesh()
    # phonopy_TD.run_mesh()
    mesh = supercell_matrix.diagonal()*2
    phonopy_orig.set_mesh(mesh,is_eigenvectors=True,is_mesh_symmetry=False)
    phonopy_TD.set_mesh(mesh,is_eigenvectors=True,is_mesh_symmetry=False)
    omega_h = phonopy_orig.mesh.frequencies # THz
    evec_h = phonopy_orig.mesh.eigenvectors
    omega_TD = phonopy_TD.mesh.frequencies # THz
    evec_TD = phonopy_TD.mesh.eigenvectors
    logger.info(f'TD_data = {TD_data}')
    logger.info(f'omega_h = {omega_h}')
    logger.info(f'omega_TD = {omega_TD}')
    logger.info(f'shape of omega_h = {omega_h.shape}')
    logger.info(f'shape of omega_TD = {omega_TD.shape}') 
    logger.info(f'evec_h = {evec_h}')
    logger.info(f'evec_TD = {evec_TD}')
    logger.info(f"phonopy_orig.mesh = {phonopy_orig.mesh}")
    logger.info(f"phonopy_TD.mesh = {phonopy_TD.mesh}")
    correction_S, correction_SC = free_energy_correction(omega_h,omega_TD,evec_h,evec_TD,[T]) # eV/atom

    TD_data["supercell_structure"] = supercell_structure
    TD_data["free_energy_correction_S"] = correction_S   # S = -(dF/dT)_V quasiparticle correction
    TD_data["free_energy_correction_SC"] = correction_SC # SCPH 4th-order correction (perturbation theory)
    TD_data["fcp"] = fcp_TD
    TD_data["fcs"] = fcs_TD
    TD_data["param"] = param_TD
    TD_data['cs'] = cs

    return TD_data

def thermodynamic_integration_ifc(
    TD_data: dict,
    fcs: ForceConstants,
    param: np.ndarray,
    imag_modes_bool: bool = True,
    lambda_array: np.ndarray = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]),
    TI_nconfig=3,
) -> dict:
    supercell_structure = TD_data["supercell_structure"]
    cs = TD_data['cs']
    fcs_TD = TD_data["fcs"]
    param_TD = TD_data["param"]
    T = TD_data['temperature'][0]
    logger.info(f"Temperature = {T}")
    supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)
    renorm = Renormalization(cs, supercell_atoms, param, fcs, T, 'least_squares', 'rfe', param_TD, fcs_TD)
    matcov_TD, matcov_BO, matcov_TDBO = renorm.born_oppenheimer_qcv(TI_nconfig)
    if not imag_modes_bool:
        lambda_array = np.array([0])
    correction_TI = renorm.thermodynamic_integration(lambda_array, matcov_TD, matcov_BO, matcov_TDBO, TI_nconfig)
    TD_data["free_energy_correction_TI"] = correction_TI
    return TD_data

def setup_TE_renorm(cs,cutoffs,parent_atoms,supercell_atoms,param,dLfrac,supercell_matrix):
    parent_atoms_TE = copy(parent_atoms)
    new_cell = Cell(np.transpose([parent_atoms_TE.get_cell()[:,i]*(1+dLfrac[0,i]) for i in range(3)]))
    parent_atoms_TE.set_cell(new_cell,scale_atoms=True)
    parent_structure_TE = AseAtomsAdaptor.get_structure(parent_atoms_TE)
    supercell_atoms_TE = copy(supercell_atoms)
    new_supercell = Cell(np.transpose([supercell_atoms_TE.get_cell()[:,i]*(1+dLfrac[0,i]) for i in range(3)]))
    supercell_atoms_TE.set_cell(new_supercell,scale_atoms=True)
    supercell_structure_TE = AseAtomsAdaptor.get_structure(supercell_atoms_TE)
    count = 0
    failed = False
    cs_TE = ClusterSpace(parent_atoms_TE,cutoffs,symprec=1e-2,acoustic_sum_rules=True)
    while True:
        count += 1
        if cs_TE.n_dofs == cs.n_dofs:
            break
        elif count>10:
            logger.warning("Could not find ClusterSpace for expanded cell identical to the original cluster space!")
            failed = True
            break
        elif count==1:
            cutoffs_TE = [i*(1+np.linalg.norm(dLfrac)) for i in cutoffs]
        elif cs_TE.n_dofs > cs.n_dofs:
            cutoffs_TE = [i*0.999 for i in cutoffs_TE]
        elif cs_TE.n_dofs < cs.n_dofs:
            cutoffs_TE = [i*1.001 for i in cutoffs_TE]
        cs_TE = ClusterSpace(parent_atoms_TE,cutoffs_TE,symprec=1e-2,acoustic_sum_rules=True)
    if failed:
        return None, None, None, None, None, failed
    else:
        fcp_TE = ForceConstantPotential(cs_TE, param)
        fcs_TE = fcp_TE.get_force_constants(supercell_atoms_TE)
        prim_TE_phonopy = PhonopyAtoms(symbols=parent_atoms_TE.get_chemical_symbols(),
                                       scaled_positions=parent_atoms_TE.get_scaled_positions(),
                                       cell=parent_atoms_TE.cell)
        phonopy_TE = Phonopy(prim_TE_phonopy, supercell_matrix=supercell_matrix, primitive_matrix=None)
        return parent_structure_TE, supercell_structure_TE, cs_TE, phonopy_TE, fcs_TE, failed


@job
def run_lattice_thermal_conductivity(
    prev_dir_hiphive: str,
    loop: int,
    temperature: float | dict,
    renormalized: bool,
    name: str = "Lattice Thermal Conductivity",
    # prev_calc_dir: Optional[str] = None,
    # db_file: str = None,
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
    # files needed to run ShengBTE

    logger.info("We are in Lattice Thermal Conductivity... 1")
    logger.info(f"previ_dir_hiphive in def run_lattice_thermal_conductivity = {prev_dir_hiphive}")


    if renormalized:
        assert type(temperature) in [float, int]
        name = f"{name} at {temperature}K"

        copy_hiphive_outputs(prev_dir_hiphive)
        with open(f"structure_data_{loop}.json") as file:
            structure_data = json.load(file)
            dumpfn(structure_data, "structure_data.json")

    else:
        # Change this later when the renormalization is implemented
        copy_hiphive_outputs(prev_dir_hiphive)
        with open(f"structure_data_{loop}.json") as file:
            structure_data = json.load(file)
            dumpfn(structure_data, "structure_data.json")

    logger.info("We are in Lattice Thermal Conductivity... 2")

    shengbte = run_thermal_cond_solver(
        renormalized=renormalized,
        temperature=temperature,
        control_kwargs=shengbte_control_kwargs,
        prev_dir_hiphive=prev_dir_hiphive,
        loop=loop,
        therm_cond_solver=therm_cond_solver
    )
    shengbte.update_config({"manager_config": {"_fworker": "gpu_reg_fworker"}}) #change to gpu_fworker
    shengbte.name += f" {temperature} {loop}"
    shengbte.metadata.update(
        {
            "tag": [
                f"run_thermal_cond_solver_{loop}",
                f"loop={loop}",
            ]
        }
    )

    return Response(addition=shengbte)
