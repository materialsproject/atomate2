"""Jobs for calculating harmonic & anharmonic props of phonon using hiPhive."""

# Basic Python packages
# Joblib parallelization
# ASE packages
import contextlib
import csv
import json
import logging
import os
import shlex
import subprocess
import time
from copy import copy
from dataclasses import field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import phonopy as phpy
import psutil
import scipy as sp
from ase import atoms
from ase.atoms import Atoms
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
from hiphive.utilities import get_displacements

# Jobflow packages
from jobflow import Flow, Response, job

# Pymatgen packages
from monty.serialization import dumpfn, loadfn
from phono3py.phonon3.gruneisen import Gruneisen

# Phonopy & Phono3py
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import (
    get_phonon_band_structure_from_fc,
    get_phonon_band_structure_symm_line_from_fc,
    get_phonon_dos_from_fc,
    get_phonopy_structure,
)
from pymatgen.io.shengbte import Control
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)
from pymatgen.transformations.standard_transformations import SupercellTransformation

from atomate2.vasp.files import copy_hiphive_outputs

# Atomate2 packages
# from atomate2.vasp.sets.core import MPStaticSetGenerator
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import (
    StaticMaker,
)

logger = logging.getLogger(__name__)

T_QHA = [
    i * 100 for i in range(21)
]  # Temp. for straight-up phonopy calculation of thermo. props. (free energy etc.)
T_RENORM = [
    1500
]  # [i*100 for i in range(0,16)] # Temp. at which renorm. is to be performed
# Temperature at which lattice thermal conductivity is calculated
# If renorm. is performed, T_RENORM overrides T_KLAT for lattice thermal conductivity
# T_KLAT = {"t_min":100,"t_max":1500,"t_step":100} #[i*100 for i in range(0,11)]
T_KLAT = 300  # [i*100 for i in range(0,11)]
T_THERMAL_CONDUCTIVITY = [0, 100, 200, 300]  # [i*100 for i in range(0,16)]
IMAGINARY_TOL = 0.025  # in THz
FIT_METHOD = "rfe"

eV2J = sp.constants.elementary_charge
hbar = sp.constants.hbar  # J-s
kB = sp.constants.Boltzmann  # J/K

__all__ = [
    "run_static_calculations",
    "quality_control",
    "run_hiphive",
    "run_shengbte",
    "run_fc_to_pdos",
    "run_hiphive_renormalization",
    "run_lattice_thermal_conductivity",
]

__author__ = "Alex Ganose, Junsoo Park, Zhuoying Zhu, Hrushikesh Sahasrabuddhe"
__email__ = "aganose@lbl.gov, jsyony37@lbl.gov, zyzhu@lbl.gov, hpsahasrabuddhe@lbl.gov"


@job
def run_static_calculations(
    supercell_matrix_kwargs: Dict[str, Any],
    n_structures: int,
    rattle_std: List[float],
    loop: int,
    prev_vasp_dir: Union[str, Path, None],
    # MPstatic_maker: BaseVaspMaker = field(default_factory=MPStaticMaker),
    # MPstatic_maker: BaseVaspMaker = field(default_factory=StaticMaker),
    MPstatic_maker: BaseVaspMaker = None,
    structure: Optional[Structure] = None,
    prev_dir_json_saver: Optional[str] = None,
):
    """
    Run static calculations.

    Note, this job will replace itself with N static calculations, where N is
    the number of deformations.

    Args:
        rattled_structures : list of Structures
            A pymatgen Structure object.
        prev_vasp_dir : str or Path or None
            A previous VASP directory to use for copying VASP outputs.
        MPstatic_maker : BaseVaspMaker
            A VaspMaker object to use for generating VASP inputs.

    Returns
    -------
        Response
    """
    # Generate the supercell ####
    if prev_dir_json_saver is not None:
        copy_hiphive_outputs(prev_dir_json_saver)
        structure = loadfn("relaxed_structure.json")
    else:
        pass

    if supercell_matrix_kwargs is not None:
        min_atoms = supercell_matrix_kwargs["min_atoms"]
        max_atoms = supercell_matrix_kwargs["max_atoms"]
        min_length = supercell_matrix_kwargs["min_length"]
        force_diagonal = supercell_matrix_kwargs["force_diagonal"]
        supercell_structure = CubicSupercellTransformation(
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_length=min_length,
            force_diagonal=force_diagonal,
        ).apply_transformation(structure)
        print(f"supercell_structure: {supercell_structure}")
    else:
        q = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
        # q = [[2, 0, 0],[0, 2, 0],[0, 0, 2]]
        supercell_structure = SupercellTransformation(
            scaling_matrix=q
        ).apply_transformation(structure)
        print(f"supercell_structure: {supercell_structure}")

    # Generate the rattled structures ####
    structures_ase_all = []
    # Convert to ASE atoms
    for i in range(len(rattle_std)):
        supercell_ase = AseAtomsAdaptor.get_atoms(supercell_structure)
        structures_ase = generate_displaced_structures(
            supercell_ase, n_structures, rattle_std[i], loop
        )
        # for j in range(len(structures_ase)):
        #     structures_ase_all.append(structures_ase[j])
        # structures_ase_all = list(structures_ase)
        structures_ase_all.append(structures_ase)
    # supercell_ase = AseAtomsAdaptor.get_atoms(supercell_structure)
    # structures_ase =
    # generate_mc_rattled_structures(supercell_ase, 3, 0.01, d_min=3.294)
    # structures_ase_all.append(structures_ase)

    print(f"structures_ase_all: {structures_ase_all}")
    # Convert back to pymatgen structure
    structures_pymatgen = []
    for atoms_ase in range(len(structures_ase_all)):
        print(f"atoms: {atoms_ase}")
        print(f"structures_ase_all[atoms]: {structures_ase_all[atoms_ase][0]}")
        structure_i = AseAtomsAdaptor.get_structure(structures_ase_all[atoms_ase])
        structures_pymatgen.append(structure_i)

    for i in range(len(structures_pymatgen)):
        structures_pymatgen[i].to(f"POSCAR_{i}", "poscar")

    # Run the static calculations ####
    all_jobs = []
    outputs: dict[str, list] = {
        "forces": [],
        "structures": [],
    }

    if MPstatic_maker is None:
        MPstatic_maker = StaticMaker()  # Assign the default value here

    for i, structure in enumerate(structures_pymatgen):
        print(structure)
        static = MPstatic_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
        static.name += f" {loop*3 - 1 - i}"
        all_jobs.append(static)
        outputs["forces"].append(static.output.output.forces)
        outputs["structures"].append(static.output.output.structure)

    static_flow = Flow(jobs=all_jobs, output=outputs)

    return Response(replace=static_flow)


@job
def collect_perturbed_structures(
    supercell_matrix_kwargs: Dict[str, Any],
    rattled_structures: list[Structure],
    forces: list[list[float]],
    loop: int = None,
    structure: Optional[Structure] = None,
    prev_dir_json_saver: Optional[str] = None,
):
    """
    Aggregate the structures and forces of perturbed supercells.

    Args:
        structure (Structure): input structure
        supercell (Structure): supercell structure
        supercell_matrix (list[list[int]]): supercell matrix
        rattled_structures (list[Structure]): list of Structures
        perturbed_tasks (int): number of perturbed tasks
    Returns:
        None.
    """
    if prev_dir_json_saver is not None:
        copy_hiphive_outputs(prev_dir_json_saver)
        structure = loadfn("relaxed_structure.json")
        # supercell = SupercellTransformation(scaling_matrix=supercell_matrix).
        # apply_transformation(structure)

        min_atoms = supercell_matrix_kwargs["min_atoms"]
        max_atoms = supercell_matrix_kwargs["max_atoms"]
        min_length = supercell_matrix_kwargs["min_length"]
        force_diagonal = supercell_matrix_kwargs["force_diagonal"]
        supercell = CubicSupercellTransformation(
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_length=min_length,
            force_diagonal=force_diagonal,
        ).apply_transformation(structure)
        # supercell_matrix = [[5, 0, 0],[0, 5, 0],[0, 0, 5]]
        # supercell_matrix = [[6, 0, 0],[0, 6, 0],[0, 0, 6]]
        supercell_matrix = []
        for i in range(3):
            lattice_vector_supercell = supercell.lattice.abc[i]
            lattice_vector_prim_cell = structure.lattice.abc[i]
            if i == 0:
                supercell_matrix.append(
                    [
                        np.round(lattice_vector_supercell / lattice_vector_prim_cell),
                        0,
                        0,
                    ]
                )
            if i == 1:
                supercell_matrix.append(
                    [
                        0,
                        np.round(lattice_vector_supercell / lattice_vector_prim_cell),
                        0,
                    ]
                )
            if i == 2:
                supercell_matrix.append(
                    [
                        0,
                        0,
                        np.round(lattice_vector_supercell / lattice_vector_prim_cell),
                    ]
                )

        structure_data = {
            "structure": structure,
            "supercell_structure": supercell,
            "supercell_matrix": supercell_matrix,
        }
    else:
        # supercell = SupercellTransformation(scaling_matrix=supercell_matrix).
        # apply_transformation(structure)

        min_atoms = supercell_matrix_kwargs["min_atoms"]
        max_atoms = supercell_matrix_kwargs["max_atoms"]
        min_length = supercell_matrix_kwargs["min_length"]
        force_diagonal = supercell_matrix_kwargs["force_diagonal"]
        supercell = CubicSupercellTransformation(
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_length=min_length,
            force_diagonal=force_diagonal,
        ).apply_transformation(structure)
        # supercell_matrix = [[5, 0, 0],[0, 5, 0],[0, 0, 5]]
        # supercell_matrix = [[6, 0, 0],[0, 6, 0],[0, 0, 6]]
        supercell_matrix = []
        for i in range(3):
            lattice_vector_supercell = supercell.lattice.abc[i]
            lattice_vector_prim_cell = structure.lattice.abc[i]
            if i == 0:
                supercell_matrix.append(
                    [
                        np.round(lattice_vector_supercell / lattice_vector_prim_cell),
                        0,
                        0,
                    ]
                )
            if i == 1:
                supercell_matrix.append(
                    [
                        0,
                        np.round(lattice_vector_supercell / lattice_vector_prim_cell),
                        0,
                    ]
                )
            if i == 2:
                supercell_matrix.append(
                    [
                        0,
                        0,
                        np.round(lattice_vector_supercell / lattice_vector_prim_cell),
                    ]
                )

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
        all_structures_loop = all_structures_loop["structures"]
        dumpfn(all_structures_loop, f"perturbed_structures_{loop}.json")

    with open(f"perturbed_forces_{loop}.json") as file:
        all_forces_loop = json.load(file)

    structure_data = loadfn(f"structure_data_{loop}.json")

    # Convert list of lists to numpy arrayx
    all_forces_loop = np.array(all_forces_loop["forces"])

    # output = []
    # for sublist in all_forces_loop:
    #     output.append(
    #         {
    #             "@module": "numpy",
    #             "@class": "array",
    #             "dtype": str(all_forces_loop.dtype),
    #             "data": sublist.tolist(),
    #         }
    #     )

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

    # all_forces_loop = loadfn(f"perturbed_forces_{loop}_new.json")
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


@job
def quality_control(
    rmse_test: float,
    n_structures: int,
    rattle_std: List[float],
    loop: int,
    fit_method: str,
    disp_cut: float,
    bulk_modulus: float,
    temperature_qha: float,
    mesh_density: float,
    imaginary_tol: float,
    prev_dir_json_saver: str,
    prev_vasp_dir: str,
    supercell_matrix_kwargs: List[List[int]],
):
    """
    Check if the desired Test RMSE is achieved.

    If not, then increase the number of structures
    """
    if rmse_test > 0.010:
        return Response(
            addition=quality_control_job(
                rmse_test,
                n_structures,
                rattle_std,
                loop,
                fit_method,
                disp_cut,
                bulk_modulus,
                temperature_qha,
                mesh_density,
                imaginary_tol,
                prev_dir_json_saver,
                prev_vasp_dir,
                supercell_matrix_kwargs,
            )
        )
    return None


@job
def quality_control_job(
    rmse_test,
    n_structures: int,
    rattle_std: List[float],
    loop: int,
    fit_method: str,
    disp_cut: float,
    bulk_modulus: float,
    temperature_qha: float,
    mesh_density: float,
    imaginary_tol: float,
    prev_dir_json_saver: str,
    prev_vasp_dir: str,
    supercell_matrix_kwargs: List[List[int]],
):
    """Increases the number of structures if the desired Test RMSE is not achieved."""
    jobs = []
    outputs = []

    # my_custom_set = MPStaticSetGenerator(user_incar_settings={"ISMEAR": 1})
    # MPstatic_maker = MPStaticMaker(input_set_generator=my_custom_set)
    MPstatic_maker: BaseVaspMaker = field(default_factory=StaticMaker)

    # 1. Generates supercell, performs fixed displ rattling and runs
    # static calculations
    vasp_static_calcs = run_static_calculations(
        prev_dir_json_saver=prev_dir_json_saver,
        supercell_matrix_kwargs=supercell_matrix_kwargs,
        n_structures=n_structures,
        rattle_std=rattle_std,
        loop=loop,
        prev_vasp_dir=prev_vasp_dir,
        MPstatic_maker=MPstatic_maker,
    )
    jobs.append(vasp_static_calcs)
    outputs.append(vasp_static_calcs.output)
    vasp_static_calcs.metadata.update(
        {
            "tag": [
                f"vasp_static_calcs_{loop}",
                f"nConfigsPerStd={n_structures}",
                f"rattleStds={rattle_std}",
                f"dispCut={disp_cut}",
                f"supercell_matrix_kwargs={supercell_matrix_kwargs}",
                f"loop={loop}",
            ]
        }
    )

    # 2.  Save "structure_data_{loop}.json", "relaxed_structures.json",
    # "perturbed_forces_{loop}.json" and "perturbed_structures_{loop}.json"
    # files locally
    json_saver = collect_perturbed_structures(
        supercell_matrix_kwargs=supercell_matrix_kwargs,
        rattled_structures=vasp_static_calcs.output,
        forces=vasp_static_calcs.output,
        prev_dir_json_saver=prev_dir_json_saver,
        loop=loop,
    )
    json_saver.name += f" {loop}"
    jobs.append(json_saver)
    outputs.append(json_saver.output)
    json_saver.metadata.update(
        {
            "tag": [
                f"json_saver_{loop}",
                f"nConfigsPerStd={n_structures}",
                f"rattleStds={rattle_std}",
                f"dispCut={disp_cut}",
                f"supercell_matrix_kwargs={supercell_matrix_kwargs}",
                f"loop={loop}",
            ]
        }
    )

    # 3. Hiphive Fitting of FCPs upto 4th order
    fw_fit_force_constant = run_hiphive(
        fit_method=fit_method,
        disp_cut=disp_cut,
        bulk_modulus=bulk_modulus,
        temperature_qha=temperature_qha,
        mesh_density=mesh_density,
        imaginary_tol=imaginary_tol,
        prev_dir_json_saver=json_saver.output[3],
        loop=loop,
    )
    fw_fit_force_constant.name += f" {loop}"
    jobs.append(fw_fit_force_constant)
    outputs.append(fw_fit_force_constant.output)
    fw_fit_force_constant.metadata.update(
        {
            "tag": [
                f"fw_fit_force_constant_{loop}",
                f"nConfigsPerStd={n_structures}",
                f"rattleStds={rattle_std}",
                f"dispCut={disp_cut}",
                f"supercell_matrix_kwargs={supercell_matrix_kwargs}",
                f"loop={loop}",
            ]
        }
    )

    # 4. Quality Control Job to check if the desired Test RMSE is achieved,
    # if not, then increase the number of structures --
    # Using "addition" feature of jobflow
    loop += 1
    n_structures += 1
    # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/InAs/
    # block_2023-06-16-04-09-51-792824/launcher_2023-06-23-23-58-57-102993/
    # launcher_2023-06-23-23-59-34-157381"
    error_check_job = quality_control(
        rmse_test=fw_fit_force_constant.output[5],
        n_structures=n_structures,
        rattle_std=rattle_std,
        loop=loop,
        fit_method=fit_method,
        disp_cut=disp_cut,
        bulk_modulus=bulk_modulus,
        temperature_qha=temperature_qha,
        mesh_density=mesh_density,
        imaginary_tol=imaginary_tol,
        prev_dir_json_saver=json_saver.output[3],
        # prev_dir_json_saver = prev_dir_json_saver,
        prev_vasp_dir=prev_vasp_dir,
        supercell_matrix_kwargs=supercell_matrix_kwargs,
    )
    error_check_job.name += f" {loop}"
    jobs.append(error_check_job)
    outputs.append(error_check_job.output)
    error_check_job.metadata.update(
        {
            "tag": [
                f"error_check_job_{loop}",
                f"nConfigsPerStd={n_structures}",
                f"rattleStds={rattle_std}",
                f"dispCut={disp_cut}",
                f"supercell_matrix_kwargs={supercell_matrix_kwargs}",
                f"loop={loop}",
            ]
        }
    )

    flow = Flow(jobs=jobs, output=outputs)

    quality_control_job.name = f"quality_control_job {loop}"

    return Response(addition=flow)


@job
# @explicit_serialize
def run_hiphive(
    cutoffs: Optional[list[list]] = None,
    fit_method: str = None,
    disp_cut: float = None,
    bulk_modulus: float = None,
    temperature_qha: float = None,
    mesh_density: float = None,
    imaginary_tol: float = None,
    prev_dir_json_saver: str = None,
    loop: int = None,
):
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
    copy_hiphive_outputs(prev_dir_json_saver)

    all_structures = loadfn(f"perturbed_structures_{loop}.json")
    all_forces = loadfn(f"perturbed_forces_{loop}_new.json")
    structure_data = loadfn(f"structure_data_{loop}.json")

    parent_structure = structure_data["structure"]
    supercell_structure = structure_data["supercell_structure"]
    supercell_matrix = np.array(structure_data["supercell_matrix"])

    # disp_cut = disp_cut

    # cutoffs = [[8.0, 5.5, 4.0], [8.0, 5.5, 4.6]]
    # cutoffs = [[3.69276149408161, 3.69276149408161, 3.69276149408161], [3, 3, 3]]
    # cutoffs = [[9.0, 6.25, 4.0], [9.0, 7.0, 4.0]]
    if cutoffs is None:
        cutoffs = get_cutoffs(supercell_structure)
        print(f"cutoffs is {cutoffs}")
        # cutoffs = cutoffs[0:-21]
        # print(f'new cutoffs is {cutoffs}')
    else:
        # cutoffs = cutoffs
        pass

    T_qha = temperature_qha if temperature_qha is not None else T_QHA
    if isinstance(T_qha, list):
        T_qha.sort()
    else:
        # Handle the case where T_qha is not a list
        # You can choose to raise an exception or handle it differently
        # For example, if T_qha is a single temperature value, you can
        # convert it to a list
        T_qha = [T_qha]
    # T_qha.sort()
    # imaginary_tol = imaginary_tol
    # bulk_modulus = bulk_modulus
    # fit_method = fit_method

    structures = []
    print(f"supercell_structure is {supercell_structure}")
    supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)
    for structure, forces in zip(all_structures, all_forces):
        print(f"structure is {structure}")
        atoms = AseAtomsAdaptor.get_atoms(structure)
        displacements = get_displacements(atoms, supercell_atoms)
        atoms.new_array("displacements", displacements)
        atoms.new_array("forces", forces)
        atoms.positions = supercell_atoms.get_positions()
        structures.append(atoms)

    logger.info(f'forces in 0th structure are {structures[0].get_array("forces")}')
    logger.info(
        f'displacements in 0th structure are {structures[0].get_array("displacements")}'
    )

    logger.info(f'forces in 1st structure are {structures[1].get_array("forces")}')
    logger.info(
        f'displacements in 1st structure are {structures[1].get_array("displacements")}'
    )

    logger.info(f'forces in 2nd structure are {structures[2].get_array("forces")}')
    logger.info(
        f'displacements in 2nd structure are {structures[2].get_array("displacements")}'
    )

    all_cutoffs = cutoffs
    fcs, param, cs, fitting_data, fcp, rmse_test = fit_force_constants(
        parent_structure=parent_structure,
        supercell_matrix=supercell_matrix,
        structures=structures,
        all_cutoffs=all_cutoffs,
        disp_cut=disp_cut,
        imaginary_tol=imaginary_tol,
        fit_method=fit_method,
    )

    if fcs is None:
        raise RuntimeError("Could not find a force constant solution")

    thermal_data, phonopy = harmonic_properties(
        parent_structure, supercell_matrix, fcs, T_qha, imaginary_tol
    )

    anharmonic_data = anharmonic_properties(
        phonopy,
        fcs,
        T_qha,
        thermal_data["heat_capacity"],
        thermal_data["n_imaginary"],
        bulk_modulus,
    )

    phonopy.save("phonopy_params.yaml")
    fitting_data["n_imaginary"] = thermal_data.pop("n_imaginary")
    thermal_data.update(anharmonic_data)
    dumpfn(fitting_data, "fitting_data.json")
    dumpfn(thermal_data, "thermal_data.json")

    logger.info("Writing cluster space and force_constants")
    logger.info(f"{type(fcs)}")
    # fcp.write("force_constants.fcp")
    if isinstance(fcp, ForceConstantPotential):
        fcp.write("force_constants.fcp")
    else:
        print("fcp is not an instance of ForceConstantPotential")
    # fcs.write("force_constants.fcs")
    if isinstance(fcs, ForceConstants):
        fcs.write("force_constants.fcs")
    else:
        print("fcs is not an instance of ForceConstants")
    np.savetxt("parameters.txt", param)
    # cs.write("cluster_space.cs")
    if isinstance(cs, ClusterSpace):
        cs.write("cluster_space.cs")
    else:
        print("cs is not an instance of ClusterSpace")

    if fitting_data["n_imaginary"] == 0:
        logger.info("No imaginary modes! Writing ShengBTE files")
        atoms = AseAtomsAdaptor.get_atoms(parent_structure)
        fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD", atoms, order=3)
        fcs.write_to_phonopy("FORCE_CONSTANTS_2ND", format="text")
        ForceConstants.write_to_phonopy(fcs, "fc2.hdf5", "hdf5")
        ForceConstants.write_to_phono3py(fcs, "fc3.hdf5", 3)
    else:
        logger.info("ShengBTE files not written due to imaginary modes.")
        logger.info("You may want to perform phonon renormalization.")

    current_dir = os.getcwd()

    return [thermal_data, anharmonic_data, fitting_data, param, current_dir, rmse_test]


def get_cutoffs(supercell_structure: Structure):
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
    # indexed as min_cutoffs[order][period]
    # DO NOT CHANGE unless you know what you are doing
    min_cutoffs = {
        2: {1: 5.0, 2: 6.0, 3: 7.0, 4: 8.0, 5: 9.0, 6: 10.0, 7: 11.0},
        3: {1: 3.0, 2: 3.5, 3: 4.5, 4: 5.5, 5: 6.0, 6: 6.5, 7: 7.0},
        4: {1: 2.5, 2: 3.0, 3: 3.5, 4: 4.0, 5: 4.5, 6: 5.0, 7: 5.5},
    }
    inc = {2: 2, 3: 1.5, 4: 0.6}
    steps = {2: 1, 3: 0.75, 4: 0.6}

    row = int(
        np.around(np.array([s.row for s in supercell_structure.species]).mean(), 0)
    )
    factor = row / 4
    mins = {2: min_cutoffs[2][row], 3: min_cutoffs[3][row], 4: min_cutoffs[4][row]}

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
    max_cutoff = estimate_maximum_cutoff(AseAtomsAdaptor.get_atoms(supercell_structure))
    cutoffs[cutoffs > max_cutoff] = max_cutoff
    logger.info(f"CUTOFFS \n {cutoffs}")
    logger.info(f"MAX_CUTOFF \n {max_cutoff}")
    good_cutoffs = np.all(cutoffs < max_cutoff - 0.1, axis=1)
    logger.info(f"GOOD CUTOFFS \n{good_cutoffs}")
    return cutoffs[good_cutoffs].tolist()


def fit_force_constants(
    parent_structure: Structure,
    supercell_matrix: np.ndarray,
    structures: List["Atoms"],
    all_cutoffs: List[List[float]],
    # separate_fit: bool,
    disp_cut: float = 0.055,
    imaginary_tol: float = IMAGINARY_TOL,
    fit_method: str = FIT_METHOD,
    n_jobs: int = -1,
    fit_kwargs: Optional[Dict] = None,
) -> Tuple:
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

    disp_cut = 0.055
    print(f"disp_cut={disp_cut}")
    fit_method = "rfe"
    # fit_method = "omp"
    print(f"fit_method={fit_method}")

    fitting_data: Dict[str, Any] = {
        "cutoffs": [],
        "rmse_test": [],
        "fit_method": fit_method,
        "disp_cut": disp_cut,
        "imaginary_tol": imaginary_tol,
        #        "max_n_imaginary": max_n_imaginary,
        "best": None,
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
    n_cutoffs = len(all_cutoffs)
    print(f"len_cutoffs={n_cutoffs}")

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

    # With Joblib's parallellization
    # cutoff_results = Parallel(n_jobs=-1, backend="multiprocessing")
    # (delayed(_run_cutoffs)(i, cutoffs, n_cutoffs, parent_structure, structures,
    # supercell_matrix, fit_method,disp_cut, imaginary_tol, fit_kwargs) for
    # i, cutoffs in enumerate(all_cutoffs))
    # cutoff_results = Parallel(
    #     n_jobs=min(os.cpu_count(), len(all_cutoffs)), backend="multiprocessing"
    # )(
    #     delayed(_run_cutoffs)(
    #         i,
    #         cutoffs,
    #         n_cutoffs,
    #         parent_structure,
    #         structures,
    #         supercell_matrix,
    #         fit_method,
    #         disp_cut,
    #         imaginary_tol,
    #         fit_kwargs,
    #     )
    #     for i, cutoffs in enumerate(all_cutoffs)
    # )

    # logger.info(f"CUTOFF RESULTS \n {cutoff_results}")

    # for result in cutoff_results:
    #     if result is None:
    #         continue

    #     fitting_data["cutoffs"].append(result["cutoffs"])
    #     fitting_data["rmse_test"].append(result["rmse_test"])
    #     #        fitting_data["n_imaginary"].append(result["n_imaginary"])
    #     #        fitting_data["min_frequency"].append(result["min_frequency"])

    #     if (
    #         result["rmse_test"]
    #         < best_fit["rmse_test"]
    #         #            and result["min_frequency"] > -np.abs(max_imaginary_freq)
    #         #            and result["n_imaginary"] <= max_n_imaginary
    #         #            and result["n_imaginary"] < best_fit["n_imaginary"]
    #     ):
    #         best_fit.update(result)
    #         fitting_data["best"] = result["cutoffs"]

    # logger.info("Finished fitting force constants.")

    # print(f'all_cutoffs={all_cutoffs}')

    # Without Joblib's parallellization
    for i, cutoffs in enumerate(all_cutoffs):
        print(f"disp_cut={disp_cut}")
        print(f"fit_method={fit_method}")
        print(f"cuttoffs={cutoffs}")
        print(f"imaginary_tol={imaginary_tol}")
        print(f"n_cutoffs={n_cutoffs}")
        print(f"parent_structure={parent_structure}")
        print(f"structures={structures}")
        print(f"supercell_matrix={supercell_matrix}")

        start_time = time.time()
        cutoff_results = _run_cutoffs(
            i,
            cutoffs,
            n_cutoffs,
            parent_structure,
            structures,
            supercell_matrix,
            fit_method,
            disp_cut,
            imaginary_tol,
            fit_kwargs,
        )
        time_taken = time.time() - start_time
        logger.info(f"Time taken for cutoffs {cutoffs} is {time_taken} seconds")

        print(f"cutoff_results={cutoff_results}")
        print(f"time taken={time_taken}")
        print(f'cutoffs={cutoff_results["cutoffs"]}')
        print(f'rmse_test={cutoff_results["rmse_test"]}')
        print(f'parameters={cutoff_results["parameters"]}')

        with open(f"timings_{cutoffs}_{fit_method}.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([cutoffs, time_taken, cutoff_results["rmse_test"]])

        cutoff_results["force_constants_potential"].write(
            f"force_constants_potential_{cutoffs}_{fit_method}.fcp"
        )
        cutoff_results["force_constants"].write(
            f"force_constants_potential_{cutoffs}_{fit_method}.fcs"
        )

        if cutoff_results["rmse_test"] < best_fit["rmse_test"]:
            best_fit["rmse_test"] = cutoff_results["rmse_test"]
            best_fit["cluster_space"] = cutoff_results["cluster_space"]
            best_fit["force_constants"] = cutoff_results["force_constants"]
            best_fit["parameters"] = cutoff_results["parameters"]
            best_fit["cutoffs"] = cutoff_results["cutoffs"]
            best_fit["force_constants_potential"] = cutoff_results[
                "force_constants_potential"
            ]
            fitting_data["best"] = cutoff_results["cutoffs"]

        fitting_data["cutoffs"].append(cutoff_results["cutoffs"])
        fitting_data["rmse_test"].append(cutoff_results["rmse_test"])

    logger.info(f"CUTOFF RESULTS \n {cutoff_results}")

    return (
        best_fit["force_constants"],
        best_fit["parameters"],
        best_fit["cluster_space"],
        fitting_data,
        best_fit["force_constants_potential"],
        best_fit["rmse_test"],
    )


def _run_cutoffs(
    i,
    cutoffs,
    n_cutoffs,
    parent_structure,
    structures,
    supercell_matrix,
    fit_method,
    # separate_fit,
    disp_cut,
    imaginary_tol,
    fit_kwargs,
) -> Dict[str, Any]:
    logger.info(f"Testing cutoffs {i+1} out of {n_cutoffs}: {cutoffs}")
    supercell_atoms = structures[0]

    if not is_cutoff_allowed(supercell_atoms, max(cutoffs)):
        logger.info("Skipping cutoff due as it is not commensurate with supercell size")
        return None

    cs = ClusterSpace(supercell_atoms, cutoffs, symprec=1e-3, acoustic_sum_rules=True)
    logger.debug(cs.__repr__())
    n2nd = cs.get_n_dofs_by_order(2)
    nall = cs.n_dofs

    logger.info("Fitting harmonic force constants separately")
    separate_fit = True
    logger.info(f"disp_cut = {disp_cut}")
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
    print(f"omega = {omega}")
    imaginary = np.any(omega < -1e-3)
    print(f"imaginary = {imaginary}")

    if imaginary:
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
        assert nall == len(parameters)
        logger.info(f"Training complete for cutoff: {i}, {cutoffs}")

    else:
        logger.info("No imaginary modes! Fitting all force constants in one shot")
        separate_fit = False
        sc = get_structure_container(
            cs, structures, separate_fit, disp_cut=None, ncut=None, param2=None
        )
        opt = Optimizer(sc.get_fit_data(), fit_method, [0, nall], **fit_kwargs)
        opt.train()
        parameters = opt.parameters
        logger.info(f"Training complete for cutoff: {i}, {cutoffs}")

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
        return None


def get_structure_container(
    cs: ClusterSpace,
    structures: List["Atoms"],
    separate_fit: bool,
    disp_cut: float,
    ncut: int,
    param2: np.ndarray,
) -> "StructureContainer":
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
    saved_structures = []
    for _, structure in enumerate(structures):
        displacements = structure.get_array("displacements")
        mean_displacements = np.linalg.norm(displacements, axis=1).mean()
        logger.info(f"Mean displacements: {mean_displacements}")
        if not separate_fit:  # fit all
            sc.add_structure(structure)
        else:  # fit separately
            if param2 is None:  # for harmonic fitting
                if mean_displacements < disp_cut:
                    print("We are in harmonic fitting if statement")
                    print(f"mean_disp = {mean_displacements}")
                    sc.add_structure(structure)
            else:  # for anharmonic fitting
                if mean_displacements >= disp_cut:
                    print("We are in anharmonic fitting if statement")
                    print(f"mean_disp = {mean_displacements}")
                    sc.add_structure(structure)
                    saved_structures.append(structure)

    print("We have completed anharmonic fitting")
    print(sc.get_fit_data())

    if separate_fit and param2 is not None:  # do after anharmonic fitting
        A_mat = sc.get_fit_data()[0]  # displacement matrix
        f_vec = sc.get_fit_data()[1]  # force vector
        f_vec -= np.dot(A_mat[:, :ncut], param2)  # subtract harmonic forces
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
    temperature: List,
    imaginary_tol: float = IMAGINARY_TOL,
) -> Tuple[Dict, Phonopy]:
    """
    Calculate harmonic properties.

    Uses the force constants to extract phonon properties. Used for comparing
    the accuracy of force constant fits.
    Args:
        structure: The parent structure.
        supercell_matrix: The supercell transformation matrix.
        force_constants: The force constants in numpy format.
        imaginary_tol: Tolerance used to decide if a phonon mode is imaginary,
            in THz.

    Returns
    -------
        A tuple of the number of imaginary modes at Gamma, the minimum phonon
        frequency at Gamma, and the free energy, entropy, and heat capacity.
    """
    logger.info("Evaluating harmonic properties...")
    fcs2 = fcs.get_fc_array(2)
    fcs.get_fc_array(3)
    parent_phonopy = get_phonopy_structure(structure)
    phonopy = Phonopy(parent_phonopy, supercell_matrix=supercell_matrix)
    natom = phonopy.primitive.get_number_of_atoms()
    mesh = supercell_matrix.diagonal() * 2

    phonopy.set_force_constants(fcs2)
    phonopy.set_mesh(
        mesh, is_eigenvectors=True, is_mesh_symmetry=False
    )  # run_mesh(is_gamma_center=True)
    phonopy.run_thermal_properties(temperatures=temperature)
    logger.info("Thermal properties successfully run!")

    _, free_energy, entropy, heat_capacity = phonopy.get_thermal_properties()
    free_energy *= 1000 / sp.constants.Avogadro / eV2J / natom  # kJ/mol to eV/atom
    entropy *= 1 / sp.constants.Avogadro / eV2J / natom  # J/K/mol to eV/K/atom
    heat_capacity *= 1 / sp.constants.Avogadro / eV2J / natom  # J/K/mol to eV/K/atom

    freq = phonopy.mesh.frequencies  # in THz
    # find imaginary modes at gamma
    #    phonopy.run_qpoints([0, 0, 0])
    #    gamma_eigs = phonopy.get_qpoints_dict()["frequencies"]
    n_imaginary = int(np.sum(freq < -np.abs(imaginary_tol)))
    np.min(freq)

    if n_imaginary == 0:
        logger.info("No imaginary modes!")
    else:  # do not calculate these if imaginary modes exist
        logger.warning("Imaginary modes found!")

    if len(temperature) == 1:
        temperature = temperature[0]
        free_energy = free_energy[0]
        entropy = entropy[0]
        heat_capacity = heat_capacity[0]

    return {
        "temperature": temperature,
        "free_energy": free_energy,
        "entropy": entropy,
        "heat_capacity": heat_capacity,
        "n_imaginary": n_imaginary,
    }, phonopy


def anharmonic_properties(
    phonopy: Phonopy,
    fcs: ForceConstants,
    temperature: List,
    heat_capacity: np.ndarray,
    n_imaginary: float,
    bulk_modulus: float = None,
) -> Dict:
    if n_imaginary == 0:
        logger.info("Evaluating anharmonic properties...")
        fcs2 = fcs.get_fc_array(2)
        fcs3 = fcs.get_fc_array(3)
        grun, cte, dLfrac = gruneisen(
            phonopy, fcs2, fcs3, temperature, heat_capacity, bulk_modulus=bulk_modulus
        )
    else:  # do not calculate these if imaginary modes exist
        logger.warning(
            "Gruneisen and thermal expansion cannot be calculated with "
            "imaginary modes. All set to 0."
        )
        grun = [np.zeros((len(temperature), 3))]
        cte = [np.zeros((len(temperature), 3))]
        dLfrac = np.zeros((len(temperature), 3))

    return {
        "gruneisen": grun,
        "thermal_expansion": cte,
        "expansion_fraction": dLfrac,
    }


def get_total_grun(
    omega: np.ndarray, grun: np.ndarray, kweight: np.ndarray, T: float
) -> np.ndarray:
    # total = 0
    total = np.zeros((3, 3))
    weight = 0
    nptk = omega.shape[0]
    nbands = omega.shape[1]
    omega = abs(omega) * 1e12 * 2 * np.pi
    if T == 0:
        total = np.zeros((3, 3))
        grun_total_diag = np.zeros(3)
    else:
        for i in range(nptk):
            for j in range(nbands):
                x = hbar * omega[i, j] / (2.0 * kB * T)
                dBE = (x / np.sinh(x)) ** 2
                weight += dBE * kweight[i]
                total += dBE * kweight[i] * grun[i, j]
        total = total / weight
        grun_total_diag = np.array([total[0, 2], total[1, 1], total[2, 0]])

        def percent_diff(a, b):
            return abs((a - b) / b)

        # This process preserves cell symmetry upon thermal expansion, i.e., it prevents
        # symmetry-identical directions from inadvertently expanding by different ratios
        # when the Gruneisen routine returns slightly different ratios for
        # those directions
        avg012 = np.mean((grun_total_diag[0], grun_total_diag[1], grun_total_diag[2]))
        avg01 = np.mean((grun_total_diag[0], grun_total_diag[1]))
        avg02 = np.mean((grun_total_diag[0], grun_total_diag[2]))
        avg12 = np.mean((grun_total_diag[1], grun_total_diag[2]))
        if percent_diff(grun_total_diag[0], avg012) < 0.1:
            if percent_diff(grun_total_diag[1], avg012) < 0.1:
                if percent_diff(grun_total_diag[2], avg012) < 0.1:  # all siilar
                    grun_total_diag[0] = avg012
                    grun_total_diag[1] = avg012
                    grun_total_diag[2] = avg012
                elif percent_diff(grun_total_diag[2], avg02) < 0.1:  # 0 and 2 similar
                    grun_total_diag[0] = avg02
                    grun_total_diag[2] = avg02
                elif percent_diff(grun_total_diag[2], avg12) < 0.1:  # 1 and 2 similar
                    grun_total_diag[1] = avg12
                    grun_total_diag[2] = avg12
                else:
                    pass
            elif percent_diff(grun_total_diag[1], avg01) < 0.1:  # 0 and 1 similar
                grun_total_diag[0] = avg01
                grun_total_diag[1] = avg01
            elif percent_diff(grun_total_diag[1], avg12) < 0.1:  # 1 and 2 similar
                grun_total_diag[1] = avg12
                grun_total_diag[2] = avg12
            else:
                pass
        elif percent_diff(grun_total_diag[0], avg01) < 0.1:  # 0 and 1 similar
            grun_total_diag[0] = avg01
            grun_total_diag[1] = avg01
        elif percent_diff(grun_total_diag[0], avg02) < 0.1:  # 0 and 2 similar
            grun_total_diag[0] = avg02
            grun_total_diag[2] = avg02
        else:  # nothing similar
            pass

    return grun_total_diag


def gruneisen(
    phonopy: Phonopy,
    fcs2: np.ndarray,
    fcs3: np.ndarray,
    temperature: List,
    heat_capacity: np.ndarray,  # in eV/K/atom
    bulk_modulus: float = None,  # in GPa
) -> Tuple[List, List, np.ndarray]:
    gruneisen = Gruneisen(fcs2, fcs3, phonopy.supercell, phonopy.primitive)
    gruneisen.set_sampling_mesh(phonopy.mesh_numbers, is_gamma_center=True)
    gruneisen.run()
    grun = gruneisen.get_gruneisen_parameters()  # (nptk,nmode,3,3)
    omega = gruneisen._frequencies
    kweight = gruneisen._weights
    grun_tot = []
    # for temp in temperature:
    #     grun_tot.append(get_total_grun(omega, grun, kweight, temp))
    # grun_tot = np.nan_to_num(np.array(grun_tot))

    grun_tot = [
        np.nan_to_num(np.array(get_total_grun(omega, grun, kweight, temp)))
        for temp in temperature
    ]

    # linear thermal expansion coefficient and fraction
    if bulk_modulus is None:
        cte = None
        dLfrac = None
    else:
        heat_capacity *= (
            eV2J * phonopy.primitive.get_number_of_atoms()
        )  # eV/K/atom to J/K
        vol = phonopy.primitive.get_volume()
        # cte = grun_tot*heat_capacity.repeat(3)/(vol/10**30)/(bulk_modulus*10**9)/3
        cte = (
            grun_tot
            * heat_capacity.repeat(3).reshape(len(heat_capacity), 3)
            / (vol / 10**30)
            / (bulk_modulus * 10**9)
            / 3
        )
        cte = np.nan_to_num(cte)
        dLfrac = thermal_expansion(temperature, cte)
        if len(temperature) == 1:
            dLfrac = dLfrac[-1]
        logger.info(f"Gruneisen: \n {grun_tot}")
        logger.info(f"Coefficient of Thermal Expansion: \n {cte}")
        logger.info(f"Linear Expansion Fraction: \n {dLfrac}")

    return grun_tot, cte, dLfrac


def thermal_expansion(
    temperature: List,
    cte: np.ndarray,
) -> np.ndarray:
    assert len(temperature) == len(cte)
    if 0 not in temperature:
        temperature = [0, *temperature]
        cte = np.array([np.array([0, 0, 0]), *list(cte)])
    temperature_np = np.array(temperature)
    ind = np.argsort(temperature_np)
    temperature_np = temperature[ind]
    cte = np.array(cte)[ind]
    # linear expansion fraction
    dLfrac = copy(cte)
    for t in range(len(temperature_np)):
        dLfrac[t, :] = np.trapz(cte[: t + 1, :], temperature_np[: t + 1], axis=0)
    return np.nan_to_num(dLfrac)


@job
# @explicit_serialize
# class RunShengBTE(FiretaskBase):
def run_shengbte(
    shengbte_cmd, renormalized, temperature, control_kwargs, prev_dir_hiphive, loop
):
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
    print("We are in ShengBTE FW 1")

    copy_hiphive_outputs(prev_dir_hiphive)
    with open(f"structure_data_{loop}.json") as file:
        structure_data = json.load(file)
        dumpfn(structure_data, "structure_data.json")

    print("We are in ShengBTE FW 2")

    # Create a symlink to ShengBTE

    ShengBTE = "ShengBTE"
    src = "/global/homes/h/hrushi99/code/shengbte_new3/shengbte/ShengBTE"
    dst = os.path.join(os.getcwd(), ShengBTE)

    with contextlib.suppress(FileExistsError):
        os.symlink(src, dst)

    print("We are in ShengBTE FW 3")

    structure_data = loadfn("structure_data.json")
    structure = structure_data["structure"]
    supercell_matrix = structure_data["supercell_matrix"]

    print(f"Temperature = {temperature}")

    temperature = temperature if temperature is not None else T_KLAT

    renormalized = renormalized if renormalized is not None else False

    if renormalized:
        assert isinstance(temperature, (int, float))
    else:
        if isinstance(temperature, (int, float)):
            pass
        elif isinstance(temperature, dict):
            temperature["t_min"]
            temperature["t_max"]
            temperature["t_step"]
        else:
            raise ValueError("Unsupported temperature type, must be float or dict")

    print("We are in ShengBTE FW 4")

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

    # shengbte_cmd = env_chk(self["shengbte_cmd"], fw_spec)
    # shengbte_cmd = env_chk(shengbte_cmd, fw_spec)

    if isinstance(shengbte_cmd, str):
        shengbte_cmd = os.path.expandvars(shengbte_cmd)
        shengbte_cmd = shlex.split(shengbte_cmd)

    print("We are in ShengBTE FW 5")

    shengbte_cmd = list(shengbte_cmd)
    logger.info(f"Running command: {shengbte_cmd}")

    with open("shengbte.out", "w") as f_std, open(
        "shengbte_err.txt", "w", buffering=1
    ) as f_err:
        # use line buffering for stderr
        return_code = subprocess.call(shengbte_cmd, stdout=f_std, stderr=f_err)
    logger.info(
        f"Command {shengbte_cmd} finished running with returncode: {return_code}"
    )

    print("We are in ShengBTE FW 6")

    if return_code == 1:
        raise RuntimeError(
            f"Running ShengBTE failed. Check '{os.getcwd()}/shengbte_err.txt' for "
            "details."
        )


@job
# @explicit_serialize
def run_fc_to_pdos(
    renormalized: bool = None,
    renorm_temperature: str = None,
    mesh_density: float = None,
    prev_dir_json_saver: str = None,
    loop: int = None,
):
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
    renorm_temperature = renorm_temperature if renorm_temperature else None
    mesh_density = mesh_density if mesh_density else 100.0

    structure_data = loadfn(f"structure_data_{loop}.json")
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
        renorm_thermal_data = loadfn("renorm_thermal_data.json")
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

    return uniform_bs, lm_bs, dos


def _get_fc_fsid(structure, supercell_matrix, fcs, mesh_density):
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

    # logger.info("Inserting phonon objects into database.")
    # dos_fsid, _ = mmdb.insert_gridfs(
    #     dos.to_json(), collection="phonon_dos_fs"
    # )
    # uniform_bs_fsid, _ = mmdb.insert_gridfs(
    #     uniform_bs.to_json(), collection="phonon_bandstructure_fs"
    # )
    # lm_bs_fsid, _ = mmdb.insert_gridfs(
    #     lm_bs.to_json(), collection="phonon_bandstructure_fs"
    # )

    # logger.info("Inserting force constants into database.")
    # fc_json = json.dumps(
    #     {str(k): v.tolist() for k, v in fcs.get_fc_dict().items()}
    # )
    # fc_fsid, _ = mmdb.insert_gridfs(
    #     fc_json, collection="phonon_force_constants_fs"
    # )

    # return dos_fsid, uniform_bs_fsid, lm_bs_fsid, fc_fsid
    return uniform_bs, lm_bs, dos


@job
# @explicit_serialize
# class RunHiPhiveRenorm(FiretaskBase):
def run_hiphive_renormalization(
    temperature: float,
    renorm_method: str,
    nconfig: int,
    conv_thresh: float,
    max_iter: int,
    renorm_TE_iter: bool,
    bulk_modulus: float,
    mesh_density: float,
    prev_dir_hiphive: str,
    loop: int,
):
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

    thermal_data = loadfn("thermal_data.json")
    thermal_data = thermal_data["heat_capacity"]

    cutoffs = fitting_data["cutoffs"]
    fit_method = fitting_data["fit_method"]
    fitting_data["n_imaginary"]
    fitting_data["imaginary_tol"]

    parent_structure = structure_data["structure"]
    supercell_structure = structure_data["supercell_structure"]
    supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)
    supercell_matrix = np.array(structure_data["supercell_matrix"])

    # temperature = temperature
    # renorm_method = renorm_method
    # nconfig = nconfig
    # conv_thresh = conv_thresh
    # max_iter = max_iter
    # renorm_TE_iter = renorm_TE_iter
    # bulk_modulus = bulk_modulus

    # Renormalization with DFT lattice
    renorm_data = run_renormalization(
        parent_structure,
        supercell_atoms,
        supercell_matrix,
        cs,
        fcs,
        param,
        temperature,
        nconfig,
        max_iter,
        conv_thresh,
        renorm_method,
        fit_method,
        bulk_modulus,
        phonopy_orig,
    )

    # Additional renormalization with thermal expansion -
    # optional - just single "iteration" for now
    if renorm_TE_iter:
        n_TE_iter = 1
        for i in range(n_TE_iter):
            if renorm_data is None or renorm_data["n_imaginary"] < 0:
                # failed, incomplete, or still imaginary
                break
            logger.info(
                f"Renormalizing with thermally expanded lattice - iteration {i}"
            )

            dLfrac = renorm_data["expansion_fraction"]
            param = renorm_data["param"]

            parent_structure_TE, supercell_atoms_TE, cs_TE, fcs_TE = setup_TE_iter(
                cs,
                cutoffs,
                parent_structure,
                param,
                temperature,
                dLfrac,
                supercell_matrix,
            )
            prim_TE_atoms = AseAtomsAdaptor.get_atoms(parent_structure_TE)
            prim_TE_phonopy = PhonopyAtoms(
                symbols=prim_TE_atoms.get_chemical_symbols(),
                scaled_positions=prim_TE_atoms.get_scaled_positions(),
                cell=prim_TE_atoms.cell,
            )
            phonopy_TE = Phonopy(
                prim_TE_phonopy,
                supercell_matrix=supercell_matrix,
                primitive_matrix=None,
            )

            renorm_data = run_renormalization(
                parent_structure_TE,
                supercell_atoms_TE,
                supercell_matrix,
                cs_TE,
                fcs_TE,
                param,
                temperature,
                nconfig,
                max_iter,
                conv_thresh,
                renorm_method,
                fit_method,
                bulk_modulus,
                phonopy_TE,
            )
        structure_data["structure"] = parent_structure_TE
        structure_data["supercell_structure"] = AseAtomsAdaptor.get_structure(
            supercell_atoms_TE
        )

    # write results
    logger.info("Writing renormalized results")
    # renorm_thermal_data = {}
    renorm_thermal_data: Dict[str, Any] = {}
    fcs = renorm_data["fcs"]
    fcs.write("force_constants.fcs")
    thermal_keys = [
        "temperature",
        "free_energy",
        "entropy",
        "heat_capacity",
        "gruneisen",
        "thermal_expansion",
        "expansion_fraction",
        "free_energy_correction_S",
        "free_energy_correction_SC",
    ]
    renorm_thermal_data = {key: [] for key in thermal_keys}
    for key in thermal_keys:
        renorm_thermal_data[key].append(renorm_data[key])

    logger.info(f"DEBUG: {renorm_data}")
    if renorm_data["n_imaginary"] > 0:
        logger.warning(f"Imaginary modes remain for {temperature} K!")
        logger.warning("ShengBTE files not written")
        logger.warning("No renormalization with thermal expansion")
    else:
        logger.info("No imaginary modes! Writing ShengBTE files")
        fcs.write_to_phonopy("FORCE_CONSTANTS_2ND".format(), format="text")

    dumpfn(structure_data, "structure_data.json".format())
    dumpfn(renorm_thermal_data, "renorm_thermal_data.json".format())


def run_renormalization(
    structure: Structure,
    supercell: Atoms,
    supercell_matrix: np.ndarray,
    cs: ClusterSpace,
    fcs: ForceConstants,
    param: np.ndarray,
    T: float,
    nconfig: int,
    max_iter: int,
    conv_tresh: float,
    renorm_method: str,
    fit_method: str,
    bulk_modulus: float = None,
    phonopy_orig: Phonopy = None,
    imaginary_tol: float = IMAGINARY_TOL,
) -> Dict:
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
    renorm = Renormalization(cs, supercell, fcs, param, T, renorm_method, fit_method)
    fcp, fcs, param = renorm.renormalize(nconfig)  # ,conv_tresh)

    renorm_data, phonopy = harmonic_properties(
        structure, supercell_matrix, fcs, [T], imaginary_tol
    )

    if renorm_data["n_imaginary"] == 0:
        logger.info(f"Renormalized phonon is completely real at T = {T} K!")
    anharmonic_data = anharmonic_properties(
        phonopy,
        fcs,
        [T],
        renorm_data["heat_capacity"],
        renorm_data["n_imaginary"],
        bulk_modulus=bulk_modulus,
    )
    #    else:
    #        anharmonic_data = dict()
    #        anharmonic_data["temperature"] = T
    #        anharmonic_data["gruneisen"] = np.array([0,0,0])
    #        anharmonic_data["thermal_expansion"] = np.array([0,0,0])
    #        anharmonic_data["expansion_fraction"] = np.array([0,0,0])
    renorm_data.update(anharmonic_data)

    phonopy_orig.run_mesh()
    phonopy.run_mesh()
    omega0 = phonopy_orig.mesh.frequencies  # THz
    omega_TD = phonopy.mesh.frequencies  # THz
    evec = phonopy.mesh.eigenvectors
    #    natom = phonopy.primitive.get_number_of_atoms()
    correction_S, correction_SC = free_energy_correction(
        omega0, omega_TD, evec, [T]
    )  # eV/atom

    renorm_data["free_energy_correction_S"] = correction_S[0]
    renorm_data["free_energy_correction_SC"] = correction_SC[0]
    renorm_data["fcp"] = fcp
    renorm_data["fcs"] = fcs
    renorm_data["param"] = param

    return renorm_data


def setup_TE_iter(
    cs, cutoffs, parent_structure, param, temperatures, dLfrac, supercell_matrix
):
    new_atoms = AseAtomsAdaptor.get_atoms(parent_structure)
    new_cell = Cell(
        np.transpose(
            [new_atoms.get_cell()[:, i] * (1 + dLfrac[0, i]) for i in range(3)]
        )
    )
    new_atoms.set_cell(new_cell, scale_atoms=True)
    parent_structure_TE = AseAtomsAdaptor.get_structure(new_atoms)
    supercell_atoms_TE = AseAtomsAdaptor.get_atoms(
        parent_structure_TE * supercell_matrix
    )
    new_cutoffs = [i * (1 + np.linalg.norm(dLfrac)) for i in cutoffs]
    fcs_TE = []  # not sure if this is correct
    while True:
        cs_TE = ClusterSpace(atoms, new_cutoffs, 1e-3, acoustic_sum_rules=True)
        if cs_TE.n_dofs == cs.n_dofs:
            break
        if cs_TE.n_dofs > cs.n_dofs:
            new_cutoffs = [i * 0.999 for i in new_cutoffs]
        if cs_TE.n_dofs < cs.n_dofs:
            new_cutoffs = [i * 1.001 for i in new_cutoffs]
        new_fcp = ForceConstantPotential(cs_TE, param)
        fcs_TE.append(new_fcp.get_force_constants(supercell_atoms_TE))
    return parent_structure_TE, supercell_atoms_TE, cs_TE, fcs_TE


@job
# @explicit_serialize
def run_lattice_thermal_conductivity(
    shengbte_cmd: str,
    prev_dir_hiphive: str,
    loop: int,
    temperature: Union[float, dict],
    renormalized: bool,
    name="Lattice Thermal Conductivity",
    # prev_calc_dir: Optional[str] = None,
    # db_file: str = None,
    shengbte_control_kwargs: Optional[dict] = None,
):
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

    print("We are in Lattice Thermal Conductivity FW 1")

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

    print("We are in Lattice Thermal Conductivity FW 2")

    shengbte = run_shengbte(
        shengbte_cmd=shengbte_cmd,
        renormalized=renormalized,
        temperature=temperature,
        control_kwargs=shengbte_control_kwargs,
        prev_dir_hiphive=prev_dir_hiphive,
        loop=loop,
    )

    # shengbte_to_db = ShengBTEToDb(db_file=db_file, additional_fields={})

    # tasks = [copy_files, run_shengbte, shengbte_to_db]

    # super().__init__(tasks, name=name, **kwargs)

    return Response(replace=shengbte)
