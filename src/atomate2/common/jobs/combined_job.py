
from __future__ import annotations

import contextlib
from phonopy import Phonopy
from pymatgen.core import Structure
from typing import TYPE_CHECKING
import warnings
from monty.json import jsanitize
from monty.serialization import dumpfn
from jobflow import job
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)
import numpy as np
from atomate2.common.jobs.utils import structure_to_conventional, structure_to_primitive
from atomate2.forcefields.utils import Relaxer
from atomate2.forcefields.schemas import ForceFieldTaskDocument
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from atomate2.common.schemas.phonons import ForceConstants, PhononBSDOSDoc, get_factor
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

if TYPE_CHECKING:
    from pathlib import Path

    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

    from atomate2.aims.jobs.base import BaseAimsMaker
    from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker

SUPPORTED_CODES = ["vasp", "aims", "forcefields"]


def get_supercell_size(
    structure: Structure, min_length: float, prefer_90_degrees: bool, **kwargs
) -> list[list[float]]:
    """
    Determine supercell size with given min_length.

    Parameters
    ----------
    structure: Structure Object
        Input structure that will be used to determine supercell
    min_length: float
        minimum length of cell in Angstrom
    prefer_90_degrees: bool
        if True, the algorithm will try to find a cell with 90 degree angles first
    **kwargs:
        Additional parameters that can be set.
    """
    kwargs.setdefault("min_atoms", None)
    kwargs.setdefault("force_diagonal", False)

    if not prefer_90_degrees:
        kwargs.setdefault("max_atoms", None)
        transformation = CubicSupercellTransformation(
            min_length=min_length,
            min_atoms=kwargs["min_atoms"],
            max_atoms=kwargs["max_atoms"],
            force_diagonal=kwargs["force_diagonal"],
            force_90_degrees=False,
        )
        transformation.apply_transformation(structure=structure)
    else:
        max_atoms = kwargs.get("max_atoms", 1000)
        kwargs.setdefault("angle_tolerance", 1e-2)
        try:
            transformation = CubicSupercellTransformation(
                min_length=min_length,
                min_atoms=kwargs["min_atoms"],
                max_atoms=max_atoms,
                force_diagonal=kwargs["force_diagonal"],
                force_90_degrees=True,
                angle_tolerance=kwargs["angle_tolerance"],
            )
            transformation.apply_transformation(structure=structure)

        except AttributeError:
            kwargs.setdefault("max_atoms", None)

            transformation = CubicSupercellTransformation(
                min_length=min_length,
                min_atoms=kwargs["min_atoms"],
                max_atoms=kwargs["max_atoms"],
                force_diagonal=kwargs["force_diagonal"],
                force_90_degrees=False,
            )
            transformation.apply_transformation(structure=structure)

    return transformation.transformation_matrix.tolist()

def structure_to_conventional(
    structure: Structure, symprec: float = 1e-4
) -> Structure:
    """
    Job hat creates a standard conventional structure.

    Parameters
    ----------
    structure: Structure object
        input structure that will be transformed
    symprec: float
        precision to determine symmetry

    Returns
    -------
    .Structure
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    return sga.get_conventional_standard_structure()

def structure_to_primitive(
    structure: Structure, symprec: float = 1e-4
) -> Structure:
    """
    Job that creates a standard primitive structure.

    Parameters
    ----------
    structure: Structure object
        input structure that will be transformed
    symprec: float
        precision to determine symmetry

    Returns
    -------
    .Structure
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    return sga.get_primitive_standard_structure()

def gap_relax_calculator(potential_param_file_name,potential_kwargs={}, optimizer_kwargs={},relax_cell=False):
    from quippy.potential import Potential

    calculator = Potential(
        args_str="IP GAP",
        param_filename=str(potential_param_file_name),
        **potential_kwargs,
    )
    relaxer = Relaxer(
        calculator, **optimizer_kwargs, relax_cell=relax_cell
    )
    return relaxer

def gap_static_calculator(potential_param_file_name, potential_kwargs={}):
    from quippy.potential import Potential

    calculator = Potential(
        args_str="IP GAP",
        param_filename=str(potential_param_file_name),
        **potential_kwargs,
    )
    relaxer = Relaxer(calculator, relax_cell=False)
    return relaxer


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
        **kwargs,
    )


@job(
    output_schema=PhononBSDOSDoc,
    data=[PhononDos, PhononBandStructureSymmLine, ForceConstants],
)
def phonon_job(
        structure: Structure,
        potential_parameter_filename: str,
        supercell_size_kwargs: dict={},
        static_kwargs: dict={},
        optimizer_kwargs: dict={},
        bulk_potential_kwargs:dict={},
        relax_steps: int = 5000,
        relax_kwargs: dict = {"interval": 5000, "fmax": 0.00001},
        generate_frequencies_eigenvectors_kwargs: dict = {"npoints_band": 50},
        sym_reduce: bool = True,
        symprec: float = 1e-4,
        displacement: float = 0.01,
        min_length: float | None = 20.0,
        prefer_90_degrees: bool = True,
        use_symmetrized_structure: str | None = None,
        store_force_constants=True,
        bulk_relax: bool = True,
        calculate_static_energy: bool = True,
        create_thermal_displacements= True,
        prev_dir: str | Path | None = None,
        born: list[Matrix3D] | None = None,
        bulk_relax_cell=True,
        static_relax_cell=False,
        epsilon_static: Matrix3D | None = None,
        total_dft_energy_per_formula_unit: float | None = None,
        supercell_matrix: Matrix3D | None = None,
        kpath_scheme: str = "seekpath",
        code: str = None
        ) -> PhononBSDOSDoc:
    use_symmetrized_structure = use_symmetrized_structure
    kpath_scheme = kpath_scheme
    valid_structs = (None, "primitive", "conventional")
    if use_symmetrized_structure not in valid_structs:
        raise ValueError(
            f"Invalid {use_symmetrized_structure=}, use one of {valid_structs}"
        )

    if use_symmetrized_structure != "primitive" and kpath_scheme != "seekpath":
        raise ValueError(
            f"You can't use {kpath_scheme=} with the primitive standard "
            "structure, please use seekpath"
        )

    valid_schemes = ("seekpath", "hinuma", "setyawan_curtarolo", "latimer_munro")
    if kpath_scheme not in valid_schemes:
        raise ValueError(
            f"{kpath_scheme=} is not implemented, use one of {valid_schemes}"
        )

    if code is None or code not in SUPPORTED_CODES:
        raise ValueError(
            "The code variable must be passed and it must be a supported code."
            f" Supported codes are: {SUPPORTED_CODES}"
        )

    if use_symmetrized_structure == "primitive":
        # These structures are compatible with many
        # of the kpath algorithms that are used for Materials Project
        prim_job = structure_to_primitive(structure, symprec)
        structure = prim_job

    elif use_symmetrized_structure == "conventional":
        # it could be beneficial to use conventional standard structures to arrive
        # faster at supercells with right angles
        conv_job = structure_to_conventional(structure, symprec)
        structure = conv_job

    optimization_run_job_dir = None
    optimization_run_uuid = None

    if bulk_relax:
        # optionally relax the structure
        # load potential
        bulk_relax_calculator = gap_relax_calculator(potential_param_file_name=potential_parameter_filename,
                                 optimizer_kwargs=optimizer_kwargs, potential_kwargs=bulk_potential_kwargs,
                                                     relax_cell=bulk_relax_cell)

        bulk = bulk_relax_calculator.relax(structure, steps=relax_steps, **relax_kwargs)
        bulk_task_doc = ForceFieldTaskDocument.from_ase_compatible_result(result=bulk, forcefield_name='GAP', relax_cell=bulk_relax_cell, steps=relax_steps,
                                                                           relax_kwargs=relax_kwargs, optimizer_kwargs=optimizer_kwargs)
        structure = bulk_task_doc.output.structure

    if supercell_matrix is None:
        supercell_matrix = get_supercell_size(
            structure,
            min_length,
            prefer_90_degrees,
            **supercell_size_kwargs,
        )

    # Initialize static calculator
    static_energy_calculator = gap_static_calculator(
        potential_param_file_name=potential_parameter_filename,
    )

    # Computation of static energy
    total_dft_energy = None
    static_run_job_dir = None
    static_run_uuid = None
    if calculate_static_energy and (
            total_dft_energy_per_formula_unit is None
    ):
        static_job_kwargs = {}
        # static_energy_calculator = gap_static_calculator(
        #     potential_param_file_name=potential_parameter_filename,
        #     )
        static_run = static_energy_calculator.relax(structure, steps=1, **static_kwargs)
        static_task_doc = ForceFieldTaskDocument.from_ase_compatible_result(result=static_run, forcefield_name='GAP',
                                                                          relax_cell=static_relax_cell,
                                                                            steps=1,
                                                                          relax_kwargs=static_kwargs,
                                                                          optimizer_kwargs=optimizer_kwargs)
        total_dft_energy = static_task_doc.output.energy

    warnings.warn(
        "Initial magnetic moments will not be considered for the determination "
        "of the symmetry of the structure and thus will be removed now.",
        stacklevel=1,
    )
    cell = get_phonopy_structure(
        structure.remove_site_property(property_name="magmom")
        if "magmom" in structure.site_properties
        else structure
    )
    factor = get_factor(code)

    if use_symmetrized_structure == "primitive" and kpath_scheme != "seekpath":
        primitive_matrix: np.ndarray | str = np.eye(3)
    else:
        primitive_matrix = "auto"

        # TARP: THIS IS BAD! Including for discussions sake
    if cell.magnetic_moments is not None and primitive_matrix == "auto":
        if np.any(cell.magnetic_moments != 0.0):
            raise ValueError(
                "For materials with magnetic moments specified "
                "use_symmetrized_structure must be 'primitive'"
            )
        cell.magnetic_moments = None

    phonon = Phonopy(
        cell,
        supercell_matrix,
        primitive_matrix=primitive_matrix,
        factor=factor,
        symprec=symprec,
        is_symmetry=sym_reduce,
    )
    phonon.generate_displacements(distance=displacement)

    supercells = phonon.supercells_with_displacements

    outputs: dict[str, list] = {
        "displacement_number": [],
        "forces": [],
        "uuids": [],
        "dirs": [],
    }

    for idx, sc in enumerate(supercells):
        if prev_dir is not None:
            phonon_job = static_energy_calculator.relax(get_pmg_structure(sc), steps=1)
        else:
            phonon_job = static_energy_calculator.relax(get_pmg_structure(sc), steps=1)

        phonon_job_taskdoc = ForceFieldTaskDocument.from_ase_compatible_result(result=phonon_job, forcefield_name='GAP',
                                                                          relax_cell=static_relax_cell,
                                                                               steps=1,
                                                                          relax_kwargs=static_kwargs,
                                                                          optimizer_kwargs=optimizer_kwargs)
        print(f'Running supercell {idx+1}/{len(supercells)}')
        # we will add some metadata
        info = {
            "displacement_number": idx,
            "original_structure": structure,
            "supercell_matrix": supercell_matrix,
            "displaced_structure": sc,
        }
        with contextlib.suppress(Exception):
            phonon_job.update_maker_kwargs(
                {"_set": {"write_additional_data->phonon_info:json": info}},
                dict_mod=True,
            )
        outputs["displacement_number"].append(idx)
        # outputs["uuids"].append(phonon_job_taskdoc.output.uuid)
        # outputs["dirs"].append(phonon_job_taskdoc.output.dir_name)
        outputs["forces"].append(phonon_job_taskdoc.output.forces)

    born_run_job_dir = None
    born_run_uuid = None

    phonon_collect = generate_frequencies_eigenvectors(
        supercell_matrix=supercell_matrix,
        displacement=displacement,
        sym_reduce=sym_reduce,
        symprec=symprec,
        use_symmetrized_structure=use_symmetrized_structure,
        kpath_scheme=kpath_scheme,
        code=code,
        structure=structure,
        displacement_data=outputs,
        epsilon_static=epsilon_static,
        born=born,
        total_dft_energy=total_dft_energy,
        **{'static_run_job_dir': static_run_job_dir,
           'static_run_uuid': static_run_uuid,
           'born_run_job_dir': born_run_job_dir,
           'born_run_uuid': born_run_uuid,
           'optimization_run_job_dir': optimization_run_job_dir,
           'optimization_run_uuid': optimization_run_uuid,
           'create_thermal_displacements':create_thermal_displacements,
           'store_force_constants':store_force_constants},
        **generate_frequencies_eigenvectors_kwargs,
    )

    # save forces for later use ??
    forces_doc = jsanitize(outputs, strict=False, allow_bson=False, enum_values=False, recursive_msonable=False)
    dumpfn(forces_doc, fn='forces_data.json.gz')

    # jsanited_doc = jsanitize(phonon_collect, strict=False, allow_bson=False,
    # enum_values=False, recursive_msonable=False)

    # dumpfn(jsanited_doc, fn='phononbsdostaskdoc.json.gz')

    return phonon_collect

