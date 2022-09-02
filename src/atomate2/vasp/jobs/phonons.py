from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
from jobflow import Flow, Response, job
from phonopy import Phonopy
from phonopy.units import VaspToTHz
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)

from atomate2.common.schemas.math import Matrix3D
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.schemas.phonons import PhononBSDOSDoc
from atomate2.vasp.sets.base import VaspInputGenerator
from atomate2.vasp.sets.core import StaticSetGenerator

logger = logging.getLogger(__name__)

__all__ = [
    "structure_to_primitive",
    "structure_to_conventional",
    "get_total_energy_per_cell",
    "get_supercell_size",
    "generate_phonon_displacements",
    "run_phonon_displacements",
    "generate_frequencies_eigenvectors",
    "PhononDisplacementMaker",
]


@job
def structure_to_primitive(structure: Structure, symprec: float):
    """
    Job hat creates a standard primitive structure.

    Parameters
    ----------
        structure: Structure object
        symprec: float
            precision to determine symmetry

    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    return sga.get_primitive_standard_structure()


@job
def structure_to_conventional(structure: Structure, symprec: float):
    """
    Job hat creates a standard conventional structure.

    Parameters
    ----------
    structure: Structure object
    symprec: float
        precision to determine symmetry


    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    return sga.get_conventional_standard_structure()


@job
def get_total_energy_per_cell(
    total_dft_energy_per_formula_unit: float, structure: Structure
):
    """
    Job that computes total dft energy of the cell.

    Parameters
    ----------
    total_dft_energy_per_formula_unit: float
        Total DFT energy in eV per formula unit.
    structure: Structure object
        Corresponding structure object.
    """
    formula_units = (
        structure.composition.num_atoms
        / structure.composition.reduced_composition.num_atoms
    )

    return total_dft_energy_per_formula_unit * formula_units


@job
def get_supercell_size(
    structure: Structure, min_length: float, prefer_90_degrees: bool, **kwargs
):
    """
    Determine supercell size with given min_length.

    Parameters
    ----------
    structure: Structure Object
    min_length: float
        minimum length of cell in Angstrom
    prefer_90_degrees: bool
        if True, the algorithm will try to find a cell with 90 degree angles first
    **kwargs:
        Additional parameters that can be set.
    """
    if "min_atoms" not in kwargs:
        kwargs["min_atoms"] = None
    if "force_diagonal" not in kwargs:
        kwargs["force_diagonal"] = False

    if not prefer_90_degrees:
        if "max_atoms" not in kwargs:
            kwargs["max_atoms"] = None
        transformation = CubicSupercellTransformation(
            min_length=min_length,
            min_atoms=kwargs["min_atoms"],
            max_atoms=kwargs["max_atoms"],
            force_diagonal=kwargs["force_diagonal"],
            force_90_degrees=False,
        )
        transformation.apply_transformation(structure=structure)

    else:
        if "max_atoms" not in kwargs:
            max_atoms = 1000
        else:
            max_atoms = kwargs["max_atoms"]
        if "angle_tolerance" not in kwargs:
            kwargs["angle_tolerance"] = 1e-2
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
            if "max_atoms" not in kwargs:
                kwargs["max_atoms"] = None

            transformation = CubicSupercellTransformation(
                min_length=min_length,
                min_atoms=kwargs["min_atoms"],
                max_atoms=kwargs["max_atoms"],
                force_diagonal=kwargs["force_diagonal"],
                force_90_degrees=False,
            )
            transformation.apply_transformation(structure=structure)

    supercell_matrix = transformation.transformation_matrix.tolist()
    return supercell_matrix


@job
def generate_phonon_displacements(
    structure: Structure,
    supercell_matrix: np.array,
    displacement: float,
    sym_reduce: bool,
    symprec: float,
    use_symmetrized_structure: Union[str, None],
    kpath_scheme: str,
    code: str,
):
    """
    Generate displaced structures with phonopy.

    Parameters
    ----------
    structure: Structure object
    supercell_matrix: np.array
        array to describe supercell matrix
    displacement: float
        displacement in Angstrom
    sym_reduce: bool
        if True, symmetry will be used to generate displacements
    symprec: float
        precision to determine symmetry
    use_symmetrized_structure: str»|None
        primitive, conventional or None
    kpath_scheme: str
        scheme to generate kpath
    code:
        code to perform the computations
    """
    cell = get_phonopy_structure(structure)
    if code == "vasp":
        factor = VaspToTHz
    # a bit of code repetition here as I currently
    # do not see how to pass the phonopy object?
    if use_symmetrized_structure == "primitive" and kpath_scheme != "seekpath":
        primitive_matrix: Union[List[List[float]], str] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    else:
        primitive_matrix = "auto"
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

    displacements = []
    for cell in supercells:
        displacements.append(get_pmg_structure(cell))
    return displacements


@job(output_schema=PhononBSDOSDoc, data=[PhononDos, PhononBandStructureSymmLine])
def generate_frequencies_eigenvectors(
    structure: Structure,
    supercell_matrix: np.array,
    displacement: float,
    sym_reduce: bool,
    symprec: float,
    use_symmetrized_structure: Union[str, None],
    kpath_scheme: str,
    code: str,
    displacement_data: dict[str, list],
    total_dft_energy: float,
    epsilon_static: Matrix3D = None,
    born: Matrix3D = None,
    **kwargs,
):
    """
    Analyze the phonon runs and summarize the results.

    Parameters
    ----------
    structure: Structure object
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
        total dft energy in eV per cell
    epsilon_static: Matrix3D
        The high-frequency dielectric constant
    born: Matrix3D
        Born charges
    kwargs:
        additional arguments that are passed

    """
    phonon_doc = PhononBSDOSDoc.from_forces_born(
        structure=structure,
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

    return phonon_doc


@job
def run_phonon_displacements(
    displacements,
    structure: Structure,
    supercell_matrix,
    phonon_maker: BaseVaspMaker = None,
):
    """
    Run phonon displacements.

    Note, this job will replace itself with N displacement calculations.

    Parameters
    ----------
    displacements
    structure: original structure for meta data
    supercell_matrix: supercell matrix for meta data
    phonon_maker : .BaseVaspMaker
        A VaspMaker to use to generate the elastic relaxation jobs.
    """
    if phonon_maker is None:
        phonon_maker = PhononDisplacementMaker()
    phonon_jobs = []
    outputs: dict[str, list] = {
        "displacement_number": [],
        "forces": [],
        "uuids": [],
        "dirs": [],
    }

    for i, displacement in enumerate(displacements):
        phonon_job = phonon_maker.make(displacement)
        phonon_job.append_name(f" {i + 1}/{len(displacements)}")

        # we will add some meta data
        info = {
            "displacement_number": i,
            "original_structure": structure,
            "supercell_matrix": supercell_matrix,
        }
        phonon_job.update_maker_kwargs(
            {"_set": {"write_additional_data->phonon_info:json": info}}, dict_mod=True
        )
        phonon_jobs.append(phonon_job)
        outputs["displacement_number"].append(i)
        outputs["uuids"].append(phonon_job.output.uuid)
        outputs["dirs"].append(phonon_job.output.dir_name)
        outputs["forces"].append(phonon_job.output.output.forces)

    displacement_flow = Flow(phonon_jobs, outputs)
    return Response(replace=displacement_flow)


@dataclass
class PhononDisplacementMaker(BaseVaspMaker):
    """
    Maker to perform a static calculation as a part of the finite displacement method.

    The input set is for a static run with tighter convergence parameters.
    Both the k-point mesh density and convergence parameters
    are stricter than a normal relaxation.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "phonon static"

    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings={"grid_density": 7000},
            user_incar_settings={
                "IBRION": 2,
                "ISIF": 3,
                "ENCUT": 700,
                "EDIFF": 1e-7,
                "LAECHG": False,
                "LREAL": False,
                "ALGO": "Normal",
                "NSW": 0,
                "LCHARG": False,
                "ISMEAR": 0,
            },
        )
    )
