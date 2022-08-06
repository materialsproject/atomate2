from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
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
    "generate_phonon_displacements",
    "run_phonon_displacements",
    "generate_frequencies_eigenvectors",
    "PhononDisplacementMaker",
]


@job
def structure_to_primitive(structure, symprec):
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    return sga.get_primitive_standard_structure()


@job
def structure_to_conventional(structure: Structure, symprec: float):
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    return sga.get_conventional_standard_structure()


# TODO: maybe add  an alternative algorithm
@job
def get_supercell_size(
    structure: Structure, min_length: float, prefer_90_degrees: bool, **kwargs
):
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
        structure = transformation.apply_transformation(structure=structure)

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
    use_standard_primitive: bool,
    kpath_scheme: str,
    code: str,
):
    """
    Generate phonon displacements.

    Parameters
    ----------
    phonopy_object: Phonopy_object

    Returns
    -------
    List[Deformation]
        A list of displacements.
    """
    cell = get_phonopy_structure(structure)
    if code == "vasp":
        factor = VaspToTHz
    # a bit of code repetition here as I currently
    # do not see how to pass the phonopy object?
    if use_standard_primitive and kpath_scheme != "seekpath":
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
    use_standard_primitive: bool,
    kpath_scheme: str,
    code: str,
    displacement_data: dict[str, list],
    total_energy: float,
    epsilon_static: Matrix3D = None,
    born: Matrix3D = None,
    full_born: bool = True,
    born_run_job_dir: str | Path | None = None,
    born_run_uuid=None,
    static_run_job_dir: str | Path | None = None,
    static_run_uuid=None,
    optimization_run_job_dir=None,
    optimization_run_uuid=None,
    # combine serval of these options
    npoints_band: int = 100,
    kpoint_density_dos: int = 7000,
    tol_imaginary_modes: float = 1e-5,
    tmin=0,
    tmax=500,
    tstep=10,
    units="THz",
    img_format="eps",
    create_thermal_displacements=True,
    freq_min_thermal_displacements=0.0,
    tmin_thermal_displacements=0,
    tmax_thermal_displacements=500,
    tstep_thermal_displacements=100,
    store_force_constants=True,
):
    """
    Compute phonon band structures and density of states.

    Parameters
    ----------

    """
    phonon_doc = PhononBSDOSDoc.from_forces_born(
        structure=structure,
        supercell_matrix=supercell_matrix,
        displacement=displacement,
        sym_reduce=sym_reduce,
        symprec=symprec,
        use_standard_primitive=use_standard_primitive,
        kpath_scheme=kpath_scheme,
        code=code,
        displacement_data=displacement_data,
        total_energy=total_energy,
        epsilon_static=epsilon_static,
        born=born,
        full_born=full_born,
        npoints_band=npoints_band,
        kpoint_density_dos=kpoint_density_dos,
        tol_imaginary_modes=tol_imaginary_modes,
        tmin=tmin,
        tmax=tmax,
        tstep=tstep,
        units=units,
        img_format=img_format,
        freq_min_thermal_displacements=freq_min_thermal_displacements,
        create_thermal_displacements=create_thermal_displacements,
        tmin_thermal_displacements=tmin_thermal_displacements,
        tmax_thermal_displacements=tmax_thermal_displacements,
        tstep_thermal_displacements=tstep_thermal_displacements,
        store_force_constants=store_force_constants,
        born_run_job_dir=born_run_job_dir,
        static_run_job_dir=static_run_job_dir,
        optimization_run_job_dir=optimization_run_job_dir,
        born_run_uuid=born_run_uuid,
        static_run_uuid=static_run_uuid,
        optimization_run_uuid=optimization_run_uuid,
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
    Note, this job will replace itself with N displacement calculations

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
    Maker to perform an static calculation as a part of the finite displacement method.

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

    # TODO: test these values!
    # TODO: change smearing?
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
