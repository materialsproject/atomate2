"""Jobs for running phonon calculations."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Response, job
from phonopy import Phonopy
from phonopy.units import VaspToTHz
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)

from atomate2.common.schemas.phonons import ForceConstants, PhononBSDOSDoc
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.sets.core import StaticSetGenerator

if TYPE_CHECKING:
    import numpy as np
    from emmet.core.math import Matrix3D
    from pymatgen.core import Structure

    from atomate2.forcefields.jobs import ForceFieldStaticMaker
    from atomate2.vasp.sets.base import VaspInputGenerator

logger = logging.getLogger(__name__)

__all__ = [
    "get_total_energy_per_cell",
    "get_supercell_size",
    "generate_phonon_displacements",
    "run_phonon_displacements",
    "generate_frequencies_eigenvectors",
    "PhononDisplacementMaker",
]


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


@job
def generate_phonon_displacements(
    structure: Structure,
    supercell_matrix: np.array,
    displacement: float,
    sym_reduce: bool,
    symprec: float,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    code: str,
):
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
    cell = get_phonopy_structure(structure)
    if code == "vasp":
        factor = VaspToTHz
    # a bit of code repetition here as I currently
    # do not see how to pass the phonopy object?
    if use_symmetrized_structure == "primitive" and kpath_scheme != "seekpath":
        primitive_matrix: list[list[float]] | str = [
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

    return [get_pmg_structure(cell) for cell in supercells]


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
    **kwargs,
):
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
        total dft energy in eV per cell
    epsilon_static: Matrix3D
        The high-frequency dielectric constant
    born: Matrix3D
        Born charges
    kwargs: dict
        Additional parameters that are passed to PhononBSDOSDoc.from_forces_born

    """
    return PhononBSDOSDoc.from_forces_born(
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


@job
def run_phonon_displacements(
    displacements,
    structure: Structure,
    supercell_matrix,
    phonon_maker: BaseVaspMaker | ForceFieldStaticMaker = None,
):
    """
    Run phonon displacements.

    Note, this job will replace itself with N displacement calculations.

    Parameters
    ----------
    displacements
    structure: Structure object
        Fully optimized structure used for phonon computations.
    supercell_matrix: Matrix3D
        supercell matrix for meta data
    phonon_maker : .BaseVaspMaker
        A VaspMaker to use to generate the phonon jobs.
    """
    if phonon_maker is None:
        phonon_maker = PhononDisplacementMaker()
    phonon_jobs = []
    outputs: dict[str, list] = {
        "displacement_number": [],
        "forces": [],
        "uuids": [],
        "dirs": [],
        "displaced_structures": [],
    }

    for i, displacement in enumerate(displacements):
        phonon_job = phonon_maker.make(displacement)
        phonon_job.append_name(f" {i + 1}/{len(displacements)}")

        # we will add some meta data
        info = {
            "displacement_number": i,
            "original_structure": structure,
            "supercell_matrix": supercell_matrix,
            "displaced_structure": displacement,
        }
        with contextlib.suppress(Exception):
            phonon_job.update_maker_kwargs(
                {"_set": {"write_additional_data->phonon_info:json": info}},
                dict_mod=True,
            )

        phonon_jobs.append(phonon_job)
        outputs["displacement_number"].append(i)
        outputs["uuids"].append(phonon_job.output.uuid)
        outputs["dirs"].append(phonon_job.output.dir_name)
        outputs["forces"].append(phonon_job.output.output.forces)
        outputs["displaced_structures"].append(displacement)

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
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
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
