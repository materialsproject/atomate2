"""Jobs for running phonon calculations."""

from __future__ import annotations

import contextlib
import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Flow, Response, job
from phonopy import Phonopy
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos

from atomate2.common.schemas.phonons import ForceConstants, PhononBSDOSDoc, get_factor
from atomate2.common.utils import get_supercell_matrix

if TYPE_CHECKING:
    from pathlib import Path

    from emmet.core.math import Matrix3D

    from atomate2.aims.jobs.base import BaseAimsMaker
    from atomate2.forcefields.jobs import ForceFieldStaticMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker


logger = logging.getLogger(__name__)


@job
def get_total_energy_per_cell(
    total_dft_energy_per_formula_unit: float, structure: Structure
) -> float:
    """
    Job that computes total DFT energy of the cell.

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
    structure: Structure,
    min_length: float,
    max_length: float,
    prefer_90_degrees: bool,
    allow_orthorhombic: bool = False,
    **kwargs,
) -> list[list[float]]:
    """
    Determine supercell size with given min_length and max_length.

    Parameters
    ----------
    structure: Structure Object
        Input structure that will be used to determine supercell
    min_length: float
        minimum length of cell in Angstrom
    max_length: float
        maximum length of cell in Angstrom
    prefer_90_degrees: bool
        if True, the algorithm will try to find a cell with 90 degree angles first
    allow_orthorhombic: bool
        if True, orthorhombic supercells are allowed
    **kwargs:
        Additional parameters that can be set.
    """
    return get_supercell_matrix(
        allow_orthorhombic=allow_orthorhombic,
        max_length=max_length,
        min_length=min_length,
        prefer_90_degrees=prefer_90_degrees,
        structure=structure,
        **kwargs,
    )


@job(data=[Structure])
def generate_phonon_displacements(
    structure: Structure,
    supercell_matrix: np.array,
    displacement: float,
    sym_reduce: bool,
    symprec: float,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    code: str,
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

    Returns
    -------
    List[Structure]
        Displaced structures
    """
    warnings.warn(
        "Initial magnetic moments will not be considered for the determination "
        "of the symmetry of the structure and thus will be removed now.",
        stacklevel=2,
    )
    if "magmom" in structure.site_properties:
        # remove_site_property is in-place so make a structure copy first
        no_mag_struct = structure.copy().remove_site_property(property_name="magmom")
    else:
        no_mag_struct = structure
    cell = get_phonopy_structure(no_mag_struct)
    factor = get_factor(code)

    # a bit of code repetition here as I currently
    # do not see how to pass the phonopy object?
    if use_symmetrized_structure == "primitive" and kpath_scheme != "seekpath":
        primitive_matrix: np.ndarray | str = np.eye(3)
    else:
        primitive_matrix = "auto"

    # TARP: THIS IS BAD! Including for discussions sake
    if cell.magnetic_moments is not None and primitive_matrix == "auto":
        if np.any(cell.magnetic_moments != 0.0):
            raise ValueError(
                "For materials with magnetic moments, "
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


@job(data=["forces", "displaced_structures"])
def run_phonon_displacements(
    displacements: list[Structure],
    structure: Structure,
    supercell_matrix: Matrix3D,
    phonon_maker: BaseVaspMaker | ForceFieldStaticMaker | BaseAimsMaker = None,
    prev_dir: str | Path = None,
    prev_dir_argname: str = None,
    socket: bool = False,
) -> Flow:
    """
    Run phonon displacements.

    Note, this job will replace itself with N displacement calculations,
    or a single socket calculation for all displacements.

    Parameters
    ----------
    displacements: Sequence
        All displacements to calculate
    structure: Structure object
        Fully optimized structure used for phonon computations.
    supercell_matrix: Matrix3D
        supercell matrix for meta data
    phonon_maker : .BaseVaspMaker or .ForceFieldStaticMaker or .BaseAimsMaker
        A maker to use to generate dispacement calculations
    prev_dir: str or Path
        The previous working directory
    prev_dir_argname: str
        argument name for the prev_dir variable
    socket: bool
        If True use the socket-io interface to increase performance
    """
    phonon_jobs = []
    outputs: dict[str, list] = {
        "displacement_number": [],
        "forces": [],
        "uuids": [],
        "dirs": [],
    }
    phonon_job_kwargs = {}
    if prev_dir is not None and prev_dir_argname is not None:
        phonon_job_kwargs[prev_dir_argname] = prev_dir

    if socket:
        phonon_job = phonon_maker.make(displacements, **phonon_job_kwargs)
        info = {
            "original_structure": structure,
            "supercell_matrix": supercell_matrix,
            "displaced_structures": displacements,
        }

        # phonon_job.update_maker_kwargs(
        #    {"_set": {"write_additional_data->phonon_info:json": info}}, dict_mod=True
        # )
        phonon_jobs.append(phonon_job)
        outputs["displacement_number"] = list(range(len(displacements)))
        outputs["uuids"] = [phonon_job.output.uuid] * len(displacements)
        outputs["dirs"] = [phonon_job.output.dir_name] * len(displacements)
        outputs["forces"] = phonon_job.output.output.all_forces
    else:
        for idx, displacement in enumerate(displacements):
            if prev_dir is not None:
                phonon_job = phonon_maker.make(displacement, prev_dir=prev_dir)
            else:
                phonon_job = phonon_maker.make(displacement)
            phonon_job.append_name(f" {idx + 1}/{len(displacements)}")

            # we will add some meta data
            info = {
                "displacement_number": idx,
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
            outputs["displacement_number"].append(idx)
            outputs["uuids"].append(phonon_job.output.uuid)
            outputs["dirs"].append(phonon_job.output.dir_name)
            outputs["forces"].append(phonon_job.output.output.forces)

    displacement_flow = Flow(phonon_jobs, outputs)
    return Response(replace=displacement_flow)


@job(data=["forces", "displaced_structures"])
def run_phonon_displacements_mod(
    displacements: list[Structure],
    structure: Structure,
    supercell_matrix: Matrix3D,
    phonon_maker: BaseVaspMaker | ForceFieldStaticMaker = None,
    prev_dir: str | Path = None,
    start_inx=0,
    batch_size=2,
    outputs: dict[str, list] = {
        "displacement_number": [],
        "forces": [],
        "uuids": [],
        "dirs": [],
    },
    stop_inx=None,
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
        supercell matrix for metadata
    phonon_maker : .BaseVaspMaker
        A VaspMaker to use to generate the elastic relaxation jobs.
    prev_dir : str or Path or None
        A previous vasp calculation directory to use for copying outputs.
    """
    stop_inx = stop_inx if stop_inx is not None else len(displacements)

    if start_inx < stop_inx:
        new_inx = start_inx + batch_size
        jobs = []
        for idx, displacement in enumerate(displacements[start_inx:new_inx]):
            if prev_dir is not None:
                phonon_job = phonon_maker.make(displacement, prev_dir=prev_dir)
            else:
                phonon_job = phonon_maker.make(displacement)
            phonon_job.append_name(f" {idx + 1 + start_inx}/{len(displacements)}")
            # print(idx+start_inx)
            # we will add some meta data
            info = {
                "displacement_number": idx + start_inx,
                "original_structure": structure,
                "supercell_matrix": supercell_matrix,
                "displaced_structure": displacement,
            }
            with contextlib.suppress(Exception):
                phonon_job.update_maker_kwargs(
                    {"_set": {"write_additional_data->phonon_info:json": info}},
                    dict_mod=True,
                )
            # outputs.append(idx+start_inx)
            outputs["displacement_number"].append(idx + start_inx)
            outputs["uuids"].append(phonon_job.output.uuid)
            outputs["dirs"].append(phonon_job.output.dir_name)
            outputs["forces"].append(phonon_job.output.output.forces)
            print(outputs)
            # print(outputs['uuids'])
            jobs.append(phonon_job)

        new_job = run_phonon_displacements_mod(
            structure=structure,
            supercell_matrix=supercell_matrix,
            displacements=displacements,
            start_inx=new_inx,
            stop_inx=stop_inx,
            phonon_maker=phonon_maker,
            prev_dir=prev_dir,
        )

        return Response(addition=[new_job, *jobs], output=outputs)


@job(data=["forces", "displaced_structures"])
def run_phonon_displacements_mod2(
    total_displacements: int,
    displacements: list[Structure],
    structure: Structure,
    supercell_matrix: Matrix3D,
    phonon_maker: BaseVaspMaker | ForceFieldStaticMaker = None,
    prev_dir: str | Path = None,
    indexs=list[int],
    outputs: dict[str, list] = {
        "displacement_number": [],
        "forces": [],
        "uuids": [],
        "dirs": [],
    },
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
        supercell matrix for metadata
    phonon_maker : .BaseVaspMaker
        A VaspMaker to use to generate the elastic relaxation jobs.
    prev_dir : str or Path or None
        A previous vasp calculation directory to use for copying outputs.
    """
    jobs = []
    for idx, displacement in enumerate(displacements):
        if prev_dir is not None:
            phonon_job = phonon_maker.make(displacement, prev_dir=prev_dir)
        else:
            phonon_job = phonon_maker.make(displacement)
        phonon_job.append_name(f" {indexs[idx]}/{total_displacements}")
        # print(idx+start_inx)
        # we will add some meta data
        info = {
            "displacement_number": indexs[idx] - 1,
            "original_structure": structure,
            "supercell_matrix": supercell_matrix,
            "displaced_structures": displacement,
        }
        with contextlib.suppress(Exception):
            phonon_job.update_maker_kwargs(
                {"_set": {"write_additional_data->phonon_info:json": info}},
                dict_mod=True,
            )
        # outputs.append(idx+start_inx)
        outputs["displacement_number"].append(indexs[idx] - 1)
        outputs["uuids"].append(phonon_job.output.uuid)
        outputs["dirs"].append(phonon_job.output.dir_name)
        outputs["forces"].append(phonon_job.output.output.forces)
        # print(outputs['uuids'])
        jobs.append(phonon_job)

    displacement_flow = Flow(jobs)
    return Response(replace=displacement_flow, output=outputs)


@job(data=["forces", "displaced_structures"])
def run_phonon_displacements_recur(
    total_displacements: int,
    index_list,
    chunks_list,
    structure: Structure,
    supercell_matrix: Matrix3D,
    phonon_maker: BaseVaspMaker | ForceFieldStaticMaker = None,
    prev_dir: str | Path = None,
    start: int = 0,
    outputs: dict[str, list] = {
        "displacement_number": [],
        "forces": [],
        "uuids": [],
        "dirs": [],
    },
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
        supercell matrix for metadata
    phonon_maker : .BaseVaspMaker
        A VaspMaker to use to generate the elastic relaxation jobs.
    prev_dir : str or Path or None
        A previous vasp calculation directory to use for copying outputs.
    """
    index_list_here = index_list[start]
    disp_here = chunks_list[start]
    jobs = []
    for idx, displacement in enumerate(disp_here):
        if prev_dir is not None:
            phonon_job = phonon_maker.make(displacement, prev_dir=prev_dir)
        else:
            phonon_job = phonon_maker.make(displacement)
        phonon_job.append_name(
            f" {index_list_here[idx]}/{total_displacements} : batch {start + 1}"
        )
        info = {
            "displacement_number": index_list_here[idx] - 1,
            "original_structure": structure,
            "supercell_matrix": supercell_matrix,
            "displaced_structures": displacement,
        }
        with contextlib.suppress(Exception):
            phonon_job.update_maker_kwargs(
                {"_set": {"write_additional_data->phonon_info:json": info}},
                dict_mod=True,
            )
        jobs.append(phonon_job)
        outputs["displacement_number"].append(index_list_here[idx] - 1)
        outputs["uuids"].append(phonon_job.output.uuid)
        outputs["dirs"].append(phonon_job.output.dir_name)
        outputs["forces"].append(phonon_job.output.output.forces)

    if start + 1 != len(index_list):
        start = start + 1
        new_job = run_phonon_displacements_recur(
            structure=structure,
            supercell_matrix=supercell_matrix,
            chunks_list=chunks_list,
            index_list=index_list,
            total_displacements=total_displacements,
            phonon_maker=phonon_maker,
            prev_dir=prev_dir,
            start=start,
        )

        displacement_flow = Flow([*jobs, new_job])
        return Response(addition=displacement_flow, output=outputs)
    displacement_flow = Flow(jobs)
    return Response(addition=displacement_flow, output=outputs)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst along with their start and end indices."""
    for start_index, i in enumerate(range(0, len(lst), n)):
        end_index = min(i + n, len(lst))
        yield lst[i:end_index], i, end_index


@job(data=["forces"])
def chunk_and_aggregate2(
    displacement: float,
    sym_reduce: bool,
    symprec: float,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    code: str,
    structure,
    supercell_matrix,
    phonon_maker,
    chunk_size=3,
):
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

    # a bit of code repetition here as I currently
    # do not see how to pass the phonopy object?
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

    displacements = [get_pmg_structure(cell) for cell in supercells]

    jobs = []
    outputs: dict[str, list] = {
        "displacement_number": None,
        "forces": None,
        "uuids": None,
        "dirs": None,
    }
    for chunk, start_index, end_index in chunks(displacements, chunk_size):
        indexs = list(range(start_index + 1, end_index + 1, 1))

        job = run_phonon_displacements_mod2(
            total_displacements=len(displacements),
            displacements=chunk,
            structure=structure,
            supercell_matrix=supercell_matrix,
            phonon_maker=phonon_maker,
            indexs=indexs,
        )
        jobs.append(job)
        outputs["displacement_number"] = job.output["displacement_number"]
        outputs["uuids"] = job.output["uuids"]
        outputs["dirs"] = job.output["dirs"]
        outputs["forces"] = job.output["forces"]

    displacement_flow = Flow(jobs)
    return Response(
        replace=displacement_flow, output=outputs
    )  # Response(addition=jobs, output=outputs)


@job
def chunk_and_aggregate(
    displacements,
    structure,
    supercell_matrix,
    phonon_maker,
    chunk_size=3,
):
    jobs = []
    outputs: dict[str, list] = {
        "displacement_number": None,
        "forces": None,
        "uuids": None,
        "dirs": None,
    }
    for chunk, start_index, end_index in chunks(displacements, chunk_size):
        indexs = list(range(start_index + 1, end_index + 1, 1))

        job = run_phonon_displacements_mod2(
            total_displacements=len(displacements),
            displacements=chunk,
            structure=structure,
            supercell_matrix=supercell_matrix,
            phonon_maker=phonon_maker,
            indexs=indexs,
        )
        jobs.append(job)
        outputs["displacement_number"] = job.output["displacement_number"]
        outputs["uuids"] = job.output["uuids"]
        outputs["dirs"] = job.output["dirs"]
        outputs["forces"] = job.output["forces"]

    return Response(replace=jobs, output=outputs)


@job(data=["forces", Structure, "chunks_list", "index_list"])
def chunk_and_aggregate_recur(
    displacement: float,
    sym_reduce: bool,
    symprec: float,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    code: str,
    structure,
    supercell_matrix,
    phonon_maker,
    chunk_size=3,
):
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

    # a bit of code repetition here as I currently
    # do not see how to pass the phonopy object?
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

    displacements = [get_pmg_structure(cell) for cell in supercells]

    total_displacements = len(displacements)
    recur_function_inputs = {"chunks_list": [], "index_list": []}
    # chunks_list = []
    # index_list = []
    for chunk, start_index, end_index in chunks(displacements, chunk_size):
        indexs = list(range(start_index + 1, end_index + 1, 1))
        recur_function_inputs["chunks_list"].append(chunk)
        recur_function_inputs["index_list"].append(indexs)
        # chunks_list.append(chunk)
        # index_list.append(indexs)

    del supercells, displacements

    displ_job = run_phonon_displacements_recur(
        total_displacements=total_displacements,
        structure=structure,
        supercell_matrix=supercell_matrix,
        phonon_maker=phonon_maker,
        chunks_list=recur_function_inputs["chunks_list"],
        index_list=recur_function_inputs["index_list"],
        start=0,
    )

    displacement_flow = Flow(displ_job, output=displ_job.output)
    return Response(
        replace=displacement_flow
    )  # Response(addition=jobs, output=outputs)


@job(data=["forces", Structure, "chunks_list", "index_list"])
def all_jobs(
    displacement: float,
    sym_reduce: bool,
    symprec: float,
    kpath_scheme: str,
    code: str,
    structure,
    supercell_matrix,
    use_symmetrized_structure: str | None,
    phonon_maker: BaseVaspMaker | ForceFieldStaticMaker = None,
    prev_dir: str | Path = None,
    chunk_size=3,
):
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

    # a bit of code repetition here as I currently
    # do not see how to pass the phonopy object?
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

    displacements = [get_pmg_structure(cell) for cell in supercells]

    del supercells

    outputs: dict[str, list] = {
        "displacement_number": [],
        "forces": [],
        "uuids": [],
        "dirs": [],
    }

    for idx, displacement in enumerate(displacements):
        if prev_dir is not None:
            phonon_job = phonon_maker.make(displacement, prev_dir=prev_dir)
        else:
            phonon_job = phonon_maker.make(displacement)
        phonon_job.append_name(f" {[idx + 1]}/{len(displacements)}")
        info = {
            "displacement_number": idx,
            "original_structure": structure,
            "supercell_matrix": supercell_matrix,
            "displaced_structures": displacement,
        }
        with contextlib.suppress(Exception):
            phonon_job.update_maker_kwargs(
                {"_set": {"write_additional_data->phonon_info:json": info}},
                dict_mod=True,
            )
        outputs["displacement_number"].append(idx - 1)
        outputs["uuids"].append(phonon_job.output.uuid)
        outputs["dirs"].append(phonon_job.output.dir_name)
        outputs["forces"].append(phonon_job.output.output.forces)

        displacement_flow = Flow(phonon_job)
        return Response(addition=displacement_flow)
