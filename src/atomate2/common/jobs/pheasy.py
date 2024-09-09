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
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)

from atomate2.common.schemas.pheasy import Forceconstants, PhononBSDOSDoc, get_factor

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
    kwargs.setdefault("force_diagonal", False)
    common_kwds = dict(
        min_length=min_length,
        min_atoms=kwargs.get("min_atoms"),
        force_diagonal=kwargs["force_diagonal"],
    )

    if not prefer_90_degrees:
        transformation = CubicSupercellTransformation(
            **common_kwds, max_atoms=kwargs.get("max_atoms"), force_90_degrees=False
        )
        transformation.apply_transformation(structure=structure)
    else:
        try:
            transformation = CubicSupercellTransformation(
                **common_kwds,
                max_atoms=kwargs.get("max_atoms", 1200),
                force_90_degrees=True,
                angle_tolerance=kwargs.get("angle_tolerance", 1e-2),
            )
            transformation.apply_transformation(structure=structure)

        except AttributeError:
            transformation = CubicSupercellTransformation(
                **common_kwds, max_atoms=kwargs.get("max_atoms"), force_90_degrees=False
            )
            transformation.apply_transformation(structure=structure)

    # matrix from pymatgen has to be transposed
    return transformation.transformation_matrix.transpose().tolist()


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
    """
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


    # following is modified by jiongzhi Zheng
    # I will use the ALM to determine how many displaced supercells we need to use for extract second order force constants here
    from alm import ALM
    supercell_z = phonon.supercell
    lattice = supercell_z.cell
    positions = supercell_z.scaled_positions
    numbers = supercell_z.numbers
    natom = len(numbers)
    with ALM(lattice, positions, numbers) as alm:
        alm.define(1)
        alm.suggest()
        n_fp = alm._get_number_of_irred_fc_elements(1)

    """Determine how many displaced supercells we need to use for extract second order force constants here
    if we want to calculate the lattice thermal conductivty here, I highly suggest you to use the finite diplacement method 
    to calculate the zero-K second order force constants which garantee you get the completely converged results"""

    num = int(np.ceil(n_fp / (3.0 * natom)))


    displacement_t = 0.01
    phonon.generate_displacements(displacement_t)
    num_disp_t = len(phonon.displacements)
    if num_disp_t > 3:
        num_d = int(np.ceil(num * 1.8))
    else:
        pass
        


    #previous version
    #if num_disp_t > 3:
    #    num_d = int(np.ceil(num_disp_t / 3.0))
    #    if num_d < num:
    #        num_d = int(num + 1)
    #    else:
    #        pass
    #else:
    #    num_d = int(num+1)

    print ("The number of free parameters of Second Order Force Constants is ", n_fp)
    print ()
    print ("The Number of Equations Used to Obtain the 2ND FCs is ", 3 * natom * num)
    print ()
    print ("Be Careful!!! Full Rank of Matrix can not always guarantee the correct result sometimes"\
           "\n if the total atoms in supercell is less than 100 and"\
           "\n lattice constants are less than 10 angstrom,"\
           "\n I highly suggest you to displace more random configurations"\
           "\n At least use one or two more configurations basd on the suggested number of displacements")
    
    displacement_f = 0.01
    phonon.generate_displacements(distance=displacement_f)

    disps = phonon.displacements

    finite_disp = False
    f_disp_n = len(disps)
    if f_disp_n > 3:
        phonon.generate_displacements(distance=displacement, number_of_snapshots=num_d, random_seed=103)
    else:
        finite_disp = True

    supercells = phonon.supercells_with_displacements

    displacements = []
    for cell in supercells:
        displacements.append(get_pmg_structure(cell))

    displacements.append(get_pmg_structure(phonon.supercell))
    return displacements


    #phonon.generate_displacements(distance=displacement)

    #supercells = phonon.supercells_with_displacements

    #return [get_pmg_structure(cell) for cell in supercells]


@job(
    output_schema=PhononBSDOSDoc,
    data=[PhononDos, PhononBandStructureSymmLine, Forceconstants],
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
    mp_id: str,
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
        mp_id=mp_id,
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
        "displaced_structures": [],
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
        phonon_job.update_maker_kwargs(
            {"_set": {"write_additional_data->phonon_info:json": info}}, dict_mod=True
        )
        phonon_jobs.append(phonon_job)
        outputs["displacement_number"] = list(range(len(displacements)))
        outputs["uuids"] = [phonon_job.output.uuid] * len(displacements)
        outputs["dirs"] = [phonon_job.output.dir_name] * len(displacements)
        outputs["forces"] = phonon_job.output.output.all_forces
        # add the displaced structures, still need to be careful with the order, experimental feature
        outputs["displaced_structures"] = displacements
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
            outputs["displaced_structures"].append(displacement)

    displacement_flow = Flow(phonon_jobs, outputs)
    return Response(replace=displacement_flow)
