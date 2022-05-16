from __future__ import annotations

import copy
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from jobflow import Flow, Response, job
from phonopy import Phonopy
from phonopy.interface.vasp import get_born_vasprunxml, parse_set_of_forces
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.units import VaspToTHz
from pymatgen.analysis.elasticity import (
    Deformation,
)
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_ph_bs_symm_line
from pymatgen.io.phonopy import get_ph_dos
from pymatgen.io.phonopy import get_pmg_structure, get_phonopy_structure
from pymatgen.io.vasp import Kpoints
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation

from atomate2 import SETTINGS
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.schemas.phonons import PhononBSDOSDoc
from atomate2.vasp.sets.base import VaspInputSetGenerator
from atomate2.vasp.sets.core import StaticSetGenerator

logger = logging.getLogger(__name__)

__all__ = [
    "generate_phonon_displacements",
    "run_phonon_displacements",
    "generate_frequencies_eigenvectors",
    "PhononDisplacementMaker"
]


def get_phonon_object(displacement, min_length, structure, sym_reduce, symprec, conventional):
    transformation = CubicSupercellTransformation(min_length=min_length)
    transformation.apply_transformation(structure=structure)
    supercell_matrix = transformation.transformation_matrix.tolist()
    cell = get_phonopy_structure(structure)
    if not conventional:
        phonon = Phonopy(cell,
                         supercell_matrix,
                         primitive_matrix=[[1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 0.0, 1.0]],
                         factor=VaspToTHz,
                         symprec=symprec,
                         is_symmetry=sym_reduce)
    else:
        phonon = Phonopy(cell,
                         supercell_matrix,
                         primitive_matrix='auto',
                         factor=VaspToTHz,
                         symprec=symprec,
                         is_symmetry=sym_reduce)
    phonon.generate_displacements(distance=displacement)
    return phonon


# check if this can also be replaced with something better
def get_kpath(structure: Structure, conventional, **kpath_kwargs):
    """
    get high-symmetry points in k-space
    Args:
        structure: Structure Object
    Returns:
    """
    highsymmkpath = HighSymmKpath(structure, **kpath_kwargs)
    kpath = highsymmkpath.kpath
    path = copy.deepcopy(kpath["path"])

    for ilabelset, labelset in enumerate(kpath["path"]):
        for ilabel, label in enumerate(labelset):
            path[ilabelset][ilabel] = kpath["kpoints"][label]
    return kpath["kpoints"], path


# TODO: check all parameters again
@job
def generate_phonon_displacements(
        structure: Structure,
        displacement: float = 0.01,
        min_length: float = 4.0,
        conventional: bool = False,
        symprec: float = SETTINGS.SYMPREC,
        sym_reduce: bool = True,
):
    """
    Generate elastic deformations.

    Parameters
    ----------
    structure : Structure
        A pymatgen structure object.
    displacement : float
        The displacement to be applied to the structure.
    min_length  : float
        minimum supercell size.
    symprec : float
        The symprec to use for the spacegroup analyzer.
    sym_reduce : bool
        Whether to reduce the symmetry of the structure.

    Returns
    -------
    List[Deformation]
        A list of diplacements.
    """
    # TODO: use functions from pymatgen instead?
    phonon = get_phonon_object(displacement, min_length, structure, sym_reduce, symprec)
    supercells = phonon.supercells_with_displacements

    displacements = []
    for cell in supercells:
        displacements.append(get_pmg_structure(cell))
    return displacements


@job(output_schema=PhononBSDOSDoc)
def generate_frequencies_eigenvectors(
        structure: Structure,
        displacement_data: list[dict],
        born_data: str | Path = None,
        symprec: float = SETTINGS.SYMPREC,
        sym_reduce: bool = True,
        displacement: float = 0.01,
        min_length: float = 4.0,
        conventional: bool = False,
        npoints_band: int = 100,
        kpoint_density_dos: int = 7000,
):
    """
    Compute phonon band structures and density of states.

    Parameters
    ----------

    """
    # get phonon object from phonopy with correct settings again

    phonon = get_phonon_object(conventional, displacement, min_length, structure, sym_reduce, symprec)

    # do this file based even though force based would be better
    forces_filenames = []
    # Vasprunxml is missing each time
    for displacement in displacement_data:
        # incompressed files should be handled as well
        forces_filenames.append(str(Path(displacement["job_dir"]) / "vasprun.xml.gz").split(":")[1])

    set_of_forces = parse_set_of_forces(num_atoms=get_pmg_structure(phonon.supercell).num_sites,
                                        forces_filenames=forces_filenames)

    # produce force constants
    # decompress_file(str((Path(born_data) / "vasprun.xml.gz").split(":")[1]))

    phonon.produce_force_constants(forces=set_of_forces)
    # for some reason server address will be included in the path?
    # deal with uncompressed files
    borns, epsilon, atom_indices = get_born_vasprunxml(str(Path(born_data) / "vasprun.xml.gz").split(":")[1],
                                                       primitive_matrix=phonon.primitive_matrix,
                                                       supercell_matrix=phonon.supercell_matrix)
    # compress_file(str(zpath(Path(born_data) / "vasprun.xml")))
    # get born charges from vasprun.xml

    phonon.nac_params = {"born": borns, "dielectric": epsilon, "factor": 14.400}

    # get phonon band structure
    tempfilename = tempfile.gettempprefix() + '.yaml'
    kpath_dict, kpath_concrete = get_kpath(structure, conventional)
    qpoints, connections = get_band_qpoints_and_path_connections(kpath_concrete, npoints=npoints_band)

    phonon.run_band_structure(qpoints, path_connections=connections)
    phonon.write_yaml_band_structure(
        filename=tempfilename)
    bs_symm_line = get_ph_bs_symm_line(tempfilename, labels_dict=kpath_dict)

    # get phonon density of states
    tempfilename = tempfile.gettempprefix() + '.yaml'
    kpoint = Kpoints.automatic_density(structure=structure, kppa=kpoint_density_dos, force_gamma=True)
    phonon.run_mesh(kpoint.kpts[0])
    phonon.run_total_dos()
    phonon.write_total_dos(filename=tempfilename)
    dos = get_ph_dos(tempfilename)

    # get thermal properties
    # TODO: add computation of thermal properties as well


    # check if any imaginary modes exist and maybe add a percentage?

    # maybe, we can just give the folder and phonopy can create it?
    # do something to generate a phonon document
    phonon_doc = PhononBSDOSDoc(structure=structure, ph_bs=bs_symm_line, ph_dos=dos)
    print(phonon_doc)
    return phonon_doc


@job
def run_phonon_displacements(
        displacements,
        phonon_maker: BaseVaspMaker = None,
):
    """
    Run elastic deformations.

    Note, this job will replace itself with N relaxation calculations, where N is
    the number of deformations.

    Parameters
    ----------
    structure : Structure
        A pymatgen structure.
    deformations : list of Deformation
        The deformations to apply.
    prev_vasp_dir : str or Path or None
        A previous VASP directory to use for copying VASP outputs.
    phonon_maker : .BaseVaspMaker
        A VaspMaker to use to generate the elastic relaxation jobs.
    """
    if phonon_maker is None:
        phonon_maker = PhononDisplacementMaker()
    phonon_runs = []
    outputs = []
    for i, displacement in enumerate(displacements):
        phonon_job = phonon_maker.make(
            displacement
        )
        phonon_job.append_name(f" {i + 1}/{len(displacements)}")
        phonon_runs.append(phonon_job)

        # extract the outputs we want
        # maybe add forces as well later on
        output = {
            "displacement_number": i,
            "uuid": phonon_job.output.uuid,
            "job_dir": phonon_job.output.dir_name,
        }

        outputs.append(output)

    relax_flow = Flow(phonon_runs, outputs)
    return Response(replace=relax_flow)


@dataclass
class PhononDisplacementMaker(BaseVaspMaker):
    """
    Maker to perform an static calculation as a part of the finite displacement method.

    The input set is for a static run with tighter convergence parameters. Both the k-point mesh density and convergence parameters
    are stricter than a normal relaxation.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputSetGenerator
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
    input_set_generator: VaspInputSetGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings={"grid_density": 100},
            user_incar_settings={
                "IBRION": 2,
                "ISIF": 3,
                "ENCUT": 700,
                "EDIFF": 1e-7,
                "LAECHG": False,
                "EDIFFG": -0.001,
                "LREAL": False,
                "ALGO": "Normal",
                "NSW": 0,
                "LCHARG": False,
                "ISMEAR": 0,
                "NPAR": 4,
            },
        )
    )
