from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
from jobflow import Flow, Response, job
from phonopy import Phonopy, load
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.symmetry import elaborate_borns_and_epsilon
from phonopy.units import VaspToTHz
from pymatgen.core import Structure
from pymatgen.io.phonopy import (
    get_ph_bs_symm_line,
    get_ph_dos,
    get_phonopy_structure,
    get_pmg_structure,
)
from pymatgen.io.vasp import Kpoints
from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.symmetry.kpath import KPathSeek
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)

from atomate2.common.schemas.math import Matrix3D
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.schemas.phonons import PhononBSDOSDoc
from atomate2.vasp.sets.base import VaspInputSetGenerator
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
def get_supercell_size(structure: Structure, min_length: float):
    transformation = CubicSupercellTransformation(min_length=min_length)
    transformation.apply_transformation(structure=structure)
    supercell_matrix = transformation.transformation_matrix.tolist()
    return supercell_matrix


def get_phonon_object(
    structure: Structure,
    supercell_matrix: np.array,
    displacement: float,
    sym_reduce: bool,
    symprec: float,
    use_standard_primitive: bool,
    kpath_scheme: str,
    code: str,
):
    if code == "vasp":
        factor = VaspToTHz
    # TODO: add other codes?

    cell = get_phonopy_structure(structure)
    if use_standard_primitive and kpath_scheme != "seekpath":
        phonon = Phonopy(
            cell,
            supercell_matrix,
            primitive_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            factor=factor,
            symprec=symprec,
            is_symmetry=sym_reduce,
        )
    else:
        phonon = Phonopy(
            cell,
            supercell_matrix,
            primitive_matrix="auto",
            factor=factor,
            symprec=symprec,
            is_symmetry=sym_reduce,
        )
    phonon.generate_displacements(distance=displacement)
    return phonon


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
    phonopy_object = get_phonon_object(
        structure=structure,
        supercell_matrix=np.array(supercell_matrix),
        displacement=displacement,
        sym_reduce=sym_reduce,
        symprec=symprec,
        use_standard_primitive=use_standard_primitive,
        kpath_scheme=kpath_scheme,
        code=code,
    )
    supercells = phonopy_object.supercells_with_displacements

    displacements = []
    for cell in supercells:
        displacements.append(get_pmg_structure(cell))
    return displacements


@job(output_schema=PhononBSDOSDoc)
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
    npoints_band: int = 100,
    kpoint_density_dos: int = 7000,
    tol_imaginary_modes: float = 1e-5,
    tmin=0,
    tmax=500,
    tstep=10,
    units="THz",
    img_format="eps",
):
    """
    Compute phonon band structures and density of states.

    Parameters
    ----------

    """
    # TODO: move his part to another class?
    def get_kpath(structure: Structure, kpath_scheme: str, **kpath_kwargs):
        """
        get high-symmetry points in k-space
        Args:
            structure: Structure Object
        Returns:
        """

        if kpath_scheme in [
            "setyawan_curtarolo",
            "hinuma",
            "latimer_munro",
            "all_pymatgen",
        ]:
            if kpath_scheme == "all_pymatgen":
                kpath_scheme = "all"
            highsymmkpath = HighSymmKpath(
                structure, path_type=kpath_scheme, **kpath_kwargs
            )
            kpath = highsymmkpath.kpath
        elif kpath_scheme == "seekpath":
            highsymmkpath = KPathSeek(structure, **kpath_kwargs)
            kpath = highsymmkpath._kpath

        path = copy.deepcopy(kpath["path"])

        for ilabelset, labelset in enumerate(kpath["path"]):
            for ilabel, label in enumerate(labelset):
                path[ilabelset][ilabel] = kpath["kpoints"][label]
        return kpath["kpoints"], path

    # have to regenerate this object as I cannot make it a job output
    # TODO: other way?
    phonon = get_phonon_object(
        structure=structure,
        supercell_matrix=supercell_matrix,
        displacement=displacement,
        sym_reduce=sym_reduce,
        symprec=symprec,
        use_standard_primitive=use_standard_primitive,
        kpath_scheme=kpath_scheme,
        code=code,
    )

    set_of_forces = [np.array(forces) for forces in displacement_data["forces"]]

    if born is not None:
        if full_born:
            # TODO: if this is a good way when user provide data
            borns, epsilon, atom_indices = elaborate_borns_and_epsilon(
                ucell=get_phonopy_structure(structure),
                borns=np.array(born),
                epsilon=np.array(epsilon_static),
                symprec=symprec,
                primitive_matrix=phonon.primitive_matrix,
                supercell_matrix=phonon.supercell_matrix,
            )
        else:
            borns = born
        if code == "vasp":
            phonon.nac_params = {
                "born": borns,
                "dielectric": epsilon,
                "factor": 14.399652,
            }
    phonon.produce_force_constants(forces=set_of_forces)

    # Currently, the setting of the born
    # charges does not work without this! I am not sure why!
    # I don't see it
    # I think I am missing something in the code
    phonon.save("phonopy.yaml")
    phonon = load("phonopy.yaml")
    # get phonon band structure
    # TODO: check if this is correct for every possible setting
    kpath_dict, kpath_concrete = get_kpath(structure, kpath_scheme)
    qpoints, connections = get_band_qpoints_and_path_connections(
        kpath_concrete, npoints=npoints_band
    )

    # add option to disable phonon bandstructure computation?
    filename_band_yaml = "phonon_band_structure.yaml"
    phonon.run_band_structure(qpoints, path_connections=connections)
    phonon.write_yaml_band_structure(filename=filename_band_yaml)
    bs_symm_line = get_ph_bs_symm_line(filename_band_yaml, labels_dict=kpath_dict)
    new_plotter = PhononBSPlotter(bs=bs_symm_line)
    new_plotter.save_plot(
        "phonon_band_structure.eps", img_format=img_format, units=units
    )
    # add a free energy document?
    imaginary_modes = bs_symm_line.has_imaginary_freq(tol=tol_imaginary_modes)

    # get phonon density of states
    filename_dos_yaml = "phonon_dos.yaml"
    kpoint = Kpoints.automatic_density(
        structure=structure, kppa=kpoint_density_dos, force_gamma=True
    )
    phonon.run_mesh(kpoint.kpts[0])
    phonon.run_total_dos()
    phonon.write_total_dos(filename=filename_dos_yaml)
    dos = get_ph_dos(filename_dos_yaml)
    new_plotter_dos = PhononDosPlotter()
    new_plotter_dos.add_dos(label="total", dos=dos)
    new_plotter_dos.save_plot(
        filename="phonon_dos.eps", img_format=img_format, units=units
    )

    # add tmin tmax tstep
    temperature_range = np.arange(tmin, tmax, tstep)
    free_energy = [
        dos.helmholtz_free_energy(structure=structure, t=temperature)
        for temperature in temperature_range
    ]

    # transfer the force constants to compute Grüneisen parameters?
    formula_units = (
        structure.composition.num_atoms
        / structure.composition.reduced_composition.num_atoms
    )
    # TODO: add more meta data here
    phonon_doc = PhononBSDOSDoc(
        structure=structure,
        ph_bs=bs_symm_line,
        ph_dos=dos,
        free_energy={
            "temp": temperature_range,
            "free_energy": free_energy,
            "total_energy": total_energy / formula_units,
        },
        imaginary_modes=imaginary_modes,
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

    # TODO: test these values!
    # TODO: change smearing?
    input_set_generator: VaspInputSetGenerator = field(
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
