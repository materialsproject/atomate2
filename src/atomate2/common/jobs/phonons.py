"""Jobs for running phonon calculations."""

from __future__ import annotations

import contextlib
import copy
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from emmet.core.phonon import PhononBSDOSDoc
from jobflow import Flow, Response, job
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon
from phonopy.units import VaspToTHz
from pymatgen.core import Structure
from pymatgen.io.phonopy import (
    get_ph_bs_symm_line,
    get_ph_dos,
    get_phonopy_structure,
    get_pmg_structure,
)
from pymatgen.io.vasp import Kpoints
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.symmetry.kpath import KPathSeek

from atomate2.aims.utils.units import omegaToTHz
from atomate2.common.utils import get_supercell_matrix

if TYPE_CHECKING:
    from emmet.core.math import Matrix3D

    from atomate2.aims.jobs.base import BaseAimsMaker
    from atomate2.forcefields.jobs import ForceFieldStaticMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker


logger = logging.getLogger(__name__)


def get_factor(code: str) -> float:
    """
    Get the frequency conversion factor to THz for each code.

    Parameters
    ----------
    code: str
        The code to get the conversion factor for

    Returns
    -------
    float
        The correct conversion factor

    Raises
    ------
    ValueError
        If code is not defined
    """
    if code in ["ase", "forcefields", "vasp"]:
        return VaspToTHz
    if code == "aims":
        return omegaToTHz  # Based on CODATA 2002
    raise ValueError(f"Frequency conversion factor for code ({code}) not defined.")


def _get_kpath(
    structure: Structure, kpath_scheme: str, symprec: float, **kpath_kwargs
) -> tuple:
    """Get high-symmetry points in k-space in phonopy format.

    Parameters
    ----------
    structure: Structure Object
    kpath_scheme: str
        string describing kpath
    symprec: float
        precision for symmetry determination
    **kpath_kwargs:
        additional parameters that can be passed to this method as a dict
    """
    valid_schemes = {"setyawan_curtarolo", "latimer_munro", "hinuma", "seekpath"}
    if kpath_scheme in (valid_schemes - {"seekpath"}):
        high_symm_kpath = HighSymmKpath(
            structure, path_type=kpath_scheme, symprec=symprec, **kpath_kwargs
        )
        kpath = high_symm_kpath.kpath
    elif kpath_scheme == "seekpath":
        high_symm_kpath = KPathSeek(structure, symprec=symprec, **kpath_kwargs)
        kpath = high_symm_kpath._kpath  # noqa: SLF001
    else:
        raise ValueError(f"Unexpected {kpath_scheme=}, must be one of {valid_schemes}")

    path = copy.deepcopy(kpath["path"])

    for set_idx, label_set in enumerate(kpath["path"]):
        for lbl_idx, label in enumerate(label_set):
            path[set_idx][lbl_idx] = kpath["kpoints"][label]
    return kpath["kpoints"], path


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


def _generate_phonon_object(
    structure: Structure,
    supercell_matrix: np.array,
    displacement: float,
    sym_reduce: bool,
    symprec: float,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    code: str,
    verbose: bool = False,
) -> Phonopy:
    """Bundle commonly-used Phonopy object construction.

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
    use_min_dof : bool = False
        Whether to use the minimal number of degrees of freedom
        in calculating randomly-displaced structures.
        Requires ALAMODE.
    verbose : bool = False
        Whether to log error messages.

    Returns
    -------
    Phonopy object.
    """
    if "magmom" in structure.site_properties and verbose:
        warnings.warn(
            "Initial magnetic moments will not be considered for the determination "
            "of the symmetry of the structure and thus will be removed now.",
            stacklevel=2,
        )

    cell = get_phonopy_structure(
        structure.copy().remove_site_property(property_name="magmom")
        if "magmom" in structure.site_properties
        else structure.copy()
    )
    factor = get_factor(code)

    # a bit of code repetition here as I currently
    # do not see how to pass the phonopy object?
    primitive_matrix: np.ndarray | str = (
        np.eye(3)
        if use_symmetrized_structure == "primitive" and kpath_scheme != "seekpath"
        else "auto"
    )

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
    return phonon


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
    verbose: bool = False,
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
    use_min_dof : bool = False
        Whether to use the minimal number of degrees of freedom
        in calculating randomly-displaced structures.
        Requires ALAMODE.
    verbose : bool = False
        Whether to log error messages.

    Returns
    -------
    List[Structure]
        Displaced structures
    """
    phonon = _generate_phonon_object(
        structure,
        supercell_matrix,
        displacement,
        sym_reduce,
        symprec,
        use_symmetrized_structure,
        kpath_scheme,
        code,
        verbose=verbose,
    )
    supercells = phonon.supercells_with_displacements
    return [get_pmg_structure(cell) for cell in supercells]


@job(
    output_schema=PhononBSDOSDoc,
    data=[PhononDos, PhononBandStructureSymmLine, "force_constants"],
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
    epsilon_static: Matrix3D | None = None,
    born: Matrix3D | None = None,
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
    phonon = _generate_phonon_object(
        structure,
        supercell_matrix,
        displacement,
        sym_reduce,
        symprec,
        use_symmetrized_structure,
        kpath_scheme,
        code,
        verbose=False,
    )
    set_of_forces = [np.array(forces) for forces in displacement_data["forces"]]

    if born is not None and epsilon_static is not None:
        if len(structure) == len(born):
            borns, epsilon = symmetrize_borns_and_epsilon(
                ucell=phonon.unitcell,
                borns=np.array(born),
                epsilon=np.array(epsilon_static),
                symprec=symprec,
                primitive_matrix=phonon.primitive_matrix,
                supercell_matrix=phonon.supercell_matrix,
                is_symmetry=kwargs.get("symmetrize_born", True),
            )
        else:
            raise ValueError(
                "Number of Born charges does not agree with number of atoms"
            )
        if code == "vasp" and not np.all(np.isclose(borns, 0.0)):
            phonon.nac_params = {
                "born": borns,
                "dielectric": epsilon,
                "factor": 14.399652,  # TODO: where is this magic number coming from?
            }
        # Other codes could be added here
    else:
        borns = None
        epsilon = None

    # Produces all force constants
    phonon.produce_force_constants(forces=set_of_forces)

    filename_phonopy_yaml = kwargs.get("filename_phonopy_yaml", "phonopy.yaml")
    create_force_constants_file = kwargs.get("create_force_constants_file", False)
    force_constants_filename = kwargs.get("force_constants_filename", "FORCE_CONSTANTS")
    phonon.save(
        filename_phonopy_yaml,
        settings={
            "force_constants": kwargs.get(
                "store_force_constants", not create_force_constants_file
            )
        },
    )
    if create_force_constants_file:
        from phonopy.file_IO import write_FORCE_CONSTANTS

        write_FORCE_CONSTANTS(  # save force_constants to text file
            phonon.force_constants, filename=force_constants_filename
        )

    # get phonon band structure
    kpath_dict, kpath_concrete = _get_kpath(
        structure=get_pmg_structure(phonon.primitive),
        kpath_scheme=kpath_scheme,
        symprec=symprec,
    )

    npoints_band = kwargs.get("npoints_band", 101)
    qpoints, connections = get_band_qpoints_and_path_connections(
        kpath_concrete, npoints=npoints_band
    )

    # phonon band structures will always be computed
    filename_band_yaml = kwargs.get("filename_band_yaml", "phonon_band_structure.yaml")
    # filename_band_yaml = "phonon_band_structure.yaml"

    # TODO: potentially add kwargs to avoid computation of eigenvectors
    phonon.run_band_structure(
        qpoints,
        path_connections=connections,
        with_eigenvectors=kwargs.get("band_structure_eigenvectors", False),
        is_band_connection=kwargs.get("band_structure_eigenvectors", False),
    )
    phonon.write_yaml_band_structure(filename=filename_band_yaml)
    bs_symm_line = get_ph_bs_symm_line(
        filename_band_yaml, labels_dict=kpath_dict, has_nac=born is not None
    )
    new_plotter = PhononBSPlotter(bs=bs_symm_line)
    new_plotter.save_plot(
        filename=kwargs.get("filename_bs", "phonon_band_structure.pdf"),
        units=kwargs.get("units", "THz"),
    )

    # will determine if imaginary modes are present in the structure
    imaginary_modes = bs_symm_line.has_imaginary_freq(
        tol=kwargs.get("tol_imaginary_modes", 1e-5)
    )

    # gets data for visualization on website - yaml is also enough
    if kwargs.get("band_structure_eigenvectors"):
        bs_symm_line.write_phononwebsite("phonon_website.json")

    # get phonon density of states
    filename_dos_yaml = kwargs.get("filename_dos_yaml", "phonon_dos.yaml")
    # filename_dos_yaml = "phonon_dos.yaml"

    kpoint_density_dos = kwargs.get("kpoint_density_dos", 7_000)
    kpoint = Kpoints.automatic_density(
        structure=get_pmg_structure(phonon.primitive),
        kppa=kpoint_density_dos,
        force_gamma=True,
    )

    # projected dos
    if kwargs.get("calculate_pdos", False):
        phonon.run_mesh(kpoint.kpts[0], with_eigenvectors=True, is_mesh_symmetry=False)
        phonon_dos_sigma = kwargs.get("phonon_dos_sigma")
        dos_use_tetrahedron_method = kwargs.get("dos_use_tetrahedron_method", True)
        phonon.run_projected_dos(
            sigma=phonon_dos_sigma,
            use_tetrahedron_method=dos_use_tetrahedron_method,
        )
        phonon.write_projected_dos()

    phonon.run_mesh(kpoint.kpts[0])
    phonon_dos_sigma = kwargs.get("phonon_dos_sigma")
    dos_use_tetrahedron_method = kwargs.get("dos_use_tetrahedron_method", True)
    phonon.run_total_dos(
        sigma=phonon_dos_sigma, use_tetrahedron_method=dos_use_tetrahedron_method
    )
    phonon.write_total_dos(filename=filename_dos_yaml)
    dos = get_ph_dos(filename_dos_yaml)
    new_plotter_dos = PhononDosPlotter()
    new_plotter_dos.add_dos(label="total", dos=dos)
    new_plotter_dos.save_plot(
        filename=kwargs.get("filename_dos", "phonon_dos.pdf"),
        units=kwargs.get("units", "THz"),
    )

    # will compute thermal displacement matrices
    # for the primitive cell (phonon.primitive!)
    # only this is available in phonopy
    if kwargs.get("create_thermal_displacements"):
        phonon.run_mesh(kpoint.kpts[0], with_eigenvectors=True, is_mesh_symmetry=False)
        freq_min_thermal_displacements = kwargs.get(
            "freq_min_thermal_displacements", 0.0
        )
        phonon.run_thermal_displacement_matrices(
            t_min=kwargs.get("tmin_thermal_displacements", 0),
            t_max=kwargs.get("tmax_thermal_displacements", 500),
            t_step=kwargs.get("tstep_thermal_displacements", 100),
            freq_min=freq_min_thermal_displacements,
        )

        temperature_range_thermal_displacements = np.arange(
            kwargs.get("tmin_thermal_displacements", 0),
            kwargs.get("tmax_thermal_displacements", 500),
            kwargs.get("tstep_thermal_displacements", 100),
        )
        for idx, temp in enumerate(temperature_range_thermal_displacements):
            phonon.thermal_displacement_matrices.write_cif(
                phonon.primitive, idx, filename=f"tdispmat_{temp}K.cif"
            )
        _disp_mat = phonon._thermal_displacement_matrices  # noqa: SLF001
        tdisp_mat = _disp_mat.thermal_displacement_matrices.tolist()

        tdisp_mat_cif = _disp_mat.thermal_displacement_matrices_cif.tolist()

    else:
        tdisp_mat = None
        tdisp_mat_cif = None

    formula_units = (
        structure.composition.num_atoms
        / structure.composition.reduced_composition.num_atoms
    )

    total_dft_energy_per_formula_unit = (
        total_dft_energy / formula_units if total_dft_energy is not None else None
    )

    volume_per_formula_unit = structure.volume / formula_units

    return PhononBSDOSDoc.from_structure(
        structure=structure,
        meta_structure=structure,
        phonon_bandstructure=bs_symm_line.as_dict(),
        phonon_dos=dos.as_dict(),
        total_dft_energy=total_dft_energy_per_formula_unit,
        volume_per_formula_unit=volume_per_formula_unit,
        formula_units=formula_units,
        has_imaginary_modes=imaginary_modes,
        force_constants={"force_constants": phonon.force_constants.tolist()}
        if kwargs["store_force_constants"]
        else None,
        born=borns.tolist() if borns is not None else None,
        epsilon_static=epsilon.tolist() if epsilon is not None else None,
        supercell_matrix=phonon.supercell_matrix.tolist(),
        primitive_matrix=phonon.primitive_matrix.tolist(),
        code=code,
        thermal_displacement_data={
            "temperatures_thermal_displacements": temperature_range_thermal_displacements.tolist(),  # noqa: E501
            "thermal_displacement_matrix_cif": tdisp_mat_cif,
            "thermal_displacement_matrix": tdisp_mat,
            "freq_min_thermal_displacements": freq_min_thermal_displacements,
        }
        if kwargs.get("create_thermal_displacements")
        else None,
        jobdirs={
            "displacements_job_dirs": displacement_data["dirs"],
            "static_run_job_dir": kwargs["static_run_job_dir"],
            "born_run_job_dir": kwargs["born_run_job_dir"],
            "optimization_run_job_dir": kwargs["optimization_run_job_dir"],
            "taskdoc_run_job_dir": str(Path.cwd()),
        },
        uuids={
            "displacements_uuids": displacement_data["uuids"],
            "born_run_uuid": kwargs["born_run_uuid"],
            "optimization_run_uuid": kwargs["optimization_run_uuid"],
            "static_run_uuid": kwargs["static_run_uuid"],
        },
        phonopy_settings={
            "npoints_band": npoints_band,
            "kpath_scheme": kpath_scheme,
            "kpoint_density_dos": kpoint_density_dos,
        },
        **kwargs.get("additional_fields", {}),
    )


@job(data=["forces", "displaced_structures"])
def run_phonon_displacements(
    displacements: list[Structure],
    structure: Structure,
    supercell_matrix: Matrix3D,
    phonon_maker: BaseVaspMaker | ForceFieldStaticMaker | BaseAimsMaker | None = None,
    prev_dir: str | Path | None = None,
    prev_dir_argname: str | None = None,
    socket: bool = False,
    store_displaced_structures: bool = False,
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
    store_displaced_structures : bool = False
        Whether to also save the displaced structures.
    """
    phonon_jobs = []
    save_props = {"displacement_number", "forces", "uuids", "dirs"}
    if store_displaced_structures:
        save_props.add("displaced_structures")
    outputs: dict[str, list] = {k: [] for k in save_props}

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

        # TODO: ensure order is correct.
        if store_displaced_structures:
            outputs["displaced_structures"] = displacements
    else:
        for idx, displacement in enumerate(displacements):
            phonon_job = phonon_maker.make(displacement, prev_dir=prev_dir)
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
            if store_displaced_structures:
                outputs["displaced_structures"].append(displacement)

    displacement_flow = Flow(phonon_jobs, outputs)
    return Response(replace=displacement_flow)
