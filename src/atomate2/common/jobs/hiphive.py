"""Jobs for running phonon calculations with phonopy and hiPhive."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.io import read as ase_read
from emmet.core import __version__ as _emmet_core_version
from emmet.core.phonon import PhononBSDOSDoc
from hiphive import (
    ClusterSpace,
    ForceConstantPotential,
    StructureContainer,
    enforce_rotational_sum_rules,
)
from hiphive.cutoffs import estimate_maximum_cutoff
from jobflow import job
from packaging.version import parse as parse_version
from phonopy.file_IO import parse_FORCE_CONSTANTS
from phonopy.interface.vasp import write_vasp
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.io.vasp import Kpoints
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)
from trainstation import Optimizer

from atomate2.common.jobs.phonons import (
    _generate_phonon_object,
    _get_kpath,
    _run_band_structure_and_plot,
    _run_total_dos_and_plot,
)

if TYPE_CHECKING:
    from ase.atoms import Atoms
    from emmet.core.math import Matrix3D

logger = logging.getLogger(__name__)

try:
    from alm import ALM
except ImportError:
    ALM = None

_DEFAULT_FILE_PATHS = {
    "force_displacements": "dataset_forces.npy",
    "displacements": "dataset_disps.npy",
    "displacements_folded": "dataset_disps_array_rr.npy",
    "phonopy": "phonopy.yaml",
    "band_structure": "phonon_band_structure.yaml",
    "band_structure_plot": "phonon_band_structure.pdf",
    "dos": "phonon_dos.yaml",
    "dos_plot": "phonon_dos.pdf",
    "force_constants": "FORCE_CONSTANTS",
    "harmonic_displacements": "disp_matrix.npy",
    "anharmonic_displacements": "disp_matrix_anhar.npy",
    "harmonic_force_matrix": "force_matrix.npy",
    "anharmonic_force_matrix": "force_matrix_anhar.npy",
    "website": "phonon_website.json",
}


def _fit_force_constants(
    prim: Atoms,
    supercell: Atoms,
    atoms_list: list[Atoms],
    cutoff: float,
    fit_method: str,
    fc_filename: str,
) -> tuple[int, float]:
    """Fit second-order force constants with hiPhive and write them to file.

    Parameters
    ----------
    prim: Atoms
        Unit cell the cluster space is built from.
    supercell: Atoms
        Ideal supercell the force constants are produced for.
    atoms_list: list[Atoms]
        Displaced supercells carrying "displacements" and "forces" arrays.
    cutoff: float
        Second-order cutoff in Angstrom.
    fit_method: str
        Regressor name passed to the trainstation optimizer.
    fc_filename: str
        Path the phonopy-format force constants are written to.

    Returns
    -------
    tuple[int, float]
        Number of free parameters and the training RMSE.
    """
    cs = ClusterSpace(prim, [cutoff])
    container = StructureContainer(cs)
    for atoms in atoms_list:
        container.add_structure(atoms)

    opt = Optimizer(container.get_fit_data(), fit_method=fit_method, train_size=1.0)
    opt.train()

    # Rotational sum rules. hiPhive equivalent of pheasy --rasr BHH.
    parameters = enforce_rotational_sum_rules(
        cs, opt.parameters, ["Huang", "Born-Huang"], alpha=1e-6
    )
    fcp = ForceConstantPotential(cs, parameters)
    fcp.get_force_constants(supercell).write_to_phonopy(fc_filename, format="text")
    return cs.n_dofs, opt.rmse_train


@job
def get_supercell_size(
    structure: Structure,
    min_length: float,
    max_atoms: int,
    force_90_degrees: bool,
    force_diagonal: bool,
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
    transformation = CubicSupercellTransformation(
        min_length=min_length,
        max_atoms=max_atoms,
        force_90_degrees=force_90_degrees,
        force_diagonal=force_diagonal,
        angle_tolerance=1e-2,
        allow_orthorhombic=False,
    )
    transformation.apply_transformation(structure=structure)
    return transformation.transformation_matrix.transpose().tolist()


@job(data=[Structure])
def generate_phonon_displacements(
    structure: Structure,
    supercell_matrix: np.array,
    displacement: float,
    num_displaced_supercells: int,
    sym_reduce: bool,
    symprec: float,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    code: str,
    random_seed: int | None = 103,
    verbose: bool = False,
) -> list[Structure]:
    """Generate small-distance perturbed structures with phonopy based on two ways.

    1. finite-displacment method (one displaced atom) when the displacement number
    is less than 3. 2. random-displacement method (all-displaced atoms) when the
    displacement number is more than 3.

    Parameters
    ----------
    structure: Structure object
        Fully optimized input structure for phonon run
    supercell_matrix: np.array
        array to describe supercell matrix
    displacement: float
        displacement in Angstrom (default: 0.01)
    num_displaced_supercells: int
        number of displaced supercells defined by users
    sym_reduce: bool
        if True, symmetry will be used to generate displacements
    symprec: float
        precision to determine symmetry
    use_symmetrized_structure: str or None
        primitive, conventional or None
    kpath_scheme: str
        scheme to generate kpath
    code: str
        code to perform the computations
    random_seed : int | None = 103
        Random seed to use in generating randomly-displaced structures.
    verbose : bool = False
        Whether to log warnings.

    """
    # TODO: remove ALMODE dependence for 2nd order force constants
    if not ALM:
        raise ImportError(
            "Error importing ALM. Please ensure the 'alm' library is installed."
        )
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

    # 1. the ALM module is used to determine the number of free parameters
    # (irreducible force constants) corresponding to the second order
    # force constants (FCs) given a supercell.
    # 2. Based on the number of free parameters, we can determine how many
    # displaced supercells we need to use to extract the second order force
    # constants. Generally, the number of free parameters should be less than
    # 3 * natom(supercell) * num_displaced_supercells. However, the full rank
    # of the matrix can not always guarantee accurate results, you
    # may need to displace more random configurations. Use at least one or
    # two more configurations based on the suggested number of displacements.
    supercell_ph = phonon.supercell
    lattice = supercell_ph.cell
    positions = supercell_ph.scaled_positions
    numbers = supercell_ph.numbers
    natom = len(numbers)

    # get the number of free parameters of 2ND FCs from ALM, labeled as n_fp
    with ALM(lattice, positions, numbers) as alm:
        alm.define(1)
        alm.suggest()
        n_fp = alm._get_number_of_irred_fc_elements(1)  # noqa: SLF001

    # get the number of displaced supercells based on the number of free parameters
    num_disp_sc = int(np.ceil(n_fp / (3.0 * natom)))

    if verbose:
        logger.info(
            f"There are {n_fp} free parameters for the second-order "
            "force constants (FCs)."
            f"There are {3 * natom * num_disp_sc} equations used to "
            "obtain the second-order FCs."
            "CAUTION: you may need to increase the number of "
            "displacements in some cases."
            "If the number of atoms in the supercell are less than 100 and "
            "all lattice constants are less than 10 Å, the user is advised "
            "to use 1-2 more randomly-displaced configurations."
        )

    # get the number of displaced supercells from phonopy to compared with the number
    # of 3, if the number of displaced supercells is less than 3, we will use the finite
    # displacement method to generate the supercells. Otherwise, we will use the random
    # displacement method to generate the supercells.
    if len(phonon.displacements) > 3:
        phonon.generate_displacements(
            distance=displacement,
            number_of_snapshots=(
                num_displaced_supercells
                if num_displaced_supercells != 0
                else int(np.ceil(num_disp_sc * 1.8)) + 1
            ),
            random_seed=random_seed,
        )

    supercells = phonon.supercells_with_displacements
    displacements = [get_pmg_structure(cell) for cell in supercells]

    # add the equilibrium structure to the list for calculating
    # the residual forces.
    displacements.append(get_pmg_structure(phonon.supercell))
    return displacements


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
    epsilon_static: Matrix3D = None,
    born: Matrix3D = None,
    cutoff_2nd: float | None = None,
    fit_method: str = "rfe",
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
    cutoff_2nd: float | None
        second-order cutoff in Angstrom for the hiPhive cluster space. If
        None, the largest cutoff the supercell allows is used.
    fit_method: str
        regressor passed to the trainstation optimizer, e.g. "rfe",
        "least-squares" or "lasso".
    verbose : bool = False
        Whether to log error messages.
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

    # Write POSCAR and SPOSCAR. hiPhive reads them as the unit cell the
    # cluster space is built on and the supercell the fit targets.
    supercell = phonon._supercell  # noqa: SLF001
    write_vasp("POSCAR", get_phonopy_structure(structure))
    write_vasp("SPOSCAR", supercell)

    # get the force-displacement dataset from previous calculations
    dataset_forces = np.array(displacement_data["forces"])
    np.save(_DEFAULT_FILE_PATHS["force_displacements"], dataset_forces)

    # To deduct the residual forces on an equilibrium structure to eliminate the
    # fitting error
    dataset_forces_array_rr = dataset_forces - dataset_forces[-1, :, :]

    # force matrix on the displaced structures
    dataset_forces_array_disp = dataset_forces_array_rr[:-1, :, :]

    # To handle the large dispalced distance in the dataset
    dataset_disps = np.array(
        [disps.frac_coords for disps in displacement_data["displaced_structures"]]
    )
    np.save(_DEFAULT_FILE_PATHS["displacements"], dataset_disps)

    supercell_scaled_positions = np.array(supercell.scaled_positions)

    dataset_disps_array_rr = np.round(
        dataset_disps - supercell_scaled_positions,
        decimals=16,
    )
    np.save(_DEFAULT_FILE_PATHS["displacements_folded"], dataset_disps_array_rr)

    dataset_disps_array_rr = np.where(
        dataset_disps_array_rr > 0.5,
        dataset_disps_array_rr - 1.0,
        dataset_disps_array_rr,
    )
    dataset_disps_array_rr = np.where(
        dataset_disps_array_rr < -0.5,
        dataset_disps_array_rr + 1.0,
        dataset_disps_array_rr,
    )

    # Transpose the displacement array on the
    # last two axes (atoms and coordinates)
    dataset_disps_array_rr_transposed = np.transpose(dataset_disps_array_rr, (0, 2, 1))

    # Perform matrix multiplication with the transposed supercell.cell
    # 'ij' for supercell.cell.T and
    # 'nkj' for the transposed dataset_disps_array_rr
    dataset_disps_array_rr_cartesian = np.einsum(
        "ij,njk->nik", supercell.cell.T, dataset_disps_array_rr_transposed
    )
    # Transpose back to the original format
    dataset_disps_array_rr_cartesian = np.transpose(
        dataset_disps_array_rr_cartesian, (0, 2, 1)
    )

    dataset_disps_array_use = dataset_disps_array_rr_cartesian[:-1, :, :]

    num_har = dataset_disps_array_use.shape[0]

    np.save(
        _DEFAULT_FILE_PATHS["harmonic_displacements"],
        dataset_disps_array_use[:num_har, :, :],
    )
    np.save(
        _DEFAULT_FILE_PATHS["harmonic_force_matrix"],
        dataset_forces_array_disp[:num_har, :, :],
    )

    # get the born charges and dielectric constant
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
                "Number of born charges does not agree with number of atoms"
            )

        if code == "vasp" and not np.all(np.isclose(borns, 0.0)):
            phonon.nac_params = {
                "born": borns,
                "dielectric": epsilon,
                "factor": 14.399652,
            }
        # Other codes could be added here

    else:
        borns = None
        epsilon = None

    prim = ase_read("POSCAR")
    supercell = ase_read("SPOSCAR")

    # Collect the displaced supercells and their forces. The displacements
    # are already residual-corrected and in Cartesian Angstrom, so they are
    # used as they are.
    atoms_list = []
    for disp, force in zip(
        dataset_disps_array_use[:num_har],
        dataset_forces_array_disp[:num_har],
        strict=True,
    ):
        # Positions stay at the ideal supercell. hiPhive aligns that cell
        # against the primitive cell to map the orbits, which only works
        # while the symmetry is intact. The displacement is carried in the
        # array, not in the coordinates.
        atoms = supercell.copy()
        atoms.new_array("displacements", np.ascontiguousarray(disp))
        atoms.new_array("forces", np.ascontiguousarray(force))
        atoms_list.append(atoms)

    # Random displacements give an underdetermined system, so a sparse
    # regressor is used above 3 configurations. Below that the dataset comes
    # from the finite-displacement path and least squares is enough. This is
    # the same criterion pheasy applies.
    fit_method_2nd = fit_method if len(phonon.displacements) > 3 else "least-squares"

    # When no cutoff is given, use the largest the supercell allows. That is
    # the hiPhive analogue of the Wigner-Seitz boundary pheasy defaults to.
    max_cutoff = estimate_maximum_cutoff(supercell) - 0.01
    cutoff = max_cutoff if cutoff_2nd is None else min(cutoff_2nd, max_cutoff)

    n_dofs, rmse = _fit_force_constants(
        prim,
        supercell,
        atoms_list,
        cutoff,
        fit_method_2nd,
        _DEFAULT_FILE_PATHS["force_constants"],
    )
    logger.info(f"Fit at {cutoff:.2f} A: {n_dofs} parameters, RMSE {rmse}")

    fc_file = Path(_DEFAULT_FILE_PATHS["force_constants"])

    if fc_file.exists():
        # Read back the force constants written by the hiPhive fit
        force_constants = parse_FORCE_CONSTANTS(filename=fc_file)
        phonon.force_constants = force_constants
        # symmetrize the force constants to make them physically correct based on
        # the space group symmetry of the crystal structure.
        phonon.symmetrize_force_constants()

    # with phonopy.load("phonopy.yaml") the phonopy API can be used
    phonon.save(_DEFAULT_FILE_PATHS["phonopy"])

    # get phonon band structure
    kpath_dict, kpath_concrete = _get_kpath(
        structure=get_pmg_structure(phonon.primitive),
        kpath_scheme=kpath_scheme,
        symprec=symprec,
    )

    bs_plot_file = kwargs.get("filename_bs", _DEFAULT_FILE_PATHS["band_structure_plot"])
    dos_plot_file = kwargs.get("filename_dos", _DEFAULT_FILE_PATHS["dos_plot"])
    npoints_band = kwargs.get("npoints_band", 101)

    bs_symm_line, imaginary_modes = _run_band_structure_and_plot(
        phonon,
        kpath_dict,
        kpath_concrete,
        _DEFAULT_FILE_PATHS["band_structure"],
        has_nac=born is not None,
        npoints_band=npoints_band,
        with_eigenvectors=kwargs.get("band_structure_eigenvectors", False),
        is_band_connection=kwargs.get("band_structure_eigenvectors", False),
        filename_bs=bs_plot_file,
        units=kwargs.get("units", "THz"),
        tol_imaginary_modes=kwargs.get("tol_imaginary_modes", 1e-5),
    )

    # If imaginary modes remain, refit at a shorter cutoff. The smaller
    # cluster space has fewer free parameters and is better conditioned,
    # which usually clears small imaginary modes near Gamma. pheasy used a
    # 10 A cutoff for the same purpose.
    if imaginary_modes:
        short_cutoff = min(10.0, max_cutoff)
        new_fc_file = f"{_DEFAULT_FILE_PATHS['force_constants']}_short_cutoff"
        n_dofs, rmse = _fit_force_constants(
            prim,
            supercell,
            atoms_list,
            short_cutoff,
            fit_method_2nd,
            new_fc_file,
        )
        logger.info(f"Refit at {short_cutoff:.2f} A: {n_dofs} params, RMSE {rmse}")

        phonon.force_constants = parse_FORCE_CONSTANTS(filename=new_fc_file)
        phonon.symmetrize_force_constants()
        phonon.save(_DEFAULT_FILE_PATHS["phonopy"])

        bs_symm_line, imaginary_modes = _run_band_structure_and_plot(
            phonon,
            kpath_dict,
            kpath_concrete,
            _DEFAULT_FILE_PATHS["band_structure"],
            has_nac=born is not None,
            npoints_band=npoints_band,
            with_eigenvectors=True,
            filename_bs=bs_plot_file,
            units=kwargs.get("units", "THz"),
            tol_imaginary_modes=kwargs.get("tol_imaginary_modes", 1e-5),
        )

    # gets data for visualization on website - yaml is also enough
    if kwargs.get("band_structure_eigenvectors"):
        bs_symm_line.write_phononwebsite(_DEFAULT_FILE_PATHS["website"])

    # get phonon density of states
    kpoint_density_dos = kwargs.get("kpoint_density_dos", 7_000)
    kpoint = Kpoints.automatic_density(
        structure=get_pmg_structure(phonon.primitive),
        kppa=kpoint_density_dos,
        force_gamma=True,
    )
    dos = _run_total_dos_and_plot(
        phonon,
        kpoint,
        _DEFAULT_FILE_PATHS["dos"],
        filename_dos=dos_plot_file,
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

    cls_constructor = (
        "migrate_fields"
        if parse_version(_emmet_core_version) >= parse_version("0.85.1")
        else "from_structure"
    )
    return getattr(PhononBSDOSDoc, cls_constructor)(
        structure=structure,
        meta_structure=structure,
        phonon_bandstructure=bs_symm_line,
        phonon_dos=dos,
        total_dft_energy=total_dft_energy_per_formula_unit,
        has_imaginary_modes=imaginary_modes,
        force_constants=(
            {"force_constants": phonon.force_constants.tolist()}
            if kwargs.get("store_force_constants")
            else None
        ),
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
        post_process_settings={
            "npoints_band": npoints_band,
            "kpath_scheme": kpath_scheme,
            "kpoint_density_dos": kpoint_density_dos,
        },
    )
