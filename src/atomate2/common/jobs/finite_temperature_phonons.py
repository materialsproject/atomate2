"""Jobs for effective harmonic phonons from MD snapshots and pheasy."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.io import read as ase_read
from ase.units import kB
from jobflow import job
from monty.io import zopen
from monty.os.path import zpath
from phonopy.file_IO import parse_FORCE_CONSTANTS
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.interface.vasp import write_vasp
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.io.vasp import Incar, Kpoints, Poscar, Xdatcar
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos

from atomate2.common.jobs.pheasy import (
    _DEFAULT_FILE_PATHS,
    _check_lasso_alpha,
    _run_harmonic_fit,
)
from atomate2.common.jobs.phonons import (
    _generate_phonon_object,
    _get_kpath,
    _run_band_structure_and_plot,
    _run_total_dos_and_plot,
)
from atomate2.common.schemas.finite_temperature_phonons import (
    FiniteTemperaturePhononDoc,
    TrajectoryHealth,
)
from atomate2.common.schemas.phonons import (
    ForceConstants,
    PhononJobDirs,
    PhononUUIDs,
    get_factor,
)
from atomate2.utils.path import strip_hostname

if TYPE_CHECKING:
    from collections.abc import Sequence

    from emmet.core.math import Matrix3D
    from phonopy import Phonopy

logger = logging.getLogger(__name__)

# file of the force field MD trajectory, written in the ASE format
ASE_TRAJECTORY_FILE = "md_trajectory.traj"
_KPATH_SCHEME = "seekpath"

# Thresholds of the trajectory check. The MD starts from the reference structure,
# so a large displacement in the first frames means the atom order does not match.
_START_LIMIT = 0.5  # Angstrom
_RMS_LIMIT = 1.0  # Angstrom
_LINDEMANN_LIMIT = 0.12
_SHIFT_RATIO_LIMIT = 1.5
_DRIFT_SIGMA = 3.0
_N_START_FRAMES = 10


def _get_phonopy(
    structure: Structure, supercell_matrix: Matrix3D, symprec: float, code: str
) -> Phonopy:
    """Get the phonopy object of the sorted structure and the supercell."""
    return _generate_phonon_object(
        structure.get_sorted_structure(),
        supercell_matrix,
        displacement=0.01,
        sym_reduce=True,
        symprec=symprec,
        use_symmetrized_structure=None,
        kpath_scheme=_KPATH_SCHEME,
        code=code,
    )


def _get_site_properties(reference: Structure) -> dict:
    """Get the magnetic moments of the reference, the only site property kept."""
    if "magmom" in reference.site_properties:
        return {"magmom": reference.site_properties["magmom"]}
    return {}


@job
def get_md_supercell(
    structure: Structure, supercell_matrix: Matrix3D, symprec: float, code: str
) -> Structure:
    """
    Build the supercell used as the reference structure of the MD and the fit.

    The atoms of the structure are sorted by electronegativity, as the VASP
    input sets do. The supercell is then the one phonopy builds, so that its
    atoms are in the order of the force constant fit. Each atom of the
    supercell gets the magnetic moment of its atom in the unit cell, if the
    structure has magnetic moments.

    Parameters
    ----------
    structure: Structure
        Relaxed unit cell.
    supercell_matrix: Matrix3D
        Supercell matrix.
    symprec: float
        Symmetry precision for phonopy.
    code: str
        Code of the phonon displacement calculations.

    Returns
    -------
    Structure
        The undisplaced supercell.
    """
    structure = structure.get_sorted_structure()
    supercell = _get_phonopy(structure, supercell_matrix, symprec, code).supercell
    site_properties = {}
    if "magmom" in structure.site_properties:
        unit_index = [supercell.u2u_map[idx] for idx in supercell.s2u_map]
        magmoms = structure.site_properties["magmom"]
        site_properties["magmom"] = [magmoms[idx] for idx in unit_index]
    return Structure(
        supercell.cell,
        supercell.symbols,
        supercell.scaled_positions,
        site_properties=site_properties,
    )


@job
def get_md_restart_structure(
    md_dir: str, md_code: str, reference: Structure
) -> Structure:
    """
    Get the final positions and velocities of an MD run.

    For VASP, they are read from CONTCAR, with the velocities in Angstrom/fs.
    For force fields, they are read from the last frame of the ASE trajectory
    file, with the velocities in ASE units. In both cases, the velocities are a
    site property that the next MD run starts from. The thermostat variables
    are not carried over, so they start again from zero in the next run. The
    magnetic moments of the reference, if any, are added as a site property.

    Parameters
    ----------
    md_dir: str
        Directory of the MD run.
    md_code: str
        Code of the MD, "vasp" or "forcefields".
    reference: Structure
        The undisplaced supercell the MD started from.

    Returns
    -------
    Structure
        The final structure, with a "velocities" site property.
    """
    directory = Path(strip_hostname(md_dir))
    if md_code == "vasp":
        structure = Poscar.from_file(zpath(str(directory / "CONTCAR"))).structure
    elif md_code == "forcefields":
        atoms = ase_read(directory / ASE_TRAJECTORY_FILE, index=-1)
        structure = Structure(
            atoms.cell[:],
            atoms.get_chemical_symbols(),
            atoms.get_positions(),
            coords_are_cartesian=True,
            site_properties={"velocities": atoms.get_velocities().tolist()},
        )
    else:
        raise ValueError(f"MD code must be 'vasp' or 'forcefields', not {md_code}.")
    for key, values in _get_site_properties(reference).items():
        structure.add_site_property(key, values)
    return structure


def _read_vasp_md(directory: Path) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    """Read the frames, energies and time step of a VASP MD run."""
    incar = Incar.from_file(zpath(str(directory / "INCAR")))
    xdatcar = Xdatcar(zpath(str(directory / "XDATCAR")))
    frac_coords = np.array([frame.frac_coords for frame in xdatcar.structures])
    species = [str(site.specie) for site in xdatcar.structures[0]]

    # the free energy F of each ionic step
    with zopen(zpath(str(directory / "OSZICAR")), mode="rt") as file:
        energies = [
            float(line.split(" F= ")[1].split()[0])
            for line in file
            if " T= " in line and " F= " in line
        ]
    if len(energies) != len(frac_coords):
        raise ValueError(
            f"{directory} has {len(frac_coords)} frames in XDATCAR but "
            f"{len(energies)} MD steps in OSZICAR. XDATCAR must have a frame at "
            "every step, with NBLOCK = 1."
        )
    return frac_coords, np.array(energies), species, float(incar["POTIM"])


def _read_ase_md(
    directory: Path, time_step: float, skip_first: bool
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    """Read the frames and energies of a force field MD run."""
    frames = ase_read(directory / ASE_TRAJECTORY_FILE, index=":")
    if skip_first:
        # the starting structure is the last frame of the previous MD run
        frames = frames[1:]
    frac_coords = np.array([atoms.get_scaled_positions() for atoms in frames])
    energies = np.array([atoms.get_potential_energy() for atoms in frames])
    return frac_coords, energies, frames[0].get_chemical_symbols(), time_step


def _get_displacements(frac_coords: np.ndarray, reference: Structure) -> np.ndarray:
    """Cartesian displacements from the reference with the minimum image."""
    diff = frac_coords - reference.frac_coords
    diff -= np.round(diff)
    return diff @ reference.lattice.matrix


def _remove_center_of_mass(disps: np.ndarray, reference: Structure) -> np.ndarray:
    """Remove the displacement of the center of mass from each frame."""
    masses = np.array([site.specie.atomic_mass for site in reference])
    center = np.einsum("i,fia->fa", masses, disps) / masses.sum()
    return disps - center[:, None, :]


def _assess_trajectory(
    frac_coords: np.ndarray,
    energies: np.ndarray,
    reference: Structure,
    temperature: float,
) -> TrajectoryHealth:
    """
    Check whether the MD trajectory stayed at the reference structure.

    The displacement of the center of mass of each frame is removed from its
    displacements. The checks are applied in this order.

    - A root mean square displacement above 0.5 Angstrom in the first 10
      frames means the atoms are not in the order of the reference.
    - A root mean square vibration u_vib above 0.12 of the nearest-neighbor
      distance means the structure melted.
    - A total displacement u_ref above 1.5 times u_vib means the mean
      positions moved away from the reference.
    - A change of the mean potential energy from the second to the last
      quarter of the frames above 3 times the standard deviation of the
      energy in the last fifth means the structure transformed (falling energy)
      or is disordering (rising energy).
    - A root mean square displacement above 1 Angstrom in the last fifth of the
      frames points at diffusion or very soft motion.

    The energy drift is measured late in the trajectory, because a run that
    starts from the reference must gain 3/2 k_B T of potential energy to reach
    equipartition.

    Parameters
    ----------
    frac_coords: np.ndarray
        Fractional coordinates of each frame, with shape (n_frames, n_atoms, 3).
    energies: np.ndarray
        Potential energy of each frame in eV.
    reference: Structure
        The undisplaced supercell.
    temperature: float
        MD temperature in K.

    Returns
    -------
    TrajectoryHealth
    """
    n_frames, n_atoms = frac_coords.shape[:2]
    disps = _remove_center_of_mass(
        np.array([_get_displacements(frame, reference) for frame in frac_coords]),
        reference,
    )
    rms = np.sqrt(np.mean(np.sum(disps**2, axis=2), axis=1))
    rms_start = float(rms[:_N_START_FRAMES].mean())
    rms_end = float(rms[-max(1, n_frames // 5) :].mean())

    # split the displacement in the second half into a static and a vibrational
    # part, u_ref**2 = u_shift**2 + u_vib**2
    tail = disps[n_frames // 2 :]
    msd_ref = float(np.mean(np.sum(tail**2, axis=2)))
    msd_shift = float(np.mean(np.sum(tail.mean(axis=0) ** 2, axis=1)))
    u_ref = np.sqrt(msd_ref)
    u_shift = np.sqrt(msd_shift)
    u_vib = np.sqrt(max(msd_ref - msd_shift, 0.0))

    distances = reference.distance_matrix.copy()
    np.fill_diagonal(distances, np.inf)
    d_nn = float(distances.min(axis=1).mean())
    lindemann_ratio = u_vib / d_nn
    shift_ratio = u_ref / max(u_vib, 1e-9)

    energy = energies / n_atoms
    quarter = len(energy) // 4
    drift = (
        float(energy[3 * quarter :].mean() - energy[quarter : 2 * quarter].mean())
        if quarter >= 5
        else None
    )
    fluctuation = float(energy[-max(1, len(energy) // 5) :].std())
    big_drift = drift is not None and abs(drift) > _DRIFT_SIGMA * max(fluctuation, 1e-9)

    if rms_start > _START_LIMIT:
        verdict = "reference_mismatch"
    elif lindemann_ratio > _LINDEMANN_LIMIT:
        verdict = "melted"
    elif shift_ratio > _SHIFT_RATIO_LIMIT:
        verdict = "transformed" if big_drift and drift < 0 else "shifted_or_diffusing"
    elif big_drift:
        verdict = "transformed" if drift < 0 else "disordering"
    elif rms_end > _RMS_LIMIT:
        verdict = "diffusing_or_soft"
    else:
        verdict = "stable"

    return TrajectoryHealth(
        verdict=verdict,
        is_stable=verdict == "stable",
        n_frames=n_frames,
        rms_displacement_start=rms_start,
        rms_displacement_end=rms_end,
        u_ref=u_ref,
        u_shift=u_shift,
        u_vib=u_vib,
        nearest_neighbor_distance=d_nn,
        lindemann_ratio=lindemann_ratio,
        shift_ratio=shift_ratio,
        energy_drift=drift,
        energy_fluctuation=fluctuation,
        equipartition_rise=1.5 * kB * temperature,
    )


@job(data=["structures"])
def select_md_snapshots(
    md_dirs: Sequence[str],
    md_code: str,
    reference: Structure,
    temperature: float,
    md_time_step: float,
    equilibration_time: float,
    n_snapshots: int,
) -> dict:
    """
    Pick snapshots from the MD trajectory and check the trajectory.

    The trajectories of all MD runs are joined in order, with a frame at every
    time step. The time of a frame is its index in the joined trajectory times
    the time step, so the first frame is at time zero. The frames of the first
    equilibration_time are left out, and n_snapshots frames are picked evenly
    spread over the rest. For VASP, XDATCAR, OSZICAR and the POTIM of INCAR are
    read from each run directory. For force fields, the ASE trajectory file is
    read. The run directories must be readable from where this job runs.

    Parameters
    ----------
    md_dirs: Sequence[str]
        Directories of the MD runs, in order.
    md_code: str
        Code of the MD, "vasp" or "forcefields".
    reference: Structure
        The undisplaced supercell the MD started from.
    temperature: float
        MD temperature in K.
    md_time_step: float
        MD time step in fs. Only used for force field trajectories.
    equilibration_time: float
        Time at the start of the trajectory that is left out, in ps.
    n_snapshots: int
        Number of snapshots.

    Returns
    -------
    dict
        The structures for the phonon displacement calculations, the
        snapshots followed by the undisplaced supercell. The MD time of each
        snapshot in ps. The number of frames times the time step in ps. The
        root mean square displacement of the snapshots in Angstrom, with the
        displacement of the center of mass removed. The trajectory check.
    """
    directories = [Path(strip_hostname(md_dir)) for md_dir in md_dirs]
    if len(set(directories)) != len(directories):
        raise ValueError(
            f"Each MD run must have its own directory, but the directories are "
            f"{md_dirs}."
        )
    frac_coords, energies, time_steps = [], [], set()
    for idx, directory in enumerate(directories):
        if md_code == "vasp":
            coords, energy, species, time_step = _read_vasp_md(directory)
        elif md_code == "forcefields":
            coords, energy, species, time_step = _read_ase_md(
                directory, md_time_step, skip_first=idx > 0
            )
        else:
            raise ValueError(f"MD code must be 'vasp' or 'forcefields', not {md_code}.")
        if species != [str(site.specie) for site in reference]:
            raise ValueError(
                f"The atoms of the MD in {directory} are not in the order of the "
                "reference supercell."
            )
        frac_coords.append(coords)
        energies.append(energy)
        time_steps.add(time_step)
    if len(time_steps) != 1:
        raise ValueError(f"The MD runs have different time steps: {time_steps} fs.")
    time_step = time_steps.pop()
    all_coords = np.concatenate(frac_coords)
    all_energies = np.concatenate(energies)
    n_frames = len(all_coords)

    n_equil = round(equilibration_time * 1000 / time_step)
    n_left = n_frames - n_equil
    if n_left < n_snapshots:
        raise ValueError(
            f"The trajectory has {n_frames} frames of {time_step} fs. After "
            f"leaving out {equilibration_time} ps, {max(n_left, 0)} frames are "
            f"left, fewer than the {n_snapshots} snapshots."
        )
    stride = n_left // n_snapshots
    indices = list(range(n_equil, n_frames, stride))[:n_snapshots]

    site_properties = _get_site_properties(reference)
    snapshots = [
        Structure(
            reference.lattice,
            reference.species,
            all_coords[idx],
            site_properties=site_properties,
        )
        for idx in indices
    ]
    disps = _remove_center_of_mass(
        np.array([_get_displacements(all_coords[idx], reference) for idx in indices]),
        reference,
    )
    health = _assess_trajectory(all_coords, all_energies, reference, temperature)
    if not health.is_stable:
        warnings.warn(
            f"The MD trajectory did not stay at the reference structure (verdict: "
            f"{health.verdict}). The fitted force constants may not describe it.",
            stacklevel=2,
        )
    return {
        "structures": [*snapshots, reference],
        "snapshot_times": [idx * time_step / 1000 for idx in indices],
        "md_time": n_frames * time_step / 1000,
        "rms_displacement": float(np.sqrt(np.mean(np.sum(disps**2, axis=2)))),
        "trajectory_health": health.model_dump(),
    }


def _get_rms_displacement(phonon: Phonopy, temperature: float, tol: float) -> float:
    """
    Get the classical root mean square displacement from the force constants.

    The modes are those of the supercell at the Gamma point. A mode with a
    frequency above tol in THz adds k_B T divided by its eigenvalue of the
    dynamical matrix, times the sum over the atoms and Cartesian components of
    its squared eigenvector component divided by the atomic mass. The sum over
    the modes is divided by the number of atoms. The modes at or below tol are
    left out. They include the three acoustic modes at the Gamma point, which
    move the center of mass.
    """
    masses = np.repeat(np.array(phonon.supercell.masses), 3)
    n_dof = len(masses)
    force_constants = phonon.force_constants.transpose(0, 2, 1, 3).reshape(n_dof, n_dof)
    inv_sqrt_mass = 1 / np.sqrt(masses)
    dynmat = force_constants * np.outer(inv_sqrt_mass, inv_sqrt_mass)
    eigvals, eigvecs = np.linalg.eigh((dynmat + dynmat.T) / 2)
    frequencies = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * get_factor("vasp")
    keep = frequencies > tol
    weights = np.sum(eigvecs[:, keep] ** 2 / masses[:, None], axis=0)
    msd = kB * temperature * np.sum(weights / eigvals[keep])
    return float(np.sqrt(msd / len(phonon.supercell.masses)))


@job(
    output_schema=FiniteTemperaturePhononDoc,
    data=[PhononDos, PhononBandStructureSymmLine, ForceConstants],
)
def fit_finite_temperature_phonons(
    structure: Structure,
    supercell_matrix: Matrix3D,
    snapshot_data: dict,
    displacement_data: dict,
    temperature: float,
    thermostat: str,
    md_time_step: float,
    equilibration_time: float,
    code: str,
    md_code: str,
    symprec: float,
    force_field_name: str | None = None,
    force_field_kwargs: dict | None = None,
    md_force_field_name: str | None = None,
    md_force_field_kwargs: dict | None = None,
    md_uuids: list[str] | None = None,
    md_dirs: list[str] | None = None,
    optimization_run_uuid: str | None = None,
    optimization_run_job_dir: str | None = None,
    born_run_uuid: str | None = None,
    born_run_job_dir: str | None = None,
    rotational_sum_rule: str | None = "BHH",
    alpha_min: int = -6,
    random_seed: int | None = 103,
    tol_imaginary_modes: float = 0.1,
    born: list[Matrix3D] | None = None,
    epsilon_static: Matrix3D | None = None,
    store_force_constants: bool = True,
    npoints_band: int = 101,
    kpoint_density_dos: int = 7_000,
) -> FiniteTemperaturePhononDoc:
    """
    Fit effective second-order force constants to MD snapshots with pheasy.

    The forces on the undisplaced supercell, the last phonon displacement
    calculation, are subtracted from the forces on the snapshots. pheasy then
    fits the second-order force constants with LASSO, with standardized data
    and without a cutoff. phonopy imposes translational and permutation
    symmetry on them. They give the phonon band structure along the seekpath
    k-path and the density of states. Imaginary modes are reported, not
    removed.

    Parameters
    ----------
    structure: Structure
        Relaxed unit cell. This job sorts its atoms by electronegativity.
    supercell_matrix: Matrix3D
        Diagonal supercell matrix.
    snapshot_data: dict
        Output of :obj:`select_md_snapshots`.
    displacement_data: dict
        Output of the phonon displacement calculations on the snapshots and on
        the undisplaced supercell, which comes last.
    temperature: float
        MD temperature in K.
    thermostat: str
        Thermostat of the MD.
    md_time_step: float
        MD time step in fs.
    equilibration_time: float
        Time at the start of the trajectory that was left out, in ps.
    code: str
        Code of the phonon displacement calculations.
    md_code: str
        Code of the MD.
    symprec: float
        Symmetry precision for phonopy and pheasy.
    force_field_name: str | None
        Force field of the phonon displacement calculations, if any.
    force_field_kwargs: dict | None
        Keyword arguments of the force field calculator of the phonon
        displacement calculations.
    md_force_field_name: str | None
        Force field of the MD, if any.
    md_force_field_kwargs: dict | None
        Keyword arguments of the force field calculator of the MD.
    md_uuids: list[str] | None
        UUIDs of the MD jobs.
    md_dirs: list[str] | None
        Directories of the MD jobs.
    optimization_run_uuid: str | None
        UUID of the relaxation.
    optimization_run_job_dir: str | None
        Directory of the relaxation.
    born_run_uuid: str | None
        UUID of the Born charge calculation.
    born_run_job_dir: str | None
        Directory of the Born charge calculation.
    rotational_sum_rule: str | None
        Rotational sum rule passed to pheasy with --rasr, or None for none.
    alpha_min: int
        Base-10 exponent of the smallest LASSO penalty in the cross-validation.
    random_seed: int | None
        Seed of the LASSO fit in pheasy.
    tol_imaginary_modes: float
        Frequencies below -tol_imaginary_modes in THz are imaginary.
    born: list[Matrix3D] | None
        Born effective charges of the sorted unit cell, for the non-analytical
        correction of the dynamical matrix.
    epsilon_static: Matrix3D | None
        High-frequency dielectric tensor, needed together with born.
    store_force_constants: bool
        Whether to store the force constants in the output document.
    npoints_band: int
        Number of q-points per band structure segment.
    kpoint_density_dos: int
        Number of q-points per reciprocal atom of the density of states mesh.

    Returns
    -------
    FiniteTemperaturePhononDoc
    """
    supercell_matrix = np.array(supercell_matrix)
    if not np.allclose(supercell_matrix, np.diag(np.diag(supercell_matrix))):
        raise ValueError("pheasy needs a diagonal supercell matrix.")

    structure = structure.get_sorted_structure()
    if born is not None and len(born) != len(structure):
        raise ValueError("The number of Born charges is not the number of atoms.")
    phonon = _get_phonopy(structure, supercell_matrix, symprec, code)
    supercell = get_pmg_structure(phonon.supercell)

    forces = np.array(displacement_data["forces"])
    structures = displacement_data["displaced_structures"]
    if len(structures) != len(snapshot_data["snapshot_times"]) + 1:
        raise ValueError(
            "The phonon displacement calculations must be the snapshots followed "
            "by the undisplaced supercell."
        )
    if not np.allclose(structures[-1].frac_coords, supercell.frac_coords, atol=1e-6):
        raise ValueError(
            "The last phonon displacement calculation is not the undisplaced supercell."
        )
    residual_forces = forces[-1]
    fit_forces = forces[:-1] - residual_forces
    disps = np.array(
        [_get_displacements(s.frac_coords, supercell) for s in structures[:-1]]
    )
    n_data = len(disps)

    write_vasp("POSCAR", get_phonopy_structure(structure))
    write_vasp("SPOSCAR", phonon.supercell)
    np.save(_DEFAULT_FILE_PATHS["harmonic_displacements"], disps)
    np.save(_DEFAULT_FILE_PATHS["harmonic_force_matrix"], fit_forces)

    # Remove the files of an earlier fit. pheasy appends to its log, and a failed
    # fit would leave the old force constants behind.
    log_file = Path(_DEFAULT_FILE_PATHS["harmonic_fit_log"])
    log_file.unlink(missing_ok=True)
    fc_file = Path(_DEFAULT_FILE_PATHS["force_constants"])
    fc_file.unlink(missing_ok=True)
    _run_harmonic_fit(
        supercell_matrix,
        symprec,
        n_data,
        rotational_sum_rule=rotational_sum_rule,
        alpha_min=alpha_min,
        random_seed=random_seed,
        log_file=str(log_file),
    )

    if not fc_file.exists():
        raise RuntimeError(f"pheasy did not write {fc_file}.")
    lasso_alpha = _check_lasso_alpha(log_file, alpha_min, alpha_min_name="alpha_min")
    phonon.force_constants = parse_FORCE_CONSTANTS(filename=str(fc_file))
    phonon.symmetrize_force_constants()

    # the fitted forces are F = -Phi u
    predicted = -np.einsum("ijab,mjb->mia", phonon.force_constants, disps)
    force_rmse = float(np.sqrt(np.mean((fit_forces - predicted) ** 2)))

    borns = epsilon = None
    if born is not None and epsilon_static is not None:
        borns, epsilon = symmetrize_borns_and_epsilon(
            ucell=phonon.unitcell,
            borns=np.array(born),
            epsilon=np.array(epsilon_static),
            symprec=symprec,
            primitive_matrix=phonon.primitive_matrix,
            supercell_matrix=phonon.supercell_matrix,
        )
        if not np.all(np.isclose(borns, 0.0)):
            phonon.nac_params = {
                "born": borns,
                "dielectric": epsilon,
                "factor": 14.399652,
            }

    # frequencies at the q-points commensurate with the supercell
    matrix = np.linalg.inv(phonon.primitive_matrix) @ phonon.supercell_matrix
    phonon.run_qpoints(get_commensurate_points(np.rint(matrix).astype(int)))
    frequencies = phonon.qpoints.frequencies

    kpath_dict, kpath_concrete = _get_kpath(
        structure=get_pmg_structure(phonon.primitive),
        kpath_scheme=_KPATH_SCHEME,
        symprec=symprec,
    )
    bs_symm_line, has_imaginary_modes = _run_band_structure_and_plot(
        phonon,
        kpath_dict,
        kpath_concrete,
        _DEFAULT_FILE_PATHS["band_structure"],
        has_nac=phonon.nac_params is not None,
        npoints_band=npoints_band,
        filename_bs=_DEFAULT_FILE_PATHS["band_structure_plot"],
        tol_imaginary_modes=tol_imaginary_modes,
    )
    kpoint = Kpoints.automatic_density(
        structure=get_pmg_structure(phonon.primitive),
        kppa=kpoint_density_dos,
        force_gamma=True,
    )
    dos = _run_total_dos_and_plot(
        phonon,
        kpoint,
        _DEFAULT_FILE_PATHS["dos"],
        filename_dos=_DEFAULT_FILE_PATHS["dos_plot"],
    )
    phonon.save(_DEFAULT_FILE_PATHS["phonopy"])

    return FiniteTemperaturePhononDoc.from_structure(
        meta_structure=structure,
        structure=structure,
        temperature=temperature,
        code=code,
        md_code=md_code,
        force_field_name=force_field_name,
        force_field_kwargs=force_field_kwargs,
        md_force_field_name=md_force_field_name,
        md_force_field_kwargs=md_force_field_kwargs,
        thermostat=thermostat,
        md_time_step=md_time_step,
        md_time=snapshot_data["md_time"],
        equilibration_time=equilibration_time,
        n_snapshots=n_data,
        snapshot_times=snapshot_data["snapshot_times"],
        supercell_matrix=phonon.supercell_matrix.tolist(),
        primitive_matrix=phonon.primitive_matrix.tolist(),
        rotational_sum_rule=rotational_sum_rule,
        lasso_alpha=lasso_alpha,
        force_rmse=force_rmse,
        max_residual_force=float(np.abs(residual_forces).max()),
        rms_displacement=snapshot_data["rms_displacement"],
        rms_displacement_from_force_constants=_get_rms_displacement(
            phonon, temperature, tol_imaginary_modes
        ),
        trajectory_health=snapshot_data["trajectory_health"],
        tol_imaginary_modes=tol_imaginary_modes,
        has_imaginary_modes=has_imaginary_modes,
        n_imaginary_modes=int(np.sum(frequencies < -tol_imaginary_modes)),
        min_frequency=float(frequencies.min()),
        phonon_bandstructure=bs_symm_line,
        phonon_dos=dos,
        force_constants=ForceConstants(phonon.force_constants.tolist())
        if store_force_constants
        else None,
        born=borns.tolist() if borns is not None else None,
        epsilon_static=epsilon.tolist() if epsilon is not None else None,
        md_uuids=md_uuids,
        md_dirs=md_dirs,
        uuids=PhononUUIDs(
            optimization_run_uuid=optimization_run_uuid,
            displacements_uuids=displacement_data["uuids"],
            born_run_uuid=born_run_uuid,
        ),
        jobdirs=PhononJobDirs(
            displacements_job_dirs=displacement_data["dirs"],
            born_run_job_dir=born_run_job_dir,
            optimization_run_job_dir=optimization_run_job_dir,
            taskdoc_run_job_dir=str(Path.cwd()),
        ),
    )
