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
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.io.vasp import Incar, Kpoints, Xdatcar
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2.common.jobs.md import ASE_TRAJECTORY_FILE, _get_site_properties
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
    _set_nac_params,
)
from atomate2.utils.path import strip_hostname

if TYPE_CHECKING:
    from collections.abc import Sequence

    from emmet.core.math import Matrix3D
    from phonopy import Phonopy

logger = logging.getLogger(__name__)

_KPATH_SCHEME = "seekpath"

# Limits of the trajectory check
# Rule of thumb: in the first 10 steps thermal motion moves an atom by about a
# tenth of an Angstrom or less, while a wrong atom order moves it by a bond length.
_N_START_FRAMES = 10
_START_LIMIT = 0.5  # Angstrom
# Lindemann ratio at melting of an fcc solid. It is about 0.18 for a bcc solid.
# Saija et al., J. Chem. Phys. 124, 244504 (2006)
_LINDEMANN_LIMIT = 0.15
# Rule of thumb: u_ref / u_vib = 1.5 means the mean positions moved by about as
# much as the atoms vibrate (u_shift = 1.1 u_vib).
_SHIFT_RATIO_LIMIT = 1.5
# Rule of thumb: the mean positions are only checked if the second half of the
# trajectory lasts at least one period of a 1 THz vibration.
_MIN_SHIFT_WINDOW = 1.0  # ps
# Three sigma rule: without a drift, a change this large has a probability below
# 5% for any unimodal distribution. Pukelsheim, Am. Stat. 48, 88 (1994)
_DRIFT_SIGMA = 3.0
# Rule of thumb: a third of a typical nearest-neighbor distance of 3 Angstrom,
# about twice the vibration at the Lindemann limit for that distance.
_RMS_LIMIT = 1.0  # Angstrom


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


def _read_vasp_md(
    directory: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    """Read the frames, cells, energies and time step of a VASP MD run."""
    incar = Incar.from_file(zpath(str(directory / "INCAR")))
    xdatcar = Xdatcar(zpath(str(directory / "XDATCAR")))
    frac_coords = np.array([frame.frac_coords for frame in xdatcar.structures])
    cells = np.array([frame.lattice.matrix for frame in xdatcar.structures])
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
    return frac_coords, cells, np.array(energies), species, float(incar["POTIM"])


def _read_ase_md(
    directory: Path, time_step: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    """Read the frames, cells and energies of a force field MD run."""
    # the first frame is the starting structure, before the first step, which
    # XDATCAR leaves out as well
    frames = ase_read(directory / ASE_TRAJECTORY_FILE, index=":")[1:]
    frac_coords = np.array([atoms.get_scaled_positions() for atoms in frames])
    cells = np.array([atoms.cell[:] for atoms in frames])
    energies = np.array([atoms.get_potential_energy() for atoms in frames])
    return frac_coords, cells, energies, frames[0].get_chemical_symbols(), time_step


def _read_md(
    md_dirs: Sequence[str], md_code: str, reference: Structure, md_time_step: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Read and join the trajectories of MD runs, in order.

    For VASP, XDATCAR, OSZICAR and the POTIM of INCAR are read from each run
    directory. For force fields, the ASE trajectory file is read, and
    md_time_step is the time step. Returns the fractional coordinates, the cell
    and the potential energy of each frame, and the time step in fs.
    """
    frac_coords, cells, energies = [], [], []
    for md_dir in md_dirs:
        directory = Path(strip_hostname(md_dir))
        if md_code == "vasp":
            coords, cell, energy, species, time_step = _read_vasp_md(directory)
        else:
            coords, cell, energy, species, time_step = _read_ase_md(
                directory, md_time_step
            )
        if species != [str(site.specie) for site in reference]:
            raise ValueError(
                f"The atoms of the MD in {directory} are not in the order of the "
                "reference supercell."
            )
        frac_coords.append(coords)
        cells.append(cell)
        energies.append(energy)
    return (
        np.concatenate(frac_coords),
        np.concatenate(cells),
        np.concatenate(energies),
        time_step,
    )


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
    time_step: float,
) -> TrajectoryHealth:
    """
    Check whether the MD trajectory stayed at the reference structure.

    The displacement of the center of mass of each frame is removed from its
    displacements. The checks are applied in this order. The limits are rules
    of thumb, except where a source is given next to them in this module.

    - A root mean square displacement above 0.5 Angstrom in the first 10
      frames means the atoms are not in the order of the reference.
    - A root mean square vibration u_vib above 0.15 of the nearest-neighbor
      distance means the structure melted.
    - A total displacement u_ref above 1.5 times u_vib means the mean
      positions moved away from the reference. This check is skipped if the
      second half of the trajectory is shorter than 1 ps.
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
    time_step: float
        MD time step in fs.

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
    shifted = (
        shift_ratio > _SHIFT_RATIO_LIMIT
        and len(tail) * time_step / 1000 >= _MIN_SHIFT_WINDOW
    )

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
    elif shifted:
        verdict = "transformed" if big_drift and drift < 0 else "shifted_or_diffusing"
    elif big_drift:
        verdict = "transformed" if drift < 0 else "disordering"
    elif rms_end > _RMS_LIMIT:
        verdict = "diffusing_or_soft"
    else:
        verdict = "stable"

    return TrajectoryHealth(
        verdict=verdict,
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
    )


def average_npt_structure(
    cells: np.ndarray,
    structure: Structure,
    supercell_matrix: Matrix3D,
    symprec: float,
) -> Structure:
    """
    Average the supercell lattices of an NPT MD into a unit cell.

    The metric tensor L L^T of the supercell, with the lattice vectors as the
    rows of L, is averaged over the cells. Unlike the lattice vectors, it does
    not change when the cell rotates. It is converted to the metric tensor of
    the unit cell with the supercell matrix and averaged over the point group of
    the structure, so that the cell keeps its symmetry. The new lattice is the
    stretch of the lattice of the structure, without a rotation, that has this
    metric tensor. The atoms keep their fractional coordinates.

    Parameters
    ----------
    cells: np.ndarray
        Supercell lattices of the frames to average, with the lattice vectors
        as rows, in Angstrom.
    structure: Structure
        Unit cell whose supercell the NPT MD started from.
    supercell_matrix: Matrix3D
        Supercell matrix.
    symprec: float
        Symmetry precision for the point group of the structure.

    Returns
    -------
    Structure
        The unit cell with the averaged lattice.
    """
    metric = np.mean([cell @ cell.T for cell in cells], axis=0)
    inv_matrix = np.linalg.inv(np.array(supercell_matrix, dtype=float))
    metric = inv_matrix @ metric @ inv_matrix.T
    # a rotation W of fractional coordinates leaves the metric tensor of the
    # symmetric cell unchanged, W^T G W = G
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    rotations = [op.rotation_matrix for op in analyzer.get_symmetry_operations()]
    metric = np.mean([rot.T @ metric @ rot for rot in rotations], axis=0)
    lattice = structure.lattice.matrix
    inv_lattice = np.linalg.inv(lattice)
    eigvals, eigvecs = np.linalg.eigh(inv_lattice @ metric @ inv_lattice.T)
    stretch = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    return Structure(
        lattice @ stretch,
        structure.species,
        structure.frac_coords,
        site_properties=structure.site_properties,
    )


@job
def get_npt_structure(
    npt_dir: str,
    md_code: str,
    structure: Structure,
    supercell_matrix: Matrix3D,
    reference: Structure,
    md_time_step: float,
    equilibration_time: float,
    symprec: float,
) -> dict:
    """
    Get the unit cell at the temperature from an NPT MD run.

    The cells after equilibration_time are averaged with
    :obj:`average_npt_structure`. The NPT trajectory is checked as in
    :obj:`select_md_snapshots`.

    Parameters
    ----------
    npt_dir: str
        Directory of the NPT MD run.
    md_code: str
        Code of the MD, "vasp" or "forcefields".
    structure: Structure
        Unit cell whose supercell the NPT MD started from.
    supercell_matrix: Matrix3D
        Supercell matrix.
    reference: Structure
        The undisplaced supercell the NPT MD started from.
    md_time_step: float
        MD time step in fs. Only used for force field trajectories.
    equilibration_time: float
        Time at the start of the trajectory that is left out, in ps.
    symprec: float
        Symmetry precision for the point group of the structure.

    Returns
    -------
    dict
        The unit cell at the temperature and the check of the NPT trajectory.
    """
    frac_coords, cells, energies, time_step = _read_md(
        [npt_dir], md_code, reference, md_time_step
    )
    n_equil = round(equilibration_time * 1000 / time_step)
    if n_equil >= len(cells):
        raise ValueError(
            f"The NPT trajectory has {len(cells)} frames of {time_step} fs, none "
            f"of them after the {equilibration_time} ps that are left out."
        )
    npt_structure = average_npt_structure(
        cells[n_equil:], structure, supercell_matrix, symprec
    )

    health = _assess_trajectory(frac_coords, energies, reference, time_step)
    if health.verdict != "stable":
        warnings.warn(
            f"The NPT trajectory did not stay at the reference structure (verdict: "
            f"{health.verdict}). The averaged cell may not describe it.",
            stacklevel=2,
        )
    return {"structure": npt_structure, "trajectory_health": health.model_dump()}


@job(data=["structures"])
def select_md_snapshots(
    md_dirs: Sequence[str],
    md_code: str,
    reference: Structure,
    md_time_step: float,
    equilibration_time: float,
    n_snapshots: int,
) -> dict:
    """
    Pick snapshots from the MD trajectory and check the trajectory.

    The trajectories of all MD runs are joined in order, with a frame after
    every time step. The time of a frame is its step number times the time
    step. The frames of the first equilibration_time are left out, and
    n_snapshots frames are picked evenly spread over the rest, from the first
    to the last. For VASP, XDATCAR, OSZICAR and the POTIM of INCAR are
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
    all_coords, _, all_energies, time_step = _read_md(
        md_dirs, md_code, reference, md_time_step
    )
    n_frames = len(all_coords)

    n_equil = round(equilibration_time * 1000 / time_step)
    n_left = n_frames - n_equil
    if n_left < n_snapshots:
        raise ValueError(
            f"The trajectory has {n_frames} frames of {time_step} fs. After "
            f"leaving out {equilibration_time} ps, {max(n_left, 0)} frames are "
            f"left, fewer than the {n_snapshots} snapshots."
        )
    indices = np.linspace(n_equil, n_frames - 1, n_snapshots).round().astype(int)

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
    health = _assess_trajectory(all_coords, all_energies, reference, time_step)
    if health.verdict != "stable":
        warnings.warn(
            f"The MD trajectory did not stay at the reference structure (verdict: "
            f"{health.verdict}). The fitted force constants may not describe it.",
            stacklevel=2,
        )
    return {
        "structures": [*snapshots, reference],
        "snapshot_times": [(idx + 1) * time_step / 1000 for idx in indices],
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
    frequencies = (
        np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * phonon.unit_conversion_factor
    )
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
    md_job_dirs: list[str] | None = None,
    optimization_run_uuid: str | None = None,
    optimization_run_job_dir: str | None = None,
    born_run_uuid: str | None = None,
    born_run_job_dir: str | None = None,
    npt_input_structure: Structure | None = None,
    pressure: float | None = None,
    npt_time: float | None = None,
    npt_equilibration_time: float | None = None,
    npt_trajectory_health: dict | None = None,
    npt_uuid: str | None = None,
    npt_job_dir: str | None = None,
    fixed_cell_relax_uuid: str | None = None,
    fixed_cell_relax_job_dir: str | None = None,
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
        Unit cell of the NVT MD, the relaxed unit cell or the cell from the NPT
        MD. This job sorts its atoms by electronegativity.
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
    md_job_dirs: list[str] | None
        Directories of the MD jobs.
    optimization_run_uuid: str | None
        UUID of the relaxation.
    optimization_run_job_dir: str | None
        Directory of the relaxation.
    born_run_uuid: str | None
        UUID of the Born charge calculation.
    born_run_job_dir: str | None
        Directory of the Born charge calculation.
    npt_input_structure: Structure | None
        Unit cell whose supercell the NPT MD started from, if there was one.
    pressure: float | None
        Pressure of the NPT MD in kbar.
    npt_time: float | None
        Length of the NPT MD in ps.
    npt_equilibration_time: float | None
        Time at the start of the NPT trajectory that was left out, in ps.
    npt_trajectory_health: dict | None
        Check of the NPT trajectory.
    npt_uuid: str | None
        UUID of the NPT MD job.
    npt_job_dir: str | None
        Directory of the NPT MD job.
    fixed_cell_relax_uuid: str | None
        UUID of the relaxation of the atoms in the cell from the NPT MD.
    fixed_cell_relax_job_dir: str | None
        Directory of the relaxation of the atoms in the cell from the NPT MD.
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
    structure = structure.get_sorted_structure()
    if born is not None and len(born) != len(structure):
        raise ValueError("The number of Born charges is not the number of atoms.")
    phonon = _get_phonopy(structure, supercell_matrix, symprec, code)
    supercell = get_pmg_structure(phonon.supercell)

    forces = np.array(displacement_data["forces"])
    structures = displacement_data["displaced_structures"]
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

    log_file = Path(_DEFAULT_FILE_PATHS["harmonic_fit_log"])
    fc_file = Path(_DEFAULT_FILE_PATHS["force_constants"])
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

    borns, epsilon = _set_nac_params(phonon, born, epsilon_static, symprec, code)

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
        lowest_frequency=float(frequencies.min()),
        phonon_bandstructure=bs_symm_line,
        phonon_dos=dos,
        force_constants=ForceConstants(phonon.force_constants.tolist())
        if store_force_constants
        else None,
        born=borns.tolist() if borns is not None else None,
        epsilon_static=epsilon.tolist() if epsilon is not None else None,
        md_uuids=md_uuids,
        md_job_dirs=md_job_dirs,
        npt_input_structure=npt_input_structure,
        pressure=pressure,
        npt_time=npt_time,
        npt_equilibration_time=npt_equilibration_time,
        npt_trajectory_health=npt_trajectory_health,
        npt_uuid=npt_uuid,
        npt_job_dir=npt_job_dir,
        fixed_cell_relax_uuid=fixed_cell_relax_uuid,
        fixed_cell_relax_job_dir=fixed_cell_relax_job_dir,
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
