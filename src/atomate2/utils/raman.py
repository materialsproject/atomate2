"""Normal-mode, selection-rule and Raman-tensor utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from phonopy import Phonopy
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Structure

VASP_TO_THZ = 15.633302153867165
THZ_TO_CM = 521.470807261 / VASP_TO_THZ


def get_encut_from_potcar(
    structure: Structure, input_set_generator: Any, factor: float = 1.3
) -> float:
    """Return factor * max(ENMAX) of the POTCARs chosen by the input set generator."""
    try:
        potcar = input_set_generator.get_input_set(structure).potcar
    except Exception as exc:
        raise RuntimeError(
            "Could not read POTCARs (check PMG_VASP_PSP_DIR) to set ENCUT; "
            "pass encut explicitly instead."
        ) from exc
    return round(factor * max(p.enmax for p in potcar), 4)


def kpoints_length_from_bandgap(
    bandgap: float | None,
    thresholds: Sequence[float] = (1.0, 2.0),
    lengths: Sequence[float] = (40.0, 30.0, 20.0),
) -> float:
    """Return Rk: lengths[2] above thresholds[1], lengths[1] in between, else [0]."""
    if bandgap is None or bandgap < thresholds[0]:
        return float(lengths[0])
    if bandgap <= thresholds[1]:
        return float(lengths[1])
    return float(lengths[2])


def build_phonopy(
    structure: Structure,
    force_constants: Any,
    supercell_matrix: Sequence[Sequence[float]],
    primitive_matrix: Sequence[Sequence[float]] | str | None = "auto",
    symprec: float = 1e-4,
) -> Phonopy:
    """Rebuild a Phonopy object from the outputs of the phonon flow."""
    if force_constants is None:
        raise ValueError("Phonon document has no force constants.")
    if hasattr(force_constants, "force_constants"):
        force_constants = force_constants.force_constants
    elif isinstance(force_constants, dict):
        force_constants = force_constants["force_constants"]

    struct = structure.copy()
    if "magmom" in struct.site_properties:
        struct.remove_site_property("magmom")
    if primitive_matrix is None:
        primitive_matrix = "auto"
    elif not isinstance(primitive_matrix, str):
        primitive_matrix = np.asarray(primitive_matrix, dtype=float)

    phonon = Phonopy(
        get_phonopy_structure(struct),
        supercell_matrix=np.asarray(supercell_matrix, dtype=int),
        primitive_matrix=primitive_matrix,
        symprec=symprec,
    )
    phonon.force_constants = np.asarray(force_constants, dtype=float)
    return phonon


def gamma_normal_modes(phonon: Phonopy) -> dict[str, Any]:
    """Real eigenvectors and frequencies of the Gamma dynamical matrix."""
    dm = phonon.dynamical_matrix
    dm.run([0.0, 0.0, 0.0])
    dyn = np.array(dm.dynamical_matrix)
    dyn = 0.5 * (dyn + dyn.conj().T).real

    eigvals, eigvecs = np.linalg.eigh(dyn)
    frequencies = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * VASP_TO_THZ
    natoms = len(phonon.primitive)
    eigvecs = eigvecs.T.reshape(3 * natoms, natoms, 3)
    for k in range(3 * natoms):
        flat = eigvecs[k].ravel()
        if flat[np.argmax(np.abs(flat))] < 0:
            eigvecs[k] *= -1

    return {
        "structure": get_pmg_structure(phonon.primitive),
        "masses": np.array(phonon.primitive.masses, dtype=float),
        "frequencies_thz": frequencies,
        "eigenvals": -np.sign(frequencies) * (frequencies / VASP_TO_THZ) ** 2,
        "eigenvecs": eigvecs,
    }


def acoustic_mode_indices(frequencies_thz: Sequence[float]) -> list[int]:
    """Return indices of the three modes with the smallest |frequency|."""
    return sorted(np.argsort(np.abs(np.asarray(frequencies_thz)))[:3].tolist())


def gamma_selection_rules(
    phonon: Phonopy, nmodes: int, degeneracy_tolerance: float = 1e-3
) -> dict[str, Any]:
    """Raman and IR activity of each mode from the Gamma irreps characters."""
    if hasattr(phonon, "run_irreps"):
        phonon.run_irreps([0, 0, 0], degeneracy_tolerance=degeneracy_tolerance)
    else:
        phonon.set_irreps([0, 0, 0], degeneracy_tolerance=degeneracy_tolerance)
    irreps = phonon.irreps

    rotations = np.array(irreps.conventional_rotations, dtype=float)
    tr = np.trace(rotations, axis1=1, axis2=2)
    tr2 = np.trace(rotations @ rotations, axis1=1, axis2=2)
    chi_raman = 0.5 * (tr**2 + tr2)
    order = len(rotations)
    labels_per_set = getattr(irreps, "_ir_labels", None)

    raman_active = [False] * nmodes
    ir_active = [False] * nmodes
    irrep_labels: list[str | None] = [None] * nmodes
    degenerate_sets = [list(map(int, s)) for s in irreps.band_indices]

    for i, (deg_set, chars) in enumerate(
        zip(degenerate_sets, irreps.characters, strict=True)
    ):
        n_raman = float(np.real(np.sum(np.conj(chars) * chi_raman)) / order)
        n_ir = float(np.real(np.sum(np.conj(chars) * tr)) / order)
        label = None
        if labels_per_set is not None and labels_per_set[i] is not None:
            label = str(labels_per_set[i])
        for band in deg_set:
            raman_active[band] = n_raman > 0.25
            ir_active[band] = n_ir > 0.25
            irrep_labels[band] = label

    return {
        "raman_active": raman_active,
        "ir_active": ir_active,
        "irrep_labels": irrep_labels,
        "degenerate_sets": degenerate_sets,
        "pointgroup": str(phonon.symmetry.pointgroup_symbol),
    }


def mode_displacement_pattern(
    eigenvec: np.ndarray, masses: np.ndarray
) -> tuple[np.ndarray, float]:
    """Unit-norm Cartesian pattern e/sqrt(m) and the norm |e/sqrt(m)|."""
    pattern = np.asarray(eigenvec, dtype=float) / np.sqrt(masses)[:, np.newaxis]
    norm = float(np.linalg.norm(pattern))
    return pattern / norm, norm


def displace_along_mode(
    structure: Structure, eigenvec: np.ndarray, masses: np.ndarray, displacement: float
) -> Structure:
    """Displace all sites along a normal mode by a total norm in Angstrom."""
    unit_pattern, _ = mode_displacement_pattern(eigenvec, masses)
    displaced = structure.copy()
    for i, vec in enumerate(unit_pattern * displacement):
        displaced.translate_sites([i], vec, frac_coords=False, to_unit_cell=False)
    return displaced


def raman_tensor_from_epsilons(
    displacements: Sequence[float],
    epsilons: Sequence[Sequence[Sequence[float]]],
    volume: float,
    pattern_norm: float,
) -> np.ndarray:
    """Central-difference Raman tensor V/(4pi) * d(eps)/du * |e/sqrt(m)|."""
    if len(displacements) != 2:
        raise ValueError(f"Need two displacements per mode, got {len(displacements)}")
    eps_a, eps_b = (np.asarray(e, dtype=float) for e in epsilons)
    d_a, d_b = displacements
    return volume / (4.0 * np.pi) * (eps_a - eps_b) / (d_a - d_b) * pattern_norm


def raman_activity(tensor: np.ndarray) -> float:
    """Orientation-averaged Raman activity 45 alpha^2 + 7 beta^2."""
    r = 0.5 * (np.asarray(tensor) + np.asarray(tensor).T)
    alpha = np.trace(r) / 3.0
    beta2 = 0.5 * (
        (r[0, 0] - r[1, 1]) ** 2
        + (r[1, 1] - r[2, 2]) ** 2
        + (r[2, 2] - r[0, 0]) ** 2
        + 6.0 * (r[0, 1] ** 2 + r[0, 2] ** 2 + r[1, 2] ** 2)
    )
    return float(45.0 * alpha**2 + 7.0 * beta2)


def crd_intensity(tensor: np.ndarray) -> float:
    """Powder intensity used in atomate-CRD plot_raman.py."""
    r = np.asarray(tensor)
    return float(
        4 * (r[0, 0] ** 2 + r[1, 1] ** 2 + r[2, 2] ** 2)
        + 7 * (r[0, 1] ** 2 + r[0, 2] ** 2 + r[1, 2] ** 2)
        + r[0, 0] * r[1, 1]
        + r[0, 0] * r[2, 2]
        + r[1, 1] * r[2, 2]
    )


def lorentzian_spectrum(
    frequencies_cm: Sequence[float],
    intensities: Sequence[float],
    width: float = 4.0,
    grid: np.ndarray | None = None,
    npoints: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised spectrum with the PRB 73, 104304 (Eq. 8) line shape."""
    freqs = np.asarray(frequencies_cm, dtype=float)
    ints = np.asarray(intensities, dtype=float)
    if grid is None:
        fmax = max(float(freqs.max()) if freqs.size else 100.0, 100.0)
        grid = np.linspace(0.0, 1.1 * fmax + 5 * width, npoints)
    spectrum = np.zeros_like(grid)
    for wj, inten in zip(freqs, ints, strict=True):
        spectrum += (
            inten
            * 4.0
            / np.pi
            * (grid**2 * width)
            / ((grid**2 - wj**2) ** 2 + 4 * width**2 * grid**2)
        )
    if spectrum.max() > 0:
        spectrum /= spectrum.max()
    return grid, spectrum
