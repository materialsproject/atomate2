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
    lengths: Sequence[int] = (40, 30, 20),
) -> int:
    """Return Rk: lengths[2] above thresholds[1], lengths[1] in between, else [0]."""
    if bandgap is None or bandgap < thresholds[0]:
        return round(lengths[0])
    if bandgap <= thresholds[1]:
        return round(lengths[1])
    return round(lengths[2])


def magnetic_order(magmoms: Sequence[float] | None, tol: float = 0.05) -> str:
    """Return "NM", "FM", or "AFM" (moments of both signs, ferrimagnets included)."""
    if magmoms is None:
        return "NM"
    moments = np.array(
        [np.linalg.norm(m) * np.sign(np.sum(m)) for m in magmoms], dtype=float
    )
    moments = moments[np.abs(moments) > tol]
    if len(moments) == 0:
        return "NM"
    if np.all(moments > 0) or np.all(moments < 0):
        return "FM"
    return "AFM"


def build_phonopy(
    structure: Structure,
    force_constants: Any,
    supercell_matrix: Sequence[Sequence[float]],
    primitive_matrix: Sequence[Sequence[float]] | str | None = "auto",
    symprec: float = 1e-4,
    magnetic: bool = False,
) -> Phonopy:
    """Rebuild a Phonopy object from the outputs of the phonon flow.

    With magnetic=True the magmom site property is kept, so the symmetry and the
    automatic primitive cell are those of the magnetic structure.
    """
    if force_constants is None:
        raise ValueError("Phonon document has no force constants.")
    if hasattr(force_constants, "force_constants"):
        force_constants = force_constants.force_constants
    elif isinstance(force_constants, dict):
        force_constants = force_constants["force_constants"]

    struct = structure.copy()
    if magnetic:
        if struct.site_properties.get("magmom") is None:
            raise ValueError("magnetic=True needs a magmom site property.")
        primitive_matrix = "auto"
    elif "magmom" in struct.site_properties:
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


def transfer_magmoms(target: Structure, source: Structure) -> Structure:
    """Copy the magmom site property of source onto the matching sites of target."""
    if len(target) != len(source):
        raise ValueError("Structures for transferring magmoms differ in size.")
    magmoms = source.site_properties["magmom"]
    frac = np.array([site.frac_coords for site in source])
    result = target.copy()
    values = []
    for site in target:
        diff = frac - site.frac_coords
        diff -= np.rint(diff)
        dist = np.linalg.norm(diff @ target.lattice.matrix, axis=1)
        j = int(np.argmin(dist))
        if dist[j] > 0.5 or source[j].species != site.species:
            raise ValueError("Could not match sites for transferring magmoms.")
        values.append(magmoms[j])
    result.add_site_property("magmom", values)
    return result


def supercell_magmoms(
    structure: Structure, supercell_matrix: Sequence[Sequence[float]]
) -> list:
    """Return the magmom of every atom of the phonopy supercell of structure."""
    magmoms = structure.site_properties["magmom"]
    struct = structure.copy()
    struct.remove_site_property("magmom")
    phonon = Phonopy(
        get_phonopy_structure(struct),
        supercell_matrix=np.asarray(supercell_matrix, dtype=int),
        primitive_matrix="P",
    )
    supercell = phonon.supercell
    return [magmoms[supercell.u2u_map[i]] for i in supercell.s2u_map]


def _primitive_operations(phonon: Phonopy) -> list[np.ndarray]:
    """Return the 3N x 3N Cartesian matrices of the primitive-cell point group."""
    prim = phonon.primitive
    lattice = np.array(prim.cell)
    positions = np.array(prim.scaled_positions)
    natoms = len(positions)
    ops = phonon.primitive_symmetry.symmetry_operations
    to_cart = lattice.T
    to_frac = np.linalg.inv(lattice.T)

    matrices = []
    for rot, trans in zip(ops["rotations"], ops["translations"], strict=True):
        rot_cart = to_cart @ rot @ to_frac
        matrix = np.zeros((3 * natoms, 3 * natoms))
        for i, pos in enumerate(positions):
            diff = positions - (rot @ pos + trans)
            diff -= np.rint(diff)
            dist = np.linalg.norm(diff @ lattice, axis=1)
            j = int(np.argmin(dist))
            if dist[j] > 10 * phonon.primitive_symmetry.tolerance:
                raise ValueError("Could not map atoms with a symmetry operation.")
            matrix[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] = rot_cart
        matrices.append(matrix)
    return matrices


def symmetrize_gamma_dynamical_matrix(
    phonon: Phonopy, dynamical_matrix: np.ndarray
) -> np.ndarray:
    """Average the Gamma dynamical matrix over the primitive-cell point group."""
    matrices = _primitive_operations(phonon)
    return sum(g @ dynamical_matrix @ g.T for g in matrices) / len(matrices)


def gamma_normal_modes(phonon: Phonopy, symmetrize: bool = True) -> dict[str, Any]:
    """Real eigenvectors and frequencies of the Gamma dynamical matrix.

    With symmetrize=True the dynamical matrix is first averaged over the
    point group of the primitive cell. This restores exact degeneracies when the
    phonon supercell has a lower symmetry than the crystal.
    """
    dm = phonon.dynamical_matrix
    dm.run([0.0, 0.0, 0.0])
    dyn = np.array(dm.dynamical_matrix)
    dyn = 0.5 * (dyn + dyn.conj().T).real
    if symmetrize:
        dyn = symmetrize_gamma_dynamical_matrix(phonon, dyn)

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


def gamma_instabilities(
    frequencies_thz: Sequence[float], max_imaginary: float | None
) -> list[int]:
    """Return the modes with a frequency below -max_imaginary (THz)."""
    if max_imaginary is None:
        return []
    return [k for k, f in enumerate(frequencies_thz) if f < -abs(max_imaginary)]


def acoustic_mode_indices(frequencies_thz: Sequence[float]) -> list[int]:
    """Return indices of the three modes with the smallest |frequency|."""
    return sorted(np.argsort(np.abs(np.asarray(frequencies_thz)))[:3].tolist())


def _run_gamma_irreps(phonon: Phonopy, degeneracy_tolerance: float) -> Any:
    """Run phonopy irreps at Gamma, grouping degenerate modes by symmetry."""
    try:
        phonon.run_irreps([0, 0, 0], symmetry_adapted=True)
    except TypeError:
        phonon.run_irreps([0, 0, 0], degeneracy_tolerance=degeneracy_tolerance)
    return phonon.irreps


def gamma_selection_rules(
    phonon: Phonopy,
    nmodes: int,
    degeneracy_tolerance: float = 1e-3,
    eigenvecs: np.ndarray | None = None,
) -> dict[str, Any]:
    """Irrep labels and Raman and IR activity of each mode at Gamma.

    Parameters
    ----------
    phonon
        Phonopy object with force constants.
    nmodes
        Number of modes (3N).
    degeneracy_tolerance
        Frequency tolerance in THz, only used if phonopy has no symmetry-adapted
        irreps.
    eigenvecs
        Mass-weighted eigenvectors (3N, N, 3) of the modes to label. Each mode is
        assigned to the phonopy degenerate set it overlaps most with. If None,
        the phonopy band order is used.
    """
    irreps = _run_gamma_irreps(phonon, degeneracy_tolerance)

    rotations = np.array(irreps.conventional_rotations, dtype=float)
    tr = np.trace(rotations, axis1=1, axis2=2)
    tr2 = np.trace(rotations @ rotations, axis1=1, axis2=2)
    chi_raman = 0.5 * (tr**2 + tr2)
    order = len(rotations)
    labels_per_set = getattr(irreps, "_ir_labels", None)
    phonopy_sets = [list(map(int, s)) for s in irreps.band_indices]

    if eigenvecs is None:
        set_of_mode = {band: i for i, s in enumerate(phonopy_sets) for band in s}
    else:
        vecs = np.asarray(eigenvecs, dtype=float).reshape(nmodes, nmodes)
        basis = np.asarray(irreps.eigenvectors)
        weights = np.array(
            [
                np.sum(np.abs(basis[:, s].conj().T @ vecs.T) ** 2, axis=0)
                for s in phonopy_sets
            ]
        )
        set_of_mode = {k: int(np.argmax(weights[:, k])) for k in range(nmodes)}

    raman_active = [False] * nmodes
    ir_active = [False] * nmodes
    irrep_labels: list[str | None] = [None] * nmodes
    degenerate_sets: list[list[int]] = [[] for _ in phonopy_sets]

    for k in range(nmodes):
        i = set_of_mode[k]
        chars = irreps.characters[i]
        n_raman = float(np.real(np.sum(np.conj(chars) * chi_raman)) / order)
        n_ir = float(np.real(np.sum(np.conj(chars) * tr)) / order)
        raman_active[k] = n_raman > 0.25
        ir_active[k] = n_ir > 0.25
        if labels_per_set is not None and labels_per_set[i] is not None:
            irrep_labels[k] = str(labels_per_set[i])
        degenerate_sets[i].append(k)

    return {
        "raman_active": raman_active,
        "ir_active": ir_active,
        "irrep_labels": irrep_labels,
        "degenerate_sets": sorted(s for s in degenerate_sets if s),
        "pointgroup": str(phonon.primitive_symmetry.pointgroup_symbol),
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
