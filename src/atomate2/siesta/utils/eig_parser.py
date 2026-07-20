"""
Utilities for parsing SIESTA .EIG (eigenvalue) files.

This module provides functions to parse SIESTA eigenvalue files and calculate
electronic properties such as band gaps, VBM, and CBM.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypedDict

    class BandGapInfo(TypedDict):
        """Type hint for band gap information dictionary."""

        vbm: float | None
        cbm: float | None
        gap: float | None
        is_metallic: bool


logger = logging.getLogger(__name__)


def parse_eig_file(filepath: str | Path) -> dict[str, float | int | list[float]]:
    """
    Parse SIESTA .EIG file.

    The .EIG file format:
    - Line 1: Fermi energy (eV)
    - Line 2: n_eigenvalues n_spin n_kpoints
    - Following lines: k-point_index eigenvalue1 eigenvalue2 ...

    Parameters
    ----------
    filepath : str | Path
        Path to the .EIG file

    Returns
    -------
    dict
        Dictionary containing:
        - fermi_energy: Fermi level in eV
        - n_eigenvalues: Number of eigenvalues per k-point
        - n_spin: Number of spin components
        - n_kpoints: Number of k-points (if available)
        - eigenvalues: List of all unique eigenvalues across all k-points

    Raises
    ------
    FileNotFoundError
        If the .EIG file does not exist
    ValueError
        If the file format is invalid
    """
    logger.info(f"parse_eig_file({filepath})")

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"EIG file not found: {filepath}")

    with open(filepath) as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError(f"Invalid EIG file format: {filepath}")

    # Line 1: Fermi energy
    try:
        fermi_energy = float(lines[0].strip())
    except ValueError as e:
        raise ValueError(f"Cannot parse Fermi energy from line 1: {lines[0]}") from e

    # Line 2: Format is "n_eigenvalues n_spin n_kpoints"
    parts = lines[1].strip().split()
    if len(parts) < 2:
        raise ValueError(f"Invalid header format in line 2: {lines[1]}")

    n_eigenvalues = int(parts[0])
    n_spin = int(parts[1])
    n_kpoints = int(parts[2]) if len(parts) > 2 else None

    # Parse eigenvalues from remaining lines
    eigenvalues = []
    for line in lines[2:]:
        if not line.strip():
            continue
        values = line.strip().split()
        # Skip k-point index (first value), extract eigenvalues
        try:
            eigs = [float(v) for v in values[1:]]
            eigenvalues.extend(eigs)
        except ValueError:
            logger.warning(f"Skipping malformed line: {line.strip()}")
            continue

    # Remove duplicates and sort (eigenvalues may be repeated for spin)
    unique_eigenvalues = sorted(set(eigenvalues))

    return {
        "fermi_energy": fermi_energy,
        "n_eigenvalues": n_eigenvalues,
        "n_spin": n_spin,
        "n_kpoints": n_kpoints,
        "eigenvalues": unique_eigenvalues,
    }


def calculate_band_gap(
    eigenvalues: list[float], fermi_energy: float, tolerance: float = 1e-6
) -> BandGapInfo:
    """
    Calculate band gap from eigenvalues and Fermi energy.

    Parameters
    ----------
    eigenvalues : list[float]
        Sorted list of unique eigenvalues in eV
    fermi_energy : float
        Fermi energy in eV
    tolerance : float, default=1e-6
        Tolerance for comparing eigenvalues to Fermi level (eV)

    Returns
    -------
    BandGapInfo
        Dictionary containing:
        - vbm: Valence band maximum (eV)
        - cbm: Conduction band minimum (eV)
        - gap: Band gap (eV), None if metallic
        - is_metallic: Boolean indicating if system is metallic
    """
    logger.info("calculate_band_gap()")

    # Find VBM: highest eigenvalue <= E_fermi
    vbm = None
    for eig in reversed(eigenvalues):
        if eig <= fermi_energy + tolerance:
            vbm = eig
            break

    # Find CBM: lowest eigenvalue > E_fermi
    cbm = None
    for eig in eigenvalues:
        if eig > fermi_energy + tolerance:
            cbm = eig
            break

    # Check if metallic (states at Fermi level)
    is_metallic = False
    gap = None

    if vbm is not None and cbm is not None:
        gap = cbm - vbm
        # Consider metallic if gap is very small (< tolerance)
        if gap < tolerance:
            is_metallic = True
            gap = 0.0
    else:
        # If we can't find both VBM and CBM, it's likely metallic
        is_metallic = True
        gap = None

    return {
        "vbm": vbm,
        "cbm": cbm,
        "gap": gap,
        "is_metallic": is_metallic,
    }


def get_band_gap_from_eig(filepath: str | Path) -> BandGapInfo:
    """
    Extract band gap information from a SIESTA .EIG file.

    This is a convenience function that combines parsing and calculation.

    Parameters
    ----------
    filepath : str | Path
        Path to the .EIG file

    Returns
    -------
    BandGapInfo
        Dictionary containing VBM, CBM, gap, and is_metallic

    Raises
    ------
    FileNotFoundError
        If the .EIG file does not exist
    ValueError
        If the file format is invalid

    Examples
    --------
    >>> gap_info = get_band_gap_from_eig("siesta.EIG")
    >>> print(f"Band gap: {gap_info['gap']:.3f} eV")
    >>> print(f"VBM: {gap_info['vbm']:.3f} eV")
    >>> print(f"CBM: {gap_info['cbm']:.3f} eV")
    """
    logger.info(f"get_band_gap_from_eig({filepath})")

    eig_data = parse_eig_file(filepath)
    gap_info = calculate_band_gap(eig_data["eigenvalues"], eig_data["fermi_energy"])

    logger.info(
        f"Band gap analysis: VBM={gap_info['vbm']:.3f} eV, "
        f"CBM={gap_info['cbm']:.3f} eV, gap={gap_info['gap']:.3f} eV, "
        f"metallic={gap_info['is_metallic']}"
    )

    return gap_info
