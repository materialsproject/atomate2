"""Utilities for reading and updating SIESTA FDF files.

This module provides functions to safely update FDF parameters
during error recovery.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_fdf_file(filepath: Path | str) -> dict[str, Any]:
    """Read FDF file and parse parameters.

    Parameters
    ----------
    filepath : Path or str
        Path to siesta.fdf file

    Returns
    -------
    dict
        Dictionary of parameter name → value

    Example
    -------
    >>> params = read_fdf_file("siesta.fdf")
    >>> print(params["MeshCutoff"])
    300 Ry
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"FDF file not found: {filepath}")
        return {}

    params = {}

    with open(filepath) as f:
        for line in f:
            # Skip comments and empty lines
            cleaned = line.split("#")[0].strip()
            if not cleaned:
                continue

            # Skip block definitions
            if cleaned.startswith(("%block", "%endblock")):
                continue

            # Parse key-value pairs
            match = re.match(r"^(\S+)\s+(.+)$", cleaned)
            if match:
                key = match.group(1)
                value = match.group(2).strip()
                params[key] = value

    return params


def update_fdf_file(
    filepath: Path | str,
    updates: dict[str, Any],
    backup: bool = True,
) -> bool:
    """Update FDF file with new parameters.

    Parameters
    ----------
    filepath : Path or str
        Path to siesta.fdf file
    updates : dict
        Dictionary of parameter → value to update
    backup : bool, optional
        Create backup before modifying (default: True)

    Returns
    -------
    bool
        True if successful, False otherwise

    Example
    -------
    >>> updates = {"SCF.Mixer.Weight": 0.01, "MeshCutoff": "400 Ry"}
    >>> update_fdf_file("siesta.fdf", updates)
    True
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"FDF file not found: {filepath}")
        return False

    # Create backup
    if backup:
        backup_path = filepath.with_suffix(".fdf.bak")
        shutil.copy2(filepath, backup_path)
        logger.debug(f"Created backup: {backup_path}")

    try:
        # Read current content
        with open(filepath) as f:
            lines = f.readlines()

        # Apply updates
        updated_lines = _apply_updates(lines, updates)

        # Write updated file
        with open(filepath, "w") as f:
            f.writelines(updated_lines)

        logger.info(f"Updated {len(updates)} parameters in {filepath}")

    except Exception as e:  # noqa: BLE001 recover from any FDF write failure
        # TRY400: message-only log intentional (no traceback needed)
        logger.error(f"Error updating FDF file: {e}")  # noqa: TRY400
        # Restore from backup if it exists
        if backup:
            backup_path = filepath.with_suffix(".fdf.bak")
            if backup_path.exists():
                shutil.copy2(backup_path, filepath)
                logger.info("Restored from backup")
        return False
    else:
        return True


def _apply_updates(lines: list[str], updates: dict[str, Any]) -> list[str]:
    """Apply parameter updates to FDF lines.

    Parameters
    ----------
    lines : list of str
        Original FDF file lines
    updates : dict
        Parameters to update

    Returns
    -------
    list of str
        Updated lines
    """
    updated_lines = []
    updated_keys = set()

    for line in lines:
        # Preserve comments and empty lines
        if line.strip().startswith("#") or not line.strip():
            updated_lines.append(line)
            continue

        # Preserve block definitions
        if line.strip().startswith("%block") or line.strip().startswith("%endblock"):
            updated_lines.append(line)
            continue

        # Check if this line contains a parameter to update
        updated = False
        for key, value in updates.items():
            # Match parameter name (case-insensitive, handle dots)
            pattern = re.compile(rf"^{re.escape(key)}\s+", re.IGNORECASE)
            if pattern.match(line.strip()):
                # Update this line
                updated_lines.append(f"{key}\t{value}\n")
                updated_keys.add(key)
                updated = True
                logger.debug(f"Updated: {key} = {value}")
                break

        if not updated:
            updated_lines.append(line)

    # Add parameters that weren't found in original file
    for key, value in updates.items():
        if key not in updated_keys:
            updated_lines.append(f"{key}\t{value}\n")
            logger.debug(f"Added: {key} = {value}")

    return updated_lines


def format_fdf_value(value: Any) -> str:
    """Format Python value for FDF file.

    Parameters
    ----------
    value : Any
        Python value (bool, int, float, str, list)

    Returns
    -------
    str
        Formatted FDF value

    Example
    -------
    >>> format_fdf_value(True)
    'T'
    >>> format_fdf_value([4, 4, 4])
    '4 4 4'
    >>> format_fdf_value(300.0)
    '300.0'
    """
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def apply_corrections(
    directory: Path | str,
    corrections: dict[str, Any],
    fdf_filename: str = "siesta.fdf",
) -> bool:
    """Apply error corrections to FDF file.

    Convenience function to format and apply corrections.

    Parameters
    ----------
    directory : Path or str
        Directory containing FDF file
    corrections : dict
        Corrections from error handler
    fdf_filename : str, optional
        Name of FDF file (default: "siesta.fdf")

    Returns
    -------
    bool
        True if successful

    Example
    -------
    >>> corrections = {"SCF.Mixer.Weight": 0.01, "kpts": [6, 6, 6]}
    >>> apply_corrections("job_001", corrections)
    True
    """
    directory = Path(directory)
    fdf_path = directory / fdf_filename

    # Format values
    formatted = {key: format_fdf_value(value) for key, value in corrections.items()}

    # Apply updates
    success = update_fdf_file(fdf_path, formatted)

    if success:
        logger.info(f"Applied {len(corrections)} corrections to {fdf_path}")
        for key, value in corrections.items():
            logger.info(f"  {key} = {value}")

    return success


def get_fdf_parameter(
    filepath: Path | str,
    parameter: str,
    default: Any = None,
) -> Any:
    """Get a specific parameter from FDF file.

    Parameters
    ----------
    filepath : Path or str
        Path to FDF file
    parameter : str
        Parameter name
    default : Any, optional
        Default value if not found

    Returns
    -------
    Any
        Parameter value or default

    Example
    -------
    >>> cutoff = get_fdf_parameter("siesta.fdf", "MeshCutoff", "300 Ry")
    >>> print(cutoff)
    400 Ry
    """
    params = read_fdf_file(filepath)
    return params.get(parameter, default)


def validate_fdf_file(filepath: Path | str) -> tuple[bool, list[str]]:
    """Validate FDF file for common issues.

    Parameters
    ----------
    filepath : Path or str
        Path to FDF file

    Returns
    -------
    tuple of (bool, list of str)
        (is_valid, list of warning messages)

    Example
    -------
    >>> is_valid, warnings = validate_fdf_file("siesta.fdf")
    >>> if not is_valid:
    ...     for warning in warnings:
    ...         print(f"Warning: {warning}")
    """
    filepath = Path(filepath)
    warnings = []

    if not filepath.exists():
        return False, ["FDF file does not exist"]

    try:
        params = read_fdf_file(filepath)

        # Check for essential parameters
        if "SystemName" not in params and "SystemLabel" not in params:
            warnings.append("Neither SystemName nor SystemLabel defined")

        # Check for k-points
        has_kpoints = False
        if any(key.startswith("kgrid") for key in params):
            has_kpoints = True
        if any(key.startswith("%block kgrid") for key in params):
            has_kpoints = True

        if not has_kpoints:
            warnings.append("No k-point sampling defined")

        # Check for basis set
        if "PAO.BasisSize" not in params:
            warnings.append("Basis size not specified")

        # Check for mesh cutoff
        if "MeshCutoff" not in params:
            warnings.append("Mesh cutoff not specified")

        is_valid = len(warnings) == 0

    except Exception as e:  # noqa: BLE001 report any FDF read failure as invalid
        return False, [f"Error reading FDF file: {e}"]
    else:
        return is_valid, warnings


def merge_fdf_updates(
    *update_dicts: dict[str, Any],
    priority: str = "last",
) -> dict[str, Any]:
    """Merge multiple update dictionaries.

    Parameters
    ----------
    *update_dicts : dict
        Dictionaries of updates to merge
    priority : str, optional
        Which dict takes priority: "first" or "last" (default: "last")

    Returns
    -------
    dict
        Merged updates

    Example
    -------
    >>> updates1 = {"SCF.Mixer.Weight": 0.05}
    >>> updates2 = {"SCF.Mixer.Weight": 0.01, "MeshCutoff": "400 Ry"}
    >>> merged = merge_fdf_updates(updates1, updates2, priority="last")
    >>> print(merged["SCF.Mixer.Weight"])
    0.01
    """
    merged = {}

    if priority == "last":
        # Later dicts override earlier ones
        for update_dict in update_dicts:
            merged.update(update_dict)
    else:
        # First dict takes priority
        for update_dict in reversed(update_dicts):
            for key, value in update_dict.items():
                if key not in merged:
                    merged[key] = value

    return merged
