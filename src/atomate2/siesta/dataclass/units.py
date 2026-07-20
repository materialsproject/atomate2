"""
Unit parsing utilities for FDF parameters.

This module provides utilities to parse SIESTA FDF parameter values
with units (e.g., "450 Ry", "2.5 Ang") and convert them to target units.
"""

from __future__ import annotations

# Metadata

__all__ = ["parse_energy", "parse_force", "parse_length"]

import re


def parse_energy(value: str | float, target_unit: str = "Ry") -> float:
    """
    Parse energy value with units and convert to target unit.

    Parameters
    ----------
    value : str or float
        Energy value, either as float or string with unit (e.g., "450 Ry", "10 eV")
    target_unit : str, optional
        Target energy unit: "Ry" (Rydberg) or "eV" (default: "Ry")

    Returns
    -------
    float
        Energy value in target unit

    Raises
    ------
    ValueError
        If value format is invalid or unit is not recognized

    Examples
    --------
    >>> parse_energy("450 Ry", target_unit="Ry")
    450.0
    >>> parse_energy("13.605693123 eV", target_unit="Ry")
    1.0
    >>> parse_energy(450, target_unit="Ry")
    450.0
    >>> parse_energy("1 Hartree", target_unit="Ry")
    2.0
    """
    # If already a number, assume it's in target unit
    if isinstance(value, (int, float)):
        return float(value)

    # Parse string: extract number and unit
    match = re.match(r"^\s*([\d.eE+-]+)\s*([a-zA-Z]*)\s*$", str(value))
    if not match:
        raise ValueError(
            f"Cannot parse energy value: '{value}'. "
            f"Expected format: '<number> <unit>' (e.g., '450 Ry', '10 eV')"
        )

    number_str, unit = match.groups()
    number = float(number_str)

    # If no unit specified, assume target unit
    if not unit:
        return number

    # Conversion factors to Rydberg
    unit_lower = unit.lower()
    ry_per_unit = {
        "ry": 1.0,
        "rydberg": 1.0,
        "ev": 1.0 / 13.605693123,  # 1 Ry = 13.6057 eV
        "hartree": 2.0,  # 1 Ry = 0.5 Hartree
        "ha": 2.0,
        "mry": 0.001,  # milliRydberg
        "mev": 0.001 / 13.605693123,  # millielectronvolt
    }

    if unit_lower not in ry_per_unit:
        raise ValueError(
            f"Unknown energy unit: '{unit}'. "
            f"Supported units: {list(ry_per_unit.keys())}"
        )

    # Convert to Rydberg first
    value_in_ry = number * ry_per_unit[unit_lower]

    # Convert to target unit
    target_lower = target_unit.lower()
    if target_lower not in ry_per_unit:
        raise ValueError(
            f"Unknown target unit: '{target_unit}'. "
            f"Supported units: {list(ry_per_unit.keys())}"
        )

    return value_in_ry / ry_per_unit[target_lower]


def parse_length(value: str | float, target_unit: str = "Ang") -> float:
    """
    Parse length value with units and convert to target unit.

    Parameters
    ----------
    value : str or float
        Length value, either as float or string with unit (e.g., "2.5 Ang", "1.0 Bohr")
    target_unit : str, optional
        Target length unit: "Ang" (Angstrom) or "Bohr" (default: "Ang")

    Returns
    -------
    float
        Length value in target unit

    Raises
    ------
    ValueError
        If value format is invalid or unit is not recognized

    Examples
    --------
    >>> parse_length("2.5 Ang", target_unit="Ang")
    2.5
    >>> parse_length("1.0 Bohr", target_unit="Ang")
    0.529177210903
    >>> parse_length(2.5, target_unit="Ang")
    2.5
    """
    # If already a number, assume it's in target unit
    if isinstance(value, (int, float)):
        return float(value)

    # Parse string: extract number and unit
    match = re.match(r"^\s*([\d.eE+-]+)\s*([a-zA-Z]*)\s*$", str(value))
    if not match:
        raise ValueError(
            f"Cannot parse length value: '{value}'. "
            f"Expected format: '<number> <unit>' (e.g., '2.5 Ang', '1.0 Bohr')"
        )

    number_str, unit = match.groups()
    number = float(number_str)

    # If no unit specified, assume target unit
    if not unit:
        return number

    # Conversion factors to Angstrom
    ang_per_unit = {
        "ang": 1.0,
        "angstrom": 1.0,
        "bohr": 0.529177210903,  # 1 Bohr = 0.529177 Ang
        "nm": 10.0,  # 1 nm = 10 Ang
        "pm": 0.01,  # 1 pm = 0.01 Ang
    }

    unit_lower = unit.lower()
    if unit_lower not in ang_per_unit:
        raise ValueError(
            f"Unknown length unit: '{unit}'. "
            f"Supported units: {list(ang_per_unit.keys())}"
        )

    # Convert to Angstrom first
    value_in_ang = number * ang_per_unit[unit_lower]

    # Convert to target unit
    target_lower = target_unit.lower()
    if target_lower not in ang_per_unit:
        raise ValueError(
            f"Unknown target unit: '{target_unit}'. "
            f"Supported units: {list(ang_per_unit.keys())}"
        )

    return value_in_ang / ang_per_unit[target_lower]


def parse_force(value: str | float, target_unit: str = "eV/Ang") -> float:
    """
    Parse force value with units and convert to target unit.

    Parameters
    ----------
    value : str or float
        Force value, either as float or string with unit (e.g., "0.04 eV/Ang", "0.01 Ry/Bohr")
    target_unit : str, optional
        Target force unit: "eV/Ang", "Ry/Bohr", etc. (default: "eV/Ang")

    Returns
    -------
    float
        Force value in target unit

    Raises
    ------
    ValueError
        If value format is invalid or unit is not recognized

    Examples
    --------
    >>> parse_force("0.04 eV/Ang", target_unit="eV/Ang")
    0.04
    >>> parse_force(0.04, target_unit="eV/Ang")
    0.04
    """
    # If already a number, assume it's in target unit
    if isinstance(value, (int, float)):
        return float(value)

    # Parse string: extract number and unit (handle compound units like eV/Ang)
    match = re.match(r"^\s*([\d.eE+-]+)\s*([a-zA-Z/]*)\s*$", str(value))
    if not match:
        raise ValueError(
            f"Cannot parse force value: '{value}'. "
            f"Expected format: '<number> <unit>' (e.g., '0.04 eV/Ang', '0.01 Ry/Bohr')"
        )

    number_str, unit = match.groups()
    number = float(number_str)

    # If no unit specified, assume target unit
    if not unit:
        return number

    # Normalize unit strings (case-insensitive)
    unit_normalized = unit.lower().replace("angstrom", "ang")
    target_normalized = target_unit.lower().replace("angstrom", "ang")

    # For simplicity, we only support direct unit matching
    # More complex conversions would require energy and length conversion
    if unit_normalized == target_normalized:
        return number
    # Could add conversion factors here if needed
    # For now, just return the value if units match commonly used variants
    unit_variants = {
        "ev/ang": ["ev/ang", "ev/angstrom"],
        "ry/bohr": ["ry/bohr"],
    }

    for canonical, variants in unit_variants.items():
        if unit_normalized in variants and target_normalized in variants:
            return number

    raise ValueError(
        f"Force unit conversion from '{unit}' to '{target_unit}' not yet supported. "
        f"Please use matching units (e.g., both 'eV/Ang' or both 'Ry/Bohr')"
    )
