"""Tools and functions common to all forcefields."""

from __future__ import annotations

from enum import Enum


class MLFF(Enum):  # TODO inherit from StrEnum when 3.11+
    """Names of ML force fields."""

    MACE = "MACE"
    GAP = "GAP"
    M3GNet = "M3GNet"
    CHGNet = "CHGNet"
    Forcefield = "Forcefield"  # default placeholder option
    NEP = "NEP"
    Nequip = "Nequip"
    SevenNet = "SevenNet"


def _get_formatted_ff_name(force_field_name: str | MLFF) -> str:
    """
    Get the standardized force field name.

    Parameters
    ----------
    force_field_name : str or .MLFF
        The name of the force field

    Returns
    -------
    str : the name of the forcefield from MLFF
    """
    if isinstance(force_field_name, str) and force_field_name in MLFF.__members__:
        # ensure `force_field_name` uses enum format
        force_field_name = MLFF(force_field_name)
    return str(force_field_name)
