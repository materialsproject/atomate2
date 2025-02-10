"""Tools and functions common to all forcefields."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from typing import Any


class MLFF(Enum):  # TODO inherit from StrEnum when 3.11+
    """Names of ML force fields."""

    MACE = "MACE"  # This is MACE-MP-0 (medium), deprecated
    MACE_MP_0 = "MACE-MP-0"
    MACE_MPA_0 = "MACE-MPA-0"
    MACE_MP_0B3 = "MACE-MP-0b3"
    GAP = "GAP"
    M3GNet = "M3GNet"
    CHGNet = "CHGNet"
    Forcefield = "Forcefield"  # default placeholder option
    NEP = "NEP"
    Nequip = "Nequip"
    SevenNet = "SevenNet"

    @classmethod
    def _missing_(cls, value: Any) -> Any:
        """Allow input of str(MLFF) as valid enum."""
        if isinstance(value, str):
            value = value.split("MLFF.")[-1]
        for member in cls:
            if member.name == value:
                return member
        return None


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
    if isinstance(force_field_name, str):
        # ensure `force_field_name` uses enum format
        if force_field_name in MLFF.__members__:
            force_field_name = MLFF[force_field_name]
        elif force_field_name in [v.value for v in MLFF]:
            force_field_name = MLFF(force_field_name)
    force_field_name = str(force_field_name)
    if force_field_name in {"MLFF.MACE","MACE"}:
        warnings.warn(
            "Because the default MP-trained MACE model is constantly evolving, "
            "we no longer recommend using `MACE` or `MLFF.MACE` to specify "
            "a MACE model. For reproducibility purposes, specifying `MACE` "
            "will still default to MACE-MP-0 (medium), which is identical to "
            "specifying `MLFF.MACE_MP_0`.",
            category=UserWarning,
        )
    return force_field_name