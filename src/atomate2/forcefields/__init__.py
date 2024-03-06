"""Tools and functions common to all forcefields."""

from enum import Enum


class MLFF(Enum):  # TODO inherit from StrEnum when 3.11+
    """Names of ML force fields."""

    MACE = "MACE"
    GAP = "GAP"
    M3GNet = "M3GNet"
    CHGNet = "CHGNet"
