"""
Module defining MatPES input set generators.

In case of questions, contact @janosh or @shyuep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pymatgen.io.vasp.sets import MatPESStaticSet

if TYPE_CHECKING:
    from typing import Literal


@dataclass
class MatPesGGAStaticSetGenerator(MatPESStaticSet):
    """Class to generate MP-compatible VASP GGA static input sets."""

    xc_functional: Literal["R2SCAN", "PBE", "PBE+U"] = "PBE"
    auto_ismear: bool = False
    auto_kspacing: bool = False


@dataclass
class MatPesMetaGGAStaticSetGenerator(MatPESStaticSet):
    """Class to generate MP-compatible VASP meta-GGA static input sets."""

    xc_functional: Literal["R2SCAN", "PBE", "PBE+U"] = "R2SCAN"
    auto_ismear: bool = False
    auto_kspacing: bool = False

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR for this calculation type.

        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        return {"GGA": None}  # unset GGA
