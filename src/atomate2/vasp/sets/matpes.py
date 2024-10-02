"""
Module defining MatPES input set generators.

In case of questions, contact @janosh or @shyuep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from monty.dev import deprecated
from pymatgen.io.vasp.sets import MatPESStaticSet

if TYPE_CHECKING:
    from typing import Literal


@deprecated(replacement=MatPESStaticSet, deadline=(2025, 1, 1))
@dataclass
class MatPesGGAStaticSetGenerator(MatPESStaticSet):
    """Class to generate MP-compatible VASP GGA static input sets."""

    xc_functional: Literal["R2SCAN", "PBE", "PBE+U"] = "PBE"
    auto_ismear: bool = False
    auto_kspacing: bool = False
    symprec: float | None = None

    def __post_init__(self) -> None:
        """Raise deprecation warning and validate."""
        if self.symprec is not None:
            self.sym_prec = self.symprec
        super().__post_init__()


@deprecated(
    replacement=MatPESStaticSet,
    deadline=(2025, 1, 1),
    message="Be sure to use `xc_functional = 'R2SCAN'` when instantiating the class.",
)
@dataclass
class MatPesMetaGGAStaticSetGenerator(MatPESStaticSet):
    """Class to generate MP-compatible VASP meta-GGA static input sets."""

    xc_functional: Literal["R2SCAN", "PBE", "PBE+U"] = "R2SCAN"
    auto_ismear: bool = False
    auto_kspacing: bool = False
    symprec: float | None = None

    def __post_init__(self) -> None:
        """Raise deprecation warning and validate."""
        if self.symprec is not None:
            self.sym_prec = self.symprec
        super().__post_init__()

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR for this calculation type.

        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        return {"GGA": None}  # unset GGA
