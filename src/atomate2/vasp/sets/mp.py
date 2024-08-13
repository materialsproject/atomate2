"""
Module defining Materials Project input set generators.

Reference: https://doi.org/10.1103/PhysRevMaterials.6.013801

In case of questions, consult @Andrew-S-Rosen, @esoteric-ephemera or @janosh.
"""

from __future__ import annotations

from dataclasses import dataclass

from monty.dev import deprecated
from pymatgen.io.vasp.sets import (
    MPRelaxSet,
    MPScanRelaxSet,
    MPScanStaticSet,
    MPStaticSet,
)


@deprecated(replacement=MPRelaxSet, deadline=(2025, 1, 1))
@dataclass
class MPGGARelaxSetGenerator(MPRelaxSet):
    """Class to generate MP-compatible VASP GGA relaxation input sets.

    reciprocal_density (int): For static calculations, we usually set the
        reciprocal density by volume. This is a convenience arg to change
        that, rather than using user_kpoints_settings. Defaults to 100,
        which is ~50% more than that of standard relaxation calculations.
    small_gap_multiply ([float, float]): If the gap is less than
        1st index, multiply the default reciprocal_density by the 2nd
        index.
    **kwargs: kwargs supported by RelaxSetGenerator.
    """

    auto_ismear: bool = False
    auto_kspacing: bool = False
    inherit_incar: bool | None = False
    bandgap_tol: float = None
    force_gamma: bool = True
    auto_metal_kpoints: bool = True
    symprec: float | None = None

    def __post_init__(self) -> None:
        """Raise deprecation warning and validate."""
        if self.symprec is not None:
            self.sym_prec = self.symprec
        super().__post_init__()


@deprecated(replacement=MPStaticSet, deadline=(2025, 1, 1))
@dataclass
class MPGGAStaticSetGenerator(MPStaticSet):
    """Class to generate MP-compatible VASP GGA static input sets."""

    auto_ismear: bool = False
    auto_kspacing: bool = False
    bandgap_tol: float = None
    inherit_incar: bool | None = False
    force_gamma: bool = True
    auto_metal_kpoints: bool = True
    symprec: float | None = None

    def __post_init__(self) -> None:
        """Raise deprecation warning and validate."""
        if self.symprec is not None:
            self.sym_prec = self.symprec
        super().__post_init__()


@deprecated(replacement=MPScanStaticSet, deadline=(2025, 1, 1))
@dataclass
class MPMetaGGAStaticSetGenerator(MPScanStaticSet):
    """Class to generate MP-compatible VASP GGA static input sets."""

    auto_ismear: bool = False
    auto_kspacing: bool = True
    bandgap_tol: float = 1e-4
    inherit_incar: bool | None = False
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
        return super().incar_updates | {
            "ALGO": "FAST",
            "GGA": None,  # unset GGA, shouldn't be set anyway but best be sure
            "LCHARG": True,
            "LWAVE": False,
            "LVHAR": False,  # this is not needed
            "LELF": False,  # prevents KPAR > 1
        }


@deprecated(replacement=MPScanRelaxSet, deadline=(2025, 1, 1))
@dataclass
class MPMetaGGARelaxSetGenerator(MPScanRelaxSet):
    """Class to generate MP-compatible VASP metaGGA relaxation input sets.

    Parameters
    ----------
    config_dict: dict
        The config dict.
    bandgap_tol: float
        Tolerance for metallic bandgap. If bandgap < bandgap_tol, KSPACING will be 0.22,
        otherwise it will increase with bandgap up to a max of 0.44.
    """

    bandgap_tol: float = 1e-4
    auto_ismear: bool = False
    auto_kspacing: bool = True
    inherit_incar: bool | None = False
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
        # unset GGA, shouldn't be set anyway but doesn't hurt to be sure
        return {"LCHARG": True, "LWAVE": True, "GGA": None, "LELF": False}
