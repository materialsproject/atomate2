"""
Module defining Materials Project input set generators.

Reference: https://doi.org/10.1103/PhysRevMaterials.6.013801

In case of questions, consult @Andrew-S-Rosen, @esoteric-ephemera or @janosh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pymatgen.io.vasp import Kpoints
from pymatgen.io.vasp.sets import MPRelaxSet, MPScanRelaxSet

from atomate2.vasp.sets.core import RelaxSetGenerator, StaticSetGenerator

@dataclass
class MPGGARelaxSetGenerator(RelaxSetGenerator):
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

    config_dict: dict = field(default_factory=lambda: MPRelaxSet.CONFIG)
    auto_ismear: bool = False
    auto_kspacing: bool = False
    inherit_incar: bool | None = False
    bandgap_tol: float = None

@dataclass
class MPGGAStaticSetGenerator(StaticSetGenerator):
    """Class to generate MP-compatible VASP GGA static input sets."""

    config_dict: dict = field(default_factory=lambda: MPRelaxSet.CONFIG)
    auto_ismear: bool = False
    auto_kspacing: bool = False
    bandgap_tol: float = None
    inherit_incar: bool | None = False
    reciprocal_density: int = 100
    small_gap_multiply: tuple[float, float] | None = None

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR for this calculation type.

        Parameters
        ----------
        structure
            A structure.
        prev_incar
            An incar from a previous calculation.
        bandgap
            The band gap.
        vasprun
            A vasprun from a previous calculation.
        outcar
            An outcar from a previous calculation.

        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        return {
            "ALGO": "FAST",
            "NSW": 0,
            "LCHARG": True,
            "LWAVE": False,
            "LREAL": False,
            "ISMEAR": -5,
        }

    @property
    def kpoints_updates(self) -> dict | Kpoints:
        """Updates to the kpoints configuration for this calculation type.
        
        This function is adapted from pymatgen.io.vasp.sets.MPStaticSet.
        Thanks to @Andrew-S-Rosen for finding this discrepancy in issue 844:
        https://github.com/materialsproject/atomate2/issues/844
        """
        factor = 1.0
        if self.bandgap is not None and self.small_gap_multiply and self.bandgap <= self.small_gap_multiply[0]:
            factor = self.small_gap_multiply[1]

        # prefer to use k-point scheme from previous run
        if self.prev_kpoints and self.prev_kpoints.style == Kpoints.supported_modes.Monkhorst:  # type: ignore
            kpoints = Kpoints.automatic_density_by_vol(
                self.structure,  # type: ignore
                int(self.reciprocal_density * factor),
                self.force_gamma,
            )
            k_div = [kp + 1 if kp % 2 == 1 else kp for kp in kpoints.kpts[0]]  # type: ignore
            return Kpoints.monkhorst_automatic(k_div)  # type: ignore

        return {"reciprocal_density": self.reciprocal_density * factor}


@dataclass
class MPMetaGGAStaticSetGenerator(StaticSetGenerator):
    """Class to generate MP-compatible VASP GGA static input sets."""

    config_dict: dict = field(default_factory=lambda: MPScanRelaxSet.CONFIG)
    auto_ismear: bool = False
    auto_kspacing: bool = True
    bandgap_tol: float = 1e-4
    inherit_incar: bool | None = False

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR for this calculation type.

                Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        return {
            "ALGO": "FAST",
            "GGA": None,  # unset GGA, shouldn't be set anyway but best be sure
            "NSW": 0,
            "LCHARG": True,
            "LWAVE": False,
            "LREAL": False,
            "ISMEAR": -5,
        }


@dataclass
class MPMetaGGARelaxSetGenerator(RelaxSetGenerator):
    """Class to generate MP-compatible VASP metaGGA relaxation input sets.

    Parameters
    ----------
    config_dict: dict
        The config dict.
    bandgap_tol: float
        Tolerance for metallic bandgap. If bandgap < bandgap_tol, KSPACING will be 0.22,
        otherwise it will increase with bandgap up to a max of 0.44.
    """

    config_dict: dict = field(default_factory=lambda: MPScanRelaxSet.CONFIG)
    bandgap_tol: float = 1e-4
    auto_ismear: bool = False
    auto_kspacing: bool = True
    inherit_incar: bool | None = False

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR for this calculation type.

        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        # unset GGA, shouldn't be set anyway but doesn't hurt to be sure
        return {"LCHARG": True, "LWAVE": True, "GGA": None}
