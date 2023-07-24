"""
Module defining jobs for Materials Project r2SCAN workflows.

Reference: https://doi.org/10.1103/PhysRevMaterials.6.013801
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from monty.serialization import loadfn
from pkg_resources import resource_filename

from atomate2.vasp.sets.base import VaspInputGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Outcar, Vasprun


_BASE_MP_R2SCAN_RELAX_SET = loadfn(
    resource_filename("atomate2.vasp.sets", "BaseMPR2SCANRelaxSet.yaml")
)


_BASE_MP_GGA_RELAX_SET = loadfn(
    resource_filename("atomate2.vasp.sets", "BaseMPGGASet.yaml")
)


@dataclass
class MPGGARelaxGenerator(VaspInputGenerator):
    """Class to generate MP-compatible VASP GGA relaxation input sets."""

    config_dict: dict = field(default_factory=lambda: _BASE_MP_GGA_RELAX_SET)

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = 0,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """
        Get updates to the INCAR for a relaxation job.

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
        return {}


@dataclass
class MPMetaGGARelaxGenerator(VaspInputGenerator):
    """Class to generate MP-compatible VASP metaGGA relaxation input sets."""

    config_dict: dict = field(default_factory=lambda: _BASE_MP_R2SCAN_RELAX_SET)
    # Tolerance for metallic bandgap. If bandgap < bandgap_tol, KSPACING will be 0.22,
    # otherwise it will increase with bandgap up to a max of 0.44.
    bandgap_tol: float = 1e-4
    bandgap_override: float | None = None

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = 0,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """
        Get updates to the INCAR for a relaxation job.

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
        bandgap = self.bandgap_override or bandgap or 0

        if bandgap < self.bandgap_tol:  # metallic
            return {"KSPACING": 0.22, "ISMEAR": 2, "SIGMA": 0.2}

        rmin = 25.22 - 2.87 * bandgap
        kspacing = 2 * np.pi * 1.0265 / (rmin - 1.0183)
        return {"KSPACING": np.clip(kspacing, 0.22, 0.44), "ISMEAR": -5, "SIGMA": 0.05}
