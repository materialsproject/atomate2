"""
Module defining MatPES input set generators.

In case of questions, contact @janosh or @shyuep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pymatgen.io.vasp.sets import MatPESStaticSet

from atomate2.vasp.sets.base import VaspInputGenerator

@dataclass
class MatPesGGAStaticSetGenerator(VaspInputGenerator):
    """Class to generate MP-compatible VASP GGA static input sets."""

    config_dict: dict = field(default_factory=lambda: MatPESStaticSet.CONFIG)
    auto_ismear: bool = False
    auto_kspacing: bool = False

@dataclass
class MatPesMetaGGAStaticSetGenerator(MatPesGGAStaticSetGenerator):
    """Class to generate MP-compatible VASP meta-GGA static input sets."""

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR for this calculation type.

        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        return {"METAGGA": "R2SCAN", "ALGO": "ALL", "GGA": None}  # unset GGA
