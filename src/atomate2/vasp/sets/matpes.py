"""
Module defining MatPES input set generators.

In case of questions, contact @janosh or @shyuep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files as get_mod_path
from typing import TYPE_CHECKING

from monty.serialization import loadfn

from atomate2.vasp.sets.base import VaspInputGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Outcar, Vasprun


# POTCAR section comes from PARENT but atomate2 does not support inheritance yet
_BASE_MATPES_PBE_STATIC_SET_NO_POTCAR = loadfn(
    get_mod_path("pymatgen.io.vasp") / "MatPESStaticSet.yaml"
)
_POTCAR_BASE_FILE = f"{_BASE_MATPES_PBE_STATIC_SET_NO_POTCAR['PARENT']}.yaml"
_POTCAR_SET = loadfn(get_mod_path("pymatgen.io.vasp") / _POTCAR_BASE_FILE)
_BASE_MATPES_PBE_STATIC_SET = {**_POTCAR_SET, **_BASE_MATPES_PBE_STATIC_SET_NO_POTCAR}


@dataclass
class MatPesGGAStaticSetGenerator(VaspInputGenerator):
    """Class to generate MP-compatible VASP GGA static input sets."""

    config_dict: dict = field(default_factory=lambda: _BASE_MATPES_PBE_STATIC_SET)
    auto_ismear: bool = False
    auto_kspacing: bool = False

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
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
        return {}


@dataclass
class MatPesMetaGGAStaticSetGenerator(MatPesGGAStaticSetGenerator):
    """Class to generate MP-compatible VASP meta-GGA static input sets."""

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
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
        return {"METAGGA": "R2SCAN", "ALGO": "ALL", "GGA": None}  # unset GGA
