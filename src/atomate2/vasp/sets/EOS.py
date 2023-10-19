"""
Module defining equation of state (EOS) parameter sets.

Three families of sets: default atomate2 params,
MP GGA compatible, and MP meta-GGA compatible
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from monty.serialization import loadfn
from pkg_resources import resource_filename

from atomate2.vasp.sets.base import VaspInputGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Outcar, Vasprun


_BASE_MP_GGA_RELAX_SET = loadfn(
    resource_filename("atomate2.vasp.sets", "BaseMPGGASet.yaml")
)
_BASE_MP_R2SCAN_RELAX_SET = loadfn(
    resource_filename("atomate2.vasp.sets", "BaseMPR2SCANRelaxSet.yaml")
)


@dataclass
class EosSetGenerator(VaspInputGenerator):
    """Class to generate VASP EOS deformation + relax input sets."""

    force_gamma: bool = True
    auto_ismear: bool = False
    auto_kspacing: bool = False
    inherit_incar: bool = False

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
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
        """
        original atomate wf_bulk_modulus had 600 eV cutoff
        Because default atomate2 set and MP's r2SCAN set use 680 eV cutoff,
        avoid needless duplication and use ENCUT = 680 eV here
        """
        return {
            "NSW": 99,
            "LCHARG": False,
            "ISIF": 2,
            "IBRION": 2,
            "EDIFF": 1.0e-6,
            "ENCUT": 680,
            "ENAUG": 1360,
            "LREAL": False,
            "LWAVE": True,
        }

    def get_kpoints_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = 0.0,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """
        Get updates to the kpoints configuration for a non-self consistent VASP job.

        Note, these updates will be ignored if the user has set user_kpoint_settings.

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
            A dictionary of updates to apply to the KPOINTS config.
        """
        return {"grid_density": 7000}


"""
MPGGA prefix = MP PBE-GGA compatible
"""


@dataclass
class MPGGAEosRelaxSetGenerator(EosSetGenerator):
    """Class to generate MP-compatible VASP GGA EOS relax input sets."""

    config_dict: dict = field(default_factory=lambda: _BASE_MP_GGA_RELAX_SET)
    auto_ismear: bool = False
    auto_kspacing: bool = False
    inherit_incar: bool = False


@dataclass
class MPGGAEosStaticSetGenerator(EosSetGenerator):
    """Class to generate MP-compatible VASP GGA EOS relax input sets."""

    config_dict: dict = field(default_factory=lambda: _BASE_MP_GGA_RELAX_SET)
    auto_ismear: bool = False
    auto_kspacing: bool = False
    inherit_incar: bool = False

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """
        Get updates to the INCAR for this calculation type.

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
            "NSW": 0,
            "IBRION": -1,
            "LCHARG": True,
            "LWAVE": False,
            "LREAL": False,
            "ISMEAR": -5,
        }


"""
MPGGA prefix = MP PBE-GGA compatible
"""


@dataclass
class MPMetaGGAEosStaticSetGenerator(EosSetGenerator):
    """Class to generate MP-compatible VASP GGA static input sets."""

    config_dict: dict = field(default_factory=lambda: _BASE_MP_R2SCAN_RELAX_SET)
    auto_ismear: bool = False
    auto_kspacing: bool = False
    inherit_incar: bool = False

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """
        Get updates to the INCAR for this calculation type.

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
            "GGA": None,  # unset GGA, shouldn't be set anyway but best be sure
            "NSW": 0,
            "LCHARG": True,
            "LWAVE": False,
            "LREAL": False,
            "ISMEAR": -5,
            "IBRION": -1,
        }


@dataclass
class MPMetaGGAEosRelaxSetGenerator(EosSetGenerator):
    """Class to generate MP-compatible VASP meta-GGA relaxation input sets.

    Parameters
    ----------
    config_dict: dict
        The config dict.
    bandgap_tol: float
        Tolerance for metallic bandgap. If bandgap < bandgap_tol, KSPACING will be 0.22,
        otherwise it will increase with bandgap up to a max of 0.44.
    """

    config_dict: dict = field(default_factory=lambda: _BASE_MP_R2SCAN_RELAX_SET)
    bandgap_tol: float = 1e-4
    auto_ismear: bool = False
    auto_kspacing: bool = False
    inherit_incar: bool = False

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """
        Get updates to the INCAR for this calculation type.

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
        # unset GGA, shouldn't be set anyway but doesn't hurt to be sure
        return {"LCHARG": True, "LWAVE": True, "GGA": None}


@dataclass
class MPMetaGGAEosPreRelaxSetGenerator(EosSetGenerator):
    """Class to generate MP-compatible VASP meta-GGA pre-relaxation input sets.

    Parameters
    ----------
    config_dict: dict
        The config dict.
    bandgap_tol: float
        Tolerance for metallic bandgap. If bandgap < bandgap_tol, KSPACING will be 0.22,
        otherwise it will increase with bandgap up to a max of 0.44.
    """

    config_dict: dict = field(default_factory=lambda: _BASE_MP_R2SCAN_RELAX_SET)
    bandgap_tol: float = 1e-4
    auto_ismear: bool = False
    auto_kspacing: bool = False
    inherit_incar: bool = False

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """
        Get updates to the INCAR for this calculation type.

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
        # unset METAGGA, shouldn't be set anyway but doesn't hurt to be sure
        return {"LCHARG": True, "LWAVE": True, "GGA": "PS", "METAGGA": None}
