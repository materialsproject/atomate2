"""
Module defining equation of state (EOS) parameter sets.

Three families of sets: default atomate2 params,
MP GGA compatible, and MP meta-GGA compatible
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files as import_files
from typing import TYPE_CHECKING

from monty.serialization import loadfn

from atomate2.vasp.sets.base import VaspInputGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Outcar, Vasprun


_BASE_MP_GGA_RELAX_SET = loadfn(
    import_files("atomate2.vasp.sets") / "BaseMPGGASet.yaml"
)
_BASE_MP_R2SCAN_RELAX_SET = loadfn(
    import_files("atomate2.vasp.sets") / "BaseMPR2SCANRelaxSet.yaml"
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
        """Get updates to the INCAR for a relaxation job.

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
            "NSW": 99,
            "LCHARG": False,
            "ISIF": 3,
            "IBRION": 2,
            "EDIFF": 1e-6,
            "ENCUT": 680,
            "ENAUG": 1360,
            "LREAL": False,
            "LWAVE": True,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "KSPACING": 0.22,
        }


# MPLegacy prefix = MP PBE-GGA compatible with atomate implementation
@dataclass
class MPLegacyEosRelaxSetGenerator(VaspInputGenerator):
    """Class to generate atomate1-MP-compatible VASP GGA EOS relax input sets."""

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
        """Get updates to the INCAR for a relaxation job.

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
            "NSW": 99,
            "LCHARG": False,
            "ISIF": 3,
            "IBRION": 2,
            "EDIFF": 1e-6,
            "ENCUT": 600,
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
        """Get updates to the kpoints configuration for a non-self consistent VASP job.

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


@dataclass
class MPLegacyEosStaticSetGenerator(EosSetGenerator):
    """Class to generate atomate1-MP-compatible VASP GGA EOS relax input sets."""

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
        """Get updates to the INCAR for a relaxation job.

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
        # Original atomate wf_bulk_modulus had 600 eV cutoff
        # and k-point grid_density = 7,000
        return {
            "NSW": 0,
            "LCHARG": False,
            "EDIFF": 1e-6,
            "ENCUT": 600,
            "LREAL": False,
            "LWAVE": False,
            "IBRION": -1,
            "ISMEAR": -5,
            "LORBIT": 11,
            "ALGO": "Normal",
        }


# MPGGA prefix = MP GGA compatible
@dataclass
class MPGGAEosRelaxSetGenerator(VaspInputGenerator):
    """Class to generate MP-compatible VASP GGA EOS relax input sets."""

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
        """Get updates to the INCAR for a relaxation job.

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
            "NSW": 99,
            "LCHARG": False,
            "ISIF": 3,
            "IBRION": 2,
            "EDIFF": 1e-6,
            "ALGO": "FAST",
            "LREAL": False,
            "LWAVE": True,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "LMAXMIX": 6,
            "KSPACING": 0.22,
            "METAGGA": None,
        }


@dataclass
class MPGGAEosStaticSetGenerator(EosSetGenerator):
    """Class to generate MP-compatible VASP GGA EOS relax input sets."""

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
        """Get updates to the INCAR for a relaxation job.

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
            "LCHARG": False,
            "IBRION": -1,
            "EDIFF": 1e-6,
            "ALGO": "NORMAL",
            "LREAL": False,
            "LWAVE": False,
            "ISMEAR": -5,
            "LMAXMIX": 6,
            "KSPACING": 0.22,
            "METAGGA": None,
        }


# MPMetaGGA prefix = MP r2SCAN meta-GGA compatible
@dataclass
class MPMetaGGAEosStaticSetGenerator(VaspInputGenerator):
    """Class to generate MP-compatible VASP Meta-GGA static input sets."""

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
            "EDIFF": 1e-6,
            "ALGO": "NORMAL",
            "GGA": None,  # unset GGA, shouldn't be set anyway but best be sure
            "NSW": 0,
            "LCHARG": True,
            "LWAVE": False,
            "LREAL": False,
            "ISMEAR": -5,
            "IBRION": -1,
            "LMAXMIX": 6,
            "KSPACING": 0.22,
        }


@dataclass
class MPMetaGGAEosRelaxSetGenerator(VaspInputGenerator):
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
        # unset GGA, shouldn't be set anyway but doesn't hurt to be sure
        return {
            "LCHARG": True,
            "LWAVE": True,
            "GGA": None,
            "LMAXMIX": 6,
            "KSPACING": 0.22,
            "NSW": 99,
            "ISIF": 3,
            "IBRION": 2,
            "EDIFF": 1e-6,
            "LREAL": False,
            "ISMEAR": 0,
            "SIGMA": 0.05,
        }


@dataclass
class MPMetaGGAEosPreRelaxSetGenerator(VaspInputGenerator):
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
        # unset METAGGA, shouldn't be set anyway but doesn't hurt to be sure
        return {
            "LCHARG": True,
            "LWAVE": True,
            "GGA": "PS",
            "METAGGA": None,
            "NSW": 99,
            "ISIF": 3,
            "IBRION": 2,
            "EDIFF": 1e-6,
            "EDIFFG": -0.05,
            "LREAL": False,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "KSPACING": 0.22,
        }
