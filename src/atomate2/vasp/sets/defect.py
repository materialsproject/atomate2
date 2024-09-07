"""Module defining VASP input set generators for defect calculations."""

from __future__ import annotations

from dataclasses import dataclass, field

from pymatgen.io.vasp.inputs import Kpoints, KpointsSupportedModes

from atomate2.vasp.sets.base import VaspInputGenerator

SPECIAL_KPOINT = Kpoints(
    comment="special k-point",
    num_kpts=1,
    style=KpointsSupportedModes.Reciprocal,
    kpts=((0.25, 0.25, 0.25),),
    kpts_shift=(0, 0, 0),
    kpts_weights=[1],
)

SPECIAL_KPOINT_GAMMA = Kpoints(
    comment="special k-point",
    num_kpts=2,
    style=KpointsSupportedModes.Reciprocal,
    kpts=((0.25, 0.25, 0.25), (0.0, 0.0, 0.0)),
    kpts_shift=(0, 0, 0),
    kpts_weights=[1, 0],
)


@dataclass
class ChargeStateRelaxSetGenerator(VaspInputGenerator):
    """Generator for atomic-only relaxation for defect supercell calculations.

    Since the defect cells are assumed to be large, we will use only a single k-point.
    """

    use_structure_charge: bool = True
    user_kpoints_settings: dict | Kpoints = field(default_factory=SPECIAL_KPOINT)

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR for a relaxation job.

        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        return {
            "IBRION": 1,
            "ISIF": 2,
            "EDIFF": 1e-5,
            "EDIFFG": -0.05,
            "LREAL": False,
            "NSW": 99,
            "ENCUT": 500,
            "LAECHG": False,
            "NELMIN": 6,
            "LCHARG": True,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "LVHAR": True,
            "KSPACING": None,
            "LWAVE": True,
        }


@dataclass
class ChargeStateStaticSetGenerator(VaspInputGenerator):
    """Generator for static defect supercell calculations.

    Since the defect cells are assumed to be large, we will use only a single k-point.
    """

    use_structure_charge: bool = True
    user_kpoints_settings: dict | Kpoints = field(default_factory=SPECIAL_KPOINT)

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR for a relaxation job.

        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        return {
            "IBRION": 1,
            "EDIFF": 1e-5,
            "EDIFFG": -0.05,
            "LREAL": False,
            "NSW": 0,
            "ENCUT": 500,
            "LAECHG": False,
            "NELMIN": 6,
            "LCHARG": True,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "LVHAR": True,
            "KSPACING": None,
            "LWAVE": True,
        }


@dataclass
class HSEChargeStateRelaxSetGenerator(VaspInputGenerator):
    """Generator for atomic-only relaxation for defect supercell calculations.

    Since the defect cells are assumed to be large, we will use only a single k-point.
    """

    use_structure_charge: bool = True
    user_kpoints_settings: dict | Kpoints = field(default_factory=SPECIAL_KPOINT)

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR for a relaxation job.

        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        return {
            "ALGO": "Normal",
            "IBRION": 1,
            "LAECHG": False,
            "ISIF": 2,
            "EDIFF": 1e-5,
            "EDIFFG": -0.05,
            "LREAL": False,
            "NSW": 99,
            "ENCUT": 500,
            "NELMIN": 6,
            "GGA": "Pe",
            "LCHARG": False,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "LVHAR": True,
            "KSPACING": None,
            "LDAU": False,
            "HFSCREEN": 0.2,
            "LHFCALC": True,
            "PRECFOCK": "Fast",
            "LASPH": True,
            "LWAVE": True,
        }


@dataclass
class HSEChargeStateStaticSetGenerator(VaspInputGenerator):
    """Generator for HSE static defect supercell calculations.

    Since the defect cells are assumed to be large, we will use only a single k-point.
    """

    use_structure_charge: bool = True
    user_kpoints_settings: dict | Kpoints = field(default_factory=SPECIAL_KPOINT)

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR for a relaxation job.

        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        return {
            "ALGO": "All",
            "IBRION": 1,
            "LAECHG": False,
            "EDIFF": 1e-5,
            "EDIFFG": -0.05,
            "LREAL": False,
            "NSW": 0,
            "ENCUT": 500,
            "NELMIN": 6,
            "LCHARG": False,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "LVHAR": True,
            "KSPACING": None,
            "LDAU": False,
            "HFSCREEN": 0.2,
            "LHFCALC": True,
            "PRECFOCK": "Fast",
            "LASPH": True,
            "LWAVE": True,
        }
