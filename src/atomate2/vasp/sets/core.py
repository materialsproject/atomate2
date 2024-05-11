"""Module defining core VASP input set generators."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from pymatgen.core.periodic_table import Element

from atomate2.vasp.sets.base import VaspInputGenerator

if TYPE_CHECKING:
    from emmet.core.math import Vector3D
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Outcar, Vasprun


logger = logging.getLogger(__name__)


@dataclass
class RelaxSetGenerator(VaspInputGenerator):
    """Class to generate VASP relaxation input sets."""

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
        return {"NSW": 99, "LCHARG": False, "ISIF": 3, "IBRION": 2}


@dataclass
class TightRelaxSetGenerator(VaspInputGenerator):
    """Class to generate tight VASP relaxation input sets."""

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """Get updates to the INCAR for a tight relaxation job.

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
            "IBRION": 2,
            "ISIF": 3,
            "ENCUT": 700,
            "EDIFF": 1e-7,
            "LAECHG": False,
            "EDIFFG": -0.001,
            "LREAL": False,
            "NSW": 99,
            "LCHARG": False,
        }


@dataclass
class StaticSetGenerator(VaspInputGenerator):
    """
    Class to generate VASP static input sets.

    Parameters
    ----------
    lepsilon
        Whether to set LEPSILON (used for calculating the high-frequency dielectric
        tensor).
    lcalcpol
        Whether to set LCALCPOL (used for calculating the electronic contribution to
        the polarization)
    **kwargs
        Other keyword arguments that will be passed to :obj:`VaspInputGenerator`.
    """

    lepsilon: bool = False
    lcalcpol: bool = False

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """Get updates to the INCAR for a static VASP job.

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
        updates = {"NSW": 0, "ISMEAR": -5, "LCHARG": True, "LORBIT": 11, "LREAL": False}
        if self.lepsilon:
            # LPEAD=T: numerical evaluation of overlap integral prevents LRF_COMMUTATOR
            # errors and can lead to better expt. agreement but produces slightly
            # different results
            updates.update({"IBRION": 8, "LEPSILON": True, "LPEAD": True, "NSW": 1})

        if self.lcalcpol:
            updates["LCALCPOL"] = True
        return updates


@dataclass
class NonSCFSetGenerator(VaspInputGenerator):
    """
    Class to generate VASP non-self-consistent field input sets.

    Parameters
    ----------
    mode
        Type of band structure mode. Options are "line", "uniform", or "boltztrap".
    dedos
        Energy difference used to set NEDOS, based on the total energy range.
    reciprocal_density
        Density of k-mesh by reciprocal volume.
    reciprocal_density_metal
        Density of k-mesh by reciprocal volume for use when the system is metallic
        and ``auto_metal_kpoints=True`` (the default).
    line_density
        Line density for line mode band structure.
    optics
        Whether to add LOPTICS (used for calculating optical response).
    nbands_factor
        Multiplicative factor for NBANDS when starting from a previous calculation.
        Choose a higher number if you are doing an LOPTICS calculation.
    **kwargs
        Other keyword arguments that will be passed to :obj:`VaspInputGenerator`.
    """

    mode: str = "line"
    dedos: float = 0.02
    reciprocal_density: float = 100
    reciprocal_density_metal: float = 400
    line_density: float = 20
    optics: bool = False
    nbands_factor: float = 1.2
    auto_ispin: bool = True

    def __post_init__(self) -> None:
        """Ensure mode is set correctly."""
        super().__post_init__()
        self.mode = self.mode.lower()

        supported_modes = ("line", "uniform", "boltztrap")
        if self.mode not in supported_modes:
            raise ValueError(f"Supported modes are: {', '.join(supported_modes)}")

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
        if self.mode == "line":
            return {"line_density": self.line_density}

        if self.mode == "boltztrap":
            return {"explicit": True, "reciprocal_density": self.reciprocal_density}

        return {
            "reciprocal_density": self.reciprocal_density,
            "reciprocal_density_metal": self.reciprocal_density_metal,
        }

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """Get updates to the INCAR for a non-self-consistent field VASP job.

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
        updates: dict[str, Any] = {
            "LCHARG": False,
            "LORBIT": 11,
            "LWAVE": False,
            "NSW": 0,
            "ISYM": 0,
            "ICHARG": 11,
            "KSPACING": None,
        }

        if vasprun is not None:
            # set NBANDS
            n_bands = int(np.ceil(vasprun.parameters["NBANDS"] * self.nbands_factor))
            updates["NBANDS"] = n_bands

        if self.mode == "uniform":
            # automatic setting of NEDOS using the energy range and the energy step
            n_edos = _get_nedos(vasprun, self.dedos)

            # use tetrahedron method for DOS and optics calculations
            updates.update({"ISMEAR": -5, "ISYM": 2, "NEDOS": n_edos})

        elif self.mode in ("line", "boltztrap"):
            # if line mode or explicit k-points (boltztrap) can't use ISMEAR=-5
            # use small sigma to avoid partial occupancies for small band gap materials
            # use a larger sigma if the material is a metal
            sigma = 0.2 if bandgap == 0 else 0.01
            updates.update({"ISMEAR": 0, "SIGMA": sigma})

        if self.optics:
            # LREAL not supported with LOPTICS = True; automatic NEDOS usually
            # underestimates, so set it explicitly
            updates.update(
                {"LOPTICS": True, "LREAL": False, "CSHIFT": 1e-5, "NEDOS": 2000}
            )

        updates["MAGMOM"] = None

        return updates


@dataclass
class HSERelaxSetGenerator(VaspInputGenerator):
    """Class to generate VASP HSE06 relaxation input sets.

    .. note::
        By default the hybrid input sets use ALGO = Normal which is only efficient for
        VASP 6.0 and higher. See https://www.vasp.at/wiki/index.php/LFOCKACE for more
        details.
    """

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """Get updates to the INCAR for a VASP HSE06 relaxation job.

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
            "ALGO": "Normal",
            "GGA": "PE",
            "HFSCREEN": 0.2,
            "LHFCALC": True,
            "IBRION": 2,
            "PRECFOCK": "Fast",
            "ISIF": 3,
            "LASPH": True,
            "LDAU": False,
        }


@dataclass
class HSETightRelaxSetGenerator(VaspInputGenerator):
    """Class to generate tight VASP HSE relaxation input sets.

    .. note::
        By default the hybrid input sets use ALGO = Normal which is only efficient for
        VASP 6.0 and higher. See https://www.vasp.at/wiki/index.php/LFOCKACE for more
        details.
    """

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """Get updates to the INCAR for an HSE tight relaxation job.

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
            "IBRION": 2,
            "ISIF": 3,
            "ENCUT": 700,
            "EDIFF": 1e-7,
            "LAECHG": False,
            "EDIFFG": -0.001,
            "LREAL": False,
            "ALGO": "Normal",
            "NSW": 99,
            "LCHARG": False,
            "GGA": "PE",
            "HFSCREEN": 0.2,
            "LHFCALC": True,
            "PRECFOCK": "Fast",
            "LASPH": True,
            "LDAU": False,
        }


@dataclass
class HSEStaticSetGenerator(VaspInputGenerator):
    """Class to generate VASP HSE06 static input sets.

    .. note::
        By default the hybrid input sets use ALGO = Normal which is only efficient for
        VASP 6.0 and higher. See https://www.vasp.at/wiki/index.php/LFOCKACE for more
        details.
    """

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """Get updates to the INCAR for a VASP HSE06 static job.

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
            "ALGO": "Normal",
            "GGA": "PE",
            "HFSCREEN": 0.2,
            "LHFCALC": True,
            "PRECFOCK": "Fast",
            "ISMEAR": -5,
            "LORBIT": 11,
            "LCHARG": True,
            "LASPH": True,
            "LREAL": False,
            "LDAU": False,
        }


@dataclass
class HSEBSSetGenerator(VaspInputGenerator):
    """
    Class to generate VASP HSE06 band structure input sets.

    HSE06 band structures must be self-consistent. A band structure along symmetry lines
    for instance needs BOTH a uniform grid with appropriate weights AND a path along the
    lines with weight 0.

    Thus, the "uniform" mode is just like regular static SCF but allows adding custom
    kpoints (e.g., corresponding to known VBM/CBM) to the uniform grid that have zero
    weight (e.g., for better gap estimate).

    The "gap" mode behaves just like the "uniform" mode, however, if starting from a
    previous calculation, the VBM and CBM k-points will automatically be added to
    ``added_kpoints``.

    The "line" mode is just like Uniform mode, but additionally adds k-points along
    symmetry lines with zero weight.

    The "uniform_dense" mode employs are regular weighted k-point mesh, in addition
    to a zero-weighted uniform mesh with higher density.

    .. note::
        By default the hybrid input sets use ALGO = Normal which is only efficient for
        VASP 6.0 and higher. See https://www.vasp.at/wiki/index.php/LFOCKACE for more
        details.

    Parameters
    ----------
    mode
        Type of band structure mode. Options are "line", "uniform", "gap", or
        "uniform_dense".
    dedos
        Energy difference used to set NEDOS, based on the total energy range.
    reciprocal_density
        Density of k-mesh by reciprocal volume.
    line_density
        Line density for line mode band structure.
    zero_weighted_reciprocal_density
        Density of uniform zero weighted k-point mesh.
    optics
        Whether to add LOPTICS (used for calculating optical response).
    nbands_factor
        Multiplicative factor for NBANDS when starting from a previous calculation.
        Choose a higher number if you are doing an LOPTICS calculation.
    added_kpoints
        A list of kpoints in fractional coordinates to add as zero-weighted points.
    **kwargs
        Other keyword arguments that will be passed to :obj:`VaspInputGenerator`.
    """

    mode: str = "gap"
    dedos: float = 0.02
    reciprocal_density: float = 64
    line_density: float = 20
    zero_weighted_reciprocal_density: float = 100
    optics: bool = False
    nbands_factor: float = 1.2
    added_kpoints: list[Vector3D] = field(default_factory=list)
    auto_ispin: bool = True

    def __post_init__(self) -> None:
        """Ensure mode is set correctly."""
        super().__post_init__()

        self.mode = self.mode.lower()
        supported_modes = ("line", "uniform", "gap", "uniform_dense")
        if self.mode not in supported_modes:
            raise ValueError(f"Supported modes are: {', '.join(supported_modes)}")

    def get_kpoints_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = 0.0,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """Get updates to the kpoints configuration for a VASP HSE06 band structure job.

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
        kpoints: dict[str, Any] = {"reciprocal_density": self.reciprocal_density}

        if self.mode == "line":
            # add line_density on top of reciprocal density
            kpoints["zero_weighted_line_density"] = self.line_density

        elif self.mode == "uniform_dense":
            kpoints["zero_weighted_reciprocal_density"] = (
                self.zero_weighted_reciprocal_density
            )

        added_kpoints = deepcopy(self.added_kpoints)
        if vasprun is not None and self.mode == "gap":
            bs = vasprun.get_band_structure()
            if not bs.is_metal():
                added_kpoints.append(bs.get_vbm()["kpoint"].frac_coords)
                added_kpoints.append(bs.get_cbm()["kpoint"].frac_coords)

        kpoints["added_kpoints"] = added_kpoints

        return kpoints

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """Get updates to the INCAR for a VASP HSE06 band structure job.

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
        updates = {
            "NSW": 0,
            "ALGO": "Normal",
            "GGA": "PE",
            "HFSCREEN": 0.2,
            "PRECFOCK": "Fast",
            "LHFCALC": True,
            "LCHARG": False,
            "NELMIN": 5,
            "KSPACING": None,
            "LORBIT": 11,
            "LREAL": False,
            "LDAU": False,
        }

        if self.mode == "uniform" and len(self.added_kpoints) == 0:
            # automatic setting of nedos using the energy range and the energy step
            nedos = _get_nedos(vasprun, self.dedos)

            # use tetrahedron method for DOS and optics calculations
            updates.update({"ISMEAR": -5, "NEDOS": nedos})

        else:
            # if line mode or explicit k-points (gap) can't use ISMEAR=-5
            # use small sigma to avoid partial occupancies for small band gap materials
            updates.update({"ISMEAR": 0, "SIGMA": 0.01})

        if vasprun is not None:
            # set nbands
            nbands = int(np.ceil(vasprun.parameters["NBANDS"] * self.nbands_factor))
            updates["NBANDS"] = nbands

        if self.optics:
            # LREAL not supported with LOPTICS
            updates.update({"LOPTICS": True, "LREAL": False, "CSHIFT": 1e-5})

        updates["MAGMOM"] = None

        return updates


@dataclass
class ElectronPhononSetGenerator(VaspInputGenerator):
    """
    Class to generate VASP electron phonon input sets.

    .. note::
        Requires VASP 6.0 and higher. See https://www.vasp.at/wiki/index.php/Electron-
        phonon_interactions_from_Monte-Carlo_sampling for more details.

    Parameters
    ----------
    temperatures : list of float
        The temperatures for which the electron-phonon interactions are evaluated.
    reciprocal_density
        Density of k-mesh by reciprocal volume.
    """

    temperatures: tuple[float, ...] = (
        0,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
    )
    reciprocal_density: float = 64
    auto_ispin: bool = True

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """Get updates to the INCAR for a static VASP job.

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
            "NSW": 1,
            "ISMEAR": 0,
            "IBRION": 6,
            "ISIF": 2,
            "ENCUT": 700,
            "EDIFF": 1e-7,
            "LAECHG": False,
            "LREAL": False,
            "LCHARG": False,
            "LVTOT": False,
            "LVHAR": False,
            "PREC": "Accurate",
            "KSPACING": None,
            "PHON_NTLIST": len(self.temperatures),
            "PHON_TLIST": list(self.temperatures),  # has to be a list otherwise error
            "PHON_NSTRUCT": 0,
            "PHON_LMC": True,
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
        return {"reciprocal_density": self.reciprocal_density}


@dataclass
class MDSetGenerator(VaspInputGenerator):
    """
    Class to generate VASP molecular dynamics input sets.

    Parameters
    ----------
    ensemble
        Molecular dynamics ensemble to run. Options include `nvt`, `nve`, and `npt`.
    start_temp
        Starting temperature. The VASP `TEBEG` parameter.
    end_temp
        Final temperature. The VASP `TEEND` parameter.
    nsteps
        Number of time steps for simulations. The VASP `NSW` parameter.
    time_step
        The time step (in femtosecond) for the simulation. The VASP `POTIM` parameter.
    **kwargs
        Other keyword arguments that will be passed to :obj:`VaspInputGenerator`.
    """

    ensemble: str = "nvt"
    start_temp: float = 300
    end_temp: float = 300
    nsteps: int = 1000
    time_step: float = 2
    auto_ispin: bool = True

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """Get updates to the INCAR for a molecular dynamics job.

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
        updates = self._get_ensemble_defaults(structure, self.ensemble)

        # Based on pymatgen.io.vasp.sets.MPMDSet.
        updates.update(
            {
                "ENCUT": 520,
                "TEBEG": self.start_temp,
                "TEEND": self.end_temp,
                "NSW": self.nsteps,
                "POTIM": self.time_step,
                "LCHARG": False,
                "NELMIN": 4,
                "MAXMIX": 20,
                "NELM": 500,
                "ISYM": 0,
                "IBRION": 0,
                "KBLOCK": 100,
                "PREC": "Normal",
            }
        )

        if Element("H") in structure.species and updates["POTIM"] > 0.5:
            logger.warning(
                f"Molecular dynamics time step is {updates['POTIM']}, which is "
                "typically too large for a structure containing H. Consider set it "
                "to a value of 0.5 or smaller."
            )

        return updates

    @staticmethod
    def _get_ensemble_defaults(structure: Structure, ensemble: str) -> dict[str, Any]:
        """Get default params for the ensemble."""
        defaults = {
            "nve": {"MDALGO": 1, "ISIF": 2, "ANDERSEN_PROB": 0.0},
            "nvt": {"MDALGO": 2, "ISIF": 2, "SMASS": 0},
            "npt": {
                "MDALGO": 3,
                "ISIF": 3,
                "LANGEVIN_GAMMA": [10] * structure.ntypesp,
                "LANGEVIN_GAMMA_L": 1,
                "PMASS": 10,
                "PSTRESS": 0,
            },
        }

        try:
            return defaults[ensemble.lower()]  # type: ignore[return-value]
        except KeyError as err:
            supported = tuple(defaults)
            raise ValueError(f"Expect {ensemble=} to be one of {supported}") from err


def _get_nedos(vasprun: Vasprun | None, dedos: float) -> int:
    """Automatic setting of nedos using the energy range and the energy step."""
    if vasprun is None:
        return 2000

    emax = max(eigs.max() for eigs in vasprun.eigenvalues.values())
    emin = min(eigs.min() for eigs in vasprun.eigenvalues.values())
    return int((emax - emin) / dedos)
