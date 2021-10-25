"""Module defining core VASP input set generators."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Outcar, Poscar, Vasprun

from atomate2.common.schemas.math import Vector3D
from atomate2.vasp.sets.base import VaspInputSet, VaspInputSetGenerator

__all__ = [
    "RelaxSetGenerator",
    "TightRelaxSetGenerator",
    "StaticSetGenerator",
    "NonSCFSetGenerator",
    "HSERelaxSetGenerator",
    "HSEBSSetGenerator",
]


class RelaxSetGenerator(VaspInputSetGenerator):
    """Class to generate VASP relaxation input sets."""

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
        return {"NSW": 99, "LCHARG": False, "ISIF": 3, "IBRION": 2}


class TightRelaxSetGenerator(VaspInputSetGenerator):
    """Class to generate tight VASP relaxation input sets."""

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = 0,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """
        Get updates to the INCAR for a tight relaxation job.

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
        }


class StaticSetGenerator(VaspInputSetGenerator):
    """
    Class to generate VASP static input sets.

    Parameters
    ----------
    lepsilon
        Whether to set LEPSILON (used for calculating the high-frequency dielectric
        tensor).
    lcalcpol
        Whether to set LCALCPOL (used tfor calculating the electronic contribution to
        the polarization)
    **kwargs
        Other keyword arguments that will be passed to :obj:`VaspInputSetGenerator`.
    """

    def __init__(self, lepsilon: bool = False, lcalcpol: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.lepsilon = lepsilon
        self.lcalcpol = lcalcpol

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = 0,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """
        Get updates to the INCAR for a static VASP job.

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
            "NSW": 1,
            "ISMEAR": -5,
            "LCHARG": True,
            "LORBIT": 11,
            "ALGO": "Normal",
        }
        if self.lepsilon:
            # LPEAD=T: numerical evaluation of overlap integral prevents LRF_COMMUTATOR
            # errors and can lead to better expt. agreement but produces slightly
            # different results
            updates = {"IBRION": 8, "LEPSILON": True, "LPEAD": True}

        if self.lcalcpol:
            updates["LCALCPOL"] = True
        return updates


class NonSCFSetGenerator(VaspInputSetGenerator):
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
    line_density
        Line density for line mode band structure.
    optics
        Whether to add LOPTICS (used for calculating optical response).
    nbands_factor
        Multiplicative factor for NBANDS when starting from a previous calculation.
        Choose a higher number if you are doing an LOPTICS calculation.
    **kwargs
        Other keyword arguments that will be passed to :obj:`VaspInputSetGenerator`.
    """

    def __init__(
        self,
        mode: str = "line",
        dedos: float = 0.005,
        reciprocal_density=100,
        line_density=20,
        optics: bool = False,
        nbands_factor: float = 1.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode.lower()
        self.dedos = dedos
        self.line_density = line_density
        self.reciprocal_density = reciprocal_density
        self.optics = optics
        self.nbands_factor = nbands_factor

        supported_modes = ("line", "uniform", "boltztrap")
        if self.mode not in supported_modes:
            raise ValueError(f"Supported modes are: {', '.join(supported_modes)}")

        self._config_dict["INCAR"].pop("KSPACING")
        if mode == "line":
            self._config_dict["KPOINTS"] = {"line_density": self.line_density}

        elif mode == "boltztrap":
            self._config_dict["KPOINTS"] = {
                "explicit": True,
                "reciprocal_density": reciprocal_density,
            }

        else:
            self._config_dict["KPOINTS"] = {
                "reciprocal_density": self.reciprocal_density
            }

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = 0,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """
        Get updates to the INCAR for a non-self-consistent field VASP job.

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
        updates: Dict[str, Any] = {
            "IBRION": -1,
            "LCHARG": False,
            "LORBIT": 11,
            "LWAVE": False,
            "NSW": 0,
            "ISYM": 0,
            "ICHARG": 11,
        }

        nedos = 2000
        if vasprun is not None:
            # automatic setting of nedos using the energy range and the energy step
            emax = max([eigs.max() for eigs in vasprun.eigenvalues.values()])
            emin = min([eigs.min() for eigs in vasprun.eigenvalues.values()])
            nedos = int((emax - emin) / self.dedos)

            # set nbands
            nbands = int(np.ceil(vasprun.parameters["NBANDS"] * self.nbands_factor))
            updates["NBANDS"] = nbands

        if outcar is not None and outcar.magnetization is not None:
            # Turn off spin when magmom for every site is smaller than 0.02.
            site_magmom = np.array([i["tot"] for i in outcar.magnetization])
            updates["ISPIN"] = 2 if np.any(np.abs(site_magmom) > 0.02) else 1
        elif vasprun is not None:
            updates["ISPIN"] = 2 if vasprun.is_spin else 1

        if self.mode == "uniform":
            # use tetrahedron method for DOS and optics calculations
            updates.update({"ISMEAR": -5, "ISYM": 2, "NEDOS": nedos})

        elif self.mode in ("line", "boltztrap"):
            # if line mode or explicit k-points (boltztrap) can't use ISMEAR=-5
            # use small sigma to avoid partial occupancies for small band gap materials
            updates.update({"ISMEAR": 0, "SIGMA": 0.01})

        if self.optics:
            updates["LOPTICS"] = True

        updates["MAGMOM"] = None

        return updates


class HSERelaxSetGenerator(VaspInputSetGenerator):
    """Class to generate VASP HSE06 relaxation input sets."""

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = 0,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """
        Get updates to the INCAR for a VASP HSE06 relaxation job.

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
            "ALGO": "All",
            "GGA": "PE",
            "HFSCREEN": 0.2,
            "LHFCALC": True,
            "IBRION": 2,
            "PRECFOCK": "Fast",
            "ISIF": 3,
        }


class HSEBSSetGenerator(VaspInputSetGenerator):
    """
    Class to generate VASP HSE06 band structure input sets.

    HSE06 band structures must be self-consistent. A band structure along symmetry lines
    for instance needs BOTH a uniform grid with appropriate weights AND a path along the
    lines with weight 0.

    Thus, the "uniform" mode is just like regular static SCF but allows adding custom
    kpoints (e.g., corresponding to known VBM/CBM) to the uniform grid that have zero
    weight (e.g., for better gap estimate).

    The "gap" mode behaves just like the "Uniform" mode, however, if starting from a
    previous calculation, the VBM and CBM k-points will automatically be added to
    ``added_kpoints``.

    The "line" mode is just like Uniform mode, but additionally adds k-points along
    symmetry lines with zero weight.

    Parameters
    ----------
    mode
        Type of band structure mode. Options are "line", "uniform", or "gap".
    reciprocal_density
        Density of k-mesh by reciprocal volume.
    line_density
        Line density for line mode band structure.
    added_kpoints
        A list of kpoints in fractional coordinates to add as zero-weighted points.
    **kwargs
        Other keyword arguments that will be passed to :obj:`VaspInputSetGenerator`.
    """

    def __init__(
        self,
        mode: str = "gap",
        reciprocal_density: int = 50,
        line_density: int = 20,
        added_kpoints: List[Vector3D] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.reciprocal_density = reciprocal_density
        self.line_density = line_density
        self.added_kpoints = [] if added_kpoints is None else added_kpoints

        supported_modes = ("line", "uniform", "gap")
        if self.mode not in supported_modes:
            raise ValueError(f"Supported modes are: {', '.join(supported_modes)}")

        self._config_dict["INCAR"].pop("KSPACING")
        self._config_dict["KPOINTS"] = {"reciprocal_density": self.reciprocal_density}

        if mode == "line":
            # add line_density on top of reciprocal density
            self._config_dict["KPOINTS"]["line_density"] = self.line_density

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = 0,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """
        Get updates to the INCAR for a VASP HSE06 band structure job.

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
            "ALGO": "All",
            "GGA": "PE",
            "HFSCREEN": 0.2,
            "LHFCALC": True,
            "LCHARG": False,
            "NELMIN": 5,
        }

    def get_input_set(  # type: ignore
        self, structure: Structure = None, prev_dir: Union[str, Path] = None
    ):
        """
        Get a VASP input set.

        Note, if both ``structure`` and ``prev_dir`` are set, then the structure
        specified will be preferred over the final structure from the last VASP run.

        Parameters
        ----------
        structure
            A structure.
        prev_dir
            A previous directory to generate the input set from.

        Returns
        -------
        VaspInputSet
            A VASP input set.
        """
        structure, prev_incar, bandgap, vasprun, outcar = self._get_previous(
            structure, prev_dir
        )
        added_kpoints = deepcopy(self.added_kpoints)
        if vasprun is not None and self.mode == "gap":
            bs = vasprun.get_band_structure()
            if not bs.is_metal():
                added_kpoints.append(bs.get_vbm()["kpoint"].frac_coords)
                added_kpoints.append(bs.get_cbm()["kpoint"].frac_coords)

        updates = self.get_incar_updates(
            structure,
            prev_incar=prev_incar,
            bandgap=bandgap,
            vasprun=vasprun,
            outcar=outcar,
        )
        kpoints = self._get_kpoints(structure, added_kpoints=added_kpoints)
        incar = self._get_incar(
            structure, kpoints, prev_incar, updates, bandgap=bandgap
        )
        return VaspInputSet(
            incar=incar,
            kpoints=kpoints,
            poscar=Poscar(structure),
            potcar=self._get_potcar(structure),
        )
