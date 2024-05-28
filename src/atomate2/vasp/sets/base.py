"""Module defining base VASP input set and generator."""

from __future__ import annotations

import glob
import os
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from importlib.resources import files as get_mod_path
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from monty.io import zopen
from monty.serialization import loadfn
from pymatgen.electronic_structure.core import Magmom
from pymatgen.io.core import InputGenerator, InputSet
from pymatgen.io.vasp import Incar, Kpoints, Outcar, Poscar, Potcar, Vasprun
from pymatgen.io.vasp.sets import (
    BadInputSetWarning,
    get_valid_magmom_struct,
    get_vasprun_outcar,
)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath

from atomate2 import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Structure

_BASE_VASP_SET = loadfn(get_mod_path("atomate2.vasp.sets") / "BaseVaspSet.yaml")


class VaspInputSet(InputSet):
    """
    A class to represent a set of VASP inputs.

    Parameters
    ----------
    incar
        An Incar object.
    kpoints
        A Kpoints object.
    poscar
        A Poscar object.
    potcar
        A list of Potcar objects.
    optional_files
        Other input files supplied as a dict of ``{filename: object}``. The objects
        should follow standard pymatgen conventions in implementing an ``as_dict()``
        and ``from_dict`` method.
    """

    def __init__(
        self,
        incar: Incar,
        poscar: Poscar,
        potcar: Potcar | list[str],
        kpoints: Kpoints | None = None,
        optional_files: dict | None = None,
    ) -> None:
        self.incar = incar
        self.poscar = poscar
        self.potcar = potcar
        self.kpoints = kpoints
        self.optional_files = optional_files or {}

    def write_input(
        self,
        directory: str | Path,
        make_dir: bool = True,
        overwrite: bool = True,
        potcar_spec: bool = False,
    ) -> None:
        """Write VASP input files to a directory.

        Parameters
        ----------
        directory
            Directory to write input files to.
        make_dir
            Whether to create the directory if it does not already exist.
        overwrite
            Whether to overwrite an input file if it already exists.
        """
        directory = Path(directory)
        if make_dir:
            os.makedirs(directory, exist_ok=True)

        inputs = {"INCAR": self.incar, "KPOINTS": self.kpoints, "POSCAR": self.poscar}
        inputs.update(self.optional_files)

        if isinstance(self.potcar, Potcar):
            inputs["POTCAR"] = self.potcar
        else:
            inputs["POTCAR.spec"] = "\n".join(self.potcar)

        for key, val in inputs.items():
            if val is not None and (overwrite or not (directory / key).exists()):
                with zopen(directory / key, mode="wt") as file:
                    if isinstance(val, Poscar):
                        # write POSCAR with more significant figures
                        file.write(val.get_str(significant_figures=16))
                    else:
                        file.write(str(val))
            elif not overwrite and (directory / key).exists():
                raise FileExistsError(f"{directory / key} already exists.")

    @staticmethod
    def from_directory(
        directory: str | Path, optional_files: dict = None
    ) -> VaspInputSet:
        """Load a set of VASP inputs from a directory.

        Note that only the standard INCAR, POSCAR, POTCAR and KPOINTS files are read
        unless optional_filenames is specified.

        Parameters
        ----------
        directory
            Directory to read VASP inputs from.
        optional_files
            Optional files to read in as well as a dict of {filename: Object class}.
            Object class must have a static/class method from_file.
        """
        directory = Path(directory)
        objs = {"INCAR": Incar, "KPOINTS": Kpoints, "POSCAR": Poscar, "POTCAR": Potcar}

        inputs = {}
        for name, obj in objs.items():
            if (directory / name).exists():
                inputs[name.lower()] = obj.from_file(directory / name)
            else:
                # handle the case where there is no KPOINTS file
                inputs[name.lower()] = None

        optional_inputs = {}
        if optional_files is not None:
            for name, obj in optional_files.items():
                optional_inputs[name] = obj.from_file(directory / name)

        return VaspInputSet(**inputs, optional_files=optional_inputs)

    @property
    def is_valid(self) -> bool:
        """Whether the input set is valid."""
        if self.incar.get("KSPACING", 0) > 0.5 and self.incar.get("ISMEAR", 0) == -5:
            warnings.warn(
                "Large KSPACING value detected with ISMEAR=-5. Ensure that VASP "
                "generates enough KPOINTS, lower KSPACING, or set ISMEAR=0",
                BadInputSetWarning,
                stacklevel=1,
            )

        ismear = self.incar.get("ISMEAR", 1)
        sigma = self.incar.get("SIGMA", 0.2)
        if (
            all(elem.is_metal for elem in self.poscar.structure.composition)
            and self.incar.get("NSW", 0) > 0
            and (ismear < 0 or (ismear == 0 and sigma > 0.05))
        ):
            ismear_docs = "https://www.vasp.at/wiki/index.php/ISMEAR"
            msg = ""
            if ismear < 0:
                msg = f"Relaxation of likely metal with ISMEAR < 0 ({ismear})."
            elif ismear == 0 and sigma > 0.05:
                msg = f"ISMEAR = 0 with a small SIGMA ({sigma}) detected."
            warnings.warn(
                f"{msg} See VASP recommendations on ISMEAR for metals ({ismear_docs}).",
                BadInputSetWarning,
                stacklevel=1,
            )

        algo = self.incar.get("ALGO", "Normal")
        if self.incar.get("LHFCALC") and algo not in ("Normal", "All", "Damped"):
            warnings.warn(
                "Hybrid functionals only support Algo = All, Damped, or Normal.",
                BadInputSetWarning,
                stacklevel=1,
            )

        if not self.incar.get("LASPH") and (
            self.incar.get("METAGGA")
            or self.incar.get("LHFCALC")
            or self.incar.get("LDAU")
            or self.incar.get("LUSE_VDW")
        ):
            msg = "LASPH = True should be set for +U, meta-GGAs, hybrids, and vdW-DFT"
            warnings.warn(msg, BadInputSetWarning, stacklevel=1)

        return True


@dataclass
class VaspInputGenerator(InputGenerator):
    """
    A class to generate VASP input sets.

    .. Note::
       Get the magmoms using the following precedence.

        1. user incar settings
        2. magmoms in input struct
        3. spins in input struct
        4. job config dict
        5. set all magmoms to 0.6

    Parameters
    ----------
    user_incar_settings
        User INCAR settings. This allows a user to override INCAR settings, e.g.,
        setting a different MAGMOM for various elements or species. The config_dict
        supports EDIFF_PER_ATOM and EDIFF keys. The former scales with # of atoms, the
        latter does not. If both are present, EDIFF is preferred. To force such
        settings, just supply user_incar_settings={"EDIFF": 1e-5, "LDAU": False} for
        example. The keys 'LDAUU', 'LDAUJ', 'LDAUL' are special cases since pymatgen
        defines different values depending on what anions are present in the structure,
        so these keys can be defined in one of two ways, e.g. either
        {"LDAUU":{"O":{"Fe":5}}} to set LDAUU for Fe to 5 in an oxide, or
        {"LDAUU":{"Fe":5}} to set LDAUU to 5 regardless of the input structure.
        To set magmoms, pass a dict mapping the strings of species to magnetic
        moments, e.g. {"MAGMOM": {"Co": 1}} or {"MAGMOM": {"Fe2+,spin=4": 3.7}} in the
        case of a site with Species("Fe2+", spin=4).
        If None is given, that key is unset. For example, {"ENCUT": None} will remove
        ENCUT from the incar settings.
    user_kpoints_settings
        Allow user to override kpoints setting by supplying a dict. E.g.,
        ``{"reciprocal_density": 1000}``. User can also supply a Kpoints object.
    user_potcar_settings
        Allow user to override POTCARs. E.g., {"Gd": "Gd_3"}.
    user_potcar_functional
        Functional to use. Default is to use the functional in the config dictionary.
        Valid values: "PBE", "PBE_52", "PBE_54", "PBE_64", "LDA", "LDA_52", "LDA_54",
        "LDA_64", "PW91", "LDA_US", "PW91_US".
    auto_ismear
        If true, the values for ISMEAR and SIGMA will be set automatically depending
        on the bandgap of the system. If the bandgap is not known (e.g., there is no
        previous VASP directory) then ISMEAR=0 and SIGMA=0.2; if the bandgap is zero (a
        metallic system) then ISMEAR=2 and SIGMA=0.2; if the system is an insulator,
        then ISMEAR=-5 (tetrahedron smearing). Note, this only works when generating the
        input set from a previous VASP directory.
    auto_ispin
        If generating input set from a previous calculation, this controls whether
        to disable magnetisation (ISPIN = 1) if the absolute value of all magnetic
        moments are less than 0.02.
    auto_lreal
        If True, automatically use the VASP recommended LREAL based on cell size.
    auto_metal_kpoints
        If true and the system is metallic, try and use ``reciprocal_density_metal``
        instead of ``reciprocal_density`` for metallic systems. Note, this only works
        when generating the input set from a previous VASP directory.
    auto_kspacing
        If true, automatically use the VASP recommended KSPACING based on bandgap,
        i.e. higher kpoint spacing for insulators than metals. Can be boolean or float.
        If a float, the value will be interpreted as the bandgap in eV to use for the
        KSPACING calculation.
    constrain_total_magmom
        Whether to constrain the total magmom (NUPDOWN in INCAR) to be the sum of the
        initial MAGMOM guess for all species.
    validate_magmom
        Ensure that missing magmom values are filled in with the default value of 1.0.
    use_structure_charge
        If set to True, then the overall charge of the structure (``structure.charge``)
        is used to set NELECT.
    sort_structure
        Whether to sort the structure (using the default sort order of
        electronegativity) before generating input files. Defaults to True, the behavior
        you would want most of the time. This ensures that similar atomic species are
        grouped together.
    force_gamma
        Force gamma centered kpoint generation.
    vdw
        Adds default parameters for van-der-Waals functionals supported by VASP to
        INCAR. Supported functionals are: DFT-D2, undamped DFT-D3, DFT-D3 with
        Becke-Jonson damping, Tkatchenko-Scheffler, Tkatchenko-Scheffler with iterative
        Hirshfeld partitioning, MBD@rSC, dDsC, Dion's vdW-DF, DF2, optPBE, optB88,
        optB86b and rVV10.
    symprec
        Tolerance for symmetry finding, used for line mode band structure k-points.
    config_dict
        The config dictionary to use containing the base input set settings.
    inherit_incar
        Whether to inherit INCAR settings from previous calculation. This might be
        useful to port Custodian fixes to child jobs but can also be dangerous e.g.
        when switching from GGA to meta-GGA or relax to static jobs. Defaults to True.
    """

    user_incar_settings: dict = field(default_factory=dict)
    user_kpoints_settings: dict | Kpoints = field(default_factory=dict)
    user_potcar_settings: dict = field(default_factory=dict)
    user_potcar_functional: str = None
    auto_ismear: bool = True
    auto_ispin: bool = False
    auto_lreal: bool = False
    auto_kspacing: bool | float = False
    auto_metal_kpoints: bool = True
    constrain_total_magmom: bool = False
    validate_magmom: bool = True
    use_structure_charge: bool = False
    sort_structure: bool = True
    force_gamma: bool = True
    symprec: float = SETTINGS.SYMPREC
    vdw: str = None
    # copy _BASE_VASP_SET to ensure each class instance has its own copy
    # otherwise in-place changes can affect other instances
    config_dict: dict = field(default_factory=lambda: _BASE_VASP_SET)
    inherit_incar: bool = None

    def __post_init__(self) -> None:
        """Post init formatting of arguments."""
        self.vdw = None if self.vdw is None else self.vdw.lower()

        if self.user_incar_settings.get("KSPACING") and self.user_kpoints_settings:
            warnings.warn(
                "You have specified KSPACING and also supplied kpoints settings. "
                "KSPACING only has effect when there is no KPOINTS file. Since both "
                "settings were given, pymatgen will generate a KPOINTS file and ignore "
                "KSPACING. Remove the `user_kpoints_settings` argument to enable "
                "KSPACING.",
                BadInputSetWarning,
                stacklevel=1,
            )

        if self.vdw:
            from pymatgen.io.vasp.sets import MODULE_DIR as PMG_SET_DIR

            vdw_par = loadfn(PMG_SET_DIR / "vdW_parameters.yaml")
            if self.vdw not in vdw_par:
                raise KeyError(
                    "Invalid or unsupported van-der-Waals functional. Supported "
                    f"functionals are {list(vdw_par)}"
                )
            self.config_dict["INCAR"].update(vdw_par[self.vdw])

        # read the POTCAR_FUNCTIONAL from the .yaml
        self.potcar_functional = self.config_dict.get("POTCAR_FUNCTIONAL", "PS")

        # warn if a user is overriding POTCAR_FUNCTIONAL
        if (
            self.user_potcar_functional
            and self.user_potcar_functional != self.potcar_functional
        ):
            warnings.warn(
                "Overriding the POTCAR functional is generally not recommended "
                "as it can significantly affect the results of calculations and "
                "compatibility with other calculations done with the same input set. "
                "Note that some POTCAR symbols specified in the configuration file may "
                "not be available in the selected functional.",
                BadInputSetWarning,
                stacklevel=1,
            )
            self.potcar_functional = self.user_potcar_functional

        if self.user_potcar_settings:
            warnings.warn(
                "Overriding POTCARs is generally not recommended as it can "
                "significantly affect the results of calculations and compatibility "
                "with other calculations done with the same input set. In many "
                "instances, it is better to write a subclass of a desired input set and"
                " override the POTCAR in the subclass to be explicit on the "
                "differences.",
                BadInputSetWarning,
                stacklevel=1,
            )
            for k, v in self.user_potcar_settings.items():
                self.config_dict["POTCAR"][k] = v

        if self.inherit_incar is None:
            self.inherit_incar = SETTINGS.VASP_INHERIT_INCAR

    def get_input_set(
        self,
        structure: Structure = None,
        prev_dir: str | Path = None,
        potcar_spec: bool = False,
    ) -> VaspInputSet:
        """Get a VASP input set.

        Note, if both ``structure`` and ``prev_dir`` are set, then the structure
        specified will be preferred over the final structure from the last VASP run.

        Parameters
        ----------
        structure
            A structure.
        prev_dir
            A previous directory to generate the input set from.
        potcar_spec
            Instead of generating a Potcar object, use a list of potcar symbols. This
            will be written as a "POTCAR.spec" file. This is intended to help sharing an
            input set with people who might not have a license to specific Potcar files.
            Given a "POTCAR.spec", the specific POTCAR file can be re-generated using
            pymatgen with the "generate_potcar" function in the pymatgen CLI.

        Returns
        -------
        VaspInputSet
            A VASP input set.
        """
        structure, prev_incar, bandgap, ispin, vasprun, outcar = self._get_previous(
            structure, prev_dir
        )
        prev_incar = prev_incar if self.inherit_incar else {}
        kwds = {
            "structure": structure,
            "prev_incar": prev_incar,
            "bandgap": bandgap,
            "vasprun": vasprun,
            "outcar": outcar,
        }
        incar_updates = self.get_incar_updates(**kwds)
        kpoints_updates = self.get_kpoints_updates(**kwds)
        kspacing = self._kspacing(incar_updates)
        kpoints = self._get_kpoints(structure, kpoints_updates, kspacing, bandgap)
        incar = self._get_incar(
            structure,
            kpoints,
            prev_incar,
            incar_updates,
            bandgap=bandgap,
            ispin=ispin,
        )
        site_properties = structure.site_properties
        poscar = Poscar(
            structure,
            velocities=site_properties.get("velocities"),
            predictor_corrector=site_properties.get("predictor_corrector"),
            predictor_corrector_preamble=structure.properties.get(
                "predictor_corrector_preamble"
            ),
            lattice_velocities=structure.properties.get("lattice_velocities"),
        )
        return VaspInputSet(
            incar=incar,
            kpoints=kpoints,
            poscar=poscar,
            potcar=self._get_potcar(structure, potcar_spec=potcar_spec),
        )

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = 0.0,
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
        raise NotImplementedError

    def get_kpoints_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = 0.0,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """Get updates to the kpoints configuration for this calculation type.

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
        return {}

    def get_nelect(self, structure: Structure) -> float:
        """Get the default number of electrons for a given structure.

        Parameters
        ----------
        structure
            A structure.

        Returns
        -------
        float
            Number of electrons for the structure.
        """
        potcar = self._get_potcar(structure, potcar_spec=False)
        map_elem_electrons = {p.element: p.nelectrons for p in potcar}
        comp = structure.composition.element_composition
        n_electrons = sum(
            num_atoms * map_elem_electrons[str(el)] for el, num_atoms in comp.items()
        )

        return n_electrons - (structure.charge if self.use_structure_charge else 0)

    def _get_previous(
        self, structure: Structure = None, prev_dir: str | Path = None
    ) -> tuple:
        """Load previous calculation outputs and decide which structure to use."""
        if structure is None and prev_dir is None:
            raise ValueError("Either structure or prev_dir must be set")

        prev_incar = {}
        prev_structure = None
        vasprun = None
        outcar = None
        bandgap = None
        ispin = None
        if prev_dir:
            vasprun, outcar = get_vasprun_outcar(prev_dir)

            path_prev_dir = Path(prev_dir)

            # CONTCAR is already renamed POSCAR
            contcars = list(glob.glob(str(path_prev_dir / "POSCAR*")))
            contcar_file_fullpath = str(path_prev_dir / "POSCAR")
            contcar_file = (
                contcar_file_fullpath
                if contcar_file_fullpath in contcars
                else max(contcars)
            )
            contcar = Poscar.from_file(contcar_file)

            if vasprun.efermi is None:
                # VASP doesn't output efermi in vasprun if IBRION = 1
                vasprun.efermi = outcar.efermi

            bs = vasprun.get_band_structure(efermi="smart")
            prev_incar = vasprun.incar
            # use structure from CONTCAR as it is written to greater
            # precision than in the vasprun
            prev_structure = contcar.structure
            bandgap = 0 if bs.is_metal() else bs.get_band_gap()["energy"]

            if self.auto_ispin:
                # turn off spin when magmom for every site is smaller than 0.02.
                ispin = _get_ispin(vasprun, outcar)

        structure = structure if structure is not None else prev_structure
        structure = self._get_structure(structure)

        return structure, prev_incar, bandgap, ispin, vasprun, outcar

    def _get_structure(self, structure: Structure) -> Structure:
        """Get the standardized structure."""
        for site in structure:
            if "magmom" in site.properties and isinstance(
                site.properties["magmom"], Magmom
            ):
                # required to fix bug in get_valid_magmom_struct
                site.properties["magmom"] = list(site.properties["magmom"])

        if self.sort_structure:
            structure = structure.get_sorted_structure()

        if self.validate_magmom:
            get_valid_magmom_struct(structure, spin_mode="auto", inplace=True)
        return structure

    def _get_potcar(self, structure: Structure, potcar_spec: bool = False) -> Potcar:
        """Get the POTCAR."""
        elements = [a[0] for a in groupby([s.specie.symbol for s in structure])]
        potcar_symbols = [self.config_dict["POTCAR"].get(el, el) for el in elements]

        if potcar_spec:
            return potcar_symbols

        potcar = Potcar(potcar_symbols, functional=self.potcar_functional)

        # warn if the selected POTCARs do not correspond to the chosen potcar_functional
        for psingle in potcar:
            if self.potcar_functional not in psingle.identify_potcar()[0]:
                warnings.warn(
                    f"POTCAR data with symbol {psingle.symbol} is not known by pymatgen"
                    " to correspond with the selected potcar_functional "
                    f"{self.potcar_functional}. This POTCAR is known to correspond with"
                    f" functionals {psingle.identify_potcar(mode='data')[0]}. Please "
                    "verify that you are using the right POTCARs!",
                    BadInputSetWarning,
                    stacklevel=1,
                )
        return potcar

    def _get_incar(
        self,
        structure: Structure,
        kpoints: Kpoints,
        previous_incar: dict = None,
        incar_updates: dict = None,
        bandgap: float = None,
        ispin: int = None,
    ) -> Incar:
        """Get the INCAR."""
        previous_incar = previous_incar or {}
        incar_updates = incar_updates or {}
        incar_settings = dict(self.config_dict["INCAR"])
        config_magmoms = incar_settings.get("MAGMOM", {})
        auto_updates = {}

        # apply user incar settings to SETTINGS not to INCAR
        _apply_incar_updates(incar_settings, self.user_incar_settings)

        # generate incar
        incar = Incar()
        for key, val in incar_settings.items():
            if key == "MAGMOM":
                incar[key] = get_magmoms(
                    structure,
                    magmoms=self.user_incar_settings.get("MAGMOM", {}),
                    config_magmoms=config_magmoms,
                )
            elif key in ("LDAUU", "LDAUJ", "LDAUL") and incar_settings.get(
                "LDAU", False
            ):
                incar[key] = _get_u_param(key, val, structure)
            elif key.startswith("EDIFF") and key != "EDIFFG":
                incar["EDIFF"] = _get_ediff(key, val, structure, incar_settings)
            else:
                incar[key] = val
        _set_u_params(incar, incar_settings, structure)

        # apply previous incar settings, be careful not to override user_incar_settings
        # also skip LDAU/MAGMOM as structure may have changed.
        skip = list(self.user_incar_settings)
        skip += ["MAGMOM", "NUPDOWN", "LDAUU", "LDAUL", "LDAUJ", "LMAXMIX"]
        _apply_incar_updates(incar, previous_incar, skip=skip)

        if self.constrain_total_magmom:
            nupdown = sum(mag if abs(mag) > 0.6 else 0 for mag in incar["MAGMOM"])
            if abs(nupdown - round(nupdown)) > 1e-5:
                warnings.warn(
                    "constrain_total_magmom was set to True, but the sum of MAGMOM "
                    "values is not an integer. NUPDOWN is meant to set the spin "
                    "multiplet and should typically be an integer. You are likely "
                    "better off changing the values of MAGMOM or simply setting "
                    "NUPDOWN directly in your INCAR settings.",
                    UserWarning,
                    stacklevel=1,
                )
            auto_updates["NUPDOWN"] = nupdown

        if self.use_structure_charge:
            auto_updates["NELECT"] = self.get_nelect(structure)

        # handle auto ISPIN
        if ispin is not None and "ISPIN" not in self.user_incar_settings:
            auto_updates["ISPIN"] = ispin

        if self.auto_ismear:
            bandgap_tol = getattr(self, "bandgap_tol", SETTINGS.BANDGAP_TOL)
            if bandgap is None:
                # don't know if we are a metal or insulator so set ISMEAR and SIGMA to
                # be safe with the most general settings
                auto_updates.update(ISMEAR=0, SIGMA=0.2)
            elif bandgap <= bandgap_tol:
                auto_updates.update(ISMEAR=2, SIGMA=0.2)  # metal
            else:
                auto_updates.update(ISMEAR=-5, SIGMA=0.05)  # insulator

        if self.auto_lreal:
            auto_updates["LREAL"] = _get_recommended_lreal(structure)

        if self.auto_kspacing is False:
            bandgap = None  # don't auto-set KSPACING based on bandgap
        elif isinstance(self.auto_kspacing, float):
            # interpret auto_kspacing as bandgap and set KSPACING based on user input
            bandgap = self.auto_kspacing

        _set_kspacing(incar, incar_settings, self.user_incar_settings, bandgap, kpoints)

        # apply updates from auto options, careful not to override user_incar_settings
        _apply_incar_updates(incar, auto_updates, skip=list(self.user_incar_settings))

        # apply updates from inputset generator
        _apply_incar_updates(incar, incar_updates, skip=list(self.user_incar_settings))

        # Remove unused INCAR parameters
        _remove_unused_incar_params(incar, skip=list(self.user_incar_settings))

        # Finally, re-apply `self.user_incar_settings` to make sure any accidentally
        # overwritten settings are changed back to the intended values.
        # skip dictionary parameters to avoid dictionaries appearing in the INCAR
        skip = ["LDAUU", "LDAUJ", "LDAUL", "MAGMOM"]
        _apply_incar_updates(incar, self.user_incar_settings, skip=skip)

        return incar

    def _get_kpoints(
        self,
        structure: Structure,
        kpoints_updates: dict[str, Any] | None,
        kspacing: float | None,
        bandgap: float | None,
    ) -> Kpoints | None:
        """Get the kpoints file."""
        kpoints_updates = kpoints_updates or {}

        # use user setting if set otherwise default to base config settings
        if self.user_kpoints_settings != {}:
            kconfig = deepcopy(self.user_kpoints_settings)
        else:
            # apply updates to k-points config
            kconfig = deepcopy(self.config_dict.get("KPOINTS", {}))
            kconfig.update(kpoints_updates)

        # Return None if KSPACING is set and no other user k-points settings have been
        # specified, because this will cause VASP to generate the kpoints automatically
        if kspacing and not self.user_kpoints_settings:
            return None

        if isinstance(kconfig, Kpoints):
            return kconfig

        explicit = (
            kconfig.get("explicit")
            or len(kconfig.get("added_kpoints", [])) > 0
            or "zero_weighted_reciprocal_density" in kconfig
            or "zero_weighted_line_density" in kconfig
        )
        # handle length generation first as this doesn't support any additional options
        if kconfig.get("length"):
            if explicit:
                raise ValueError(
                    "length option cannot be used with explicit k-point generation, "
                    "added_kpoints, or zero weighted k-points."
                )
            # If length is in kpoints settings use Kpoints.automatic
            return Kpoints.automatic(kconfig["length"])

        base_kpoints = None
        if kconfig.get("line_density"):
            # handle line density generation
            kpath = HighSymmKpath(structure, **kconfig.get("kpath_kwargs", {}))
            frac_k_points, k_points_labels = kpath.get_kpoints(
                line_density=kconfig["line_density"], coords_are_cartesian=False
            )
            base_kpoints = Kpoints(
                comment="Non SCF run along symmetry lines",
                style=Kpoints.supported_modes.Reciprocal,
                num_kpts=len(frac_k_points),
                kpts=frac_k_points,
                labels=k_points_labels,
                kpts_weights=[1] * len(frac_k_points),
            )
        elif kconfig.get("grid_density") or kconfig.get("reciprocal_density"):
            # handle regular weighted k-point grid generation
            if kconfig.get("grid_density"):
                base_kpoints = Kpoints.automatic_density(
                    structure, int(kconfig["grid_density"]), self.force_gamma
                )
            elif kconfig.get("reciprocal_density"):
                if (
                    bandgap == 0
                    and kconfig.get("reciprocal_density_metal")
                    and self.auto_metal_kpoints
                ):
                    density = kconfig["reciprocal_density_metal"]
                else:
                    density = kconfig["reciprocal_density"]
                base_kpoints = Kpoints.automatic_density_by_vol(
                    structure, density, self.force_gamma
                )
            if explicit:
                sga = SpacegroupAnalyzer(structure, symprec=self.symprec)
                mesh = sga.get_ir_reciprocal_mesh(base_kpoints.kpts[0])
                base_kpoints = Kpoints(
                    comment="Uniform grid",
                    style=Kpoints.supported_modes.Reciprocal,
                    num_kpts=len(mesh),
                    kpts=[i[0] for i in mesh],
                    kpts_weights=[i[1] for i in mesh],
                )
            else:
                # if not explicit that means no other options have been specified
                # so we can return the k-points as is
                return base_kpoints

        zero_weighted_kpoints = None
        if kconfig.get("zero_weighted_line_density"):
            # zero_weighted k-points along line mode path
            kpath = HighSymmKpath(structure)
            frac_k_points, k_points_labels = kpath.get_kpoints(
                line_density=kconfig["zero_weighted_line_density"],
                coords_are_cartesian=False,
            )
            zero_weighted_kpoints = Kpoints(
                comment="Hybrid run along symmetry lines",
                style=Kpoints.supported_modes.Reciprocal,
                num_kpts=len(frac_k_points),
                kpts=frac_k_points,
                labels=k_points_labels,
                kpts_weights=[0] * len(frac_k_points),
            )
        elif kconfig.get("zero_weighted_reciprocal_density"):
            zero_weighted_kpoints = Kpoints.automatic_density_by_vol(
                structure, kconfig["zero_weighted_reciprocal_density"], self.force_gamma
            )
            sga = SpacegroupAnalyzer(structure, symprec=self.symprec)
            mesh = sga.get_ir_reciprocal_mesh(zero_weighted_kpoints.kpts[0])
            zero_weighted_kpoints = Kpoints(
                comment="Uniform grid",
                style=Kpoints.supported_modes.Reciprocal,
                num_kpts=len(mesh),
                kpts=[i[0] for i in mesh],
                kpts_weights=[0 for i in mesh],
            )

        added_kpoints = None
        if kconfig.get("added_kpoints"):
            added_kpoints = Kpoints(
                comment="Specified k-points only",
                style=Kpoints.supported_modes.Reciprocal,
                num_kpts=len(kconfig.get("added_kpoints")),
                kpts=kconfig.get("added_kpoints"),
                labels=["user-defined"] * len(kconfig.get("added_kpoints")),
                kpts_weights=[0] * len(kconfig.get("added_kpoints")),
            )

        if base_kpoints and not (added_kpoints or zero_weighted_kpoints):
            return base_kpoints
        if added_kpoints and not (base_kpoints or zero_weighted_kpoints):
            return added_kpoints

        # do some sanity checking
        if "line_density" in kconfig and zero_weighted_kpoints:
            raise ValueError(
                "Cannot combined line_density and zero weighted k-points options"
            )
        if zero_weighted_kpoints and not base_kpoints:
            raise ValueError(
                "Zero weighted k-points must be used with reciprocal_density or "
                "grid_density options"
            )
        if not (base_kpoints or zero_weighted_kpoints or added_kpoints):
            raise ValueError(
                "Invalid k-point generation algo. Supported Keys are 'grid_density' "
                "for Kpoints.automatic_density generation, 'reciprocal_density' for "
                "KPoints.automatic_density_by_vol generation, 'length' for "
                "Kpoints.automatic generation, 'line_density' for line mode generation,"
                " 'added_kpoints' for specific k-points to include, "
                " 'zero_weighted_reciprocal_density' for a zero weighted uniform mesh,"
                " or 'zero_weighted_line_density' for a zero weighted line mode mesh."
            )

        return _combine_kpoints(base_kpoints, zero_weighted_kpoints, added_kpoints)

    def _kspacing(self, incar_updates: dict[str, Any]) -> float | None:
        """Get KSPACING value based on the config dict, updates and user settings."""
        key = "KSPACING"
        return self.user_incar_settings.get(
            key, incar_updates.get(key, self.config_dict["INCAR"].get(key))
        )


def get_magmoms(
    structure: Structure,
    magmoms: dict[str, float] = None,
    config_magmoms: dict[str, float] = None,
) -> list[float]:
    """Get the mamgoms using the following precedence.

    1. user incar settings
    2. magmoms in input struct
    3. spins in input struct
    4. job config dict
    5. set all magmoms to 0.6
    """
    magmoms = magmoms or {}
    config_magmoms = config_magmoms or {}
    mag = []
    msg = (
        "Co without an oxidation state is initialized as low spin by default in "
        "Atomate2. If this default behavior is not desired, please set the spin on the "
        "magmom on the site directly to ensure correct initialization."
    )
    for site in structure:
        specie = str(site.specie)
        if specie in magmoms:
            mag.append(magmoms.get(specie))
        elif hasattr(site, "magmom"):
            mag.append(site.magmom)
        elif hasattr(site.specie, "spin") and site.specie.spin is not None:
            mag.append(site.specie.spin)
        elif specie in config_magmoms:
            if site.specie.symbol == "Co" and config_magmoms[specie] <= 1.0:
                warnings.warn(msg, stacklevel=2)
            mag.append(config_magmoms.get(specie))
        else:
            if site.specie.symbol == "Co":
                warnings.warn(msg, stacklevel=2)
            mag.append(magmoms.get(site.specie.symbol, 0.6))
    return mag


def _get_u_param(
    lda_param: str, lda_config: dict[str, Any], structure: Structure
) -> list[float]:
    """Get U parameters."""
    comp = structure.composition
    elements = sorted((el for el in comp.elements if comp[el] > 0), key=lambda e: e.X)
    most_electroneg = elements[-1].symbol
    poscar = Poscar(structure)

    if hasattr(structure[0], lda_param.lower()):
        m = {site.specie.symbol: getattr(site, lda_param.lower()) for site in structure}
        return [m[sym] for sym in poscar.site_symbols]
    if isinstance(lda_config.get(most_electroneg, 0), dict):
        # lookup specific LDAU if specified for most_electroneg atom
        return [lda_config[most_electroneg].get(sym, 0) for sym in poscar.site_symbols]
    return [
        lda_config.get(sym, 0)
        if isinstance(lda_config.get(sym, 0), (float, int))
        else 0
        for sym in poscar.site_symbols
    ]


def _get_ediff(
    param: str, value: str | float, structure: Structure, incar_settings: dict[str, Any]
) -> float:
    """Get EDIFF."""
    if incar_settings.get("EDIFF") is None and param == "EDIFF_PER_ATOM":
        return float(value) * structure.num_sites
    return float(incar_settings["EDIFF"])


def _set_u_params(
    incar: Incar, incar_settings: dict[str, Any], structure: Structure
) -> None:
    """Modify INCAR for use with U parameters."""
    has_u = incar_settings.get("LDAU") and sum(incar["LDAUU"]) > 0

    if not has_u:
        ldau_keys = [key for key in incar if key.startswith("LDAU")]
        for key in ldau_keys:
            incar.pop(key, None)

    # Modify LMAXMIX if you have d or f electrons present. Note that if the user
    # explicitly sets LMAXMIX in settings it will override this logic (setdefault keeps
    # current value). Previously, this was only set if Hubbard U was enabled as per the
    # VASP manual but following an investigation it was determined that this would lead
    # to a significant difference between SCF -> NonSCF even without Hubbard U enabled.
    # Thanks to Andrew Rosen for investigating and reporting.
    blocks = [site.specie.block for site in structure]
    if "f" in blocks:  # contains f-electrons
        incar.setdefault("LMAXMIX", 6)
    elif "d" in blocks:  # contains d-electrons
        incar.setdefault("LMAXMIX", 4)


def _apply_incar_updates(
    incar: dict[str, Any], updates: dict[str, Any], skip: Sequence[str] = ()
) -> None:
    """
    Apply updates to an INCAR file.

    Parameters
    ----------
    incar
        An incar.
    updates
        Updates to apply.
    skip
        Keys to skip.
    """
    for key, val in updates.items():
        if key in skip:
            continue

        if val is None:
            incar.pop(key, None)
        else:
            incar[key] = val


def _remove_unused_incar_params(
    incar: dict[str, Any], skip: Sequence[str] = ()
) -> None:
    """
    Remove INCAR parameters that are not actively used by VASP.

    Parameters
    ----------
    incar
        An incar.
    skip
        Keys to skip.
    """
    # Turn off IBRION/ISIF/POTIM if NSW = 0
    opt_flags = ["EDIFFG", "IBRION", "ISIF", "POTIM"]
    if incar.get("NSW", 0) == 0:
        for opt_flag in opt_flags:
            if opt_flag not in skip:
                incar.pop(opt_flag, None)

    # Remove MAGMOMs if they aren't used
    if incar.get("ISPIN", 1) == 1 and "MAGMOM" not in skip:
        incar.pop("MAGMOM", None)

    # Turn off +U flags if +U is not even used
    ldau_flags = ("LDAUU", "LDAUJ", "LDAUL", "LDAUTYPE")
    if not incar.get("LDAU"):
        for ldau_flag in ldau_flags:
            if ldau_flag not in skip:
                incar.pop(ldau_flag, None)


def _combine_kpoints(*kpoints_objects: Kpoints) -> Kpoints:
    """Combine k-points files together."""
    labels, kpoints, weights = [], [], []

    recip_mode = Kpoints.supported_modes.Reciprocal
    for kpoints_object in filter(None, kpoints_objects):
        if kpoints_object.style != recip_mode:
            raise ValueError(
                f"Can only combine kpoints with style {recip_mode}, "
                f"got {kpoints_object.style}"
            )
        labels.append(kpoints_object.labels or [""] * len(kpoints_object.kpts))

        weights.append(kpoints_object.kpts_weights)
        kpoints.append(kpoints_object.kpts)

    labels = np.concatenate(labels).tolist()
    weights = np.concatenate(weights).tolist()
    kpoints = np.concatenate(kpoints)
    return Kpoints(
        comment="Combined k-points",
        style=recip_mode,
        num_kpts=len(kpoints),
        kpts=kpoints,
        labels=labels,
        kpts_weights=weights,
    )


def _get_ispin(vasprun: Vasprun | None, outcar: Outcar | None) -> int:
    """Get value of ISPIN depending on the magnetisation in the OUTCAR and vasprun."""
    if outcar is not None and outcar.magnetization is not None:
        # Turn off spin when magmom for every site is smaller than 0.02.
        site_magmom = np.array([i["tot"] for i in outcar.magnetization])
        return 2 if np.any(np.abs(site_magmom) > 0.02) else 1
    if vasprun is not None:
        return 2 if vasprun.is_spin else 1
    return 2


def _get_recommended_lreal(structure: Structure) -> str | bool:
    """Get recommended LREAL flag based on the structure."""
    return "Auto" if structure.num_sites > 16 else False


def _get_kspacing(bandgap: float, tol: float = 1e-4) -> float:
    """Get KSPACING based on a band gap."""
    if bandgap <= tol:  # metallic
        return 0.22

    rmin = max(1.5, 25.22 - 2.87 * bandgap)  # Eq. 25
    kspacing = 2 * np.pi * 1.0265 / (rmin - 1.0183)  # Eq. 29

    # cap kspacing at a max of 0.44, per internal benchmarking
    return min(kspacing, 0.44)


def _set_kspacing(
    incar: Incar,
    incar_settings: dict,
    user_incar_settings: dict,
    bandgap: float | None,
    kpoints: Kpoints | None,
) -> Incar:
    """
    Set KSPACING in an INCAR.

    if kpoints is not None then unset any KSPACING
    if kspacing set in user_incar_settings then use that
    if auto_kspacing then do that
    if kspacing is set in config use that.
    if from_prev is True, ISMEAR will be set according to the band gap.
    """
    if kpoints is not None:
        # unset KSPACING as we are using a KPOINTS file
        incar.pop("KSPACING", None)

        # Ensure adequate number of KPOINTS are present for the tetrahedron method
        # (ISMEAR=-5). If KSPACING is in the INCAR file the number of kpoints is not
        # known before calling VASP, but a warning is raised when the KSPACING value is
        # > 0.5 (2 reciprocal Angstrom). An error handler in Custodian is available to
        # correct overly large KSPACING values (small number of kpoints) if necessary.
        if np.prod(kpoints.kpts) < 4 and incar.get("ISMEAR", 0) == -5:
            incar["ISMEAR"] = 0

    elif "KSPACING" in user_incar_settings:
        incar["KSPACING"] = user_incar_settings["KSPACING"]

    elif incar_settings.get("KSPACING") and isinstance(bandgap, (int, float)):
        # will always default to 0.22 in first run as one
        # cannot be sure if one treats a metal or
        # semiconductor/insulator
        incar["KSPACING"] = _get_kspacing(bandgap)
        # This should default to ISMEAR=0 if band gap is not known (first computation)
        # if not from_prev:
        #     # be careful to not override user_incar_settings

    elif incar_settings.get("KSPACING"):
        incar["KSPACING"] = incar_settings["KSPACING"]

    return incar
