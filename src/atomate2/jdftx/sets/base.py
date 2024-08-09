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
from atomate2.jdftx.io.JDFTXInfile import JDFTXInfile #TODO update this to the pymatgen module
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath

from atomate2 import SETTINGS
from atomate2.jdftx.io.inputs import JdftxInput

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Structure

_BASE_JDFTX_SET = loadfn(get_mod_path("atomate2.jdftx.sets") / "BaseJdftxSet.yaml")


class JdftxInputSet(InputSet):
    """
    A class to represent a JDFTx input file as a JDFTx InputSet.

    Parameters
    ----------
    jdftxinput
        A JdftxInput object
    """

    def __init__( 
        self,
        jdftxinput: JdftxInput
    ) -> None:
        self.jdftxinput = jdftxinput


    def write_input(
        self,
        directory: str | Path,
        make_dir: bool = True,
        overwrite: bool = True,
    ) -> None:
        """Write JDFTx input file to a directory.

        Parameters
        ----------
        directory
            Directory to write input files to.
        make_dir
            Whether to create the directory if it does not already exist.
        overwrite
            Whether to overwrite an input file if it already exists.
        """
        infile = "inputs.in"
        directory = Path(directory)
        if make_dir:
            os.makedirs(directory, exist_ok=True)

        if not overwrite and (directory / infile).exists():
            raise FileExistsError(f"{directory / infile} already exists.")

        self.jdftxinput.write_file(filename=(directory / infile))

    @staticmethod
    def from_directory(
        directory: str | Path, optional_files: dict = None
    ) -> JdftxInputSet:
        """Load a set of JDFTx inputs from a directory.

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
class JdftxInputGenerator(InputGenerator):
    """
    A class to generate JDFTx input sets.

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
    # copy _BASE_JDFTX_SET to ensure each class instance has its own copy
    # otherwise in-place changes can affect other instances
    config_dict: dict = field(default_factory=lambda: _BASE_JDFTX_SET)

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

    def get_input_set(
        self,
        structure: Structure = None,
        prev_dir: str | Path = None,
    ) -> JdftxInputSet:
        """Get a JDFTx input set.

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
        JdftxInputSet
            A JDFTx input set.
        """
        # _get_previous will load in default values and structure from specified
        # directory. If prev_dir isn't specified, it'll return none
        if prev_dir != None:
            structure, prev_inputs = self._get_previous(
                structure, prev_dir
            )
        # prev_incar = prev_incar if self.inherit_incar else {} # TODO do we want an inherit_incar bool equivalent
        # input_updates = self.get_incar_updates(**kwds)
        # kspacing = self._kspacing(incar_updates)
        # kpoints = self._get_kpoints(structure, kpoints_updates, kspacing)
        jdftinputs = self._get_jdftxinputs(
            structure,
            # kpoints,
            # prev_incar,
        )
        return JdftxInputSet(
            jdftxinput=jdftinputs # TODO make the inputs object above and pass it here
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

    def _get_previous( # TODO adapt this for JDFTx
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

        return structure, prev_inputs

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

    def _get_jdftxinputs(
        self,
        structure: Structure=None,
        kpoints: Kpoints=None,
        incar_updates: dict=None, # ignore this for now
    ) -> JDFTXInfile:
        """Get the JDFTx input file object"""
        default_inputs = dict(self.config_dict)

        # generate incar
        print(default_inputs)


        jdftxinputs = JDFTXInfile.from_dict(default_inputs)

        return jdftxinputs

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
