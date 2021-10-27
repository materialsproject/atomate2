"""Module defining base VASP input set and generator."""

import os
import warnings
from copy import deepcopy
from itertools import groupby
from math import pi
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from monty.io import zopen
from monty.serialization import loadfn
from pkg_resources import resource_filename
from pymatgen.core import Structure
from pymatgen.io.vasp import Incar, Kpoints, Outcar, Poscar, Potcar, Vasprun
from pymatgen.io.vasp.sets import (
    BadInputSetWarning,
    get_valid_magmom_struct,
    get_vasprun_outcar,
)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath

from atomate2.common.schemas.math import Vector3D
from atomate2.common.sets import InputSet, InputSetGenerator
from atomate2.settings import settings

_BASE_VASP_SET = loadfn(resource_filename("atomate2.vasp.sets", "BaseVaspSet.yaml"))

__all__ = ["VaspInputSet", "VaspInputSetGenerator"]


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
        potcar: Potcar,
        kpoints: Kpoints,
        optional_files: Optional[Dict] = None,
    ):
        self.incar = incar
        self.poscar = poscar
        self.potcar = potcar
        self.kpoints = kpoints
        self.optional_files = {} if optional_files is None else optional_files

    def write_input(
        self,
        directory: Union[str, Path],
        make_dir: bool = True,
        overwrite: bool = True,
        potcar_spec: bool = False,
    ):
        """
        Write VASP input files to a directory.

        Parameters
        ----------
        directory
            Directory to write input files to.
        make_dir
            Whether to create the directory if it does not already exist.
        overwrite
            Whether to overwrite an input file if it already exists.
        potcar_spec
            Instead of writing the POTCAR, write a "POTCAR.spec". This is intended to
            help sharing an input set with people who might not have a license to
            specific Potcar files. Given a "POTCAR.spec", the specific POTCAR file can
            be re-generated using pymatgen with the "generate_potcar" function in the
            pymatgen CLI.
        """
        directory = Path(directory)
        if make_dir and not directory.exists():
            os.makedirs(directory)

        inputs = {
            "INCAR": self.incar,
            "POTCAR": self.potcar,
            "KPOINTS": self.kpoints,
            "POSCAR": self.poscar,
        }
        inputs.update(self.optional_files)

        if potcar_spec:
            inputs.pop("POTCAR")
            inputs["POTCAR.spec"] = "\n".join(self.potcar.symbols)

        for k, v in inputs.items():
            if v is not None and (overwrite or not (directory / k).exists()):
                with zopen(directory / k, "wt") as f:
                    f.write(v.__str__())
            elif not overwrite and (directory / k).exists():
                raise FileExistsError(f"{directory / k} already exists.")

    @staticmethod
    def from_directory(directory: Union[str, Path], optional_files: Dict = None):
        """
        Load a set of VASP inputs from a directory.

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
        """
        Whether the input set is valid.

        Returns
        -------
        bool
            Whether the input set is valid.
        """
        if self.incar.get("KSPACING", 0) > 0.5 and self.incar.get("ISMEAR", 0) == -5:
            warnings.warn(
                "Large KSPACING value detected with ISMEAR=-5. Ensure that VASP "
                "generates enough KPOINTS, lower KSPACING, or set ISMEAR=0",
                BadInputSetWarning,
            )

        if (
            all(k.is_metal for k in self.poscar.structure.composition.keys())
            and self.incar.get("NSW", 0) > 0
            and self.incar.get("ISMEAR", 1) < 1
        ):
            warnings.warn(
                "Relaxation of likely metal with ISMEAR < 1 detected. Please see VASP "
                "recommendations on ISMEAR for metals.",
                BadInputSetWarning,
            )

        return True


class VaspInputSetGenerator(InputSetGenerator):
    """
    A class to generate VASP input sets.

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
        If None is given, that key is unset. For example, {"ENCUT": None} will remove
        ENCUT from the incar settings.
    user_kpoints_settings
        Allow user to override kpoints setting by supplying a dict. E.g.,
        ``{"reciprocal_density": 1000}``. User can also supply a Kpoints object.
    user_potcar_settings
        Allow user to override POTCARs. E.g., {"Gd": "Gd_3"}.
    user_potcar_functional
        Functional to use. Default is to use the functional in the config dictionary.
        Valid values: "PBE", "PBE_52", "PBE_54", "LDA", "LDA_52", "LDA_54", "PW91",
        "LDA_US", "PW91_US".
    auto_kspacing
        Whether to set the kspacing value based on the band gap. Note, this only works
        when generating the input set from a previous VASP directory.
    constrain_total_magmom
        Whether to constrain the total magmom (NUPDOWN in INCAR) to be the sum of the
        expected MAGMOM for all species.
    validate_magmom
        Ensure that missing magmom values are filled in with the default value of 1.0.
    use_structure_charge
        If set to True, then the overall charge of the structure (``structure.charge``)
        is  used to set NELECT.
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
    """

    def __init__(
        self,
        user_incar_settings: Dict = None,
        user_kpoints_settings: Union[Dict, Kpoints] = None,
        user_potcar_settings: Dict = None,
        user_potcar_functional: str = None,
        auto_kspacing: bool = True,
        constrain_total_magmom: bool = False,
        validate_magmom: bool = True,
        use_structure_charge: bool = False,
        sort_structure: bool = True,
        force_gamma: bool = True,
        symprec: float = settings.SYMPREC,
        vdw: str = None,
        config_dict: Dict = None,
    ):
        if config_dict is None:
            config_dict = _BASE_VASP_SET

        self._config_dict = deepcopy(config_dict)
        self.user_incar_settings = user_incar_settings or {}
        self.user_kpoints_settings: Union[Dict, Kpoints] = user_kpoints_settings or {}
        self.user_potcar_settings = user_potcar_settings
        self.user_potcar_functional = user_potcar_functional
        self.auto_kspacing = auto_kspacing
        self.constrain_total_magmom = constrain_total_magmom
        self.validate_magmom = validate_magmom
        self.sort_structure = sort_structure
        self.force_gamma = force_gamma
        self.symprec = symprec  # used in k-point generation
        self.vdw = None if vdw is None else vdw.lower()
        self.use_structure_charge = use_structure_charge

        if self.user_incar_settings.get("KSPACING") and self.user_kpoints_settings:
            warnings.warn(
                "You have specified KSPACING and also supplied kpoints settings. "
                "KSPACING only has effect when there is no KPOINTS file. Since both "
                "settings were given, pymatgen will generate a KPOINTS file and ignore "
                "KSPACING. Remove the `user_kpoints_settings` argument to enable "
                "KSPACING.",
                BadInputSetWarning,
            )

        if self.vdw:
            from pymatgen.io.vasp.sets import MODULE_DIR as PMG_SET_DIR

            vdw_par = loadfn(PMG_SET_DIR / "vdW_parameters.yaml")
            if self.vdw not in vdw_par:
                raise KeyError(
                    "Invalid or unsupported van-der-Waals functional. Supported "
                    f"functionals are {vdw_par.keys()}"
                )
            self._config_dict["INCAR"].update(vdw_par[self.vdw])

        # read the POTCAR_FUNCTIONAL from the .yaml
        self.potcar_functional = self._config_dict.get("POTCAR_FUNCTIONAL", "PS")

        # warn if a user is overriding POTCAR_FUNCTIONAL
        if user_potcar_functional and user_potcar_functional != self.potcar_functional:
            warnings.warn(
                "Overriding the POTCAR functional is generally not recommended "
                "as it can significantly affect the results of calculations and "
                "compatibility with other calculations done with the same input set. "
                "Note that some POTCAR symbols specified in the configuration file may "
                "not be available in the selected functional.",
                BadInputSetWarning,
            )
            self.potcar_functional = user_potcar_functional

        if self.user_potcar_settings:
            warnings.warn(
                "Overriding POTCARs is generally not recommended as it can "
                "significantly affect the results of calculations and compatibility "
                "with other calculations done with the same input set. In many "
                "instances, it is better to write a subclass of a desired input set and"
                " override the POTCAR in the subclass to be explicit on the "
                "differences.",
                BadInputSetWarning,
            )
            for k, v in self.user_potcar_settings.items():
                self._config_dict["POTCAR"][k] = v

    def get_input_set(  # type: ignore
        self, structure: Structure = None, prev_dir: Union[str, Path] = None
    ) -> VaspInputSet:
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
        updates = self.get_incar_updates(
            structure,
            prev_incar=prev_incar,
            bandgap=bandgap,
            vasprun=vasprun,
            outcar=outcar,
        )
        kpoints = self._get_kpoints(structure)
        incar = self._get_incar(
            structure, kpoints, prev_incar, updates, bandgap=bandgap
        )
        return VaspInputSet(
            incar=incar,
            kpoints=kpoints,
            poscar=Poscar(structure),
            potcar=self._get_potcar(structure),
        )

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = 0.0,
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
        raise NotImplementedError

    def get_nelect(self, structure: Structure) -> float:
        """
        Get the default number of electrons for a given structure.

        Parameters
        ----------
        structure
            A structure.

        Returns
        -------
        float
            Number of electrons for the structure.
        """
        potcar = self._get_potcar(structure)
        nelec = {p.element: p.nelectrons for p in potcar}
        comp = structure.composition.element_composition
        nelect = sum([num_atoms * nelec[str(el)] for el, num_atoms in comp.items()])

        if self.use_structure_charge:
            return nelect - self.structure.charge

        return nelect

    def _get_previous(
        self, structure: Structure = None, prev_dir: Union[str, Path] = None
    ):
        """Load previous calculation outputs and decide which structure to use."""
        if structure is None and prev_dir is None:
            raise ValueError("Either structure or prev_dir must be set.")

        prev_incar = {}
        prev_structure = None
        vasprun = None
        outcar = None
        bandgap = 0
        if prev_dir:
            vasprun, outcar = get_vasprun_outcar(prev_dir)
            bs = vasprun.get_band_structure()
            prev_incar = vasprun.incar
            prev_structure = vasprun.final_structure
            bandgap = 0 if bs.is_metal() else bs.get_band_gap()["energy"]

        structure = structure if structure is not None else prev_structure
        structure = self._get_structure(structure)
        return structure, prev_incar, bandgap, vasprun, outcar

    def _get_structure(self, structure):
        """Get the standardized structure."""
        if self.sort_structure:
            structure = structure.get_sorted_structure()

        if self.validate_magmom:
            get_valid_magmom_struct(structure, spin_mode="auto", inplace=True)
        return structure

    def _get_potcar(self, structure):
        """Get the POTCAR."""
        elements = [a[0] for a in groupby([s.specie.symbol for s in structure])]
        potcar_symbols = [self._config_dict["POTCAR"].get(el, el) for el in elements]
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
                )
        return potcar

    def _get_incar(
        self,
        structure,
        kpoints: Kpoints,
        previous_incar: Dict = None,
        incar_updates: Dict = None,
        bandgap: float = 0.0,
    ):
        """Get the INCAR."""
        previous_incar = {} if previous_incar is None else previous_incar
        incar_updates = {} if incar_updates is None else incar_updates
        incar_settings = dict(self._config_dict["INCAR"])

        # apply user incar settings to SETTINGS not to INCAR
        _apply_incar_updates(incar_settings, self.user_incar_settings)

        # generate incar
        incar = Incar()
        for k, v in incar_settings.items():
            if k == "MAGMOM":
                incar[k] = _get_magmoms(v, structure)
            elif k in ("LDAUU", "LDAUJ", "LDAUL") and incar_settings.get("LDAU", False):
                incar[k] = _get_u_param(k, v, structure)
            elif k.startswith("EDIFF") and k != "EDIFFG":
                incar["EDIFF"] = _get_ediff(k, v, structure, incar_settings)
            else:
                incar[k] = v
        _set_u_params(incar, incar_settings, structure)

        # apply previous incar settings, be careful not to override user_incar_settings
        # also skip LDAU/MAGMOM as structure may have changed.
        skip = list(self.user_incar_settings.keys())
        skip += ["MAGMOM", "NUPDOWN", "LDAUU", "LDAUL", "LDAUJ", "LMAXMIX"]
        _apply_incar_updates(incar, previous_incar, skip=skip)

        if self.constrain_total_magmom:
            nupdown = sum([mag if abs(mag) > 0.6 else 0 for mag in incar["MAGMOM"]])
            incar["NUPDOWN"] = nupdown

        if self.use_structure_charge:
            incar["NELECT"] = self.get_nelect(structure)

        # handle kspacing
        _set_kspacing(
            incar,
            incar_settings,
            self.user_incar_settings,
            self.auto_kspacing,
            bandgap,
            kpoints,
        )

        # apply specified updates, be careful not to override user_incar_settings
        _apply_incar_updates(incar, incar_updates, skip=self.user_incar_settings.keys())

        return incar

    def _get_kpoints(
        self,
        structure: Structure,
        added_kpoints: List[Vector3D] = None,
    ) -> Union[Kpoints, None]:
        """Get the kpoints file."""
        # Return None if KSPACING is present in the INCAR and no other user k-points
        # settings have been specified, because this will cause VASP to generate the
        # kpoints automatically
        if not self.user_kpoints_settings and (
            self.user_incar_settings.get("KSPACING", 0) is not None
            or self._config_dict["INCAR"].get("KSPACING")
        ):
            return None

        # use user setting if set otherwise default to base config settings
        if self.user_incar_settings != {}:
            kconfig = deepcopy(self.user_kpoints_settings)
        else:
            kconfig = deepcopy(self._config_dict.get("KPOINTS"))

        if isinstance(kconfig, Kpoints):
            return kconfig

        added_kpoints = [] if added_kpoints is None else added_kpoints
        if len(added_kpoints) > 0 and kconfig.get("length"):
            raise ValueError(
                "added_kpoints only compatible with grid_density, reciprocal_density "
                "or line_density generation schemes."
            )
        elif len(added_kpoints) > 0:
            # force explicit k-point generation if adding additional k-points.
            kconfig["explicit"] = True

        # If length is in kpoints settings use Kpoints.automatic
        if kconfig.get("length"):
            return Kpoints.automatic(kconfig["length"])

        kpoints = None
        # If grid_density is in kpoints settings use automatic_density
        if kconfig.get("grid_density"):
            kpoints = Kpoints.automatic_density(
                structure, int(kconfig["grid_density"]), self.force_gamma
            )

        # If reciprocal_density is in kpoints settings use automatic_density_by_vol
        if kconfig.get("reciprocal_density"):
            kpoints = Kpoints.automatic_density_by_vol(
                structure, kconfig["reciprocal_density"], self.force_gamma
            )

        # if reciprocal_density or grid_density is set with line_density we have a
        # weighted uniform mesh and zero-weighted line-mode mesh
        if kpoints is not None and kconfig.get("line_density", False):
            kconfig["explicit"] = True

        if kpoints is not None and kconfig.get("explicit"):
            sga = SpacegroupAnalyzer(structure, symprec=self.symprec)
            mesh = sga.get_ir_reciprocal_mesh(kpoints.kpts[0])
            kpoints = Kpoints(
                comment="Uniform grid",
                style=Kpoints.supported_modes.Reciprocal,
                num_kpts=len(mesh),
                kpts=[i[0] for i in mesh],
                kpts_weights=[i[1] for i in mesh],
            )

        # If line_density is in kpoints settings generate line mode band structure
        if kconfig.get("line_density"):
            kpath = HighSymmKpath(structure)
            frac_k_points, k_points_labels = kpath.get_kpoints(
                line_density=kconfig["line_density"], coords_are_cartesian=False
            )
            if kpoints is None:
                kpoints = Kpoints(
                    comment="Non SCF run along symmetry lines",
                    style=Kpoints.supported_modes.Reciprocal,
                    num_kpts=len(frac_k_points),
                    kpts=frac_k_points,
                    labels=k_points_labels,
                    kpts_weights=[1] * len(frac_k_points),
                )
            else:
                # hybrid zero_weighted k-points
                if kpoints.labels is None:
                    labels = [""] * len(kpoints.kpts) + k_points_labels
                else:
                    labels = kpoints.labels + k_points_labels

                weights = kpoints.kpts_weights + [0] * len(frac_k_points)
                kpts = list(kpoints.kpts) + frac_k_points
                kpoints = Kpoints(
                    comment="Hybrid run along symmetry lines",
                    style=Kpoints.supported_modes.Reciprocal,
                    num_kpts=len(kpts),
                    kpts=kpts,
                    labels=labels,
                    kpts_weights=weights,
                )

        # finally, add any additional k-points if specified
        if added_kpoints and kpoints is None:
            kpoints = Kpoints(
                comment="Specified k-points only",
                style=Kpoints.supported_modes.Reciprocal,
                num_kpts=len(added_kpoints),
                kpts=added_kpoints,
                labels=["user-defined"] * len(added_kpoints),
                kpts_weights=[1] * len(added_kpoints),
            )
        elif added_kpoints:
            if kpoints.labels is None:
                labels = [""] * len(kpoints.kpts)
            else:
                labels = kpoints.labels
            labels += ["user-defined"] * len(added_kpoints)
            weights = kpoints.kpts_weights + [0] * len(added_kpoints)
            kpts = list(kpoints.kpts) + added_kpoints
            comment = "High symmetry run" if kconfig.get("line_density") else "Uniform"
            kpoints = Kpoints(
                comment=comment,
                style=Kpoints.supported_modes.Reciprocal,
                num_kpts=len(kpts),
                kpts=kpts,
                labels=labels,
                kpts_weights=weights,
            )

        if kpoints is None:
            raise ValueError(
                "Invalid k-point generation algo. Supported Keys are 'grid_density' "
                "for Kpoints.automatic_density generation, 'reciprocal_density' for "
                "KPoints.automatic_density_by_vol generation, 'length' for "
                "Kpoints.automatic generation, 'line_density' for line mode generation."
            )
        return kpoints


def _get_kspacing(bandgap: float) -> float:
    """Get KSPACING based on a band gap."""
    if bandgap == 0:
        return 0.22

    rmin = 25.22 - 2.87 * bandgap  # Eq. 25
    kspacing = 2 * pi * 1.0265 / (rmin - 1.0183)  # Eq. 29

    # cap kspacing at a max of 0.44, per internal benchmarking
    return min(kspacing, 0.44)


def _get_magmoms(magmoms, structure):
    """Get the mamgoms."""
    mag = []
    for site in structure:
        if hasattr(site, "magmom"):
            mag.append(site.magmom)
        elif hasattr(site.specie, "spin"):
            mag.append(site.specie.spin)
        elif str(site.specie) in magmoms:
            if site.specie.symbol == "Co":
                warnings.warn(
                    "Co without oxidation state is initialized low spin by default. If "
                    "this is not desired, please set the spin on the magmom on the "
                    "site directly to ensure correct initialization."
                )
            mag.append(magmoms.get(str(site.specie)))
        else:
            if site.specie.symbol == "Co":
                warnings.warn(
                    "Co without oxidation state is initialized low spin by default. If "
                    "this is not desired, please set the spin on the magmom on the "
                    "site directly to ensure correct initialization."
                )
            mag.append(magmoms.get(site.specie.symbol, 0.6))
    return mag


def _get_u_param(lda_param, lda_config, structure):
    """Get U parameters."""
    comp = structure.composition
    elements = sorted([el for el in comp.elements if comp[el] > 0], key=lambda e: e.X)
    most_electroneg = elements[-1].symbol
    poscar = Poscar(structure)

    if hasattr(structure[0], lda_param.lower()):
        m = {site.specie.symbol: getattr(site, lda_param.lower()) for site in structure}
        return [m[sym] for sym in poscar.site_symbols]
    elif isinstance(lda_config.get(most_electroneg, 0), dict):
        # lookup specific LDAU if specified for most_electroneg atom
        return [lda_config[most_electroneg].get(sym, 0) for sym in poscar.site_symbols]
    else:
        return [
            lda_config.get(sym, 0)
            if isinstance(lda_config.get(sym, 0), (float, int))
            else 0
            for sym in poscar.site_symbols
        ]


def _get_ediff(param, value, structure, incar_settings):
    """Get EDIFF."""
    if "EDIFF" not in incar_settings and param == "EDIFF_PER_ATOM":
        return float(value) * structure.num_sites
    else:
        return float(incar_settings["EDIFF"])


def _set_u_params(incar, incar_settings, structure):
    """Modify INCAR for use with U parameters."""
    has_u = incar_settings.get("LDAU", False) and sum(incar["LDAUU"]) > 0

    if has_u:
        # modify LMAXMIX if LSDA+U and you have d or f electrons
        # note that if the user explicitly sets LMAXMIX in settings it will
        # override this logic.
        if "LMAXMIX" not in incar_settings.keys():
            # contains f-electrons
            if any(el.Z > 56 for el in structure.composition):
                incar["LMAXMIX"] = 6
            # contains d-electrons
            elif any(el.Z > 20 for el in structure.composition):
                incar["LMAXMIX"] = 4

    else:
        for key in list(incar.keys()):
            if key.startswith("LDAU"):
                del incar[key]


def _apply_incar_updates(incar, updates, skip=None):
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
    skip = () if skip is None else skip
    for k, v in updates.items():
        if k in skip:
            continue

        if v is None:
            try:
                del incar[k]
            except KeyError:
                incar[k] = v
        else:
            incar[k] = v


def _set_kspacing(
    incar, incar_settings, user_incar_settings, auto_kspacing, bandgap, kpoints
):
    """
    Set KSPACING in an INCAR.

    if kpoints is not None then unset any KSPACING
    if kspacing set in user_incar_settings then use that
    if auto_kspacing then do that
    if kspacing is set in config use that.
    """
    if kpoints is not None:
        # unset KSPACING as we are using a KPOINTS file
        incar.pop("KSPACING", None)

        # Ensure adequate number of KPOINTS are present for the tetrahedron method
        # (ISMEAR=-5). If KSPACING is in the INCAR file the number of kpoints is not
        # known before calling VASP, but a warning is raised when the KSPACING value is
        # > 0.5 (2 reciprocal Angstrom). An error handler in Custodian is available to
        # correct overly large KSPACING values (small number of kpoints) if necessary.
        if np.product(kpoints.kpts) < 4 and incar.get("ISMEAR", 0) == -5:
            incar["ISMEAR"] = 0

    elif "KSPACING" in user_incar_settings:
        incar["KSPACING"] = user_incar_settings["KSPACING"]
    elif incar_settings.get("KSPACING") and auto_kspacing:
        incar["KSPACING"] = _get_kspacing(bandgap)

        # be careful to not override user_incar_settings
        if bandgap == 0:
            incar["SIGMA"] = user_incar_settings.get("SIGMA", 0.2)
            incar["ISMEAR"] = user_incar_settings.get("ISMEAR", 2)
        else:
            incar["SIGMA"] = user_incar_settings.get("SIGMA", 0.05)
            incar["ISMEAR"] = user_incar_settings.get("ISMEAR", -5)
    elif incar_settings.get("KSPACING"):
        incar["KSPACING"] = incar_settings["KSPACING"]

    return incar
