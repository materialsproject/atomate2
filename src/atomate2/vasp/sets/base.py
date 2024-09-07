"""Module defining base VASP input set and generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pymatgen.io.vasp.sets import MPRelaxSet, MPScanRelaxSet, VaspInputSet

from atomate2 import SETTINGS

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Kpoints
    from pymatgen.io.vasp.sets import UserPotcarFunctional

_BASE_VASP_SET = {**MPScanRelaxSet()._config_dict, "KPOINTS": {}}  # noqa: SLF001
_ATOMATE2_BASE_VASP_SET_UPDATES = {
    "INCAR": {
        "ALGO": "Fast",
        "GGA": "PS",
        "LREAL": False,
        "KSPACING": None,
        "METAGGA": None,
        **{
            k: v
            for k, v in MPRelaxSet()._config_dict["INCAR"].items()  # noqa: SLF001
            if k.startswith("LDAU")
        },
    },
    "KPOINTS": {"reciprocal_density": 64, "reciprocal_density_metal": 200},
    "POTCAR": {
        "Be": "Be",
        "Bi": "Bi_d",
        "Cu": "Cu",
        "Eu": "Eu_2",
        "Fe": "Fe",
        "Gd": "Gd_3",
        "Mg": "Mg",
        "Mo": "Mo_sv",
        "Nb": "Nb_sv",
        "Ni": "Ni",
        "Os": "Os",
        "Re": "Re",
        "Ti": "Ti_sv",
        "V": "V_sv",
    },
}
for k, updates in _ATOMATE2_BASE_VASP_SET_UPDATES.items():
    for g, v in updates.items():
        if v is None:
            _BASE_VASP_SET[k].pop(g)
        else:
            _BASE_VASP_SET[k][g] = v


@dataclass
class VaspInputGenerator(VaspInputSet):
    """Base atomate2 implementation of a VASP input set.

    Base class representing a set of VASP input parameters with a structure
    supplied as init parameters and initialized from a dict of settings.
    This allows arbitrary settings to be input. In general,
    this is rarely used directly unless there is a source of settings in yaml
    format (e.g., from a REST interface). It is typically used by other
    VaspInputSets for initialization.

    Special consideration should be paid to the way the MAGMOM initialization
    for the INCAR is done. The initialization differs depending on the type of
    structure and the configuration settings. The order in which the magmom is
    determined is as follows:

    1. If the site is specified in user_incar_settings, use that setting.
    2. If the site itself has a magmom setting (i.e. site.properties["magmom"] = float),
        that is used. This can be set with structure.add_site_property().
    3. If the species of the site has a spin setting, that is used. This can be set
        with structure.add_spin_by_element().
    4. If the species itself has a particular setting in the config file, that
       is used, e.g. Mn3+ may have a different magmom than Mn4+.
    5. Lastly, the element symbol itself is checked in the config file. If
       there are no settings, a default value of 0.6 is used.

    Args:
        structure (Structure): The Structure to create inputs for. If None, the input
            set is initialized without a Structure but one must be set separately before
            the inputs are generated.
        config_dict (dict): The config dictionary to use.
        files_to_transfer (dict): A dictionary of {filename: filepath}. This allows the
            transfer of files from a previous calculation.
        user_incar_settings (dict): User INCAR settings. This allows a user to override
            INCAR settings, e.g. setting a different MAGMOM for various elements or
            species. Note that in the new scheme, ediff_per_atom and hubbard_u are no
            longer args. Instead, the CONFIG supports EDIFF_PER_ATOM and EDIFF keys.
            The former scales with # of atoms, the latter does not. If both are present,
            EDIFF is preferred. To force such settings, just supply
            user_incar_settings={"EDIFF": 1e-5, "LDAU": False} for example. The keys
            'LDAUU', 'LDAUJ', 'LDAUL' are special cases since pymatgen defines different
            values depending on what anions are present in the structure, so these keys
            can be defined in one of two ways, e.g. either {"LDAUU":{"O":{"Fe":5}}} to
            set LDAUU for Fe to 5 in an oxide, or {"LDAUU":{"Fe":5}} to set LDAUU to 5
            regardless of the input structure. If a None value is given, that key is
            unset. For example, {"ENCUT": None} will remove ENCUT from the
            incar settings. Finally, KSPACING is a special setting and can be set to
            "auto" in which the KSPACING is set automatically based on the band gap.
        user_kpoints_settings (dict or .Kpoints): Allow user to override kpoints setting
            by supplying a dict. e.g. {"reciprocal_density": 1000}. User can also
            supply Kpoints object.
        user_potcar_settings (dict): Allow user to override POTCARs. e.g. {"Gd":
            "Gd_3"}. This is generally not recommended.
        constrain_total_magmom (bool): Whether to constrain the total magmom (NUPDOWN in
            INCAR) to be the sum of the expected MAGMOM for all species.
        sort_structure (bool): Whether to sort the structure (using the default sort
            order of electronegativity) before generating input files. Defaults to True,
            the behavior you would want most of the time. This ensures that similar
            atomic species are grouped together.
        user_potcar_functional (str): Functional to use. Default (None) is to use the
            functional in the config dictionary. Valid values: "PBE", "PBE_52",
            "PBE_54", "LDA", "LDA_52", "LDA_54", "PW91", "LDA_US", "PW91_US".
        force_gamma (bool): Force gamma centered kpoint generation.
            Default (False) is to use the Automatic Density kpoint scheme, which
            will use the Gamma centered generation scheme for hexagonal cells,
            and Monkhorst-Pack otherwise.
        reduce_structure (None/str): Before generating the input files, generate the
            reduced structure. Default (None), does not alter the structure. Valid
            values: None, "niggli", "LLL".
        vdw: Adds default parameters for van-der-Waals functionals supported by VASP to
            INCAR. Supported functionals are: DFT-D2, undamped DFT-D3, DFT-D3 with
            Becke-Jonson damping, Tkatchenko-Scheffler, Tkatchenko-Scheffler with
            iterative Hirshfeld partitioning, MBD@rSC, dDsC, Dion's vdW-DF, DF2, optPBE,
            optB88, optB86b and rVV10.
        use_structure_charge (bool): If set to True, then the overall charge of the
            structure (structure.charge) is used to set the NELECT variable in the
            INCAR. Default is False.
        standardize (float): Whether to standardize to a primitive standard cell.
            Defaults to False.
        sym_prec (float): Tolerance for symmetry finding.
        international_monoclinic (bool): Whether to use international convention (vs
            Curtarolo) for monoclinic. Defaults True.
        validate_magmom (bool): Ensure that the missing magmom values are filled in with
            the VASP default value of 1.0.
        inherit_incar (bool or list[str]): Whether to inherit INCAR settings
            from a previous calculation. This might be useful to port Custodian
            fixes to child jobs but can also be dangerous, e.g., when switching
            from GGA to meta-GGA or relax to static jobs.
            Defaults to True.
            Can also be a list of strings to specify which parameters are inherited.
        auto_kspacing (bool): If true, determines the value of KSPACING from the bandgap
            of a previous calculation.
        auto_ismear (bool): If true, the values for ISMEAR and SIGMA will be set
            automatically depending on the bandgap of the system. If the bandgap is not
            known (e.g., there is no previous VASP directory) then ISMEAR=0 and
            SIGMA=0.2; if the bandgap is zero (a metallic system) then ISMEAR=2 and
            SIGMA=0.2; if the system is an insulator, then ISMEAR=-5 (tetrahedron
            smearing). Note, this only works when generating the input set from a
            previous VASP directory.
        auto_ispin (bool) = False:
            If generating input set from a previous calculation, this controls whether
            to disable magnetisation (ISPIN = 1) if the absolute value of all magnetic
            moments are less than 0.02.
        auto_lreal (bool) = False:
            If True, automatically use the VASP recommended LREAL based on cell size.
        auto_metal_kpoints
            If true and the system is metallic, try and use ``reciprocal_density_metal``
            instead of ``reciprocal_density`` for metallic systems.
            Note, this only works if the bandgap is not None.
        bandgap_tol (float): Tolerance for determining if a system is metallic when
            KSPACING is set to "auto". If the bandgap is less than this value, the
            system is considered metallic. Defaults to 1e-4 (eV).
        bandgap (float): Used for determining KSPACING if KSPACING == "auto" or
            ISMEAR if auto_ismear == True. Set automatically when using from_prev_calc.
        prev_incar (str or dict): Previous INCAR used for setting parent INCAR when
            inherit_incar == True. Set automatically when using from_prev_calc.
        prev_kpoints (str or .Kpoints): Previous Kpoints. Set automatically when using
            from_prev_calc.
    """

    structure: Structure | None = None
    config_dict: dict = field(default_factory=lambda: _BASE_VASP_SET)
    files_to_transfer: dict = field(default_factory=dict)
    user_incar_settings: dict = field(default_factory=dict)
    user_kpoints_settings: dict = field(default_factory=dict)
    user_potcar_settings: dict = field(default_factory=dict)
    constrain_total_magmom: bool = False
    sort_structure: bool = True
    user_potcar_functional: UserPotcarFunctional = None
    force_gamma: bool = True
    reduce_structure: Literal["niggli", "LLL"] | None = None
    vdw: str | None = None
    use_structure_charge: bool = False
    standardize: bool = False
    sym_prec: float = SETTINGS.SYMPREC
    international_monoclinic: bool = True
    validate_magmom: bool = True
    inherit_incar: bool | list[str] = SETTINGS.VASP_INHERIT_INCAR
    auto_ismear: bool = True
    auto_ispin: bool = False
    auto_lreal: bool = False
    auto_metal_kpoints: bool = True
    auto_kspacing: bool = False
    bandgap_tol: float = SETTINGS.BANDGAP_TOL
    bandgap: float | None = None
    prev_incar: str | dict | None = None
    prev_kpoints: str | Kpoints | None = None
