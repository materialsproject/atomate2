"""
Module defining base SIESTA input set and generator.

class BasisSetsAndProjectors

Based on User's Guide Siesta 5.4.0
Section: 6.3 Basis set and KB projectors
"""

# Metadata

__all__ = ["BasisSetsAndProjectors"]

import logging
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from typing import Any

from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.dataclass.units import parse_energy, parse_length
from atomate2.siesta.utils.common import console
from atomate2.siesta.utils.verbosity import VerbosityLevel

logger = logging.getLogger(__name__)


@dataclass
class BasisSetsAndProjectors(FDFDataclass):
    """
    Data class to manage basis sets and KB projectors for SIESTA input.

    Currently supported (Option 1):
        perform_siesta_default_basis = True and user specifies these four flags:
            (a) pao_basis_type
            (b) pao_basissize
            (c) pao_split_norm
            (d) pao_energy_shift

    Future enhancements (not yet implemented):

    (Option 2) Custom PAO.Basis blocks:
        Allow perform_siesta_default_basis = False with user-defined basis
        specifications:

        Example format:
            %block PAO.Basis
                Na.1 3 # Species label, number of l-shells
                n=2 0 1 # n, l, Nzeta
                2.609
                1.000
                n=3 0 2 # n, l, Nzeta
                8.808 6.635
                1.000 1.000
                ...
            %endblock PAO.Basis

        Implementation requires: Basis block parsing and validation
        Priority: Medium (advanced users feature)

    (Option 3) NetCDF basis format:
        Support user-defined basis sets in NetCDF (.nc) format for pre-optimized
        basis sets from external tools.

        Implementation requires: NetCDF file handling and SIESTA integration
        Priority: Low (niche use case)
    """

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = (
        VerbosityLevel.INFO
    )  # Default to show info messages

    perform_siesta_default_basis: bool = field(
        default=True,
        metadata={
            "description": (
                "A wrapper-level flag to control whether a default basis set "
                "should be generated using the subsequent PAO parameters. This "
                "is not a direct SIESTA keyword."
            ),
            "SIESTA keyword": None,
        },
    )
    pao_basis_type: str = field(
        default="split",
        metadata={
            "description": (
                "The method used to generate the PAO basis set. Common options "
                "are 'split' for split-valence or 'gamess' for reading GAMESS "
                "output."
            ),
            "SIESTA keyword": "PAO.BasisType",
        },
    )
    pao_basissize: str = field(
        default="DZP",
        metadata={
            "description": (
                "A convenience flag for defining standard basis set sizes like "
                "SZ, DZ, SZP, DZP. This is a scalar parameter, not a block."
            ),
            "SIESTA keyword": "PAO.BasisSize",
        },
    )
    pao_basissizes_block: list[str] = field(
        default_factory=list,
        metadata={
            "description": (
                "Allows per-species basis size specification. Accepts TWO "
                "formats: (1) List: ['Si DZP', 'O TZP'], (2) Dict: {'Si': "
                "'DZP', 'O_surface': 'TZP', 'O_bulk': 'DZ'} (enables species "
                "variants). Dict format is auto-converted to list internally. "
                "Mutually exclusive with PAO.BasisSize scalar and "
                "%block PAO.Basis."
            ),
            "SIESTA keyword": "%block PAO.BasisSizes",
        },
    )
    pao_basis_block: list[str] = field(
        default_factory=list,
        metadata={
            "description": (
                "Full custom basis specification with complete orbital details. "
                "This is the highest priority and overrides both PAO.BasisSize "
                "and %block PAO.BasisSizes."
            ),
            "SIESTA keyword": "%block PAO.Basis",
        },
    )
    pao_energy_shift: float = field(
        default=0.01,
        metadata={
            "description": (
                "An energy shift (in Rydberg) to soften the confining potential "
                "of orbitals, which can improve stability. Note: This is "
                "overridden by settings in the PAO.BasisSize block."
            ),
            "SIESTA keyword": "PAO.EnergyShift",
            "unit": "Ry",
        },
    )
    write_graphviz: str = field(
        default="none",
        metadata={
            "description": (
                "If set (e.g., to 'iteration'), writes the calculation "
                "dependency graph in Graphviz format for debugging."
            ),
            "SIESTA keyword": "WriteGraphviz",
        },
    )
    pao_split_norm: float = field(
        default=0.15,
        metadata={
            "description": (
                "In the 'split' basis type, this is the norm of the tail of the "
                "first-zeta orbital that determines the split point for creating "
                "the second-zeta."
            ),
            "SIESTA keyword": "PAO.SplitNorm",
        },
    )
    pao_split_norm_h: float = field(
        default=1.0,
        metadata={
            "description": (
                "A special value of PAO.SplitNorm that is applied only to "
                "Hydrogen atoms."
            ),
            "SIESTA keyword": "PAO.SplitNormH",
        },
    )
    pao_split_tail_norm: bool = field(
        default=True,
        metadata={
            "description": (
                "Determines the split criterion. If true, uses the tail norm; "
                "if false, uses the overlap between first and second zeta "
                "orbitals. Disabling is recommended for larger basis sets like "
                "DZP/TZP."
            ),
            "SIESTA keyword": "PAO.SplitTailNorm",
        },
    )
    pao_split_valence_legacy: bool = field(
        default=True,
        metadata={
            "description": "Use the legacy algorithm for the split-valence procedure.",
            "SIESTA keyword": "PAO.SplitValenceLegacy",
        },
    )
    pao_fix_split_table: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, uses a fixed internal table for split-valence "
                "parameters, ensuring reproducibility across different "
                "architectures."
            ),
            "SIESTA keyword": "PAO.FixSplitTable",
        },
    )
    pao_energy_cutoff: float = field(
        default=20.0,
        metadata={
            "description": (
                "Energy cutoff (in Rydberg) for the orbital-filtering process, "
                "which discards redundant or high-energy orbitals from the "
                "basis."
            ),
            "SIESTA keyword": "PAO.Filter.Cutoff",
            "unit": "Ry",
        },
    )
    pao_energy_pol_cutoff: float = field(
        default=20.0,
        metadata={
            "description": (
                "A specific energy cutoff (in Rydberg) applied only to "
                "polarization orbitals during the filtering process."
            ),
            "SIESTA keyword": "PAO.Filter.PolarizationCutoff",
            "unit": "Ry",
        },
    )
    pao_contraction_cutoff: float = field(
        default=0.0,
        metadata={
            "description": (
                "Radial overlap cutoff used to contract basis orbitals into "
                "more efficient, combined orbitals."
            ),
            "SIESTA keyword": "PAO.ContractionCutoff",
        },
    )
    pao_polarization_non_perturbative: bool = field(
        default=True,
        metadata={
            "description": (
                "If true, enables the non-perturbative generation of "
                "polarization orbitals. Recommended for 'allowed' basis sets."
            ),
            "SIESTA keyword": "PAO.Polarization.NonPerturbative",
        },
    )
    pao_polarization_scheme_block: dict[str, Any] | None = field(
        default_factory=dict,
        metadata={
            "description": (
                "Allows for the detailed customization of the polarization "
                "shell scheme on a per-species basis."
            ),
            "SIESTA keyword": "%block PAO.Polarization.Scheme",
        },
    )
    pao_polarization_rc_expansion_factor: float = field(
        default=1.0,
        metadata={
            "description": (
                "A factor to scale the cutoff radii (rc) of the orbitals used "
                "to generate the polarization orbitals."
            ),
            "SIESTA keyword": "PAO.Polarization.RcExpansionFactor",
        },
    )
    pao_soft_default: bool = field(
        default=True,
        metadata={
            "description": (
                "A wrapper-level flag to use a soft confinement potential for "
                "generating orbitals, controlled by the subsequent parameters. "
                "This is not a direct SIESTA keyword."
            ),
            "SIESTA keyword": None,
        },
    )
    pao_soft_inner_radius: float = field(
        default=0.9,
        metadata={
            "description": (
                "The inner radius (in Bohr) at which the soft repulsive "
                "potential begins."
            ),
            "SIESTA keyword": "PAO.SoftDefault.InnerRadius",
            "unit": "Bohr",
        },
    )
    pao_soft_potential: float = field(
        default=40.0,
        metadata={
            "description": "The height of the soft repulsive potential (in Rydberg).",
            "SIESTA keyword": "PAO.SoftDefault.Potential",
            "unit": "Ry",
        },
    )
    ps_lmax_block: dict[str, Any] | None = field(
        default_factory=dict,
        metadata={
            "description": (
                "Specifies the maximum angular momentum (l) of the "
                "pseudopotential to use for each species."
            ),
            "SIESTA keyword": "%block PS.lmax",
        },
    )
    kb_projectors_block: dict[str, Any] | None = field(
        default_factory=dict,
        metadata={
            "description": (
                "Defines the number of Kleinman-Bylander (KB) projectors for "
                "each angular momentum channel."
            ),
            "SIESTA keyword": "%block PS.KBprojectors",
        },
    )
    filter_cutoff: float = field(
        default=0.0,
        metadata={
            "description": (
                "Energy cutoff (in eV) for the orbital-filtering process. "
                "Note: The native SIESTA unit is Rydberg."
            ),
            "SIESTA keyword": "PAO.Filter.Cutoff",
            "unit": "eV",
        },
    )
    filter_tol: float = field(
        default=0.0,
        metadata={
            "description": (
                "Tolerance (in eV) to discard linearly-dependent orbitals "
                "during the filtering stage. Note: The native SIESTA unit is "
                "Rydberg."
            ),
            "SIESTA keyword": "PAO.Filter.Tolerance",
            "unit": "eV",
        },
    )
    user_basis: bool = field(
        default=False,
        metadata={
            "description": (
                "A wrapper-level flag to indicate that a user-provided basis "
                "set file should be used, potentially bypassing automatic "
                "generation."
            ),
            "SIESTA keyword": "User.Basis",
        },
    )
    user_basis_netcdf: bool = field(
        default=False,
        metadata={
            "description": (
                "A flag to specify that the user-provided basis set is in the "
                "NetCDF (.nc) format. This is not a direct SIESTA keyword; file "
                "type is inferred from the filename."
            ),
            "SIESTA keyword": "User.Basis.NetCDF",
        },
    )
    basis_pressure: float = field(
        default=0.2,
        metadata={
            "description": (
                "Sets the confining pressure (in GPa) on the atom to generate "
                "basis orbitals. A higher pressure results in more localized "
                "(less diffuse) orbitals."
            ),
            "SIESTA keyword": "BasisPressure",
            "unit": "GPa",
        },
    )
    reparametrize_pseudos: bool = field(
        default=True,
        metadata={
            "description": (
                "A wrapper-level flag to enable a custom reparametrization of "
                "pseudopotentials before the main calculation. This is not a "
                "direct SIESTA keyword."
            ),
            "SIESTA keyword": "Reparametrize.Pseudos",
        },
    )
    new_a_parameter: float = field(
        default=0.001,
        metadata={
            "description": (
                "A custom 'a' parameter for the pseudopotential "
                "reparametrization scheme. Its meaning is defined by the "
                "external tool used. Not a SIESTA keyword."
            ),
            "SIESTA keyword": "New.A.Parameter",
        },
    )
    new_b_parameter: float = field(
        default=0.01,
        metadata={
            "description": (
                "A custom 'b' parameter for the pseudopotential "
                "reparametrization scheme. Its meaning is defined by the "
                "external tool used. Not a SIESTA keyword."
            ),
            "SIESTA keyword": "New.B.Parameter",
        },
    )
    rmax_radial_grid: float = field(
        default=50.0,
        metadata={
            "description": (
                "The maximum radius (in Bohr) for the radial grid used during "
                "the atomic (e.g., pseudopotential generation) calculation. Not "
                "a direct SIESTA keyword."
            ),
            "SIESTA keyword": "Rmax.Radial.Grid",
            "unit": "Bohr",
        },
    )
    restricted_radial_grid: bool = field(
        default=True,
        metadata={
            "description": (
                "A flag to use a restricted radial grid that adapts to the "
                "orbital extent, instead of a fixed grid. This is a feature of "
                "auxiliary atomic codes. Not a SIESTA keyword."
            ),
            "SIESTA keyword": "Restricted.Radial.Grid",
        },
    )
    pao_rc_unbound_state: float = field(
        default=0.0,
        metadata={
            "description": (
                "Sets the cutoff radius (in Bohr) for unbound atomic states "
                "that are used for generating basis orbitals."
            ),
            "SIESTA keyword": "PAO.rc.unbound.state",
            "unit": "Bohr",
        },
    )
    basis_set_fdf_arguments: OrderedDict[str, Any] = field(
        default_factory=OrderedDict,
        metadata={
            "description": (
                "A dictionary for any additional or arbitrary FDF (Flexible "
                "Data Format) flags related to the basis set. This allows for "
                "using keywords not explicitly defined elsewhere."
            ),
            "SIESTA keyword": None,
        },
    )
    comments: str = field(
        default="BasisSetsAndProjectors Settings",
        metadata={
            "description": (
                "User-provided comments to be included as a comment block in "
                "the FDF file."
            ),
            "SIESTA keyword": None,
        },
    )

    def __post_init__(self) -> None:
        """Register FDF parameters handled by this dataclass."""
        if not hasattr(self.__class__, "_registered"):
            self.register_fdf_params(
                "PAO.BasisType",
                "%block PAO.Basis",  # Highest priority - custom basis
                "%block PAO.BasisSizes",  # Medium priority - per-species sizes
                "PAO.BasisSize",  # Lowest priority - global scalar
                "PAO.EnergyShift",
                "WriteGraphviz",
                "PAO.SplitNorm",
                "PAO.SplitNormH",
                "PAO.SplitTailNorm",
                "PAO.SplitValenceLegacy",
                "PAO.FixSplitTable",
                "PAO.Filter.Cutoff",
                "PAO.Filter.PolarizationCutoff",
                "PAO.ContractionCutoff",
                "PAO.Polarization.NonPerturbative",
                "%block PAO.Polarization.Scheme",
                "PAO.Polarization.RcExpansionFactor",
                "PAO.SoftDefault.InnerRadius",
                "PAO.SoftDefault.Potential",
                "%block PS.lmax",
                "%block PS.KBprojectors",
                "PAO.Filter.Tolerance",
                "User.Basis",
                "User.Basis.NetCDF",
                "BasisPressure",
                "Reparametrize.Pseudos",
                "New.A.Parameter",
                "New.B.Parameter",
                "Rmax.Radial.Grid",
                "Restricted.Radial.Grid",
                "PAO.rc.unbound.state",
            )
            self.__class__._registered = True  # noqa: SLF001

    @classmethod
    def setup_basis_sets_and_projectors(
        cls, user_params: dict[str, Any] | None = None, **kwargs
    ) -> "BasisSetsAndProjectors":
        """
        Create and configure a BasisSetsAndProjectors instance from user params.

        All default values are retained for unspecified fields.

        Args:
            user_params (dict, optional): Dictionary of user-defined parameters
                (case-insensitive, may include dots). If None or empty, all
                default BasisSetsAndProjectors values are used.
            **kwargs: Additional keyword arguments to override or supplement
                user_params.

        Returns
        -------
            BasisSetsAndProjectors: Configured instance with all fields (default
                and user-specified) and FDF arguments.
        """
        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]BasisSetsAndProjectors.setup_basis_sets_and_projectors()[/green]"
            )

        # Initialize BasisSetsAndProjectors instance with defaults
        basis_instance = cls()

        # Handle case where user_params is None or empty
        if user_params is None or not user_params:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    "[blue]No user parameters provided; using all default "
                    "BasisSetsAndProjectors values.[/blue]"
                )
        else:
            # Get valid BasisSetsAndProjectors attribute names (lowercase)
            basis_attributes = {field.name.lower() for field in fields(cls)}
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                console.print(
                    f"[blue]Available BasisSetsAndProjectors attributes: "
                    f"{basis_attributes}[/blue]"
                )

            # Process user parameters
            for key, value in user_params.items():
                # Normalize key: handle camelCase properly
                # Insert underscore before capital letters that follow lowercase letters
                import re

                key_with_underscores = re.sub(r"([a-z])([A-Z])", r"\1_\2", key)
                # Then replace dots with underscores and convert to lowercase
                key_normalized = key_with_underscores.replace(".", "_").lower()
                if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                    console.print(
                        f"[blue]Processing key: {key} -> {key_with_underscores} "
                        f"-> {key_normalized}, value: {value}[/blue]"
                    )

                # Check if normalized key matches any BasisSetsAndProjectors attribute
                # First try exact match, then try fuzzy match (remove all underscores)
                matched_attr = None
                if key_normalized in basis_attributes:
                    matched_attr = key_normalized
                else:
                    # Fuzzy match: remove all underscores and compare
                    key_no_underscores = key_normalized.replace("_", "")
                    for attr in basis_attributes:
                        if attr.replace("_", "") == key_no_underscores:
                            matched_attr = attr
                            if (
                                cls.CONSOLE_VERBOSITY.value
                                >= VerbosityLevel.DEBUG.value
                            ):
                                console.print(
                                    f"[blue]Fuzzy matched: {key_normalized} -> "
                                    f"{attr} (both normalize to "
                                    f"{key_no_underscores})[/blue]"
                                )
                            break

                if matched_attr:
                    # Find the original attribute name (preserving case)
                    original_key = next(
                        field.name
                        for field in fields(cls)
                        if field.name.lower() == matched_attr
                    )
                    if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                        console.print(
                            f"[blue]Matched BasisSetsAndProjectors field: "
                            f"{original_key} = {value}[/blue]"
                        )

                    # Handle type conversion for specific fields
                    if original_key in [
                        "pao_basissizes_block",
                        "pao_basis_block",
                        "pao_polarization_scheme_block",
                        "ps_lmax_block",
                        "kb_projectors_block",
                    ]:
                        # Special handling for pao_basissizes_block:
                        # accepts dict OR list
                        if original_key == "pao_basissizes_block":
                            if isinstance(value, dict):
                                # Convert dict to list format
                                setattr(
                                    basis_instance,
                                    original_key,
                                    [
                                        f"{label}  {basis}"
                                        for label, basis in value.items()
                                    ],
                                )
                            elif isinstance(value, list):
                                setattr(basis_instance, original_key, value)
                            elif (
                                cls.CONSOLE_VERBOSITY.value
                                >= VerbosityLevel.WARNING.value
                            ):
                                console.print(
                                    f"[yellow]Invalid value type for "
                                    f"{original_key}: expected dict or list, "
                                    f"got {type(value)}[/yellow]"
                                )
                        # Other block fields: list only
                        elif isinstance(value, list):
                            setattr(basis_instance, original_key, value)
                        elif (
                            cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value
                        ):
                            console.print(
                                f"[yellow]Invalid value type for "
                                f"{original_key}: expected list, got "
                                f"{type(value)}[/yellow]"
                            )
                    elif original_key in [
                        "perform_siesta_default_basis",
                        "pao_split_tail_norm",
                        "pao_split_valence_legacy",
                        "pao_fix_split_table",
                        "pao_polarization_non_perturbative",
                        "pao_soft_default",
                        "user_basis",
                        "user_basis_netcdf",
                        "reparametrize_pseudos",
                        "restricted_radial_grid",
                    ]:
                        bool_value = (
                            value.lower() in ("true", "t", "1", "yes")
                            if isinstance(value, str)
                            else value
                        )
                        setattr(basis_instance, original_key, bool(bool_value))
                    elif original_key in [
                        "pao_energy_shift",
                        "pao_split_norm",
                        "pao_split_norm_h",
                        "pao_energy_cutoff",
                        "pao_energy_pol_cutoff",
                        "pao_contraction_cutoff",
                        "pao_polarization_rc_expansion_factor",
                        "pao_soft_inner_radius",
                        "pao_soft_potential",
                        "basis_pressure",
                        "new_a_parameter",
                        "new_b_parameter",
                        "rmax_radial_grid",
                        "pao_rc_unbound_state",
                        "filter_cutoff",
                        "filter_tol",
                    ]:
                        try:
                            setattr(basis_instance, original_key, float(value))
                        except (ValueError, TypeError):
                            if (
                                cls.CONSOLE_VERBOSITY.value
                                >= VerbosityLevel.WARNING.value
                            ):
                                console.print(
                                    f"[yellow]Invalid value type for "
                                    f"{original_key}: expected float, got "
                                    f"{value}[/yellow]"
                                )
                    elif original_key == "pao_basis_type":
                        allowed_basis_types = [
                            "split",
                            "splitgauss",
                            "nodes",
                            "nonodes",
                            "filteret",
                        ]
                        if value.lower() not in allowed_basis_types:
                            if (
                                cls.CONSOLE_VERBOSITY.value
                                >= VerbosityLevel.WARNING.value
                            ):
                                console.print(
                                    f"[red]Invalid basis type "
                                    f"[bold]'{value}'[/bold] for "
                                    f"[bold]{original_key}[/bold].[/red]"
                                )
                                console.print(
                                    f"[red]Allowed basis types are "
                                    f"[bold]{allowed_basis_types}[/bold][/red]"
                                )
                            raise ValueError(
                                f"Invalid basis type '{value}'. Allowed values "
                                f"are: {allowed_basis_types}"
                            )
                        setattr(basis_instance, original_key, value.lower())
                    elif original_key == "pao_basissize":
                        allowed_basis_sizes = [
                            "SZ",
                            "MINIMAL",
                            "SZP",
                            "SZSP",
                            "SZ1P",
                            "SZP1",
                            "DZ",
                            "DZP",
                            "DZSP",
                            "DZP1",
                            "DZ1P",
                            "STANDARD",
                            "DZDP",
                            "DZP2",
                            "DZ2P",
                            "TZ",
                            "TZP",
                            "TZSP",
                            "TZP1",
                            "TZ1P",
                            "TZDP",
                            "TZP2",
                            "TZ2P",
                            "TZTP",
                            "TZP3",
                            "TZ3P",
                        ]
                        if value.upper() not in allowed_basis_sizes:
                            if (
                                cls.CONSOLE_VERBOSITY.value
                                >= VerbosityLevel.WARNING.value
                            ):
                                console.print(
                                    f"[red]Invalid basis size "
                                    f"[bold]'{value}'[/bold] for "
                                    f"[bold]{original_key}[/bold].[/red]"
                                )
                                console.print(
                                    f"[red]Allowed basis sizes are "
                                    f"[bold]{allowed_basis_sizes}[/bold][/red]"
                                )
                            raise ValueError(
                                f"Invalid basis size '{value}'. Allowed values "
                                f"are: {allowed_basis_sizes}"
                            )
                        setattr(basis_instance, original_key, value.upper())
                    else:
                        setattr(basis_instance, original_key, value)
                elif cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                    console.print(
                        f"[yellow]Key '{key}' does not match any "
                        f"BasisSetsAndProjectors field, skipping.[/yellow]"
                    )

        # Update with kwargs (kwargs take precedence over user_params)
        for key, value in kwargs.items():
            key_normalized = key.lower().replace(".", "_")
            if key_normalized in basis_attributes:
                original_key = next(
                    field.name
                    for field in fields(cls)
                    if field.name.lower() == key_normalized
                )
                if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.DEBUG.value:
                    console.print(
                        f"[blue]Matched BasisSetsAndProjectors kwarg: "
                        f"{original_key} = {value}[/blue]"
                    )
                # Apply similar type conversion logic for kwargs
                if original_key in [
                    "pao_basissizes_block",
                    "pao_basis_block",
                    "pao_polarization_scheme_block",
                    "ps_lmax_block",
                    "kb_projectors_block",
                ]:
                    # Special handling for pao_basissizes_block:
                    # accepts dict OR list
                    if original_key == "pao_basissizes_block":
                        if isinstance(value, dict):
                            # Convert dict to list format
                            setattr(
                                basis_instance,
                                original_key,
                                [f"{label}  {basis}" for label, basis in value.items()],
                            )
                        elif isinstance(value, list):
                            setattr(basis_instance, original_key, value)
                    # Other block fields: list only
                    elif isinstance(value, list):
                        setattr(basis_instance, original_key, value)
                elif original_key in [
                    "perform_siesta_default_basis",
                    "pao_split_tail_norm",
                    "pao_split_valence_legacy",
                    "pao_fix_split_table",
                    "pao_polarization_non_perturbative",
                    "pao_soft_default",
                    "user_basis",
                    "user_basis_netcdf",
                    "reparametrize_pseudos",
                    "restricted_radial_grid",
                ]:
                    bool_value = (
                        value.lower() in ("true", "t", "1", "yes")
                        if isinstance(value, str)
                        else value
                    )
                    setattr(basis_instance, original_key, bool(bool_value))
                elif original_key in [
                    "pao_energy_shift",
                    "pao_split_norm",
                    "pao_split_norm_h",
                    "pao_energy_cutoff",
                    "pao_energy_pol_cutoff",
                    "pao_contraction_cutoff",
                    "pao_polarization_rc_expansion_factor",
                    "pao_soft_inner_radius",
                    "pao_soft_potential",
                    "basis_pressure",
                    "new_a_parameter",
                    "new_b_parameter",
                    "rmax_radial_grid",
                    "pao_rc_unbound_state",
                    "filter_cutoff",
                    "filter_tol",
                ]:
                    try:
                        setattr(basis_instance, original_key, float(value))
                    except (ValueError, TypeError):
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                            console.print(
                                f"[yellow]Invalid value type for "
                                f"{original_key}: expected float, got "
                                f"{value}[/yellow]"
                            )
                elif original_key == "pao_basis_type":
                    allowed_basis_types = [
                        "split",
                        "splitgauss",
                        "nodes",
                        "nonodes",
                        "filteret",
                    ]
                    if value.lower() not in allowed_basis_types:
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                            console.print(
                                f"[red]Invalid basis type "
                                f"[bold]'{value}'[/bold] for "
                                f"[bold]{original_key}[/bold].[/red]"
                            )
                            console.print(
                                f"[red]Allowed basis types are "
                                f"[bold]{allowed_basis_types}[/bold][/red]"
                            )
                        raise ValueError(
                            f"Invalid basis type '{value}'. Allowed values are: "
                            f"{allowed_basis_types}"
                        )
                    setattr(basis_instance, original_key, value.lower())
                elif original_key == "pao_basissize":
                    allowed_basis_sizes = [
                        "SZ",
                        "MINIMAL",
                        "SZP",
                        "SZSP",
                        "SZ1P",
                        "SZP1",
                        "DZ",
                        "DZP",
                        "DZSP",
                        "DZP1",
                        "DZ1P",
                        "STANDARD",
                        "DZDP",
                        "DZP2",
                        "DZ2P",
                        "TZ",
                        "TZP",
                        "TZSP",
                        "TZP1",
                        "TZ1P",
                        "TZDP",
                        "TZP2",
                        "TZ2P",
                        "TZTP",
                        "TZP3",
                        "TZ3P",
                    ]
                    if value.upper() not in allowed_basis_sizes:
                        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.WARNING.value:
                            console.print(
                                f"[red]Invalid basis size "
                                f"[bold]'{value}'[/bold] for "
                                f"[bold]{original_key}[/bold].[/red]"
                            )
                            console.print(
                                f"[red]Allowed basis sizes are "
                                f"[bold]{allowed_basis_sizes}[/bold][/red]"
                            )
                        raise ValueError(
                            f"Invalid basis size '{value}'. Allowed values are: "
                            f"{allowed_basis_sizes}"
                        )
                    setattr(basis_instance, original_key, value.upper())
                else:
                    setattr(basis_instance, original_key, value)

        # Validate basis settings
        try:
            basis_instance.validate()
        except ValueError as e:
            if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.ERROR.value:
                console.print(f"[red]Validation failed: {e}[/red]")
            raise

        # Generate FDF basis block
        basis_instance.generate_basis_block()

        if cls.CONSOLE_VERBOSITY.value >= VerbosityLevel.INFO.value:
            console.print(
                "[green]Validation & Generation: "
                "[yellow]BasisSetsAndProjectors[/yellow] Successful![/green]"
            )

        return basis_instance

    def validate(self) -> None:
        """Validate the basis set and KB projectors settings."""
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]BasisSetsAndProjectors.validate()[/green]")

        allowed_basis_size = [
            "SZ",
            "MINIMAL",
            "SZP",
            "SZSP",
            "SZ1P",
            "SZP1",
            "DZ",
            "DZP",
            "DZSP",
            "DZP1",
            "DZ1P",
            "STANDARD",
            "DZDP",
            "DZP2",
            "DZ2P",
            "TZ",
            "TZP",
            "TZSP",
            "TZP1",
            "TZ1P",
            "TZDP",
            "TZP2",
            "TZ2P",
            "TZTP",
            "TZP3",
            "TZ3P",
        ]
        # Basis Size
        if self.pao_basissize.upper() not in allowed_basis_size:
            raise ValueError(
                f"Invalid basis set '{self.pao_basissize}'. Allowed values "
                f"are: {allowed_basis_size}"
            )

        # Basis Type
        allowed_pao_basis_type = ["split", "splitgauss", "nodes", "nonodes", "filteret"]
        if self.pao_basis_type not in allowed_pao_basis_type:
            raise ValueError(
                f"Invalid basis set '{self.pao_basis_type}'. Allowed values "
                f"are: {allowed_pao_basis_type}"
            )

        if self.perform_siesta_default_basis and self.pao_basis_type in [
            "split",
            "nodes",
        ]:
            if not self.pao_energy_shift:
                raise ValueError(
                    "pao_energy_shift must be specified when pao_basis_type "
                    "is 'split' or 'nodes'."
                )
            if not self.pao_split_norm:
                raise ValueError(
                    "pao_split_norm must be specified when pao_basis_type "
                    "is 'split' or 'nodes'."
                )
            if not self.pao_basissize:
                raise ValueError(
                    "pao_basissize must be specified when pao_basis_type "
                    "is 'split' or 'nodes'."
                )

        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.INFO.value:
            console.print(
                "[green]Validation: [yellow]BasisSetsAndProjectors[/yellow] "
                "Successful![/green]"
            )

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        Args:
            fdf_dict: Dictionary of FDF parameters (from user_params)
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]BasisSetsAndProjectors.update_from_fdf()[/green]")

        for key, value in fdf_dict.items():
            key_lower = key.lower()

            # Basis parameters
            if key_lower == "pao.basistype":
                self.pao_basis_type = value
            elif key_lower == "%block pao.basis":
                # Highest priority - full custom basis
                if isinstance(value, list):
                    self.pao_basis_block = value
            elif key_lower in ["%block pao.basissizes", "pao_basissizes_block"]:
                # Medium priority - per-species basis sizes
                # Accepts TWO formats:
                # 1. List: ["Si DZP", "O TZP"]
                # 2. Dict: {"Si": "DZP", "O": "TZP"} (auto-converted to list)
                if isinstance(value, dict):
                    # Convert dict to list format
                    self.pao_basissizes_block = [
                        f"{label}  {basis}" for label, basis in value.items()
                    ]
                elif isinstance(value, list):
                    # List format (existing)
                    self.pao_basissizes_block = value
            elif key_lower in ["pao.basissize", "pao_basissize"]:
                # Lowest priority - global scalar
                self.pao_basissize = str(value).upper()
            elif key_lower == "pao.energyshift":
                self.pao_energy_shift = parse_energy(value, target_unit="Ry")
            elif key_lower == "pao.splitnorm":
                self.pao_split_norm = float(value) if isinstance(value, str) else value
            elif key_lower == "pao.splitnormh":
                self.pao_split_norm_h = (
                    float(value) if isinstance(value, str) else value
                )
            elif key_lower == "pao.splittailnorm":
                self.pao_split_tail_norm = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower == "pao.splitvalencelegacy":
                self.pao_split_valence_legacy = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower == "pao.fixsplittable":
                self.pao_fix_split_table = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )

            # Filter parameters
            elif key_lower == "pao.filter.cutoff":
                self.pao_energy_cutoff = parse_energy(value, target_unit="Ry")
            elif key_lower == "pao.filter.polarizationcutoff":
                self.pao_energy_pol_cutoff = parse_energy(value, target_unit="Ry")
            elif key_lower == "pao.filter.tolerance":
                self.filter_tol = parse_energy(value, target_unit="eV")
            elif key_lower == "pao.contractioncutoff":
                self.pao_contraction_cutoff = (
                    float(value) if isinstance(value, str) else value
                )

            # Polarization parameters
            elif key_lower == "pao.polarization.nonperturbative":
                self.pao_polarization_non_perturbative = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower == "%block pao.polarization.scheme":
                if isinstance(value, dict):
                    self.pao_polarization_scheme_block = value
            elif key_lower == "pao.polarization.rcexpansionfactor":
                self.pao_polarization_rc_expansion_factor = (
                    float(value) if isinstance(value, str) else value
                )

            # Soft confinement parameters
            elif key_lower == "pao.softdefault.innerradius":
                self.pao_soft_inner_radius = parse_length(value, target_unit="Bohr")
            elif key_lower == "pao.softdefault.potential":
                self.pao_soft_potential = parse_energy(value, target_unit="Ry")

            # Pseudopotential parameters
            elif key_lower == "%block ps.lmax":
                if isinstance(value, dict):
                    self.ps_lmax_block = value
            elif key_lower == "%block ps.kbprojectors":
                if isinstance(value, dict):
                    self.kb_projectors_block = value

            # Other parameters
            elif key_lower == "writegraphviz":
                self.write_graphviz = value
            elif key_lower == "user.basis":
                self.user_basis = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower == "user.basis.netcdf":
                self.user_basis_netcdf = (
                    value.lower() in ["true", "t", "yes", "1"]
                    if isinstance(value, str)
                    else bool(value)
                )
            elif key_lower == "basispressure":
                self.basis_pressure = float(value) if isinstance(value, str) else value
            elif key_lower == "pao.rc.unbound.state":
                self.pao_rc_unbound_state = parse_length(value, target_unit="Bohr")

    def generate_fdf(self) -> dict[str, Any]:
        """
        Generate SIESTA FDF format parameters.

        Returns
        -------
            Dictionary of FDF parameters with proper units
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print("[green]BasisSetsAndProjectors.generate_fdf()[/green]")

        fdf: dict[str, Any] = OrderedDict()
        fdf["#BasisSetsAndProjectors"] = "BasisSetsAndProjectors"

        # Basis parameters with default markers
        # PAO.BasisType
        if self.pao_basis_type == "split":
            fdf["PAO.BasisType"] = f"{self.pao_basis_type}  # SIESTA DEFAULT VALUE"
        else:
            fdf["PAO.BasisType"] = self.pao_basis_type

        # PAO.Basis / PAO.BasisSizes / PAO.BasisSize - Can coexist with priority
        # SIESTA priority: %block PAO.Basis > %block PAO.BasisSizes > PAO.BasisSize
        # When blocks are present, they override PAO.BasisSize for their species only
        # PAO.BasisSize serves as fallback for species not in blocks
        # SIESTA documentation: See PAO.Basis and PAO.BasisSizes in SIESTA manual

        # PRIORITY 1: Custom basis block (highest priority for species defined in it)
        if self.pao_basis_block:
            fdf["%block PAO.Basis"] = self.pao_basis_block

        # PRIORITY 2: Per-species basis sizes (medium priority, mutually
        # exclusive with PAO.BasisSize)
        if self.pao_basissizes_block:
            fdf["%block PAO.BasisSizes"] = self.pao_basissizes_block
            # Don't write scalar - %block PAO.BasisSizes completely replaces
            # PAO.BasisSize

        # PRIORITY 3: Global basis size scalar (fallback for species not in blocks)
        elif not self.pao_basissizes_block:
            # Write scalar if no %block PAO.BasisSizes (can coexist with
            # %block PAO.Basis)
            if self.pao_basissize == "DZP":
                fdf["PAO.BasisSize"] = "DZP  # SIESTA DEFAULT VALUE"
            else:
                fdf["PAO.BasisSize"] = self.pao_basissize

        # PAO.EnergyShift
        if self.pao_energy_shift == 0.01:
            fdf["PAO.EnergyShift"] = (
                f"{self.pao_energy_shift} Ry  # SIESTA DEFAULT VALUE"
            )
        else:
            fdf["PAO.EnergyShift"] = f"{self.pao_energy_shift} Ry"

        # PAO.SplitNorm
        if self.pao_split_norm == 0.15:
            fdf["PAO.SplitNorm"] = f"{self.pao_split_norm}  # SIESTA DEFAULT VALUE"
        else:
            fdf["PAO.SplitNorm"] = self.pao_split_norm

        # PAO.SplitNormH
        if self.pao_split_norm_h == 1.0:
            fdf["PAO.SplitNormH"] = f"{self.pao_split_norm_h}  # SIESTA DEFAULT VALUE"
        else:
            fdf["PAO.SplitNormH"] = self.pao_split_norm_h

        # PAO.SplitTailNorm
        if self.pao_split_tail_norm:
            fdf["PAO.SplitTailNorm"] = (
                f"{str(self.pao_split_tail_norm).lower()}  # SIESTA DEFAULT VALUE"
            )
        else:
            fdf["PAO.SplitTailNorm"] = str(self.pao_split_tail_norm).lower()

        # PAO.SplitValenceLegacy
        if self.pao_split_valence_legacy:
            fdf["PAO.SplitValenceLegacy"] = (
                f"{str(self.pao_split_valence_legacy).lower()}  # SIESTA DEFAULT VALUE"
            )
        else:
            fdf["PAO.SplitValenceLegacy"] = str(self.pao_split_valence_legacy).lower()

        # PAO.FixSplitTable
        if not self.pao_fix_split_table:
            fdf["PAO.FixSplitTable"] = (
                f"{str(self.pao_fix_split_table).lower()}  # SIESTA DEFAULT VALUE"
            )
        else:
            fdf["PAO.FixSplitTable"] = str(self.pao_fix_split_table).lower()

        # Filter parameters with default markers
        # PAO.Filter.Cutoff
        if self.pao_energy_cutoff == 20.0:
            fdf["PAO.Filter.Cutoff"] = (
                f"{self.pao_energy_cutoff} Ry  # SIESTA DEFAULT VALUE"
            )
        else:
            fdf["PAO.Filter.Cutoff"] = f"{self.pao_energy_cutoff} Ry"

        # PAO.Filter.PolarizationCutoff
        if self.pao_energy_pol_cutoff == 20.0:
            fdf["PAO.Filter.PolarizationCutoff"] = (
                f"{self.pao_energy_pol_cutoff} Ry  # SIESTA DEFAULT VALUE"
            )
        else:
            fdf["PAO.Filter.PolarizationCutoff"] = f"{self.pao_energy_pol_cutoff} Ry"

        # PAO.Filter.Tolerance
        if self.filter_tol == 0.0:
            fdf["PAO.Filter.Tolerance"] = (
                f"{self.filter_tol} eV  # SIESTA DEFAULT VALUE"
            )
        else:
            fdf["PAO.Filter.Tolerance"] = f"{self.filter_tol} eV"

        # PAO.ContractionCutoff
        if self.pao_contraction_cutoff == 0.0:
            fdf["PAO.ContractionCutoff"] = (
                f"{self.pao_contraction_cutoff}  # SIESTA DEFAULT VALUE"
            )
        else:
            fdf["PAO.ContractionCutoff"] = self.pao_contraction_cutoff

        # Polarization parameters with default markers
        # PAO.Polarization.NonPerturbative
        if self.pao_polarization_non_perturbative:
            fdf["PAO.Polarization.NonPerturbative"] = (
                f"{str(self.pao_polarization_non_perturbative).lower()}  "
                "# SIESTA DEFAULT VALUE"
            )
        else:
            fdf["PAO.Polarization.NonPerturbative"] = str(
                self.pao_polarization_non_perturbative
            ).lower()
        if self.pao_polarization_scheme_block:
            fdf["%block PAO.Polarization.Scheme"] = self.pao_polarization_scheme_block

        # PAO.Polarization.RcExpansionFactor
        if self.pao_polarization_rc_expansion_factor == 1.0:
            fdf["PAO.Polarization.RcExpansionFactor"] = (
                f"{self.pao_polarization_rc_expansion_factor}  # SIESTA DEFAULT VALUE"
            )
        else:
            fdf["PAO.Polarization.RcExpansionFactor"] = (
                self.pao_polarization_rc_expansion_factor
            )

        # Soft confinement parameters
        if self.pao_soft_default:
            # PAO.SoftDefault.InnerRadius - always write with default marker
            if self.pao_soft_inner_radius == 0.9:
                fdf["PAO.SoftDefault.InnerRadius"] = "0.9 Bohr  # SIESTA DEFAULT VALUE"
            else:
                fdf["PAO.SoftDefault.InnerRadius"] = (
                    f"{self.pao_soft_inner_radius} Bohr"
                )

            # PAO.SoftDefault.Potential - always write with default marker
            if self.pao_soft_potential == 40.0:
                fdf["PAO.SoftDefault.Potential"] = "40.0 Ry  # SIESTA DEFAULT VALUE"
            else:
                fdf["PAO.SoftDefault.Potential"] = f"{self.pao_soft_potential} Ry"

        # Pseudopotential parameters
        if self.ps_lmax_block:
            fdf["%block PS.lmax"] = self.ps_lmax_block
        if self.kb_projectors_block:
            fdf["%block PS.KBprojectors"] = self.kb_projectors_block

        # Other parameters
        if self.write_graphviz != "none":
            fdf["WriteGraphviz"] = self.write_graphviz
        if self.user_basis:
            fdf["User.Basis"] = str(self.user_basis).lower()
        if self.user_basis_netcdf:
            fdf["User.Basis.NetCDF"] = str(self.user_basis_netcdf).lower()

        # BasisPressure - always write with default marker
        if self.basis_pressure == 0.2:
            fdf["BasisPressure"] = "0.2 GPa  # SIESTA DEFAULT VALUE"
        else:
            fdf["BasisPressure"] = f"{self.basis_pressure} GPa"
        if self.pao_rc_unbound_state > 0:
            fdf["PAO.rc.unbound.state"] = f"{self.pao_rc_unbound_state} Bohr"

        return fdf

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters (optional).

        Returns
        -------
            Dictionary of ASE parameters
        """
        # ASE uses different parameter names for basis settings
        return {
            "pao_basis_type": self.pao_basis_type,
            "pao_basissize": self.pao_basissize,
            "pao_energyshift": self.pao_energy_shift,
            "pao_splitnorm": self.pao_split_norm,
        }

    def generate_basis_block(self) -> None:
        """
        Generate the PAO.Basis block or flags for the FDF file.

        Includes all relevant fields (default and user-specified).

        This is a wrapper around generate_fdf() to maintain backward
        compatibility with code that calls this method directly (e.g.,
        setup_basis_sets_and_projectors()).

        By calling generate_fdf(), we ensure:
        - Single source of truth for FDF generation
        - Proper "# SIESTA DEFAULT VALUE" markers on default parameters
        - Consistency with user_params, powerups, and tier presets
        - DRY principle (no parameter duplication)
        - Values updated via update_from_fdf() are properly reflected
        """
        if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.VERBOSE.value:
            console.print(
                "[green]BasisSetsAndProjectors.generate_basis_block()[/green]"
            )

        # Call generate_fdf() which uses the current dataclass attributes
        # (these have been updated from user_params/powerups/tiers via
        # update_from_fdf())
        fdf = self.generate_fdf()

        # Add comment header
        fdf_with_header = OrderedDict()
        if self.comments:
            fdf_with_header["#BasisSetsAndProjectors"] = self.comments
        fdf_with_header.update(fdf)

        self.basis_set_fdf_arguments = fdf_with_header
