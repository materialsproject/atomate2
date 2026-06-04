"""Module defining base SIESTA input set and generator."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

import os
import json
import logging
from pathlib import Path

from dataclasses import dataclass
from dataclasses import field

from typing import TYPE_CHECKING
from typing import List
from typing import Any
from typing import Optional

from collections import OrderedDict

from monty.json import MontyDecoder
from monty.json import MontyEncoder

from pymatgen.core import Molecule
from pymatgen.core import Structure
from pymatgen.io.core import InputFile
from pymatgen.io.core import InputGenerator
from pymatgen.io.core import InputSet

from atomate2.siesta.sets.parser import SiestaParseError
from atomate2.siesta.sets.parser import read_siesta_output_structure
from atomate2.siesta.utils.verbosity import VerbosityLevel, get_verbosity_value

from atomate2.siesta.dataclass.base import merge_fdf_parameters
from atomate2.siesta.dataclass.general_system_descriptors import (
    GeneralSystemDescriptors,
)
from atomate2.siesta.dataclass.pseudopotentials import Pseudopotentials
from atomate2.siesta.dataclass.external_control_and_scripting import (
    ExternalControlAndScripting,
)
from atomate2.siesta.dataclass.registry import (
    get_modules_for_tier,
    get_sorted_modules,
)

# ASE-based SIESTA interface imports (required for phonon/VIBRA calculations)
# Future: migrate phonon workflows from ASE to sisl for better integration
from atomate2.siesta.sets.ase import Species, PAOBasisBlock, Siesta
from atomate2.siesta.sets.utils import pymatgen_to_ase
from atomate2.siesta.sets.utils import siesta_fdf_to_json
from ase.units import Ry

from atomate2.siesta import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pymatgen.util.typing import PathLike


# ============================================================================
# Internal Parameter Naming System
# ============================================================================
# atomate2siesta uses two types of parameters:
# 1. SIESTA FDF parameters (no prefix) - written to siesta.fdf
# 2. atomate2siesta internal parameters (prefixed) - control framework behavior
#
# Internal parameters support BOTH prefixes (user choice):
#   - Full: "atomate2siesta_magnetic_ordering" (explicit)
#   - Alias: "a2s_magnetic_ordering" (concise)
# ============================================================================

INTERNAL_PARAM_PREFIX_FULL = "atomate2siesta_"
INTERNAL_PARAM_PREFIX_ALIAS = "a2s_"

# Legacy internal parameter names (NO LONGER SUPPORTED as of v1.0.0)
# These are used to provide helpful error messages
# NOTE: Only parameters that are ACTUALLY IMPLEMENTED are listed here
LEGACY_INTERNAL_PARAMS = {
    "magnetic_ordering": "a2s_magnetic_ordering",  # SpinSettings
    "kpts": "a2s_kpts",  # KPointSampling
    "pseudo_path": "a2s_pseudo_path",  # Pseudopotentials
    "pseudo_base_path": "a2s_pseudo_base_path",  # Pseudopotentials
    "pseudo_family": "a2s_pseudo_family",  # Pseudopotentials
    "pseudo_version": "a2s_pseudo_version",  # Pseudopotentials
    "pseudo_quality": "a2s_pseudo_quality",  # Pseudopotentials
    "pseudo_relativistic": "a2s_pseudo_relativistic",  # Pseudopotentials
}


def filter_internal_params(user_params: dict) -> tuple[dict, dict]:
    """
    Separate internal control parameters from FDF parameters.

    Supports both full prefix (atomate2siesta_) and short alias (a2s_).
    Internal parameters are atomate2siesta-specific controls that modify
    framework behavior but are NOT written to SIESTA FDF files.

    Args:
        user_params: User-provided parameters

    Returns:
        Tuple of (fdf_params, internal_params) where internal_params
        have their prefixes stripped

    Example:
        >>> params = {
        ...     "Mesh.Cutoff": "300 Ry",                    # SIESTA parameter
        ...     "atomate2siesta_magnetic_ordering": "AFM",  # Internal (full)
        ...     "a2s_auto_kpoints": True,                   # Internal (alias)
        ... }
        >>> fdf, internal = filter_internal_params(params)
        >>> fdf
        {'Mesh.Cutoff': '300 Ry'}
        >>> internal
        {'magnetic_ordering': 'AFM', 'auto_kpoints': True}
    """
    # Top-level InputGenerator fields that are NOT FDF parameters.
    # If these leak into user_params they must be filtered out so they
    # never reach merge_fdf_parameters() validation.
    _INPUT_GEN_FIELDS = {"enabled_modules", "disabled_modules"}

    fdf_params = {}
    internal_params = {}

    for key, value in user_params.items():
        # Check for full prefix
        if key.startswith(INTERNAL_PARAM_PREFIX_FULL):
            clean_key = key[len(INTERNAL_PARAM_PREFIX_FULL) :]
            internal_params[clean_key] = value
            logger.debug(f"Internal parameter (full): {key} → {clean_key}")
        # Check for alias prefix
        elif key.startswith(INTERNAL_PARAM_PREFIX_ALIAS):
            clean_key = key[len(INTERNAL_PARAM_PREFIX_ALIAS) :]
            internal_params[clean_key] = value
            logger.debug(f"Internal parameter (alias): {key} → {clean_key}")
        # Check for InputGenerator-level fields that are not FDF params
        elif key in _INPUT_GEN_FIELDS:
            internal_params[key] = value
            logger.debug(f"InputGenerator field in user_params (filtered): {key}")
        else:
            fdf_params[key] = value

    return fdf_params, internal_params


def normalize_internal_params(user_params: dict) -> dict:
    """
    Validate internal parameter names (v1.0.0+).

    Legacy unprefixed parameter names are NO LONGER SUPPORTED.
    All atomate2siesta-specific parameters must use prefix:
    - 'atomate2siesta_' (full) or 'a2s_' (alias)

    Args:
        user_params: User-provided parameters

    Returns:
        Dictionary (unchanged - validation only)

    Raises:
        ValueError: If legacy unprefixed parameter names are used

    Example:
        >>> params = {
        ...     "Mesh.Cutoff": "300 Ry",       # ✅ SIESTA parameter (no prefix needed)
        ...     "a2s_kpts": [4, 4, 4],         # ✅ atomate2siesta parameter (prefixed)
        ... }
        >>> normalize_internal_params(params)
        {'Mesh.Cutoff': '300 Ry', 'a2s_kpts': [4, 4, 4]}

        >>> params = {"kpts": [4, 4, 4]}  # ❌ Legacy unprefixed
        >>> normalize_internal_params(params)
        ValueError: Legacy parameter 'kpts' is no longer supported...
    """
    # Check for legacy unprefixed parameters
    legacy_params_found = []

    for key in user_params.keys():
        if key in LEGACY_INTERNAL_PARAMS:
            legacy_params_found.append(key)

    if legacy_params_found:
        # Build helpful error message with suggestions
        suggestions = []
        for old_key in legacy_params_found:
            new_key = LEGACY_INTERNAL_PARAMS[old_key]
            suggestions.append(
                f"  - '{old_key}' → '{new_key}' or '{INTERNAL_PARAM_PREFIX_FULL}{old_key.replace('a2s_', '')}'"
            )

        error_msg = (
            f"Legacy unprefixed parameter(s) detected: {', '.join(legacy_params_found)}\n\n"
            f"As of v1.0.0, all atomate2siesta-specific parameters MUST use prefix.\n"
            f"Use 'a2s_' (alias) or 'atomate2siesta_' (full):\n\n"
            + "\n".join(suggestions)
            + "\n\nSIESTA parameters (from SIESTA manual) do NOT need prefixes:\n"
            "  ✅ 'Mesh.Cutoff', 'PAO.BasisSize', 'Spin', etc.\n\n"
            "atomate2siesta shortcuts/controls REQUIRE prefixes:\n"
            "  ✅ 'a2s_kpts', 'a2s_pseudo_path', 'a2s_magnetic_ordering', etc."
        )

        raise ValueError(error_msg)

    # No changes - validation only
    return user_params


# Initialize FDF registry by instantiating all dataclasses
# This ensures the registry is populated before any validation happens
def _initialize_fdf_registry():
    """
    Initialize the FDF parameter registry.

    Instantiates all dataclasses to trigger FDF parameter registration.
    This must happen at module import time to ensure the registry is
    populated before setup_fdf_arguments() is called.
    """
    # Import all dataclass modules
    from atomate2.siesta.dataclass.basis_sets_and_projectors import (
        BasisSetsAndProjectors,
    )
    from atomate2.siesta.dataclass.kpoint_sampling import KPointSampling
    from atomate2.siesta.dataclass.exchange_correlation_functionals import (
        ExchangeCorrelationFunctionals,
    )
    from atomate2.siesta.dataclass.spin_settings import SpinSettings
    from atomate2.siesta.dataclass.scf_loop_parameters import SCFLoopParameters
    from atomate2.siesta.dataclass.real_space_grid_parameters import (
        RealSpaceGridParameters,
    )
    from atomate2.siesta.dataclass.hamiltonian_and_overlap_parameters import (
        HamiltonianAndOverlapParameters,
    )
    from atomate2.siesta.dataclass.electronic_structure_calculation_options import (
        ElectronicStructureCalculationOptions,
    )
    from atomate2.siesta.dataclass.density_of_states_and_band_structure import (
        DensityOfStatesAndBandStructure,
    )
    from atomate2.siesta.dataclass.chemical_analysis import ChemicalAnalysis
    from atomate2.siesta.dataclass.optical_properties import OpticalProperties
    from atomate2.siesta.dataclass.wannier90 import Wannier90
    from atomate2.siesta.dataclass.charge_dipole_electric_field import (
        ChargeDipoleElectricField,
    )
    from atomate2.siesta.dataclass.grids import Grids
    from atomate2.siesta.dataclass.auxiliary_force_field import AuxiliaryForceField
    from atomate2.siesta.dataclass.parallel_options import ParallelOptions
    from atomate2.siesta.dataclass.efficiency_options import EfficiencyOptions
    from atomate2.siesta.dataclass.denchar import Denchar
    from atomate2.siesta.dataclass.netcdf_options import NetcdfOptions
    from atomate2.siesta.dataclass.general_constraints import GeneralConstraints
    from atomate2.siesta.dataclass.phonon_calculations import PhononCalculations
    from atomate2.siesta.dataclass.dftu import DFTU
    from atomate2.siesta.dataclass.rttddft import RTTDDFT
    from atomate2.siesta.dataclass.structural_information import (
        StructuralInformationVersion1,
        StructuralInformationVersion2,
    )
    from atomate2.siesta.dataclass.molecular_dynamics_and_relaxation import (
        MolecularDynamicsAndRelaxation,
    )
    from atomate2.siesta.dataclass.solvers_and_performance_options import (
        SolversAndPerformanceOptions,
    )

    # Instantiate all dataclasses to trigger registration
    _ = GeneralSystemDescriptors()
    _ = Pseudopotentials()
    _ = BasisSetsAndProjectors()
    _ = KPointSampling()
    _ = ExchangeCorrelationFunctionals()
    _ = SpinSettings()
    _ = SCFLoopParameters()
    _ = RealSpaceGridParameters()
    _ = HamiltonianAndOverlapParameters()
    _ = ElectronicStructureCalculationOptions()
    _ = DensityOfStatesAndBandStructure()
    _ = ChemicalAnalysis()
    _ = OpticalProperties()
    _ = Wannier90()
    _ = ChargeDipoleElectricField()
    _ = Grids()
    _ = AuxiliaryForceField()
    _ = ParallelOptions()
    _ = EfficiencyOptions()
    _ = Denchar()
    _ = NetcdfOptions()
    _ = ExternalControlAndScripting()
    _ = GeneralConstraints()
    _ = PhononCalculations()
    _ = DFTU()
    _ = RTTDDFT()
    _ = StructuralInformationVersion1()
    _ = StructuralInformationVersion2()
    _ = MolecularDynamicsAndRelaxation()
    _ = SolversAndPerformanceOptions()


# Initialize registry on module import
_initialize_fdf_registry()

SIESTA_OUTPUT_FILE_NAME: str = "siesta.out"
SIESTA_FDF_FILE_NAME: str = "siesta.fdf"
SIESTA_STRUCTURE_FDF_FILE_NAME: str = "structure.fdf"
SIESTA_PARAMS_JSON_FILE_NAME: str = "siesta_parameters.json"
SIESTA_GEOMETRY_FILE_NAME: str = "siesta.XV"
SIESTA_VIBRA_FILE_NAME: str = "siesta_fcbuild.fdf"
SIESTA_OPTICAL_FILE_NAME: str = "siesta.EPSIMG"


logger = logging.getLogger(__name__)
console = Console()


class SiestaInputSet(InputSet):
    """
    A class to represent a set of SIESTA inputs.
    """

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = (
        VerbosityLevel.ERROR
    )  # Default to show info & errors messages

    def __init__(
        self,
        structure: Structure | Molecule,
        siesta_input=Siesta,
    ) -> None:
        """
        Construct the SiestaInputSet.

        Args:
            parameters (dict[str, Any]): The ASE parameters object for the calculation
            structure (Structure or Molecule): The Structure/Molecule objects to
                create the inputs for
        """
        logger.info("SiestaInputSet.__init__()")

        self._structure = structure
        self._siesta_input = siesta_input

        super().__init__(
            inputs={
                SIESTA_FDF_FILE_NAME: siesta_input,
                SIESTA_PARAMS_JSON_FILE_NAME: json.dumps(
                    self._siesta_input.parameters, cls=MontyEncoder
                ),
            }
        )

    @property
    def siesta_input(self) -> Siesta:
        """
        Get the Siesta object.
        """
        logger.info("SiestaInputSet.siesta_input()")
        return self[SIESTA_FDF_FILE_NAME]

    @property
    def params_json(self) -> str | slice | InputFile:
        """
        The JSON representation of the parameters dict.
        """
        logger.info("SiestaInputSet.params_json()")
        return self[SIESTA_PARAMS_JSON_FILE_NAME]

    def write_input(
        self,
        directory: str | Path,
        make_dir: bool = True,
        overwrite: bool = True,
        zip_inputs: bool = False,
    ):
        """
        Write SIESTA input files to a directory.

        Overrides InputSet.write_input() to properly write the FDF file
        using the Siesta calculator's write_input() method.

        Args:
            directory: Directory to write input files to.
            make_dir: Whether to create the directory if it doesn't exist.
            overwrite: Whether to overwrite files if they already exist.
            zip_inputs: Not used for SIESTA (kept for compatibility).
        """
        logger.info("SiestaInputSet.write_input()")
        directory = Path(directory)

        if make_dir:
            directory.mkdir(parents=True, exist_ok=True)

        # Change to directory to write files there
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(directory)

            # Write the FDF file using write_siesta_fdf()
            self.write_siesta_fdf(self._structure, directory=directory)

        finally:
            os.chdir(old_cwd)

    def write_siesta_fdf(self, structure: Structure, directory=None):
        """
        Writes SIESTA FDF input file and converts to JSON format.
        """
        logger.info("SiestaInputSet.write_siesta_fdf()")

        # DEBUG: Check what's in fdf_arguments before writing
        fdf_args = self.siesta_input["fdf_arguments"]
        logger.info(f"write_siesta_fdf: Total FDF arguments: {len(fdf_args)}")
        logger.debug(f"write_siesta_fdf: All FDF keys: {list(fdf_args.keys())}")

        ase_atoms = pymatgen_to_ase(structure=structure)
        # Ensure the latest calculator settings are used
        self.siesta_input.write_input(ase_atoms, "energy")  #'density'
        siesta_fdf_to_json(
            "siesta.fdf", json_output_path=SIESTA_PARAMS_JSON_FILE_NAME
        )  # "siesta_parameters.json"

        # Write CIF files with ghost atom support
        # ASE/FileIOCalculator may write CIF files, so we overwrite them
        from pathlib import Path

        from atomate2.siesta.sets.utils.structure_io import write_cif_with_ghost

        # Overwrite any CIF files in current directory with ghost-aware versions
        cwd = Path.cwd()
        for cif_file in cwd.glob("*.cif"):
            write_cif_with_ghost(structure, cif_file)


@dataclass
class SiestaInputGenerator(InputGenerator):
    """
       A class to generate SIESTA input sets.

    Attributes:
        user_params (OrderedDict[str, Any]): Updates the default parameters for the SIESTA calculator.
        user_kpoints_settings (OrderedDict[str, Any]): Settings used to create the k-grid parameters for SIESTA.
        fdf_arguments (OrderedDict[str, Any]): Explicitly given fdf arguments using SIESTA keywords as in the manual.
            List values are written as fdf blocks with each element on a separate line, while tuples write elements
            on a single line. ASE units are assumed.
            Example:
            ```python
            fdf_arguments={union() {
                'DM.MixingWeight': 0.1,
                'MaxSCFIterations': 100,
                'XC.mix': [2, 'LDA CA 0.5 0.75', 'GGA PBE 0.5 0.25']
            }
            ```
            Produces:
            ```
            DM.MixingWeight 0.1
            MaxSCFIterations 100
            %block XC.mix
                2
                LDA CA 0.5 0.75
                GGA PBE 0.5 0.25
            %endblock XC.mix
            ```
        CONSOLE_VERBOSITY (VerbosityLevel): Controls console output verbosity, defaults to ERROR.
        enable_lua (bool): Flag to enable Lua settings, defaults to True.
        perform_siesta_default_basis (bool): Whether to use SIESTA's default basis, defaults to True.
        energy_shift (float): Energy shift for basis set generation in eV, defaults to 0.01.
        basis_set_size (str): Basis set size, defaults to 'SZ'.
        basis_set_block (Optional[List[PAOBasisBlock]]): Custom basis set block, defaults to None.
        xc (str): Exchange-correlation functional, defaults to 'PBE'.
        mesh_cutoff (float): Mesh cutoff energy in eV, defaults to 100.0.
        kpts (List[int]): K-point grid, defaults to [1, 1, 1].
        species (List[Species]): List of species for the calculation, defaults to empty list.
        pseudo_path (Optional[str]): Path to pseudopotential files, defaults to SIESTA_PP_PATH or SETTINGS.SIESTA_PP_PATH.

    Methods:
        get_input_set(structure: Structure | Molecule | None = None, prev_dir: PathLike | None = None) -> SiestaInputSet:
            Generates a SiestaInputSet object for the given structure or from a previous calculation directory.
            Raises ValueError if no structure can be determined.
        _read_previous(prev_dir: PathLike | None = None) -> tuple[Structure | Molecule | None, dict[str, Any], dict[str, Any]]:
            Reads previous calculation results from a specified directory, returning the structure, parameters, and results.
        _get_input_parameters(structure: Structure | Molecule, prev_parameters: dict[str, Any] | None = None) -> dict[str, Any]:
            Generates SIESTA input parameters for a given structure, incorporating user parameters and settings for basis, spin, XC, mesh, k-points, and pseudopotentials.
        get_parameter_updates(structure: Structure | Molecule, prev_parameters: dict[str, Any]) -> dict[str, Any]:
            Updates parameters for a given calculation type based on the structure and previous parameters.
        setup_fdf_arguments(user_params):
            Sets up fdf arguments for SIESTA input by updating with user-provided parameters and adding default settings for 'LongOutput' and 'WriteForces'.

    """

    # Class-level verbosity control
    CONSOLE_VERBOSITY: VerbosityLevel = (
        VerbosityLevel.ERROR
    )  # Default to show info & errors messages

    # Tier-based initialization control
    tier: str = (
        "intermediate"  # Calculation tier: basic, intermediate, advanced, expert
    )
    enabled_modules: Optional[List[str]] = None  # Override: explicitly enable modules
    disabled_modules: Optional[List[str]] = None  # Override: explicitly disable modules

    enable_lua: bool = True  # Flag to control Lua settings
    force_unknown: bool = (
        False  # Allow unknown FDF parameters not registered by dataclasses
    )
    user_params: Optional[OrderedDict[str, Any]] = field(default_factory=OrderedDict)
    user_kpoints_settings: OrderedDict[str, Any] = field(default_factory=OrderedDict)
    # Parameters for Basis
    perform_siesta_default_basis: bool = True
    energy_shift: float = field(default=0.01)  # in eV
    basis_set_size: str = field(default="SZ")  # For Default Siesta Basis
    basis_set_block: Optional[List[PAOBasisBlock]] = field(
        default=None
    )  # For Siesta Basis Block and Made optional
    xc: str = field(default="PBE")
    mesh_cutoff: float = field(default=100.0)  # in eV
    kpts: List[int] = field(default_factory=lambda: [1, 1, 1])
    # fdf_arguments: Dict[str, Any] = field(default_factory=dict)
    fdf_arguments: OrderedDict[str, Any] = field(default_factory=OrderedDict)
    species: List[Species] = field(default_factory=list)
    pseudo_path: Optional[str] = field(
        default_factory=lambda: os.getenv("SIESTA_PP_PATH")
        or getattr(SETTINGS, "SIESTA_PP_PATH", None),
        metadata={
            "description": "Path to pseudopotential files, defaults to SIESTA_PP_PATH or SETTINGS.SIESTA_PP_PATH."
        },
    )

    # Instance of GeneralSystemDescriptors
    # AA
    # general_system_descriptors: GeneralSystemDescriptors = field(default_factory=GeneralSystemDescriptors)
    # spin_settings : SpinSettings = field(default_factory=SpinSettings)

    def __post_init__(self):
        """Apply tier-specific default parameters and validate parameter names."""
        from atomate2.siesta.sets.tiers import TIER_DEFAULTS

        # Store ORIGINAL user params before any defaults are applied
        # This allows us to distinguish explicit user params from maker defaults
        self._explicit_user_params = (
            OrderedDict(self.user_params) if self.user_params else OrderedDict()
        )

        # Apply tier defaults, then merge with user_params (user params take precedence)
        if self.tier in TIER_DEFAULTS:
            tier_defaults = OrderedDict(TIER_DEFAULTS[self.tier])
            if self.user_params:
                # Merge: tier defaults first, then user params override
                merged_params = OrderedDict(tier_defaults)
                merged_params.update(self.user_params)
                self.user_params = merged_params
                logger.info(
                    f"Applied tier '{self.tier}' defaults and merged with "
                    f"{len(self._explicit_user_params)} user params"
                )
            else:
                # No user params, use tier defaults only
                self.user_params = tier_defaults
                logger.info(f"Applied tier '{self.tier}' defaults: {self.user_params}")

        # Validate internal parameter names AFTER tier merging (v1.0.0: strict validation)
        # This ensures legacy unprefixed parameters are caught early
        if self.user_params:
            normalize_internal_params(self.user_params)

        # Separate FDF parameters from internal parameters before validation
        # Internal params (a2s_*, atomate2siesta_*) should not be validated as FDF
        if self.user_params:
            from atomate2.siesta.dataclass.base import merge_fdf_parameters

            # Filter out internal parameters - they're not FDF parameters
            fdf_params, _internal_params = filter_internal_params(self.user_params)

            # Validate only the FDF parameters
            # This will raise ValueError if unknown parameters found and force_unknown=False
            if fdf_params:
                merge_fdf_parameters(fdf_params, force_unknown=self.force_unknown)

    def get_input_set(
        self,
        structure: Structure | Molecule | None = None,
        prev_dir: PathLike | None = None,
    ) -> SiestaInputSet:
        """
        Generate a SiestaInputSet object.

        Args:
            structure (Structure or Molecule, optional): Structure or Molecule to generate the input set for.
            prev_dir (str or Path, optional): Path to the previous working directory.

        Returns:
            SiestaInputSet: The input set for the calculation of the structure.

        Raises:
            ValueError: If no structure can be determined to generate the input set.
        """
        logger.info("SiestaInputGenerator.get_input_set()")

        prev_structure, prev_parameters, _ = self._read_previous(prev_dir)

        structure = structure or prev_structure

        if structure is None:
            raise ValueError("No structure can be determined to generate the input set")

        parameters = self._get_input_parameters(structure, prev_parameters)
        # parameters update(self.user_params)
        # parameters = Siesta()

        return SiestaInputSet(structure=structure, siesta_input=parameters)

    @staticmethod
    def _read_previous(
        prev_dir: PathLike | None = None,
    ) -> tuple[Structure | Molecule | None, dict[str, Any], dict[str, Any]]:
        """
        Read in previous calculation results.

        Args:
            prev_dir (str or Path, optional): The previous directory for the calculation.

        Returns:
            tuple: A tuple containing the previous structure (Structure or Molecule or None),
                   previous parameters (dict), and previous results (dict).
        """
        logger.info("SiestaInputGenerator._read_previous()")

        prev_structure: Structure | Molecule | None = None
        prev_params: dict[str, Any] = {}
        prev_results: dict[str, Any] = {}

        if prev_dir:
            # strip hostname from the directory (not good, works only with run_locally.
            # Should be checked with Fireworks, will not for sure work with
            # jobflow_remote)
            split_prev_dir = str(prev_dir).split(":")[-1]

            # Search for siesta_parameters.json in multiple locations
            # (handles both compressed and uncompressed files)
            import gzip
            from pathlib import Path

            param_file_path = None
            split_prev_path = Path(split_prev_dir)

            # Check main directory (uncompressed)
            if (split_prev_path / "siesta_parameters.json").exists():
                param_file_path = split_prev_path / "siesta_parameters.json"
            # Check main directory (compressed)
            elif (split_prev_path / "siesta_parameters.json.gz").exists():
                param_file_path = split_prev_path / "siesta_parameters.json.gz"
            # Check compressed subfolder
            elif (
                split_prev_path / "siesta_compressed" / "siesta_parameters.json.gz"
            ).exists():
                param_file_path = (
                    split_prev_path / "siesta_compressed" / "siesta_parameters.json.gz"
                )

            if param_file_path is None:
                # Check if prev_dir contains dry_run_output subdirectory
                # (dry_run creates files in prev_dir/dry_run_output/*/)
                dry_run_subdir = split_prev_path / "dry_run_output"
                if dry_run_subdir.exists() and dry_run_subdir.is_dir():
                    # Find the first subdirectory in dry_run_output
                    dry_run_dirs = [d for d in dry_run_subdir.iterdir() if d.is_dir()]
                    if dry_run_dirs:
                        # Check in the dry_run subdirectory
                        actual_dir = dry_run_dirs[0]
                        if (actual_dir / "siesta_parameters.json").exists():
                            param_file_path = actual_dir / "siesta_parameters.json"
                            split_prev_path = (
                                actual_dir  # Update path for later file reading
                            )
                        elif (actual_dir / "siesta_parameters.json.gz").exists():
                            param_file_path = actual_dir / "siesta_parameters.json.gz"
                            split_prev_path = actual_dir

                if param_file_path is None:
                    # Check if this is a dry-run directory (contains .cif/.xsf files but no siesta_parameters.json)
                    # In dry-run mode, we don't have previous calculation parameters to read
                    dry_run_files = list(split_prev_path.glob("*.cif")) + list(
                        split_prev_path.glob("*.xsf")
                    )
                    if dry_run_files:
                        logger.info(
                            f"Skipping prev_dir reading - appears to be dry-run directory: {split_prev_dir}"
                        )
                        return prev_structure, prev_params, prev_results
                    else:
                        raise FileNotFoundError(
                            f"Could not find siesta_parameters.json in {split_prev_dir} "
                            f"(checked main directory, dry_run_output/, and siesta_compressed/ subfolder)"
                        )

            # Read with appropriate method
            if str(param_file_path).endswith(".gz"):
                with gzip.open(param_file_path, "rt") as param_file:
                    prev_params = json.load(param_file, cls=MontyDecoder)
            else:
                with open(param_file_path) as param_file:
                    prev_params = json.load(param_file, cls=MontyDecoder)

            try:
                # siesta_output: Sequence[Structure | Molecule] = read_siesta_output(
                #    f"{split_prev_dir}/siesta.out", index=slice(-1, None))
                # prev_structure = siesta_output[0]
                siesta_output: Sequence[
                    Structure | Molecule
                ] = read_siesta_output_structure(
                    f"{split_prev_dir}/siesta.XV", index=slice(-1, None)
                )
                prev_structure = siesta_output[0] if siesta_output else None

                if prev_structure is not None:
                    prev_results = prev_structure.properties
                    # Only Structure has site_properties, Molecule does not
                    if isinstance(prev_structure, Structure):
                        prev_results.update(prev_structure.site_properties)
            except (IndexError, SiestaParseError, FileNotFoundError):
                # If siesta.XV doesn't exist (dry-run mode), try reading from .cif/.xsf
                try:
                    # Structure is already imported at module level (line 25)
                    cif_files = list(split_prev_path.glob("*.cif"))
                    xsf_files = list(split_prev_path.glob("*.xsf"))
                    structure_files = cif_files + xsf_files

                    if structure_files:
                        prev_structure = Structure.from_file(str(structure_files[0]))
                        logger.info(
                            f"Read structure from dry-run file: {structure_files[0].name}"
                        )
                except Exception as e:
                    logger.warning(f"Could not read structure from prev_dir: {e}")
                    pass

        return prev_structure, prev_params, prev_results

    def _get_input_parameters(
        self,
        structure: Structure | Molecule,
        prev_parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate SIESTA input parameters.

        Args:
            structure (Structure or Molecule): The system to generate input parameters for.
            prev_parameters (dict[str, Any], optional): Previous calculation parameters.

        Returns:
            dict: A dictionary of SIESTA input parameters.
        """
        logger.info("SiestaInputGenerator._get_input_parameters()")

        # Ensure self.user_params is an OrderedDict
        if not hasattr(self, "user_params") or self.user_params is None:
            self.user_params = OrderedDict()

        # Convert all keys to lowercase for case-insensitive matching
        user_params_lower = {
            k.lower(): v.lower() if isinstance(v, str) else v
            for k, v in self.user_params.items()
        }
        user_params_ = OrderedDict(
            {k: v if isinstance(v, str) else v for k, v in self.user_params.items()}
        )

        if user_params_:
            # Filter out non-SIESTA parameters:
            # 1. Internal atomate2siesta parameters (with prefix)
            # 2. Pseudopotential metadata (for path construction only)
            # Note: xc.functional and xc.authors ARE real SIESTA parameters, so we keep them

            fdf_params_only, internal_params_found = filter_internal_params(
                user_params_
            )

            # Also filter out unprefixed pseudo metadata (legacy support)
            non_siesta_params = {
                "pseudo_base_path",
                "pseudo_family",
                "pseudo_version",
                "pseudo_quality",
                "pseudo_relativistic",
                "pseudo_fdf_arguments",
            }

            # Create filtered dict for display (only actual SIESTA parameters)
            siesta_params_display = {
                k: v
                for k, v in fdf_params_only.items()
                if k.lower() not in non_siesta_params
            }

            # Create a Rich Table for fancy display of SIESTA parameters only
            if siesta_params_display:
                # Check SIESTA_SHOW_PARAMETER_EVOLUTION setting
                display_level = SETTINGS.SIESTA_SHOW_PARAMETER_EVOLUTION

                if display_level == "none":
                    # User disabled parameter evolution display - skip all output
                    # but still store initial params for potential later use
                    self._initial_user_params = dict(siesta_params_display)
                else:
                    # Separate explicit user params from maker defaults
                    explicit_params = {}
                    maker_defaults = {}

                    for key, value in siesta_params_display.items():
                        if key in self._explicit_user_params:
                            explicit_params[key] = value
                        else:
                            maker_defaults[key] = value

                    # Display explicit user parameters if any
                    if explicit_params:
                        table = Table(
                            title="[bold cyan]Explicit User-Provided Parameters[/bold cyan]",
                            show_header=True,
                            header_style="bold magenta",
                        )
                        table.add_column("Parameter", style="cyan", justify="left")
                        table.add_column("Value", style="magenta", justify="left")
                        table.add_column("Source", style="yellow", justify="left")

                        for key, value in explicit_params.items():
                            # Determine source
                            if key in internal_params_found:
                                source = "Internal (atomate2siesta)"
                            else:
                                source = "User (SIESTA FDF)"
                            table.add_row(key.upper(), str(value), source)
                        console.print(table)

                    # Display maker defaults if any
                    if maker_defaults:
                        table = Table(
                            title="[bold blue]Maker Default Parameters[/bold blue]",
                            show_header=True,
                            header_style="bold blue",
                        )
                        table.add_column("Parameter", style="blue", justify="left")
                        table.add_column("Value", style="cyan", justify="left")
                        table.add_column("Source", style="dim yellow", justify="left")

                        for key, value in maker_defaults.items():
                            # Determine source
                            if key in internal_params_found:
                                source = "Internal (atomate2siesta)"
                            else:
                                source = "Maker (SIESTA FDF)"
                            table.add_row(key.upper(), str(value), source)
                        console.print(table)

                    # Store initial parameters for later comparison
                    self._initial_user_params = dict(siesta_params_display)

        # Initialize fdf_arguments
        self.setup_fdf_arguments(user_params=user_params_)

        # =====================================================================
        # PSEUDO_PATH PARSING (must happen BEFORE module initialization)
        # =====================================================================
        # If user provided explicit pseudo_path, parse it to extract XC metadata
        # This allows XC parameters to be auto-detected and used by ExchangeCorrelationFunctionals
        if "pseudo_path" in user_params_:
            parsed_metadata = Pseudopotentials.parse_pseudo_path(
                user_params_["pseudo_path"]
            )
            if parsed_metadata:
                # Add parsed XC to user_params if not already specified
                if (
                    "xc_functional" not in user_params_lower
                    and "xc.functional" not in user_params_lower
                ):
                    user_params_["xc.functional"] = parsed_metadata["xc_functional"]
                    user_params_lower["xc.functional"] = parsed_metadata[
                        "xc_functional"
                    ].lower()
                if (
                    "xc_authors" not in user_params_lower
                    and "xc.authors" not in user_params_lower
                ):
                    user_params_["xc.authors"] = parsed_metadata["xc_authors"]
                    user_params_lower["xc.authors"] = parsed_metadata[
                        "xc_authors"
                    ].lower()
                if self.CONSOLE_VERBOSITY.value >= VerbosityLevel.INFO.value:
                    console.print(
                        f"[green]Auto-detected XC from pseudo_path: {parsed_metadata['xc_functional']}/{parsed_metadata['xc_authors']}[/green]"
                    )

        # =====================================================================
        # REGISTRY-BASED AUTO-INITIALIZATION
        # =====================================================================
        # Determine which modules to initialize based on tier AND user_params
        modules_to_init = self._get_modules_to_initialize(user_params=user_params_)

        # Initialize all modules in priority order
        self._initialize_modules(
            modules_to_init, structure, user_params_, user_params_lower
        )

        # =====================================================================
        # SPECIAL HANDLING (modules with unique initialization)
        # =====================================================================
        # General system descriptors (no FDF arguments, just validation)
        self.general_system_descriptors = GeneralSystemDescriptors()
        self.general_system_descriptors.validate_label_and_name()

        # Pseudopotentials (returns path, not FDF arguments)
        self.pseudo_path = SETTINGS.SIESTA_PP_PATH

        # Prepare pseudopotential parameters with XC information for automatic path construction
        pseudo_params = dict(
            user_params_
        )  # Copy user params (includes any parsed XC info)

        # Extract XC information from user_params (includes auto-detected XC from pseudo_path parsing)
        xc_functional = user_params_lower.get(
            "xc_functional", user_params_lower.get("xc.functional", "GGA")
        )
        xc_authors = user_params_lower.get(
            "xc_authors",
            user_params_lower.get(
                "xc.authors", "PBE"
            ),  # Match default in ExchangeCorrelationFunctionals
        )

        # Add XC info to pseudo params if not already present
        if "xc_functional" not in pseudo_params:
            pseudo_params["xc_functional"] = xc_functional
        if "xc_authors" not in pseudo_params:
            pseudo_params["xc_authors"] = xc_authors

        # Use SIESTA_PP_PATH as pseudo_base_path if not explicitly set
        if "pseudo_base_path" not in pseudo_params and not pseudo_params.get(
            "pseudo_path"
        ):
            # Try to use parent directory of SIESTA_PP_PATH as base path
            if self.pseudo_path and os.path.isdir(self.pseudo_path):
                parent_dir = os.path.dirname(self.pseudo_path)
                if os.path.isdir(parent_dir):
                    pseudo_params["pseudo_base_path"] = parent_dir

        pseudos_settings = Pseudopotentials.setup_pseudos(
            pseudo_params, default_pseudo_path=self.pseudo_path
        )
        # Store for later XC validation
        self.pseudos_settings = pseudos_settings

        # =====================================================================
        # XC CONSISTENCY VALIDATION
        # =====================================================================
        # Validate that XC functional matches pseudopotentials
        if pseudos_settings and pseudos_settings.pseudo_path:
            # Extract XC from FDF arguments (already set by ExchangeCorrelationFunctionals module)
            xc_functional = self.fdf_arguments.get("XC.Functional", "GGA")
            xc_authors = self.fdf_arguments.get("XC.Authors", "PBE")

            # Validate consistency
            Pseudopotentials.validate_xc_consistency(
                pseudo_path=pseudos_settings.pseudo_path,
                xc_functional=xc_functional,
                xc_authors=xc_authors,
                structure=structure,
            )

        # For lua
        if self.enable_lua:  # Enable Lua
            if (
                get_verbosity_value(self.CONSOLE_VERBOSITY)
                >= VerbosityLevel.VERBOSE.value
            ):
                console.print(
                    "[yellow]Using Lua settings due to used [bold]Maker[/bold]...[/yellow]"
                )
            self.lua_settings = ExternalControlAndScripting.setup_lua_settings(
                user_params_
            )
            self.fdf_arguments.update(self.lua_settings.lua_fdf_arguments)
        else:
            if (
                get_verbosity_value(self.CONSOLE_VERBOSITY)
                >= VerbosityLevel.VERBOSE.value
            ):
                console.print(
                    "[yellow]Skipping Lua settings due to used [bold]Maker[/bold]...[/yellow]"
                )

        # To check
        self.setup_fdf_arguments(user_params=user_params_)

        # Call get_parameter_updates() to allow subclasses to add FDF parameters
        # (e.g., RelaxSetGenerator adds relaxation-specific FDF)
        # This updates self.fdf_arguments in place
        self.get_parameter_updates(structure=structure, prev_parameters={})

        # Display final parameter state before creating Siesta object
        # This shows all changes from user → dataclass → powerups
        self._display_parameter_evolution(
            stage="final", final_fdf_params=dict(self.fdf_arguments)
        )

        # Now create Siesta object with the UPDATED fdf_arguments
        # (includes both module FDF and get_parameter_updates() additions)
        parameters = Siesta(
            label=self.general_system_descriptors.system_label,
            xc=self.xc,
            mesh_cutoff=self.mesh_cutoff * Ry,
            energy_shift=self.energy_shift * Ry,
            basis_set=self.basis_set_size,
            # basis_set= basis_set_size,
            kpts=self.kpts,
            fdf_arguments=self.fdf_arguments,
            species=self.species,
            pseudo_path=pseudos_settings.pseudo_path,
            # spin = spin,
            # pseudo_path = self.pseudo_path, #"/home/akhtar/.siesta/pseudos/ONCVPSP-PBEsol-FR-PDv0.4-Standard/",
        )

        return parameters

    def _display_parameter_evolution(
        self, stage: str = "after_dataclass", final_fdf_params: dict = None
    ):
        """
        Display parameter changes at different stages of processing.

        Tracks and displays:
        1. Initial user parameters (from user_params)
        2. After dataclass processing (auto-generated/modified by dataclasses)
        3. After flow modifications (powerups/flow changes)
        4. Final FDF parameters (what gets written to file)

        Display level controlled by SETTINGS.SIESTA_SHOW_PARAMETER_EVOLUTION:
        - 'none': No display
        - 'user': Only initial user parameters
        - 'diff': Only changes (added/modified)
        - 'summary': Initial + changes summary (default)
        - 'full': All stages with complete final table

        Args:
            stage: Processing stage ("after_dataclass", "after_powerups", "final")
            final_fdf_params: Final FDF parameters dict (for "final" stage only)
        """
        # Check verbosity level first
        verbosity_value = get_verbosity_value(self.CONSOLE_VERBOSITY)
        if verbosity_value < VerbosityLevel.INFO.value:
            return  # Skip if not verbose enough

        # Check SIESTA_SHOW_PARAMETER_EVOLUTION setting
        display_level = SETTINGS.SIESTA_SHOW_PARAMETER_EVOLUTION

        if display_level == "none":
            return  # User disabled parameter evolution display

        # Get current FDF parameters
        current_fdf = dict(self.fdf_arguments)

        # Remove internal tracking parameters
        internal_tracking = ["_user_set_spin", "_user_requested_cutoff"]
        for key in internal_tracking:
            current_fdf.pop(key, None)

        # Get initial parameters if stored
        initial_params = getattr(self, "_initial_user_params", {})

        if stage == "after_dataclass" and initial_params:
            # Show what changed after dataclass processing
            added_params = {}
            modified_params = {}
            unchanged_params = {}

            for key, value in current_fdf.items():
                key_upper = key.upper()
                # Check if this was in initial params
                if key_upper in {k.upper() for k in initial_params.keys()}:
                    # Find the original key (might have different case)
                    orig_key = next(
                        (k for k in initial_params.keys() if k.upper() == key_upper),
                        None,
                    )
                    if orig_key and str(initial_params[orig_key]) != str(value):
                        modified_params[key] = {
                            "old": initial_params[orig_key],
                            "new": value,
                        }
                    elif orig_key:
                        unchanged_params[key] = value
                else:
                    added_params[key] = value

            # Display based on level setting
            # Skip display if level is 'user' (only show initial, not changes)
            if display_level == "user":
                return

            # Display changes (for 'diff', 'summary', and 'full' levels)
            if added_params or modified_params:
                from rich.panel import Panel
                from rich import box

                console.print(
                    Panel.fit(
                        "[bold yellow]Parameter Evolution After Dataclass Processing[/bold yellow]",
                        border_style="yellow",
                        box=box.DOUBLE,
                    )
                )

                # Show added parameters
                if added_params:
                    table_added = Table(
                        title="[bold green]Parameters Added by Dataclasses[/bold green]",
                        show_header=True,
                        header_style="bold green",
                        border_style="green",
                    )
                    table_added.add_column("Parameter", style="cyan", justify="left")
                    table_added.add_column("Value", style="green", justify="left")
                    table_added.add_column("Source", style="yellow", justify="left")

                    for key, value in sorted(added_params.items()):
                        # Try to identify which dataclass added this
                        source = "Auto-generated"
                        if "kgrid" in key.lower() or "kpts" in key.lower():
                            source = "KPointSampling"
                        elif "pao" in key.lower() or "basis" in key.lower():
                            source = "BasisSetsAndProjectors"
                        elif "mesh" in key.lower():
                            source = "MeshParameters"
                        elif "md" in key.lower() or "relax" in key.lower():
                            source = "MDAndRelaxation"
                        elif "scf" in key.lower():
                            source = "SCFLoopParameters"
                        elif "spin" in key.lower() or "dm.init" in key.lower():
                            source = "SpinSettings"

                        table_added.add_row(key.upper(), str(value), source)

                    console.print(table_added)

                # Show modified parameters
                if modified_params:
                    table_modified = Table(
                        title="[bold yellow]Parameters Modified by Dataclasses[/bold yellow]",
                        show_header=True,
                        header_style="bold yellow",
                        border_style="yellow",
                    )
                    table_modified.add_column("Parameter", style="cyan", justify="left")
                    table_modified.add_column(
                        "Initial Value", style="red", justify="left"
                    )
                    table_modified.add_column("→", style="white", justify="center")
                    table_modified.add_column(
                        "Current Value", style="green", justify="left"
                    )

                    for key, values in sorted(modified_params.items()):
                        table_modified.add_row(
                            key.upper(), str(values["old"]), "→", str(values["new"])
                        )

                    console.print(table_modified)

                # Show summary
                console.print("\n[bold]Summary:[/bold]")
                console.print(f"  • Initial user parameters: {len(initial_params)}")
                console.print(
                    f"  • Added by dataclasses: [green]{len(added_params)}[/green]"
                )
                console.print(
                    f"  • Modified by dataclasses: [yellow]{len(modified_params)}[/yellow]"
                )
                console.print(f"  • Unchanged: {len(unchanged_params)}")
                console.print(
                    f"  • Total current parameters: [bold]{len(current_fdf)}[/bold]\n"
                )

        elif stage == "final" and final_fdf_params:
            # Stage 4: Display final FDF parameters before writing to file
            from rich.panel import Panel
            from rich import box

            # Store previous state for comparison
            prev_fdf = getattr(
                self, "_after_dataclass_params", dict(self.fdf_arguments)
            )

            # Remove internal tracking parameters
            internal_tracking = ["_user_set_spin", "_user_requested_cutoff"]
            for key in internal_tracking:
                final_fdf_params.pop(key, None)
                prev_fdf.pop(key, None)

            # Find changes from powerups/flow modifications
            powerup_added = {}
            powerup_modified = {}
            powerup_removed = {}

            for key, value in final_fdf_params.items():
                key_upper = key.upper()
                # Check if this was in previous FDF
                if key_upper in {k.upper() for k in prev_fdf.keys()}:
                    orig_key = next(
                        (k for k in prev_fdf.keys() if k.upper() == key_upper), None
                    )
                    if orig_key and str(prev_fdf[orig_key]) != str(value):
                        powerup_modified[key] = {
                            "old": prev_fdf[orig_key],
                            "new": value,
                        }
                else:
                    powerup_added[key] = value

            # Check for removed parameters
            for key in prev_fdf.keys():
                if key.upper() not in {k.upper() for k in final_fdf_params.keys()}:
                    powerup_removed[key] = prev_fdf[key]

            console.print(
                Panel.fit(
                    "[bold blue]Final FDF Parameters - Ready to Write[/bold blue]",
                    border_style="blue",
                    box=box.DOUBLE,
                )
            )

            if powerup_added or powerup_modified or powerup_removed:
                console.print(
                    "\n[bold cyan]Changes from Powerups/Flow Modifications:[/bold cyan]\n"
                )

                # Show added parameters
                if powerup_added:
                    table_added = Table(
                        title="[bold green]Parameters Added by Powerups/Flows[/bold green]",
                        show_header=True,
                        header_style="bold green",
                        border_style="green",
                    )
                    table_added.add_column("Parameter", style="cyan", justify="left")
                    table_added.add_column("Value", style="green", justify="left")

                    for key, value in sorted(powerup_added.items()):
                        table_added.add_row(key.upper(), str(value))

                    console.print(table_added)

                # Show modified parameters
                if powerup_modified:
                    table_modified = Table(
                        title="[bold yellow]Parameters Modified by Powerups/Flows[/bold yellow]",
                        show_header=True,
                        header_style="bold yellow",
                        border_style="yellow",
                    )
                    table_modified.add_column("Parameter", style="cyan", justify="left")
                    table_modified.add_column("Before", style="red", justify="left")
                    table_modified.add_column("→", style="white", justify="center")
                    table_modified.add_column("After", style="green", justify="left")

                    for key, values in sorted(powerup_modified.items()):
                        table_modified.add_row(
                            key.upper(), str(values["old"]), "→", str(values["new"])
                        )

                    console.print(table_modified)

                # Show removed parameters (if any)
                if powerup_removed:
                    table_removed = Table(
                        title="[bold red]Parameters Removed[/bold red]",
                        show_header=True,
                        header_style="bold red",
                        border_style="red",
                    )
                    table_removed.add_column("Parameter", style="cyan", justify="left")
                    table_removed.add_column(
                        "Previous Value", style="red", justify="left"
                    )

                    for key, value in sorted(powerup_removed.items()):
                        table_removed.add_row(key.upper(), str(value))

                    console.print(table_removed)

            # Show final summary table with ALL parameters (only for 'full' level)
            if display_level == "full":
                from rich.text import Text

                table_final = Table(
                    title="[bold blue]Complete Final FDF Parameters[/bold blue]",
                    show_header=True,
                    header_style="bold blue",
                    border_style="dim blue",
                    show_lines=False,  # Remove internal lines for compactness
                    padding=(0, 1),  # Reduce padding
                    collapse_padding=True,
                )
                table_final.add_column("#", style="dim", justify="right", width=3)
                table_final.add_column(
                    "Parameter", style="cyan", justify="left", max_width=35
                )
                table_final.add_column(
                    "Value", style="white", justify="left", max_width=40
                )
                table_final.add_column("Src", style="yellow", justify="center", width=4)

                initial_params = getattr(self, "_initial_user_params", {})

                # Filter out metadata comments (lines starting with #)
                real_params = {
                    k: v for k, v in final_fdf_params.items() if not k.startswith("#")
                }

                for idx, (key, value) in enumerate(sorted(real_params.items()), 1):
                    # Determine status
                    key_upper = key.upper()
                    if key_upper in {k.upper() for k in initial_params.keys()}:
                        status = "U"  # User
                        status_style = "bold green"
                    elif key in powerup_added or key_upper in {
                        k.upper() for k in powerup_added.keys()
                    }:
                        status = "P"  # Powerup
                        status_style = "bold yellow"
                    elif key in powerup_modified or key_upper in {
                        k.upper() for k in powerup_modified.keys()
                    }:
                        status = "M"  # Modified
                        status_style = "bold yellow"
                    else:
                        status = "A"  # Auto
                        status_style = "dim"

                    # Format value for display (truncate if too long, remove SIESTA DEFAULT comments)
                    value_str = str(value)

                    # Remove SIESTA DEFAULT VALUE comments for cleaner display
                    if "# SIESTA DEFAULT VALUE" in value_str:
                        value_str = value_str.split("# SIESTA DEFAULT VALUE")[0].strip()

                    # Truncate if still too long
                    if len(value_str) > 40:
                        value_str = value_str[:37] + "..."

                    # Color code the status
                    status_text = Text(status, style=status_style)

                    table_final.add_row(str(idx), key.upper(), value_str, status_text)

                console.print("\n")
                console.print(table_final)

                # Add legend for status codes
                console.print(
                    "\n[dim]Status: [bold green]U[/bold green]=User, [bold yellow]P[/bold yellow]=Powerup, [bold yellow]M[/bold yellow]=Modified, A=Auto[/dim]"
                )

            # Final statistics
            console.print("\n[bold]Final Parameter Statistics:[/bold]")
            console.print(
                f"  • Total parameters in FDF: [bold cyan]{len(final_fdf_params)}[/bold cyan]"
            )
            console.print(
                f"  • From user: {len([k for k in final_fdf_params if k.upper() in {p.upper() for p in initial_params}])}"
            )
            console.print(
                f"  • Auto-generated: {len(final_fdf_params) - len(initial_params)}"
            )
            if powerup_added:
                console.print(
                    f"  • Added by powerups: [green]{len(powerup_added)}[/green]"
                )
            if powerup_modified:
                console.print(
                    f"  • Modified by powerups: [yellow]{len(powerup_modified)}[/yellow]"
                )
            if powerup_removed:
                console.print(
                    f"  • Removed by powerups: [red]{len(powerup_removed)}[/red]"
                )
            console.print("")

            # Write parameter evolution log to file
            from atomate2.siesta.sets.utils import write_parameter_evolution_log

            log_file = write_parameter_evolution_log(
                log_file_path="parameter_evolution.log",
                explicit_user_params=self._explicit_user_params
                if hasattr(self, "_explicit_user_params")
                else {},
                initial_params=initial_params,
                after_dataclass_params=self._after_dataclass_params
                if hasattr(self, "_after_dataclass_params")
                else {},
                final_fdf_params=final_fdf_params,
                powerup_added=powerup_added,
                powerup_modified=powerup_modified,
                powerup_removed=powerup_removed,
            )
            console.print(
                f"[dim]Parameter evolution log written to: {log_file.absolute()}[/dim]"
            )

    def get_parameter_updates(
        self,
        structure: Structure | Molecule,
        prev_parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Update the parameters for a given calculation type.

        Args:
            structure (Structure or Molecule): The system to run.
            prev_parameters (dict[str, Any]): Previous calculation parameters.

        Returns:
            dict: A dictionary of updates to apply to the parameters.
        """
        logger.info("SiestaInputGenerator.get_parameter_updates()")
        return prev_parameters

    def setup_fdf_arguments(self, user_params):
        """
        Set up fdf arguments for SIESTA input.

        Uses the FDF registry system to identify known vs unknown FDF parameters.
        Known parameters are handled by dataclasses, unknown parameters are either
        passed through (if force_unknown=True) or raise an error (if force_unknown=False).

        FDF parameters can be provided in two ways:
        1. Nested in "fdf_arguments" key (explicit)
        2. Directly in user_params (validated against FDF registry)

        Args:
            user_params: User-provided parameters dictionary that may contain
                        'fdf_arguments' key or direct FDF parameters.

        Note:
            This method updates self.fdf_arguments in-place. It's called during
            input parameter generation to ensure all FDF settings are properly configured.
        """
        logger.info("SiestaInputGenerator.setup_fdf_arguments()")

        # Method 1: Update with nested fdf_arguments if present
        if "fdf_arguments" in user_params:
            self.fdf_arguments.update(user_params.get("fdf_arguments"))
            # Track if user explicitly set Spin in fdf_arguments
            if "Spin" in user_params.get("fdf_arguments", {}):
                self.fdf_arguments["_user_set_spin"] = True

        # Method 2: Use merge_fdf_parameters to separate known from unknown FDF parameters
        # Non-FDF parameters (handled by other mechanisms, not FDF registry)
        non_fdf_params = {
            # Internal parameters (not SIESTA FDF parameters)
            "kpts",  # Internal shorthand for %block kgrid.monkhorst.pack
            "basis_set_size",  # Internal shorthand for PAO.BasisSize
            "mesh_cutoff",  # Internal shorthand for Mesh.Cutoff
            "energy_shift",  # Internal shorthand for PAO.EnergyShift
            "pseudo_path",
            "pseudo_base_path",
            "pseudo_family",
            "pseudo_version",
            "pseudo_quality",
            "pseudo_relativistic",
            "pseudo_fdf_arguments",
            "tier",
            "enabled_modules",
            "disabled_modules",
            "fdf_arguments",
            "force_unknown",  # Internal control flag
        }

        # Step 1: Separate internal parameters from FDF parameters
        # Note: normalize_internal_params() already called in __post_init__()
        fdf_candidate_params, internal_params = filter_internal_params(user_params)

        # Store internal_params for use in _initialize_modules()
        self._internal_params = internal_params

        # Step 3: Filter out non-FDF parameters (Python-specific controls)
        final_fdf_params = {}
        for k, v in fdf_candidate_params.items():
            k_lower = k.lower()
            k_normalized = k_lower.replace("_", "").replace(".", "").replace("-", "")

            # Check if it's a non-FDF parameter (try both original and normalized)
            if k_lower in non_fdf_params or k_normalized in {
                p.replace("_", "").replace(".", "").replace("-", "")
                for p in non_fdf_params
            }:
                continue

            # It's a potential FDF parameter
            final_fdf_params[k] = v

        # Use final_fdf_params for validation
        fdf_candidate_params = final_fdf_params

        # Use merge_fdf_parameters to validate
        try:
            known_params, unknown_params = merge_fdf_parameters(
                fdf_candidate_params, force_unknown=self.force_unknown
            )

            # Add known FDF parameters (handled by dataclasses - will be processed later)
            # These don't go to fdf_arguments directly, they're passed to dataclasses
            logger.debug(
                f"Known FDF parameters (handled by dataclasses): {list(known_params.keys())}"
            )

            # Group known FDF parameters by which dataclass handles them
            from atomate2.siesta.dataclass.base import FDFDataclass

            params_by_dataclass = {}
            for key, value in known_params.items():
                handler = FDFDataclass.get_handler(key)
                if handler not in params_by_dataclass:
                    params_by_dataclass[handler] = {}
                params_by_dataclass[handler][key] = value

            logger.debug(
                f"Grouped {len(known_params)} known FDF parameters "
                f"into {len(params_by_dataclass)} dataclasses"
            )

            # Store for later routing during module initialization
            self._params_by_dataclass = params_by_dataclass

            # Add unknown FDF parameters to fdf_arguments (if force_unknown=True)
            for key, value in unknown_params.items():
                self.fdf_arguments[key] = value
                # Track if user explicitly set Spin parameter (case-insensitive)
                if key.lower() == "spin":
                    self.fdf_arguments["_user_set_spin"] = True

        except ValueError:
            # Unknown parameters found and force_unknown=False
            # Colored error already printed by merge_fdf_parameters()
            raise

        # Always add LongOutput (needed for proper output)
        # Note: WriteForces is now handled by MolecularDynamicsAndRelaxation dataclass
        self.fdf_arguments.update({"LongOutput": "True"})

    def _detect_modules_from_params(self, user_params: dict[str, Any]) -> set[str]:
        """
        Auto-detect which modules are needed based on user-provided parameters.

        This enables smart activation: users can use any parameter without
        worrying about tier settings. The system automatically activates
        the required dataclass modules based on the parameters provided.

        Args:
            user_params: Dictionary of user-provided SIESTA parameters

        Returns:
            set: Set of module names that should be activated based on parameters
        """
        needed_modules = set()

        # Check for special value-based detection first
        # MD.TypeOfRun = "FC" triggers phonons module
        for key, value in user_params.items():
            key_normalized = (
                key.lower()
                .replace("_", "")
                .replace(".", "")
                .replace("-", "")
                .replace(" ", "")
            )
            if key_normalized == "mdtypeofrun" and str(value).upper() == "FC":
                needed_modules.add("phonons")

        # Convert all param keys to lowercase with no separators for matching
        # Also strip %block prefix for better matching
        param_keys_lower = set()
        for k in user_params.keys():
            k_normalized = (
                k.lower()
                .replace("_", "")
                .replace(".", "")
                .replace("-", "")
                .replace(" ", "")
            )
            # Strip %block prefix if present
            if k_normalized.startswith("%block"):
                k_normalized = k_normalized[6:]  # Remove "%block"
            param_keys_lower.add(k_normalized)

        # COMPREHENSIVE MODULE PARAMETER MAPPINGS
        # ========================================
        # Maps parameter name patterns to their corresponding dataclass modules
        # This enables automatic module activation for ANY SIESTA parameter

        # Use prefix-based detection for robustness
        for param in param_keys_lower:
            # SCF Loop Parameters
            if param.startswith(("scf", "dm")) or "mixer" in param or "pulay" in param:
                needed_modules.add("scf_loop")

            # Electronic Structure (Occupation, Fermi, etc.)
            elif param.startswith(("occupation", "electronic", "fermi")):
                needed_modules.add("electronic_structure")

            # MD/Relaxation parameters
            # Note: MD.FC* parameters (FCDispl, FCfirst, FClast) go to phonons module
            elif param.startswith("md"):
                # Check if it's a phonon FC parameter (MD.FCDispl, MD.FCfirst, MD.FClast)
                if "mdfc" in param:
                    needed_modules.add("phonons")
                else:
                    needed_modules.add("md_relaxation")

            # Spin parameters
            elif param.startswith("spin") or param in ("noncollinearspin",):
                needed_modules.add("spin")

            # DFT+U parameters
            elif param.startswith("dftu") or "hubbard" in param:
                needed_modules.add("dftu")

            # Optical properties
            elif param.startswith("optical") or param.startswith("polarization"):
                needed_modules.add("optical")

            # DOS/Band structure
            elif param.startswith(("projecteddens", "coop", "pdos", "writeeigen")):
                needed_modules.add("dos_bands")

            # NetCDF parameters
            elif param.startswith("cdf") or param in ("cdfsave", "cdfcompress"):
                needed_modules.add("netcdf")

            # Parallel parameters
            elif param in ("blocksize", "processory") or "decomposition" in param:
                needed_modules.add("parallel")

            # Efficiency/Performance parameters
            elif param.startswith(("alloc", "timer", "maxwall")) or param in (
                "directphi",
                "usesavedata",
            ):
                needed_modules.add("efficiency")

            # Hamiltonian/Overlap parameters
            elif (
                param
                in ("neglnonoverlapint", "savehs", "forceauxcell", "hcutoff", "scutoff")
                or "scfwriteextra" in param
                or "auxcell" in param
            ):
                needed_modules.add("hamiltonian_overlap")

            # Phonon parameters
            elif param.startswith("fc") or param in ("bandlinesscale",):
                needed_modules.add("phonons")

            # Constraints
            elif (
                param.startswith("constraint")
                or "geometryconstraints" in param
                or "geometry.constraints" in param
            ):
                needed_modules.add("constraints")

            # Charge/Dipole/Electric field
            elif (
                param.startswith(
                    (
                        "netcharge",
                        "simulatedoping",
                        "externalelectric",
                        "slabdipole",
                        "bulkbias",
                        "geometrycharge",
                        "geometryhartree",
                    )
                )
                or "dipolecorrection" in param
                or "electricfield" in param
            ):
                needed_modules.add("charge_dipole")

            # Grids (advanced) - Save density/potential grids
            elif param.startswith(
                (
                    "saverho",
                    "savedeltarho",
                    "saveelectrostatic",
                    "savetotalpotential",
                    "saveioniccharge",
                    "savebader",
                    "savegridfunc",  # SaveGridFunc.Format
                    "gridsprecision",
                    "gridscheck",
                    "gridsmax",
                )
            ) or (
                param.startswith(("grid", "fftmesh"))
                and param not in ("gridcellsampling",)
            ):
                needed_modules.add("grids_advanced")

            # Denchar
            elif param.startswith(("denchar", "writedenchar")):
                needed_modules.add("denchar")

            # Chemical Analysis (Mulliken, COOP, bond analysis)
            elif (
                param.startswith(("writemulliken", "writecoor", "coop"))
                or "mullikenpop" in param
                or "bondlength" in param
            ):
                needed_modules.add("chemical_analysis")

            # Wannier90
            elif param.startswith("wannier"):
                needed_modules.add("wannier90")

            # Molecular Mechanics / Auxiliary Force Field
            elif param.startswith("mm") or "grimme" in param:
                needed_modules.add("auxiliary_force_field")

            # RTTDDFT (Real-time time-dependent DFT)
            elif param.startswith("td") or "rttddft" in param:
                needed_modules.add("rttddft")

            # Solvers
            elif param.startswith("diag") or param in ("solutionmethod",):
                needed_modules.add("solvers")

        return needed_modules

    def _get_modules_to_initialize(
        self, user_params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Determine which dataclass modules to initialize based on tier, user_params, and overrides.

        Args:
            user_params: User-provided parameters to check for module-specific parameters

        Returns:
            dict: Dictionary of module names to DataclassModule metadata objects
                 that should be initialized for this input generator.

        Notes:
            - Gets all modules for the specified tier (hierarchical: basic < intermediate < advanced < expert)
            - Auto-detects needed modules from user_params (smart activation)
            - Applies enabled_modules override to force-enable specific modules
            - Applies disabled_modules override to force-disable specific modules
        """
        from atomate2.siesta.dataclass.registry import MODULE_REGISTRY

        # Get modules for tier (hierarchical)
        modules = get_modules_for_tier(self.tier)

        # Auto-detect modules needed from user_params (SMART ACTIVATION)
        # This allows users to use any parameter without worrying about tiers
        if user_params:
            param_to_module = self._detect_modules_from_params(user_params)
            for module_name in param_to_module:
                if module_name in MODULE_REGISTRY and module_name not in modules:
                    modules[module_name] = MODULE_REGISTRY[module_name]

        # Apply enabled_modules override
        if self.enabled_modules:
            for module_name in self.enabled_modules:
                if module_name in MODULE_REGISTRY:
                    modules[module_name] = MODULE_REGISTRY[module_name]

        # Apply disabled_modules override
        if self.disabled_modules:
            for module_name in self.disabled_modules:
                modules.pop(module_name, None)

        return modules

    def _initialize_modules(
        self,
        modules: dict[str, Any],
        structure: Structure | Molecule,
        user_params_: dict[str, Any],
        user_params_lower: dict[str, Any],
    ) -> None:
        """
        Initialize all dataclass modules in priority order.

        Args:
            modules: Dictionary of module names to DataclassModule metadata
            structure: The structure/molecule being calculated
            user_params_: User parameters (original case)
            user_params_lower: User parameters (lowercased for matching)

        Notes:
            - Modules are initialized in priority order (lower priority = initialized first)
            - Each module's setup_*() method is called to process user_params
            - FDF arguments from each module are collected and merged
            - Special handling for modules that need structure (e.g., kpoints)
            - Special handling for modules with non-standard attributes (e.g., basis_sets)
        """
        import importlib

        # Sort modules by priority (lower = earlier)
        sorted_modules = get_sorted_modules(modules)

        for module_meta in sorted_modules:
            # Skip modules handled specially elsewhere
            if module_meta.name in [
                "pseudopotentials",
                "general_system",
                "lua_scripting",
            ]:
                # pseudopotentials: Handled separately via setup_pseudos() in _get_input_parameters
                # general_system: Handled separately (system descriptors)
                # lua_scripting: Handled separately at lines 841-844 (creates self.lua_settings attribute)
                continue

            try:
                # Dynamically import the module
                module = importlib.import_module(module_meta.module_path)
                klass = getattr(module, module_meta.class_name)
                setup_method = getattr(klass, module_meta.setup_method)

                # Call setup method - special cases for modules needing structure
                if module_meta.name == "kpoints":
                    settings = setup_method(
                        structure=structure, user_params=user_params_lower
                    )
                elif (
                    module_meta.name == "spin"
                ):  # Module name is "spin", not "spin_settings"
                    # SpinSettings needs structure for auto-generating DM.InitSpin from magmom
                    # Get magnetic_ordering from internal_params (prefixed or legacy)
                    magnetic_ordering = self._internal_params.get(
                        "magnetic_ordering", "antiferromagnetic"
                    )

                    settings = setup_method(
                        structure=structure,
                        user_params=user_params_,
                        magnetic_ordering=magnetic_ordering,
                    )
                elif module_meta.name == "lua_scripting":
                    # Lua scripting requires enable_lua flag
                    if self.enable_lua:
                        settings = setup_method(user_params_)
                    else:
                        continue  # Skip if Lua disabled
                else:
                    settings = setup_method(user_params_)

                # Route FDF parameters to this dataclass if applicable
                dataclass_name = module_meta.class_name
                if (
                    hasattr(self, "_params_by_dataclass")
                    and dataclass_name in self._params_by_dataclass
                ):
                    fdf_params_for_this_class = self._params_by_dataclass[
                        dataclass_name
                    ]
                    logger.info(
                        f"Routing {len(fdf_params_for_this_class)} FDF parameters "
                        f"to {dataclass_name}: {list(fdf_params_for_this_class.keys())}"
                    )
                    try:
                        # Update dataclass attributes from user FDF parameters
                        settings.update_from_fdf(fdf_params_for_this_class)
                        logger.info(
                            f"Successfully updated {dataclass_name} from FDF parameters"
                        )

                        # CRITICAL: Regenerate FDF after updating from user parameters
                        # The setup_method() already generated FDF with default/auto values.
                        # Now that we've updated attributes from user FDF, regenerate to reflect user values.
                        if hasattr(settings, "generate_fdf"):
                            regenerated_fdf = settings.generate_fdf()
                            if regenerated_fdf:
                                # Store regenerated FDF in the module's FDF attribute
                                setattr(
                                    settings, module_meta.fdf_attribute, regenerated_fdf
                                )
                                logger.info(
                                    f"Regenerated FDF for {dataclass_name} with user values"
                                )

                    except Exception as e:
                        logger.warning(
                            f"Failed to update {dataclass_name} from FDF: {e}. "
                            f"Using default values."
                        )

                # Extract and update FDF arguments
                if hasattr(settings, module_meta.fdf_attribute):
                    fdf_args = getattr(settings, module_meta.fdf_attribute)
                    if fdf_args:  # Only update if non-empty
                        self.fdf_arguments.update(fdf_args)

                # GENERAL: Store the updated module instance for reuse by other code
                # This ensures ONE instance per module, updated with user params
                # Store with underscore prefix to avoid conflicts with existing attributes
                instance_attr_name = f"_{module_meta.instance_attribute}_module"
                setattr(self, instance_attr_name, settings)
                logger.debug(
                    f"Stored updated {module_meta.name} instance as self.{instance_attr_name}"
                )

                # Handle special module-specific attributes
                if module_meta.name == "basis_sets":
                    # Basis sets module sets additional attributes
                    if hasattr(settings, "pao_basissize"):
                        self.basis_set_size = settings.pao_basissize
                    if hasattr(settings, "pao_energy_shift"):
                        self.energy_shift = settings.pao_energy_shift

                logger.debug(
                    f"Initialized module: {module_meta.name} (tier={module_meta.tier}, priority={module_meta.priority})"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to initialize module {module_meta.name}: {e}. "
                    "This module's parameters will not be available for this calculation."
                )

        # Store state after dataclass processing for later comparison
        self._after_dataclass_params = dict(self.fdf_arguments)

        # Display parameter changes after dataclass processing
        self._display_parameter_evolution(stage="after_dataclass")
