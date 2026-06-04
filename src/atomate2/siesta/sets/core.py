from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field

from atomate2.siesta.sets.base import SiestaInputGenerator
from typing import TYPE_CHECKING
from typing import Any

from pymatgen.core import Structure


from atomate2.siesta.dataclass.molecular_dynamics_and_relaxation import (
    MolecularDynamicsAndRelaxation,
)
from atomate2.siesta.dataclass.density_of_states_and_band_structure import (
    DensityOfStatesAndBandStructure,
)
from atomate2.siesta.dataclass.phonon_calculations import PhononCalculations
from atomate2.siesta.dataclass.optical_properties import OpticalProperties

if TYPE_CHECKING:
    from pymatgen.core import Molecule

logger = logging.getLogger(__name__)


@dataclass
class RelaxSetGenerator(SiestaInputGenerator):
    """
    RelaxSetGenerator class for relaxation-specific calculations.

    Notes
    -----
    This generator automatically enables the ``md_relaxation`` module regardless
    of the tier level, since relaxation calculations require MD/relaxation parameters.
    """

    relax_cell: bool = False
    relaxation: MolecularDynamicsAndRelaxation = field(
        init=False, repr=False
    )  # Initialized lazily in _initialize_modules(); repr=False so repr()
    # works before the input set is generated
    enable_lua: bool = False  # Disable Lua for RelaxSetGenerator

    def __post_init__(self):
        """
        Initialize the relaxation settings after the class is initialized.
        """
        logger.info("RelaxSetGenerator.__post_init__()")

        # Force-enable md_relaxation module since RelaxSetGenerator requires it
        # This allows tier="basic" or tier="dirty" to work with RelaxMaker
        if self.enabled_modules is None:
            self.enabled_modules = ["md_relaxation"]
        elif "md_relaxation" not in self.enabled_modules:
            self.enabled_modules = list(self.enabled_modules) + ["md_relaxation"]

        # Call parent's __post_init__ to apply tier defaults
        super().__post_init__()

        # Don't create self.relaxation here - it will be created and updated by
        # _initialize_modules() as self._md_relaxation_module

    def get_parameter_updates(
        self, structure: Structure | Molecule, prev_parameters: dict[str, Any]
    ) -> dict:
        """
        Generate and update FDF parameters with relaxation-specific settings.

        This method validates the relaxation configuration, generates the relaxation
        block for the SIESTA FDF file, and combines it with user-defined parameters.

        Parameters
        ----------
        structure : Structure | Molecule
            The structure for which to generate relaxation parameters
        prev_parameters : dict[str, Any]
            Previous calculation parameters (currently unused but required by interface)

        Returns
        -------
        dict
            Updated FDF arguments dictionary containing both user-defined and
            relaxation-specific parameters
        """
        logger.info("RelaxSetGenerator.get_parameter_updates()")

        # Use the relaxation instance from _initialize_modules() which has been
        # updated with user_params via update_from_fdf()
        relaxation_instance = self._md_relaxation_module

        # Enable relaxation for RelaxSetGenerator
        relaxation_instance.perform_relaxation = True
        relaxation_instance.md_variable_cell = self.relax_cell

        relaxation_instance.validate()
        relaxation_instance.generate_relaxation_block()

        self.relaxation_fdf_arguments = relaxation_instance.relaxation_fdf_arguments

        # Combine user-defined FDF arguments with relaxation-specific ones
        self.fdf_arguments.update(self.relaxation_fdf_arguments)
        return self.fdf_arguments


@dataclass
class BandStructureSetGenerator(SiestaInputGenerator):
    """
    Generator for SIESTA band structure calculations.

    This class generates input parameters for band structure calculations,
    automatically generating k-point paths along high-symmetry lines in
    the Brillouin zone based on the input structure.
    """

    def get_parameter_updates(
        self, structure: Structure | Molecule, prev_parameters: dict[str, Any]
    ) -> dict:
        """
        Generate and update FDF parameters with band structure settings.

        This method automatically generates a k-point path through the Brillouin zone
        based on the structure's symmetry and combines it with user-defined parameters.

        Parameters
        ----------
        structure : Structure | Molecule
            The structure for which to generate the band structure k-path
        prev_parameters : dict[str, Any]
            Previous calculation parameters (currently unused but required by interface)

        Returns
        -------
        dict
            Updated FDF arguments dictionary containing both user-defined and
            band structure-specific parameters (k-point path, number of points, etc.)
        """
        logger.info("BandStructureSetGenerator.get_parameter_updates()")
        bands = DensityOfStatesAndBandStructure()
        bands.generate_band_structure_block(structure=structure)

        bands_fdf_arguments = bands.bands_fdf_arguments
        logger.debug(f"bands_fdf_arguments={bands_fdf_arguments}")

        # Combine user-defined FDF arguments with relaxation-specific ones
        self.fdf_arguments.update(bands_fdf_arguments)
        return self.fdf_arguments


@dataclass
class LuaSetGenerator(SiestaInputGenerator):
    """
    Generator for SIESTA calculations using Lua scripting (FLOS library).

    This class enables advanced simulation techniques through Lua scripting in SIESTA,
    such as custom relaxation algorithms and nudged elastic band (NEB) calculations.
    It uses the FLOS (Flexible Lua-based Object System) library for complex
    optimization tasks.

    Parameters
    ----------
    lua_type : str, optional
        Type of Lua calculation: "lua_relaxation" for custom relaxation algorithms
        or "lua_neb" for nudged elastic band calculations
    relax_cell : bool, default=False
        Whether to allow cell relaxation during the calculation
    enable_lua : bool, default=True
        Enable Lua scripting for this calculation

    Notes
    -----
    Requires FLOS library and appropriate Lua scripts (e.g., neb.lua) in the
    working directory.
    """

    lua_type: str = None  # "lua_relaxation" or "lua_neb"
    relax_cell: bool = False
    enable_lua: bool = True  # Enable Lua for LuaSetGenerator

    relaxation: MolecularDynamicsAndRelaxation = field(
        init=False, repr=False
    )  # Initialized lazily in _initialize_modules(); repr=False so repr()
    # works before the input set is generated

    def __post_init__(self):
        """
        Initialize the relaxation settings after the class is initialized.

        Sets MD.TypeOfRun to "LUA" for Lua-based relaxation.
        """
        logger.info("LuaSetGenerator.__post_init__()")

        # IMPORTANT: Call parent __post_init__ to initialize _explicit_user_params and other base attributes
        super().__post_init__()

        self.relaxation = MolecularDynamicsAndRelaxation(
            perform_relaxation=True,
            md_variable_cell=self.relax_cell,
            md_type_of_run="LUA",  # Set MD.TypeOfRun to LUA for Lua scripting
        )

    def get_parameter_updates(
        self, structure: Structure | Molecule, prev_parameters: dict[str, Any]
    ) -> dict:
        """
        Generate and update FDF parameters with Lua scripting settings.

        Depending on the lua_type, this method configures SIESTA to use Lua scripts
        for either custom relaxation algorithms or nudged elastic band calculations.

        Parameters
        ----------
        structure : Structure | Molecule
            The structure for the Lua-based calculation
        prev_parameters : dict[str, Any]
            Previous calculation parameters (currently unused but required by interface)

        Returns
        -------
        dict
            Updated FDF arguments dictionary containing Lua script configurations
            and associated parameters for the selected calculation type

        Raises
        ------
        ValueError
            If lua_type is not one of "lua_relaxation" or "lua_neb"
        """
        logger.info("LuaSetGenerator.get_parameter_updates()")

        if self.lua_type == "lua_relaxation":
            self.relaxation.validate()
            self.relaxation.generate_relaxation_block()
            self.relaxation_fdf_arguments = self.relaxation.relaxation_fdf_arguments
            # Keep sections separate by updating fdf_arguments with both dictionaries
            # This ensures ExternalControlAndScripting section appears separately
            self.fdf_arguments.update(self.relaxation_fdf_arguments)
            self.fdf_arguments.update(self.lua_settings.lua_fdf_arguments)

        if self.lua_type == "lua_neb":
            logger.debug(f"lua_type={self.lua_type}")
            lua_arguments = {
                "Lua.Script": "neb.lua",
                "MD.TypeOfRun": "LUA",
            }
            self.fdf_arguments.update(lua_arguments)

        return self.fdf_arguments


@dataclass
class PhononSetGenerator(SiestaInputGenerator):
    """
    A generator class for creating SIESTA input files for phonon calculations.

    This class extends the `SiestaInputGenerator` to generate the necessary input
    files for performing phonon calculations using SIESTA. It initializes the
    phonon-related settings such as the type of phonon run, displacement, and
    frequency calculation parameters.

    Parameters
    ----------
    md_type_of_run : str
        The type of molecular dynamics run. Defaults to "FC" (Force Constant).
    md_fc_first : int
        The first atom index to calculate force constants for. Defaults to 1.
    md_fc_last : int
        The last atom index to calculate force constants for. Defaults to 1.
    md_fc_displ : float
        The displacement to use when calculating the force constants. Defaults to 0.04.

    Attributes
    ----------
    phonon : PhononCalculations
        An instance of the `PhononCalculations` class that stores phonon-related settings
        and methods for generating the phonon-related blocks for the input file.

    Methods
    -------
    __post_init__():
        Initializes the phonon-related settings after the class is initialized.

    get_parameter_updates(structure, prev_parameters):
        Updates the FDF (SIESTA input) arguments with phonon-specific and band structure
        parameters, ensuring they are properly generated and validated before the SIESTA run.
    """

    md_type_of_run: str = "FC"
    md_fc_first: int = 1
    md_fc_last: int = 1
    md_fc_displ: float = 0.04

    phonon: PhononCalculations = field(init=False)

    def __post_init__(self):
        """
        Initialize the phonon settings after the class is initialized.
        """
        logger.info("PhononSetGenerator.__post_init__()")
        self.phonon = PhononCalculations(
            md_type_of_run=self.md_type_of_run,
            md_fc_first=self.md_fc_first,
            md_fc_last=self.md_fc_last,
            md_fc_displ=self.md_fc_displ,
        )

    def get_parameter_updates(
        self, structure: Structure | Molecule, prev_parameters: dict[str, Any]
    ) -> dict:
        """
        Generate and update FDF parameters for phonon calculations.

        This method validates the phonon configuration, generates both the phonon
        calculation block and band structure k-path, and combines them with
        user-defined parameters for SIESTA phonon calculations.

        Parameters
        ----------
        structure : Structure | Molecule
            The structure for which to calculate phonons
        prev_parameters : dict[str, Any]
            Previous calculation parameters (currently unused but required by interface)

        Returns
        -------
        dict
            Updated FDF arguments dictionary containing phonon calculation parameters
            (force constant settings, displacement) and band structure parameters for
            phonon dispersion curves
        """
        logger.info("PhononSetGenerator.get_parameter_updates()")

        self.phonon.validate()
        self.phonon.generate_phonon_block()

        self.phonon_fdf_arguments = self.phonon.phonon_fdf_arguments
        logger.debug(f"phonon_fdf_arguments={self.phonon_fdf_arguments}")

        # Combine user-defined FDF arguments with relaxation-specific ones
        self.fdf_arguments.update(self.phonon_fdf_arguments)

        # Generate k-path for phonon dispersion using HighSymmKpath
        # (automatically determines correct reciprocal lattice directions from structure symmetry)
        bands = DensityOfStatesAndBandStructure()
        bands.generate_band_structure_block(structure=structure)

        bands_fdf_arguments = bands.bands_fdf_arguments
        logger.debug(f"bands_fdf_arguments={bands_fdf_arguments}")

        # Combine user-defined FDF arguments with relaxation-specific ones
        self.fdf_arguments.update(bands_fdf_arguments)

        return self.fdf_arguments


@dataclass
class OpticalSetGenerator(SiestaInputGenerator):
    """
    This class is responsible for generating input files for optical property calculations
    using the SIESTA package. It extends the `SiestaInputGenerator` to include settings
    for optical calculations.

    Attributes:
        optical_calculation (str | None): Specifies the type of optical calculation to be performed.
        optical (OpticalProperties): Instance of `OpticalProperties` containing validated
                                     optical calculation parameters.
    """

    def __init__(self, *args, optical_calculation=None, **kwargs):
        """
        Initialize the OpticalSetGenerator class with specific settings for optical calculations.

        Args:
            optical_calculation (str | None): Type of optical calculation to be performed.
            *args: Variable length argument list passed to the SiestaInputGenerator.
            **kwargs: Keyword arguments containing both optical-specific and general SIESTA parameters.
        """
        logger.info("OpticalSetGenerator.__init__()")
        # Extract optical properties kwargs
        optical_properties_keys = OpticalProperties.__dataclass_fields__.keys()
        optical_kwargs = {
            k: v for k, v in kwargs.items() if k in optical_properties_keys
        }
        siesta_kwargs = {
            k: v for k, v in kwargs.items() if k not in optical_properties_keys
        }

        # Initialize the parent SiestaInputGenerator with its relevant kwargs
        super().__init__(*args, **siesta_kwargs)

        self.optical_calculation = optical_calculation
        self.optical = OpticalProperties(
            optical_calculation=self.optical_calculation, **optical_kwargs
        )

    def get_parameter_updates(
        self, structure: Structure | Molecule, prev_parameters: dict[str, Any]
    ) -> dict:
        """
        Updates the FDF (input) arguments for the SIESTA calculation by incorporating optical and band
        structure settings.

        Args:
            structure (Structure | Molecule): The structure for which the optical properties are calculated.
            prev_parameters (dict): Previous calculation parameters.

        Returns:
            dict: Updated FDF parameters including optical properties and band structure information.
        """
        logger.info("OpticalSetGenerator.get_parameter_updates()")
        self.optical.validate()
        self.optical.generate_optical_properties_block()

        self.optical_fdf_arguments = self.optical.optical_fdf_arguments

        # Combine user-defined FDF arguments with optical-specific ones
        self.fdf_arguments.update(self.optical_fdf_arguments)

        # Generate k-path for electronic band structure using HighSymmKpath
        # (automatically determines correct reciprocal lattice directions from structure symmetry)
        bands = DensityOfStatesAndBandStructure()
        bands.generate_band_structure_block(structure=structure)

        bands_fdf_arguments = bands.bands_fdf_arguments

        # Combine user-defined FDF arguments with relaxation-specific ones
        self.fdf_arguments.update(bands_fdf_arguments)

        return self.fdf_arguments


@dataclass
class SocketIOSetGenerator(SiestaInputGenerator):
    """
    Generator for SIESTA calculations using socket-based I/O communication.

    This class enables communication between SIESTA and external programs through
    sockets, allowing for on-the-fly structure updates and forces calculations.
    Commonly used for integration with molecular dynamics drivers (e.g., i-PI) or
    machine learning force fields.

    Notes
    -----
    This is currently a placeholder implementation. Socket I/O functionality
    requires additional configuration parameters for host, port, and communication
    protocol settings.

    See Also
    --------
    i-PI : Universal force engine interface (https://ipi-code.org/)
    """

    pass


@dataclass
class DOSSetGenerator(SiestaInputGenerator):
    """
    Generator for SIESTA total density of states (DOS) calculations.

    This class generates input parameters for calculating the total electronic
    density of states. It automatically configures the energy grid and DOS
    calculation settings based on the DensityOfStatesAndBandStructure dataclass.

    Examples
    --------
    >>> from pymatgen.core import Structure
    >>> structure = Structure(...)
    >>> dos_gen = DOSSetGenerator()
    >>> input_set = dos_gen.get_input_set(structure)

    Notes
    -----
    For projected DOS (PDOS), use PDOSSetGenerator instead.
    The DOS is calculated on an energy grid around the Fermi level.
    """

    def __post_init__(self):
        """Initialize the DOS settings after the class is initialized."""
        logger.info("DOSSetGenerator.__post_init__()")
        # Call parent's __post_init__ to apply tier defaults and initialize modules
        super().__post_init__()

    def get_parameter_updates(
        self, structure: Structure | Molecule, prev_parameters: dict[str, Any]
    ) -> dict:
        """
        Generate and update FDF parameters with DOS calculation settings.

        This method automatically configures the energy grid, k-point sampling for DOS,
        and enables total DOS calculation using the ProjectedDensityOfStates block.

        Parameters
        ----------
        structure : Structure | Molecule
            The structure for which to calculate DOS
        prev_parameters : dict[str, Any]
            Previous calculation parameters (currently unused but required by interface)

        Returns
        -------
        dict
            Updated FDF arguments dictionary containing DOS-specific parameters
            including ProjectedDensityOfStates and DOS.kgrid.MonkhorstPack blocks
        """
        logger.info("DOSSetGenerator.get_parameter_updates()")

        # Create DOS dataclass instance
        dos = DensityOfStatesAndBandStructure()

        # Enable total DOS calculation
        dos.calculate_total_dos = True
        dos.calculate_partial_dos = False

        # Set DOS k-grid
        # Use user-provided grid if available, otherwise use default 10x10x10
        if "%block DOS.kgrid.MonkhorstPack" in self.user_params:
            dos.dos_kgrid_monkhorst_pack_block = self.user_params[
                "%block DOS.kgrid.MonkhorstPack"
            ]
        else:
            dos.dos_kgrid_monkhorst_pack_block = [
                "10 0 0 0.0",
                "0 10 0 0.0",
                "0 0 10 0.0",
            ]

        # Check if user provided custom ProjectedDensityOfStates block
        if "%block ProjectedDensityOfStates" in self.user_params:
            dos.projected_density_of_states_block = self.user_params[
                "%block ProjectedDensityOfStates"
            ]

        # Generate DOS block (includes ProjectedDensityOfStates)
        dos.generate_dos_block()

        dos_fdf_arguments = dos.bands_fdf_arguments
        logger.debug(f"dos_fdf_arguments={dos_fdf_arguments}")

        # Combine user-defined FDF arguments with DOS-specific ones
        self.fdf_arguments.update(dos_fdf_arguments)
        return self.fdf_arguments


@dataclass
class PDOSSetGenerator(SiestaInputGenerator):
    """
    Generator for SIESTA projected density of states (PDOS) calculations.

    This class generates input parameters for calculating both total and projected
    electronic density of states. PDOS provides orbital-resolved contributions from
    all atoms to the total DOS.

    Examples
    --------
    >>> from pymatgen.core import Structure
    >>> structure = Structure(...)
    >>> pdos_gen = PDOSSetGenerator()
    >>> input_set = pdos_gen.get_input_set(structure)

    Notes
    -----
    The PDOS calculation automatically includes total DOS.
    SIESTA generates PDOS for all atoms when ProjectedDensityOfStates block is present.
    Output files: siesta.DOS (total), siesta.PDOS (projected), siesta.PDOS.xml
    """

    def __post_init__(self):
        """Initialize the PDOS settings after the class is initialized."""
        logger.info("PDOSSetGenerator.__post_init__()")
        # Call parent's __post_init__ to apply tier defaults and initialize modules
        super().__post_init__()

    def get_parameter_updates(
        self, structure: Structure | Molecule, prev_parameters: dict[str, Any]
    ) -> dict:
        """
        Generate and update FDF parameters with PDOS calculation settings.

        This method automatically configures the energy grid, k-point sampling for PDOS,
        and enables both total and projected DOS calculations using the
        ProjectedDensityOfStates block.

        Parameters
        ----------
        structure : Structure | Molecule
            The structure for which to calculate PDOS
        prev_parameters : dict[str, Any]
            Previous calculation parameters (currently unused but required by interface)

        Returns
        -------
        dict
            Updated FDF arguments dictionary containing PDOS-specific parameters
            including ProjectedDensityOfStates and PDOS.kgrid.MonkhorstPack blocks
        """
        logger.info("PDOSSetGenerator.get_parameter_updates()")

        # Use the module instance if available (set by _initialize_modules)
        # This ensures we use the same instance that received user params
        if hasattr(self, "_dos_bands_module") and self._dos_bands_module is not None:
            dos = self._dos_bands_module
        else:
            # Create DOS dataclass instance (fallback)
            dos = DensityOfStatesAndBandStructure()

        # Enable both total and projected DOS
        dos.calculate_total_dos = True
        dos.calculate_partial_dos = True

        # Set PDOS k-grid
        # Use user-provided grid if available, otherwise use default 10x10x10
        # Accept both PDOS.kgrid and DOS.kgrid (PDOS takes precedence)
        if "%block PDOS.kgrid.MonkhorstPack" in self.user_params:
            dos.pdos_kgrid_monkhorst_pack_block = self.user_params[
                "%block PDOS.kgrid.MonkhorstPack"
            ]
        elif "%block DOS.kgrid.MonkhorstPack" in self.user_params:
            dos.pdos_kgrid_monkhorst_pack_block = self.user_params[
                "%block DOS.kgrid.MonkhorstPack"
            ]
        else:
            dos.pdos_kgrid_monkhorst_pack_block = [
                "10 0 0 0.0",
                "0 10 0 0.0",
                "0 0 10 0.0",
            ]

        # Check if user provided custom ProjectedDensityOfStates block
        # If so, store it directly so generate_dos_block() uses it instead of generating a new one
        user_pdos_block = self.user_params.get("%block ProjectedDensityOfStates")
        if user_pdos_block:
            # Store user's block - generate_dos_block() will use it directly
            dos.projected_density_of_states_block = user_pdos_block
            logger.info(
                "User provided ProjectedDensityOfStates block, will use directly"
            )

        # Generate DOS block (includes ProjectedDensityOfStates)
        # Note: SIESTA automatically generates PDOS for all atoms
        dos.generate_dos_block()

        pdos_fdf_arguments = dos.bands_fdf_arguments
        logger.debug(f"pdos_fdf_arguments={pdos_fdf_arguments}")

        # Combine user-defined FDF arguments with PDOS-specific ones
        self.fdf_arguments.update(pdos_fdf_arguments)
        return self.fdf_arguments


@dataclass
class StaticSetGenerator(SiestaInputGenerator):
    """
    Generator for static (single-point) SIESTA calculations.

    This class generates input parameters for static energy calculations without
    geometric relaxation or molecular dynamics. It uses default settings from the
    base SiestaInputGenerator class, making it suitable for evaluating energies,
    forces, and electronic properties at fixed atomic positions.

    Examples
    --------
    >>> from pymatgen.core import Structure
    >>> structure = Structure(...)
    >>> static_gen = StaticSetGenerator()
    >>> input_set = static_gen.get_input_set(structure)

    Notes
    -----
    For geometry optimization, use RelaxSetGenerator instead.
    For electronic band structure calculations, use BandStructureSetGenerator.
    """

    pass
