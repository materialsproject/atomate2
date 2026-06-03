"""
Base class for FDF dataclasses with self-registration.

This module provides the foundation for dataclass-based FDF parameter handling
with automatic registration and case-insensitive parameter matching.

class GeneralSystemDescriptors
class Pseudopotentials
class BasisSetsAndProjectors
class StructuralInformationVersion1
class StructuralInformationVersion2
class KPointSampling
class ExchangeCorrelationFunctionals
class SpinSettings
class SCFLoopParameters
class RealSpaceGridParameters
class HamiltonianAndOverlapParameters
class ElectronicStructureCalculationOptions
class SolversAndPerformanceOptions
class DensityOfStatesAndBandStructure
class ChemicalAnalysis
class OpticalProperties
class Wannier90
class ChargeDipoleElectricField
class Grids
class AuxiliaryForceField
class ParallelOptions
class EfficiencyOptions
class Denchar
class NetcdfOptions
class ExternalControlAndScripting
class GeneralConstraints
class PhononCalculations
class DFTU
class RTTDDFT
"""

from __future__ import annotations

# Metadata

__all__ = ["FDFDataclass"]

import logging
from typing import Any

logger = logging.getLogger(__name__)


class FDFDataclass:
    """
    Base class for all FDF dataclasses.

    Provides:
    - Self-registration of FDF parameters
    - Case-insensitive FDF parameter matching
    - Common interface for FDF handling

    Each dataclass that inherits from this should:
    1. Call register_fdf_params() in __post_init__
    2. Implement update_from_fdf()
    3. Implement validate()
    4. Implement generate_fdf()
    5. Optionally implement to_ase()

    Example:
        >>> @dataclass
        >>> class MeshSettings(FDFDataclass):
        ...     mesh_cutoff: float = 200.0
        ...
        ...     def __post_init__(self):
        ...         if not hasattr(self.__class__, '_registered'):
        ...             self.register_fdf_params("Mesh.Cutoff")
        ...             self.__class__._registered = True
        ...
        ...     def update_from_fdf(self, fdf_dict):
        ...         # Parse FDF parameters
        ...         pass
    """

    # Global registry: fdf_name_lower → dataclass_name
    _FDF_REGISTRY: dict[str, str] = {}

    @classmethod
    def register_fdf_params(cls, *fdf_names: str) -> None:
        """
        Register FDF parameters handled by this dataclass.

        Args:
            *fdf_names: SIESTA FDF parameter names (any case)

        Example:
            >>> MeshSettings.register_fdf_params("Mesh.Cutoff")
            >>> BasisSettings.register_fdf_params(
            ...     "PAO.BasisSize",
            ...     "PAO.EnergyShift",
            ...     "PAO.SplitNorm"
            ... )
        """
        for name in fdf_names:
            name_lower = name.lower()
            if name_lower in cls._FDF_REGISTRY:
                existing = cls._FDF_REGISTRY[name_lower]
                if existing != cls.__name__:
                    logger.warning(
                        f"FDF parameter '{name}' already registered by {existing}. "
                        f"Overriding with {cls.__name__}."
                    )
            cls._FDF_REGISTRY[name_lower] = cls.__name__
            logger.debug(f"Registered FDF parameter '{name}' → {cls.__name__}")

    @classmethod
    def handles_fdf_param(cls, fdf_name: str) -> bool:
        """
        Check if any dataclass handles this FDF parameter.

        Args:
            fdf_name: SIESTA FDF parameter name (any case)

        Returns:
            True if parameter is registered

        Example:
            >>> FDFDataclass.handles_fdf_param("Mesh.Cutoff")
            True
            >>> FDFDataclass.handles_fdf_param("mesh.cutoff")  # Case-insensitive
            True
            >>> FDFDataclass.handles_fdf_param("MD.TypeOfRun")
            False  # Not registered
        """
        return fdf_name.lower() in cls._FDF_REGISTRY

    @classmethod
    def get_handler(cls, fdf_name: str) -> str | None:
        """
        Get dataclass name that handles this FDF parameter.

        Args:
            fdf_name: SIESTA FDF parameter name (any case)

        Returns:
            Dataclass name or None if not registered

        Example:
            >>> FDFDataclass.get_handler("Mesh.Cutoff")
            "MeshSettings"
            >>> FDFDataclass.get_handler("PAO.BasisSize")
            "BasisSetOptimization"
            >>> FDFDataclass.get_handler("MD.TypeOfRun")
            None
        """
        return cls._FDF_REGISTRY.get(fdf_name.lower())

    @classmethod
    def get_all_registered_params(cls) -> list[str]:
        """
        Get all registered FDF parameter names.

        Returns:
            List of registered FDF parameter names (lowercase)

        Example:
            >>> FDFDataclass.get_all_registered_params()
            ['mesh.cutoff', 'pao.basissize', 'pao.energyshift', ...]
        """
        return sorted(cls._FDF_REGISTRY.keys())

    @classmethod
    def get_field_unit(cls, field_name: str) -> str | None:
        """
        Get the unit for a dataclass field from its metadata.

        Args:
            field_name: Name of the dataclass field

        Returns:
            Unit string (e.g., "Ry", "Ang", "fs") or None if not specified

        Example:
            >>> MeshSettings.get_field_unit("mesh_cutoff")
            "Ry"
            >>> BasisSettings.get_field_unit("pao_energy_shift")
            "Ry"
        """

        if not hasattr(cls, "__dataclass_fields__"):
            return None

        field_info = cls.__dataclass_fields__.get(field_name)
        if field_info is None:
            return None

        return field_info.metadata.get("unit")

    @classmethod
    def get_fdf_parameter_unit(cls, fdf_name: str) -> str | None:
        """
        Get the unit for an FDF parameter by searching all dataclass fields.

        Args:
            fdf_name: SIESTA FDF parameter name (e.g., "Mesh.Cutoff", "PAO.EnergyShift")

        Returns:
            Unit string (e.g., "Ry", "Ang", "fs") or None if not found

        Example:
            >>> FDFDataclass.get_fdf_parameter_unit("Mesh.Cutoff")
            "Ry"
            >>> FDFDataclass.get_fdf_parameter_unit("PAO.EnergyShift")
            "Ry"
            >>> FDFDataclass.get_fdf_parameter_unit("PAO.BasisSize")
            None  # No unit (string parameter)
        """

        if not hasattr(cls, "__dataclass_fields__"):
            return None

        fdf_name_lower = fdf_name.lower()

        # Search all fields for matching SIESTA keyword
        for field_name, field_info in cls.__dataclass_fields__.items():
            siesta_keyword = field_info.metadata.get("SIESTA keyword")
            if siesta_keyword and siesta_keyword.lower() == fdf_name_lower:
                return field_info.metadata.get("unit")

        return None

    @classmethod
    def get_all_fields_with_units(cls) -> dict[str, dict[str, str]]:
        """
        Get all dataclass fields that have units, with their metadata.

        Returns:
            Dictionary mapping field names to their metadata (description, keyword, unit)

        Example:
            >>> MeshSettings.get_all_fields_with_units()
            {
                "mesh_cutoff": {
                    "description": "Sets the energy cutoff...",
                    "SIESTA keyword": "Mesh.Cutoff",
                    "unit": "Ry"
                }
            }
        """

        if not hasattr(cls, "__dataclass_fields__"):
            return {}

        fields_with_units = {}
        for field_name, field_info in cls.__dataclass_fields__.items():
            unit = field_info.metadata.get("unit")
            if unit is not None:
                fields_with_units[field_name] = {
                    "description": field_info.metadata.get("description", ""),
                    "SIESTA keyword": field_info.metadata.get("SIESTA keyword", ""),
                    "unit": unit,
                }

        return fields_with_units

    def update_from_fdf(self, fdf_dict: dict[str, Any]) -> None:
        """
        Update this dataclass from FDF parameters.

        This method must be implemented by subclasses to parse
        FDF format parameters and update internal attributes.

        Args:
            fdf_dict: Dictionary of FDF parameters

        Raises:
            NotImplementedError: If not implemented by subclass

        Example:
            >>> def update_from_fdf(self, fdf_dict):
            ...     for key, value in fdf_dict.items():
            ...         if key.lower() == "mesh.cutoff":
            ...             self.mesh_cutoff = parse_energy(value)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement update_from_fdf()"
        )

    def validate(self) -> None:
        """
        Validate parameter values.

        This method must be implemented by subclasses to check
        if parameter values are valid.

        Raises:
            NotImplementedError: If not implemented by subclass
            ValueError: If validation fails

        Example:
            >>> def validate(self):
            ...     if not (50 <= self.mesh_cutoff <= 10000):
            ...         raise ValueError("Invalid Mesh.Cutoff")
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement validate()"
        )

    def generate_fdf(self) -> dict[str, Any]:
        """
        Generate FDF output for this dataclass.

        This method must be implemented by subclasses to convert
        internal attributes to SIESTA FDF format.

        Returns:
            Dictionary of FDF parameters

        Raises:
            NotImplementedError: If not implemented by subclass

        Example:
            >>> def generate_fdf(self):
            ...     return {"Mesh.Cutoff": f"{self.mesh_cutoff} Ry"}
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement generate_fdf()"
        )

    def to_ase(self) -> dict[str, Any]:
        """
        Generate ASE-format parameters (optional).

        Override this method if the dataclass generates ASE parameters.
        Default implementation returns empty dict.

        Returns:
            Dictionary of ASE parameters

        Example:
            >>> def to_ase(self):
            ...     return {"mesh": self.mesh_cutoff}  # ASE uses 'mesh'
        """
        return {}


def merge_fdf_parameters(
    user_params: dict[str, Any] | None,
    force_unknown: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Separate user_params into known FDF parameters and unknown parameters.

    This function uses the FDF registry to identify which parameters are
    handled by dataclasses and which are unknown. Unknown parameters can
    either raise an error or be allowed based on force_unknown flag.

    Args:
        user_params: User-provided FDF parameters (case-insensitive)
        force_unknown: If True, allow unknown parameters to pass through

    Returns:
        Tuple of (known_params, unknown_params):
            - known_params: Parameters registered by dataclasses
            - unknown_params: Parameters not in registry

    Raises:
        ValueError: If unknown parameters found and force_unknown=False

    Example:
        >>> user_params = {
        ...     "Mesh.Cutoff": "300 Ry",      # Known (registered)
        ...     "PAO.BasisSize": "DZP",        # Known (registered)
        ...     "CustomParam": "value",        # Unknown
        ... }
        >>> known, unknown = merge_fdf_parameters(user_params)
        ValueError: Unknown FDF parameters: CustomParam

        >>> known, unknown = merge_fdf_parameters(user_params, force_unknown=True)
        >>> known
        {"Mesh.Cutoff": "300 Ry", "PAO.BasisSize": "DZP"}
        >>> unknown
        {"CustomParam": "value"}
    """
    if user_params is None:
        return {}, {}

    known_params: dict[str, Any] = {}
    unknown_params: dict[str, Any] = {}

    for key, value in user_params.items():
        if FDFDataclass.handles_fdf_param(key):
            known_params[key] = value
            handler = FDFDataclass.get_handler(key)
            logger.debug(f"Known FDF parameter '{key}' → handled by {handler}")
        else:
            unknown_params[key] = value
            logger.debug(f"Unknown FDF parameter '{key}'")

    # Handle unknown parameters
    if unknown_params and not force_unknown:
        from rich.console import Console

        console = Console()

        unknown_keys_plain = ", ".join(sorted(unknown_params.keys()))
        unknown_keys_colored = ", ".join(
            f"[yellow]{key}[/yellow]" for key in sorted(unknown_params.keys())
        )

        # Print colored version for terminal
        console.print(
            f"\n[bold red]Unknown FDF parameter(s):[/bold red] {unknown_keys_colored}\n"
        )

        # Error message for exception (plain text)
        error_msg = (
            f"Unknown FDF parameter(s): {unknown_keys_plain}\n\n"
            f"These parameters are not registered in the FDF registry.\n\n"
            f"To fix this:\n"
            f"  1. Search for correct parameter name:\n"
            f"     atomate2siesta-inputs search <keyword>\n"
            f"  2. Check spelling against SIESTA manual (case-insensitive)\n"
            f"  3. Allow unknown parameters with force_unknown=True:\n"
            f"     • RelaxMaker(user_params={{...}}, force_unknown=True)\n"
            f"     • update_user_siesta_settings(flow, {{...}}, force_unknown=True)\n"
        )

        raise ValueError(error_msg)

    if unknown_params:
        logger.warning(
            f"Allowing {len(unknown_params)} unknown FDF parameters "
            f"(force_unknown=True): {', '.join(sorted(unknown_params.keys()))}"
        )

    return known_params, unknown_params
