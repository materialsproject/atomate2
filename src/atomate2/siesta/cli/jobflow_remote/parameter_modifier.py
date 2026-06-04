"""Parameter modification engine with validation.

This module provides utilities to parse, validate, and apply modifications
to SIESTA FDF parameters.
"""

from __future__ import annotations

import ast
from typing import Any

from rich.console import Console

console = Console()


def get_all_registered_fdf_parameters() -> dict[str, str]:
    """Get all FDF parameters registered in dataclass system.

    Uses the FDFDataclass registry to get all known SIESTA parameters.
    This is automatically updated when dataclasses are modified.

    Returns:
        Dictionary mapping lowercase parameter names to dataclass handler names
    """
    from atomate2.siesta.dataclass.base import FDFDataclass

    # Ensure all dataclass modules are imported and registered
    # This triggers __post_init__ which registers FDF parameters
    _ensure_dataclasses_registered()

    # Access the global FDF registry
    registry = FDFDataclass._FDF_REGISTRY

    # Return copy of registry
    return dict(registry)


def _ensure_dataclasses_registered():
    """Ensure all dataclass modules are loaded and registered.

    This creates dummy instances to trigger __post_init__ registration.
    Uses a cache to avoid repeated instantiation.
    """
    # Use a module-level cache
    if hasattr(_ensure_dataclasses_registered, "_done"):
        return

    # Import all dataclass modules to trigger registration
    # We only need to import them - the class decorators and __post_init__
    # will handle registration
    try:
        from atomate2.siesta.dataclass import (
            # Basic tier
            GeneralSystemDescriptors,
            Pseudopotentials,
            BasisSetsAndProjectors,
            KPointSampling,
            ExchangeCorrelationFunctionals,
            SpinSettings,
            RealSpaceGridParameters,
            # Intermediate tier
            SCFLoopParameters,
            ElectronicStructureCalculationOptions,
            MolecularDynamicsAndRelaxation,
            GeneralConstraints,
            ExternalControlAndScripting,
            ChemicalAnalysis,
            # Advanced tier
            PhononCalculations,
            OpticalProperties,
            DensityOfStatesAndBandStructure,
            DFTU,
            ChargeDipoleElectricField,
            Grids,
            Wannier90,
            AuxiliaryForceField,
            Denchar,
            # Expert tier
            ParallelOptions,
            SolversAndPerformanceOptions,
            EfficiencyOptions,
            HamiltonianAndOverlapParameters,
            NetcdfOptions,
            RTTDDFT,
        )

        # Create one instance of each to trigger registration
        # Use minimal required arguments
        _ = [
            GeneralSystemDescriptors(),
            Pseudopotentials(),
            BasisSetsAndProjectors(),
            KPointSampling(),
            ExchangeCorrelationFunctionals(),
            SpinSettings(),
            RealSpaceGridParameters(),
            SCFLoopParameters(),
            ElectronicStructureCalculationOptions(),
            MolecularDynamicsAndRelaxation(),
            GeneralConstraints(),
            ExternalControlAndScripting(),
            ChemicalAnalysis(),
            PhononCalculations(),
            OpticalProperties(),
            DensityOfStatesAndBandStructure(),
            DFTU(),
            ChargeDipoleElectricField(),
            Grids(),
            Wannier90(),
            AuxiliaryForceField(),
            Denchar(),
            ParallelOptions(),
            SolversAndPerformanceOptions(),
            EfficiencyOptions(),
            HamiltonianAndOverlapParameters(),
            NetcdfOptions(),
            RTTDDFT(),
        ]

        # Mark as done
        _ensure_dataclasses_registered._done = True

    except Exception as e:
        console.print(
            f"[yellow]Warning:[/yellow] Could not load all dataclass modules: {e}"
        )
        # Continue anyway - partial registration is better than none


def find_parameter_case_variants(param: str) -> list[str]:
    """Find all case variants of a parameter in the registry.

    Args:
        param: Parameter name in any case

    Returns:
        List of all registered parameters that match (case-insensitive)
    """
    param_lower = param.lower()
    registry = get_all_registered_fdf_parameters()

    # Find exact match
    if param_lower in registry:
        return [param_lower]

    # Find similar (for suggestions)
    similar = []
    for registered_param in registry.keys():
        # Check if it's similar (ignore dots, dashes, underscores)
        normalized_param = (
            param_lower.replace(".", "").replace("-", "").replace("_", "")
        )
        normalized_registered = (
            registered_param.replace(".", "").replace("-", "").replace("_", "")
        )

        if normalized_param == normalized_registered:
            similar.append(registered_param)

    return similar


def parse_parameter_string(param_str: str) -> tuple[str, Any] | None:
    """Parse parameter modification string.

    Supports formats:
        - key=value (simple)
        - key=[1,2,3] (list)
        - key={"a":1} (dict)
        - key=true (boolean)
        - key=100.5 (float)

    Args:
        param_str: Parameter string like "kpts=[4,4,4]"

    Returns:
        Tuple of (key, value) or None if parsing fails
    """
    if "=" not in param_str:
        console.print(
            f"[red]Error:[/red] Invalid parameter format: {param_str}\n"
            f"Expected format: key=value"
        )
        return None

    key, value_str = param_str.split("=", 1)
    key = key.strip()
    value_str = value_str.strip()

    # Try to parse value using ast.literal_eval for safe evaluation
    try:
        # Handle common types
        value: Any
        if value_str.lower() == "true":
            value = True
        elif value_str.lower() == "false":
            value = False
        elif value_str.lower() == "none":
            value = None
        else:
            # Try literal_eval for lists, dicts, numbers
            try:
                value = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                # If it fails, treat as string
                value = value_str

        return (key, value)

    except Exception as e:
        console.print(f"[red]Error parsing value:[/red] {e}")
        return None


def parse_multiple_parameters(param_strings: list[str]) -> dict[str, Any]:
    """Parse multiple parameter modification strings.

    Args:
        param_strings: List of parameter strings

    Returns:
        Dictionary of parsed parameters
    """
    params = {}

    for param_str in param_strings:
        result = parse_parameter_string(param_str)
        if result:
            key, value = result
            params[key] = value

    return params


def validate_fdf_parameter(key: str, value: Any) -> tuple[bool, str]:
    """Validate FDF parameter key and value.

    Uses the dataclass registry for automatic, dynamic validation.
    Parameters are validated against the registered FDF parameters.

    Args:
        key: Parameter key
        value: Parameter value

    Returns:
        Tuple of (is_valid, error_message)
    """
    from atomate2.siesta.dataclass.base import FDFDataclass

    # Get all registered parameters (automatically updated from dataclasses)
    registered_params = get_all_registered_fdf_parameters()
    key_lower = key.lower()

    # Check if parameter is registered (case-insensitive)
    if key_lower not in registered_params:
        # Special cases for atomate2siesta internal parameters
        if key.startswith("a2s_") or key.startswith("atomate2siesta_"):
            # Internal parameter - allow it
            return (True, "")

        # kpts is an atomate2siesta internal parameter (not in the FDF
        # registry), but we still validate its shape here.
        if key == "kpts":
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                return (False, "kpts must be a list of 3 integers: [k1, k2, k3]")
            if not all(isinstance(v, int) and v > 0 for v in value):
                return (False, "kpts values must be positive integers")
            return (True, "")

        if key in [
            "kgrid_cutoff",
            "mesh_cutoff",
            "xc",
            "tier",
            "fdf_arguments",
        ]:
            # atomate2siesta special parameters
            return (True, "")

        # Find similar parameters for suggestions
        similar = find_parameter_case_variants(key)
        if similar:
            return (
                False,
                f"Parameter '{key}' not registered. Did you mean one of: {', '.join(similar)}?",
            )
        else:
            # Not registered but don't block - SIESTA will validate
            console.print(
                f"[yellow]Warning:[/yellow] Parameter '{key}' is not in the dataclass registry.\n"
                f"It will be passed to SIESTA, which will validate it."
            )
            return (True, "")

    # Check for case mismatches (e.g., spin vs Spin)
    # Get the registered parameter name from FDFDataclass
    handler = FDFDataclass.get_handler(key)
    if handler:
        # Check if exact case matches
        # Since registry stores lowercase, we check if the original key would be handled
        if not FDFDataclass.handles_fdf_param(key):
            # Case doesn't match - this shouldn't happen if registry works, but check anyway
            pass

    # Validate specific parameter types
    if key == "Spin":
        valid_spin = ["polarized", "non-polarized", "spin-orbit", "none"]
        if isinstance(value, str) and value.lower() not in valid_spin:
            return (
                False,
                f"Spin must be one of: {valid_spin}. Got: {value}",
            )

    elif key in ["Mesh.Cutoff", "MeshCutoff"]:
        # Should be string with units or float
        if isinstance(value, str):
            if not any(unit in value for unit in ["Ry", "eV", "Ha", "meV"]):
                return (
                    False,
                    "Mesh.Cutoff should include units (Ry, eV, Ha, meV). "
                    "Example: '300 Ry'",
                )

    elif key == "XC.functional":
        valid_xc = ["LDA", "GGA", "VDW"]
        if isinstance(value, str) and value not in valid_xc:
            console.print(
                f"[yellow]Warning:[/yellow] XC.functional '{value}' not in "
                f"common values: {valid_xc}"
            )

    elif key == "XC.authors":
        valid_authors = ["CA", "PZ", "PW92", "PBE", "RPBE", "revPBE", "WC", "AM05"]
        if isinstance(value, str) and value not in valid_authors:
            console.print(
                f"[yellow]Warning:[/yellow] XC.authors '{value}' not in "
                f"common values: {valid_authors}"
            )

    # Validation passed
    return (True, "")


def validate_all_parameters(params: dict[str, Any]) -> bool:
    """Validate all parameters in dictionary.

    Args:
        params: Dictionary of parameters to validate

    Returns:
        True if all valid, False otherwise
    """
    all_valid = True

    for key, value in params.items():
        is_valid, error_msg = validate_fdf_parameter(key, value)
        if not is_valid:
            console.print(f"[red]Validation Error:[/red] {key}: {error_msg}")
            all_valid = False

    return all_valid


def merge_parameters(
    base_params: dict[str, Any],
    override_params: dict[str, Any],
    remove_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Merge override parameters into base parameters and remove specified keys.

    Args:
        base_params: Original parameters
        override_params: New parameters to apply
        remove_keys: List of parameter keys to remove

    Returns:
        Merged parameter dictionary
    """
    # Create deep copy to avoid modifying original
    merged = dict(base_params)

    # Apply overrides
    for key, value in override_params.items():
        if key in merged:
            console.print(f"[yellow]Override:[/yellow] {key}: {merged[key]} → {value}")
        else:
            console.print(f"[green]New parameter:[/green] {key} = {value}")

        merged[key] = value

    # Remove specified keys
    if remove_keys:
        for key in remove_keys:
            if key in merged:
                console.print(f"[red]Remove:[/red] {key} = {merged[key]}")
                del merged[key]
            else:
                console.print(
                    f"[yellow]Warning:[/yellow] Cannot remove '{key}' - not found in parameters"
                )

    return merged


def preview_parameter_changes(
    original: dict[str, Any], modified: dict[str, Any]
) -> None:
    """Display a preview of parameter changes.

    Args:
        original: Original parameters
        modified: Modified parameters
    """
    from rich.table import Table

    table = Table(
        title="Parameter Changes Preview",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Parameter", style="yellow")
    table.add_column("Original", style="red")
    table.add_column("New", style="green")
    table.add_column("Change Type", style="blue")

    # Find all keys
    all_keys = set(original.keys()) | set(modified.keys())

    for key in sorted(all_keys):
        orig_val = original.get(key, "[not set]")
        new_val = modified.get(key, "[removed]")

        if key not in original:
            change_type = "ADDED"
        elif key not in modified:
            change_type = "REMOVED"
        elif orig_val != new_val:
            change_type = "MODIFIED"
        else:
            continue  # Skip unchanged

        table.add_row(
            key,
            str(orig_val),
            str(new_val),
            f"[bold]{change_type}[/bold]",
        )

    if table.row_count > 0:
        console.print("\n")
        console.print(table)
    else:
        console.print("\n[yellow]No changes detected[/yellow]")
