"""Core tier system functions.

Functions for applying tier presets to makers and retrieving preset configurations.
"""

from __future__ import annotations

from typing import Any

from .presets import TIER_PRESETS


def get_tier_preset(preset_name: str) -> dict[str, Any]:
    """
    Get a tier preset configuration by name.

    Parameters
    ----------
    preset_name : str
        Name of the tier preset (e.g., "surface_metal", "phonon_high_accuracy")

    Returns
    -------
    dict
        Tier preset configuration containing:
        - description: Human-readable description
        - tier: Base tier level
        - enabled_modules: List of additional modules to enable
        - disabled_modules: List of modules to disable
        - recommended_params: Suggested user_params for this preset

    Raises
    ------
    ValueError
        If preset_name is not found in TIER_PRESETS

    Examples
    --------
    >>> preset = get_tier_preset("surface_metal")
    >>> print(preset["description"])
    'Metallic surface calculations (occupation smearing)'
    >>> print(preset["tier"])
    'intermediate'
    """
    if preset_name not in TIER_PRESETS:
        available = list(TIER_PRESETS.keys())
        raise ValueError(
            f"Unknown tier preset '{preset_name}'. Available presets: {available}"
        )

    return TIER_PRESETS[preset_name].copy()


def apply_tier_preset(
    maker: Any,
    preset_name: str,
    override_params: dict[str, Any] | None = None,
) -> Any:
    """
    Apply a tier preset to a Maker by updating its input_set_generator.

    This function modifies the maker's input_set_generator to use the
    tier configuration and recommended parameters from the preset.

    Parameters
    ----------
    maker : BaseSiestaMaker
        The maker to apply the preset to (e.g., RelaxMaker, StaticMaker)
    preset_name : str
        Name of the tier preset to apply
    override_params : dict, optional
        Additional user_params to override preset recommendations

    Returns
    -------
    BaseSiestaMaker
        Modified maker with updated tier configuration

    Examples
    --------
    Apply surface_metal preset to relaxation:
        >>> from atomate2.siesta.jobs.core import RelaxMaker
        >>> maker = RelaxMaker.fixed_cell_relaxation()
        >>> maker = apply_tier_preset(maker, "surface_metal")

    Apply preset with parameter overrides:
        >>> maker = RelaxMaker.fixed_cell_relaxation()
        >>> maker = apply_tier_preset(
        ...     maker,
        ...     "phonon_high_accuracy",
        ...     override_params={"a2s_kpts": [10, 10, 10]}
        ... )
    """
    preset = get_tier_preset(preset_name)

    # Get current input_set_generator
    input_gen = maker.input_set_generator

    # Update tier configuration
    input_gen.tier = preset["tier"]

    if preset["enabled_modules"]:
        # Merge with existing enabled_modules if present
        existing_enabled = input_gen.enabled_modules or []
        input_gen.enabled_modules = list(
            set(existing_enabled + preset["enabled_modules"])
        )

    if preset["disabled_modules"]:
        # Merge with existing disabled_modules if present
        existing_disabled = input_gen.disabled_modules or []
        input_gen.disabled_modules = list(
            set(existing_disabled + preset["disabled_modules"])
        )

    # Update user_params with recommended settings
    if preset["recommended_params"]:
        current_params = input_gen.user_params or {}
        # Merge: current defaults < preset params < override_params
        # Preset parameters take precedence over defaults, but override_params win
        merged_params = current_params.copy()
        merged_params.update(preset["recommended_params"])
        if override_params:
            merged_params.update(override_params)
        input_gen.user_params = merged_params

    return maker


def list_tier_presets() -> dict[str, str]:
    """
    List all available tier presets with their descriptions.

    Returns
    -------
    dict
        Dictionary mapping preset names to descriptions

    Examples
    --------
    >>> presets = list_tier_presets()
    >>> for name, desc in presets.items():
    ...     print(f"{name}: {desc}")
    basic_relax: Basic structural relaxation (minimal parameters)
    surface_metal: Metallic surface calculations (occupation smearing)
    ...
    """
    return {name: config["description"] for name, config in TIER_PRESETS.items()}


def print_tier_presets() -> None:
    """
    Print a formatted table of all available tier presets.

    This is a convenience function for interactive use.

    Examples
    --------
    >>> from atomate2.siesta.sets.tiers import print_tier_presets
    >>> print_tier_presets()
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(
        title="[bold cyan]Available SIESTA Tier Presets[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Preset Name", style="cyan", justify="left")
    table.add_column("Description", style="white", justify="left")
    table.add_column("Base Tier", style="green", justify="center")
    table.add_column("Extra Modules", style="yellow", justify="left")

    for name, config in TIER_PRESETS.items():
        extra_modules = (
            ", ".join(config["enabled_modules"]) if config["enabled_modules"] else "-"
        )
        table.add_row(
            name,
            config["description"],
            config["tier"],
            extra_modules,
        )

    console.print(table)
