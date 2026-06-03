#!/usr/bin/env python
"""
CLI tool to display available material-specific presets and their configurations.

Command: atomate2siesta-presets

This tool provides:
- List of all 27 material-specific presets from the codebase
- Preset details and parameters
- Search by category or tier level
- Dynamic loading from TIER_PRESETS registry
- Tier-level defaults (basic/intermediate/advanced/expert)
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# Import actual tier presets and defaults from the codebase
try:
    from atomate2.siesta.sets.tiers.presets import TIER_PRESETS
    from atomate2.siesta.sets.tiers.defaults import TIER_DEFAULTS
except ImportError:
    # Fallback if import fails
    TIER_PRESETS = {}
    TIER_DEFAULTS = {}
    console.print("[red]Warning: Could not load TIER_PRESETS from atomate2[/red]")


# Automatically build category mapping from preset names
def build_category_map():
    """Automatically categorize presets based on their names."""
    categories = {}

    for preset_name in TIER_PRESETS.keys():
        # Determine category from preset name prefix
        if preset_name.startswith("2d_"):
            category = "2d"
        elif preset_name.startswith("surface_") or preset_name == "adsorbate_screening":
            category = "surface"
        elif preset_name.startswith("molecular_") or preset_name.startswith(
            "molecule_"
        ):
            category = "molecular"
        elif preset_name.startswith("magnetic_"):
            category = "magnetic"
        elif preset_name.startswith("phonon_"):
            category = "phonon"
        elif preset_name.startswith("optical_"):
            category = "optical"
        elif preset_name in ["band_structure", "bulk_metal", "bulk_semiconductor"]:
            category = "electronic"
        elif preset_name in ["large_system", "parallel_hpc", "convergence_test"]:
            category = "performance"
        elif "relax" in preset_name:
            category = "structural"
        else:
            # Default category for unmatched presets
            category = "other"

        if category not in categories:
            categories[category] = []
        categories[category].append(preset_name)

    return categories


CATEGORY_MAP = build_category_map()


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Display material-specific presets and tier configurations.

    Explore 27 presets across 9 categories (2d, surface, magnetic, etc.)
    plus tier-level defaults (basic/intermediate/advanced/expert).
    """
    if ctx.invoked_subcommand is None:
        show_list()


@cli.command()
def list():
    """List all tier presets by category."""
    show_list()


@cli.command()
@click.argument("preset_name")
def show(preset_name):
    """Show detailed information for a specific preset."""
    if preset_name not in TIER_PRESETS:
        console.print(f"[red]Error: Preset '{preset_name}' not found[/red]")
        console.print(
            f"\n[dim]Available presets: {', '.join(sorted(TIER_PRESETS.keys()))}[/dim]"
        )
        return

    show_preset_details(preset_name)


@cli.command()
@click.argument("category_name")
def category(category_name):
    """Show presets for a specific category."""
    if category_name not in CATEGORY_MAP:
        console.print(f"[red]Error: Category '{category_name}' not found[/red]")
        console.print(
            f"\n[dim]Available categories: {', '.join(sorted(CATEGORY_MAP.keys()))}[/dim]"
        )
        return

    show_category(category_name)


@cli.command()
@click.option("--category", "-c", help="Filter by category")
@click.option(
    "--tier", "-t", help="Filter by tier level (basic/intermediate/advanced/expert)"
)
def search(category, tier):
    """Search presets by category or tier."""
    results = []

    for name, preset in TIER_PRESETS.items():
        match = True

        if category:
            # Find which category this preset belongs to
            preset_categories = [
                cat for cat, presets in CATEGORY_MAP.items() if name in presets
            ]
            if category not in preset_categories:
                match = False

        if tier and preset.get("tier", "intermediate") != tier:
            match = False

        if match:
            results.append(name)

    if not results:
        console.print("[yellow]No presets found matching criteria[/yellow]")
        return

    console.print(f"\n[bold cyan]Found {len(results)} preset(s)[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Preset", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Tier", style="yellow")

    for name in sorted(results):
        preset = TIER_PRESETS[name]
        table.add_row(
            name, preset.get("description", ""), preset.get("tier", "intermediate")
        )

    console.print(table)
    console.print()


@cli.command()
def examples():
    """Show usage examples."""
    show_examples()


@cli.command()
def defaults():
    """Show tier-level default parameters."""
    show_tier_defaults()


def show_list():
    """Display all tier presets organized by category."""
    console.print()

    # Header
    header = Panel(
        f"[bold cyan]Tier Presets[/bold cyan]\n"
        f"[dim]{len(TIER_PRESETS)} presets across {len(CATEGORY_MAP)} categories[/dim]",
        border_style="cyan",
        box=box.DOUBLE,
    )
    console.print(header)
    console.print()

    # Show presets by category
    for category_name in sorted(CATEGORY_MAP.keys()):
        presets_in_category = CATEGORY_MAP[category_name]

        # Count how many actually exist
        existing_presets = [p for p in presets_in_category if p in TIER_PRESETS]

        if not existing_presets:
            continue

        console.print(f"[bold yellow]Category: {category_name}[/bold yellow]")

        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column(style="cyan", width=30)
        table.add_column(style="white")
        table.add_column(style="yellow", width=15)

        for preset_name in sorted(existing_presets):
            if preset_name in TIER_PRESETS:
                preset = TIER_PRESETS[preset_name]
                table.add_row(
                    preset_name,
                    preset.get("description", "")[:60],
                    preset.get("tier", "intermediate"),
                )

        console.print(table)
        console.print()

    console.print(f"[dim]Total: {len(TIER_PRESETS)} presets[/dim]")
    console.print(
        "[dim]Use: atomate2siesta-presets show <name> for details and usage examples[/dim]"
    )
    console.print("[dim]Tip: Customize with override_params={'kpts': [6,6,6]}[/dim]\n")


def show_preset_details(preset_name):
    """Display detailed information for a preset."""
    preset = TIER_PRESETS[preset_name]

    console.print()

    # Header
    header = Panel(
        f"[bold cyan]{preset_name}[/bold cyan]\n"
        f"{preset.get('description', '')}\n\n"
        f"[dim]Tier:[/dim] [yellow]{preset.get('tier', 'intermediate')}[/yellow]",
        border_style="cyan",
        box=box.DOUBLE,
        title=f"[bold]{preset_name}[/bold]",
    )
    console.print(header)
    console.print()

    # Enabled modules
    enabled = preset.get("enabled_modules", [])
    if enabled:
        console.print("[bold]Enabled Modules:[/bold]")
        for module in enabled:
            console.print(f"  • {module}")
        console.print()

    # Parameters
    params = preset.get("recommended_params", {})
    if params:
        console.print("[bold]Parameters:[/bold]")

        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column(style="yellow", width=30)
        table.add_column(style="white")

        for key, value in params.items():
            table.add_row(key, str(value))

        console.print(table)
        console.print()

    # Usage examples
    console.print("[bold]Usage Examples:[/bold]")
    console.print()

    # Basic usage
    console.print("[dim]# Basic usage:[/dim]")
    console.print("[cyan]from atomate2.siesta.jobs.core import RelaxMaker[/cyan]")
    console.print(
        "[cyan]from atomate2.siesta.sets.tiers import apply_tier_preset[/cyan]"
    )
    console.print()
    console.print("[cyan]maker = RelaxMaker.fixed_cell_relaxation()[/cyan]")
    console.print(f"[cyan]maker = apply_tier_preset(maker, '{preset_name}')[/cyan]")
    console.print("[cyan]job = maker.make(structure)[/cyan]")
    console.print()

    # With overrides
    console.print("[dim]# Modify parameters:[/dim]")
    console.print("[cyan]maker = RelaxMaker.fixed_cell_relaxation()[/cyan]")
    console.print("[cyan]maker = apply_tier_preset([/cyan]")
    console.print("[cyan]    maker,[/cyan]")
    console.print(f"[cyan]    '{preset_name}',[/cyan]")
    console.print("[cyan]    override_params={'kpts': [6, 6, 6]},[/cyan]")
    console.print("[cyan])[/cyan]")
    console.print()


def show_category(category_name):
    """Display all presets in a category."""
    presets_in_category = CATEGORY_MAP[category_name]
    existing_presets = [p for p in presets_in_category if p in TIER_PRESETS]

    if not existing_presets:
        console.print(
            f"[yellow]No presets found in category '{category_name}'[/yellow]"
        )
        return

    console.print()
    console.print(f"[bold cyan]{category_name.upper()} Presets[/bold cyan]")
    console.print(f"[dim]{len(existing_presets)} preset(s)[/dim]\n")

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Preset", style="cyan", width=30)
    table.add_column("Description", style="white")
    table.add_column("Tier", style="yellow", width=15)

    for preset_name in sorted(existing_presets):
        preset = TIER_PRESETS[preset_name]
        table.add_row(
            preset_name,
            preset.get("description", ""),
            preset.get("tier", "intermediate"),
        )

    console.print(table)
    console.print()
    console.print(
        "[dim]Use: atomate2siesta-presets show <name> for usage examples[/dim]"
    )
    console.print("[dim]Tip: Customize with override_params={'kpts': [6,6,6]}[/dim]\n")


def show_examples():
    """Show usage examples."""
    console.print("\n[bold cyan]Usage Examples[/bold cyan]\n")

    # Example 1: List all presets
    console.print("[bold yellow]1. List all presets[/bold yellow]")
    console.print("[green]$ atomate2siesta-presets list[/green]")
    console.print()

    # Example 2: Show specific preset
    console.print("[bold yellow]2. Show details for a preset[/bold yellow]")
    console.print("[green]$ atomate2siesta-presets show adsorbate_screening[/green]")
    console.print()

    # Example 3: Search by category
    console.print("[bold yellow]3. Show presets in a category[/bold yellow]")
    console.print("[green]$ atomate2siesta-presets category surface[/green]")
    console.print()

    # Example 4: Search by tier
    console.print("[bold yellow]4. Search by tier level[/bold yellow]")
    console.print("[green]$ atomate2siesta-presets search --tier basic[/green]")
    console.print()

    # Example 5: Use in Python
    console.print("[bold yellow]5. Use in Python code[/bold yellow]")
    console.print("[cyan]from atomate2.siesta.jobs.core import RelaxMaker[/cyan]")
    console.print(
        "[cyan]from atomate2.siesta.sets.tiers import apply_tier_preset[/cyan]"
    )
    console.print()
    console.print("[cyan]maker = RelaxMaker.fixed_cell_relaxation()[/cyan]")
    console.print(
        "[cyan]maker = apply_tier_preset(maker, 'adsorbate_screening')[/cyan]"
    )
    console.print("[cyan]job = maker.make(structure)[/cyan]")
    console.print()


def show_tier_defaults():
    """Display tier-level default parameters."""
    console.print()

    if not TIER_DEFAULTS:
        console.print("[yellow]No tier defaults available[/yellow]")
        return

    # Dynamically count tier levels
    num_tiers = len(TIER_DEFAULTS)
    tier_word = "tier level" if num_tiers == 1 else "tier levels"

    # Header
    header = Panel(
        "[bold cyan]Tier-Level Defaults[/bold cyan]\n"
        f"[dim]Base parameter sets for the {num_tiers} {tier_word}[/dim]\n\n"
        "These defaults are used as starting points that can be overridden by specific presets.",
        border_style="cyan",
        box=box.DOUBLE,
    )
    console.print(header)
    console.print()

    # Automatically determine tier order (show all tiers dynamically)
    # Prefer standard order if they exist, then add any custom tiers
    standard_order = ["dirty", "basic", "intermediate", "advanced", "expert", "ultra"]
    tier_order = [t for t in standard_order if t in TIER_DEFAULTS]

    # Add any custom tiers not in standard order
    for tier_name in sorted(TIER_DEFAULTS.keys()):
        if tier_name not in tier_order:
            tier_order.append(tier_name)

    for tier_name in tier_order:
        params = TIER_DEFAULTS[tier_name]

        console.print(f"[bold yellow]Tier: {tier_name}[/bold yellow]")

        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column(style="cyan", width=30)
        table.add_column(style="white")

        for key, value in params.items():
            table.add_row(key, str(value))

        console.print(table)
        console.print()

    console.print("[dim]Note: Specific presets may override these defaults[/dim]")
    console.print("[dim]Use: atomate2siesta-presets list to see all presets[/dim]\n")


if __name__ == "__main__":
    cli()
