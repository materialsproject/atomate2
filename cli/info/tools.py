"""Display available CLI tools."""

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def show_cli_tools():
    """Display all CLI tools."""
    console.print("\n[bold cyan]Available CLI Tools[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Command", style="cyan", width=30)
    table.add_column("Description", style="white")
    table.add_column("Category", style="yellow", width=15)

    # Workflow Generation
    table.add_row(
        "atomate2siesta-maker",
        "Generate ready-to-run workflow scripts (13 templates)",
        "Workflows",
    )

    # Information & Discovery
    table.add_row(
        "atomate2siesta-info",
        "Display capabilities and information (this tool)",
        "Information",
    )
    table.add_row(
        "atomate2siesta-tutorials",
        "Browse and search 57+ tutorials",
        "Information",
    )
    table.add_row(
        "atomate2siesta-presets",
        "Show available tier presets and configurations",
        "Information",
    )
    table.add_row(
        "atomate2siesta-recipe",
        "Browse and search recipe book (39 recipes)",
        "Information",
    )

    # Infrastructure
    table.add_row(
        "atomate2siesta-database",
        "MongoDB database management and queries",
        "Infrastructure",
    )
    table.add_row(
        "atomate2siesta-cluster",
        "Remote HPC cluster setup with SIESTA",
        "Infrastructure",
    )
    table.add_row(
        "atomate2siesta-jobflow-remote",
        "Configure job submission to HPC clusters",
        "Infrastructure",
    )

    # Configuration
    table.add_row(
        "atomate2siesta-config",
        "Create configuration file (~/.atomate2siesta.yaml)",
        "Configuration",
    )
    table.add_row(
        "atomate2siesta-inputs",
        "Generate SIESTA input files from structure",
        "Configuration",
    )

    # Pseudopotentials
    table.add_row(
        "atomate2siesta-pseudos",
        "Install and manage pseudopotential libraries",
        "Pseudopotentials",
    )

    # Utilities
    table.add_row(
        "atomate2siesta-plot-pseudo",
        "Plot pseudopotential data from PSML files",
        "Utilities",
    )
    table.add_row(
        "atomate2siesta-structure",
        "Structure manipulation and conversion tools (16 commands)",
        "Utilities",
    )

    console.print(table)

    console.print("\n[dim]Use --help with any command for detailed options[/dim]")
    console.print("[dim]Example: atomate2siesta-maker --help[/dim]\n")
