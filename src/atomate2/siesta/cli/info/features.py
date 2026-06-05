"""Display major features of atomate2siesta."""

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def show_features():
    """Display major features."""
    console.print("\n[bold cyan]Major Features[/bold cyan]\n")

    features = [
        ("🍳 Recipe Book", "one-line workflow recipes (significant code reduction)"),
        ("🎯 Tier System", "Material-specific parameter presets"),
        ("🛡️ Custodian", "Automatic error detection and recovery"),
        ("🔍 Convergence", "Automated k-point, basis, and cutoff testing"),
        ("📊 Visualization", "Automatic publication-quality plots"),
        ("🧪 Dry-Run Mode", "Preview workflows without running calculations"),
        ("🔧 Powerups", "Dynamic workflow modification functions"),
        ("💾 Database", "MongoDB integration for result storage"),
        ("☁️ HPC Support", "SLURM, PBS, SGE cluster integration"),
        ("📝 Schemas", "Structured output with SiestaTaskDoc"),
        ("🔬 Magnetic", "Automatic DM.InitSpin generation (FM/AFM/custom)"),
        ("📐 Structure Tools", "16 commands for structure manipulation"),
        ("📚 Tutorials", "Comprehensive tutorials with MyST integration"),
    ]

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column(style="bold yellow", width=20)
    table.add_column(style="white")

    for feature, description in features:
        table.add_row(feature, description)

    console.print(table)
    console.print()
