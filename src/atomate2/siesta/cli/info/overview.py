"""Display complete overview of atomate2siesta."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


def show_overview():
    """Display complete overview."""
    console.print()

    # Header
    header = Panel(
        "[bold cyan]atomate2siesta[/bold cyan]\n"
        "[dim]Automated SIESTA Workflows for Materials Science[/dim]\n\n"
        "🔬 Production-ready DFT workflows with intelligent error handling\n"
        "⚡ significant code reduction with Recipe Book\n"
        "🛠️ 13 CLI tools for workflow automation\n"
        "📚 57+ tutorials and comprehensive documentation",
        border_style="cyan",
        box=box.DOUBLE,
    )
    console.print(header)
    console.print()

    # Quick stats
    stats_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    stats_table.add_column(style="bold yellow")
    stats_table.add_column(style="green")

    stats_table.add_row("📦 Workflows", "13+ production workflows")
    stats_table.add_row("🍳 Recipes", "39 one-line workflows")
    stats_table.add_row("🛠️ CLI Tools", "13 command-line utilities")
    stats_table.add_row("🎯 Tier Presets", "26 material-specific configurations")
    stats_table.add_row("📝 Tutorials", "57+ comprehensive guides")
    stats_table.add_row("✅ Tests", "2,038 tests (58-60% coverage)")
    stats_table.add_row("🔧 Default XC", "PBE (GGA)")

    console.print(stats_table)
    console.print()

    # Quick commands
    console.print("[bold]Quick Commands:[/bold]")
    console.print("  [cyan]atomate2siesta-info tools[/cyan]      - List all CLI tools")
    console.print("  [cyan]atomate2siesta-info workflows[/cyan]  - List workflow types")
    console.print("  [cyan]atomate2siesta-info features[/cyan]   - List major features")
    console.print("  [cyan]atomate2siesta-info examples[/cyan]   - Show quick examples")
    console.print()

    # Links
    console.print("[bold]Resources:[/bold]")
    console.print("  📖 Docs:   [link]https://atomate2siesta.readthedocs.io[/link]")
    console.print(
        "  🐙 GitHub: [link]https://github.com/arsalan-akhtar/atomate2siesta[/link]"
    )
    console.print(
        "  💬 Issues: [link]https://github.com/arsalan-akhtar/atomate2siesta/issues[/link]"
    )
    console.print()
