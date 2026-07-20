"""Display complete overview of atomate2siesta."""

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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
        "📚 Comprehensive tutorials and documentation",
        border_style="cyan",
        box=box.DOUBLE,
    )
    console.print(header)
    console.print()

    # Quick stats
    stats_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    stats_table.add_column(style="bold yellow")
    stats_table.add_column(style="green")

    stats_table.add_row("📦 Workflows", "production workflows")
    stats_table.add_row("🍳 Recipes", "one-line workflow recipes")
    stats_table.add_row("🛠️ CLI Tools", "13 command-line utilities")
    stats_table.add_row("🎯 Tier Presets", "material-specific configurations")
    stats_table.add_row("📝 Tutorials", "comprehensive guides")
    stats_table.add_row("✅ Tests", "comprehensive automated test suite")
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
    console.print(
        "  📖 Docs:   [link]https://materialsproject.github.io/atomate2/[/link]"
    )
    console.print(
        "  🐙 GitHub: [link]https://github.com/materialsproject/atomate2[/link]"
    )
    console.print(
        "  💬 Issues: [link]https://github.com/materialsproject/atomate2/issues[/link]"
    )
    console.print()
