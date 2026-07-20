"""Display version information."""

from rich.console import Console
from rich.panel import Panel

console = Console()


def show_version():
    """Show version information."""
    try:
        import atomate2

        version = atomate2.__version__
    except (ImportError, AttributeError):
        version = "1.0.1 (development)"

    console.print()
    console.print(
        Panel(
            f"[bold cyan]atomate2siesta[/bold cyan]\n"
            f"[dim]Version:[/dim] [yellow]{version}[/yellow]\n"
            f"[dim]Default XC:[/dim] [green]PBE (GGA)[/green]\n"
            f"[dim]Config:[/dim] [blue]~/.atomate2siesta.yaml[/blue]\n"
            f"[dim]Python:[/dim] 3.8+",
            border_style="cyan",
            title="Version Info",
        )
    )
    console.print()
