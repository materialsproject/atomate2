"""CLI tutorial browser for atomate2siesta.

Interactive terminal UI for browsing, viewing, and copying tutorial files.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()


def get_tutorials_dir() -> Path:
    """Get the tutorials directory path."""
    # Get package root
    # __file__ is: src/atomate2/siesta/cli/tutorial/browser.py
    # package_dir should be: project root (6 levels up)
    package_dir = Path(__file__).parent.parent.parent.parent.parent.parent
    tutorials_dir = package_dir / "tutorials"

    if not tutorials_dir.exists():
        # If running from installed package, tutorials might not be in same location
        # For now, raise an error with helpful message
        msg = (
            f"Tutorials directory not found at {tutorials_dir}\n"
            "Please ensure atomate2siesta is installed from source "
            "with tutorials included."
        )
        raise FileNotFoundError(msg)

    return tutorials_dir


def discover_tutorials() -> dict[str, list[dict[str, Any]]]:
    """Discover all tutorial READMEs organized by category."""
    tutorials_dir = get_tutorials_dir()
    structure: dict[str, list[dict[str, Any]]] = {}

    for readme in tutorials_dir.rglob("README.md"):
        # Skip certain files
        if any(
            skip in str(readme)
            for skip in ["archive", "QUICKSTART", "TUTORIAL_TEMPLATES"]
        ):
            continue

        rel_path = readme.relative_to(tutorials_dir)
        if len(rel_path.parts) > 1:
            category = rel_path.parts[0]

            # Parse metadata
            content = readme.read_text()
            lines = content.split("\n")

            # Extract title
            title = "Untitled"
            for line in lines[:20]:
                if line.startswith("# "):
                    title = line[2:].strip()
                    if title.startswith("Tutorial:"):
                        title = title[9:].strip()
                    break

            # Extract metadata
            metadata = {}
            for line in lines[:15]:
                if line.startswith("**Category**:"):
                    metadata["category"] = line.split(":", 1)[1].strip()
                elif line.startswith("**Difficulty**:"):
                    metadata["difficulty"] = line.split(":", 1)[1].strip()
                elif line.startswith("**Time**:"):
                    metadata["time"] = line.split(":", 1)[1].strip()

            if category not in structure:
                structure[category] = []

            structure[category].append(
                {
                    "path": readme,
                    "rel_path": str(rel_path),
                    "name": rel_path.parent.name
                    if len(rel_path.parts) > 2
                    else category,
                    "title": title,
                    **metadata,
                }
            )

    return structure


def display_tutorial_list(structure: dict[str, list[dict[str, Any]]]) -> None:
    """Display organized list of tutorials."""
    # Category titles
    category_titles = {
        "00-structures": "Structure Files",
        "01-basics": "Basic Tutorials",
        "02-convergence": "Convergence Studies",
        "03-advanced-workflows": "Advanced Workflows",
        "04-infrastructure": "Infrastructure & Setup",
        "05-vibrational-properties": "Vibrational Properties",
        "06-surfaces-and-adsorption": "Surfaces & Adsorption",
        "07-advanced-features": "Advanced Features",
        "08-recipe-book": "Recipe Book",
        "09-structure-manipulation": "Structure Manipulation",
        "notebooks": "Jupyter Notebooks",
    }

    tree = Tree("📚 [bold cyan]Atomate2-SIESTA Tutorials[/bold cyan]")

    total = 0
    for category in sorted(structure.keys()):
        tutorials = sorted(structure[category], key=lambda x: x["name"])
        count = len(tutorials)
        total += count

        category_title = category_titles.get(
            category, category.replace("-", " ").title()
        )
        branch = tree.add(
            f"[bold yellow]{category_title}[/bold yellow] ({count} tutorials)"
        )

        for tutorial in tutorials:
            name = tutorial["name"]
            title = tutorial["title"]

            # Add difficulty and time if available
            meta = []
            if "difficulty" in tutorial:
                meta.append(f"[dim]{tutorial['difficulty']}[/dim]")
            if "time" in tutorial:
                meta.append(f"[dim]{tutorial['time']}[/dim]")

            meta_str = " • ".join(meta) if meta else ""
            branch.add(f"{name}: {title} {meta_str}")

    console.print(tree)
    console.print(f"\n[bold green]Total: {total} tutorials[/bold green]")


def display_tutorial(tutorial: dict[str, Any]) -> None:
    """Display a single tutorial's README."""
    console.clear()

    # Read the full README
    content = tutorial["path"].read_text()

    # Display metadata panel
    meta_table = Table.grid(padding=(0, 2))
    meta_table.add_column(style="bold cyan")
    meta_table.add_column()

    meta_table.add_row("📁 Location:", str(tutorial["rel_path"]))
    if "difficulty" in tutorial:
        meta_table.add_row("⭐ Difficulty:", tutorial["difficulty"])
    if "time" in tutorial:
        meta_table.add_row("⏱️  Time:", tutorial["time"])

    console.print(
        Panel(
            meta_table,
            title=f"[bold]{tutorial['title']}[/bold]",
            border_style="cyan",
        )
    )

    # Display markdown content
    console.print()
    md = Markdown(content)
    console.print(md)


def copy_tutorial(tutorial: dict[str, Any], dest: Path) -> None:
    """Copy tutorial files to destination directory."""
    source_dir = tutorial["path"].parent

    if dest.exists() and any(dest.iterdir()):
        console.print(
            f"[yellow]Warning: Directory {dest} exists and is not empty.[/yellow]"
        )
        if not click.confirm("Overwrite?", default=False):
            console.print("[red]Cancelled.[/red]")
            return

    # Copy all files from tutorial directory
    dest.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    for file in source_dir.iterdir():
        if file.is_file():
            shutil.copy2(file, dest / file.name)
            console.print(f"  Copied: {file.name}")
            copied_count += 1

    console.print(f"\n[green]✓ {copied_count} files copied to {dest}[/green]")


@click.group(invoke_without_command=True)
@click.pass_context
def tutorials(ctx: click.Context) -> None:
    """Browse and manage atomate2siesta tutorials.

    Interactive tutorial browser with rich terminal UI.
    Run without arguments for interactive mode.

    \b
    Examples:
      atomate2siesta-tutorials              # Interactive mode
      atomate2siesta-tutorials list         # List all tutorials
      atomate2siesta-tutorials search phonon # Search tutorials
      atomate2siesta-tutorials show relaxation # Show specific tutorial
      atomate2siesta-tutorials copy relaxation # Copy tutorial files
    """  # noqa: D301
    if ctx.invoked_subcommand is None:
        # Interactive mode
        browse_interactive()


@tutorials.command("list")
def list_tutorials() -> None:
    """List all available tutorials organized by category."""
    structure = discover_tutorials()
    display_tutorial_list(structure)


@tutorials.command()
@click.argument("query", required=False)
def search(query: str | None) -> None:
    """Search tutorials by name or keywords.

    \b
    Examples:
      atomate2siesta-tutorials search phonon
      atomate2siesta-tutorials search relaxation
      atomate2siesta-tutorials search "band structure"
    """  # noqa: D301
    structure = discover_tutorials()

    if not query:
        console.print("[yellow]Usage: atomate2siesta-tutorials search <query>[/yellow]")
        console.print("\nExample: atomate2siesta-tutorials search phonon")
        return

    query_lower = query.lower()
    results = []

    for tutorials in structure.values():
        for tutorial in tutorials:
            # Search in title, name, and path
            searchable = (
                f"{tutorial['title']} {tutorial['name']} {tutorial['rel_path']}".lower()
            )
            if query_lower in searchable:
                results.append(tutorial)

    if not results:
        console.print(f"[yellow]No tutorials found matching '{query}'[/yellow]")
        return

    console.print(
        f"[bold cyan]Found {len(results)} tutorials matching '{query}':[/bold cyan]\n"
    )
    for tutorial in results:
        console.print(f"  • [bold]{tutorial['name']}[/bold]: {tutorial['title']}")
        console.print(f"    [dim]{tutorial['rel_path']}[/dim]")
        if "difficulty" in tutorial and "time" in tutorial:
            console.print(
                f"    [dim]{tutorial['difficulty']} • {tutorial['time']}[/dim]"
            )
        console.print()


@tutorials.command()
@click.argument("tutorial_name")
def show(tutorial_name: str) -> None:
    """Display a specific tutorial's README.

    \b
    Examples:
      atomate2siesta-tutorials show 01-relaxation
      atomate2siesta-tutorials show phonons
    """  # noqa: D301
    structure = discover_tutorials()

    # Find tutorial (fuzzy match)
    found = None
    for tutorials in structure.values():
        for tutorial in tutorials:
            if (
                tutorial["name"] == tutorial_name
                or tutorial_name in tutorial["name"]
                or tutorial_name in tutorial["rel_path"]
            ):
                found = tutorial
                break
        if found:
            break

    if not found:
        console.print(f"[red]Tutorial '{tutorial_name}' not found.[/red]")
        console.print("\n[yellow]Try: atomate2siesta-tutorials list[/yellow]")
        return

    display_tutorial(found)


@tutorials.command()
@click.argument("tutorial_name")
@click.option(
    "-o", "--output", type=click.Path(), help="Output directory (default: current dir)"
)
def copy(tutorial_name: str, output: str | None) -> None:
    """Copy tutorial files to current or specified directory.

    \b
    Examples:
      atomate2siesta-tutorials copy 01-relaxation
      atomate2siesta-tutorials copy phonon -o my-phonon-calc
    """  # noqa: D301
    structure = discover_tutorials()

    # Find tutorial
    found = None
    for tutorials in structure.values():
        for tutorial in tutorials:
            if (
                tutorial["name"] == tutorial_name
                or tutorial_name in tutorial["name"]
                or tutorial_name in tutorial["rel_path"]
            ):
                found = tutorial
                break
        if found:
            break

    if not found:
        console.print(f"[red]Tutorial '{tutorial_name}' not found.[/red]")
        console.print("\n[yellow]Try: atomate2siesta-tutorials list[/yellow]")
        return

    # Determine destination
    dest = Path(output) if output else Path.cwd() / tutorial["name"]

    console.print(f"[cyan]Copying tutorial to: {dest}[/cyan]\n")
    copy_tutorial(found, dest)


def browse_interactive() -> None:
    """Interactive tutorial browser."""
    try:
        import questionary
    except ImportError:
        console.print(
            "[red]Error: 'questionary' package required for interactive mode.[/red]"
        )
        console.print("[yellow]Install with: pip install questionary[/yellow]")
        console.print("\n[cyan]You can still use non-interactive commands:[/cyan]")
        console.print("  atomate2siesta-tutorials list")
        console.print("  atomate2siesta-tutorials search <query>")
        console.print("  atomate2siesta-tutorials show <name>")
        console.print("  atomate2siesta-tutorials copy <name>")
        return

    structure = discover_tutorials()

    while True:
        # Show welcome
        console.clear()
        console.print(
            Panel.fit(
                "[bold cyan]Atomate2-SIESTA Tutorial Browser[/bold cyan]\n"
                f"Browse {sum(len(t) for t in structure.values())} "
                "tutorials interactively",
                border_style="cyan",
            )
        )

        # Main menu
        choices = [
            "📚 Browse all tutorials",
            "🔍 Search tutorials",
            "📋 List by category",
            "❌ Exit",
        ]

        action = questionary.select("What would you like to do?", choices=choices).ask()

        if not action or action == "❌ Exit":
            console.print("[cyan]Goodbye![/cyan]")
            break

        if action == "📚 Browse all tutorials":
            browse_all_tutorials(structure)
        elif action == "🔍 Search tutorials":
            search_interactive(structure)
        elif action == "📋 List by category":
            display_tutorial_list(structure)
            console.input("\n[dim]Press Enter to continue...[/dim]")


def browse_all_tutorials(structure: dict[str, list[dict[str, Any]]]) -> None:
    """Browse tutorials interactively."""
    import questionary

    # Select category
    category_titles = {
        "00-structures": "Structure Files",
        "01-basics": "⭐ Basic Tutorials (START HERE)",
        "02-convergence": "Convergence Studies",
        "03-advanced-workflows": "Advanced Workflows",
        "04-infrastructure": "Infrastructure & Setup",
        "05-vibrational-properties": "Vibrational Properties",
        "06-surfaces-and-adsorption": "Surfaces & Adsorption",
        "07-advanced-features": "Advanced Features",
        "08-recipe-book": "Recipe Book",
        "09-structure-manipulation": "Structure Manipulation",
        "notebooks": "Jupyter Notebooks",
    }

    category_choices = [
        f"{category_titles.get(cat, cat)} ({len(structure[cat])} tutorials)"
        for cat in sorted(structure.keys())
    ]
    category_choices.append("← Back")

    category = questionary.select("Select category:", choices=category_choices).ask()

    if not category or category == "← Back":
        return

    # Extract category key from selection
    category_key = None
    for key in sorted(structure.keys()):
        if category_titles.get(key, key) in category:
            category_key = key
            break

    if not category_key:
        return

    # Select tutorial
    tutorials = sorted(structure[category_key], key=lambda x: x["name"])
    tutorial_choices = []
    for t in tutorials:
        meta = []
        if "difficulty" in t:
            meta.append(t["difficulty"])
        if "time" in t:
            meta.append(t["time"])
        meta_str = f" ({' • '.join(meta)})" if meta else ""
        tutorial_choices.append(f"{t['name']}: {t['title']}{meta_str}")

    tutorial_choices.append("← Back")

    tutorial_choice = questionary.select(
        "Select tutorial:", choices=tutorial_choices
    ).ask()

    if not tutorial_choice or tutorial_choice == "← Back":
        return

    # Find selected tutorial
    tutorial_name = tutorial_choice.split(":")[0].strip()
    tutorial = next(t for t in tutorials if t["name"] == tutorial_name)

    # Show tutorial actions
    while True:
        display_tutorial(tutorial)

        action = questionary.select(
            "\nWhat would you like to do?",
            choices=[
                "📋 Copy to current directory",
                "📂 Copy to custom directory",
                "← Back",
            ],
        ).ask()

        if not action or action == "← Back":
            break

        if action == "📋 Copy to current directory":
            dest = Path.cwd() / tutorial["name"]
            copy_tutorial(tutorial, dest)
            console.input("\n[dim]Press Enter to continue...[/dim]")

        elif action == "📂 Copy to custom directory":
            dest_path = questionary.path("Enter destination path:").ask()
            if dest_path:
                copy_tutorial(tutorial, Path(dest_path))
                console.input("\n[dim]Press Enter to continue...[/dim]")


def search_interactive(structure: dict[str, list[dict[str, Any]]]) -> None:
    """Interactive search."""
    import questionary

    query = questionary.text("Search query:").ask()
    if not query:
        return

    query_lower = query.lower()
    results = []

    for tutorials in structure.values():
        for tutorial in tutorials:
            searchable = (
                f"{tutorial['title']} {tutorial['name']} {tutorial['rel_path']}".lower()
            )
            if query_lower in searchable:
                results.append(tutorial)

    if not results:
        console.print(f"[yellow]No tutorials found matching '{query}'[/yellow]")
        console.input("\n[dim]Press Enter to continue...[/dim]")
        return

    # Show results
    console.print(f"\n[bold cyan]Found {len(results)} tutorials:[/bold cyan]\n")

    tutorial_choices = [f"{t['name']}: {t['title']}" for t in results]
    tutorial_choices.append("← Back")

    choice = questionary.select(
        "Select tutorial to view:", choices=tutorial_choices
    ).ask()

    if choice and choice != "← Back":
        tutorial_name = choice.split(":")[0].strip()
        tutorial = next(t for t in results if t["name"] == tutorial_name)
        display_tutorial(tutorial)
        console.input("\n[dim]Press Enter to continue...[/dim]")


if __name__ == "__main__":
    tutorials()
