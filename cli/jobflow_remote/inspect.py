"""Job inspection command for viewing job details and FDF parameters.

This module provides read-only inspection of jobflow-remote jobs,
including SIESTA FDF parameter extraction.
"""

from __future__ import annotations

import click
from rich.console import Console

from .db_query import (
    display_job_info,
    extract_fdf_parameters,
    get_actual_fdf_file,
    get_job_details_from_db,
    get_job_info_from_jf,
    get_tier_defaults,
)

console = Console()


@click.command()
@click.argument("job_id")
@click.option(
    "--full",
    is_flag=True,
    help="Show full job details including FDF parameters",
)
@click.option(
    "--fdf-only",
    is_flag=True,
    help="Show only FDF parameters",
)
@click.option(
    "--show-all-defaults",
    is_flag=True,
    help="Show what tier preset contributes (includes default parameters)",
)
@click.option(
    "--show-actual-fdf",
    is_flag=True,
    help="Show the actual generated siesta.fdf file from job run directory",
)
@click.pass_context
def inspect(
    ctx,
    job_id: str,
    full: bool,
    fdf_only: bool,
    show_all_defaults: bool,
    show_actual_fdf: bool,
):
    """Inspect job details and SIESTA FDF parameters.

    This command provides read-only inspection of job information stored
    in the jobflow-remote database. It can display:

    \b
    - Basic job information (name, state, worker, UUID)
    - SIESTA FDF input parameters (requires pymongo)
    - Complete job document structure (--full)

    \b
    Examples:
        # Basic job info
        atomate2siesta-jobflow-remote -p production job inspect 70

        # Include FDF parameters
        atomate2siesta-jobflow-remote -p production job inspect 70 --full

        # Show only FDF parameters
        atomate2siesta-jobflow-remote -p production job inspect 70 --fdf-only

    Args:
        job_id: Job index number or UUID
    """
    project_name = ctx.obj.get("project_name", "atomate2siesta")

    console.print(
        f"\n[bold cyan]Inspecting job {job_id} in project '{project_name}'[/bold cyan]\n"
    )

    # Get basic job info from jf CLI
    job_info = get_job_info_from_jf(project_name, job_id)

    if not job_info:
        console.print("[red]Failed to retrieve job information[/red]")
        raise click.Abort()

    # Get detailed job document from MongoDB if requested
    fdf_params = None
    tier_level = None
    job_doc = None

    if full or fdf_only or show_all_defaults or show_actual_fdf:
        job_doc = get_job_details_from_db(project_name, job_id)
        if job_doc:
            fdf_params = extract_fdf_parameters(job_doc)
            job_info["fdf_params"] = fdf_params

            # Extract tier level if present
            if fdf_params and "tier" in fdf_params:
                tier_level = fdf_params["tier"]
        else:
            console.print(
                "\n[yellow]Warning:[/yellow] Could not retrieve FDF parameters from database"
            )
            console.print("[yellow]Hint:[/yellow] Install pymongo: pip install pymongo")

    # Display information
    if fdf_only:
        if fdf_params:
            from rich.panel import Panel
            from rich.syntax import Syntax

            import yaml

            fdf_text = yaml.dump(fdf_params, default_flow_style=False, sort_keys=False)
            syntax = Syntax(fdf_text, "yaml", theme="monokai", line_numbers=True)
            console.print(
                Panel(
                    syntax,
                    title=f"[bold yellow]SIESTA FDF Parameters (Job {job_id})[/bold yellow]",
                    border_style="yellow",
                )
            )
        else:
            console.print(
                "[yellow]No FDF parameters found or could not access database[/yellow]"
            )
    else:
        display_job_info(job_info, include_fdf=(full and fdf_params is not None))

    # Add explanatory note about user parameters vs full FDF
    if fdf_params and not fdf_only:
        from rich.panel import Panel

        console.print("\n")
        console.print(
            Panel.fit(
                "[bold cyan]ℹ️  Parameter Display Explanation[/bold cyan]\n\n"
                "The parameters shown above are [bold]user-configurable parameters[/bold] from the job definition.\n"
                "The actual generated [bold]siesta.fdf[/bold] file contains many more parameters:\n\n"
                "  • SIESTA defaults (marked with '# SIESTA DEFAULT VALUE')\n"
                "  • Tier preset contributions (from dataclass modules)\n"
                "  • Auto-generated blocks (k-points, structure, etc.)\n\n"
                "[bold]Options to see more:[/bold]\n"
                "  • [cyan]--show-all-defaults[/cyan] - Show what tier preset contributes\n"
                "  • [cyan]--show-actual-fdf[/cyan] - Show the actual generated FDF file\n\n"
                "[dim]The parameters shown above are the ones you can modify with 'job recreate'.[/dim]",
                title="Understanding FDF Parameters",
                border_style="cyan",
            )
        )

    # Show tier defaults if requested
    if show_all_defaults and tier_level:
        console.print("\n")
        _display_tier_defaults(tier_level)

    # Show actual FDF file if requested
    if show_actual_fdf and job_doc:
        console.print("\n")
        _display_actual_fdf(project_name, job_doc, job_id)

    # Show usage hints
    if not fdf_only:
        console.print("\n[dim]" + "─" * 70 + "[/dim]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print(
            "  • Modify parameters: [cyan]job modify-db[/cyan] (modifies in place)"
        )
        console.print("  • View full params: [cyan]job inspect --full[/cyan]")
        console.print(
            "  • View tier defaults: [cyan]job inspect --show-all-defaults[/cyan]"
        )
        console.print("  • View actual FDF: [cyan]job inspect --show-actual-fdf[/cyan]")
        console.print()


def _display_tier_defaults(tier_level: str) -> None:
    """Display tier default parameters.

    Args:
        tier_level: Tier level (basic_dirty, basic, intermediate, advanced, expert)
    """
    from rich.panel import Panel
    from rich.syntax import Syntax
    import yaml

    console.print(
        f"[bold cyan]Fetching tier '{tier_level}' default parameters...[/bold cyan]"
    )

    tier_defaults = get_tier_defaults(tier_level)

    if tier_defaults:
        # Convert to YAML for display
        yaml_text = yaml.dump(tier_defaults, default_flow_style=False, sort_keys=False)
        syntax = Syntax(yaml_text, "yaml", theme="monokai", line_numbers=True)

        console.print(
            Panel(
                syntax,
                title=f"[bold yellow]Tier '{tier_level}' Default Parameters[/bold yellow]",
                subtitle=f"[dim]{len(tier_defaults)} parameters from dataclass modules[/dim]",
                border_style="yellow",
            )
        )

        console.print(
            "\n[dim]These are the default parameters that the tier preset contributes.[/dim]"
        )
        console.print(
            "[dim]User parameters (shown above) override these defaults.[/dim]"
        )
    else:
        console.print("[yellow]Could not retrieve tier defaults[/yellow]")


def _display_actual_fdf(project_name: str, job_doc: dict, job_id: str) -> None:
    """Display the actual generated siesta.fdf file.

    Args:
        project_name: Project name
        job_doc: Job document from MongoDB
        job_id: Job ID
    """
    from rich.panel import Panel
    from rich.syntax import Syntax

    console.print(
        "[bold cyan]Fetching actual siesta.fdf file from run directory...[/bold cyan]"
    )

    fdf_content = get_actual_fdf_file(project_name, job_doc)

    if fdf_content:
        # Display with syntax highlighting
        syntax = Syntax(fdf_content, "fortran", theme="monokai", line_numbers=True)

        run_dir = job_doc.get("run_dir", "unknown")
        console.print(
            Panel(
                syntax,
                title=f"[bold green]Actual Generated siesta.fdf (Job {job_id})[/bold green]",
                subtitle=f"[dim]From: {run_dir}[/dim]",
                border_style="green",
            )
        )

        # Count lines
        line_count = len(fdf_content.split("\n"))
        console.print(f"\n[dim]File contains {line_count} lines[/dim]")
    else:
        console.print("[yellow]Could not retrieve actual FDF file[/yellow]")
        console.print(
            "[dim]The job may not have run yet, or the run directory is not accessible.[/dim]"
        )
