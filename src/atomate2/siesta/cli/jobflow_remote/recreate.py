"""Job recreation command for safely modifying FDF parameters.

This module provides the SAFE method for parameter modification:
it creates a NEW flow with modified parameters instead of editing
the existing job in the database.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel

from .db_query import extract_fdf_parameters, get_job_details_from_db
from .parameter_modifier import (
    merge_parameters,
    parse_multiple_parameters,
    preview_parameter_changes,
    validate_all_parameters,
)

console = Console()


@click.command()
@click.argument("job_id")
@click.option(
    "--modify",
    "-m",
    multiple=True,
    help="Parameter to modify (format: key=value). Can be used multiple times.",
)
@click.option(
    "--output-script",
    "-o",
    type=click.Path(),
    help="Output Python script file (default: recreate_job_{job_id}.py)",
)
@click.option(
    "--preview-only",
    is_flag=True,
    help="Preview changes without creating script",
)
@click.pass_context
def recreate(
    ctx,
    job_id: str,
    modify: tuple[str],
    output_script: str | None,
    preview_only: bool,
):
    """Recreate job with modified FDF parameters (SAFE method).

    This command extracts the job configuration from the database and
    generates a NEW Python script with modified parameters. This is the
    SAFE method because it:

    \b
    - Does NOT modify existing job in database
    - Creates a new flow that you review before submission
    - Preserves original job for comparison
    - Allows full customization via Python script

    \b
    Examples:
        # Modify k-points mesh
        atomate2siesta-jobflow-remote -p prod job recreate 70 -m "kpts=[6,6,1]"

        # Modify multiple parameters
        atomate2siesta-jobflow-remote -p prod job recreate 70 \\
            -m "kpts=[6,6,1]" \\
            -m "Mesh.Cutoff=350 Ry" \\
            -m "SCF.Mixer.Weight=0.01"

        # Preview changes without creating script
        atomate2siesta-jobflow-remote -p prod job recreate 70 \\
            -m "kpts=[8,8,1]" --preview-only

    Args:
        job_id: Job index number or UUID to recreate
    """
    project_name = ctx.obj.get("project_name", "atomate2siesta")

    console.print(
        f"\n[bold cyan]Recreating job {job_id} from project '{project_name}'[/bold cyan]\n"
    )

    # Check if modifications specified
    if not modify:
        console.print(
            "[red]Error:[/red] No modifications specified. "
            "Use -m/--modify to specify parameters."
        )
        console.print("\n[yellow]Example:[/yellow]")
        console.print(
            f"  atomate2siesta-jobflow-remote -p {project_name} job recreate {job_id} "
            '-m "kpts=[6,6,1]"'
        )
        raise click.Abort()

    # Parse parameter modifications
    console.print("[bold]Parsing parameter modifications...[/bold]")
    new_params = parse_multiple_parameters(list(modify))

    if not new_params:
        console.print("[red]Error:[/red] No valid parameters parsed")
        raise click.Abort()

    # Validate parameters
    console.print("[bold]Validating parameters...[/bold]")
    if not validate_all_parameters(new_params):
        console.print("\n[red]Parameter validation failed[/red]")
        raise click.Abort()

    console.print("[green]✓[/green] All parameters valid\n")

    # Get original job details from database
    console.print("[bold]Fetching job details from database...[/bold]")
    job_doc = get_job_details_from_db(project_name, job_id)

    if not job_doc:
        console.print(
            "[red]Error:[/red] Could not retrieve job from database. "
            "Is pymongo installed?"
        )
        raise click.Abort()

    # Extract original FDF parameters
    original_params = extract_fdf_parameters(job_doc) or {}
    console.print(
        f"[green]✓[/green] Found {len(original_params)} original parameters\n"
    )

    # Merge parameters
    merged_params = merge_parameters(original_params, new_params)

    # Preview changes
    console.print()
    preview_parameter_changes(original_params, merged_params)
    console.print()

    if preview_only:
        console.print("[yellow]Preview mode - no script created[/yellow]")
        return

    # Generate Python script
    if output_script is None:
        output_script = f"recreate_job_{job_id}.py"

    console.print(f"[bold]Generating Python script: {output_script}[/bold]\n")

    # Extract job metadata
    job_data = job_doc.get("job", {})
    job_name = job_data.get("name", "RelaxJob")
    job_index = job_doc.get("db_id", job_id)

    # Generate script content
    script_content = _generate_recreation_script(
        job_name=job_name,
        job_index=job_index,
        project_name=project_name,
        original_params=original_params,
        modified_params=merged_params,
    )

    # Write script
    try:
        with open(output_script, "w") as f:
            f.write(script_content)

        console.print(f"[green]✓[/green] Script created: {output_script}\n")

        # Display next steps
        console.print(
            Panel.fit(
                f"[bold green]Success![/bold green] Recreation script generated.\n\n"
                f"[bold]Review and edit:[/bold]\n"
                f"  {output_script}\n\n"
                f"[bold]Submit new flow:[/bold]\n"
                f"  python {output_script}\n\n"
                f"[bold yellow]Note:[/bold yellow] This creates a NEW flow. "
                f"The original job {job_id} is unchanged.",
                title="Next Steps",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"[red]Error writing script:[/red] {e}")
        raise click.Abort()


def _generate_recreation_script(
    job_name: str,
    job_index: int,
    project_name: str,
    original_params: dict,
    modified_params: dict,
) -> str:
    """Generate Python script for job recreation.

    Args:
        job_name: Original job name
        job_index: Original job index
        project_name: Jobflow-remote project name
        original_params: Original FDF parameters
        modified_params: Modified FDP parameters

    Returns
    -------
        Python script content as string
    """
    import pprint

    # Convert params to pretty Python dict representation
    params_str = pprint.pformat(modified_params, width=80, compact=False)

    script = f'''#!/usr/bin/env python3
"""Recreated job from jobflow-remote job {job_index}.

This script was auto-generated by atomate2siesta-jobflow-remote.
Review and modify as needed before running.

Original job: {job_name} (index {job_index})
Project: {project_name}
"""

from pathlib import Path
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow.managers.local import run_locally

# TODO: Load or define your structure
# Option 1: Load from file
structure_file = "path/to/structure.cif"  # CHANGE THIS
if Path(structure_file).exists():
    structure = Structure.from_file(structure_file)
else:
    raise FileNotFoundError(
        f"Structure file not found: {{structure_file}}\\n"
        "Please update the structure_file path or create the structure programmatically."
    )

# Modified FDF parameters
# (Original job {job_index} parameters with your modifications applied)
user_params = {params_str}

# Create job maker with modified parameters
# TODO: Adjust maker type if needed (RelaxMaker, StaticMaker, etc.)
maker = RelaxMaker(user_params=user_params)

# Generate job
job = maker.make(structure)

# Run locally (or submit to jobflow-remote)
# Option 1: Run locally for testing
# run_locally(job, create_folders=True)

# Option 2: Submit to jobflow-remote
# from jobflow_remote import submit_flow
# submit_flow(job, worker="{project_name}")

print("Job created successfully!")
print("\\nNext steps:")
print("  1. Review this script")
print("  2. Update structure file path")
print("  3. Choose execution method (local or remote)")
print("  4. Run: python {{Path(__file__).name}}")
'''

    return script
