"""Job database modification command (RISKY method).

This module provides DIRECT database modification of job parameters.
This is RISKY and should only be used when absolutely necessary.

⚠️  WARNING: This modifies jobs directly in MongoDB!
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from .db_query import (
    extract_fdf_parameters,
    get_job_details_from_db,
    get_project_config,
)
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
    "--param",
    "-p",
    multiple=True,
    help="Parameter to modify (format: key=value). Can be used multiple times.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Skip safety confirmation prompts (USE WITH EXTREME CAUTION)",
)
@click.option(
    "--remove",
    "-r",
    help="Parameter(s) to remove. Use comma to separate multiple keys (e.g., 'Key1,Key2,Key3').",
)
@click.option(
    "--backup",
    is_flag=True,
    default=True,
    help="Create backup of job document before modification (default: True)",
)
@click.pass_context
def modify_db(
    ctx,
    job_id: str,
    param: tuple[str],
    remove: str | None,
    force: bool,
    backup: bool,
):
    """Modify job FDF parameters directly in database (RISKY).

    ⚠️  WARNING: This command modifies jobs DIRECTLY in the MongoDB database!

    This is the RISKY method and should ONLY be used when:
    \b
    - You understand the risks
    - You have database backups
    - Recreating the flow is not feasible
    - The job is in a state that allows modification

    \b
    SAFER ALTERNATIVE: Use 'job recreate' instead!

    \b
    Examples:
        # Modify with confirmation
        atomate2siesta-jobflow-remote -p prod job modify-db 70 \\
            --param "kpts=[6,6,1]"

        # Modify multiple parameters
        atomate2siesta-jobflow-remote -p prod job modify-db 70 \\
            --param "kpts=[6,6,1]" \\
            --param "Mesh.Cutoff=350 Ry"

        # Remove a parameter
        atomate2siesta-jobflow-remote -p prod job modify-db 70 \\
            --remove "Slab.DipoleCorrection"

        # Remove multiple parameters (comma-separated)
        atomate2siesta-jobflow-remote -p prod job modify-db 70 \\
            --remove "Slab.DipoleCorrection,OtherParam,ThirdParam"

        # Modify and remove in same command
        atomate2siesta-jobflow-remote -p prod job modify-db 70 \\
            --param "kpts=[6,6,1]" \\
            --remove "Slab.DipoleCorrection"

        # Force mode (skip confirmations - dangerous!)
        atomate2siesta-jobflow-remote -p prod job modify-db 70 \\
            --param "kpts=[8,8,1]" --force

    Args:
        job_id: Job index number or UUID to modify
    """
    project_name = ctx.obj.get("project_name", "atomate2siesta")

    # Display BIG warning
    console.print()
    console.print(
        Panel.fit(
            "[bold red]⚠️  DANGER - DATABASE MODIFICATION ⚠️[/bold red]\n\n"
            "You are about to DIRECTLY modify a job in the MongoDB database.\n"
            "This can cause:\n"
            "  • Job execution failures\n"
            "  • Data corruption\n"
            "  • Loss of reproducibility\n"
            "  • Broken workflow tracking\n\n"
            "[bold yellow]SAFER ALTERNATIVE:[/bold yellow] Use 'job recreate' to create a new flow.",
            title="⚠️  WARNING",
            border_style="red",
        )
    )
    console.print()

    if not force:
        if not Confirm.ask(
            "[bold red]Do you understand the risks and wish to proceed?[/bold red]",
            default=False,
        ):
            console.print("[yellow]Operation cancelled[/yellow]")
            return

    # Check if modifications specified
    if not param and not remove:
        console.print(
            "[red]Error:[/red] No modifications specified. "
            "Use --param to add/modify or --remove to delete parameters."
        )
        console.print("\n[yellow]Examples:[/yellow]")
        console.print(
            f"  atomate2siesta-jobflow-remote -p {project_name} job modify-db {job_id} "
            '--param "kpts=[6,6,1]"'
        )
        console.print(
            f"  atomate2siesta-jobflow-remote -p {project_name} job modify-db {job_id} "
            '--remove "Slab.DipoleCorrection"'
        )
        raise click.Abort()

    # Parse parameter modifications
    console.print("[bold]Parsing parameter modifications...[/bold]")
    new_params = parse_multiple_parameters(list(param)) if param else {}

    # Parse comma-separated remove keys
    remove_keys = []
    if remove:
        remove_keys = [k.strip() for k in remove.split(",") if k.strip()]

    if not new_params and not remove_keys:
        console.print("[red]Error:[/red] No valid parameters parsed")
        raise click.Abort()

    if remove_keys:
        console.print(f"[cyan]Parameters to remove:[/cyan] {', '.join(remove_keys)}")

    # Validate parameters
    console.print("[bold]Validating parameters...[/bold]")
    if not validate_all_parameters(new_params):
        console.print("\n[red]Parameter validation failed[/red]")
        raise click.Abort()

    console.print("[green]✓[/green] All parameters valid\n")

    # Get job details from database
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

    # Merge parameters (add/modify new_params, remove remove_keys)
    merged_params = merge_parameters(original_params, new_params, remove_keys)

    # Preview changes
    console.print()
    preview_parameter_changes(original_params, merged_params)
    console.print()

    # Final confirmation
    if not force:
        console.print(
            Panel.fit(
                "[bold yellow]Final confirmation required[/bold yellow]\n\n"
                f"You are about to modify job {job_id} in project '{project_name}'.\n"
                "This will PERMANENTLY change the job parameters in the database.\n\n"
                f"Backup will be created: {backup}",
                border_style="yellow",
            )
        )
        console.print()

        if not Confirm.ask(
            "[bold]Proceed with database modification?[/bold]",
            default=False,
        ):
            console.print("[yellow]Operation cancelled[/yellow]")
            return

    # Perform database modification
    console.print("\n[bold]Modifying database...[/bold]")

    try:
        success = _modify_job_in_database(
            project_name=project_name,
            job_id=job_id,
            job_doc=job_doc,
            new_params=new_params,  # Pass only NEW params (not pre-merged)
            remove_keys=remove_keys,  # Pass keys to remove
            create_backup=backup,
        )

        if success:
            console.print()
            console.print(
                Panel.fit(
                    "[bold green]✓ Database modified successfully[/bold green]\n\n"
                    f"Job {job_id} parameters have been updated.\n\n"
                    "[bold]Next steps:[/bold]\n"
                    "  • Verify parameters: [cyan]job inspect {job_id} --full[/cyan]\n"
                    f"  • Rerun job: [cyan]jf -p {project_name} job rerun {job_id}[/cyan]\n"
                    "  • Monitor execution: [cyan]jf -p {project_name} job info {job_id}[/cyan]",
                    title="Success",
                    border_style="green",
                )
            )
        else:
            console.print("[red]✗ Database modification failed[/red]")
            raise click.Abort()

    except Exception as e:
        console.print(f"[red]Error during modification:[/red] {e}")
        raise click.Abort()


def _modify_job_in_database(
    project_name: str,
    job_id: str,
    job_doc: dict,
    new_params: dict,
    remove_keys: list[str] | None = None,
    create_backup: bool = True,
) -> bool:
    """Modify job parameters directly in MongoDB.

    Args:
        project_name: Project name
        job_id: Job ID
        job_doc: Original job document
        new_params: New parameters to apply
        remove_keys: List of parameter keys to remove
        create_backup: Whether to create backup collection

    Returns:
        True if successful, False otherwise
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        console.print("[red]Error:[/red] pymongo not installed")
        return False

    # Get MongoDB connection details
    config = get_project_config(project_name)
    if not config:
        return False

    try:
        # Connect to MongoDB
        queue_store = config.get("queue", {}).get("store", {})
        host = queue_store.get("host", "localhost")
        port = queue_store.get("port", 27017)
        database = queue_store.get("database", "jobflow_remote")
        collection_name = queue_store.get("collection_name", "jobs")

        client = MongoClient(host, port)
        db = client[database]
        collection = db[collection_name]

        # Create backup if requested
        if create_backup:
            import datetime

            backup_collection = db[f"{collection_name}_backup"]

            # Create backup document with timestamp to avoid _id conflicts
            backup_doc = job_doc.copy()
            backup_doc["_backup_timestamp"] = datetime.datetime.utcnow()
            backup_doc["_original_id"] = job_doc["_id"]

            # Remove original _id to avoid duplicate key error
            if "_id" in backup_doc:
                del backup_doc["_id"]

            backup_collection.insert_one(backup_doc)
            console.print(
                f"[green]✓[/green] Backup created in collection '{collection_name}_backup'"
            )

        # Update job document with new parameters
        # Based on inspection, the path is: job.function.@bound.input_set_generator.user_params

        query = {}
        if job_id.isdigit():
            # jobflow-remote stores db_id as STRING, not int
            query = {"db_id": str(job_id)}
        else:
            query = {"uuid": job_id}

        # Filter parameters: separate FDF params from atomate2siesta internal params
        # Internal params should NOT go in user_params (they cause deserialization errors)
        internal_params = {
            "tier",
            "xc",
            "mesh_cutoff",
            "kpts",
            "kgrid_cutoff",
            "fdf_arguments",
        }

        # Split params into FDF and internal
        # Note: a2s_ parameters (like a2s_kpts, a2s_magnetic_ordering) are VALID user params!
        # Only skip the truly internal ones in the internal_params set
        fdf_params = {}
        for key, value in new_params.items():
            # Skip internal atomate2siesta parameters
            if key in internal_params:
                console.print(f"[dim]Skipping internal parameter: {key}[/dim]")
                continue
            # Keep all other parameters (including a2s_ prefix - they're valid!)
            fdf_params[key] = value

        # Get current user_params from job document
        current_user_params = {}
        job_data = job_doc.get("job", {})
        function_data = job_data.get("function", {})
        if "@bound" in function_data:
            bound_data = function_data["@bound"]
            input_gen = bound_data.get("input_set_generator", {})
            current_user_params = input_gen.get("user_params", {})

        # DON'T filter out internal params from current user_params!
        # Bug fix: We should preserve existing internal params (kpts, tier, etc.)
        # Only filter them from NEW params being added
        # Merge: new FDF params added to ALL existing params (FDF + internal)
        merged_fdf_params = {**current_user_params, **fdf_params}

        # Remove specified keys AFTER merging
        if remove_keys:
            for key in remove_keys:
                if key in merged_fdf_params:
                    del merged_fdf_params[key]
                    console.print(f"[dim]Removed key from final params: {key}[/dim]")

        if not merged_fdf_params and not remove_keys:
            console.print(
                "[yellow]Warning:[/yellow] No FDF parameters to update (only internal params)"
            )
            return False

        console.print(f"[dim]Updating {len(merged_fdf_params)} FDF parameters[/dim]")

        # The correct path for jobflow-remote job documents
        # job -> function -> @bound -> input_set_generator -> user_params
        update_path = "job.function.@bound.input_set_generator.user_params"

        # Update with merged FDF params (no internal params)
        result = collection.update_one(
            query,
            {"$set": {update_path: merged_fdf_params}},
        )

        if result.modified_count > 0:
            console.print(
                f"[green]✓[/green] Modified {result.modified_count} document(s)"
            )
            return True
        else:
            console.print("[yellow]Warning:[/yellow] No documents modified")
            console.print(f"[yellow]Query:[/yellow] {query}")
            console.print(f"[yellow]Update path:[/yellow] {update_path}")

            # Try to find the job to debug
            found_job = collection.find_one(query)
            if not found_job:
                console.print("[red]Job not found with query![/red]")
            else:
                console.print(
                    "[yellow]Job found but update failed - checking structure...[/yellow]"
                )
                # Check if the path exists
                job_data = found_job.get("job", {})
                function_data = job_data.get("function", {})
                if "@bound" in function_data:
                    bound_data = function_data["@bound"]
                    if "input_set_generator" in bound_data:
                        console.print("[green]✓[/green] Path exists in job document")
                    else:
                        console.print("[red]✗[/red] input_set_generator not in @bound")
                else:
                    console.print("[red]✗[/red] @bound not in function")

            return False

    except Exception as e:
        console.print(f"[red]Database error:[/red] {e}")
        return False
