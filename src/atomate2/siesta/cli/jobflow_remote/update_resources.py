"""Job resource update command for jobflow-remote.

Update SLURM/PBS resources (nodes, cores, memory, walltime, partition, etc.)
for jobs that have not yet started running.
"""

from __future__ import annotations

import datetime
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from .db_query import (
    extract_job_resources,
    extract_job_state,
    get_job_details_from_db,
    get_project_config,
)

console = Console()

# Job states that allow resource modification
MODIFIABLE_STATES = {"READY", "WAITING", "PAUSED", "REMOTE_ERROR", "FAILED"}


@click.command("update-resources")
@click.argument("job_id")
@click.option("--nodes", type=int, help="Number of nodes")
@click.option("--ntasks-per-node", type=int, help="Tasks (cores) per node")
@click.option("--cpus-per-task", type=int, help="CPUs per task (for threaded jobs)")
@click.option("--mem-per-cpu", type=str, help='Memory per CPU (e.g., "4G", "2000M")')
@click.option("--time", "walltime", type=str, help="Walltime in HH:MM:SS format")
@click.option("--partition", type=str, help="SLURM partition name")
@click.option("--account", type=str, help="SLURM account name")
@click.option("--qos", type=str, help="Quality of Service")
@click.option("--gres", type=str, help='Generic resources (e.g., "gpu:2")')
@click.option(
    "--auto",
    "auto_allocate",
    is_flag=True,
    help="Auto-allocate resources based on atom count heuristic",
)
@click.option(
    "--cluster-profile",
    type=str,
    help='Named cluster profile for --auto (e.g., "mn5", "agustina")',
)
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.option(
    "--backup/--no-backup",
    default=True,
    help="Create backup before modification (default: True)",
)
@click.pass_context
def update_resources(
    ctx,
    job_id: str,
    nodes: int | None,
    ntasks_per_node: int | None,
    cpus_per_task: int | None,
    mem_per_cpu: str | None,
    walltime: str | None,
    partition: str | None,
    account: str | None,
    qos: str | None,
    gres: str | None,
    auto_allocate: bool,
    cluster_profile: str | None,
    force: bool,
    backup: bool,
):
    """Update SLURM/PBS resources for a pending job.

    Modifies the job's resource allocation (nodes, cores, memory, walltime,
    partition, etc.) directly in MongoDB. Only works for jobs in READY,
    WAITING, or PAUSED state.

    \b
    Examples:
        # Update walltime and partition
        atomate2siesta-jobflow-remote -p prod job update-resources 70 \\
            --time "24:00:00" --partition "large"

        # Scale up to 2 nodes with 48 cores each
        atomate2siesta-jobflow-remote -p prod job update-resources 70 \\
            --nodes 2 --ntasks-per-node 48 --mem-per-cpu "4G"

        # Auto-allocate based on atom count
        atomate2siesta-jobflow-remote -p prod job update-resources 70 \\
            --auto --cluster-profile mn5

        # Force mode (skip confirmation)
        atomate2siesta-jobflow-remote -p prod job update-resources 70 \\
            --nodes 4 --time "48:00:00" --force

    Args:
        job_id: Job index number or UUID
    """
    project_name = ctx.obj.get("project_name", "atomate2siesta")

    # Check that at least one resource option is specified
    manual_opts = [
        nodes,
        ntasks_per_node,
        cpus_per_task,
        mem_per_cpu,
        walltime,
        partition,
        account,
        qos,
        gres,
    ]
    if not auto_allocate and all(opt is None for opt in manual_opts):
        console.print(
            "[red]Error:[/red] No resource options specified. "
            "Use --help to see available options."
        )
        console.print(
            f"\n[yellow]Example:[/yellow] atomate2siesta-jobflow-remote -p {project_name} "
            f'job update-resources {job_id} --time "24:00:00" --nodes 2'
        )
        raise click.Abort()

    # Fetch job from database
    console.print("[bold]Fetching job details...[/bold]")
    job_doc = get_job_details_from_db(project_name, job_id)
    if not job_doc:
        console.print(
            "[red]Error:[/red] Could not retrieve job from database. "
            "Is pymongo installed?"
        )
        raise click.Abort()

    # Check job state
    state = extract_job_state(job_doc)
    if state and state not in MODIFIABLE_STATES:
        console.print(
            f"[red]Error:[/red] Job {job_id} is in state [bold]{state}[/bold]. "
            f"Resources can only be updated for jobs in: {', '.join(sorted(MODIFIABLE_STATES))}"
        )
        raise click.Abort()

    console.print(
        f"[green]>[/green] Job {job_id} — state: [bold]{state or 'UNKNOWN'}[/bold]"
    )

    # Extract current resources
    current_resources = extract_job_resources(job_doc) or {}
    console.print(
        f"[green]>[/green] Current resources: {len(current_resources)} field(s)\n"
    )

    # Build new resources
    if auto_allocate:
        new_resources = _build_auto_resources(job_doc, cluster_profile)
        if new_resources is None:
            raise click.Abort()
    else:
        new_resources = {}

    # Apply manual overrides (these take precedence over --auto)
    _apply_manual_overrides(
        new_resources,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        cpus_per_task=cpus_per_task,
        mem_per_cpu=mem_per_cpu,
        walltime=walltime,
        partition=partition,
        account=account,
        qos=qos,
        gres=gres,
    )

    # Merge: current resources + new overrides
    merged_resources = {**current_resources, **new_resources}

    # Preview changes
    _display_resource_comparison(current_resources, merged_resources)
    console.print()

    # Confirm
    if not force:
        if not Confirm.ask("[bold]Apply these resource changes?[/bold]", default=True):
            console.print("[yellow]Operation cancelled[/yellow]")
            return

    # Perform update
    console.print("\n[bold]Updating resources in database...[/bold]")
    success = _update_resources_in_db(
        project_name=project_name,
        job_id=job_id,
        job_doc=job_doc,
        new_resources=merged_resources,
        create_backup=backup,
    )

    if success:
        console.print()
        console.print(
            Panel.fit(
                f"[bold green]Resources updated for job {job_id}[/bold green]\n\n"
                "[bold]Next steps:[/bold]\n"
                f"  - Inspect: [cyan]atomate2siesta-jobflow-remote -p {project_name} job inspect {job_id} --full[/cyan]\n"
                f"  - Rerun:   [cyan]jf -p {project_name} job rerun {job_id}[/cyan]",
                border_style="green",
            )
        )
    else:
        console.print("[red]Failed to update resources[/red]")
        raise click.Abort()


def _build_auto_resources(
    job_doc: dict[str, Any], profile_name: str | None
) -> dict[str, Any] | None:
    """Build resources using auto-allocation heuristic.

    Args:
        job_doc: Job document from MongoDB
        profile_name: Optional cluster profile name

    Returns:
        Resource dict or None on failure
    """
    try:
        from atomate2.siesta.powerups import (
            _estimate_resources,
            _estimate_resources_heuristic,
        )
    except ImportError:
        console.print(
            "[red]Error:[/red] Could not import resource estimation functions"
        )
        return None

    # Extract atom count from job document
    num_atoms = _extract_atom_count_from_doc(job_doc)
    if num_atoms is None:
        console.print(
            "[red]Error:[/red] Could not determine atom count from job document. "
            "--auto requires a structure in the job."
        )
        return None

    console.print(f"[green]>[/green] Detected {num_atoms} atoms")

    # Get cluster profile if specified
    profile = None
    if profile_name:
        try:
            from atomate2.siesta.cluster_profiles import ClusterProfile

            predefined = {p.name: p for p in ClusterProfile.list_predefined()}
            if profile_name in predefined:
                profile = predefined[profile_name]
                console.print(
                    f"[green]>[/green] Using cluster profile: {profile.summary()}"
                )
            else:
                console.print(
                    f"[yellow]Warning:[/yellow] Unknown profile '{profile_name}'. "
                    f"Available: {', '.join(predefined.keys())}"
                )
                console.print("[dim]Falling back to heuristic without profile[/dim]")
        except ImportError:
            console.print(
                "[yellow]Warning:[/yellow] cluster_profiles not available, "
                "using heuristic only"
            )

    if profile:
        resources = _estimate_resources(num_atoms, profile=profile)
    else:
        resources = _estimate_resources_heuristic(num_atoms)

    console.print(f"[green]>[/green] Auto-allocated resources for {num_atoms} atoms\n")
    return resources


def _extract_atom_count_from_doc(job_doc: dict[str, Any]) -> int | None:
    """Extract atom count from a MongoDB job document.

    Searches the job's function arguments for a structure with num_sites.

    Args:
        job_doc: Job document from MongoDB

    Returns:
        Number of atoms or None if not found
    """
    try:
        job_data = job_doc.get("job", {})

        # Check function_kwargs for structure-like objects
        function_kwargs = job_data.get("function_kwargs", {})
        for _key, value in function_kwargs.items():
            if isinstance(value, dict):
                # pymatgen Structure serialized as dict
                if "lattice" in value and "sites" in value:
                    sites = value.get("sites", [])
                    return len(sites)
                # Check for num_sites directly
                if "num_sites" in value:
                    return value["num_sites"]

        # Check function_args (positional)
        function_args = job_data.get("function_args", [])
        for arg in function_args:
            if isinstance(arg, dict):
                if "lattice" in arg and "sites" in arg:
                    return len(arg.get("sites", []))
                if "num_sites" in arg:
                    return arg["num_sites"]

        # Check @bound maker for prev_dir pattern (structure may be in output ref)
        # In this case we can't get atom count — return None
        return None

    except Exception:
        return None


def _apply_manual_overrides(
    resources: dict[str, Any],
    *,
    nodes: int | None,
    ntasks_per_node: int | None,
    cpus_per_task: int | None,
    mem_per_cpu: str | None,
    walltime: str | None,
    partition: str | None,
    account: str | None,
    qos: str | None,
    gres: str | None,
) -> None:
    """Apply manual CLI overrides to resource dict (in-place)."""
    if nodes is not None:
        resources["nodes"] = nodes
    if ntasks_per_node is not None:
        resources["ntasks_per_node"] = ntasks_per_node
    if cpus_per_task is not None:
        resources["cpus_per_task"] = cpus_per_task
    if mem_per_cpu is not None:
        resources["mem_per_cpu"] = mem_per_cpu
    if walltime is not None:
        resources["time"] = walltime
    if partition is not None:
        resources["partition"] = partition
    if account is not None:
        resources["account"] = account
    if qos is not None:
        resources["qos"] = qos
    if gres is not None:
        resources["gres"] = gres


def _display_resource_comparison(current: dict[str, Any], new: dict[str, Any]) -> None:
    """Display a Rich table comparing current vs new resources."""
    all_keys = sorted(set(list(current.keys()) + list(new.keys())))

    table = Table(
        title="Resource Changes",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Resource", style="green")
    table.add_column("Current", style="white")
    table.add_column("New", style="white")
    table.add_column("Status", style="white")

    for key in all_keys:
        cur_val = current.get(key)
        new_val = new.get(key)

        cur_str = str(cur_val) if cur_val is not None else "-"
        new_str = str(new_val) if new_val is not None else "-"

        if cur_val == new_val:
            status = "[dim]unchanged[/dim]"
        elif cur_val is None:
            status = "[cyan]added[/cyan]"
        elif new_val is None:
            status = "[yellow]removed[/yellow]"
        else:
            status = "[bold yellow]changed[/bold yellow]"

        table.add_row(key, cur_str, new_str, status)

    console.print(table)


def _update_resources_in_db(
    project_name: str,
    job_id: str,
    job_doc: dict[str, Any],
    new_resources: dict[str, Any],
    create_backup: bool = True,
) -> bool:
    """Update job resources directly in MongoDB.

    Args:
        project_name: Project name
        job_id: Job ID
        job_doc: Original job document
        new_resources: New resources dict to set
        create_backup: Whether to create backup

    Returns:
        True if successful
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        console.print("[red]Error:[/red] pymongo not installed")
        return False

    config = get_project_config(project_name)
    if not config:
        return False

    try:
        queue_store = config.get("queue", {}).get("store", {})
        host = queue_store.get("host", "localhost")
        port = queue_store.get("port", 27017)
        database = queue_store.get("database", "jobflow_remote")
        collection_name = queue_store.get("collection_name", "jobs")
        username = queue_store.get("username")
        password = queue_store.get("password")

        if username and password:
            client = MongoClient(
                host, port, username=username, password=password, authSource=database
            )
        else:
            client = MongoClient(host, port)

        db = client[database]
        collection = db[collection_name]

        # Create backup
        if create_backup:
            backup_collection = db[f"{collection_name}_backup"]
            backup_doc = job_doc.copy()
            backup_doc["_backup_timestamp"] = datetime.datetime.utcnow()
            backup_doc["_backup_type"] = "update_resources"
            backup_doc["_original_id"] = job_doc["_id"]
            if "_id" in backup_doc:
                del backup_doc["_id"]
            backup_collection.insert_one(backup_doc)
            console.print(
                f"[green]>[/green] Backup created in '{collection_name}_backup'"
            )

        # Build query
        if job_id.isdigit():
            query = {"db_id": str(job_id)}
        else:
            query = {"uuid": job_id}

        # Update resources at BOTH paths:
        # 1. Top-level "resources" — what jobflow-remote runner actually reads
        # 2. "job.config.manager_config.resources" — the job definition copy
        result = collection.update_one(
            query,
            {
                "$set": {
                    "resources": new_resources,
                    "job.config.manager_config.resources": new_resources,
                }
            },
        )

        if result.modified_count > 0:
            console.print(
                f"[green]>[/green] Modified {result.modified_count} document(s)"
            )
            return True
        else:
            console.print("[yellow]Warning:[/yellow] No documents modified")
            found = collection.find_one(query)
            if not found:
                console.print(f"[red]Job not found with query: {query}[/red]")
            else:
                console.print(
                    "[yellow]Note:[/yellow] Resources may already match the target values"
                )
            return False

    except Exception as e:
        console.print(f"[red]Database error:[/red] {e}")
        return False
