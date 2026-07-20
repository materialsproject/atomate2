"""MongoDB query utilities for jobflow-remote job inspection.

This module provides utilities to query and inspect jobs stored in the
jobflow-remote MongoDB database.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

if TYPE_CHECKING:
    from collections import OrderedDict

console = Console()


def get_project_config(project_name: str) -> dict[str, Any] | None:
    """Load jobflow-remote project configuration.

    Args:
        project_name: Name of the jobflow-remote project

    Returns
    -------
        Configuration dictionary or None if not found
    """
    config_dir = Path.home() / ".jfremote"
    config_file = config_dir / f"{project_name}.yaml"

    if not config_file.exists():
        console.print(
            f"[red]Error:[/red] Configuration file not found: {config_file}",
            style="bold",
        )
        console.print(
            "\n[yellow]Hint:[/yellow] Run "
            f"'atomate2siesta-jobflow-remote -p {project_name} setup' first"
        )
        return None

    try:
        with open(config_file) as f:
            return yaml.safe_load(f)
    except Exception as e:  # noqa: BLE001 friendly CLI error handler
        console.print(f"[red]Error loading config:[/red] {e}")
        return None


def get_job_info_from_jf(project_name: str, job_id: str) -> dict[str, Any] | None:
    """Get job information using jf CLI.

    Args:
        project_name: Name of the jobflow-remote project
        job_id: Job ID (UUID or index)

    Returns
    -------
        Job information dictionary or None if not found
    """
    try:
        # Use jf job info command to get job details
        # jf: jobflow-remote CLI, intentionally invoked by name from PATH
        result = subprocess.run(
            ["jf", "-p", project_name, "job", "info", job_id, "-v"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse the output (jf returns YAML-like output)
        # We'll extract key information from the text output
        output_lines = result.stdout.strip().split("\n")

        job_info = {
            "db_id": job_id,
            "name": None,
            "state": None,
            "worker": None,
            "index": None,
            "uuid": None,
        }

        for raw_line in output_lines:
            line = raw_line.strip()
            if line.startswith("Name:"):
                job_info["name"] = line.split("Name:", 1)[1].strip()
            elif line.startswith("State:"):
                job_info["state"] = line.split("State:", 1)[1].strip()
            elif line.startswith("Worker:"):
                job_info["worker"] = line.split("Worker:", 1)[1].strip()
            elif line.startswith("Index:"):
                job_info["index"] = line.split("Index:", 1)[1].strip()
            elif line.startswith("UUID:"):
                job_info["uuid"] = line.split("UUID:", 1)[1].strip()

        return job_info  # noqa: TRY300 direct return after parsing loop

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error getting job info:[/red] {e.stderr}")
        return None
    except FileNotFoundError:
        console.print(
            "[red]Error:[/red] 'jf' command not found. Is jobflow-remote installed?"
        )
        return None


def get_job_details_from_db(project_name: str, job_id: str) -> dict[str, Any] | None:
    """Get complete job details from MongoDB (requires pymongo).

    This function directly queries MongoDB to get ALL job details including
    input parameters, function arguments, and metadata.

    Args:
        project_name: Name of the jobflow-remote project
        job_id: Job ID (UUID or index)

    Returns
    -------
        Complete job document or None if not found
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        console.print(
            "[yellow]Warning:[/yellow] pymongo not installed. "
            "Install with: pip install pymongo"
        )
        return None

    # Load config to get MongoDB connection details
    config = get_project_config(project_name)
    if not config:
        return None

    try:
        # Get MongoDB connection details from queue store
        queue_store = config.get("queue", {}).get("store", {})
        host = queue_store.get("host", "localhost")
        port = queue_store.get("port", 27017)
        database = queue_store.get("database", "jobflow_remote")
        collection_name = queue_store.get("collection_name", "jobs")
        username = queue_store.get("username")
        password = queue_store.get("password")

        # Connect to MongoDB (with authentication if provided)
        client: MongoClient
        if username and password:
            client = MongoClient(
                host, port, username=username, password=password, authSource=database
            )
        else:
            client = MongoClient(host, port)

        db = client[database]
        collection = db[collection_name]

        # Try multiple query strategies for finding the job
        # jobflow-remote uses different field names than we initially assumed
        job_doc = None

        if job_id.isdigit():
            # Try different possible field names and types for the numeric ID
            # jobflow-remote stores db_id as STRING, index as INT
            for field_name, value in [
                ("db_id", str(job_id)),  # String field
                ("index", int(job_id)),  # Integer field
                ("db_id", int(job_id)),  # Try int for db_id too
            ]:
                try:
                    query = {field_name: value}
                    job_doc = collection.find_one(query)
                    if job_doc:
                        break
                except Exception:  # noqa: S112, BLE001 try next query strategy
                    continue

            # Also try _id as ObjectId if nothing found
            if not job_doc:
                try:
                    from bson import ObjectId

                    query = {"_id": ObjectId(job_id)}
                    job_doc = collection.find_one(query)
                except Exception:  # noqa: S110, BLE001 optional ObjectId lookup
                    pass
        else:
            # Try UUID search
            query = {"uuid": job_id}
            job_doc = collection.find_one(query)

        if not job_doc:
            console.print(f"[red]Error:[/red] Job {job_id} not found in database")
            console.print("[yellow]Debug info:[/yellow]")
            console.print(f"  Database: {database}")
            console.print(f"  Collection: {collection_name}")
            console.print(f"  Host: {host}:{port}")

            # Show sample document structure to help debug
            sample = collection.find_one()
            if sample:
                console.print("\n[yellow]Sample document fields:[/yellow]")
                for key in list(sample.keys())[:10]:
                    console.print(f"  - {key}: {type(sample[key]).__name__}")

            return None

        return job_doc  # noqa: TRY300 direct return after query strategies

    except Exception as e:  # noqa: BLE001 friendly CLI error handler
        console.print(f"[red]Error querying database:[/red] {e}")
        return None


def extract_fdf_parameters(job_doc: dict[str, Any]) -> dict[str, Any] | None:
    """Extract SIESTA FDF parameters from job document.

    Args:
        job_doc: Complete job document from MongoDB

    Returns
    -------
        Dictionary of FDF parameters or None if not found
    """
    try:
        # Navigate job document structure to find input parameters
        # jobflow-remote stores the maker in the function's @bound object

        job_data = job_doc.get("job", {})
        fdf_params = {}

        # Method 1: Check function_kwargs (for jobs created directly)
        function_kwargs = job_data.get("function_kwargs", {})

        if "user_params" in function_kwargs:
            fdf_params.update(function_kwargs["user_params"])

        if "maker" in function_kwargs:
            maker = function_kwargs["maker"]
            if isinstance(maker, dict):
                input_gen = maker.get("input_set_generator", {})
                if "user_params" in input_gen:
                    fdf_params.update(input_gen["user_params"])

        # Method 2: Check function @bound (jobflow-remote typical structure)
        function_def = job_data.get("function", {})
        if isinstance(function_def, dict) and "@bound" in function_def:
            bound_maker = function_def["@bound"]

            # Extract from bound maker's input_set_generator
            if "input_set_generator" in bound_maker:
                input_gen = bound_maker["input_set_generator"]
                if isinstance(input_gen, dict) and "user_params" in input_gen:
                    fdf_params.update(input_gen["user_params"])

                # Also get fdf_arguments if present
                if isinstance(input_gen, dict) and "fdf_arguments" in input_gen:
                    fdf_args = input_gen["fdf_arguments"]
                    if fdf_args:  # Only add if not empty
                        fdf_params.update(fdf_args)

                # Get other important parameters
                for key in ["kpts", "mesh_cutoff", "xc", "tier"]:
                    # Skip if already in user_params with a2s_ prefix
                    if (
                        input_gen.get(key)
                        and f"a2s_{key}" not in fdf_params
                        and key not in fdf_params
                    ):
                        fdf_params[key] = input_gen[key]

        # Method 3: Check stored_data (if parameters were stored separately)
        if job_doc.get("stored_data"):
            stored = job_doc["stored_data"]
            if isinstance(stored, dict):
                if "user_params" in stored:
                    fdf_params.update(stored["user_params"])
                if "fdf_arguments" in stored:
                    fdf_params.update(stored["fdf_arguments"])

        return fdf_params or None  # noqa: TRY300 direct return after extraction

    except Exception as e:  # noqa: BLE001 friendly CLI error handler
        console.print(
            f"[yellow]Warning:[/yellow] Could not extract FDF parameters: {e}"
        )
        import traceback

        traceback.print_exc()
        return None


def get_tier_defaults(tier: str) -> dict[str, Any]:
    """Get default parameters that a tier preset contributes.

    Args:
        tier: Tier level (basic_dirty, basic, intermediate, advanced, expert)

    Returns
    -------
        Dictionary of default parameters from dataclass modules
    """
    try:
        import io
        import sys

        from pymatgen.core import Lattice, Structure

        from atomate2.siesta.sets.base import SiestaInputGenerator

        # Suppress validation output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            # Create a dummy structure for parameter generation
            dummy_structure = Structure(Lattice.cubic(5.0), ["Si"], [[0, 0, 0]])

            # Create generator with specified tier (no user params)
            generator = SiestaInputGenerator(
                tier=tier, user_params=cast("OrderedDict[str, Any]", {})
            )

            # Generate parameters (this activates all modules)
            input_set = generator.get_input_set(dummy_structure)

            # Get the Siesta object from input_set.inputs
            siesta_calc = input_set.inputs.get("siesta.fdf")

            if siesta_calc:
                # The Siesta calculator object has parameters attribute
                return dict(siesta_calc.parameters)  # type: ignore[union-attr]  # dynamic InputFile payload
            return {}

        finally:
            sys.stdout = old_stdout

    except Exception as e:  # noqa: BLE001 friendly CLI error handler
        import traceback

        console.print(f"[yellow]Warning:[/yellow] Could not get tier defaults: {e}")
        console.print("[dim]" + traceback.format_exc() + "[/dim]")
        return {}


def get_actual_fdf_file(project_name: str, job_doc: dict[str, Any]) -> str | None:
    """Get the actual generated siesta.fdf file from job run directory.

    Tries multiple methods:
    1. Direct file access (if filesystem is mounted)
    2. SSH access (if worker is remote)
    3. jobflow-remote download (via jf CLI)

    Args:
        project_name: Jobflow-remote project name
        job_doc: Job document from MongoDB

    Returns
    -------
        Content of siesta.fdf file or None if not found
    """
    try:
        # Get run directory from job document
        run_dir = job_doc.get("run_dir")

        if not run_dir:
            console.print("[yellow]Warning:[/yellow] No run_dir found in job document")
            return None

        # Import Path
        from pathlib import Path

        # Method 1: Try direct file access (local or mounted filesystem)
        fdf_path = Path(run_dir) / "siesta.fdf"

        if fdf_path.exists():
            return fdf_path.read_text()

        # Method 2: Try SSH access if worker is remote
        console.print("[dim]Local file not accessible, trying remote access...[/dim]")

        # Get worker config to determine if remote
        config = get_project_config(project_name)
        if config:
            worker_name = job_doc.get("worker")
            if worker_name:
                workers = config.get("workers", {})
                worker_config = workers.get(worker_name, {})

                if worker_config.get("type") == "remote":
                    # Try SSH access
                    host = worker_config.get("host")
                    user = worker_config.get("user")

                    if host and user:
                        content = _fetch_via_ssh(host, user, str(fdf_path))
                        if content:
                            return content

        # Method 3: Try using jf download
        console.print("[dim]Trying jobflow-remote download...[/dim]")
        content = _fetch_via_jf_download(project_name, job_doc)
        if content:
            return content

        # All methods failed
        console.print(f"[yellow]Could not access FDF file at:[/yellow] {fdf_path}")
        console.print(
            "[dim]File may be on remote system not accessible via SSH "
            "or jobflow-remote[/dim]"
        )
        return None  # noqa: TRY300 direct return after fallback attempts

    except Exception as e:  # noqa: BLE001 friendly CLI error handler
        console.print(f"[yellow]Warning:[/yellow] Could not read FDF file: {e}")
        return None


def _fetch_via_ssh(host: str, user: str, remote_path: str) -> str | None:
    """Fetch file via SSH.

    Args:
        host: Remote hostname
        user: Username
        remote_path: Path to file on remote system

    Returns
    -------
        File content or None if failed
    """
    try:
        import subprocess

        # Use ssh to cat the file
        ssh_command = f"ssh {user}@{host} cat {remote_path}"

        result = subprocess.run(
            ssh_command.split(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if result.returncode == 0:
            console.print("[green]✓[/green] Retrieved via SSH")
            return result.stdout
        console.print(f"[dim]SSH access failed: {result.stderr[:100]}[/dim]")
        return None  # noqa: TRY300 direct return on non-zero exit

    except Exception as e:  # noqa: BLE001 friendly CLI error handler
        console.print(f"[dim]SSH method failed: {e}[/dim]")
        return None


def _fetch_via_jf_download(project_name: str, job_doc: dict[str, Any]) -> str | None:
    """Fetch file using jobflow-remote download mechanism.

    Args:
        project_name: Project name
        job_doc: Job document

    Returns
    -------
        File content or None if failed
    """
    try:
        import subprocess
        import tempfile
        from pathlib import Path

        # Get job ID
        db_id = job_doc.get("db_id")
        if not db_id:
            return None

        # Create temp directory for download
        with tempfile.TemporaryDirectory() as tmpdir:
            # Try to download specific file using jf
            cmd = [
                "jf",
                "-p",
                project_name,
                "job",
                "files",
                str(db_id),
                "--file",
                "siesta.fdf",
                "--output-dir",
                tmpdir,
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=False
            )

            if result.returncode == 0:
                # Check if file was downloaded
                downloaded_file = Path(tmpdir) / "siesta.fdf"
                if downloaded_file.exists():
                    console.print("[green]✓[/green] Retrieved via jobflow-remote")
                    return downloaded_file.read_text()
            else:
                console.print(f"[dim]jf download failed: {result.stderr[:100]}[/dim]")

        return None  # noqa: TRY300 direct return after download attempt

    except Exception as e:  # noqa: BLE001 friendly CLI error handler
        console.print(f"[dim]jobflow-remote download failed: {e}[/dim]")
        return None


def extract_job_resources(job_doc: dict[str, Any]) -> dict[str, Any] | None:
    """Extract SLURM/PBS resources from job document.

    jobflow-remote stores resources at the top-level ``resources`` field,
    which is what the runner actually reads when submitting to SLURM/PBS.

    Args:
        job_doc: Complete job document from MongoDB

    Returns
    -------
        Dictionary of resources or None if not found
    """
    try:
        # Primary: top-level resources (what jobflow-remote actually uses)
        resources = job_doc.get("resources")
        if resources and isinstance(resources, dict):
            return dict(resources)
        # Fallback: nested path (initial job definition)
        resources = (
            job_doc.get("job", {})
            .get("config", {})
            .get("manager_config", {})
            .get("resources", None)
        )
        if resources and isinstance(resources, dict):
            return dict(resources)
        return None  # noqa: TRY300 direct return after resource lookup
    except Exception:  # noqa: BLE001 resources are optional
        return None


def extract_job_state(job_doc: dict[str, Any]) -> str | None:
    """Extract job state from job document.

    Args:
        job_doc: Complete job document from MongoDB

    Returns
    -------
        Job state string (e.g., READY, WAITING, RUNNING) or None
    """
    return job_doc.get("state")


def display_job_info(job_info: dict[str, Any], include_fdf: bool = False) -> None:
    """Display job information in a formatted table.

    Args:
        job_info: Job information dictionary
        include_fdf: If True, display FDF parameters if available
    """
    # Create basic info table
    table = Table(title="Job Information", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="green")
    table.add_column("Value", style="white")

    for key, value in job_info.items():
        if key != "fdf_params" and value is not None:
            table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)

    # Display FDF parameters if available
    if include_fdf and "fdf_params" in job_info:
        fdf_params = job_info["fdf_params"]
        if fdf_params:
            console.print("\n")
            fdf_text = yaml.dump(fdf_params, default_flow_style=False, sort_keys=False)
            syntax = Syntax(fdf_text, "yaml", theme="monokai", line_numbers=True)
            console.print(
                Panel(
                    syntax,
                    title="[bold yellow]SIESTA FDF Parameters[/bold yellow]",
                    border_style="yellow",
                )
            )
        else:
            console.print(
                "\n[yellow]Note:[/yellow] No FDF parameters found in job document"
            )
