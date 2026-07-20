"""Setup and configuration commands for jobflow-remote."""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .utils import _backup_config, _load_yaml_config, _save_yaml_config

console = Console()


@click.command()
@click.option(
    "--dev",
    is_flag=True,
    help="Install development version from GitHub",
)
def install(dev: bool) -> None:
    """Install jobflow-remote package.

    This command installs jobflow-remote using pip. By default, it installs
    the stable version from PyPI. Use --dev to install the latest development
    version from GitHub.

    Examples
    --------
        # Install stable version
        atomate2siesta-jobflow-remote install

        # Install development version
        atomate2siesta-jobflow-remote install --dev
    """
    console.print("\n[bold cyan]Installing jobflow-remote...[/bold cyan]\n")

    # Determine installation command
    if dev:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/Matgenix/jobflow-remote.git",
        ]
        console.print("[yellow]Installing development version from GitHub[/yellow]")
    else:
        cmd = [sys.executable, "-m", "pip", "install", "jobflow-remote"]
        console.print("[yellow]Installing stable version from PyPI[/yellow]")

    # Show installation location
    import site

    site_packages = site.getsitepackages()[0] if site.getsitepackages() else "unknown"
    console.print(f"\n[dim]Installing to: {site_packages}[/dim]\n")

    # Run installation
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        console.print("\n[bold green]✓ Installation successful![/bold green]")
        console.print(f"[dim]Location: {site_packages}/jobflow_remote/[/dim]\n")

        # Show next steps
        next_steps = Panel(
            "[bold]Next Steps:[/bold]\n\n"
            "1. Check installation:\n"
            "   [cyan]jf --version[/cyan]\n\n"
            "2. Generate project configuration:\n"
            "   [cyan]atomate2siesta-jobflow-remote setup[/cyan]\n\n"
            "3. Check configuration:\n"
            "   [cyan]jf project check --errors[/cyan]\n\n"
            "4. Initialize database:\n"
            "   [cyan]jf admin reset[/cyan]\n\n"
            "Documentation: [blue]https://matgenix.github.io/jobflow-remote/[/blue]",
            title="Installation Complete",
            style="green",
        )
        console.print(next_steps)

    except subprocess.CalledProcessError as e:
        console.print("\n[bold red]✗ Installation failed![/bold red]")
        console.print(f"\n[red]Error: {e.stderr}[/red]")
        sys.exit(1)


@click.command()
@click.option(
    "--project-name",
    default="atomate2siesta",
    help="Name for the jobflow-remote project",
)
@click.option(
    "--worker-name",
    default="local_shell",
    help="Name for the worker",
)
@click.option(
    "--database",
    default="atomate2siesta",
    help="MongoDB database name",
)
@click.option(
    "--host",
    default="localhost",
    help="MongoDB host",
)
@click.option(
    "--port",
    default=27017,
    type=int,
    help="MongoDB port",
)
@click.option(
    "--update",
    is_flag=True,
    help="Update existing configuration file instead of generating new one",
)
@click.option(
    "--backup/--no-backup",
    default=True,
    help="Create backup before updating (default: True)",
)
def setup(
    project_name: str,
    worker_name: str,
    database: str,
    host: str,
    port: int,
    update: bool,
    backup: bool,
) -> None:
    """Generate or update jobflow-remote project configuration.

    This is a convenience wrapper around the jobflow-remote command:
        jf project generate <project_name>

    It creates a jobflow-remote project configuration file at
    ~/.jfremote/<project_name>.yaml with settings pre-configured for
    atomate2siesta workflows.

    The configuration includes:
    - Worker setup (local shell or remote HPC)
    - Queue store connection (MongoDB)
    - Job store configuration

    Note: You can also use the original jobflow-remote command directly:
        jf project generate atomate2siesta

    Examples
    --------
        # Generate default configuration
        atomate2siesta-jobflow-remote setup

        # Update existing configuration with new MongoDB settings
        atomate2siesta-jobflow-remote setup --update --host newhost --port 27018

        # Update without creating backup
        atomate2siesta-jobflow-remote setup --update --no-backup --database mydb

        # Custom project name and worker
        atomate2siesta-jobflow-remote setup --project-name my_project \
            --worker-name hpc_worker
    """
    # Determine config file path
    config_path = Path.home() / ".jfremote" / f"{project_name}.yaml"

    # Handle update mode
    if update:
        if not config_path.exists():
            console.print(
                f"[bold red]✗ Configuration file not found: {config_path}[/bold red]\n"
            )
            console.print(
                "Generate it first with: "
                "[cyan]atomate2siesta-jobflow-remote setup[/cyan]"
            )
            sys.exit(1)

        console.print(
            "\n[bold cyan]Updating jobflow-remote configuration...[/bold cyan]\n"
        )

        # Create backup if requested
        if backup:
            backup_path = _backup_config(config_path)
            console.print(f"[green]✓ Backup created:[/green] {backup_path}\n")

        # Load existing configuration
        try:
            config = _load_yaml_config(config_path)
        except Exception as e:  # noqa: BLE001 friendly CLI error reporting
            console.print("[bold red]✗ Failed to load configuration![/bold red]")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

        # Update queue store settings
        if "queue" not in config:
            config["queue"] = {}
        if "store" not in config["queue"]:
            config["queue"]["store"] = {}

        config["queue"]["store"]["database"] = database
        config["queue"]["store"]["host"] = host
        config["queue"]["store"]["port"] = port

        # Update jobstore settings
        if "jobstore" not in config:
            config["jobstore"] = {}
        if "docs_store" not in config["jobstore"]:
            config["jobstore"]["docs_store"] = {}

        config["jobstore"]["docs_store"]["database"] = database
        config["jobstore"]["docs_store"]["host"] = host
        config["jobstore"]["docs_store"]["port"] = port

        # Update additional stores if they exist
        if "additional_stores" in config["jobstore"]:
            for store_config in config["jobstore"]["additional_stores"].values():
                if isinstance(store_config, dict):
                    store_config["database"] = database
                    store_config["host"] = host
                    store_config["port"] = port

        # Save updated configuration
        try:
            _save_yaml_config(config_path, config)
            console.print(
                "[bold green]✓ Configuration updated successfully![/bold green]\n"
            )
        except Exception as e:  # noqa: BLE001 friendly CLI error reporting
            console.print("[bold red]✗ Failed to save configuration![/bold red]")
            console.print(f"[red]Error: {e}[/red]")
            if backup:
                console.print(f"\n[yellow]Restore from backup: {backup_path}[/yellow]")
            sys.exit(1)

        # Show what was updated
        update_panel = Panel(
            f"[bold]Updated Configuration:[/bold] {config_path}\n\n"
            "[bold]MongoDB Settings:[/bold]\n"
            f"  • Database: [cyan]{database}[/cyan]\n"
            f"  • Host: [cyan]{host}[/cyan]\n"
            f"  • Port: [cyan]{port}[/cyan]\n\n"
            "[bold]Updated Sections:[/bold]\n"
            "  • queue.store\n"
            "  • jobstore.docs_store\n"
            "  • jobstore.additional_stores (if present)",
            title="Update Complete",
            style="green",
        )
        console.print(update_panel)

        # Show next steps
        next_steps = Panel(
            "[bold]Next Steps:[/bold]\n\n"
            "1. Verify configuration:\n"
            "   [cyan]jf project check --errors[/cyan]\n\n"
            "2. Reinitialize database if needed:\n"
            "   [cyan]jf admin reset[/cyan]\n\n"
            "3. Restart runner if running:\n"
            "   [cyan]jf runner restart[/cyan]",
            title="Verification",
            style="green",
        )
        console.print(next_steps)
        return

    # Generate new configuration (original behavior)
    console.print(
        "\n[bold cyan]Generating jobflow-remote configuration...[/bold cyan]\n"
    )

    # Check if jobflow-remote is installed
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "show", "jobflow-remote"],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        console.print("[bold red]✗ jobflow-remote is not installed![/bold red]\n")
        console.print(
            "Install it first with: [cyan]atomate2siesta-jobflow-remote install[/cyan]"
        )
        sys.exit(1)

    # Generate project
    console.print(f"[yellow]Generating project: {project_name}[/yellow]")
    console.print(f"[dim]Running: jf project generate {project_name}[/dim]\n")

    try:
        subprocess.run(
            ["jf", "project", "generate", project_name],  # noqa: S607 jf on PATH
            capture_output=True,
            text=True,
            check=True,
        )

        console.print("[green]✓ Project generated successfully![/green]")
        console.print(f"[dim]Equivalent to: jf project generate {project_name}[/dim]\n")

        if config_path.exists():
            # Read and modify the configuration
            console.print(
                f"[yellow]Updating configuration at: {config_path}[/yellow]\n"
            )

            # Show configuration instructions
            config_panel = Panel(
                f"[bold]Configuration File:[/bold] {config_path}\n\n"
                "[bold]Key Settings to Configure:[/bold]\n\n"
                "1. [cyan]workers[/cyan]:\n"
                f"   - Name: {worker_name}\n"
                "   - Type: local (for testing) or remote (for HPC)\n"
                "   - Scheduler: shell, slurm, pbs, etc.\n\n"
                "2. [cyan]queue.store[/cyan]:\n"
                f"   - Database: {database}\n"
                f"   - Host: {host}\n"
                f"   - Port: {port}\n\n"
                "3. [cyan]jobstore[/cyan]:\n"
                f"   - Database: {database}\n"
                f"   - Host: {host}\n"
                f"   - Port: {port}\n\n"
                "[bold]Edit this file manually to customize settings.[/bold]",
                title="Configuration Generated",
                style="green",
            )
            console.print(config_panel)

        # Show next steps
        next_steps = Panel(
            "[bold]Next Steps:[/bold]\n\n"
            f"1. Edit configuration file:\n"
            f"   [cyan]{config_path}[/cyan]\n\n"
            "2. [bold yellow]Install atomate2siesta on remote cluster:[/bold yellow]\n"
            "   SSH to cluster and run:\n"
            "   [cyan]conda activate your_env[/cyan]\n"
            "   [cyan]pip install atomate2siesta[/cyan]\n\n"
            "3. Update with CLI (optional):\n"
            "   [cyan]atomate2siesta-jobflow-remote setup --update[/cyan]\n\n"
            "4. Check configuration:\n"
            "   [cyan]jf project check --errors[/cyan]\n\n"
            "5. Initialize database:\n"
            "   [cyan]jf admin reset[/cyan]\n\n"
            "6. Start runner:\n"
            "   [cyan]jf runner start[/cyan]\n\n"
            "7. Submit test job:\n"
            "   [cyan]atomate2siesta-jobflow-remote test[/cyan]",
            title="Setup Complete",
            style="green",
        )
        console.print(next_steps)

    except subprocess.CalledProcessError as e:
        console.print("\n[bold red]✗ Project generation failed![/bold red]")
        console.print(f"\n[red]Error: {e.stderr}[/red]")
        sys.exit(1)


@click.command()
def test() -> None:
    """Submit a test job to verify jobflow-remote setup.

    This command submits a simple test job to verify that jobflow-remote
    is correctly configured and can submit jobs.

    The test job performs a simple addition operation and stores the result
    in the database.

    Examples
    --------
        # Submit test job
        atomate2siesta-jobflow-remote test
    """
    console.print("\n[bold cyan]Submitting test job...[/bold cyan]\n")

    try:
        # Import required modules
        from jobflow import Flow
        from jobflow_remote import submit_flow
        from jobflow_remote.utils.examples import add

        # Create test jobs
        job1 = add(1, 2)
        job2 = add(job1.output, 3)
        flow = Flow([job1, job2])

        # Submit flow
        console.print("[yellow]Creating test flow: add(1, 2) + 3[/yellow]")
        result = submit_flow(flow, worker="local_shell")

        console.print("\n[bold green]✓ Test job submitted successfully![/bold green]")
        console.print(f"\n[cyan]Job ID: {result}[/cyan]\n")

        # Show next steps
        next_steps = Panel(
            "[bold]Check Job Status:[/bold]\n\n"
            "1. List all jobs:\n"
            "   [cyan]jf job list[/cyan]\n\n"
            "2. Start runner (if not running):\n"
            "   [cyan]jf runner start[/cyan]\n\n"
            "3. Check runner status:\n"
            "   [cyan]jf runner status[/cyan]\n\n"
            f"4. Get job output:\n"
            f"   [cyan]jf job output {result}[/cyan]\n\n"
            "5. View job info:\n"
            f"   [cyan]jf job info {result}[/cyan]",
            title="Test Job Submitted",
            style="green",
        )
        console.print(next_steps)

    except ImportError as e:
        console.print("[bold red]✗ jobflow-remote is not installed![/bold red]\n")
        console.print(f"Error: {e}\n")
        console.print(
            "Install it first with: [cyan]atomate2siesta-jobflow-remote install[/cyan]"
        )
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 friendly CLI error reporting
        console.print("\n[bold red]✗ Test job submission failed![/bold red]")
        console.print(f"\n[red]Error: {e}[/red]\n")
        console.print("Make sure jobflow-remote is properly configured:")
        console.print("  1. [cyan]jf project check --errors[/cyan]")
        console.print("  2. [cyan]jf admin reset[/cyan]")
        sys.exit(1)


@click.command()
@click.option(
    "--project-name",
    help="Show details for a specific project",
)
def info(project_name: str) -> None:
    """Show information about jobflow-remote setup.

    This command displays information about the current jobflow-remote
    installation, configuration, and provides helpful documentation links.

    Examples
    --------
        # Show general information and all projects
        atomate2siesta-jobflow-remote info

        # Show details for a specific project
        atomate2siesta-jobflow-remote info --project-name atomate2siesta
    """
    console.print()

    # Header
    header = Panel(
        Text("Jobflow Remote Setup Helper", style="bold cyan", justify="center"),
        style="cyan",
    )
    console.print(header)

    # Installation status
    console.print("\n[bold]Installation Status:[/bold]\n")

    try:
        # Check using pip show instead of jf --version (which doesn't exist)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "jobflow-remote"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Extract version from pip show output
        for line in result.stdout.split("\n"):
            if line.startswith("Version:"):
                version = line.split(":", 1)[1].strip()
                console.print(f"  ✓ jobflow-remote: [green]{version}[/green]")
                break
    except subprocess.CalledProcessError:
        console.print("  ✗ jobflow-remote: [red]Not installed[/red]")

    # Configuration files
    console.print("\n[bold]Available Projects:[/bold]\n")

    jfremote_dir = Path.home() / ".jfremote"
    if jfremote_dir.exists():
        config_files = list(jfremote_dir.glob("*.yaml"))
        # Filter out backup files
        config_files = [
            f
            for f in config_files
            if not f.stem.startswith(".") and "backup" not in f.stem
        ]

        if config_files:
            # Create table for projects
            projects_table = Table(show_header=True, box=None)
            projects_table.add_column("Project Name", style="cyan")
            projects_table.add_column("Config File", style="dim")
            projects_table.add_column("Workers", style="yellow")

            for config_file in config_files:
                project = config_file.stem
                try:
                    config = _load_yaml_config(config_file)
                    num_workers = len(config.get("workers", {}))
                    worker_names = ", ".join(config.get("workers", {}).keys())
                    projects_table.add_row(
                        project,
                        str(config_file),
                        f"{num_workers} ({worker_names})" if num_workers > 0 else "0",
                    )
                except Exception:  # noqa: BLE001 friendly CLI error reporting
                    projects_table.add_row(
                        project, str(config_file), "[red]Error loading[/red]"
                    )

            console.print(projects_table)
            console.print()

            # Show detailed info for specific project if requested
            if project_name:
                config_path = jfremote_dir / f"{project_name}.yaml"
                if config_path.exists():
                    console.print(f"\n[bold]Project Details: {project_name}[/bold]\n")
                    try:
                        config = _load_yaml_config(config_path)

                        # Workers info
                        if config.get("workers"):
                            console.print("[cyan]Workers:[/cyan]")
                            for worker_name, worker_config in config["workers"].items():
                                console.print(f"  • {worker_name}")
                                if isinstance(worker_config, dict):
                                    wtype = worker_config.get("type", "N/A")
                                    scheduler = worker_config.get(
                                        "scheduler_type", "N/A"
                                    )
                                    console.print(f"    - Type: {wtype}")
                                    console.print(f"    - Scheduler: {scheduler}")
                                    if worker_config.get("host"):
                                        console.print(
                                            f"    - Host: {worker_config.get('host')}"
                                        )

                        # Database info
                        console.print("\n[cyan]Database Configuration:[/cyan]")
                        if "queue" in config and "store" in config["queue"]:
                            store = config["queue"]["store"]
                            console.print("  Queue Store:")
                            console.print(f"    - Host: {store.get('host', 'N/A')}")
                            console.print(
                                f"    - Database: {store.get('database', 'N/A')}"
                            )
                            console.print(f"    - Port: {store.get('port', 27017)}")

                        if "jobstore" in config and "docs_store" in config["jobstore"]:
                            store = config["jobstore"]["docs_store"]
                            console.print("  Job Store:")
                            console.print(f"    - Host: {store.get('host', 'N/A')}")
                            console.print(
                                f"    - Database: {store.get('database', 'N/A')}"
                            )
                            console.print(f"    - Port: {store.get('port', 27017)}")

                        console.print()
                    except Exception as e:  # noqa: BLE001 friendly CLI error reporting
                        console.print(
                            f"[red]Error loading project details: {e}[/red]\n"
                        )
                else:
                    console.print(f"[red]Project '{project_name}' not found[/red]\n")
        else:
            console.print("  [yellow]No configuration files found[/yellow]")
    else:
        console.print("  [yellow]~/.jfremote directory does not exist[/yellow]")

    # Features table
    console.print("\n[bold]Key Features:[/bold]\n")

    features_table = Table(show_header=False, box=None)
    features_table.add_column("Feature", style="cyan")
    features_table.add_column("Description")

    features_table.add_row("Remote Submission", "Submit jobs to HPC clusters")
    features_table.add_row("Queue Management", "Automatic job queue handling")
    features_table.add_row("Worker Support", "Multiple worker configurations")
    features_table.add_row("MongoDB Backend", "Persistent job storage")
    features_table.add_row("CLI Interface", "Comprehensive command-line tools")

    console.print(features_table)

    # Quick commands
    console.print("\n[bold]Quick Commands:[/bold]\n")

    commands_table = Table(show_header=False, box=None)
    commands_table.add_column("Command", style="cyan")
    commands_table.add_column("Description")

    commands_table.add_row("install", "Install jobflow-remote")
    commands_table.add_row("setup", "Generate project configuration")
    commands_table.add_row("test", "Submit test job")
    commands_table.add_row("info", "Show this information")

    console.print(commands_table)

    # Documentation links
    console.print("\n[bold]Documentation:[/bold]\n")
    console.print(
        "  • Official Docs: [blue]https://matgenix.github.io/jobflow-remote/[/blue]"
    )
    console.print("  • GitHub: [blue]https://github.com/Matgenix/jobflow-remote[/blue]")
    console.print(
        "  • Installation: [blue]https://matgenix.github.io/jobflow-remote/user/install.html[/blue]"
    )
    console.print(
        "  • Quickstart: [blue]https://matgenix.github.io/jobflow-remote/user/quickstart.html[/blue]"
    )

    # Example workflow
    console.print("\n[bold]Example Workflow:[/bold]\n")

    workflow_panel = Panel(
        "# 1. Install jobflow-remote\n"
        "[cyan]atomate2siesta-jobflow-remote install[/cyan]\n"
        "[dim]→ Runs: pip install jobflow-remote[/dim]\n\n"
        "# 2. Generate configuration\n"
        "[cyan]atomate2siesta-jobflow-remote setup[/cyan]\n"
        "[dim]→ Runs: jf project generate atomate2siesta[/dim]\n\n"
        "# 3. [yellow]Install atomate2siesta on remote cluster[/yellow]\n"
        "[dim]SSH to cluster, activate conda environment, and:[/dim]\n"
        "[cyan]pip install atomate2siesta[/cyan]\n\n"
        "# 4. Edit config file (optional)\n"
        "[cyan]nano ~/.jfremote/atomate2siesta.yaml[/cyan]\n\n"
        "# 5. Check configuration\n"
        "[cyan]jf project check --errors[/cyan]\n\n"
        "# 6. Initialize database\n"
        "[cyan]jf admin reset[/cyan]\n\n"
        "# 7. Submit test job\n"
        "[cyan]atomate2siesta-jobflow-remote test[/cyan]\n\n"
        "# 8. Start runner\n"
        "[cyan]jf runner start[/cyan]\n\n"
        "# 9. Check job status\n"
        "[cyan]jf job list[/cyan]",
        style="green",
    )
    console.print(workflow_panel)

    # Add note about wrapper commands
    console.print("\n[bold]Note:[/bold]")
    console.print(
        "  [dim]• Commands like 'setup' and 'install' are convenience wrappers[/dim]"
    )
    console.print("  [dim]• You can also use the original 'jf' commands directly[/dim]")
    console.print("  [dim]• See documentation for all available 'jf' commands[/dim]")

    console.print()


@click.command()
def runner() -> None:
    """Show runner management commands.

    This command displays common runner management commands for controlling
    the jobflow-remote runner daemon that executes jobs.

    Examples
    --------
        # Show runner commands
        atomate2siesta-jobflow-remote runner
    """
    console.print()

    header = Panel(
        Text("Jobflow Remote Runner Management", style="bold cyan", justify="center"),
        style="cyan",
    )
    console.print(header)

    console.print("\n[bold]Runner Commands:[/bold]\n")

    commands_table = Table(box=None)
    commands_table.add_column("Command", style="cyan", no_wrap=True)
    commands_table.add_column("Description")
    commands_table.add_column("Example", style="dim")

    commands_table.add_row(
        "jf runner start",
        "Start the runner daemon",
        "jf runner start",
    )
    commands_table.add_row(
        "jf runner stop",
        "Stop the runner daemon",
        "jf runner stop",
    )
    commands_table.add_row(
        "jf runner status",
        "Check runner status",
        "jf runner status",
    )
    commands_table.add_row(
        "jf runner restart",
        "Restart the runner",
        "jf runner restart",
    )
    commands_table.add_row(
        "jf runner logs",
        "View runner logs",
        "jf runner logs --tail 50",
    )

    console.print(commands_table)

    console.print("\n[bold]Job Management:[/bold]\n")

    job_table = Table(box=None)
    job_table.add_column("Command", style="cyan", no_wrap=True)
    job_table.add_column("Description")
    job_table.add_column("Example", style="dim")

    job_table.add_row(
        "jf job list",
        "List all jobs",
        "jf job list --state READY",
    )
    job_table.add_row(
        "jf job info <id>",
        "Show job details",
        "jf job info 1",
    )
    job_table.add_row(
        "jf job output <id>",
        "Get job output",
        "jf job output 1",
    )
    job_table.add_row(
        "jf job cancel <id>",
        "Cancel a job",
        "jf job cancel 1",
    )
    job_table.add_row(
        "jf job retry <id>",
        "Retry failed job",
        "jf job retry 1",
    )

    console.print(job_table)

    console.print("\n[bold]Useful Tips:[/bold]\n")

    tips_panel = Panel(
        "• Keep runner running in background: [cyan]jf runner start -d[/cyan]\n"
        "• Monitor runner in real-time: [cyan]jf runner logs -f[/cyan]\n"
        "• Check database connection: [cyan]jf project check --errors[/cyan]\n"
        "• View all jobs: [cyan]jf job list --all[/cyan]\n"
        "• Get job count by state: [cyan]jf job list --count[/cyan]",
        style="green",
    )
    console.print(tips_panel)

    console.print()


@click.command()
@click.option(
    "--project-name",
    default="atomate2siesta",
    help="Name of the jobflow-remote project to update",
)
@click.option(
    "--database",
    help="MongoDB database name to update",
)
@click.option(
    "--host",
    help="MongoDB host to update",
)
@click.option(
    "--port",
    type=int,
    help="MongoDB port to update",
)
@click.option(
    "--add-comments",
    is_flag=True,
    help="Rewrite config file with descriptive comments for all entries",
)
@click.option(
    "--backup/--no-backup",
    default=True,
    help="Create backup before updating (default: True)",
)
def update(
    project_name: str,
    database: str | None,
    host: str | None,
    port: int | None,
    add_comments: bool,
    backup: bool,
) -> None:
    """Update existing jobflow-remote project configuration.

    This command allows you to update specific settings in an existing
    jobflow-remote project configuration file. You can update MongoDB
    connection settings and optionally add descriptive comments to all
    configuration entries.

    Examples
    --------
        # Update MongoDB settings
        atomate2siesta-jobflow-remote update --database mydb --host localhost \
            --port 27017

        # Add descriptive comments to all configuration entries
        atomate2siesta-jobflow-remote update --add-comments

        # Update settings and add comments
        atomate2siesta-jobflow-remote update --database mydb --add-comments

        # Update specific project
        atomate2siesta-jobflow-remote update --project-name my_project --database newdb

        # Update without backup (not recommended)
        atomate2siesta-jobflow-remote update --no-backup --database mydb
    """
    console.print("\n[bold cyan]Updating jobflow-remote configuration...[/bold cyan]\n")

    # Determine config file path
    config_path = Path.home() / ".jfremote" / f"{project_name}.yaml"

    if not config_path.exists():
        console.print(
            f"[bold red]✗ Configuration file not found: {config_path}[/bold red]\n"
        )
        console.print(
            "Generate it first with: [cyan]atomate2siesta-jobflow-remote setup[/cyan]"
        )
        sys.exit(1)

    # Create backup if requested
    if backup:
        backup_path = _backup_config(config_path)
        console.print(f"[green]✓ Backup created:[/green] {backup_path}\n")

    # Load existing configuration
    try:
        config = _load_yaml_config(config_path)
    except Exception as e:  # noqa: BLE001 friendly CLI error reporting
        console.print("[bold red]✗ Failed to load configuration![/bold red]")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    # Track what was updated
    updates_made = []

    # Update MongoDB settings if provided
    if database or host or port:
        console.print("[yellow]Updating MongoDB settings...[/yellow]\n")

        # Update queue store settings
        if "queue" not in config:
            config["queue"] = {}
        if "store" not in config["queue"]:
            config["queue"]["store"] = {}

        if database:
            config["queue"]["store"]["database"] = database
            updates_made.append(f"queue.store.database → {database}")

        if host:
            config["queue"]["store"]["host"] = host
            updates_made.append(f"queue.store.host → {host}")

        if port:
            config["queue"]["store"]["port"] = port
            updates_made.append(f"queue.store.port → {port}")

        # Update jobstore settings
        if "jobstore" not in config:
            config["jobstore"] = {}
        if "docs_store" not in config["jobstore"]:
            config["jobstore"]["docs_store"] = {}

        if database:
            config["jobstore"]["docs_store"]["database"] = database
            updates_made.append(f"jobstore.docs_store.database → {database}")

        if host:
            config["jobstore"]["docs_store"]["host"] = host
            updates_made.append(f"jobstore.docs_store.host → {host}")

        if port:
            config["jobstore"]["docs_store"]["port"] = port
            updates_made.append(f"jobstore.docs_store.port → {port}")

        # Update additional stores if they exist
        if "additional_stores" in config["jobstore"]:
            for store_name, store_config in config["jobstore"][
                "additional_stores"
            ].items():
                if isinstance(store_config, dict):
                    if database:
                        store_config["database"] = database
                    if host:
                        store_config["host"] = host
                    if port:
                        store_config["port"] = port
                    if any([database, host, port]):
                        updates_made.append(
                            f"jobstore.additional_stores.{store_name} → updated"
                        )

    # Save updated configuration
    try:
        _save_yaml_config(config_path, config, add_comments=add_comments)
        console.print(
            "[bold green]✓ Configuration updated successfully![/bold green]\n"
        )
    except Exception as e:  # noqa: BLE001 friendly CLI error reporting
        console.print("[bold red]✗ Failed to save configuration![/bold red]")
        console.print(f"[red]Error: {e}[/red]")
        if backup:
            console.print(f"\n[yellow]Restore from backup: {backup_path}[/yellow]")
        sys.exit(1)

    # Show what was updated
    if updates_made or add_comments:
        update_text = f"[bold]Updated Configuration:[/bold] {config_path}\n\n"

        if updates_made:
            update_text += "[bold]Settings Updated:[/bold]\n"
            for update in updates_made:
                update_text += f"  • {update}\n"
            update_text += "\n"

        if add_comments:
            update_text += (
                "[bold green]✓ Descriptive comments added to all "
                "configuration entries[/bold green]\n"
            )

        update_panel = Panel(
            update_text,
            title="Update Complete",
            style="green",
        )
        console.print(update_panel)

        # Show next steps
        next_steps = Panel(
            "[bold]Next Steps:[/bold]\n\n"
            "1. Review updated configuration:\n"
            f"   [cyan]cat {config_path}[/cyan]\n\n"
            "2. Verify configuration:\n"
            "   [cyan]jf project check --errors[/cyan]\n\n"
            "3. Reinitialize database if MongoDB settings changed:\n"
            "   [cyan]jf admin reset[/cyan]\n\n"
            "4. Restart runner if running:\n"
            "   [cyan]jf runner restart[/cyan]",
            title="Verification",
            style="green",
        )
        console.print(next_steps)
    else:
        console.print(
            "[yellow]No updates were made. "
            "Use --help to see available options.[/yellow]\n"
        )


@click.command()
@click.option(
    "-f",
    "--flow-id",
    type=int,
    help="Flow ID to download (mutually exclusive with -j)",
)
@click.option(
    "-j",
    "--job-id",
    help="Job ID (db_id) to download (mutually exclusive with -f)",
)
@click.option(
    "-d",
    "--output-dir",
    default="./flow_outputs",
    help="Output directory for downloaded files (default: ./flow_outputs)",
)
@click.option(
    "--files",
    default=None,
    help="Comma-separated list of files to download (default: download ALL files)",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Resume interrupted downloads (default: enabled)",
)
@click.pass_context
def download(
    ctx: click.Context,
    flow_id: int | None,
    job_id: str | None,
    output_dir: str,
    files: str | None,  # noqa: ARG001 reserved CLI option, not yet used
    resume: bool,
) -> None:
    """Download job outputs from a flow or single job.

    This command downloads output files using jobflow-remote. It creates
    a structured directory with one folder per job containing all files,
    using the same folder names as on the remote cluster.

    Supports resume for interrupted downloads (enabled by default).

    Note: Project name is taken from the top-level -p flag.

    Examples
    --------
        # Download ALL jobs from flow 1
        atomate2siesta-jobflow-remote -p alberto download -f 1

        # Download single job by ID
        atomate2siesta-jobflow-remote -p alberto download -j 10

        # Resume interrupted download
        atomate2siesta-jobflow-remote -p alberto download -f 1 --resume

        # Download to custom directory
        atomate2siesta-jobflow-remote -p alberto download -f 1 -d ./my_results
    """
    import json
    from pathlib import Path

    # Get project name from parent context
    project = ctx.obj.get("project_name", "atomate2siesta")

    # Validate mutually exclusive options
    if not flow_id and not job_id:
        console.print(
            "[red]Error: Must specify either -f/--flow-id or -j/--job-id[/red]"
        )
        return
    if flow_id and job_id:
        console.print(
            "[red]Error: Cannot specify both -f/--flow-id and -j/--job-id[/red]"
        )
        return

    console.print("\n[bold cyan]Downloading Outputs[/bold cyan]\n")
    console.print(f"Project: [yellow]{project}[/yellow]")
    if flow_id:
        console.print(f"Flow ID: [yellow]{flow_id}[/yellow]")
    else:
        console.print(f"Job ID: [yellow]{job_id}[/yellow]")
    console.print(f"Output directory: [yellow]{output_dir}[/yellow]")
    console.print(f"Resume: [yellow]{'enabled' if resume else 'disabled'}[/yellow]\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Get jobs list (either from flow or single job)
        jobs_info = []

        if flow_id:
            # Get all jobs from flow using jf flow info
            console.print("[cyan]Getting flow information...[/cyan]")
            result = subprocess.run(
                # jf resolved from PATH (jobflow-remote CLI)
                ["jf", "-p", project, "flow", "info", str(flow_id)],  # noqa: S607
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse the table to extract DB ids
            lines = result.stdout.strip().split("\n")

            for line in lines:
                # Skip header lines and separator lines
                if "│" not in line or "━" in line or "DB id" in line or "Flow:" in line:
                    continue

                # Split by │ and clean up
                parts = [p.strip() for p in line.split("│") if p.strip()]

                if len(parts) >= 2:
                    try:
                        db_id = parts[0]
                        job_name = parts[1]
                        # Make sure db_id is numeric
                        int(db_id)  # Test if it's a number
                        jobs_info.append({"db_id": db_id, "name": job_name})
                    except (IndexError, ValueError):
                        continue

            if not jobs_info:
                console.print("[yellow]No jobs found in this flow![/yellow]")
                return

            console.print(f"[green]Found {len(jobs_info)} jobs in flow[/green]\n")

        else:
            # Single job mode - just add the job ID
            console.print("[cyan]Getting job information...[/cyan]")
            # Verify job exists
            result = subprocess.run(
                # jf resolved from PATH (jobflow-remote CLI)
                ["jf", "-p", project, "job", "info", job_id],  # noqa: S607
                capture_output=True,
                text=True,
                check=True,
            )

            # Extract job name from info
            job_name = "unknown"
            for line in result.stdout.split("\n"):
                if "name" in line.lower() and "=" in line:
                    cleaned = line.replace("│", "").strip()
                    match = re.search(r"name\s*=\s*'([^']+)'", cleaned)
                    if match:
                        job_name = match.group(1)
                        break

            jobs_info = [{"db_id": job_id, "name": job_name}]
            console.print(f"[green]Found job: {job_name}[/green]\n")

        # Download each job using db_id
        success_count = 0
        failed_jobs = []

        # Temporarily disable progress bar for debugging
        # from rich.progress import Progress, SpinnerColumn, TextColumn,
        # BarColumn, TaskProgressColumn

        # with Progress(
        #     SpinnerColumn(),
        #     TextColumn("[bold blue]{task.description}"),
        #     BarColumn(),
        #     TaskProgressColumn(),
        #     console=console,
        # ) as progress:
        #     overall_task = progress.add_task(
        #         "[cyan]Downloading jobs...",
        #         total=len(jobs_info)
        #     )

        if flow_id:
            console.print("[cyan]Downloading jobs from flow...[/cyan]\n")
        else:
            console.print("[cyan]Downloading job...[/cyan]\n")

        for idx, job in enumerate(jobs_info, 1):
            db_id = job["db_id"]
            job_name = job["name"]

            console.print(f"[cyan]Job {idx}/{len(jobs_info)}: {job_name}[/cyan]")

            try:
                # First get the remote directory name from jf job info
                info_cmd = ["jf", "-p", project, "job", "info", db_id]
                info_result = subprocess.run(
                    info_cmd, capture_output=True, text=True, check=True
                )

                # Extract run_dir path to get the folder name
                remote_folder_name = None
                lines = info_result.stdout.split("\n")
                for i, line in enumerate(lines):
                    if "run_dir" in line.lower() and "=" in line:
                        # Get this line and next line
                        cleaned = line.replace("│", "").strip()
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].replace("│", "").strip()
                            cleaned = cleaned + next_line

                        # Extract the path
                        # Format: run_dir = '/path/to/folder_name'
                        match = re.search(r"run_dir\s*=\s*'([^']+)'", cleaned)
                        if match:
                            full_path = match.group(1)
                            # Get the last part of the path (the folder name)
                            remote_folder_name = full_path.split("/")[-1]
                            break

                if not remote_folder_name:
                    console.print(
                        "  [yellow]Could not get remote folder name, "
                        "using db_id[/yellow]"
                    )
                    remote_folder_name = f"job_{db_id}"

                # Create job-specific directory using remote folder name
                job_dir = output_path / remote_folder_name
                job_dir.mkdir(parents=True, exist_ok=True)

                # Store remote folder name in job dict
                job["local_folder"] = remote_folder_name

                # Get remote directory path from earlier extraction
                # Extract run_dir to get full remote path
                remote_dir = None
                lines = info_result.stdout.split("\n")
                for i, line in enumerate(lines):
                    if "run_dir" in line.lower() and "=" in line:
                        cleaned = line.replace("│", "").strip()
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].replace("│", "").strip()
                            cleaned = cleaned + next_line
                        match = re.search(r"run_dir\s*=\s*'([^']+)'", cleaned)
                        if match:
                            remote_dir = match.group(1)
                            break

                if not remote_dir:
                    console.print("  [red]Could not find remote directory[/red]\n")
                    failed_jobs.append((db_id, job_name))
                    continue

                # Get worker info for rsync
                worker_name = None
                for line in lines:
                    if "worker" in line.lower() and "=" in line:
                        cleaned = line.replace("│", "").strip()
                        match = re.search(r"worker\s*=\s*'([^']+)'", cleaned)
                        if match:
                            worker_name = match.group(1)
                            break

                # Get worker config for SSH
                config_path = Path.home() / ".jfremote" / f"{project}.yaml"
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                worker_config = config.get("workers", {}).get(worker_name, {})
                worker_host = worker_config.get("host")
                worker_user = worker_config.get("user")

                if not worker_host:
                    console.print("  [red]Could not find worker host[/red]\n")
                    failed_jobs.append((db_id, job_name))
                    continue

                # Use rsync to download entire directory tree
                ssh_dest = (
                    f"{worker_user}@{worker_host}" if worker_user else worker_host
                )
                rsync_cmd = [
                    "rsync",
                    "-az",  # archive mode + compress
                ]

                # Add resume support
                if resume:
                    rsync_cmd.extend(
                        [
                            "--partial",  # Keep partially transferred files
                            # Store partial files in subdirectory
                            "--partial-dir=.rsync-partial",
                            "--progress",  # Show progress
                        ]
                    )

                rsync_cmd.extend([f"{ssh_dest}:{remote_dir}/", str(job_dir) + "/"])

                result = subprocess.run(
                    rsync_cmd, capture_output=True, text=True, check=True
                )

                # Store remote path in job dict for summary
                job["remote_path"] = remote_dir

                console.print(f"  [green]✓ Downloaded all files to {job_dir}[/green]\n")
                success_count += 1

            except subprocess.CalledProcessError as e:
                console.print("  [red]✗ Failed to download[/red]")
                console.print(f"  [red]Error: {e.stderr}[/red]\n")
                failed_jobs.append((db_id, job_name))
            except Exception as e:  # noqa: BLE001 friendly CLI error reporting
                console.print(f"  [red]✗ Failed: {type(e).__name__}: {e}[/red]\n")
                failed_jobs.append((db_id, job_name))

        # Summary
        console.print("\n[bold]Download Summary:[/bold]")
        console.print(f"  Total jobs: {len(jobs_info)}")
        console.print(f"  [green]Successfully downloaded: {success_count}[/green]")

        if failed_jobs:
            console.print(f"  [yellow]Failed: {len(failed_jobs)}[/yellow]")
            console.print("\n[yellow]Failed jobs:[/yellow]")
            for db_id, name in failed_jobs:
                console.print(f"    - Job {db_id}: {name}")

        console.print(f"\n[bold green]✓ Outputs saved to: {output_path}[/bold green]\n")

        # Create summary file
        summary_file = output_path / "WORKFLOW_SUMMARY.md"
        with open(summary_file, "w") as f:
            f.write("# Workflow Download Summary\n\n")
            f.write(f"**Project**: {project}\n")
            f.write(f"**Flow ID**: {flow_id}\n")
            # local time intentional for human-readable summary
            download_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # noqa: DTZ005
            f.write(f"**Download Date**: {download_date}\n")
            f.write(f"**Total Jobs**: {len(jobs_info)}\n")
            f.write(f"**Successfully Downloaded**: {success_count}\n")
            f.write(f"**Failed**: {len(failed_jobs)}\n\n")

            f.write("## Directory Structure\n\n")
            f.write("```\n")
            f.write(f"{output_path.name}/\n")
            for job in jobs_info:
                db_id = job["db_id"]
                local_folder = job.get("local_folder", f"job_{db_id}")
                status = "✓" if db_id not in [str(fj[0]) for fj in failed_jobs] else "✗"
                f.write(f"├── {status} {local_folder}/\n")
            f.write("└── WORKFLOW_SUMMARY.md (this file)\n")
            f.write("```\n\n")

            f.write("## Job Details\n\n")
            f.write(
                "| DB ID | Job Name | Status | Local Directory "
                "| Remote Path (Cluster) |\n"
            )
            f.write(
                "|-------|----------|--------|-----------------|----------------------|\n"
            )
            for job in jobs_info:
                db_id = job["db_id"]
                job_name = job["name"]
                remote_path = job.get("remote_path", "N/A")
                local_folder = job.get("local_folder", f"job_{db_id}")
                status = (
                    "✓ Success"
                    if db_id not in [str(fj[0]) for fj in failed_jobs]
                    else "✗ Failed"
                )
                f.write(
                    f"| {db_id} | {job_name} | {status} "
                    f"| `{local_folder}/` | `{remote_path}` |\n"
                )

            if failed_jobs:
                f.write("\n## Failed Jobs\n\n")
                f.write("The following jobs failed to download:\n\n")
                for db_id, name in failed_jobs:
                    f.write(f"- **Job {db_id}**: {name}\n")
                f.write(
                    "\n**Tip**: Check if these jobs completed "
                    "successfully on the cluster.\n"
                )

            f.write("\n## How to Use These Files\n\n")
            f.write("Each job directory contains:\n")
            f.write("- `siesta.out` - Main SIESTA output file\n")
            f.write("- `siesta.fdf` - Input parameters\n")
            f.write("- `siesta.XV` - Final structure coordinates\n")
            f.write("- `*.json` - Task documentation and results\n")
            f.write("- `*.psml` or `*.ion.nc` - Pseudopotential files\n")
            f.write("- `submit.sh` - SLURM submission script\n")
            f.write("- `queue.out/err` - Cluster output/error logs\n\n")

            f.write("### Quick Commands\n\n")
            f.write("```bash\n")
            f.write("# View main output of a job\n")
            f.write("cat job_*/siesta.out\n\n")
            f.write("# Check energy convergence\n")
            f.write("grep 'Total =' job_*/siesta.out\n\n")
            f.write("# View final structures\n")
            f.write("ls -lh job_*/siesta.XV\n\n")
            f.write("# Extract JSON results\n")
            f.write("cat job_*/*.json | jq .\n")
            f.write("```\n\n")

            f.write("---\n")
            f.write(
                "*Downloaded using `atomate2siesta-jobflow-remote download "
                f"-p {project} -f {flow_id}`*\n"
            )

        console.print(f"[green]✓ Summary saved to: {summary_file}[/green]\n")

    except subprocess.CalledProcessError as e:
        console.print("\n[bold red]✗ Error getting flow information[/bold red]")
        console.print(f"[red]{e.stderr}[/red]\n")
        console.print("Make sure:")
        console.print(f"  1. Project '{project}' exists")
        console.print(f"  2. Flow ID {flow_id} is valid")
        console.print(f"  3. Run: [cyan]jf -p {project} flow info {flow_id}[/cyan]\n")
        sys.exit(1)
    except json.JSONDecodeError as e:
        console.print("\n[bold red]✗ Error parsing flow data[/bold red]")
        console.print(f"[red]{e}[/red]\n")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 friendly CLI error reporting
        console.print("\n[bold red]✗ Unexpected error[/bold red]")
        console.print(f"[red]{e}[/red]\n")
        sys.exit(1)
