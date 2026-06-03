#!/usr/bin/env python
import click
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Initialize rich console
console = Console()


def ensure_config_file(
    file_name,
    config_dir,
    siesta_cmd,
    siesta_pp_path,
    flos_path,
    optical_input_cmd,
    optical_cmd,
    show_banner,
    parameter_evolution="summary",
):
    """Create the default .atomate2siesta-local.yaml file if it doesn't exist."""
    try:
        config_path = Path(config_dir) / file_name
        if not config_path.exists():
            config_content = f"""SIESTA_CMD: "{siesta_cmd}"
SIESTA_PP_PATH: "{siesta_pp_path}"
FLOS_PATH: "{flos_path}"
OPTICAL_INPUT_CMD: "{optical_input_cmd}"
OPTICAL_CMD: "{optical_cmd}"
SIESTA_SHOW_BANNER: {show_banner}
SIESTA_SHOW_PARAMETER_EVOLUTION: "{parameter_evolution}"  # Options: none, user, diff, summary, full
"""
            os.makedirs(config_dir, exist_ok=True)
            with open(config_path, "w") as f:
                f.write(config_content)
            console.print(
                Panel(
                    f"Created default config file at: [bold cyan]{config_path}[/bold cyan]",
                    style="green",
                )
            )
        else:
            console.print(
                f"[yellow]Config file already exists at:[/yellow] [bold cyan]{config_path}[/bold cyan]"
            )
            # return
            raise FileExistsError(f"Config file already exists at: {config_path}")
        return config_path
    except Exception as e:
        console.print(f"[red]Error creating config file:[/red] {e}")
        return None


@click.group()
def cli():
    """CLI for Atomate2 SIESTA configuration."""
    console.print(
        "[bold magenta]Atomate2 SIESTA Configuration CLI[/bold magenta]", style="bold"
    )


@cli.command()
@click.option(
    "--file-name",
    default=".atomate2siesta-local.yaml",
    help="Name of the config file (default: .atomate2siesta-local.yaml)",
)
@click.option(
    "--output-dir",
    default=None,
    help="Directory to save the config file (defaults to current directory)",
)
@click.option(
    "--siesta-cmd",
    default=None,
    help="Command to run SIESTA (if not specified, auto-generated based on --use-srun)",
)
@click.option(
    "--siesta-pp-path",
    default=None,
    help="Path to SIESTA pseudopotentials (defaults to ~/.siesta/pseudos)",
)
@click.option(
    "--flos-path",
    default=None,
    help="Path to FLOS library (defaults to ~/flos if it exists)",
)
@click.option(
    "--optical-input-cmd",
    default=None,
    help="Command for optical input (if not specified, auto-generated based on --use-srun)",
)
@click.option(
    "--optical-cmd",
    default=None,
    help="Command for optical calculation (if not specified, auto-generated based on --use-srun)",
)
@click.option(
    "--use-srun/--no-use-srun",
    default=False,
    help="Use 'srun' prefix for cluster/MPI execution (required for SLURM systems). Default: False",
)
@click.option(
    "--show-banner/--no-show-banner",
    default=True,
    help="Display welcome banner and logo on module import (default: True)",
)
@click.option(
    "--parameter-evolution",
    type=click.Choice(
        ["none", "user", "diff", "summary", "full"], case_sensitive=False
    ),
    default="summary",
    help="Parameter evolution display level (default: summary)",
)
def create(
    file_name,
    output_dir,
    siesta_cmd,
    siesta_pp_path,
    flos_path,
    optical_input_cmd,
    optical_cmd,
    use_srun,
    show_banner,
    parameter_evolution,
):
    """Create an Atomate2 SIESTA configuration file in the specified directory."""
    try:
        # Set default output_dir to current working directory if not provided
        if output_dir is None:
            output_dir = os.getcwd()
        console.print(
            f"[blue]Using output directory:[/blue] [bold cyan]{output_dir}[/bold cyan]"
        )

        # Auto-generate paths based on home directory if not explicitly provided
        home_dir = Path.home()

        if siesta_pp_path is None:
            siesta_pp_path = str(home_dir / ".siesta" / "pseudos")
            console.print(
                f"[dim]Auto-detected pseudopotential path:[/dim] [cyan]{siesta_pp_path}[/cyan]"
            )

        if flos_path is None:
            # Check if ~/flos exists, otherwise use a placeholder
            default_flos = home_dir / "flos"
            if default_flos.exists():
                flos_path = str(default_flos)
                console.print(
                    f"[dim]Auto-detected FLOS path:[/dim] [cyan]{flos_path}[/cyan]"
                )
            else:
                flos_path = str(home_dir / "flos")
                console.print(
                    f"[dim]FLOS path (not verified):[/dim] [yellow]{flos_path}[/yellow]"
                )

        # Auto-generate commands based on use_srun flag if not explicitly provided
        if siesta_cmd is None:
            siesta_cmd = (
                "srun siesta < siesta.fdf > siesta.out"
                if use_srun
                else "siesta < siesta.fdf > siesta.out"
            )
        if optical_input_cmd is None:
            optical_input_cmd = (
                "srun optical_input < siesta.EPSIMG"
                if use_srun
                else "optical_input < siesta.EPSIMG"
            )
        if optical_cmd is None:
            optical_cmd = (
                "srun optical < siesta.EPSIMG"
                if use_srun
                else "optical < siesta.EPSIMG"
            )

        # Display configuration mode
        if use_srun:
            console.print(
                "[green]✓ Using SLURM mode (srun)[/green] - Suitable for cluster environments"
            )
        else:
            console.print(
                "[yellow]Using direct mode (no srun)[/yellow] - Suitable for local/workstation use"
            )

        # Create config file with provided or default parameters
        config_path = ensure_config_file(
            file_name,
            output_dir,
            siesta_cmd=siesta_cmd,
            siesta_pp_path=siesta_pp_path,
            flos_path=flos_path,
            optical_input_cmd=optical_input_cmd,
            optical_cmd=optical_cmd,
            show_banner=show_banner,
            parameter_evolution=parameter_evolution,
        )
        if config_path:
            console.print(
                Panel(
                    f"Created SIESTA configuration file at: [bold cyan]{config_path}[/bold cyan]",
                    style="green",
                )
            )
            # Print export command for user to run
            export_command = f'export ATOMATE2_CONFIG_FILE="{config_path}"'
            console.print(
                Panel(
                    Text("Run the following command in your shell:\n", style="bold")
                    + Text(export_command, style="bold cyan"),
                    style="green",
                )
            )
    except Exception as e:
        console.print(f"[red]Error in create_siesta_config:[/red] {e}")


@cli.command()
@click.argument("file_path")
def set(file_path):
    """Set the ATOMATE2_CONFIG_FILE environment variable for the current session."""
    try:
        console.print(
            f"[blue]Received file_path:[/blue] [bold cyan]{file_path}[/bold cyan]"
        )
        config_dir = os.getcwd()
        config_path = Path(file_path)

        # If the path is not absolute, assume it's relative to the current directory
        if not config_path.is_absolute():
            config_path = Path(config_dir) / file_path
        console.print(
            f"[blue]Resolved config path:[/blue] [bold cyan]{config_path}[/bold cyan]"
        )

        # Check if the config file exists
        if not config_path.exists():
            console.print(
                f"[red]Error: Config file '[bold]{config_path}[/bold]' does not exist.[/red]"
            )
            return

        # Check file permissions
        if not os.access(config_path, os.R_OK):
            console.print(
                f"[red]Error: Config file '[bold]{config_path}[/bold]' is not readable. Check permissions.[/red]"
            )
            return

        # Set the environment variable
        os.environ["ATOMATE2_CONFIG_FILE"] = str(config_path)
        export_command = f'export ATOMATE2_CONFIG_FILE="{config_path}"'
        console.print(
            Panel(
                Text("Run the following command in your shell:\n", style="bold")
                + Text(export_command, style="bold cyan"),
                style="green",
            )
        )
    except Exception as e:
        console.print(f"[red]Error in set_config:[/red] {e}")


@cli.command()
def status():
    """Check the current configuration status and ATOMATE2_CONFIG_FILE environment variable."""
    from rich.table import Table

    console.print("\n[bold cyan]Configuration Status[/bold cyan]\n")

    # Check if ATOMATE2_CONFIG_FILE is set
    env_config_file = os.getenv("ATOMATE2_CONFIG_FILE")

    # Default config file locations to check
    default_config = Path("~/.atomate2siesta.yaml").expanduser()
    local_config = Path("atomate2siesta-local.yaml")  # Old non-hidden format
    local_hidden_config = Path(
        ".atomate2siesta-local.yaml"
    )  # New hidden format (current default)

    # Create status table
    table = Table(
        title="Environment Variable Status",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Variable", style="cyan", width=25)
    table.add_column("Status", style="green", width=15)
    table.add_column("Value", style="yellow")

    if env_config_file:
        env_path = Path(env_config_file).expanduser()
        if env_path.exists():
            status_icon = "✓ Set & Exists"
            status_style = "green"
        else:
            status_icon = "✗ Set but Missing"
            status_style = "red"
        table.add_row(
            "ATOMATE2_CONFIG_FILE", Text(status_icon, style=status_style), str(env_path)
        )
    else:
        table.add_row(
            "ATOMATE2_CONFIG_FILE",
            Text("Not Set", style="yellow"),
            Text("Using defaults or file discovery", style="dim"),
        )

    console.print(table)

    # Create file discovery table
    file_table = Table(
        title="Configuration File Discovery",
        show_header=True,
        header_style="bold magenta",
    )
    file_table.add_column("Location", style="cyan", width=40)
    file_table.add_column("Status", style="green", width=15)
    file_table.add_column("Priority", style="yellow", width=10)

    # Check priority order
    priority = 1

    # If ATOMATE2_CONFIG_FILE is set and exists, it has highest priority
    if env_config_file:
        env_path = Path(env_config_file).expanduser()
        if env_path.exists():
            file_table.add_row(
                str(env_path),
                Text("✓ Active", style="green bold"),
                Text(f"#{priority}", style="green bold"),
            )
            priority += 1

    # Default file
    if default_config.exists():
        is_active = (
            not env_config_file or not Path(env_config_file).expanduser().exists()
        )
        file_table.add_row(
            str(default_config),
            Text("✓ Exists", style="green")
            if not is_active
            else Text("✓ Active", style="green bold"),
            Text(f"#{priority}", style="green" if not is_active else "green bold"),
        )
    else:
        file_table.add_row(
            str(default_config),
            Text("✗ Not Found", style="dim"),
            Text(f"#{priority}", style="dim"),
        )
    priority += 1

    # Local config files
    for local_file, description in [
        (local_config, "Local config (current dir)"),
        (local_hidden_config, "Local hidden config (current dir)"),
    ]:
        if local_file.exists():
            file_table.add_row(
                f"{local_file} ({description})",
                Text("✓ Exists", style="blue"),
                Text("Manual", style="blue"),
            )

    console.print(file_table)

    # Show what will be used
    console.print("\n[bold]Active Configuration:[/bold]")
    if env_config_file and Path(env_config_file).expanduser().exists():
        console.print(f"  [green]Using:[/green] {Path(env_config_file).expanduser()}")
        console.print("  [dim]Source: ATOMATE2_CONFIG_FILE environment variable[/dim]")
    elif default_config.exists():
        console.print(f"  [green]Using:[/green] {default_config}")
        console.print("  [dim]Source: Default location (~/.atomate2siesta.yaml)[/dim]")
    else:
        console.print("  [yellow]Using built-in defaults[/yellow]")
        console.print("  [dim]No configuration file found[/dim]")

    # Show current settings by loading them
    console.print("\n[bold]Current Settings:[/bold]")
    try:
        from atomate2.siesta import SETTINGS
        import shutil

        def check_command_exists(cmd_string):
            """Check if the main executable in a command string exists."""
            if not cmd_string:
                return "-"

            # Parse command to extract main executable
            # Handle cases like: "mpirun -n 4 /path/to/siesta < input > output"
            parts = cmd_string.split()
            if not parts:
                return "-"

            # Skip common MPI wrappers and their flags
            mpi_wrappers = ["mpirun", "mpiexec", "srun", "ibrun"]
            executable = None
            skip_next = False  # Flag to skip the value after a flag

            for part in parts:
                # Skip this part if it's a flag value
                if skip_next:
                    skip_next = False
                    continue

                # Skip MPI wrappers
                if part in mpi_wrappers:
                    continue

                # Skip flags and mark to skip their values
                if part.startswith("-"):
                    skip_next = True  # Skip next part (flag value like "4" in "-n 4")
                    continue

                # Stop at redirections
                if part in ["<", ">", ">>", "|"]:
                    break

                # Found the executable (first non-flag, non-wrapper part)
                executable = part
                break

            if not executable:
                return "-"

            # Check if executable exists
            # First try as absolute/relative path
            exec_path = Path(executable).expanduser()
            if exec_path.exists() and exec_path.is_file():
                # Check if it's executable
                if os.access(exec_path, os.X_OK):
                    return "[green]✓ Found[/green]"
                else:
                    return "[red]✗ Not Executable[/red]"

            # Then check in PATH
            if shutil.which(executable):
                return "[green]✓ Found[/green]"

            return "[red]✗ Not Found[/red]"

        settings_table = Table(show_header=True, header_style="bold magenta")
        settings_table.add_column("Setting", style="cyan", width=35)
        settings_table.add_column("Value", style="yellow")
        settings_table.add_column("Status", width=15)

        # Commands (with executable checks)
        settings_table.add_row(
            "SIESTA_CMD", SETTINGS.SIESTA_CMD, check_command_exists(SETTINGS.SIESTA_CMD)
        )
        settings_table.add_row(
            "VIBRA_CMD", SETTINGS.VIBRA_CMD, check_command_exists(SETTINGS.VIBRA_CMD)
        )
        settings_table.add_row(
            "OPTICAL_INPUT_CMD",
            SETTINGS.OPTICAL_INPUT_CMD,
            check_command_exists(SETTINGS.OPTICAL_INPUT_CMD),
        )
        settings_table.add_row(
            "OPTICAL_CMD",
            SETTINGS.OPTICAL_CMD,
            check_command_exists(SETTINGS.OPTICAL_CMD),
        )

        # Paths (with existence checks)
        # Check SIESTA_PP_PATH
        if SETTINGS.SIESTA_PP_PATH:
            pp_path = Path(SETTINGS.SIESTA_PP_PATH)
            pp_status = (
                "[green]✓ Exists[/green]"
                if pp_path.exists()
                else "[red]✗ Not Found[/red]"
            )
            pp_value = str(SETTINGS.SIESTA_PP_PATH)
        else:
            pp_value = "Not set"
            pp_status = "-"
        settings_table.add_row("SIESTA_PP_PATH", pp_value, pp_status)

        # Check FLOS_PATH
        if SETTINGS.FLOS_PATH:
            flos_path = Path(SETTINGS.FLOS_PATH)
            flos_status = (
                "[green]✓ Exists[/green]"
                if flos_path.exists()
                else "[red]✗ Not Found[/red]"
            )
            flos_value = str(SETTINGS.FLOS_PATH)
        else:
            flos_value = "Not set"
            flos_status = "-"
        settings_table.add_row("FLOS_PATH", flos_value, flos_status)

        # Check CONFIG_FILE
        if SETTINGS.CONFIG_FILE:
            config_path = Path(SETTINGS.CONFIG_FILE)
            config_status = (
                "[green]✓ Exists[/green]"
                if config_path.exists()
                else "[red]✗ Not Found[/red]"
            )
            config_value = str(SETTINGS.CONFIG_FILE)
        else:
            config_value = "Not set"
            config_status = "-"
        settings_table.add_row("CONFIG_FILE", config_value, config_status)

        # Display options (no status check needed)
        settings_table.add_row(
            "SIESTA_SHOW_BANNER", str(SETTINGS.SIESTA_SHOW_BANNER), "-"
        )
        settings_table.add_row(
            "SIESTA_SHOW_DOCSTRINGS", str(SETTINGS.SIESTA_SHOW_DOCSTRINGS), "-"
        )
        settings_table.add_row(
            "SIESTA_SHOW_PARAMETER_EVOLUTION",
            SETTINGS.SIESTA_SHOW_PARAMETER_EVOLUTION,
            "-",
        )
        settings_table.add_row("SIESTA_ZIP_FILES", str(SETTINGS.SIESTA_ZIP_FILES), "-")

        # Calculation settings (no status check needed)
        settings_table.add_row("SYMPREC", str(SETTINGS.SYMPREC), "-")
        settings_table.add_row("PHONON_SYMPREC", str(SETTINGS.PHONON_SYMPREC), "-")
        settings_table.add_row(
            "ELASTIC_FITTING_METHOD", SETTINGS.ELASTIC_FITTING_METHOD, "-"
        )

        console.print(settings_table)
    except Exception as e:
        console.print(f"[red]Could not load current settings: {e}[/red]")

    # Check database configuration
    console.print("\n[bold cyan]Database Configuration[/bold cyan]\n")
    db_table = Table(
        title="MongoDB Connection Status",
        show_header=True,
        header_style="bold magenta",
    )
    db_table.add_column("Parameter", style="cyan", width=20)
    db_table.add_column("Value", style="yellow", width=30)
    db_table.add_column("Status", style="green", width=20)

    # Default MongoDB settings (could be made configurable)
    db_host = "localhost"
    db_port = 27017
    db_name = "atomate2siesta"
    db_collection = "tasks"

    # Try to import pymongo and test connection
    try:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
        import warnings

        # Suppress pymongo warnings during connection test
        warnings.filterwarnings("ignore", category=UserWarning, module="pymongo")

        # Add connection parameters to table
        db_table.add_row("Host", db_host, "-")
        db_table.add_row("Port", str(db_port), "-")
        db_table.add_row("Database", db_name, "-")
        db_table.add_row("Collection", db_collection, "-")

        # Try to connect with suppressed output
        try:
            client = MongoClient(
                host=db_host,
                port=db_port,
                serverSelectionTimeoutMS=2000,  # Reduced timeout for faster feedback
                connectTimeoutMS=2000,
            )
            # Attempt connection with timeout
            client.admin.command("ping")

            # Connection successful
            db_table.add_row(
                "Connection",
                "Connected",
                Text("✓ Success", style="green"),
            )

            # Check if database exists
            db = client[db_name]
            db_exists = db_name in client.list_database_names()

            if db_exists:
                db_table.add_row(
                    "Database Exists",
                    "Yes",
                    Text("✓ Found", style="green"),
                )

                # Check collection
                coll = db[db_collection]
                doc_count = coll.count_documents({})
                db_table.add_row(
                    "Collection Exists",
                    "Yes",
                    Text("✓ Found", style="green"),
                )
                db_table.add_row(
                    "Document Count",
                    str(doc_count),
                    Text("✓ Active", style="green")
                    if doc_count > 0
                    else Text("-", style="dim"),
                )
            else:
                db_table.add_row(
                    "Database Exists",
                    "No",
                    Text("✗ Not Found", style="red"),
                )
                db_table.add_row(
                    "Collection Exists",
                    "N/A",
                    Text("-", style="dim"),
                )

            client.close()

        except (ConnectionFailure, ServerSelectionTimeoutError, Exception):
            # Clean error message without traceback
            db_table.add_row(
                "Connection",
                f"No database found on port {db_port}",
                Text("✗ Not Found", style="red"),
            )
            db_table.add_row(
                "Database Exists",
                "N/A",
                Text("-", style="dim"),
            )
            db_table.add_row(
                "Collection Exists",
                "N/A",
                Text("-", style="dim"),
            )

    except ImportError:
        db_table.add_row(
            "PyMongo",
            "Not Installed",
            Text("✗ Missing", style="red"),
        )
        db_table.add_row(
            "Connection",
            "N/A",
            Text("-", style="dim"),
        )

    console.print(db_table)

    # Check jobflow.yaml configuration
    console.print("\n[bold cyan]Jobflow Configuration[/bold cyan]\n")
    jf_table = Table(
        title="Jobflow Database Store Configuration",
        show_header=True,
        header_style="bold magenta",
    )
    jf_table.add_column("Parameter", style="cyan", width=25)
    jf_table.add_column("Value", style="yellow", width=30)
    jf_table.add_column("Status", style="green", width=20)

    # Check for jobflow.yaml in common locations
    jobflow_locations = [
        Path("~/.jobflow.yaml").expanduser(),
        Path("jobflow.yaml"),
        Path(".jobflow.yaml"),
    ]

    jobflow_file = None
    for loc in jobflow_locations:
        if loc.exists():
            jobflow_file = loc
            break

    if jobflow_file:
        jf_table.add_row(
            "Config File",
            str(jobflow_file),
            Text("✓ Found", style="green"),
        )

        # Try to parse the jobflow.yaml file
        try:
            import yaml

            with open(jobflow_file) as f:
                jf_config = yaml.safe_load(f)

            # Check for JOB_STORE configuration
            if jf_config and "JOB_STORE" in jf_config:
                job_store = jf_config["JOB_STORE"]
                jf_table.add_row(
                    "JOB_STORE Defined",
                    "Yes",
                    Text("✓ Configured", style="green"),
                )

                # Check database settings
                if isinstance(job_store, dict):
                    store_type = job_store.get("docs_store", {}).get("type", "N/A")
                    jf_table.add_row("Store Type", store_type, "-")

                    # Check MongoDB settings
                    if "docs_store" in job_store:
                        docs_store = job_store["docs_store"]
                        store_host = docs_store.get("host", "N/A")
                        store_port = docs_store.get("port", "N/A")
                        store_db = docs_store.get("database", "N/A")
                        store_coll = docs_store.get("collection_name", "N/A")

                        jf_table.add_row("DB Host", str(store_host), "-")
                        jf_table.add_row("DB Port", str(store_port), "-")
                        jf_table.add_row("DB Name", str(store_db), "-")
                        jf_table.add_row("DB Collection", str(store_coll), "-")

                        # Check if settings match MongoDB connection
                        match_host = str(store_host) == str(db_host)
                        match_port = str(store_port) == str(db_port)
                        match_db = str(store_db) == str(db_name)

                        if match_host and match_port and match_db:
                            jf_table.add_row(
                                "Settings Match",
                                "Yes",
                                Text("✓ Consistent", style="green"),
                            )
                        else:
                            jf_table.add_row(
                                "Settings Match",
                                "No",
                                Text("⚠ Mismatch", style="yellow"),
                            )
            else:
                jf_table.add_row(
                    "JOB_STORE Defined",
                    "No",
                    Text("✗ Missing", style="red"),
                )
                jf_table.add_row(
                    "Store Type",
                    "N/A",
                    Text("-", style="dim"),
                )

        except Exception as e:
            jf_table.add_row(
                "Parse Error",
                str(e)[:30],
                Text("✗ Invalid YAML", style="red"),
            )

    else:
        jf_table.add_row(
            "Config File",
            "Not Found",
            Text("✗ Missing", style="red"),
        )
        jf_table.add_row(
            "JOB_STORE Defined",
            "N/A",
            Text("-", style="dim"),
        )
        jf_table.add_row(
            "Note",
            "Calculations won't save to DB",
            Text("⚠ Action Required", style="yellow"),
        )

    console.print(jf_table)

    # Show helpful commands
    console.print("\n[bold]Helpful Commands:[/bold]")
    console.print("  [bold]Configuration:[/bold]")
    console.print("    • Create config: [cyan]atomate2siesta-config create[/cyan]")
    console.print(
        "    • Create config in home: [cyan]atomate2siesta-config create --output-dir ~[/cyan]"
    )
    console.print(
        '    • Set environment: [cyan]export ATOMATE2_CONFIG_FILE="path/to/config.yaml"[/cyan]'
    )
    console.print("    • Check status: [cyan]atomate2siesta-config status[/cyan]")
    console.print("\n  [bold]Database:[/bold]")
    console.print("    • Test connection: [cyan]atomate2siesta-database test[/cyan]")
    console.print("    • Create database: [cyan]atomate2siesta-database create[/cyan]")
    console.print("    • List calculations: [cyan]atomate2siesta-database list[/cyan]")
    console.print(
        "    • Query database: [cyan]atomate2siesta-database query --formula Si[/cyan]"
    )
    console.print("    • Database stats: [cyan]atomate2siesta-database stats[/cyan]")

    # Add jobflow setup note if jobflow.yaml is missing
    if not jobflow_file:
        console.print("\n  [bold yellow]⚠ Note:[/bold yellow]")
        console.print(
            "    [dim]No jobflow.yaml found. Calculations will only save to database if[/dim]"
        )
        console.print(
            "    [dim]you explicitly define a JobStore in your Python scripts.[/dim]"
        )
        console.print(
            "    [dim]Create ~/.jobflow.yaml for automatic database integration.[/dim]"
        )

        # Show Python example for explicit JobStore definition
        from rich.syntax import Syntax

        example_code = f"""from jobflow import SETTINGS, JobStore, run_locally
from maggma.stores import MongoStore
from atomate2.siesta.jobs.core import RelaxMaker

# Define MongoDB store
store = MongoStore(
    database="{db_name}",
    collection_name="{db_collection}",
    host="{db_host}",
    port={db_port},
)

# Create JobStore and set in SETTINGS
job_store = JobStore(docs_store=store)
SETTINGS.JOB_STORE = job_store

# Create and run job
maker = RelaxMaker.fixed_cell_relaxation()
job = maker.make(structure)
results = run_locally(job, create_folders=True)
"""

        console.print(
            "\n  [bold cyan]Example - Explicit JobStore in Python:[/bold cyan]"
        )
        syntax = Syntax(example_code, "python", theme="monokai", line_numbers=False)
        console.print(syntax)

    console.print()


@cli.command()
@click.argument("config_path", required=False)
@click.option(
    "--shell",
    type=click.Choice(["zsh", "bash", "fish", "auto"], case_sensitive=False),
    default="auto",
    help="Shell type to configure (default: auto-detect)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be added without modifying files",
)
def persist(config_path, shell, dry_run):
    """Add ATOMATE2_CONFIG_FILE export to your shell configuration file (.zshrc/.bashrc).

    This command helps you persist the configuration across shell sessions by adding
    the export statement to your shell's configuration file.

    Examples:
        # Auto-detect shell and add current/default config
        atomate2siesta-config persist

        # Specify config file path
        atomate2siesta-config persist ~/.atomate2siesta.yaml

        # Preview changes without modifying files
        atomate2siesta-config persist --dry-run

        # Specify shell type explicitly
        atomate2siesta-config persist --shell zsh
    """
    from rich.prompt import Confirm

    console.print("\n[bold cyan]Persist Configuration to Shell Profile[/bold cyan]\n")

    # Determine config path
    if config_path:
        config_file = Path(config_path).expanduser().resolve()
    else:
        # Check if ATOMATE2_CONFIG_FILE is already set
        env_config = os.getenv("ATOMATE2_CONFIG_FILE")
        if env_config:
            config_file = Path(env_config).expanduser().resolve()
            console.print(
                f"[dim]Using current ATOMATE2_CONFIG_FILE: {config_file}[/dim]"
            )
        else:
            # Check for default location
            default_config = Path("~/.atomate2siesta.yaml").expanduser()
            local_hidden = Path(".atomate2siesta-local.yaml")

            if default_config.exists():
                config_file = default_config
                console.print(
                    f"[dim]Found config at default location: {config_file}[/dim]"
                )
            elif local_hidden.exists():
                config_file = local_hidden.resolve()
                console.print(
                    f"[dim]Found config in current directory: {config_file}[/dim]"
                )
            else:
                console.print("[yellow]⚠ No configuration file found![/yellow]\n")
                console.print("Please specify a config file path or create one first:")
                console.print(
                    "  [cyan]atomate2siesta-config create --output-dir ~[/cyan]"
                )
                console.print(
                    "  [cyan]atomate2siesta-config persist ~/.atomate2siesta.yaml[/cyan]"
                )
                return

    # Verify config file exists
    if not config_file.exists():
        console.print(f"[red]✗ Config file does not exist: {config_file}[/red]")
        console.print("\nCreate it first with:")
        console.print(
            f"  [cyan]atomate2siesta-config create --output-dir {config_file.parent}[/cyan]"
        )
        return

    # Detect shell
    if shell == "auto":
        shell_env = os.getenv("SHELL", "")
        if "zsh" in shell_env:
            detected_shell = "zsh"
            shell_file = Path.home() / ".zshrc"
        elif "bash" in shell_env:
            detected_shell = "bash"
            shell_file = Path.home() / ".bashrc"
        elif "fish" in shell_env:
            detected_shell = "fish"
            shell_file = Path.home() / ".config" / "fish" / "config.fish"
        else:
            console.print(
                f"[yellow]⚠ Could not detect shell from $SHELL: {shell_env}[/yellow]"
            )
            console.print("\nPlease specify shell explicitly:")
            console.print("  [cyan]atomate2siesta-config persist --shell zsh[/cyan]")
            console.print("  [cyan]atomate2siesta-config persist --shell bash[/cyan]")
            return
    else:
        detected_shell = shell.lower()
        if detected_shell == "zsh":
            shell_file = Path.home() / ".zshrc"
        elif detected_shell == "bash":
            shell_file = Path.home() / ".bashrc"
        elif detected_shell == "fish":
            shell_file = Path.home() / ".config" / "fish" / "config.fish"

    console.print(f"[green]✓ Detected shell:[/green] {detected_shell}")
    console.print(f"[green]✓ Config file:[/green] {config_file}")
    console.print(f"[green]✓ Shell profile:[/green] {shell_file}")

    # Create export line based on shell
    marker_comment = "# Added by atomate2siesta-config"
    if detected_shell == "fish":
        export_line = f'set -gx ATOMATE2_CONFIG_FILE "{config_file}"'
    else:
        export_line = f'export ATOMATE2_CONFIG_FILE="{config_file}"'

    # Check if shell file exists
    if not shell_file.exists():
        if not dry_run:
            console.print(
                f"\n[yellow]⚠ Shell config file does not exist: {shell_file}[/yellow]"
            )
            if Confirm.ask(f"Create {shell_file.name}?", default=True):
                shell_file.parent.mkdir(parents=True, exist_ok=True)
                shell_file.touch()
                console.print(f"[green]✓ Created {shell_file}[/green]")
            else:
                console.print("[yellow]Cancelled.[/yellow]")
                return
        else:
            console.print(
                f"\n[yellow]⚠ Shell config file does not exist: {shell_file}[/yellow]"
            )
            console.print("[dim](Would be created in non-dry-run mode)[/dim]")

    # Check if already exists
    if shell_file.exists():
        with open(shell_file, "r") as f:
            content = f.read()

        if (
            marker_comment in content
            or f'ATOMATE2_CONFIG_FILE="{config_file}"' in content
        ):
            console.print(
                f"\n[yellow]⚠ Export already exists in {shell_file.name}![/yellow]"
            )
            console.print("[dim]The configuration is already persisted.[/dim]")

            # Show the existing line
            for line in content.split("\n"):
                if "ATOMATE2_CONFIG_FILE" in line:
                    console.print(f"\n[dim]Existing line:[/dim] {line}")
            return

    # Show what will be added
    console.print("\n[bold]Lines to be added:[/bold]")
    console.print(
        Panel(
            f"{marker_comment}\n{export_line}", style="cyan", title="Export Statement"
        )
    )

    # Dry run mode
    if dry_run:
        console.print("\n[yellow]Dry-run mode:[/yellow] No changes made.")
        console.print("\nTo apply changes, run:")
        console.print(f"  [cyan]atomate2siesta-config persist {config_file}[/cyan]")
        return

    # Ask for confirmation
    console.print()
    if not Confirm.ask(f"Add these lines to {shell_file}?", default=True):
        console.print("\n[yellow]Cancelled.[/yellow]")
        console.print("\nTo add manually, run:")
        console.print(f"  [cyan]echo '{marker_comment}' >> {shell_file}[/cyan]")
        console.print(f"  [cyan]echo '{export_line}' >> {shell_file}[/cyan]")
        return

    # Add to shell file
    try:
        with open(shell_file, "a") as f:
            f.write(f"\n{marker_comment}\n")
            f.write(f"{export_line}\n")

        console.print(f"\n[green]✓ Successfully added to {shell_file}![/green]")

        # Show next steps
        console.print("\n[bold]Next Steps:[/bold]")
        console.print("  1. Reload your shell configuration:")
        console.print(f"     [cyan]source {shell_file}[/cyan]")
        console.print("  2. Or restart your terminal")
        console.print("  3. Verify with:")
        console.print("     [cyan]atomate2siesta-config status[/cyan]")

    except Exception as e:
        console.print(f"\n[red]✗ Failed to modify {shell_file}: {e}[/red]")
        console.print("\nTo add manually, run:")
        console.print(f"  [cyan]echo '{marker_comment}' >> {shell_file}[/cyan]")
        console.print(f"  [cyan]echo '{export_line}' >> {shell_file}[/cyan]")


if __name__ == "__main__":
    cli()
