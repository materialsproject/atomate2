"""Main CLI entry point for jobflow-remote tools."""

import click
from rich.console import Console

from .inspect import inspect
from .modify_db import modify_db
from .recreate import recreate
from .setup import download, info, install, runner, setup, test, update
from .update_resources import update_resources

console = Console()


@click.group()
@click.option(
    "-p",
    "--project-name",
    default="atomate2siesta",
    help="Jobflow-remote project name (matches jf -p PROJECTNAME pattern)",
)
@click.pass_context
def cli(ctx, project_name):
    """Atomate2-SIESTA jobflow-remote management tools.

    Manage HPC workflow submission, job inspection, parameter tuning,
    and resource allocation for SIESTA calculations via jobflow-remote.

    Uses the same -p/--project-name convention as the 'jf' command.
    Config files are stored in ~/.jfremote/<project>.yaml.

    \b
    Quick start:
        atomate2siesta-jobflow-remote install          # Install jobflow-remote
        atomate2siesta-jobflow-remote -p mn5 setup     # Configure project
        atomate2siesta-jobflow-remote -p mn5 test      # Verify setup
        atomate2siesta-jobflow-remote -p mn5 info      # Show project details

    \b
    Job management:
        ... job inspect <ID>                           # View job details & FDF params
        ... job modify-db <ID> --param "kpts=[6,6,1]"  # Edit FDF params in DB
        ... job update-resources <ID> --nodes 2        # Change SLURM resources

    \b
    Operations:
        ... runner                                     # Show runner daemon commands
        ... download -f <FLOW_ID>                      # Download job outputs
    """
    # Store project_name in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["project_name"] = project_name


cli.add_command(install)
cli.add_command(setup)
cli.add_command(update)
cli.add_command(info)
cli.add_command(test)
cli.add_command(runner)
cli.add_command(download)


# Create job command group
@cli.group()
@click.pass_context
def job(ctx):
    """Inspect, modify, and tune jobflow-remote jobs.

    View job details, edit SIESTA FDF parameters, and update
    SLURM/PBS resource allocations directly in MongoDB.

    \b
    Examples:
        # Inspect job details and FDF parameters
        atomate2siesta-jobflow-remote -p mn5 job inspect 70
        atomate2siesta-jobflow-remote -p mn5 job inspect 70 --full

        # Modify SIESTA input parameters
        atomate2siesta-jobflow-remote -p mn5 job modify-db 70 \\
            --param "Spin=polarized" --param "kpts=[6,6,1]"

        # Update SLURM resources (nodes, cores, walltime, etc.)
        atomate2siesta-jobflow-remote -p mn5 job update-resources 70 \\
            --ntasks-per-node 64 --time "24:00:00"

        # Auto-allocate resources with cluster profile
        atomate2siesta-jobflow-remote -p mn5 job update-resources 70 \\
            --auto --cluster-profile mn5
    """
    # Context is already set by parent, just pass through


job.add_command(inspect)
job.add_command(modify_db)
job.add_command(recreate)
job.add_command(update_resources)

if __name__ == "__main__":
    cli()
