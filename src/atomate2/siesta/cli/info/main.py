#!/usr/bin/env python
"""
Main CLI entry point for atomate2siesta-info.

This tool provides an overview of:
- Available CLI tools
- Workflow types
- Features and capabilities
- Quick start examples
- Version information
"""

import click

from atomate2.siesta.cli.info.overview import show_overview
from atomate2.siesta.cli.info.tools import show_cli_tools
from atomate2.siesta.cli.info.workflows import show_workflows
from atomate2.siesta.cli.info.features import show_features
from atomate2.siesta.cli.info.examples import show_examples
from atomate2.siesta.cli.info.version import show_version


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Display atomate2siesta information and capabilities."""
    if ctx.invoked_subcommand is None:
        show_overview()


@cli.command()
def overview():
    """Show complete overview of atomate2siesta."""
    show_overview()


@cli.command()
def tools():
    """List all available CLI tools."""
    show_cli_tools()


@cli.command()
@click.argument("workflow_name", required=False)
@click.option("--list-all", is_flag=True, help="List all discovered FlowMaker classes")
@click.option("--full", is_flag=True, help="Show full documentation in Rich panel")
def workflows(workflow_name, list_all, full):
    """
    Show workflow information.

    \b
    Examples:
      atomate2siesta-info workflows                        # List all workflows
      atomate2siesta-info workflows phonon                 # Show phonon workflow details
      atomate2siesta-info workflows SiestaEosFlowMaker --full  # Show full docstring in Rich panel
      atomate2siesta-info workflows --list-all             # List all FlowMaker classes
    """
    if list_all:
        from atomate2.siesta.cli.info.workflow_details import list_all_flowmakers

        list_all_flowmakers()
    elif workflow_name:
        from atomate2.siesta.cli.info.workflow_details import show_workflow_details

        show_workflow_details(workflow_name, full=full)
    else:
        show_workflows()


@cli.command()
def features():
    """List all major features."""
    show_features()


@cli.command()
def examples():
    """Show quick start examples."""
    show_examples()


@cli.command()
def version():
    """Show version information."""
    show_version()


if __name__ == "__main__":
    cli()
