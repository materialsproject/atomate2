"""Module containing the atomate2 command-line interface."""

import click

from atomate2.cli.dev import dev


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Command-line interface for atomate2."""


cli.add_command(dev)
