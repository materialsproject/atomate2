"""Extract FDF parameters from SIESTA input files.

This module provides functionality to parse SIESTA FDF files and extract
all parameters in various output formats (Python dict, CLI flags, YAML, etc.).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()


def parse_fdf_file(fdf_path: str | Path) -> dict[str, Any]:
    """Parse SIESTA FDF file and extract all parameters.

    Parameters
    ----------
    fdf_path : str | Path
        Path to the SIESTA FDF file

    Returns
    -------
    dict[str, Any]
        Dictionary of extracted parameters

    Notes
    -----
    Handles:
    - Simple key-value pairs (Mesh.Cutoff 300 Ry)
    - Boolean flags (Spin polarized)
    - Block parameters (%block DM.InitSpin ... %endblock)
    - Comments (lines starting with #)
    - Include statements (%include file.fdf)
    """
    fdf_path = Path(fdf_path)
    if not fdf_path.exists():
        raise FileNotFoundError(f"FDF file not found: {fdf_path}")

    params: dict[str, Any] = {}
    in_block = False
    block_name: str | None = None
    block_lines: list[str] = []

    with open(fdf_path) as f:
        for line in f:
            # Remove inline comments
            if "#" in line:
                line = line[: line.index("#")]

            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Handle block start
            if line.lower().startswith("%block"):
                in_block = True
                block_name = line.split()[1]  # Get block name
                block_lines = []
                continue

            # Handle block end
            if line.lower().startswith("%endblock"):
                if in_block and block_name:
                    params[block_name] = block_lines
                in_block = False
                block_name = None
                continue

            # If inside block, accumulate lines
            if in_block:
                block_lines.append(line)
                continue

            # Handle %include directives (store as special key)
            if line.lower().startswith("%include"):
                include_file = (
                    line.split(maxsplit=1)[1] if len(line.split()) > 1 else ""
                )
                params.setdefault("_includes", []).append(include_file)
                continue

            # Parse regular key-value pairs
            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                # Boolean flag (e.g., "MD.UseSaveXV")
                params[parts[0]] = True
            elif len(parts) == 2:
                key, value = parts
                # Try to parse as number
                try:
                    # Try integer first
                    if "." not in value and "e" not in value.lower():
                        params[key] = int(value)
                    else:
                        params[key] = float(value)
                except ValueError:
                    # Keep as string (includes units like "300 Ry")
                    # Check for boolean values
                    if value.lower() in ("true", "yes", ".true.", "t"):
                        params[key] = True
                    elif value.lower() in ("false", "no", ".false.", "f"):
                        params[key] = False
                    else:
                        params[key] = value

    return params


def format_as_python_dict(params: dict[str, Any], indent: int = 4) -> str:
    """Format parameters as Python dictionary code.

    Parameters
    ----------
    params : dict
        Parameters dictionary
    indent : int
        Indentation spaces

    Returns
    -------
    str
        Formatted Python dict code
    """
    lines = ["{"]

    for key, value in params.items():
        # Skip internal keys
        if key.startswith("_"):
            continue

        # Format value based on type
        if isinstance(value, list):
            # Block parameters
            if len(value) == 1:
                formatted_value = f'["{value[0]}"]'
            else:
                formatted_value = "[\n" + " " * (indent + 4)
                formatted_value += (",\n" + " " * (indent + 4)).join(
                    f'"{line}"' for line in value
                )
                formatted_value += f"\n{' ' * indent}]"
        elif isinstance(value, str):
            formatted_value = f'"{value}"'
        elif isinstance(value, bool):
            formatted_value = str(value)
        else:
            formatted_value = str(value)

        lines.append(f'{" " * indent}"{key}": {formatted_value},')

    lines.append("}")
    return "\n".join(lines)


def format_as_cli_params(params: dict[str, Any]) -> list[str]:
    """Format parameters as CLI --param flags.

    Parameters
    ----------
    params : dict
        Parameters dictionary

    Returns
    -------
    list[str]
        List of --param "key=value" strings
    """
    cli_params = []

    for key, value in params.items():
        # Skip internal keys
        if key.startswith("_"):
            continue

        # Format value based on type
        if isinstance(value, list):
            # Block parameters - convert to dict format for fdf_arguments
            formatted_value = "{" + f'"{key}": {value}' + "}"
            cli_params.append(f"--param 'fdf_arguments={formatted_value}'")
        elif isinstance(value, str) or isinstance(value, bool):
            cli_params.append(f'--param "{key}={value}"')
        else:
            cli_params.append(f'--param "{key}={value}"')

    return cli_params


def format_as_yaml(params: dict[str, Any]) -> str:
    """Format parameters as YAML.

    Parameters
    ----------
    params : dict
        Parameters dictionary

    Returns
    -------
    str
        YAML formatted string
    """
    lines = []

    for key, value in params.items():
        # Skip internal keys
        if key.startswith("_"):
            continue

        if isinstance(value, list):
            # Block parameters
            lines.append(f"{key}:")
            for line in value:
                lines.append(f"  - {line}")
        elif isinstance(value, str):
            lines.append(f'{key}: "{value}"')
        elif isinstance(value, bool):
            lines.append(f"{key}: {str(value).lower()}")
        else:
            lines.append(f"{key}: {value}")

    return "\n".join(lines)


def format_as_maker_code(params: dict[str, Any]) -> str:
    """Format parameters as Maker initialization code.

    Parameters
    ----------
    params : dict
        Parameters dictionary

    Returns
    -------
    str
        Python code for Maker initialization
    """
    # Separate FDF params from special params
    fdf_params = {}
    special_params = {}

    for key, value in params.items():
        if key.startswith("_"):
            continue
        if key in ["a2s_kpts", "a2s_magnetic_ordering", "a2s_dm_init_spin_format"]:
            special_params[key] = value
        else:
            fdf_params[key] = value

    lines = [
        "from atomate2.siesta.jobs.core import RelaxMaker",
        "",
        "# Create maker with extracted parameters",
        "maker = RelaxMaker.fixed_cell_relaxation(",
        "    user_params={",
    ]

    # Add FDF parameters
    for key, value in fdf_params.items():
        if isinstance(value, list):
            # Block parameters
            formatted_value = "[\n            "
            formatted_value += ",\n            ".join(f'"{line}"' for line in value)
            formatted_value += ",\n        ]"
            lines.append(f'        "{key}": {formatted_value},')
        elif isinstance(value, str):
            lines.append(f'        "{key}": "{value}",')
        else:
            lines.append(f'        "{key}": {value},')

    lines.append("    }")
    lines.append(")")

    return "\n".join(lines)


@click.command()
@click.argument("fdf_file", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    type=click.Choice(
        ["dict", "cli", "yaml", "maker", "json", "table"], case_sensitive=False
    ),
    default="dict",
    help="Output format (dict, cli, yaml, maker, json, table)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file (default: print to console)",
)
@click.option(
    "--filter",
    "-F",
    multiple=True,
    help="Filter parameters by key pattern (can use multiple times)",
)
@click.option(
    "--exclude-blocks",
    is_flag=True,
    help="Exclude block parameters from output",
)
def extract(
    fdf_file: str,
    format: str,
    output: str | None,
    filter: tuple[str],
    exclude_blocks: bool,
):
    """Extract parameters from SIESTA FDF file.

    Parse a SIESTA FDF file and extract all parameters in various formats:

    \b
    - dict: Python dictionary format
    - cli: CLI --param flags (for modify_db)
    - yaml: YAML format
    - maker: Python Maker initialization code
    - json: JSON format
    - table: Rich table display

    Examples
    --------
    \b
        # Extract as Python dict
        atomate2siesta-inputs extract siesta.fdf

    \b
        # Extract as CLI parameters
        atomate2siesta-inputs extract siesta.fdf --format cli

    \b
        # Extract as Maker code
        atomate2siesta-inputs extract siesta.fdf --format maker

    \b
        # Filter specific parameters
        atomate2siesta-inputs extract siesta.fdf --filter "Mesh*" --filter "SCF*"

    \b
        # Save to file
        atomate2siesta-inputs extract siesta.fdf --format yaml -o params.yaml
    """
    try:
        # Parse FDF file
        console.print(f"[cyan]Parsing FDF file:[/cyan] {fdf_file}")
        params = parse_fdf_file(fdf_file)

        # Apply filters if specified
        if filter:
            filtered_params = {}
            for pattern in filter:
                pattern_re = re.compile(pattern.replace("*", ".*"))
                for key, value in params.items():
                    if pattern_re.match(key):
                        filtered_params[key] = value
            params = filtered_params

        # Exclude blocks if requested
        if exclude_blocks:
            params = {k: v for k, v in params.items() if not isinstance(v, list)}

        # Count parameters
        n_params = len([k for k in params.keys() if not k.startswith("_")])
        n_blocks = len([v for v in params.values() if isinstance(v, list)])

        console.print(
            f"[green]✓[/green] Found {n_params} parameters ({n_blocks} blocks)\n"
        )

        # Format output based on format choice
        if format == "dict":
            formatted = format_as_python_dict(params)
            syntax = Syntax(formatted, "python", theme="monokai", line_numbers=False)
            if output:
                Path(output).write_text(formatted)
                console.print(f"[green]✓[/green] Saved to: {output}")
            else:
                console.print(
                    Panel(
                        syntax,
                        title="Python Dictionary",
                        border_style="blue",
                    )
                )

        elif format == "cli":
            cli_params = format_as_cli_params(params)
            formatted = "\\\n    ".join(cli_params)
            if output:
                Path(output).write_text("\n".join(cli_params))
                console.print(f"[green]✓[/green] Saved to: {output}")
            else:
                console.print(
                    Panel(
                        formatted,
                        title="CLI Parameters (for modify_db)",
                        border_style="green",
                    )
                )
                console.print("\n[yellow]Usage example:[/yellow]")
                console.print(
                    "atomate2siesta-jobflow-remote job modify-db <job_id> \\\n    "
                    + formatted
                )

        elif format == "yaml":
            formatted = format_as_yaml(params)
            if output:
                Path(output).write_text(formatted)
                console.print(f"[green]✓[/green] Saved to: {output}")
            else:
                syntax = Syntax(formatted, "yaml", theme="monokai", line_numbers=False)
                console.print(
                    Panel(
                        syntax,
                        title="YAML Format",
                        border_style="cyan",
                    )
                )

        elif format == "maker":
            formatted = format_as_maker_code(params)
            if output:
                Path(output).write_text(formatted)
                console.print(f"[green]✓[/green] Saved to: {output}")
            else:
                syntax = Syntax(formatted, "python", theme="monokai", line_numbers=True)
                console.print(
                    Panel(
                        syntax,
                        title="Maker Initialization Code",
                        border_style="magenta",
                    )
                )

        elif format == "json":
            import json

            # Convert to JSON-serializable format
            json_params = {}
            for key, value in params.items():
                if key.startswith("_"):
                    continue
                json_params[key] = value

            formatted = json.dumps(json_params, indent=2)
            if output:
                Path(output).write_text(formatted)
                console.print(f"[green]✓[/green] Saved to: {output}")
            else:
                syntax = Syntax(formatted, "json", theme="monokai", line_numbers=False)
                console.print(
                    Panel(
                        syntax,
                        title="JSON Format",
                        border_style="yellow",
                    )
                )

        elif format == "table":
            # Create rich table
            table = Table(title=f"Parameters from {Path(fdf_file).name}")
            table.add_column("Parameter", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")
            table.add_column("Type", style="magenta")

            for key, value in params.items():
                if key.startswith("_"):
                    continue

                if isinstance(value, list):
                    value_str = f"[{len(value)} lines]"
                    type_str = "block"
                elif isinstance(value, bool):
                    value_str = str(value)
                    type_str = "bool"
                elif isinstance(value, (int, float)):
                    value_str = str(value)
                    type_str = "number"
                else:
                    value_str = str(value)
                    type_str = "string"

                table.add_row(key, value_str, type_str)

            console.print(table)

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error parsing FDF file:[/red] {e}")
        console.print("[yellow]Make sure the file is a valid SIESTA FDF file.[/yellow]")
        raise click.Abort()


if __name__ == "__main__":
    extract()
