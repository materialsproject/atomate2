"""CLI tool for generating atomate2siesta workflow scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from atomate2.siesta.cli.maker.templates import TEMPLATES

console = Console()


@click.group(invoke_without_command=True)
@click.option(
    "--interactive",
    is_flag=True,
    help="Interactive mode with guided prompts",
)
@click.pass_context
def cli(ctx, interactive):
    """Generate ready-to-run atomate2siesta workflow scripts."""
    if ctx.invoked_subcommand is None and interactive:
        run_interactive_mode()
    elif ctx.invoked_subcommand is None:
        # Show help if no command and not interactive
        click.echo(ctx.get_help())


def run_interactive_mode():
    """Run interactive mode with questionary prompts."""
    try:
        import questionary
        from questionary import Style
    except ImportError:
        console.print(
            "\n[red]Error:[/red] Interactive mode requires questionary package"
        )
        console.print("Install with: [cyan]pip install questionary[/cyan]\n")
        sys.exit(1)

    # Custom style for prompts
    custom_style = Style(
        [
            ("qmark", "fg:#673ab7 bold"),  # question mark
            ("question", "bold"),  # question text
            ("answer", "fg:#2196f3 bold"),  # user answer
            ("pointer", "fg:#673ab7 bold"),  # pointer used in select
            ("highlighted", "fg:#673ab7 bold"),  # selected choice
            ("selected", "fg:#2196f3"),  # selected option
            ("separator", "fg:#cc5454"),  # separator between choices
            ("instruction", ""),  # user instructions
            ("text", ""),  # plain text
        ]
    )

    console.print("\n[bold cyan]Welcome to atomate2siesta-maker![/bold cyan]")
    console.print("[dim]Interactive workflow generation[/dim]\n")

    # Step 1: Select workflow type
    workflows = {
        "relax": "Structure relaxation (fixed or variable cell)",
        "static": "Single-point energy calculation",
        "bands": "Electronic band structure",
        "dos": "Density of states",
        "phonon": "Phonon calculation with automatic plotting",
        "gruneisen": "Grüneisen parameters and thermal expansion",
        "qha": "Quasi-harmonic approximation",
        "eos": "Equation of state",
        "elastic": "Elastic constants",
        "bulk-modulus": "Bulk modulus calculation",
        "neb": "Nudged elastic band (transition states)",
        "surface": "Surface energy calculation",
        "adsorption": "Adsorption site scanning",
    }

    workflow_type = questionary.select(
        "Select workflow type:",
        choices=[
            questionary.Choice(title=f"{k}: {v}", value=k) for k, v in workflows.items()
        ],
        style=custom_style,
    ).ask()

    if workflow_type is None:  # User cancelled
        console.print("\n[yellow]Cancelled[/yellow]\n")
        return

    # Step 2: Get structure file
    while True:
        structure_file = questionary.path(
            "Structure file (CIF/POSCAR/etc.):",
            style=custom_style,
        ).ask()

        if structure_file is None:  # User cancelled
            console.print("\n[yellow]Cancelled[/yellow]\n")
            return

        # Strip whitespace from path
        structure_file = structure_file.strip()

        # Validate file exists and is a file
        if not structure_file:
            console.print("\n[red]Error:[/red] No file path provided\n")
            retry = questionary.confirm(
                "Try another file path?",
                default=True,
                style=custom_style,
            ).ask()
            if not retry:
                console.print("\n[yellow]Cancelled[/yellow]\n")
                return
            continue

        if not Path(structure_file).exists():
            console.print(f"\n[red]Error:[/red] File not found: {structure_file}")
            console.print(
                "\nTip: Use Tab key for path completion or provide absolute path\n"
            )

            retry = questionary.confirm(
                "Try another file path?",
                default=True,
                style=custom_style,
            ).ask()

            if not retry:
                console.print("\n[yellow]Cancelled[/yellow]\n")
                return
            continue

        if not Path(structure_file).is_file():
            console.print(f"\n[red]Error:[/red] Not a file: {structure_file}")
            console.print(
                "Please provide a path to a structure file (not a directory)\n"
            )

            retry = questionary.confirm(
                "Try another file path?",
                default=True,
                style=custom_style,
            ).ask()

            if not retry:
                console.print("\n[yellow]Cancelled[/yellow]\n")
                return
            continue

        # Valid file path
        console.print(f"[green]✓[/green] Valid file: {structure_file}")
        break

    # Step 3: Get tier preset (optional)
    preset_choices = [
        "None (use default parameters)",
        "relax_standard",
        "surface_metal",
        "adsorbate_screening",
        "2d_metal",
        "band_structure",
        "Other (enter manually)",
    ]

    preset_choice = questionary.select(
        "Use tier preset?",
        choices=preset_choices,
        default=preset_choices[0],
        style=custom_style,
    ).ask()

    preset = None
    if preset_choice and preset_choice != preset_choices[0]:
        if preset_choice == "Other (enter manually)":
            # Validate preset name
            from atomate2.siesta.sets.tiers import list_tier_presets

            available_presets = list_tier_presets()

            while True:
                preset = questionary.text(
                    "Enter preset name:",
                    style=custom_style,
                ).ask()

                if preset is None:  # User cancelled
                    console.print("\n[yellow]Cancelled[/yellow]\n")
                    return

                if preset:
                    preset = preset.strip()

                if not preset:  # Empty string - skip preset
                    preset = None
                    break

                # Validate preset exists
                if preset in available_presets:
                    console.print(f"[green]✓[/green] Valid preset: {preset}")
                    break
                else:
                    console.print(f"\n[red]Error:[/red] Preset '{preset}' not found")
                    console.print("\n[cyan]Available presets:[/cyan]")
                    for name in sorted(available_presets.keys())[:10]:
                        console.print(f"  • {name}")
                    if len(available_presets) > 10:
                        console.print(f"  ... and {len(available_presets) - 10} more")
                    console.print(
                        "\nTip: Run [cyan]atomate2siesta-presets list[/cyan] to see all"
                    )

                    retry = questionary.confirm(
                        "Try another preset name?",
                        default=True,
                        style=custom_style,
                    ).ask()

                    if not retry:
                        preset = None
                        break
        else:
            preset = preset_choice

    # Step 4: Execution mode
    exec_mode = questionary.select(
        "Execution mode:",
        choices=[
            "Local (run_locally) - Run on this machine",
            "Remote (jobflow-remote) - Submit to HPC cluster",
            "Dry-run - Preview only, don't run SIESTA",
        ],
        default="Local (run_locally) - Run on this machine",
        style=custom_style,
    ).ask()

    dry_run = "Dry-run" in exec_mode if exec_mode else False
    remote = "Remote" in exec_mode if exec_mode else False
    worker = None

    if remote:
        worker = questionary.text(
            "Jobflow-remote worker name:",
            default="default",
            style=custom_style,
        ).ask()
        if worker:
            worker = worker.strip()

    # Step 4.5: Database configuration
    use_database = questionary.confirm(
        "Save results to MongoDB?",
        default=False,
        style=custom_style,
    ).ask()

    db_config = None
    if use_database:
        db_method = questionary.select(
            "Database configuration method:",
            choices=[
                "Use jobflow.yaml (automatic)",
                "Explicit JobStore in script",
            ],
            default="Use jobflow.yaml (automatic)",
            style=custom_style,
        ).ask()

        if "Explicit" in db_method:
            db_host = questionary.text(
                "MongoDB host:",
                default="localhost",
                style=custom_style,
            ).ask()
            if db_host:
                db_host = db_host.strip()

            db_port = questionary.text(
                "MongoDB port:",
                default="27017",
                validate=lambda x: x.isdigit(),
                style=custom_style,
            ).ask()
            if db_port:
                db_port = db_port.strip()

            db_name = questionary.text(
                "Database name:",
                default="atomate2siesta",
                style=custom_style,
            ).ask()
            if db_name:
                db_name = db_name.strip()

            db_collection = questionary.text(
                "Collection name:",
                default="tasks",
                style=custom_style,
            ).ask()
            if db_collection:
                db_collection = db_collection.strip()

            db_config = {
                "method": "explicit",
                "host": db_host or "localhost",
                "port": int(db_port) if db_port else 27017,
                "database": db_name or "atomate2siesta",
                "collection": db_collection or "tasks",
            }
        else:
            db_config = {"method": "jobflow_yaml"}

    # Step 5: Workflow-specific parameters
    workflow_options = {}
    if workflow_type == "relax":
        cell_type = questionary.select(
            "Cell relaxation type:",
            choices=["fixed", "variable"],
            default="fixed",
            style=custom_style,
        ).ask()
        workflow_options["cell_type"] = cell_type

    elif workflow_type == "phonon":
        supercell_method = questionary.select(
            "Supercell generation method:",
            choices=["Automatic (min_length)", "Explicit matrix"],
            default="Automatic (min_length)",
            style=custom_style,
        ).ask()

        if "Automatic" in supercell_method:
            min_length = questionary.text(
                "Minimum supercell length (Å):",
                default="10.0",
                validate=lambda x: x.replace(".", "").isdigit(),
                style=custom_style,
            ).ask()
            workflow_options["min_length"] = float(min_length) if min_length else 10.0
        else:
            supercell = questionary.text(
                "Supercell matrix (e.g., 2 2 2):",
                default="2 2 2",
                style=custom_style,
            ).ask()
            if supercell:
                workflow_options["supercell"] = [int(x) for x in supercell.split()]

    elif workflow_type == "neb":
        console.print(
            "\n[yellow]Note:[/yellow] NEB requires two structure files (initial and final)"
        )
        final_structure = questionary.path(
            "Final structure file:",
            style=custom_style,
        ).ask()

        if final_structure:
            final_structure = final_structure.strip()

        if final_structure and Path(final_structure).exists():
            workflow_options["final_structure"] = final_structure
        else:
            console.print("\n[red]Error:[/red] Invalid final structure file\n")
            return

        num_images = questionary.text(
            "Number of intermediate images:",
            default="5",
            validate=lambda x: x.isdigit() and int(x) > 0,
            style=custom_style,
        ).ask()
        workflow_options["number_of_images"] = int(num_images) if num_images else 5

    elif workflow_type == "bands":
        kpath_density = questionary.text(
            "K-path density (k-points per Å⁻¹):",
            default="20",
            validate=lambda x: x.isdigit() and int(x) > 0,
            style=custom_style,
        ).ask()
        workflow_options["kpath_density"] = int(kpath_density) if kpath_density else 20

    elif workflow_type == "gruneisen":
        min_length = questionary.text(
            "Minimum supercell length (Å):",
            default="10.0",
            validate=lambda x: x.replace(".", "").replace("-", "").isdigit(),
            style=custom_style,
        ).ask()
        workflow_options["min_length"] = float(min_length) if min_length else 10.0

        displacement = questionary.text(
            "Atomic displacement (Å):",
            default="0.01",
            validate=lambda x: x.replace(".", "").replace("-", "").isdigit(),
            style=custom_style,
        ).ask()
        workflow_options["displacement"] = float(displacement) if displacement else 0.01

    elif workflow_type == "qha":
        min_length = questionary.text(
            "Minimum supercell length (Å):",
            default="10.0",
            validate=lambda x: x.replace(".", "").replace("-", "").isdigit(),
            style=custom_style,
        ).ask()
        workflow_options["min_length"] = float(min_length) if min_length else 10.0

        displacement = questionary.text(
            "Atomic displacement (Å):",
            default="0.01",
            validate=lambda x: x.replace(".", "").replace("-", "").isdigit(),
            style=custom_style,
        ).ask()
        workflow_options["displacement"] = float(displacement) if displacement else 0.01

    elif workflow_type == "eos":
        num_frames = questionary.text(
            "Number of volume frames:",
            default="7",
            validate=lambda x: x.isdigit() and int(x) > 0,
            style=custom_style,
        ).ask()
        workflow_options["number_of_frames"] = int(num_frames) if num_frames else 7

        strain_range = questionary.text(
            "Strain range (±fraction):",
            default="0.05",
            validate=lambda x: x.replace(".", "").replace("-", "").isdigit(),
            style=custom_style,
        ).ask()
        workflow_options["strain_range"] = float(strain_range) if strain_range else 0.05

    elif workflow_type == "elastic":
        strain_magnitude = questionary.text(
            "Strain magnitude:",
            default="0.005",
            validate=lambda x: x.replace(".", "").replace("-", "").isdigit(),
            style=custom_style,
        ).ask()
        workflow_options["strain_magnitude"] = (
            float(strain_magnitude) if strain_magnitude else 0.005
        )

    elif workflow_type == "surface":
        miller = questionary.text(
            "Miller indices (e.g., 1,1,1):",
            style=custom_style,
        ).ask()
        if miller:
            miller = miller.strip()
            workflow_options["miller_indices"] = miller

    elif workflow_type == "adsorption":
        grid_size = questionary.text(
            "Grid size (e.g., 3 3):",
            default="3 3",
            style=custom_style,
        ).ask()
        if grid_size:
            grid_size = grid_size.strip()
            workflow_options["grid_size"] = [int(x) for x in grid_size.split()]

        height = questionary.text(
            "Adsorbate height above surface (Å):",
            default="2.0",
            validate=lambda x: x.replace(".", "").isdigit(),
            style=custom_style,
        ).ask()
        if height:
            workflow_options["height"] = float(height)

    # Step 6: Output filename
    from pymatgen.core import Structure

    try:
        struct = Structure.from_file(structure_file)
        formula = struct.composition.reduced_formula
        default_output = f"{workflow_type}_{formula}.py"
    except Exception:
        default_output = f"{workflow_type}_workflow.py"

    output = questionary.text(
        "Output filename:",
        default=default_output,
        style=custom_style,
    ).ask()

    if output is None:
        output = default_output
    else:
        output = output.strip()

    # Build options dictionary
    options = {
        "output": output,
        "preset": preset,
        "dry_run": dry_run,
        "remote": remote,
        "worker": worker,
        "db_config": db_config,
        **workflow_options,
    }

    # Generate the workflow
    console.print(f"\n[cyan]Generating {workflow_type} workflow...[/cyan]")

    try:
        generate_workflow_script(workflow_type, structure_file, options)
    except Exception as e:
        console.print(f"\n[red]Error generating workflow:[/red] {e}")
        sys.exit(1)


# Common options decorator
def common_options(func):
    """Add common options to all workflow commands."""
    func = click.option(
        "-o",
        "--output",
        type=click.Path(),
        help="Output Python file (default: <workflow>_<formula>.py)",
    )(func)
    func = click.option(
        "--preset",
        help="Tier preset to apply (e.g., relax_standard, band_structure)",
    )(func)
    func = click.option(
        "--dry-run",
        is_flag=True,
        help="Generate script with dry_run=True for testing",
    )(func)
    func = click.option(
        "--remote",
        is_flag=True,
        help="Generate script for jobflow-remote submission",
    )(func)
    func = click.option(
        "--worker",
        default="default",
        help="Jobflow-remote worker name (requires --remote)",
    )(func)
    func = click.option(
        "--database",
        is_flag=True,
        help="Include MongoDB database configuration in script",
    )(func)
    func = click.option(
        "--db-host",
        default="localhost",
        help="MongoDB host (requires --database)",
    )(func)
    func = click.option(
        "--db-port",
        default=27017,
        type=int,
        help="MongoDB port (requires --database)",
    )(func)
    func = click.option(
        "--db-name",
        default="atomate2siesta",
        help="MongoDB database name (requires --database)",
    )(func)
    func = click.option(
        "--db-collection",
        default="tasks",
        help="MongoDB collection name (requires --database)",
    )(func)
    func = click.argument("structure_file", type=click.Path(exists=True))(func)
    return func


def generate_workflow_script(workflow_type, structure_file, options):
    """Common logic for generating workflow scripts."""
    # Build database config if database flag is set
    if options.get("database"):
        options["db_config"] = {
            "method": "explicit",
            "host": options.get("db_host", "localhost"),
            "port": options.get("db_port", 27017),
            "database": options.get("db_name", "atomate2siesta"),
            "collection": options.get("db_collection", "tasks"),
        }

    # Get template
    template = TEMPLATES[workflow_type]

    # Validate inputs
    try:
        template.validate_inputs(structure_file, options)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    # Generate output filename if not provided
    output = options.get("output")
    if not output:
        from pymatgen.core import Structure

        struct = Structure.from_file(structure_file)
        formula = struct.composition.reduced_formula
        output = f"{workflow_type}_{formula}.py"

    # Generate command string for header
    command_parts = [f"atomate2siesta-maker {workflow_type} {structure_file}"]
    if options.get("preset"):
        command_parts.append(f"--preset {options['preset']}")
    if options.get("dry_run"):
        command_parts.append("--dry-run")
    if options.get("remote"):
        command_parts.append(f"--remote --worker {options['worker']}")

    # Add workflow-specific options to command
    if workflow_type == "relax" and options.get("cell_type") != "fixed":
        command_parts.append(f"--cell-type {options['cell_type']}")
    if workflow_type == "bands" and options.get("kpath_density") != 20:
        command_parts.append(f"--kpath-density {options['kpath_density']}")
    if workflow_type == "phonon":
        supercell = options.get("supercell")
        if supercell:
            command_parts.append(f"--supercell {' '.join(map(str, supercell))}")
        if options.get("min_length") != 10.0:
            command_parts.append(f"--min-length {options['min_length']}")
        if options.get("displacement") != 0.01:
            command_parts.append(f"--displacement {options['displacement']}")
        if options.get("custom_params"):
            command_parts.append("--custom-params")
    if workflow_type == "adsorption":
        grid_size = options.get("grid_size", (3, 3))
        if grid_size != (3, 3):
            command_parts.append(f"--grid-size {' '.join(map(str, grid_size))}")
        if options.get("height", 2.0) != 2.0:
            command_parts.append(f"--height {options['height']}")
        if options.get("miller_indices") != "1,0,0":
            command_parts.append(f"--miller-indices {options['miller_indices']}")
        if options.get("adsorbate"):
            command_parts.append(f"--adsorbate {options['adsorbate']}")

    command = " ".join(command_parts)

    # Generate script
    console.print(f"\n[bold]Generating {workflow_type} workflow script...[/bold]")
    console.print(f"  Structure: {structure_file}")
    console.print(f"  Output: {output}")

    if options.get("preset"):
        console.print(f"  Preset: {options['preset']}")
    if options.get("dry_run"):
        console.print("  Mode: [yellow]Dry-run (preview only)[/yellow]")
    if options.get("remote"):
        console.print(f"  Remote worker: {options['worker']}")

    script_content = template.generate(structure_file, command, options)

    # Write script to file
    output_path = Path(output)
    output_path.write_text(script_content)

    # Make executable
    output_path.chmod(0o755)

    console.print(f"\n[green]✓[/green] Generated: {output}")
    console.print(f"\nRun with: [cyan]python {output}[/cyan]")

    # Show expected outputs
    if template.output_files:
        console.print("\n[bold]Expected output files:[/bold]")
        for file in template.output_files:
            console.print(f"  • {file}")


# Basic workflows
@cli.command()
@common_options
@click.option(
    "--cell-type",
    type=click.Choice(["fixed", "variable"]),
    default="fixed",
    help="Cell relaxation type",
)
def relax(
    structure_file,
    output,
    preset,
    dry_run,
    remote,
    worker,
    database,
    db_host,
    db_port,
    db_name,
    db_collection,
    cell_type,
):
    """Generate structure relaxation workflow.

    Performs atomic position optimization (fixed-cell) or full structural relaxation
    (variable-cell) to find the minimum energy configuration. Fixed-cell relaxation
    optimizes atomic positions while keeping lattice parameters constant - ideal for
    surfaces, interfaces, or when cell parameters are constrained. Variable-cell
    relaxation optimizes both atomic positions and lattice parameters simultaneously.

    Use fixed-cell (--cell-type fixed) for:
      • Surface slabs (preserve vacuum spacing)
      • Heterostructures (maintain interface geometry)
      • Adsorbate systems (slab should stay fixed)
      • Constrained geometries

    Use variable-cell (--cell-type variable) for:
      • Bulk materials (find equilibrium lattice)
      • Crystal structure optimization
      • Molecules (to optimize box size)

    The workflow automatically checks convergence and outputs relaxed structure.
    """
    options = {
        "output": output,
        "preset": preset,
        "dry_run": dry_run,
        "remote": remote,
        "worker": worker,
        "database": database,
        "db_host": db_host,
        "db_port": db_port,
        "db_name": db_name,
        "db_collection": db_collection,
        "cell_type": cell_type,
    }
    generate_workflow_script("relax", structure_file, options)


@cli.command()
@common_options
def static(structure_file, output, preset, dry_run, remote, worker):
    """Generate static (single-point energy) calculation workflow.

    Performs a single SCF calculation at fixed atomic positions without any geometry
    optimization. Computes total energy, forces, stress tensor, and electronic
    properties (e.g., Fermi energy, charge density) for a given structure.

    Use this workflow for:
      • Evaluating energy at a specific geometry
      • Post-processing relaxed structures
      • Computing forces/stress without relaxation
      • Quick energy comparisons between structures
      • Testing SIESTA parameters before production runs

    Typical workflow: relax → static (for accurate final energy with tight settings)

    This is the fastest calculation type - ideal for testing configurations.
    """
    options = {
        "output": output,
        "preset": preset,
        "dry_run": dry_run,
        "remote": remote,
        "worker": worker,
    }
    generate_workflow_script("static", structure_file, options)


@cli.command()
@common_options
@click.option(
    "--kpath-density",
    type=int,
    default=20,
    help="K-path density (k-points per Å⁻¹)",
)
def bands(structure_file, output, preset, dry_run, remote, worker, kpath_density):
    """Generate electronic band structure calculation workflow.

    Computes electronic band structure along high-symmetry k-point paths in the
    Brillouin zone. The workflow first performs a self-consistent calculation,
    then non-self-consistently computes eigenvalues along the k-path. Automatically
    determines the high-symmetry path based on crystal symmetry.

    Use this workflow for:
      • Identifying band gaps (direct vs indirect)
      • Understanding electronic structure
      • Analyzing semiconductor/insulator properties
      • Studying band dispersion and effective masses
      • Visualizing Fermi surface crossings

    Key parameter:
      --kpath-density: Controls k-point sampling density along the path (default: 20).
                      Higher values give smoother bands but increase computation time.

    Typical workflow: relax → bands (compute band structure of optimized geometry)

    Output: Band structure plot and eigenvalues at each k-point.
    """
    options = {
        "output": output,
        "preset": preset,
        "dry_run": dry_run,
        "remote": remote,
        "worker": worker,
        "kpath_density": kpath_density,
    }
    generate_workflow_script("bands", structure_file, options)


@cli.command()
@common_options
def dos(structure_file, output, preset, dry_run, remote, worker):
    """Generate total density of states (DOS) workflow.

    Computes the total electronic density of states - the number of electronic
    states per energy interval. DOS reveals the distribution of electronic states
    as a function of energy and is essential for understanding electronic properties.

    Use this workflow for:
      • Determining band gap and Fermi level position
      • Analyzing metallic vs semiconducting behavior
      • Studying electronic state distribution
      • Identifying peak positions (bonding/antibonding states)
      • Quick electronic structure overview

    The total DOS shows overall electronic structure. For element- or orbital-specific
    information, use the 'pdos' workflow instead.

    Typical workflow: relax → dos (analyze electronic structure of optimized geometry)

    Output: DOS plot (states/eV vs energy) with Fermi level marked.
    """
    options = {
        "output": output,
        "preset": preset,
        "dry_run": dry_run,
        "remote": remote,
        "worker": worker,
    }
    generate_workflow_script("dos", structure_file, options)


@cli.command()
@common_options
def pdos(structure_file, output, preset, dry_run, remote, worker):
    """Generate projected density of states (PDOS) workflow.

    Computes orbital- and atom-resolved density of states, showing the contribution
    of specific atoms or orbitals (s, p, d, f) to the total DOS. PDOS reveals which
    atoms and orbitals contribute to states at different energies - critical for
    understanding bonding, reactivity, and electronic structure.

    Use this workflow for:
      • Identifying orbital character near Fermi level
      • Understanding chemical bonding (which orbitals overlap)
      • Analyzing d-band centers (catalysis descriptors)
      • Studying magnetic properties (spin-polarized PDOS)
      • Determining atomic contributions to band edges

    Difference from 'dos': DOS shows total states, PDOS breaks down by atom/orbital.

    Typical workflow: relax → pdos (detailed electronic analysis of optimized structure)

    Output: PDOS plots showing contribution from each element and orbital type.
    """
    options = {
        "output": output,
        "preset": preset,
        "dry_run": dry_run,
        "remote": remote,
        "worker": worker,
    }
    generate_workflow_script("pdos", structure_file, options)


@cli.command()
@common_options
def optical(structure_file, output, preset, dry_run, remote, worker):
    """Generate optical properties calculation workflow.

    Computes frequency-dependent optical properties including dielectric function,
    absorption coefficient, reflectivity, and refractive index. Uses the independent
    particle approximation to calculate optical transitions between occupied and
    unoccupied states.

    Use this workflow for:
      • Solar cell materials (absorption spectrum analysis)
      • Transparent conductors (optical transparency windows)
      • Photocatalysts (light absorption characteristics)
      • Plasmonic materials (dielectric function)
      • Material identification (characteristic optical signatures)

    Key outputs:
      • ε(ω): Complex dielectric function (real and imaginary parts)
      • α(ω): Absorption coefficient
      • R(ω): Reflectivity
      • n(ω), k(ω): Complex refractive index

    Typical workflow: relax → optical (analyze light-matter interaction)

    Note: Uses independent particle approximation (no many-body effects like excitons).
    """
    options = {
        "output": output,
        "preset": preset,
        "dry_run": dry_run,
        "remote": remote,
        "worker": worker,
    }
    generate_workflow_script("optical", structure_file, options)


# Vibrational properties
@cli.command()
@common_options
@click.option(
    "--supercell",
    nargs=3,
    type=int,
    help="Supercell size (3 integers: nx ny nz)",
)
@click.option(
    "--min-length",
    type=float,
    default=10.0,
    help="Minimum supercell length (Å)",
)
@click.option(
    "--displacement",
    type=float,
    default=0.01,
    help="Atomic displacement (Å)",
)
@click.option(
    "--custom-params",
    is_flag=True,
    help="Use separate relax/force parameters",
)
def phonon(
    structure_file,
    output,
    preset,
    dry_run,
    remote,
    worker,
    supercell,
    min_length,
    displacement,
    custom_params,
):
    """Generate phonon (lattice dynamics) calculation workflow.

    Computes vibrational properties using the finite displacement method with
    phonopy integration. Calculates phonon dispersion, density of states, and
    thermodynamic properties (heat capacity, entropy, free energy). Automatically
    generates publication-quality plots.

    Use this workflow for:
      • Verifying structural stability (no imaginary frequencies)
      • Computing thermal properties (Cv, entropy, free energy)
      • Understanding lattice vibrations and phonon modes
      • Analyzing Raman/IR active modes
      • Temperature-dependent properties

    Key parameters:
      --supercell: Explicit supercell matrix (e.g., 2 2 2)
      --min-length: Auto-generate supercell with minimum dimension (default: 10 Å)
      --displacement: Atomic displacement for force calculations (default: 0.01 Å)
      --custom-params: Use separate tight settings for relax vs force calculations

    Typical workflow: relax → phonon (compute vibrations of optimized structure)

    Outputs: phonon_bands.png, phonon_dos.png, thermal_properties.png, phonopy.yaml
    """
    options = {
        "output": output,
        "preset": preset,
        "dry_run": dry_run,
        "remote": remote,
        "worker": worker,
        "supercell": supercell,  # Now a tuple of 3 ints or None
        "min_length": min_length,
        "displacement": displacement,
        "custom_params": custom_params,
    }
    generate_workflow_script("phonon", structure_file, options)


@cli.command()
@common_options
@click.option(
    "--min-length",
    type=float,
    default=10.0,
    help="Minimum supercell length (Å)",
)
@click.option(
    "--displacement",
    type=float,
    default=0.01,
    help="Atomic displacement (Å)",
)
def gruneisen(
    structure_file, output, preset, dry_run, remote, worker, min_length, displacement
):
    """Generate Grüneisen parameters and thermal expansion workflow.

    Computes mode-dependent Grüneisen parameters (γ) which describe how phonon
    frequencies change with volume. These parameters are essential for understanding
    thermal expansion, heat conduction, and anharmonicity. The workflow performs
    phonon calculations at multiple volumes (compressed, equilibrium, expanded).

    Use this workflow for:
      • Predicting thermal expansion coefficient (α_V)
      • Understanding anharmonic effects
      • Analyzing pressure dependence of phonons
      • Studying temperature-dependent lattice dynamics
      • Identifying soft modes sensitive to volume changes

    Key outputs:
      • Mode-resolved Grüneisen parameters γ(q,j) for each phonon mode
      • Volume dependence of phonon frequencies
      • Thermal expansion predictions
      • 6 publication-quality plots (dispersion, DOS, mesh, thermal, parameters, heatmap)

    Typical workflow: relax → gruneisen (analyze volume-dependent vibrations)

    Note: More expensive than phonon workflow (requires 3× phonon calculations).
    """
    options = {
        "output": output,
        "preset": preset,
        "dry_run": dry_run,
        "remote": remote,
        "worker": worker,
        "min_length": min_length,
        "displacement": displacement,
    }
    generate_workflow_script("gruneisen", structure_file, options)


@cli.command()
@common_options
@click.option(
    "--min-length",
    type=float,
    default=10.0,
    help="Minimum supercell length (Å)",
)
@click.option(
    "--displacement",
    type=float,
    default=0.01,
    help="Atomic displacement (Å)",
)
def qha(
    structure_file, output, preset, dry_run, remote, worker, min_length, displacement
):
    """Generate quasi-harmonic approximation (QHA) workflow.

    Computes temperature-dependent thermodynamic properties by combining phonon
    calculations at multiple volumes with equation of state fitting. QHA accounts
    for volume expansion with temperature while treating phonons harmonically at
    each volume - essential for finite-temperature phase diagrams and stability.

    Use this workflow for:
      • Temperature-dependent free energy F(T,V)
      • Thermal expansion α(T) predictions
      • Heat capacity at constant pressure Cp(T)
      • Bulk modulus temperature dependence B(T)
      • Phase transition temperatures
      • Finite-temperature equations of state

    How it works:
      1. Perform phonon calculations at 5-7 different volumes
      2. Compute free energy F(T,V) = E(V) + F_phonon(T,V)
      3. Minimize F(T,V) to find equilibrium volume at each temperature
      4. Extract thermodynamic properties from volume-temperature relationship

    Typical workflow: relax → qha (predict finite-temperature thermodynamics)

    Note: Computationally expensive (5-7× phonon calculations). High accuracy needed.
    """
    options = {
        "output": output,
        "preset": preset,
        "dry_run": dry_run,
        "remote": remote,
        "worker": worker,
        "min_length": min_length,
        "displacement": displacement,
    }
    generate_workflow_script("qha", structure_file, options)


# Mechanical properties
@cli.command()
@common_options
@click.option(
    "--number-of-frames",
    type=int,
    default=7,
    help="Number of volume points",
)
@click.option(
    "--strain-range",
    type=float,
    default=0.05,
    help="Strain range (±fraction)",
)
def eos(
    structure_file,
    output,
    preset,
    dry_run,
    remote,
    worker,
    number_of_frames,
    strain_range,
):
    """Generate equation of state (EOS) workflow.

    Computes energy vs volume relationship by performing calculations at multiple
    volumes (compressed and expanded) around equilibrium. Fits the E(V) curve to
    standard EOS models (Birch-Murnaghan, Vinet, etc.) to extract bulk modulus,
    equilibrium volume, and pressure derivatives.

    Use this workflow for:
      • Determining bulk modulus (B₀) and its pressure derivative (B₀')
      • Finding equilibrium lattice parameters
      • Understanding compressibility
      • Validating DFT accuracy (compare to experiment)
      • Studying pressure-induced phase transitions

    Key parameters:
      --number-of-frames: Number of volume points (default: 7, typical range: 5-11)
      --strain-range: Volume variation ±fraction (default: 0.05 = ±5%)

    Output:
      • E-V curve with multiple EOS fits
      • B₀, V₀, E₀ extracted from fitting
      • Comparison of different EOS models

    Typical workflow: Quick check for lattice parameters and bulk modulus.
    """
    options = {
        "output": output,
        "preset": preset,
        "dry_run": dry_run,
        "remote": remote,
        "worker": worker,
        "number_of_frames": number_of_frames,
        "strain_range": strain_range,
    }
    generate_workflow_script("eos", structure_file, options)


@cli.command()
@common_options
@click.option(
    "--strain-magnitude",
    type=float,
    default=0.005,
    help="Strain magnitude",
)
def elastic(structure_file, output, preset, dry_run, remote, worker, strain_magnitude):
    """Generate elastic constants (elastic tensor) workflow.

    Computes the full elastic tensor (C_ij) by applying small deformations to the
    structure and calculating the stress response. The elastic tensor completely
    describes the linear elastic behavior - how the material responds to applied
    stress or strain. Automatically derives mechanical properties from the tensor.

    Use this workflow for:
      • Full elastic tensor C_ij (all independent components)
      • Derived properties: bulk modulus, shear modulus, Young's modulus
      • Poisson's ratio and elastic anisotropy
      • Sound velocities and Debye temperature
      • Mechanical stability analysis (Born stability criteria)
      • Brittle vs ductile behavior (Pugh's ratio)

    Key parameter:
      --strain-magnitude: Deformation amplitude (default: 0.005 = 0.5%)
                         Smaller = more accurate but needs tighter convergence

    Output:
      • Full elastic tensor in Voigt notation
      • Derived mechanical properties
      • Eigenvalues (stability check)

    Typical workflow: relax → elastic (compute mechanical properties of optimized structure)
    """
    options = {
        "output": output,
        "preset": preset,
        "dry_run": dry_run,
        "remote": remote,
        "worker": worker,
        "strain_magnitude": strain_magnitude,
    }
    generate_workflow_script("elastic", structure_file, options)


@cli.command(name="bulk-modulus")
@common_options
def bulk_modulus(structure_file, output, preset, dry_run, remote, worker):
    """Generate bulk modulus calculation workflow (quick EOS variant).

    Performs a streamlined equation of state calculation optimized for rapid bulk
    modulus determination. Uses fewer volume points and faster convergence criteria
    compared to the full 'eos' workflow - ideal for high-throughput screening or
    quick material property estimation.

    Use this workflow for:
      • Quick bulk modulus estimates
      • High-throughput screening
      • Rapid compressibility assessment
      • Testing before running full EOS
      • When only B₀ is needed (not full E-V curve)

    Difference from 'eos' workflow:
      • bulk-modulus: Faster, fewer points, optimized for B₀ only
      • eos: More accurate, more points, full E(V) analysis with multiple fits

    Typical use: Preliminary screening or when computational resources are limited.

    Output: Bulk modulus B₀ and equilibrium volume V₀ (less detailed than 'eos').
    """
    options = {
        "output": output,
        "preset": preset,
        "dry_run": dry_run,
        "remote": remote,
        "worker": worker,
    }
    generate_workflow_script("bulk-modulus", structure_file, options)


# Surface/Catalysis
@cli.command()
@common_options
@click.option(
    "--slab-directory",
    default="./slabs",
    help="Directory containing slab structures",
)
@click.option(
    "--miller-indices",
    default="1,0,0",
    help="Miller indices (e.g., 1,0,0 or 1,1,1)",
)
@click.option(
    "--relax-slabs",
    is_flag=True,
    help="Relax slab structures before energy calculation",
)
def surface(
    structure_file,
    output,
    preset,
    dry_run,
    remote,
    worker,
    slab_directory,
    miller_indices,
    relax_slabs,
):
    """Generate surface energy calculation workflow.

    Computes surface formation energy for a specific Miller index by comparing the
    slab energy to bulk energy. Surface energy (γ) represents the energy cost of
    creating a surface - critical for understanding crystal morphology, catalytic
    activity, and material stability.

    Use this workflow for:
      • Surface stability analysis (Wulff construction)
      • Catalyst design (active facet identification)
      • Crystal growth predictions (preferred orientations)
      • Comparing different terminations
      • Adsorption energy calculations (reference state)

    Key parameters:
      --miller-indices: Surface orientation (e.g., 1,0,0 or 1,1,1)
      --relax-slabs: Optimize slab geometry before energy calculation
      --slab-directory: Directory containing pre-generated slab structures

    Calculation: γ = (E_slab - n × E_bulk) / (2A)
      where A is surface area and factor of 2 accounts for two surfaces.

    Typical workflow: Generate slab → surface energy calculation

    Output: Surface energy in eV/Å² or J/m² and relaxed slab structure.
    """
    options = {
        "output": output,
        "preset": preset,
        "dry_run": dry_run,
        "remote": remote,
        "worker": worker,
        "slab_directory": slab_directory,
        "miller_indices": miller_indices,
        "relax_slabs": relax_slabs,
    }
    generate_workflow_script("surface", structure_file, options)


@cli.command()
@common_options
@click.option(
    "--grid-size",
    nargs=2,
    type=int,
    default=(3, 3),
    help="Grid size for site scanning (2 integers: nx ny)",
)
@click.option(
    "--height",
    type=float,
    default=2.0,
    help="Initial adsorbate height above surface (Å)",
)
@click.option(
    "--miller-indices",
    default="1,0,0",
    help="Miller indices of the surface",
)
@click.option(
    "--adsorbate",
    help="Path to adsorbate molecule file (XYZ, CIF, etc.)",
)
def adsorption(
    structure_file,
    output,
    preset,
    dry_run,
    remote,
    worker,
    grid_size,
    height,
    miller_indices,
    adsorbate,
):
    """Generate adsorption site scanning workflow to find the best adsorption site.

    Systematically scans multiple grid positions on the surface to identify the most
    favorable adsorption site. Places the adsorbate at each grid point, performs
    energy calculation, and generates heatmaps showing adsorption energy landscape.
    This is the PRIMARY tool for discovering unknown adsorption sites.

    Use this workflow when:
      • You DON'T know where the adsorbate prefers to bind
      • Exploring different binding sites (atop, bridge, hollow, etc.)
      • Comparing adsorption at multiple surface locations
      • Screening adsorbate-surface combinations
      • Generating adsorption energy heatmaps

    Key parameters:
      --grid-size: Number of grid points (e.g., 3 3 for 3×3 grid = 9 sites)
      --height: Initial adsorbate height above surface (Å)
      --adsorbate: Path to adsorbate molecule file (XYZ, CIF, etc.)

    Workflow steps:
      1. Generate grid of positions across surface
      2. Place adsorbate at each position
      3. Relax or compute energy at each site
      4. Identify site with lowest energy (best site)
      5. Generate heatmap and site visualization

    Output: Heatmap PNG, best_structure.cif, adsorption_sites.png, summary.txt

    **After scanning, use 'adsorption-optimize' to refine geometry at best site.**
    """
    options = {
        "output": output,
        "preset": preset,
        "dry_run": dry_run,
        "remote": remote,
        "worker": worker,
        "grid_size": grid_size,
        "height": height,
        "miller_indices": miller_indices,
        "adsorbate": adsorbate,
    }
    generate_workflow_script("adsorption", structure_file, options)


@cli.command(name="multi-surface")
@common_options
@click.option(
    "--miller-list",
    help="List of Miller indices (e.g., '(1,0,0),(1,1,0),(1,1,1)')",
)
def multi_surface(structure_file, output, preset, dry_run, remote, worker, miller_list):
    """Generate multiple surface energy comparison workflow.

    Computes surface energies for multiple Miller indices simultaneously, enabling
    direct comparison of different crystal facets. Generates comparative plots and
    tables ranking surfaces by stability. Essential for understanding equilibrium
    crystal shape (Wulff construction) and identifying most stable facets.

    Use this workflow for:
      • Comparing multiple surface orientations (e.g., (100) vs (110) vs (111))
      • Wulff construction (equilibrium crystal shape prediction)
      • Identifying most stable facets for catalysis
      • Surface phase diagrams
      • High-throughput surface screening

    Key parameter:
      --miller-list: Comma-separated Miller indices (e.g., '(1,0,0),(1,1,0),(1,1,1)')

    Workflow process:
      1. Generate slab for each Miller index
      2. Relax each slab (optional)
      3. Compute surface energy for each
      4. Compare and rank by stability
      5. Generate comparative plots

    Output:
      • multi_surface_comparison.png (bar chart of surface energies)
      • multi_surface_analysis.txt (detailed comparison)
      • Individual surface energy results for each facet

    Typical use: Comprehensive surface stability analysis of a material.
    """
    options = {
        "output": output,
        "preset": preset,
        "dry_run": dry_run,
        "remote": remote,
        "worker": worker,
        "miller_list": miller_list,
    }
    generate_workflow_script("multi-surface", structure_file, options)


@cli.command(name="adsorption-optimize")
@common_options
@click.option(
    "--site",
    nargs=2,
    type=float,
    default=(0.5, 0.5),
    help="Adsorption site (fractional xy coordinates)",
)
@click.option(
    "--height",
    type=float,
    default=2.0,
    help="Initial height above surface (Å)",
)
@click.option(
    "--adsorbate",
    type=click.Path(exists=True),
    help="Adsorbate molecule file (XYZ, CIF, etc.)",
)
def adsorption_optimize(
    structure_file, output, preset, dry_run, remote, worker, site, height, adsorbate
):
    """Generate adsorption geometry optimization workflow at a KNOWN site.

    Performs detailed structural relaxation of an adsorbate at a specific,
    pre-determined adsorption site. Use this when you ALREADY KNOW where the
    adsorbate should be placed (e.g., from previous 'adsorption' scan or
    chemical intuition). Optimizes both adsorbate and surface atoms.

    Use this workflow when:
      • You ALREADY KNOW the preferred binding site
      • Refining geometry after 'adsorption' scan identified best site
      • Computing accurate adsorption energy at a specific site
      • Optimizing geometry at chemically-known sites (atop, bridge, hollow)
      • Following up on screening results

    Key parameters:
      --site: Fractional xy coordinates of adsorption site (e.g., 0.5 0.5)
      --height: Initial z-height above surface (Å)
      --adsorbate: Path to adsorbate molecule file

    Difference from 'adsorption' workflow:
      • adsorption: SCANS grid to FIND best site (many positions tested)
      • adsorption-optimize: OPTIMIZES geometry at ONE KNOWN site (single position)

    Typical workflow sequence:
      1. Run 'adsorption' to scan and identify best site → (0.33, 0.67)
      2. Run 'adsorption-optimize --site 0.33 0.67' to refine geometry

    Output: Optimized structure, adsorption energy, binding geometry analysis.

    **Use 'adsorption' first if you don't know the binding site!**
    """
    options = {
        "output": output,
        "preset": preset,
        "dry_run": dry_run,
        "remote": remote,
        "worker": worker,
        "site": site,
        "height": height,
        "adsorbate": adsorbate,
    }
    generate_workflow_script("adsorption-optimize", structure_file, options)


# Transition states
@cli.command()
@click.argument("initial_structure", type=click.Path(exists=True))
@click.argument("final_structure", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output Python file (default: <workflow>_<formula>.py)",
)
@click.option(
    "--preset",
    help="Tier preset to apply (e.g., relax_standard, band_structure)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Generate script with dry_run=True for testing",
)
@click.option(
    "--remote",
    is_flag=True,
    help="Generate script for jobflow-remote submission",
)
@click.option(
    "--worker",
    default="default",
    help="Jobflow-remote worker name (requires --remote)",
)
@click.option(
    "--number-of-images",
    type=int,
    default=5,
    help="Number of intermediate NEB images",
)
@click.option(
    "--relax-endpoints",
    is_flag=True,
    help="Relax initial and final structures before NEB",
)
@click.option(
    "--interpolation",
    type=click.Choice(["idpp", "linear"]),
    default="idpp",
    help="Interpolation method for generating images",
)
def neb(
    initial_structure,
    final_structure,
    output,
    preset,
    dry_run,
    remote,
    worker,
    number_of_images,
    relax_endpoints,
    interpolation,
):
    """Generate nudged elastic band (NEB) workflow for transition state search.

    Finds the minimum energy path (MEP) and transition state between two known
    configurations (initial and final). NEB generates intermediate images along
    the reaction path and optimizes them to find the saddle point - essential for
    understanding reaction mechanisms, diffusion barriers, and kinetics.

    Use this workflow for:
      • Finding activation barriers (E_a) for chemical reactions
      • Diffusion barrier calculations (ion/atom migration)
      • Reaction mechanism elucidation
      • Transition state identification
      • Computing reaction rate constants (Arrhenius prefactor)
      • Surface reaction pathways (catalysis)

    Key parameters:
      --number-of-images: Intermediate NEB images (default: 5, typical: 5-9)
      --relax-endpoints: Optimize initial/final structures before NEB
      --interpolation: Image generation method (idpp or linear, idpp recommended)

    Workflow process:
      1. Optionally relax initial and final structures
      2. Generate intermediate images by interpolation
      3. Optimize NEB path to find MEP
      4. Identify transition state (highest energy image)
      5. Extract activation barrier

    Output:
      • Energy vs reaction coordinate plot
      • Transition state structure
      • Activation barrier E_a
      • NEB summary with convergence info

    Note: Requires BOTH initial and final structure files as arguments.
    """
    options = {
        "output": output,
        "preset": preset,
        "dry_run": dry_run,
        "remote": remote,
        "worker": worker,
        "final_structure": final_structure,
        "number_of_images": number_of_images,
        "relax_endpoints": relax_endpoints,
        "interpolation": interpolation,
    }
    generate_workflow_script("neb", initial_structure, options)


@cli.command()
def list():
    """List all available workflow templates."""
    console.print("\n[bold]Available Workflow Templates[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Workflow", style="cyan")
    table.add_column("Description")
    table.add_column("Runtime")

    for name, template in TEMPLATES.items():
        table.add_row(name, template.description, template.runtime_estimate)

    console.print(table)

    console.print(
        "\n[bold]Usage:[/bold] atomate2siesta-maker [WORKFLOW] [STRUCTURE_FILE] [OPTIONS]"
    )
    console.print("[bold]Examples:[/bold]")
    console.print("  atomate2siesta-maker relax Si.cif")
    console.print("  atomate2siesta-maker eos Si.cif --number-of-frames 9")
    console.print("  atomate2siesta-maker phonon Si.cif --supercell 2 2 2")
    console.print("  atomate2siesta-maker adsorption slab.cif --grid-size 4 4")
    console.print("\n[bold]Help:[/bold] atomate2siesta-maker [WORKFLOW] --help\n")


if __name__ == "__main__":
    cli()
