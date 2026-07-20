"""Automatic workflow discovery and detailed information display."""

import inspect

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def discover_flowmakers():
    """
    Automatically discover all FlowMaker classes.

    Returns
    -------
    dict
        Mapping of workflow_name -> FlowMaker class info
    """
    from atomate2.siesta.flows import (
        bands,
        convergence,
        core,
        elastic,
        eos,
        phonon,
        surface,
    )
    from atomate2.siesta.flows.neb import ase_neb, direct, vacancy_exchange
    from atomate2.siesta.flows.phonon import gruneisen, phonopy_maker, qha
    from atomate2.siesta.flows.surface import adsorption, multi_surface

    flowmakers = {}

    # List of modules to scan
    modules = [
        eos,
        convergence,
        core,
        elastic,
        surface,
        bands,
        phonon,
        gruneisen,
        qha,
        phonopy_maker,
        adsorption,
        multi_surface,
        vacancy_exchange,
        ase_neb,
        direct,
    ]

    for module in modules:
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Only include classes that end with FlowMaker
            if name.endswith("FlowMaker") and name != "BaseSiestaFlowMaker":
                # Get module path for context
                module_name = module.__name__.split(".")[-1]

                # Extract docstring
                docstring = inspect.getdoc(obj) or "No description available"
                first_line = docstring.split("\n")[0]

                # Get __init__ signature to understand inputs
                try:
                    sig = inspect.signature(obj.__init__)
                    params = sig.parameters
                except Exception:
                    params = {}

                flowmakers[name] = {
                    "class": obj,
                    "module": module_name,
                    "description": first_line,
                    "docstring": docstring,
                    "parameters": params,
                    "full_name": f"{obj.__module__}.{name}",
                }

    return flowmakers


def discover_makers():
    """
    Automatically discover all single-job Maker classes.

    Returns
    -------
    dict
        Mapping of Maker name -> Maker class info
    """
    from atomate2.siesta.jobs import core

    makers = {}

    for name, obj in inspect.getmembers(core, inspect.isclass):
        # Only include classes that end with Maker (but not FlowMaker)
        if (
            name.endswith("Maker")
            and not name.endswith("FlowMaker")
            and name != "BaseSiestaMaker"
        ):
            # Extract docstring
            docstring = inspect.getdoc(obj) or "No description available"
            first_line = docstring.split("\n")[0]

            # Get __init__ signature
            try:
                sig = inspect.signature(obj.__init__)
                params = sig.parameters
            except Exception:
                params = {}

            # Find classmethod constructors
            classmethods = []
            for method_name in dir(obj):
                if method_name.startswith("_"):
                    continue
                try:
                    method = getattr(obj, method_name)
                    # Check if it's a classmethod
                    if isinstance(
                        inspect.getattr_static(obj, method_name), classmethod
                    ):
                        # Get the signature
                        sig = inspect.signature(method)
                        # Check if return type matches this Maker
                        if sig.return_annotation != inspect.Signature.empty:
                            return_type = sig.return_annotation
                            # Handle both direct class and string annotations
                            if (
                                hasattr(return_type, "__name__")
                                and return_type.__name__ == name
                            ) or (isinstance(return_type, str) and return_type == name):
                                classmethods.append(method_name)
                except Exception:
                    pass

            makers[name] = {
                "class": obj,
                "module": "jobs.core",
                "description": first_line,
                "docstring": docstring,
                "parameters": params,
                "full_name": f"{obj.__module__}.{name}",
                "classmethods": classmethods,
            }

    return makers


def get_workflow_name_mapping():
    """
    Map friendly workflow names to FlowMaker classes.

    This mapping connects CLI workflow names (like 'phonon', 'eos')
    to their FlowMaker classes.

    Returns
    -------
    dict
        Mapping of friendly_name -> FlowMaker info
    """
    flowmakers = discover_flowmakers()

    # Create reverse mapping from friendly names
    name_mapping = {
        # Basic (these are single jobs, not flows)
        "relax": {
            "type": "job",
            "maker": "RelaxMaker",
            "description": "Structure relaxation (fixed or variable cell)",
            "module": "jobs.core",
        },
        "static": {
            "type": "job",
            "maker": "StaticMaker",
            "description": "Single-point energy calculation",
            "module": "jobs.core",
        },
        "bands": {
            "type": "job",
            "maker": "BandStructureMaker",
            "description": "Electronic band structure",
            "module": "jobs.core",
        },
        "dos": {
            "type": "job",
            "maker": "DOSMaker/PDOSMaker",
            "description": "Density of states (total or projected)",
            "module": "jobs.core",
        },
        # Flows - automatically discovered
        "phonon": flowmakers.get("SiestaPhononFlowMaker", {}),
        "gruneisen": flowmakers.get("SiestaGruneisenFlowMaker", {}),
        "qha": flowmakers.get("SiestaQhaFlowMaker", {}),
        "eos": flowmakers.get("SiestaEosFlowMaker", {}),
        "elastic": flowmakers.get("ElasticFlowMaker", {}),
        "surface": flowmakers.get("SurfaceEnergyFlowMaker", {}),
        "multi-surface": flowmakers.get("MultiSurfaceEnergyFlowMaker", {}),
        "adsorption": flowmakers.get("AdsorptionScanFlowMaker", {}),
        "neb": flowmakers.get(
            "NebDirectFlowMaker", {}
        ),  # Could also be AseNebFlowMaker
    }

    return name_mapping, flowmakers


def show_workflow_details(workflow_name: str, full: bool = False):
    """
    Display detailed information about a specific workflow.

    Automatically extracts information from FlowMaker class:
    - Docstring for description
    - __init__ parameters for inputs
    - Class attributes for configuration

    Parameters
    ----------
    workflow_name : str
        Name of workflow (e.g., 'phonon', 'eos', 'neb')
        Can be friendly name ('phonon') or real class name ('SiestaPhononFlowMaker' or 'BandStructureMaker')
    full : bool, optional
        If True, display full docstring in Rich panel (default: False)
    """
    name_mapping, all_flowmakers = get_workflow_name_mapping()
    all_makers = discover_makers()

    # Check if it's a Maker class name directly
    if workflow_name in all_makers:
        info = all_makers[workflow_name]
        is_job = True
    # Check if it's a FlowMaker class name directly
    elif workflow_name in all_flowmakers:
        info = all_flowmakers[workflow_name]
        is_job = False
    elif workflow_name in name_mapping:
        info = name_mapping[workflow_name]
        is_job = info.get("type") == "job"
    else:
        console.print(f"\n[red]Error:[/red] Unknown workflow '{workflow_name}'")
        console.print("\n[dim]Available Single Job Makers:[/dim]")
        for name in sorted(all_makers.keys()):
            console.print(f"  - {name}")
        console.print("\n[dim]Available FlowMakers:[/dim]")
        for name in sorted(all_flowmakers.keys()):
            console.print(f"  - {name}")
        console.print("\n[dim]Available friendly names:[/dim]")
        for name in sorted(name_mapping.keys()):
            console.print(f"  - {name}")
        console.print()
        return

    console.print()

    # If --full flag is set, display the full docstring in Rich panel
    if full:
        from atomate2.siesta.utils.common import print_docstring_in_box

        maker_class = info.get("class")
        if maker_class and maker_class.__doc__:
            print_docstring_in_box(maker_class.__doc__, title=maker_class.__name__)
            console.print()
            return
        console.print(
            "[yellow]Warning:[/yellow] No docstring available for this workflow\n"
        )
        # Fall through to show standard details

    if is_job:
        # Single job maker
        title = f"Workflow: {workflow_name}"

        # Check if we have the class or just metadata
        maker_class = info.get("class")
        if maker_class:
            # Auto-discovered Maker
            class_name = maker_class.__name__
            description = info.get("description", "No description")
            full_path = info.get("full_name", "Unknown")

            header = Panel(
                f"[bold cyan]{workflow_name.title()}[/bold cyan]\n"
                f"[dim]Type:[/dim] [yellow]Single Job[/yellow]\n"
                f"[dim]Maker:[/dim] [green]{class_name}[/green]\n"
                f"[dim]Module:[/dim] [blue]{full_path}[/blue]\n\n"
                f"{description}",
                title=f"[bold]{title}[/bold]",
                border_style="cyan",
                box=box.DOUBLE,
            )
            console.print(header)
            console.print()

            # Extract parameters from __init__ signature
            params = info.get("parameters", {})
            if params:
                params_table = Table(
                    title="📥 Maker Parameters",
                    box=box.ROUNDED,
                    show_header=True,
                    header_style="bold magenta",
                )
                params_table.add_column("Parameter", style="cyan", width=25)
                params_table.add_column("Type", style="yellow", width=20)
                params_table.add_column("Default", style="green", width=20)

                for param_name, param in params.items():
                    if param_name in ("self", "args", "kwargs"):
                        continue

                    # Get type annotation
                    param_type = "Any"
                    if param.annotation != inspect.Parameter.empty:
                        if hasattr(param.annotation, "__name__"):
                            param_type = param.annotation.__name__
                        else:
                            param_type = str(param.annotation).replace("typing.", "")

                    # Get default value
                    default = "required"
                    if param.default != inspect.Parameter.empty:
                        default = str(param.default)
                        if len(default) > 20:
                            default = default[:17] + "..."

                    params_table.add_row(param_name, param_type, default)

                console.print(params_table)

            # Show docstring if available
            docstring = info.get("docstring", "")
            if docstring and len(docstring) > len(description):
                console.print()
                console.print("[bold]Full Documentation:[/bold]")
                # Show first 10 lines of docstring
                lines = docstring.split("\n")[:10]
                for line in lines:
                    console.print(f"[dim]{line}[/dim]")
                if len(docstring.split("\n")) > 10:
                    console.print("[dim]...[/dim]")
        else:
            # Legacy metadata-based Maker
            maker_name = info.get("maker", "Unknown")
            description = info.get("description", "No description")
            module_path = f"atomate2.siesta.{info.get('module', 'jobs.core')}"

            header = Panel(
                f"[bold cyan]{workflow_name.title()}[/bold cyan]\n"
                f"[dim]Type:[/dim] [yellow]Single Job[/yellow]\n"
                f"[dim]Maker:[/dim] [green]{maker_name}[/green]\n"
                f"[dim]Module:[/dim] [blue]{module_path}[/blue]\n\n"
                f"{description}",
                title=f"[bold]{title}[/bold]",
                border_style="cyan",
                box=box.DOUBLE,
            )
            console.print(header)

    else:
        # Flow maker - extract from class
        flowmaker_class = info.get("class")
        if not flowmaker_class:
            console.print(
                f"[red]Error:[/red] FlowMaker class not found for {workflow_name}"
            )
            return

        class_name = flowmaker_class.__name__
        description = info.get("description", "No description")
        full_path = info.get("full_name", "Unknown")

        header = Panel(
            f"[bold cyan]{workflow_name.title()}[/bold cyan]\n"
            f"[dim]Type:[/dim] [yellow]Multi-Step Flow[/yellow]\n"
            f"[dim]FlowMaker:[/dim] [green]{class_name}[/green]\n"
            f"[dim]Module:[/dim] [blue]{full_path}[/blue]\n\n"
            f"{description}",
            title=f"[bold]Workflow: {workflow_name}[/bold]",
            border_style="cyan",
            box=box.DOUBLE,
        )
        console.print(header)
        console.print()

        # Extract parameters from __init__ signature
        params = info.get("parameters", {})
        if params:
            params_table = Table(
                title="📥 FlowMaker Parameters",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta",
            )
            params_table.add_column("Parameter", style="cyan", width=25)
            params_table.add_column("Type", style="yellow", width=20)
            params_table.add_column("Default", style="green", width=20)

            for param_name, param in params.items():
                if param_name in ("self", "args", "kwargs"):
                    continue

                # Get type annotation
                param_type = "Any"
                if param.annotation != inspect.Parameter.empty:
                    if hasattr(param.annotation, "__name__"):
                        param_type = param.annotation.__name__
                    else:
                        param_type = str(param.annotation).replace("typing.", "")

                # Get default value
                default = "required"
                if param.default != inspect.Parameter.empty:
                    default = str(param.default)
                    if len(default) > 20:
                        default = default[:17] + "..."

                params_table.add_row(param_name, param_type, default)

            console.print(params_table)

        # Show docstring if available
        docstring = info.get("docstring", "")
        if docstring and len(docstring) > len(description):
            console.print()
            console.print("[bold]Full Documentation:[/bold]")
            # Show first 10 lines of docstring
            lines = docstring.split("\n")[:10]
            for line in lines:
                console.print(f"[dim]{line}[/dim]")
            if len(docstring.split("\n")) > 10:
                console.print("[dim]...[/dim]")

    console.print()

    # Quick start example
    console.print("[bold]Quick Start:[/bold]")
    console.print(
        f"[green]$ atomate2siesta-maker {workflow_name} structure.cif[/green]"
    )
    console.print()

    # Show Python API usage
    console.print("[bold]Python API:[/bold]")
    if is_job:
        # Check if we have auto-discovered class info
        if maker_class:
            class_name = maker_class.__name__
            classmethods = info.get("classmethods", [])

            # Show all classmethods if available
            if classmethods:
                console.print(
                    f"[cyan]from atomate2.siesta.jobs.core import {class_name}[/cyan]"
                )
                console.print("[cyan]from pymatgen.core import Structure[/cyan]")
                console.print("[cyan]from jobflow import run_locally[/cyan]")
                console.print()
                console.print(
                    "[cyan]structure = Structure.from_file('structure.cif')[/cyan]"
                )

                if len(classmethods) == 1:
                    # Single classmethod
                    method_name = classmethods[0]
                    console.print(
                        f"[cyan]job = {class_name}.{method_name}(dry_run=True).make(structure)[/cyan]"
                    )
                else:
                    # Multiple classmethods - show all options
                    console.print("[cyan]# Available classmethods:[/cyan]")
                    for method_name in classmethods:
                        console.print(
                            f"[cyan]job = {class_name}.{method_name}(dry_run=True).make(structure)[/cyan]"
                        )
                    console.print()
                    console.print("[dim]# Use one of the above methods[/dim]")

                console.print(
                    "[cyan]results = run_locally(job, create_folders=True)[/cyan]"
                )
            else:
                console.print(
                    f"[cyan]from atomate2.siesta.jobs.core import {class_name}[/cyan]"
                )
                console.print("[cyan]from pymatgen.core import Structure[/cyan]")
                console.print()
                console.print(f"[cyan]maker = {class_name}()[/cyan]")
                console.print("[cyan]job = maker.make(structure)[/cyan]")
        else:
            # Legacy metadata-based
            maker_name = info.get("maker", "Maker")
            console.print(
                f"[cyan]from atomate2.siesta.jobs.core import {maker_name}[/cyan]"
            )
            console.print(f"[cyan]job = {maker_name}().make(structure)[/cyan]")
    else:
        class_name = flowmaker_class.__name__
        module_path = info.get("full_name", "").rsplit(".", 1)[0]
        console.print(f"[cyan]from {module_path} import {class_name}[/cyan]")
        console.print("[cyan]from pymatgen.core import Structure[/cyan]")
        console.print("[cyan]from jobflow import run_locally[/cyan]")
        console.print()
        console.print("[cyan]structure = Structure.from_file('structure.cif')[/cyan]")
        console.print(f"[cyan]flow = {class_name}().make(structure)[/cyan]")
        console.print("[cyan]results = run_locally(flow, create_folders=True)[/cyan]")
    console.print()


def list_all_flowmakers():
    """List all discovered FlowMaker classes."""
    _, flowmakers = get_workflow_name_mapping()

    console.print("\n[bold cyan]All Discovered FlowMakers[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("FlowMaker Class", style="cyan", width=40)
    table.add_column("Module", style="yellow", width=20)
    table.add_column("Description", style="white")

    for name, info in sorted(flowmakers.items()):
        module = info.get("module", "unknown")
        desc = info.get("description", "No description")
        if len(desc) > 60:
            desc = desc[:57] + "..."
        table.add_row(name, module, desc)

    console.print(table)
    console.print(
        f"\n[dim]Total: {len(flowmakers)} FlowMaker classes discovered[/dim]\n"
    )
