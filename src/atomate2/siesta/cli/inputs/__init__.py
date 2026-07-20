#!/usr/bin/env python
import inspect
import re
from dataclasses import fields
from typing import get_type_hints

import click
from rich.box import ROUNDED  # Changed import to ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Import extract command
from atomate2.siesta.cli.inputs.extract import extract
from atomate2.siesta.dataclass.auxiliary_force_field import AuxiliaryForceField
from atomate2.siesta.dataclass.basis_sets_and_projectors import BasisSetsAndProjectors
from atomate2.siesta.dataclass.charge_dipole_electric_field import (
    ChargeDipoleElectricField,
)
from atomate2.siesta.dataclass.chemical_analysis import ChemicalAnalysis
from atomate2.siesta.dataclass.denchar import Denchar
from atomate2.siesta.dataclass.density_of_states_and_band_structure import (
    DensityOfStatesAndBandStructure,
)
from atomate2.siesta.dataclass.dftu import DFTU
from atomate2.siesta.dataclass.efficiency_options import EfficiencyOptions
from atomate2.siesta.dataclass.electronic_structure_calculation_options import (
    ElectronicStructureCalculationOptions,
)
from atomate2.siesta.dataclass.exchange_correlation_functionals import (
    ExchangeCorrelationFunctionals,
)
from atomate2.siesta.dataclass.external_control_and_scripting import (
    ExternalControlAndScripting,
)
from atomate2.siesta.dataclass.general_constraints import GeneralConstraints
from atomate2.siesta.dataclass.general_system_descriptors import (
    GeneralSystemDescriptors,
)
from atomate2.siesta.dataclass.grids import Grids
from atomate2.siesta.dataclass.hamiltonian_and_overlap_parameters import (
    HamiltonianAndOverlapParameters,
)
from atomate2.siesta.dataclass.kpoint_sampling import KPointSampling
from atomate2.siesta.dataclass.molecular_dynamics_and_relaxation import (
    MolecularDynamicsAndRelaxation,
)
from atomate2.siesta.dataclass.netcdf_options import NetcdfOptions
from atomate2.siesta.dataclass.optical_properties import OpticalProperties
from atomate2.siesta.dataclass.parallel_options import ParallelOptions
from atomate2.siesta.dataclass.phonon_calculations import PhononCalculations
from atomate2.siesta.dataclass.pseudopotentials import Pseudopotentials
from atomate2.siesta.dataclass.real_space_grid_parameters import RealSpaceGridParameters
from atomate2.siesta.dataclass.rttddft import RTTDDFT
from atomate2.siesta.dataclass.scf_loop_parameters import SCFLoopParameters
from atomate2.siesta.dataclass.solvers_and_performance_options import (
    SolversAndPerformanceOptions,
)
from atomate2.siesta.dataclass.spin_settings import SpinSettings
from atomate2.siesta.dataclass.structural_information import (
    StructuralInformationVersion1,
    StructuralInformationVersion2,
)
from atomate2.siesta.dataclass.wannier90 import Wannier90

# Initialize rich console
console = Console()

# Define the DATA_CLASSES dictionary
DATA_CLASSES = {
    "GeneralSystemDescriptors": GeneralSystemDescriptors,
    "Pseudopotentials": Pseudopotentials,
    "BasisSetsAndProjectors": BasisSetsAndProjectors,
    "StructuralInformationVersion1": StructuralInformationVersion1,
    "StructuralInformationVersion2": StructuralInformationVersion2,
    "KPointSampling": KPointSampling,
    "ExchangeCorrelationFunctionals": ExchangeCorrelationFunctionals,
    "SpinSettings": SpinSettings,
    "SCFLoopParameters": SCFLoopParameters,
    "RealSpaceGridParameters": RealSpaceGridParameters,
    "HamiltonianAndOverlapParameters": HamiltonianAndOverlapParameters,
    "ElectronicStructureCalculationOptions": ElectronicStructureCalculationOptions,
    "SolversAndPerformanceOptions": SolversAndPerformanceOptions,
    "DensityOfStatesAndBandStructure": DensityOfStatesAndBandStructure,
    "ChemicalAnalysis": ChemicalAnalysis,
    "OpticalProperties": OpticalProperties,
    "Wannier90": Wannier90,
    "ChargeDipoleElectricField": ChargeDipoleElectricField,
    "Grids": Grids,
    "AuxiliaryForceField": AuxiliaryForceField,
    "ParallelOptions": ParallelOptions,
    "EfficiencyOptions": EfficiencyOptions,
    "Denchar": Denchar,
    "NetcdfOptions": NetcdfOptions,
    "MolecularDynamicsAndRelaxation": MolecularDynamicsAndRelaxation,
    "ExternalControlAndScripting": ExternalControlAndScripting,
    "GeneralConstraints": GeneralConstraints,
    "PhononCalculations": PhononCalculations,
    "DFTU": DFTU,
    "RTTDDFT": RTTDDFT,
}


def format_default_value(value):
    """Helper function to format default values for display."""
    if value is None:
        return "None"
    if callable(value):
        try:
            result = value()
            if isinstance(result, (list, dict)) and not result:
                return str(result)
            return str(result)
        except Exception:
            return "<callable>"
    elif isinstance(value, (list, dict)) and not value:
        return str(value)
    else:
        return str(value)


def get_class_docstring(cls):
    """Extract the docstring of a class, if available."""
    doc = inspect.getdoc(cls)
    return doc or "No docstring available."


@click.group()
def cli():
    """Command-line interface for SIESTA input file operations.

    Provides tools for:
    - Displaying information about SIESTA data classes
    - Extracting parameters from existing FDF files
    - Searching and exploring SIESTA parameters
    """


@cli.command(name="list")
def list_classes():
    """List all available data classes."""
    console.print(
        Panel(
            Text("Available SIESTA Data Classes", style="bold cyan"),
            border_style="blue",
            expand=False,
        )
    )
    table = Table(show_header=False, box=ROUNDED)  # Changed to ROUNDED
    table.add_column("Class Name", style="green")
    for name in sorted(DATA_CLASSES.keys()):
        table.add_row(name)
    console.print(table)


@cli.command()
@click.argument("class_name")
@click.option("-c", "--complete", is_flag=True, help="Show complete detailed info.")
@click.option("-s", "--siesta", is_flag=True, help="Show SIESTA keywords info.")
@click.option("-u", "--unit", is_flag=True, help="Show unit of keywords.")
def show(class_name, complete, siesta, unit):
    """Show detailed information about a specific data class."""
    if class_name not in DATA_CLASSES:
        console.print(f"[red]Error: Data class '{class_name}' not found.[/red]")
        console.print(
            "[yellow]Use 'list' command to see available data classes.[/yellow]"
        )
        return

    cls = DATA_CLASSES[class_name]

    # Display class information in a panel
    console.print(
        Panel(
            Text(f"Data Class: {class_name}", style="bold cyan"),
            border_style="blue",
            expand=False,
        )
    )

    # Display docstring
    console.print(
        Panel(
            Text(get_class_docstring(cls), style="white"),
            title="Description",
            title_align="left",
            border_style="yellow",
        )
    )

    # Create table for attributes
    table = Table(title="Attributes", box=ROUNDED)  # Changed to ROUNDED
    table.add_column("Name", style="green")
    table.add_column("Type", style="magenta")
    table.add_column("Default", style="blue")
    if complete:
        table.add_column("SIESTA Keyword", style="yellow")
        table.add_column("Description", style="white")
    elif siesta:
        table.add_column("SIESTA Keyword", style="yellow")
    elif unit:
        table.add_column("Unit", style="yellow")

    # Get type hints for better type information
    type_hints = get_type_hints(cls)

    for field in fields(cls):
        field_type = type_hints.get(field.name, field.type)
        type_name = getattr(field_type, "__name__", str(field_type))
        default_value = (
            format_default_value(field.default)
            if field.default is not field.default_factory
            else format_default_value(field.default_factory)
        )

        if complete:
            description = field.metadata.get("description", "No description available.")
            siesta_keyword = field.metadata.get(
                "SIESTA keyword", "No SIESTA keyword available."
            )
            table.add_row(
                field.name, type_name, default_value, siesta_keyword, description
            )
        elif siesta:
            siesta_keyword = field.metadata.get(
                "SIESTA keyword", "No SIESTA keyword available."
            )
            table.add_row(field.name, type_name, default_value, siesta_keyword)
        elif unit:
            siesta_unit = field.metadata.get("Unit", "No unit available.")
            table.add_row(field.name, type_name, default_value, siesta_unit)
        else:
            table.add_row(field.name, type_name, default_value)

    console.print(table)


@cli.command()
@click.argument("keyword")
@click.option(
    "-r", "--restrict", is_flag=True, help="Restrict search to exact word matches."
)
def search(keyword, restrict):
    """Search for data class attributes by keyword in name, type, default, description, or SIESTA keyword."""
    keyword = keyword.lower()
    keyword_pattern = rf"\b{re.escape(keyword)}\b" if restrict else keyword
    matches = []

    for class_name, cls in DATA_CLASSES.items():
        type_hints = get_type_hints(cls)
        for field in fields(cls):
            field_type = type_hints.get(field.name, field.type)
            type_name = getattr(field_type, "__name__", str(field_type)).lower()
            default_value = (
                format_default_value(field.default)
                if field.default is not field.default_factory
                else format_default_value(field.default_factory)
            )
            description = field.metadata.get(
                "description", "No description available."
            ).lower()
            siesta_keyword = field.metadata.get(
                "SIESTA keyword", "No SIESTA keyword available."
            )
            siesta_keyword_lower = (
                siesta_keyword.lower()
                if siesta_keyword is not None
                else "no siesta keyword available."
            )

            match_details = []
            if restrict:
                if re.search(keyword_pattern, field.name.lower()):
                    match_details.append(f"field name: {field.name}")
                if re.search(keyword_pattern, type_name):
                    match_details.append(f"field type: {field.name} ({type_name})")
                if re.search(keyword_pattern, default_value.lower()):
                    match_details.append(
                        f"field default: {field.name} ({default_value})"
                    )
                if re.search(keyword_pattern, description):
                    match_details.append(f"field description: {field.name}")
                if re.search(keyword_pattern, siesta_keyword_lower):
                    match_details.append(
                        f"field SIESTA keyword: {field.name} ({siesta_keyword})"
                    )
            else:
                if keyword in field.name.lower():
                    match_details.append(f"field name: {field.name}")
                if keyword in type_name:
                    match_details.append(f"field type: {field.name} ({type_name})")
                if keyword in default_value.lower():
                    match_details.append(
                        f"field default: {field.name} ({default_value})"
                    )
                if keyword in description:
                    match_details.append(f"field description: {field.name}")
                if keyword in siesta_keyword_lower:
                    match_details.append(
                        f"field SIESTA keyword: {field.name} ({siesta_keyword})"
                    )

            if match_details:
                matches.append(
                    (
                        class_name,
                        field,
                        field_type,
                        default_value,
                        description,
                        siesta_keyword,
                    )
                )

    if not matches:
        console.print(f"[red]No attributes found matching keyword '{keyword}'.[/red]")
        console.print(
            "[yellow]Use 'list' command to see available data classes.[/yellow]"
        )
        return

    console.print(
        Panel(
            Text(
                f"Found {len(matches)} attribute(s) matching '{keyword}'",
                style="bold cyan",
            ),
            border_style="blue",
            expand=False,
        )
    )

    table = Table(title="Search Results", box=ROUNDED)  # Changed to ROUNDED
    table.add_column("Class", style="yellow")
    table.add_column("Attribute", style="green")
    table.add_column("Type", style="magenta")
    table.add_column("Default", style="blue")
    table.add_column("SIESTA Keyword", style="yellow")
    table.add_column("Description", style="white")

    for (
        class_name,
        field,
        field_type,
        default_value,
        description,
        siesta_keyword,
    ) in matches:
        type_name = getattr(field_type, "__name__", str(field_type))
        siesta_keyword_display = (
            siesta_keyword
            if siesta_keyword is not None
            else "No SIESTA keyword available."
        )
        table.add_row(
            class_name,
            field.name,
            type_name,
            default_value,
            siesta_keyword_display,
            description,
        )

    console.print(table)


# Register extract command
cli.add_command(extract)


if __name__ == "__main__":
    cli()
