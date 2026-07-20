#!/usr/bin/env python
import sisl
import json
import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import pickle
from pathlib import Path

# Initialize rich console for enhanced output formatting
console = Console()


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types for serialization.

    This class extends json.JSONEncoder to convert NumPy integers, floats, and arrays
    into standard Python types that can be serialized to JSON.

    Methods:
        default: Converts NumPy types to Python types.
    """

    def default(self, obj):
        """Convert NumPy types to Python types for JSON serialization.

        Args:
            obj: Object to be serialized.

        Returns:
            int, float, or list: Converted Python type for JSON serialization.

        Raises:
            TypeError: If the object cannot be serialized.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def print_species_info(species_dict, species_Z_dict):
    """Display species information in a formatted table using rich.

    Args:
        species_dict (dict): Dictionary mapping species indices to labels.
        species_Z_dict (dict): Dictionary mapping species indices to atomic numbers.
    """
    species_table = Table(
        title="Species Information", show_header=True, header_style="bold magenta"
    )
    species_table.add_column("Index", style="cyan")
    species_table.add_column("Label", style="green")
    species_table.add_column("Atomic Number (Z)", style="yellow")
    for idx, label in species_dict.items():
        species_table.add_row(str(idx), label, str(species_Z_dict[idx]))
    console.print(species_table)


def print_atom_info(atom_species, tags, species_labels, species_Z):
    """Display atom-related information in rich panels.

    Args:
        atom_species (numpy.ndarray): Array of species indices for each atom.
        tags (list): List of tags for each atom.
        species_labels (list): List of species labels for each atom.
        species_Z (list): List of atomic numbers for each atom.
    """
    console.print(
        Panel(
            f"Atom species indices: {atom_species.tolist()}",
            title="Atom Species",
            border_style="blue",
        )
    )
    console.print(Panel(f"Assigned tags: {tags}", title="Tags", border_style="blue"))
    console.print(
        Panel(
            f"Per-atom species labels: {species_labels}",
            title="Species Labels",
            border_style="green",
        )
    )
    console.print(
        Panel(
            f"Per-atom species Z: {species_Z}",
            title="Atomic Numbers",
            border_style="yellow",
        )
    )


def print_structure_info(
    structure_ase,
    structure_ase_no_ghost,
    structure_pymatgen,
    structure_pymatgen_no_ghost,
):
    """Display structure information for ASE and pymatgen objects in rich panels.

    Args:
        structure_ase (ase.Atoms): ASE structure with all atoms.
        structure_ase_no_ghost (ase.Atoms): ASE structure without ghost atoms.
        structure_pymatgen (pymatgen.core.Structure): Pymatgen structure with all atoms.
        structure_pymatgen_no_ghost (pymatgen.core.Structure): Pymatgen structure without ghost atoms.
    """
    console.print(
        Panel(
            f"ASE Atoms tags: {structure_ase.get_tags().tolist()}",
            title="ASE Tags",
            border_style="blue",
        )
    )
    console.print(
        Panel(
            f"ASE Atoms species_dict: {structure_ase.info.get('species_dict')}",
            title="ASE Species Dict",
            border_style="green",
        )
    )
    console.print(
        Panel(
            f"ASE Atoms species_Z_dict: {structure_ase.info.get('species_Z_dict')}",
            title="ASE Species Z Dict",
            border_style="yellow",
        )
    )
    console.print(
        Panel(
            f"ASE Atoms species_labels: {structure_ase.info.get('species_labels')}",
            title="ASE Species Labels",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            f"ASE Atoms species_Z: {structure_ase.info.get('species_Z')}",
            title="ASE Species Z",
            border_style="magenta",
        )
    )
    console.print(
        Panel(
            f"ASE Atoms (no ghost) tags: {structure_ase_no_ghost.get_tags().tolist()}",
            title="ASE (No Ghost) Tags",
            border_style="blue",
        )
    )
    console.print(
        Panel(
            f"ASE Atoms (no ghost) species_dict: {structure_ase_no_ghost.info.get('species_dict')}",
            title="ASE (No Ghost) Species Dict",
            border_style="green",
        )
    )
    console.print(
        Panel(
            f"ASE Atoms (no ghost) species_Z_dict: {structure_ase_no_ghost.info.get('species_Z_dict')}",
            title="ASE (No Ghost) Species Z Dict",
            border_style="yellow",
        )
    )
    console.print(
        Panel(
            f"ASE Atoms (no ghost) species_labels: {structure_ase_no_ghost.info.get('species_labels')}",
            title="ASE (No Ghost) Species Labels",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            f"ASE Atoms (no ghost) species_Z: {structure_ase_no_ghost.info.get('species_Z')}",
            title="ASE (No Ghost) Species Z",
            border_style="magenta",
        )
    )
    console.print(
        Panel(
            f"Pymatgen Structure: {structure_pymatgen}",
            title="Pymatgen Structure",
            border_style="blue",
        )
    )
    console.print(
        Panel(
            f"Pymatgen site properties: {structure_pymatgen.site_properties}",
            title="Pymatgen Site Properties",
            border_style="green",
        )
    )
    console.print(
        Panel(
            f"Pymatgen Structure (no ghost): {structure_pymatgen_no_ghost}",
            title="Pymatgen (No Ghost) Structure",
            border_style="blue",
        )
    )
    console.print(
        Panel(
            f"Pymatgen (no ghost) site properties: {structure_pymatgen_no_ghost.site_properties}",
            title="Pymatgen (No Ghost) Site Properties",
            border_style="green",
        )
    )


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--write-xsf", is_flag=True, help="Write ASE structure (no ghost) to XSF file"
)
@click.option(
    "--write-cif", is_flag=True, help="Write pymatgen structure (no ghost) to CIF file"
)
@click.option("--write-json", is_flag=True, help="Write structure data to JSON file")
@click.option(
    "--write-sisl-pickle", is_flag=True, help="Write sisl structure to pickle file"
)
@click.option(
    "--write-ase-pickle",
    is_flag=True,
    help="Write ASE structures (with and without ghost) to pickle files",
)
@click.option(
    "--write-pymatgen-pickle",
    is_flag=True,
    help="Write pymatgen structures (with and without ghost) to pickle files",
)
@click.option(
    "--write-fdf", is_flag=True, help="Write sisl structure (with ghost) to FDF file"
)
@click.option(
    "--write-fdf-no-ghost",
    is_flag=True,
    help="Write sisl structure (no ghost) to FDF file",
)
@click.option(
    "--output-prefix",
    default="structure",
    help="Prefix for output file names (default: structure)",
)
def main(
    input_file,
    write_xsf,
    write_cif,
    write_json,
    write_sisl_pickle,
    write_ase_pickle,
    write_pymatgen_pickle,
    write_fdf,
    write_fdf_no_ghost,
    output_prefix,
):
    """
    Convert structure files between different formats (SIESTA FDF/XV, CIF, XSF) with automatic format detection.

    This script automatically detects the input file format from its extension (.fdf, .xv/.XV, .cif, .xsf),
    reads the structure, converts it to ASE and Pymatgen formats, removes ghost species (those with negative
    atomic numbers or '_ghost' in the label), and saves the results to specified output files with a user-defined prefix.

    Supported input formats:
        - .fdf: SIESTA FDF file (with ChemicalSpeciesLabel block)
        - .xv/.XV: SIESTA XV file (restart geometry)
        - .cif: Crystallographic Information File
        - .xsf: XCrySDen Structure File

    Args:
        input_file (str): Path to structure file (.fdf, .xv, .XV, .cif, or .xsf).
        write_xsf (bool): If True, write ASE structure (no ghost) to <output_prefix>_no_ghost.xsf.
        write_cif (bool): If True, write pymatgen structure (no ghost) to <output_prefix>_no_ghost.cif.
        write_json (bool): If True, write structure data to <output_prefix>_data.json.
        write_sisl_pickle (bool): If True, write sisl structure to <output_prefix>_sisl.pkl.
        write_ase_pickle (bool): If True, write ASE structures to <output_prefix>_ase.pkl and <output_prefix>_ase_no_ghost.pkl.
        write_pymatgen_pickle (bool): If True, write pymatgen structures to <output_prefix>_pymatgen.pkl and <output_prefix>_pymatgen_no_ghost.pkl.
        write_fdf (bool): If True, write sisl structure (with ghost) to <output_prefix>.fdf.
        write_fdf_no_ghost (bool): If True, write sisl structure (no ghost) to <output_prefix>_no_ghost.fdf.
        output_prefix (str): Prefix for output file names (default: 'structure').

        One can use pickle to read the different pickle objects:

        import pickle

        with open('test_pymatgen.pkl', 'rb') as file:

            loaded_data = pickle.load(file)

    Raises:
        ValueError: If reading geometry, unsupported format, or converting structures fails.
        IOError: If writing to any output file fails.

    Examples:
        Convert SIESTA FDF to all formats (automatic detection):
        $ atomate2siesta-structure convert siesta.fdf --write-xsf --write-cif --write-json

        Convert XV file to CIF (automatic detection):
        $ atomate2siesta-structure convert siesta.XV --write-cif --output-prefix mgo

        Convert CIF to SIESTA FDF (automatic detection):
        $ atomate2siesta-structure convert structure.cif --write-fdf --output-prefix siesta_input

        Convert XSF to multiple formats:
        $ atomate2siesta-structure convert structure.xsf --write-cif --write-fdf --write-json
    """
    # Automatic format detection based on file extension
    input_path = Path(input_file)
    file_ext = input_path.suffix.lower()

    console.print(f"[cyan]Detecting file format from extension: {file_ext}[/cyan]")

    # Read geometry based on detected format
    try:
        if file_ext in [".xv"]:
            # SIESTA XV file (restart geometry)
            console.print("[cyan]Detected: SIESTA XV file[/cyan]")
            structure_sisl = sisl.get_sile(input_file).read_geometry()
            file_type = "XV file"

        elif file_ext in [".fdf"]:
            # SIESTA FDF file
            console.print("[cyan]Detected: SIESTA FDF file[/cyan]")
            structure_sisl = sisl.get_sile(input_file).read_geometry()
            file_type = "FDF file"

        elif file_ext in [".cif"]:
            # CIF file - read using pymatgen, then convert to sisl
            console.print("[cyan]Detected: CIF file[/cyan]")
            structure_pmg = Structure.from_file(input_file)
            structure_ase_temp = AseAtomsAdaptor.get_atoms(structure_pmg)
            # Convert ASE to sisl Geometry
            cell = structure_ase_temp.get_cell()[:]
            xyz = structure_ase_temp.get_positions()
            atoms = [sisl.Atom(Z=atom.number) for atom in structure_ase_temp]
            structure_sisl = sisl.Geometry(
                xyz=xyz, atoms=sisl.Atoms(atoms=atoms, na=len(xyz)), lattice=cell
            )
            file_type = "CIF file"

        elif file_ext in [".xsf"]:
            # XSF file - read using ASE, then convert to sisl
            console.print("[cyan]Detected: XSF file[/cyan]")
            structure_ase_temp = ase_read(input_file)
            # Convert ASE to sisl Geometry
            cell = structure_ase_temp.get_cell()[:]
            xyz = structure_ase_temp.get_positions()
            atoms = [sisl.Atom(Z=atom.number) for atom in structure_ase_temp]
            structure_sisl = sisl.Geometry(
                xyz=xyz, atoms=sisl.Atoms(atoms=atoms, na=len(xyz)), lattice=cell
            )
            file_type = "XSF file"

        else:
            # Unsupported format
            raise ValueError(
                f"Unsupported file format: {file_ext}\n"
                f"Supported formats: .fdf, .xv, .XV, .cif, .xsf"
            )

        console.print(f"[green]Successfully read structure from {file_type}[/green]")

    except Exception as e:
        console.print(
            f"[red]Error reading geometry from {file_type} ({input_file}): {str(e)}[/red]"
        )
        raise ValueError(
            f"Error reading geometry from {file_type} ({input_file}): {str(e)}"
        )

    # Write sisl structure to pickle if requested
    if write_sisl_pickle:
        try:
            with open(f"{output_prefix}_sisl.pkl", "wb") as f:
                pickle.dump(structure_sisl, f)
            console.print(
                f"[green]Wrote sisl structure to {output_prefix}_sisl.pkl[/green]"
            )
        except Exception as e:
            console.print(
                f"[red]Error writing sisl structure to pickle: {str(e)}[/red]"
            )
            raise IOError(f"Error writing sisl structure to pickle: {str(e)}")

    # Write sisl structure to FDF if requested
    if write_fdf:
        try:
            structure_sisl.write(f"{output_prefix}.fdf")
            console.print(
                f"[green]Wrote sisl structure (with ghost) to {output_prefix}.fdf[/green]"
            )
        except Exception as e:
            console.print(f"[red]Error writing sisl structure to FDF: {str(e)}[/red]")
            raise IOError(f"Error writing sisl structure to FDF: {str(e)}")

    # Get species labels from the ChemicalSpeciesLabel block
    # Note: ChemicalSpeciesLabel is only in SIESTA FDF files
    # Determine the FDF file path based on input file format
    if file_ext in [".xv"]:
        # Input is an XV file, try to find the corresponding FDF file
        fdf_file = str(input_path.with_suffix(".fdf"))
    elif file_ext in [".fdf"]:
        # Input is already an FDF file
        fdf_file = input_file
    else:
        # For CIF/XSF files, there is no FDF file - will use fallback
        fdf_file = None

    # Convert sisl geometry to ASE Atoms (needed for fallback)
    structure_ase = structure_sisl.to.ase()

    # Get the species indices for each atom (from AtomicCoordinatesAndAtomicSpecies)
    atom_species = (
        structure_sisl.atoms.species
    )  # Array of species indices for each atom

    # Try to read species information from FDF file
    species_dict = {}
    species_Z_dict = {}
    species_block = None

    # Only try to read FDF file if it's a SIESTA format
    if fdf_file is not None:
        try:
            sile = sisl.get_sile(fdf_file)
            sile.read()
            # Access the species information (ChemicalSpeciesLabel block)
            species_block = sile.get(
                "ChemicalSpeciesLabel"
            )  # Returns a list of strings, e.g., ['1 12 Mg', '2 8 O', '3 -8 O_ghost', '4 8 O_surface']

            for entry in species_block:
                # Split the string, e.g., '1 12 Mg' -> ['1', '12', 'Mg']
                parts = entry.strip().split()
                if len(parts) >= 3:
                    index = int(parts[0])  # Species index (1-based in .fdf)
                    Z = int(parts[1])  # Atomic number (e.g., 12, -8)
                    label = parts[2]  # Species label (e.g., Mg, O_ghost, O_surface)
                    species_dict[index] = label
                    species_Z_dict[index] = Z

            console.print(
                f"[green]Successfully read ChemicalSpeciesLabel from {fdf_file}[/green]"
            )

        except (FileNotFoundError, IOError) as e:
            # FDF file not found, fall back to extracting species from geometry
            console.print(
                f"[yellow]Warning: Could not read FDF file '{fdf_file}': {str(e)}[/yellow]"
            )
            console.print(
                "[yellow]Falling back to extracting species information from geometry[/yellow]"
            )
            fdf_file = None  # Mark as unavailable for fallback logic

    # If no FDF file or reading failed, extract species from geometry
    if fdf_file is None or not species_dict:
        # Extract species information from the geometry
        from pymatgen.core import Element

        # Get unique species from the structure
        unique_atoms = structure_sisl.atoms.atom
        for i, atom in enumerate(unique_atoms):
            species_idx = i + 1  # 1-based indexing
            Z = atom.Z  # Atomic number
            try:
                element = Element.from_Z(Z)
                label = element.symbol
            except (ValueError, KeyError):
                # Handle ghost atoms or invalid Z
                if Z < 0:
                    label = f"Ghost_{abs(Z)}"
                else:
                    label = f"X_{Z}"

            species_dict[species_idx] = label
            species_Z_dict[species_idx] = Z

        console.print(
            f"[green]Extracted {len(species_dict)} species from geometry[/green]"
        )

        # Reconstruct species_block for JSON output
        species_block = []
        for idx in sorted(species_dict.keys()):
            label = species_dict[idx]
            Z = species_Z_dict[idx]
            species_block.append(f"{idx} {Z} {label}")

    # Assign tags and collect species labels and Z values
    tags = []
    species_labels = []  # Store per-atom species labels
    species_Z = []  # Store per-atom Z values
    for specie_idx in atom_species:
        # Adjust for 0-based indexing (sisl returns 0-based, .fdf uses 1-based)
        fdf_index = specie_idx + 1 if specie_idx < len(species_dict) else specie_idx
        species_label = species_dict.get(fdf_index, "")
        Z = species_Z_dict.get(fdf_index, 0)
        species_labels.append(species_label)
        species_Z.append(Z)
        tags.append(int(specie_idx))  # Tag is 0-based species index

    # Display species and atom information
    print_species_info(species_dict, species_Z_dict)
    print_atom_info(atom_species, tags, species_labels, species_Z)

    # Store species information in ASE Atoms
    structure_ase.info["species_dict"] = species_dict  # Store the full dictionary
    structure_ase.info["species_Z_dict"] = species_Z_dict  # Store Z dictionary
    structure_ase.info["species_labels"] = species_labels  # Store per-atom labels
    structure_ase.info["species_Z"] = species_Z  # Store per-atom Z values
    structure_ase.set_tags(tags)  # Set tags (numerical, stored in arrays)

    # Write ASE structure to pickle if requested
    if write_ase_pickle:
        try:
            with open(f"{output_prefix}_ase.pkl", "wb") as f:
                pickle.dump(structure_ase, f)
            console.print(
                f"[green]Wrote ASE structure to {output_prefix}_ase.pkl[/green]"
            )
        except Exception as e:
            console.print(f"[red]Error writing ASE structure to pickle: {str(e)}[/red]")
            raise IOError(f"Error writing ASE structure to pickle: {str(e)}")

    # Convert to pymatgen Structure for Atomate2
    try:
        structure_pymatgen = AseAtomsAdaptor.get_structure(structure_ase)
    except Exception as e:
        console.print(
            f"[red]Error converting ASE Atoms to pymatgen Structure: {str(e)}[/red]"
        )
        raise ValueError(f"Error converting ASE Atoms to pymatgen Structure: {str(e)}")
    # Add tags, species labels, and Z as site properties for pymatgen
    structure_pymatgen.add_site_property("tags", tags)
    structure_pymatgen.add_site_property("species_label", species_labels)
    structure_pymatgen.add_site_property("species_Z", species_Z)

    # Write pymatgen structure to pickle if requested
    if write_pymatgen_pickle:
        try:
            with open(f"{output_prefix}_pymatgen.pkl", "wb") as f:
                pickle.dump(structure_pymatgen, f)
            console.print(
                f"[green]Wrote pymatgen structure to {output_prefix}_pymatgen.pkl[/green]"
            )
        except Exception as e:
            console.print(
                f"[red]Error writing pymatgen structure to pickle: {str(e)}[/red]"
            )
            raise IOError(f"Error writing pymatgen structure to pickle: {str(e)}")

    # Create new ASE Atoms object without ghost species
    non_ghost_indices = [
        i
        for i, label in enumerate(species_labels)
        if "_ghost" not in label and species_Z[i] > 0
    ]
    positions_no_ghost = structure_ase.get_positions()[non_ghost_indices]
    symbols_no_ghost = [
        structure_ase.get_chemical_symbols()[i] for i in non_ghost_indices
    ]
    tags_no_ghost = [tags[i] for i in non_ghost_indices]
    species_labels_no_ghost = [species_labels[i] for i in non_ghost_indices]
    species_Z_no_ghost = [species_Z[i] for i in non_ghost_indices]

    structure_ase_no_ghost = Atoms(
        symbols=symbols_no_ghost,
        positions=positions_no_ghost,
        cell=structure_ase.get_cell(),
        pbc=structure_ase.get_pbc(),
        tags=tags_no_ghost,
        info={
            "species_dict": species_dict,
            "species_Z_dict": {
                k: v
                for k, v in species_Z_dict.items()
                if v > 0 and "_ghost" not in species_dict.get(k, "")
            },
            "species_labels": species_labels_no_ghost,
            "species_Z": species_Z_no_ghost,
        },
    )

    # Convert ASE Atoms (no ghost) to sisl Geometry for FDF writing
    if write_fdf_no_ghost:
        try:
            # Create a new sisl Geometry for non-ghost atoms
            cell = structure_ase_no_ghost.get_cell()[:]
            xyz = structure_ase_no_ghost.get_positions()
            # Map species labels to sisl Atoms, preserving original labels and Z
            unique_labels = sorted(
                set(species_labels_no_ghost)
            )  # e.g., ['Mg', 'O', 'O_surface']
            species_Z_map = {
                label: species_Z_no_ghost[species_labels_no_ghost.index(label)]
                for label in unique_labels
            }
            atoms = []
            for label in species_labels_no_ghost:
                Z = species_Z_map[label]
                atoms.append(sisl.Atom(Z=Z, tag=label))
            structure_sisl_no_ghost = sisl.Geometry(
                xyz=xyz, atoms=sisl.Atoms(atoms=atoms, na=len(xyz)), lattice=cell
            )
            # Write to FDF
            structure_sisl_no_ghost.write(f"{output_prefix}_no_ghost.fdf")
            console.print(
                f"[green]Wrote sisl structure (no ghost) to {output_prefix}_no_ghost.fdf[/green]"
            )
        except Exception as e:
            console.print(
                f"[red]Error writing sisl structure (no ghost) to FDF: {str(e)}[/red]"
            )
            raise IOError(f"Error writing sisl structure (no ghost) to FDF: {str(e)}")

    # Write ASE (no ghost) structure to pickle if requested
    if write_ase_pickle:
        try:
            with open(f"{output_prefix}_ase_no_ghost.pkl", "wb") as f:
                pickle.dump(structure_ase_no_ghost, f)
            console.print(
                f"[green]Wrote ASE structure (no ghost) to {output_prefix}_ase_no_ghost.pkl[/green]"
            )
        except Exception as e:
            console.print(
                f"[red]Error writing ASE structure (no ghost) to pickle: {str(e)}[/red]"
            )
            raise IOError(f"Error writing ASE structure (no ghost) to pickle: {str(e)}")

    # Convert ASE Atoms (no ghost) to pymatgen Structure
    try:
        structure_pymatgen_no_ghost = AseAtomsAdaptor.get_structure(
            structure_ase_no_ghost
        )
    except Exception as e:
        console.print(
            f"[red]Error converting ASE Atoms (no ghost) to pymatgen Structure: {str(e)}[/red]"
        )
        raise ValueError(
            f"Error converting ASE Atoms (no ghost) to pymatgen Structure: {str(e)}"
        )
    # Add site properties for pymatgen
    structure_pymatgen_no_ghost.add_site_property("tags", tags_no_ghost)
    structure_pymatgen_no_ghost.add_site_property(
        "species_label", species_labels_no_ghost
    )
    structure_pymatgen_no_ghost.add_site_property("species_Z", species_Z_no_ghost)

    # Write pymatgen (no ghost) structure to pickle if requested
    if write_pymatgen_pickle:
        try:
            with open(f"{output_prefix}_pymatgen_no_ghost.pkl", "wb") as f:
                pickle.dump(structure_pymatgen_no_ghost, f)
            console.print(
                f"[green]Wrote pymatgen structure (no ghost) to {output_prefix}_pymatgen_no_ghost.pkl[/green]"
            )
        except Exception as e:
            console.print(
                f"[red]Error writing pymatgen structure (no ghost) to pickle: {str(e)}[/red]"
            )
            raise IOError(
                f"Error writing pymatgen structure (no ghost) to pickle: {str(e)}"
            )

    # Write to XSF file using ASE if requested
    if write_xsf:
        try:
            structure_ase_no_ghost.write(f"{output_prefix}_no_ghost.xsf")
            console.print(
                f"[green]Wrote ASE structure (no ghost) to {output_prefix}_no_ghost.xsf[/green]"
            )
        except Exception as e:
            console.print(f"[red]Error writing XSF file: {str(e)}[/red]")
            raise IOError(f"Error writing XSF file: {str(e)}")

    # Write to CIF file using pymatgen if requested
    if write_cif:
        try:
            structure_pymatgen_no_ghost.to_file(f"{output_prefix}_no_ghost.cif")
            console.print(
                f"[green]Wrote pymatgen structure (no ghost) to {output_prefix}_no_ghost.cif[/green]"
            )
        except Exception as e:
            console.print(f"[red]Error writing CIF file: {str(e)}[/red]")
            raise IOError(f"Error writing CIF file: {str(e)}")

    # Prepare data for JSON
    json_data = {
        "sisl": {
            "lattice": structure_sisl.cell.tolist(),  # Convert NumPy array to list
            "coordinates": structure_sisl.xyz.tolist(),  # Atomic coordinates
            "species_indices": structure_sisl.atoms.species.tolist(),  # 0-based species indices
            "chemical_species_label": species_block,  # Raw ChemicalSpeciesLabel block
        },
        "ase": {
            "positions": structure_ase.get_positions().tolist(),  # Convert NumPy array to list
            "chemical_symbols": structure_ase.get_chemical_symbols(),
            "tags": structure_ase.get_tags().tolist(),  # Convert NumPy array to list
            "species_dict": structure_ase.info.get("species_dict"),
            "species_Z_dict": structure_ase.info.get("species_Z_dict"),
            "species_labels": structure_ase.info.get("species_labels"),
            "species_Z": structure_ase.info.get("species_Z"),
            "cell": structure_ase.get_cell().tolist(),  # Convert NumPy array to list
            "pbc": structure_ase.get_pbc().tolist(),  # Periodic boundary conditions
        },
        "ase_no_ghost": {
            "positions": structure_ase_no_ghost.get_positions().tolist(),
            "chemical_symbols": structure_ase_no_ghost.get_chemical_symbols(),
            "tags": structure_ase_no_ghost.get_tags().tolist(),
            "species_dict": structure_ase_no_ghost.info.get("species_dict"),
            "species_Z_dict": structure_ase_no_ghost.info.get("species_Z_dict"),
            "species_labels": structure_ase_no_ghost.info.get("species_labels"),
            "species_Z": structure_ase_no_ghost.info.get("species_Z"),
            "cell": structure_ase_no_ghost.get_cell().tolist(),
            "pbc": structure_ase_no_ghost.get_pbc().tolist(),
        },
        "pymatgen": {
            "lattice": {
                "matrix": structure_pymatgen.lattice.matrix.tolist(),  # Lattice vectors
                "a": float(structure_pymatgen.lattice.a),  # Convert to float
                "b": float(structure_pymatgen.lattice.b),
                "c": float(structure_pymatgen.lattice.c),
                "alpha": float(structure_pymatgen.lattice.alpha),
                "beta": float(structure_pymatgen.lattice.beta),
                "gamma": float(structure_pymatgen.lattice.gamma),
            },
            "sites": [
                {
                    "species": site.species_string,
                    "coords": site.coords.tolist(),  # Convert NumPy array to list
                    "tags": int(
                        structure_pymatgen.site_properties["tags"][i]
                    ),  # Ensure Python int
                    "species_label": structure_pymatgen.site_properties[
                        "species_label"
                    ][i],
                    "species_Z": int(
                        structure_pymatgen.site_properties["species_Z"][i]
                    ),  # Ensure Python int
                }
                for i, site in enumerate(structure_pymatgen)
            ],
            "site_properties": {
                "tags": [
                    int(tag) for tag in structure_pymatgen.site_properties["tags"]
                ],  # Convert to Python int
                "species_label": structure_pymatgen.site_properties["species_label"],
                "species_Z": [
                    int(Z) for Z in structure_pymatgen.site_properties["species_Z"]
                ],  # Convert to Python int
            },
            "formula": structure_pymatgen.formula,
            "reduced_formula": structure_pymatgen.composition.reduced_formula,
        },
        "pymatgen_no_ghost": {
            "lattice": {
                "matrix": structure_pymatgen_no_ghost.lattice.matrix.tolist(),
                "a": float(structure_pymatgen_no_ghost.lattice.a),
                "b": float(structure_pymatgen_no_ghost.lattice.b),
                "c": float(structure_pymatgen_no_ghost.lattice.c),
                "alpha": float(structure_pymatgen_no_ghost.lattice.alpha),
                "beta": float(structure_pymatgen_no_ghost.lattice.beta),
                "gamma": float(structure_pymatgen_no_ghost.lattice.gamma),
            },
            "sites": [
                {
                    "species": site.species_string,
                    "coords": site.coords.tolist(),
                    "tags": int(structure_pymatgen_no_ghost.site_properties["tags"][i]),
                    "species_label": structure_pymatgen_no_ghost.site_properties[
                        "species_label"
                    ][i],
                    "species_Z": int(
                        structure_pymatgen_no_ghost.site_properties["species_Z"][i]
                    ),
                }
                for i, site in enumerate(structure_pymatgen_no_ghost)
            ],
            "site_properties": {
                "tags": [
                    int(tag)
                    for tag in structure_pymatgen_no_ghost.site_properties["tags"]
                ],
                "species_label": structure_pymatgen_no_ghost.site_properties[
                    "species_label"
                ],
                "species_Z": [
                    int(Z)
                    for Z in structure_pymatgen_no_ghost.site_properties["species_Z"]
                ],
            },
            "formula": structure_pymatgen_no_ghost.formula,
            "reduced_formula": structure_pymatgen_no_ghost.composition.reduced_formula,
        },
    }

    # Write to JSON file if requested
    if write_json:
        json_file = f"{output_prefix}_data.json"
        try:
            with open(json_file, "w") as f:
                json.dump(json_data, f, indent=4, cls=NumpyEncoder)
            console.print(f"[green]Data saved to {json_file}[/green]")
        except Exception as e:
            console.print(f"[red]Error writing to {json_file}: {str(e)}[/red]")
            raise IOError(f"Error writing to {json_file}: {str(e)}")

    # Display structure information
    print_structure_info(
        structure_ase,
        structure_ase_no_ghost,
        structure_pymatgen,
        structure_pymatgen_no_ghost,
    )


if __name__ == "__main__":
    main()
