import logging
from pathlib import Path

import click
import sisl
from ase import Atoms

from atomate2.siesta.cli.structure.convert import main as sisl_main

logger = logging.getLogger(__name__)


def generate_structure_fdf(
    atoms: Atoms = None,
    input_file: str = None,
    output_file: str = "structure.fdf",
    xv: bool = False,
) -> None:
    """
    Generate structure.fdf file from ASE Atoms object or existing FDF/XV file.

    This function creates a SIESTA-compatible structure.fdf file containing
    lattice vectors, atomic coordinates, and chemical species information.
    It uses sisl for the underlying file format conversion.

    Args:
        atoms: ASE Atoms object to convert to FDF. If provided, takes precedence.
        input_file: Path to input FDF or XV file to read structure from.
        output_file: Path to output structure.fdf file (default: 'structure.fdf').
        xv: If True and input_file is provided, read geometry from XV file instead of FDF.

    Returns:
        None

    Raises:
        ValueError: If neither atoms nor input_file is provided, or if input validation fails.
        FileNotFoundError: If input_file is specified but does not exist.
        IOError: If writing to output_file fails or directory cannot be created.

    Example:
        >>> from ase.build import bulk
        >>> atoms = bulk('Si', 'diamond', a=5.43)
        >>> generate_structure_fdf(atoms, output_file='Si.fdf')
    """
    logger.info("generate_structure_fdf()")

    # Validate inputs
    if atoms is None and input_file is None:
        raise ValueError("Either 'atoms' or 'input_file' must be provided.")

    # Ensure output directory exists
    output_path = Path(output_file)
    if output_path.parent != Path("."):
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created output directory: {output_path.parent}")
        except OSError as e:
            raise IOError(f"Cannot create output directory {output_path.parent}: {e}")

    if input_file:
        # Read from existing FDF/XV file
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        logger.info(f"Reading structure from {input_file} (xv={xv})")
        try:
            # Call the original sisl script's main function
            ctx = click.Context(sisl_main)
            ctx.invoke(
                sisl_main,
                input_file=input_file,
                xv=xv,
                write_fdf=True,
                output_prefix=output_path.stem,
            )
            logger.info(f"Structure file generated: {output_file}")
        except Exception as e:
            raise IOError(f"Failed to convert {input_file} to FDF: {e}") from e
    else:
        # Convert ASE Atoms to sisl Geometry and write FDF
        logger.info(f"Converting ASE Atoms to structure.fdf: {output_file}")

        # Validate atoms object
        if not isinstance(atoms, Atoms):
            raise ValueError(f"Expected ASE Atoms object, got {type(atoms)}")
        if len(atoms) == 0:
            raise ValueError("Atoms object is empty")

        try:
            # Extract atomic structure information
            cell = atoms.get_cell()[:]
            xyz = atoms.get_positions()
            symbols = atoms.get_chemical_symbols()
            atomic_numbers = atoms.get_atomic_numbers()

            # Get species information from atoms.info if available
            species_labels = atoms.info.get("species_labels", symbols)
            species_Z = atoms.info.get("species_Z", atomic_numbers.tolist())

            # Validate species information consistency
            if len(species_labels) != len(atoms):
                raise ValueError(
                    f"species_labels length ({len(species_labels)}) does not match "
                    f"number of atoms ({len(atoms)})"
                )

            # Build mapping from unique species labels to atomic numbers
            unique_labels = sorted(set(species_labels))
            species_Z_map = {}
            for label in unique_labels:
                # Find first occurrence of this label to get its atomic number
                idx = species_labels.index(label)
                species_Z_map[label] = species_Z[idx]

            logger.debug(f"Unique species: {unique_labels}")
            logger.debug(f"Species Z map: {species_Z_map}")

            # Build sisl Atoms list
            atoms_sisl = [
                sisl.Atom(Z=species_Z_map[label], tag=label) for label in species_labels
            ]

            # Create sisl Geometry
            structure_sisl = sisl.Geometry(
                xyz=xyz, atoms=sisl.Atoms(atoms=atoms_sisl, na=len(xyz)), lattice=cell
            )

            # Write to FDF
            structure_sisl.write(str(output_path))

            # Verify file was written
            if not output_path.exists():
                raise IOError(f"Structure file was not created: {output_file}")

            logger.info(f"Successfully wrote structure to {output_file}")

        except (ValueError, KeyError, IndexError) as e:
            raise ValueError(
                f"Invalid atoms structure or species information: {e}"
            ) from e
        except Exception as e:
            raise IOError(f"Error writing structure to {output_file}: {e}") from e
