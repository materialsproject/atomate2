import logging
from typing import Any

from ase import Atoms
from atomate2.siesta.sets.ase.siesta_input import SiestaInput

logger = logging.getLogger(__name__)


def _nonpolarized_alias(_: list, kwargs: dict[str, Any]) -> bool:
    """
    Handle deprecated 'UNPOLARIZED' spin keyword for backward compatibility.

    Args:
        _: Unused argument (list).
        kwargs (Dict[str, Any]): Keyword arguments to check and modify.

    Returns
    -------
        bool: True if 'UNPOLARIZED' was replaced with 'non-polarized', False otherwise.
    """
    logger.info("_nonpolarized_alias()")
    if kwargs.get("spin") == "UNPOLARIZED":
        kwargs["spin"] = "non-polarized"
        return True
    return False


# Utilities for generating bits of strings.
#
def format_block(name, block):
    """
    Format a SIESTA block for FDF input file.

    Args:
        name (str): Name of the block (e.g., 'BandPoints').
        block: Iterable of rows, where each row contains data to be formatted.

    Returns
    -------
        str: Formatted block string with %block and %endblock directives.
    """
    logger.info("format_block()")
    lines = [f"%block {name}"]
    for row in block:
        data = " ".join(str(obj) for obj in row)
        lines.append(f"    {data}")
    lines.append(f"%endblock {name}")
    return "\n".join(lines)


def bandpath2bandpoints(path):
    """
    Convert a band path to SIESTA BandPoints block format.

    Args:
        path: Band path object containing k-points.

    Returns
    -------
        str: Formatted BandPoints block for FDF file.
    """
    logger.info("bandpath2bandpoints()")
    return "\n".join(
        [
            "BandLinesScale ReciprocalLatticeVectors",
            format_block("BandPoints", path.kpts),
        ]
    )


# We are re-aliasing format_fdf and format_block in the anticipation
# that they may change, or we might move this onto a Formatter object
# which applies consistent spacings etc.
def var(key, value):
    """
    Format a single FDF variable.

    Args:
        key (str): FDF variable name.
        value: Variable value.

    Returns
    -------
        str: Formatted FDF variable line.
    """
    logger.info("var()")
    return format_fdf(key, value)


def block(name, data):
    """
    Format a SIESTA FDF block.

    Args:
        name (str): Block name.
        data: Block data as an iterable.

    Returns
    -------
        str: Formatted block string.
    """
    logger.info("block()")
    return format_block(name, data)


def format_fdf(key, value):
    """
    Write an fdf key-word value pair.

    Parameters
    ----------
        - key : The fdf-key
        - value : The fdf value.
    """
    logger.debug(f"format_fdf called with key={key}, value={value}")
    if isinstance(value, (list, tuple)) and len(value) == 0:
        logger.debug("Returning empty string for empty list/tuple")
        return ""
    if key.startswith("#"):
        # Special case: box the full value as a comment for keys starting with '#'
        if value is None or str(value).strip() == "":
            logger.warning(
                "Value for comment key is None or empty, using default comment"
            )
            comment_text = "Comment"
        else:
            comment_text = str(value)  # Use the full value
        result = "".join(list(comment_in_box([comment_text])))
        logger.debug(f"Generated comment box: {result}")
        return result
    # Check if key already has %block prefix before formatting
    key_has_block_prefix = key.startswith("%block")

    if not key_has_block_prefix:
        key = format_key(key)

    new_value = format_value(value)
    if isinstance(value, list):
        if key_has_block_prefix:
            # Key already has %block prefix, don't add it again
            block_name = key.replace("%block", "").strip()
            string = key + "\n" + new_value + "\n" + "%endblock " + block_name + "\n"
        else:
            string = (
                "%block " + key + "\n" + new_value + "\n" + "%endblock " + key + "\n"
            )
    else:
        string = f"{key}\t{new_value}\n"
    logger.debug(f"Generated FDF string: {string}")
    return string


def format_value(value):
    """
    Format python values to fdf-format.

    Parameters
    ----------
        - value : The value to format.
    """
    if isinstance(value, tuple):
        sub_values = [format_value(v) for v in value]
        value = "\t".join(sub_values)
    elif isinstance(value, list):
        sub_values = [format_value(v) for v in value]
        value = "\n".join(sub_values)
    else:
        value = str(value)

    return value


def format_key(key):
    """Fix the fdf-key replacing '_' with '.' and '__' with '_'"""
    key = key.replace("__", "#")
    key = key.replace("_", ".")
    # key = key.replace('#', '_')
    # if key == '#':
    #     # If the key is '#', treat the value as a comment and use comment_in_box
    #     # return comment_in_box([key])
    #     return ''.join(list(comment_in_box([key])))

    return key


def generate_atomic_coordinates(
    atoms: Atoms, species_numbers, atomic_coord_format: str
):
    """
    Generate atomic coordinates block for FDF file.

    Args:
        atoms (Atoms): ASE Atoms object.
        species_numbers: Species numbers for each atom.
        atomic_coord_format (str): Format ('xyz' or 'zmatrix').

    Yields
    ------
        str: Lines for the atomic coordinates block.

    Raises
    ------
        RuntimeError: If atomic_coord_format is unknown.
    Write atomic coordinates.

    Parameters
    ----------
    fd : IO
        An open file object.
    atoms : Atoms
        An atoms object.
    """
    logger.info("Siesta.generate_atomic_coordinates()")
    if atomic_coord_format == "xyz":
        return generate_atomic_coordinates_xyz(atoms, species_numbers)
    if atomic_coord_format == "zmatrix":
        return generate_atomic_coordinates_zmatrix(atoms, species_numbers)
    raise RuntimeError(f"Unknown atomic_coord_format: {atomic_coord_format}")


def generate_atomic_coordinates_zmatrix(atoms: Atoms, species_numbers):
    """
    Generate atomic coordinates in Z-matrix format.

    Args:
        atoms (Atoms): ASE Atoms object.
        species_numbers: Species numbers for each atom.

    Yields
    ------
        str: Lines for the Zmatrix block.

    Write atomic coordinates in Z-matrix format.

    Parameters
    ----------
    fd : IO
        An open file object.
    atoms : Atoms
        An atoms object.
    """
    logger.info("Siesta.generate_atomic_coordinates_zmatrix()")
    yield "\n"
    yield var("ZM.UnitsLength", "Ang")
    yield "%block Zmatrix\n"
    yield "  cartesian\n"

    fstr = "{:5d}" + "{:20.10f}" * 3 + "{:3d}" * 3 + "{:7d} {:s}\n"
    a2constr = SiestaInput.make_xyz_constraints(atoms)
    a2p, a2s = atoms.get_positions(), atoms.symbols
    for ia, (sp, xyz, ccc, sym) in enumerate(zip(species_numbers, a2p, a2constr, a2s)):
        yield fstr.format(
            sp, xyz[0], xyz[1], xyz[2], ccc[0], ccc[1], ccc[2], ia + 1, sym
        )
    yield "%endblock Zmatrix\n"

    # origin = tuple(-atoms.get_celldisp().flatten())
    # yield block('AtomicCoordinatesOrigin', [origin])


def generate_atomic_coordinates_xyz(atoms: Atoms, species_numbers):
    """
    Generate atomic coordinates in XYZ format.

    Args:
        atoms (Atoms): ASE Atoms object.
        species_numbers: Species numbers for each atom.

    Yields
    ------
        str: Lines for the AtomicCoordinatesAndAtomicSpecies block.

    Write atomic coordinates.

    Parameters
    ----------
    fd : IO
        An open file object.
    atoms : Atoms
        An atoms object.
    """
    logger.info("Siesta.generate_atomic_coordinates_xyz()")
    yield "\n"
    # yield var('AtomicCoordinatesFormat', 'Ang')
    yield var("AtomicCoordinatesFormat", "NotScaledCartesianAng")
    yield block(
        "AtomicCoordinatesAndAtomicSpecies",
        # [[*atom.position, number]
        # for atom, number in zip(atoms, species_numbers)])
        # AA: Had to change for vibra cz it needs masses
        [
            [*atom.position, number, mass]
            for atom, number, mass in zip(atoms, species_numbers, atoms.get_masses())
        ],
    )
    yield "\n"

    # origin = tuple(-atoms.get_celldisp().flatten())
    # yield block('AtomicCoordinatesOrigin', [origin])


def comment_in_box(text_lines):
    """
    Format a list of text lines into a boxed comment block for SIESTA FDF files.
    Each line is prefixed with '#' and enclosed in a border of '#'.

    Args:
        text_lines (list[str]): List of strings to be included in the comment box.

    Yields
    ------
        str: Lines of the boxed comment block, each starting with '#'.
    """
    logger.info("comment_in_box()")
    if not text_lines:
        yield "#\n"
        return

    # Find the length of the longest line for proper box sizing
    max_length = max(len(line) for line in text_lines)

    # Top border
    yield "#" + "-" * (max_length + 4) + "#\n"

    # Each line with side borders and # prefix
    for line in text_lines:
        # yield f'# | {line.ljust(max_length)} |\n'
        yield f"#  {line.ljust(max_length)} \n"

    # Bottom border
    yield "#" + "-" * (max_length + 4) + "#\n"
