"""Utilities for structure conversion, magnetism, and SIESTA I/O helpers."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, cast

import numpy as np
import sisl
import yaml
from ase import Atoms

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import PeriodicSite, Structure

logger = logging.getLogger(__name__)


def _get_site_atomic_number(site: PeriodicSite) -> int:
    """
    Get atomic number from a pymatgen site, handling regular and fractional occupancy.

    This handles sites from read_cif_with_ghost() which have fractional occupancy
    (e.g., {'S': 0.001} for ghost atoms) where site.specie doesn't work.

    Parameters
    ----------
    site : PeriodicSite
        A pymatgen PeriodicSite object.

    Returns
    -------
    int
        The atomic number (Z) of the site's element.
    """
    if hasattr(site.species, "Z"):
        # Regular Element/Species with direct Z attribute
        return site.species.Z
    # Composition (fractional occupancy) - get first element's Z
    return next(iter(site.species.keys())).Z


# Define magnetic elements and their default moments as module-level constants
MAGNETIC_ELEMENTS = {
    24,
    25,
    26,
    27,
    28,
    29,  # Cr, Mn, Fe, Co, Ni, Cu (3d)
    42,
    43,
    44,
    45,  # Mo, Tc, Ru, Rh (4d)
    # Lanthanides
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    # Actinides
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
}

ELEMENT_MAGMOMS = {
    24: 4.0,  # Cr
    25: 5.0,  # Mn
    26: 4.0,  # Fe
    27: 3.0,  # Co
    28: 2.0,  # Ni
    29: 0.6,  # Cu (d⁹, one unpaired electron, ~0.5-1.0 μB)
    64: 7.0,  # Gd (has highest magnetic moment)
}


def get_default_initial_magnetic_moments(
    structure: Structure,
    default_magmom: float = 1.0,
    magnetic_ordering: str = "ferromagnetic",
) -> list[float] | None:
    """
    Automatically detect magnetic elements and assign initial magnetic moments.

    Parameters
    ----------
    structure : Structure
        Pymatgen Structure object
    default_magmom : float, optional
        Default magnetic moment to assign to magnetic elements (default: 1.0 μB)
    magnetic_ordering : str, optional
        Magnetic ordering pattern (default: "ferromagnetic")
        Options:
        - "ferromagnetic" or "FM": All moments positive
        - "antiferromagnetic" or "AFM": Alternating signs (+/-)
        - "custom": Use exact values from structure

    Returns
    -------
    list[float] or None
        List of initial magnetic moments for each atom, or None if no magnetic elements
        - Magnetic atoms: element-specific moments (e.g., Fe: 4.0, Cu: 0.6)
        - Non-magnetic atoms: 0.0

    Notes
    -----
    Magnetic elements include:
    - 3d transition metals: Cr, Mn, Fe, Co, Ni, Cu (24-29)
    - 4d transition metals: Mo, Tc, Ru, Rh (42-45)
    - Lanthanides: La-Tm (57-69)
    - Actinides: Ac-Es (89-99)
    - Element-specific defaults: Cr(4.0), Mn(5.0), Fe(4.0), Co(3.0), Ni(2.0),
      Cu(0.6), Gd(7.0)

    Examples
    --------
    Ferromagnetic (default):
    >>> magmoms = get_default_initial_magnetic_moments(structure)
    >>> # Fe: [4.0, 4.0, ...], Cu: [0.6, 0.6, ...], O: [0.0, 0.0, ...]

    Antiferromagnetic:
    >>> magmoms = get_default_initial_magnetic_moments(
    ...     structure, magnetic_ordering="AFM"
    ... )
    >>> # Fe: [4.0, -4.0, 4.0, -4.0, ...], O: [0.0, 0.0, ...]
    """
    logger.info("get_default_initial_magnetic_moments()")

    # Check if structure has any magnetic elements
    atomic_numbers = [_get_site_atomic_number(site) for site in structure]
    has_magnetic = any(z in MAGNETIC_ELEMENTS for z in atomic_numbers)

    if not has_magnetic:
        logger.debug("No magnetic elements detected, returning None")
        return None

    # Normalize magnetic ordering string
    ordering_lower = magnetic_ordering.lower()
    is_fm = ordering_lower in ["ferromagnetic", "fm"]
    is_afm = ordering_lower in ["antiferromagnetic", "afm"]

    # Assign magnetic moments
    magmoms = []
    magnetic_atom_index = 0  # Counter for magnetic atoms only (for AFM pattern)

    for z in atomic_numbers:
        if z in MAGNETIC_ELEMENTS:
            # Use element-specific or default values
            magmom = ELEMENT_MAGMOMS.get(z, default_magmom)

            # Apply magnetic ordering to sign
            if is_afm:
                sign = +1 if magnetic_atom_index % 2 == 0 else -1
                magmom = sign * abs(magmom)
            elif is_fm:
                magmom = abs(magmom)  # Ensure positive
            # else: custom - keep as-is

            magmoms.append(magmom)
            magnetic_atom_index += 1
            logger.debug(f"Assigned magnetic moment {magmoms[-1]} to element Z={z}")
        else:
            # Non-magnetic element
            magmoms.append(0.0)

    logger.info(
        f"Assigned magnetic moments to {sum(1 for m in magmoms if m != 0)} atoms "
        f"(ordering={magnetic_ordering})"
    )
    return magmoms


def set_magnetic_ordering(
    structure: Structure,
    ordering: str = "ferromagnetic",
    default_magmom: float | None = None,
    magnetic_species: list[str | int] | None = None,
    afm_pattern: list[int] | None = None,
) -> list[float]:
    """
    Set magnetic ordering for a structure with various magnetic configurations.

    Parameters
    ----------
    structure : Structure
        Pymatgen Structure object
    ordering : str, optional
        Type of magnetic ordering. Options:
        - "ferromagnetic" or "FM": All magnetic moments aligned parallel
        - "antiferromagnetic" or "AFM": Alternating up/down moments
        - "ferrimagnetic" or "FiM": Different magnitudes, antiparallel alignment
        - "custom": User-defined pattern via afm_pattern
        Default: "ferromagnetic"
    default_magmom : float, optional
        Default magnetic moment magnitude. If None, uses element-specific defaults.
    magnetic_species : list of str or int, optional
        List of species (element symbols or atomic numbers) to make magnetic.
        If None, auto-detects all magnetic elements.
    afm_pattern : list of int, optional
        Custom pattern for magnetic ordering. Values should be +1 (up), -1 (down),
        or 0 (non-magnetic).
        Used when ordering="custom". Length must match number of atoms.

    Returns
    -------
    list[float]
        List of initial magnetic moments for each atom

    Examples
    --------
    Ferromagnetic Fe:
    >>> structure = Structure(...)  # Fe structure
    >>> magmoms = set_magnetic_ordering(structure, "FM")
    >>> # All Fe: [+4.0, +4.0, ...]

    Antiferromagnetic NiO (rock salt):
    >>> nio = Structure(...)  # NiO rock salt
    >>> magmoms = set_magnetic_ordering(nio, "AFM")
    >>> # Ni alternating: [+2.0, -2.0, +2.0, -2.0, ...], O: [0.0, 0.0, ...]

    Custom pattern:
    >>> pattern = [+1, -1, +1, -1, 0, 0]  # First 4 atoms AFM, last 2 non-magnetic
    >>> magmoms = set_magnetic_ordering(structure, "custom", afm_pattern=pattern)

    Notes
    -----
    For antiferromagnetic ordering, the function alternates moments based on atom index.
    For complex magnetic structures (e.g., spin spirals), use custom patterns or
    specify moments manually via structure.site_properties["magmoms"].
    """
    logger.info(f"set_magnetic_ordering(ordering={ordering})")

    ordering = ordering.upper()

    # Get atomic numbers and identify magnetic atoms
    atomic_numbers = [_get_site_atomic_number(site) for site in structure]
    n_atoms = len(atomic_numbers)

    # Determine which atoms are magnetic
    if magnetic_species is not None:
        # Convert to set of atomic numbers
        magnetic_z = set()
        for spec in magnetic_species:
            if isinstance(spec, str):
                from pymatgen.core import Element

                magnetic_z.add(Element(spec).Z)
            else:
                magnetic_z.add(spec)
    else:
        # Auto-detect magnetic elements
        magnetic_z = MAGNETIC_ELEMENTS

    # Get base magnetic moments
    base_magmoms = []
    for z in atomic_numbers:
        if z in magnetic_z:
            if default_magmom is not None:
                magmom = default_magmom
            else:
                magmom = ELEMENT_MAGMOMS.get(z, 1.0)
            base_magmoms.append(magmom)
        else:
            base_magmoms.append(0.0)

    # Apply magnetic ordering
    magmoms = []

    if ordering in ["FERROMAGNETIC", "FM"]:
        # All moments parallel (positive)
        magmoms = base_magmoms.copy()
        n_up_atoms = sum(1 for m in magmoms if m > 0)
        logger.info(f"Applied ferromagnetic ordering: {n_up_atoms} spin-up atoms")

    elif ordering in ["ANTIFERROMAGNETIC", "AFM"]:
        # Alternating up/down for magnetic atoms
        magnetic_count = 0
        for base_mag in base_magmoms:
            if base_mag != 0:
                # Alternate sign for magnetic atoms
                sign = 1 if (magnetic_count % 2 == 0) else -1
                magmoms.append(sign * base_mag)
                magnetic_count += 1
            else:
                magmoms.append(0.0)

        n_up = sum(1 for m in magmoms if m > 0)
        n_down = sum(1 for m in magmoms if m < 0)
        logger.info(
            f"Applied antiferromagnetic ordering: {n_up} spin-up, "
            f"{n_down} spin-down atoms"
        )

    elif ordering in ["FERRIMAGNETIC", "FIM"]:
        # Different sublattices with different moments
        # For simplicity: alternating with different magnitudes
        # Users should customize this for specific materials
        magnetic_count = 0
        for base_mag in base_magmoms:
            if base_mag != 0:
                if magnetic_count % 2 == 0:
                    # First sublattice: full moment, spin up
                    magmoms.append(base_mag)
                else:
                    # Second sublattice: reduced moment, spin down
                    magmoms.append(-base_mag * 0.5)
                magnetic_count += 1
            else:
                magmoms.append(0.0)

        logger.info("Applied ferrimagnetic ordering with unequal sublattice moments")

    elif ordering == "CUSTOM":
        # User-defined pattern
        if afm_pattern is None:
            raise ValueError("Must provide afm_pattern for custom magnetic ordering")
        if len(afm_pattern) != n_atoms:
            raise ValueError(
                f"afm_pattern length ({len(afm_pattern)}) must match number "
                f"of atoms ({n_atoms})"
            )

        magmoms = [
            pattern * base_mag
            for pattern, base_mag in zip(afm_pattern, base_magmoms, strict=False)
        ]
        logger.info(f"Applied custom magnetic ordering: {afm_pattern}")

    else:
        raise ValueError(
            f"Unknown magnetic ordering '{ordering}'. "
            f"Options: 'ferromagnetic'/'FM', 'antiferromagnetic'/'AFM', "
            f"'ferrimagnetic'/'FiM', 'custom'"
        )

    return magmoms


def get_magnetic_structure_info(structure: Structure) -> dict:
    """
    Analyze a structure and return information about magnetic elements.

    Parameters
    ----------
    structure : Structure
        Pymatgen Structure object

    Returns
    -------
    dict
        Dictionary containing:
        - 'has_magnetic': bool - Whether structure contains magnetic elements
        - 'magnetic_elements': list - List of magnetic element symbols
        - 'magnetic_indices': list - Indices of magnetic atoms
        - 'n_magnetic': int - Number of magnetic atoms
        - 'suggested_moments': list - Suggested initial moments
        - 'suggested_ordering': str - Suggested magnetic ordering type

    Examples
    --------
    >>> info = get_magnetic_structure_info(nio_structure)
    >>> print(info["magnetic_elements"])
    ['Ni']
    >>> print(info["suggested_ordering"])
    'antiferromagnetic'
    """
    logger.info("get_magnetic_structure_info()")

    atomic_numbers = [_get_site_atomic_number(site) for site in structure]

    # Find magnetic atoms
    magnetic_indices = [
        i for i, z in enumerate(atomic_numbers) if z in MAGNETIC_ELEMENTS
    ]
    magnetic_elements = list({structure[i].specie.symbol for i in magnetic_indices})

    has_magnetic = len(magnetic_indices) > 0
    n_magnetic = len(magnetic_indices)

    # Get suggested moments
    suggested_moments = (
        get_default_initial_magnetic_moments(structure) if has_magnetic else None
    )

    # Suggest ordering based on structure
    suggested_ordering = "ferromagnetic"  # Default
    if has_magnetic:
        # Check if it's a binary oxide (common AFM materials)
        composition = structure.composition
        if len(composition.elements) == 2:
            elements = [el.symbol for el in composition.elements]
            if "O" in elements:
                # Metal oxide - likely antiferromagnetic
                suggested_ordering = "antiferromagnetic"

    info = {
        "has_magnetic": has_magnetic,
        "magnetic_elements": sorted(magnetic_elements),
        "magnetic_indices": magnetic_indices,
        "n_magnetic": n_magnetic,
        "suggested_moments": suggested_moments,
        "suggested_ordering": suggested_ordering,
    }

    logger.info(f"Found {n_magnetic} magnetic atoms: {magnetic_elements}")
    return info


def pymatgen_to_ase(
    structure: Structure,
    ghost_tags: list[bool] | None = None,
    auto_magmoms: bool = True,
    magnetic_ordering: str | None = None,
) -> Atoms:
    """
    Convert a Pymatgen Structure object to an ASE Atoms object.

    Parameters
    ----------
    structure : Structure
        A Pymatgen Structure object
    ghost_tags : list of bool, optional
        List indicating which atoms are ghost atoms.
        If None, the function will check for 'ghost_tags' in structure.site_properties.
    auto_magmoms : bool, optional
        If True, automatically detect and set initial magnetic moments for
        magnetic elements.
        Default: True
    magnetic_ordering : str, optional
        Type of magnetic ordering to apply. Options: "FM", "AFM", "FiM", "custom".
        If None and auto_magmoms=True, uses ferromagnetic ordering.
        Default: None

    Returns
    -------
    ase.Atoms
        An ASE Atoms object with adjusted atomic numbers for ghost atoms and
        initial magnetic moments set for magnetic elements.

    Examples
    --------
    Ferromagnetic ordering (default):
    >>> ase_atoms = pymatgen_to_ase(fe_structure)

    Antiferromagnetic ordering:
    >>> ase_atoms = pymatgen_to_ase(nio_structure, magnetic_ordering="AFM")

    See Also
    --------
    set_magnetic_ordering : Set custom magnetic orderings
    """
    logger.info("pymatgen_to_ase()")
    # Add ghost_tags as a site property if provided
    if ghost_tags is not None:
        structure.add_site_property("ghost_tags", ghost_tags)

    # Extract lattice and atomic positions from Pymatgen structure
    lattice_matrix = (
        structure.lattice.matrix
    )  # 3x3 array representing the lattice vectors
    atomic_positions = structure.frac_coords  # Fractional atomic coordinates

    # Get atomic numbers
    # Get atomic numbers - handles both regular sites and fractional occupancy
    # (ghost atoms)
    atomic_numbers = [_get_site_atomic_number(site) for site in structure]

    # Check if 'ghost_tags' is available in site properties
    if "ghost_tags" in structure.site_properties:
        ghost_tags = cast("list[bool]", structure.site_properties["ghost_tags"])
        # Adjust atomic numbers based on ghost tags
        atomic_numbers = [
            -num if ghost else num
            for num, ghost in zip(atomic_numbers, ghost_tags, strict=False)
        ]
    else:
        # If ghost_tags are not available, show a message and proceed without them
        logger.info(
            "The structure does not have 'ghost_tags' site property. "
            "Proceeding without ghost atoms."
        )

    # Convert fractional coordinates to Cartesian coordinates
    cartesian_positions = np.dot(atomic_positions, lattice_matrix)

    # Create ASE Atoms object
    ase_atoms = Atoms(
        numbers=atomic_numbers,
        positions=cartesian_positions,
        cell=lattice_matrix,
        pbc=True,
    )

    # Handle species_label site property for species variants
    # This enables O_surface, O_bulk, Ti_surface, Ti_bulk, etc.
    if "species_label" in structure.site_properties:
        species_labels = structure.site_properties["species_label"]

        # Create species_dict and species_Z_dict for ASE integration
        # species_dict: maps species number (1-based) to species label
        # species_Z_dict: maps species number (1-based) to atomic number
        unique_labels = sorted(set(species_labels))
        species_dict = {}
        species_Z_dict = {}  # noqa: N806  Z = atomic number

        for idx, label in enumerate(unique_labels, start=1):
            # Find first atom with this label to get atomic number
            for site_label, atom_num in zip(
                species_labels, atomic_numbers, strict=False
            ):
                if site_label == label:
                    species_dict[idx] = label
                    species_Z_dict[idx] = atom_num
                    break

        # Create species_labels list for each atom (same order as atomic_numbers)
        # This is used to assign species numbers
        ase_atoms.info["species_dict"] = species_dict
        ase_atoms.info["species_Z_dict"] = species_Z_dict
        ase_atoms.info["species_labels"] = species_labels

        logger.info(f"Set species variants from site property: {unique_labels}")

    # Automatically set initial magnetic moments if requested
    if auto_magmoms:
        # Check if user already provided magnetic moments in structure properties
        if "magmoms" in structure.site_properties:
            magmoms = structure.site_properties["magmoms"]
            logger.info(
                "Using user-provided magnetic moments from structure.site_properties"
            )
        # Apply magnetic ordering if specified
        elif magnetic_ordering is not None:
            magmoms = set_magnetic_ordering(structure, ordering=magnetic_ordering)
        else:
            # Auto-detect magnetic elements and assign ferromagnetic moments
            magmoms = get_default_initial_magnetic_moments(structure)

        if magmoms is not None:
            ase_atoms.set_initial_magnetic_moments(magmoms)
            n_mag = len([m for m in magmoms if m != 0])
            ordering_type = magnetic_ordering or "FM"
            logger.info(
                f"Set {ordering_type} magnetic moments for {n_mag} magnetic atoms"
            )

    return ase_atoms


def pymatgen_to_ase_v2(structure: Structure, auto_magmoms: bool = True) -> Atoms:
    """
    Convert a pymatgen Structure to an ASE Atoms object (version 2).

    This is a simplified conversion function using pymatgen's built-in ASE adaptor.

    Parameters
    ----------
    structure : Structure
        Pymatgen Structure object to convert
    auto_magmoms : bool, optional
        If True, automatically detect and set initial magnetic moments for
        magnetic elements.
        Default: True

    Returns
    -------
    ase.Atoms
        ASE Atoms object with the same atomic positions, cell, and species,
        and initial magnetic moments set for magnetic elements.

    See Also
    --------
    pymatgen_to_ase : Alternative conversion with ghost atom support
    ase_v2_to_pymatgen : Reverse conversion
    """
    logger.info("pymatgen_to_ase_v2()")
    from pymatgen.io.ase import AseAtomsAdaptor

    # Convert pymatgen structure to ASE Atoms object
    ase_atoms = AseAtomsAdaptor.get_atoms(structure)

    # Automatically set initial magnetic moments if requested
    if auto_magmoms:
        # Check if user already provided magnetic moments in structure properties
        if "magmoms" in structure.site_properties:
            magmoms = structure.site_properties["magmoms"]
            logger.info(
                "Using user-provided magnetic moments from structure.site_properties"
            )
        else:
            # Auto-detect magnetic elements and assign moments
            magmoms = get_default_initial_magnetic_moments(structure)

        if magmoms is not None:
            ase_atoms.set_initial_magnetic_moments(magmoms)
            n_mag = len([m for m in magmoms if m != 0])
            logger.info(f"Set initial magnetic moments for {n_mag} magnetic atoms")

    return ase_atoms


def ase_v2_to_pymatgen(ase_atoms: Atoms) -> Structure:
    """
    Convert an ASE Atoms object to a pymatgen Structure (version 2).

    This is a simplified conversion function using pymatgen's built-in ASE adaptor.

    Parameters
    ----------
    ase_atoms : ase.Atoms
        ASE Atoms object to convert

    Returns
    -------
    Structure
        Pymatgen Structure object with the same atomic positions, cell, and species

    See Also
    --------
    pymatgen_to_ase_v2 : Reverse conversion
    """
    logger.info("ase_v2_to_pymatgen()")
    from pymatgen.io.ase import AseAtomsAdaptor

    # Convert ASE Atoms to pymatgen Structure
    return AseAtomsAdaptor.get_structure(ase_atoms)


def pymatgen_to_sisl(
    structure: Structure, ghost_tags: list[bool] | None = None
) -> sisl.Geometry:
    """
    Convert a Pymatgen Structure object to a sisl Geometry object.

    Parameters
    ----------
    structure (pymatgen.core.Structure): A Pymatgen Structure object.

    Returns
    -------
    sisl.Geometry: A sisl Geometry object with adjusted atomic numbers for ghost atoms.
    """
    logger.info("pymatgen_to_sisl()")
    if ghost_tags is not None:
        structure.add_site_property("ghost_tags", ghost_tags)

    # Extract lattice and atomic positions from Pymatgen structure
    lattice_matrix = (
        structure.lattice.matrix
    )  # 3x3 array representing the lattice vectors
    atomic_positions = structure.frac_coords  # Fractional atomic coordinates

    # Get atomic numbers and adjust based on ghost tags
    atomic_numbers = [_get_site_atomic_number(site) for site in structure]

    # Check if 'ghost_tags' is available in site properties
    if "ghost_tags" in structure.site_properties:
        ghost_tags = cast("list[bool]", structure.site_properties["ghost_tags"])
        atomic_numbers = [
            -num if ghost else num
            for num, ghost in zip(atomic_numbers, ghost_tags, strict=False)
        ]
    else:
        logger.info(
            "The structure does not have 'ghost_tags' site property. "
            "Proceeding without ghost atoms."
        )
        # Optionally, you could also use a warning
        # warnings.warn("The structure does not have 'ghost_tags' site "
        #               "property. Proceeding without ghost atoms.")
        ghost_tags = [False] * len(atomic_numbers)  # No ghost atoms
        # raise ("The structure does not have 'ghost_tags' site property.")
        # pass

    # Create a list of sisl Atom objects
    atoms = [sisl.Atom(an) for an in atomic_numbers]

    # Create sisl Geometry object
    return sisl.Geometry(atomic_positions, atoms, lattice=lattice_matrix)


def read_outvars(file_path: str) -> dict | None:
    """
    Read the OUTVARS.yml file and return its contents as a dictionary.

    Parameters
    ----------
    file_path : str
        The path to the OUTVARS.yml file.

    Returns
    -------
    dict
        A dictionary containing the contents of the YAML file.
    """
    logger.info("read_outvars()")
    try:
        with open(file_path) as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        logger.exception(f"The file {file_path} does not exist.")
    except yaml.YAMLError:
        logger.exception("Error reading the YAML file")
    else:
        return data
    return None


def siesta_fdf_to_json(
    siesta_fdf_path: str | Path,
    json_output_path: str | Path,
    fdf_data: dict | None = None,
) -> None:
    """
    Convert a SIESTA FDF (Flexible Data Format) file to JSON format.

    Reads a SIESTA input file and extracts key parameters and settings into a JSON file.
    Uses sisl to parse the FDF file.

    Parameters
    ----------
    siesta_fdf_path : str or Path
        Path to the SIESTA FDF input file
    json_output_path : str or Path
        Path where the JSON output file will be written
    fdf_data : dict, optional
        Pre-existing FDF data dictionary. If None, extracts data from FDF file.

    Notes
    -----
    Extracts common SIESTA parameters including:
    - SCF settings (convergence, mixer parameters)
    - XC functional and authors
    - Mesh cutoff and basis parameters
    - System information (species, atoms, lattice)
    - K-point sampling
    - Initial spin configuration
    """
    logger.info("siesta_fdf_to_json()")
    # Read the FDF file using sisl
    siesta_fdf = sisl.get_sile(siesta_fdf_path)

    if fdf_data is None:
        # Create a dictionary to store the extracted information
        fdf_data = {}

        # Extracting simple key-value pairs
        fdf_data["SCFMustConverge"] = siesta_fdf.get("SCFMustConverge")
        fdf_data["Spin"] = siesta_fdf.get("Spin")
        fdf_data["XC.functional"] = siesta_fdf.get("XC.functional")
        fdf_data["XC.authors"] = siesta_fdf.get("XC.authors")
        fdf_data["MeshCutoff"] = siesta_fdf.get("MeshCutoff")
        fdf_data["PAO.BasisSize"] = siesta_fdf.get("PAO.BasisSize")
        fdf_data["PAO.EnergyShift"] = siesta_fdf.get("PAO.EnergyShift")
        fdf_data["NumberOfSpecies"] = siesta_fdf.get("NumberOfSpecies")
        fdf_data["NumberOfAtoms"] = siesta_fdf.get("NumberOfAtoms")
        fdf_data["LatticeConstant"] = siesta_fdf.get("LatticeConstant")
        fdf_data["AtomicCoordinatesFormat"] = siesta_fdf.get("AtomicCoordinatesFormat")
        fdf_data["DM.UseSaveDM"] = siesta_fdf.get("DM.UseSaveDM")
        fdf_data["SaveRho"] = siesta_fdf.get("SaveRho")
        fdf_data["WriteFoces"] = siesta_fdf.get("WriteFoces")
        fdf_data["LongOutput"] = siesta_fdf.get("LongOutput")

        # Extracting blocks
        fdf_data["ChemicalSpeciesLabel"] = siesta_fdf.get("ChemicalSpeciesLabel")
        fdf_data["PAO.BasisSizes"] = siesta_fdf.get("PAO.BasisSizes")
        fdf_data["LatticeVectors"] = siesta_fdf.get("LatticeVectors")
        fdf_data["AtomicCoordinatesAndAtomicSpecies"] = siesta_fdf.get(
            "AtomicCoordinatesAndAtomicSpecies"
        )
        fdf_data["DM.InitSpin"] = siesta_fdf.get("DM.InitSpin")
        fdf_data["kgrid_Monkhorst_Pack"] = siesta_fdf.get("kgrid_Monkhorst_Pack")

    # Convert the dictionary to a JSON formatted string
    json_data = json.dumps(fdf_data, indent=4)

    # Write the JSON string to a file
    with open(json_output_path, "w") as json_file:
        json_file.write(json_data)


def write_parameter_evolution_log(
    log_file_path: str,
    explicit_user_params: dict,
    initial_params: dict,
    after_dataclass_params: dict,
    final_fdf_params: dict,
    powerup_added: dict = None,
    powerup_modified: dict = None,
    powerup_removed: dict = None,
) -> Path:
    """Write parameter evolution to a log file.

    Args:
        log_file_path: Path to the log file
        explicit_user_params: Explicitly provided user parameters
        initial_params: All initial parameters (user + maker defaults)
        after_dataclass_params: Parameters after dataclass processing
        final_fdf_params: Final FDF parameters
        powerup_added: Parameters added by powerups
        powerup_modified: Parameters modified by powerups
            (dict of {key: (old_val, new_val)})
        powerup_removed: Parameters removed by powerups
    """
    from datetime import datetime
    from pathlib import Path

    log_file = Path(log_file_path)

    with open(log_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SIESTA PARAMETER EVOLUTION LOG\n")
        f.write("=" * 80 + "\n")
        # Local timestamp is intentional for a human-readable log
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # noqa: DTZ005
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 80 + "\n\n")

        # Add legend
        f.write("LEGEND:\n")
        f.write("  [+] = Parameter added\n")
        f.write("  [~] = Parameter modified (shows: new value (was: old value))\n")
        f.write("  [-] = Parameter removed\n")
        f.write("\n")

        # Section 1: Initial User Parameters
        f.write("1. INITIAL USER-PROVIDED PARAMETERS\n")
        f.write("-" * 80 + "\n")
        if explicit_user_params:
            f.write("Explicit User Parameters:\n")
            f.writelines(
                f"  {key.upper():30s} = {value}\n"
                for key, value in explicit_user_params.items()
            )
            f.write("\n")

        # Maker defaults
        maker_defaults = {
            k: v for k, v in initial_params.items() if k not in explicit_user_params
        }
        if maker_defaults:
            f.write("Maker Default Parameters:\n")
            f.writelines(
                f"  {key.upper():30s} = {value}\n"
                for key, value in maker_defaults.items()
            )
            f.write("\n")

        if not explicit_user_params and not maker_defaults:
            f.write("  (No user parameters provided)\n\n")

        # Section 2: After Dataclass Processing
        f.write("2. AFTER DATACLASS PROCESSING\n")
        f.write("-" * 80 + "\n")

        if after_dataclass_params:
            added = {
                k: v
                for k, v in after_dataclass_params.items()
                if k.upper() not in {p.upper() for p in initial_params}
            }
            modified = {
                k: v
                for k, v in after_dataclass_params.items()
                if k.upper() in {p.upper() for p in initial_params}
                and str(v)
                != str(
                    initial_params.get(
                        k, initial_params.get(k.upper(), initial_params.get(k.lower()))
                    )
                )
            }

            if added:
                f.write("Parameters Added by Dataclasses:\n")
                f.writelines(
                    f"  [+] {key.upper():30s} = {value}\n"
                    for key, value in added.items()
                )
                f.write("\n")

            if modified:
                f.write("Parameters Modified by Dataclasses:\n")
                for key, value in modified.items():
                    # Find original value
                    orig_val = None
                    for orig_key, orig_value in initial_params.items():
                        if orig_key.upper() == key.upper():
                            orig_val = orig_value
                            break
                    f.write(f"  [~] {key.upper():30s} = {value} (was: {orig_val})\n")
                f.write("\n")

            if not added and not modified:
                f.write("  (No changes from dataclass processing)\n\n")

        # Section 3: Powerup/Flow Modifications
        f.write("3. POWERUP/FLOW MODIFICATIONS\n")
        f.write("-" * 80 + "\n")

        has_powerup_changes = False
        if powerup_added:
            f.write("Parameters Added by Powerups/Flows:\n")
            for key, value in powerup_added.items():
                f.write(f"  [+] {key.upper():30s} = {value}\n")
            f.write("\n")
            has_powerup_changes = True

        if powerup_modified:
            f.write("Parameters Modified by Powerups/Flows:\n")
            for key, (old_val, new_val) in powerup_modified.items():
                f.write(f"  [~] {key.upper():30s} = {new_val} (was: {old_val})\n")
            f.write("\n")
            has_powerup_changes = True

        if powerup_removed:
            f.write("Parameters Removed by Powerups/Flows:\n")
            for key, value in powerup_removed.items():
                f.write(f"  [-] {key.upper():30s} (was: {value})\n")
            f.write("\n")
            has_powerup_changes = True

        if not has_powerup_changes:
            f.write("  (No powerup/flow modifications)\n\n")

        # Section 4: Final Statistics
        f.write("4. FINAL PARAMETER STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total parameters in FDF:        {len(final_fdf_params)}\n")
        f.write(f"From user (explicit):           {len(explicit_user_params)}\n")
        f.write(f"From maker defaults:            {len(maker_defaults)}\n")
        n_auto = len(final_fdf_params) - len(initial_params)
        f.write(f"Auto-generated by dataclasses:  {n_auto}\n")
        if powerup_added:
            f.write(f"Added by powerups:              {len(powerup_added)}\n")
        if powerup_modified:
            f.write(f"Modified by powerups:           {len(powerup_modified)}\n")
        if powerup_removed:
            f.write(f"Removed by powerups:            {len(powerup_removed)}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF PARAMETER EVOLUTION LOG\n")
        f.write("=" * 80 + "\n")

    return log_file
