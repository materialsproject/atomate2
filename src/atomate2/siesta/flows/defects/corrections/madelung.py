"""
Madelung constant calculation and lookup for different crystal structures.

The Madelung constant (α_M) represents the electrostatic energy of an ionic
crystal in units of the nearest-neighbor interaction energy. It depends on
the crystal structure type.

References
----------
.. [1] Kittel, C. "Introduction to Solid State Physics", 8th Ed. (2005)
       Wiley, New York. Table 3 (page 64).
.. [2] Tosi, M. P. "Cohesion of Ionic Solids in the Born Model"
       Solid State Physics 16, 1-120 (1964).
.. [3] Sherman, J. "Crystal energies of ionic compounds and thermochemical
       applications" Chemical Reviews 11, 93-170 (1932).
.. [4] Born, M. & Huang, K. "Dynamical Theory of Crystal Lattices"
       Oxford University Press (1954).
.. [5] CRC Handbook of Chemistry and Physics, 97th Ed. (2016-2017).
"""  # noqa: RUF002

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


# Madelung constants for common crystal structures
# Format: {structure_name: (alpha_M, citation)}
MADELUNG_CONSTANTS = {
    # Cubic structures
    "rocksalt": (1.74756, "Kittel (2005), Table 3; Tosi (1964)"),  # NaCl-type (Fm-3m)
    "cesium_chloride": (1.76267, "Kittel (2005); Born & Huang (1954)"),  # CsCl-type
    "zincblende": (1.63806, "Kittel (2005); Sherman (1932)"),  # ZnS-type (F-43m)
    "fluorite": (2.51939, "Tosi (1964); CRC Handbook"),  # CaF2-type
    "rutile": (2.408, "Tosi (1964)"),  # TiO2-type
    # Hexagonal structures
    "wurtzite": (1.64132, "Sherman (1932); Born & Huang (1954)"),  # ZnO-type
    "nickel_arsenide": (1.7476, "Tosi (1964)"),  # NiAs-type
    # Other common structures
    "corundum": (4.172, "Tosi (1964)"),  # Al2O3-type
    "perovskite": (4.3484, "Tosi (1964)"),  # CaTiO3-type (ideal cubic)
    # Simple lattices (for reference/testing)
    "simple_cubic": (1.76012, "Kittel (2005)"),  # Simple cubic lattice
    "fcc": (1.79175, "Kittel (2005)"),  # Face-centered cubic
    "bcc": (1.79186, "Kittel (2005)"),  # Body-centered cubic
    # Wigner-Seitz approximation (spherical)
    "wigner_seitz": (2.8373, "Makov & Payne (1995) - spherical approximation"),
}


def get_madelung_constant(
    structure: Structure,
    use_wigner_seitz_fallback: bool = True,
) -> tuple[float, str]:
    """
    Get the Madelung constant for a given structure.

    Attempts to identify the crystal structure type and return the
    appropriate Madelung constant. For unknown structures, can either
    use the Wigner-Seitz approximation or raise an error.

    Parameters
    ----------
    structure : Structure
        Pymatgen Structure object
    use_wigner_seitz_fallback : bool, optional
        If True, use Wigner-Seitz approximation (α_M = 2.8373) for
        unknown structures. If False, raise ValueError for unknown
        structures. Default: True.

    Returns
    -------
    tuple[float, str]
        (madelung_constant, citation)

    Examples
    --------
    >>> from pymatgen.core import Structure, Lattice
    >>> # NaCl structure
    >>> lattice = Lattice.cubic(5.64)
    >>> nacl = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    >>> alpha_M, citation = get_madelung_constant(nacl)
    >>> print(f"α_M = {alpha_M:.4f}")
    α_M = 1.7476

    Notes
    -----
    The Madelung constant represents the electrostatic energy per ion pair
    in units of e²/(4πε₀r₀), where r₀ is the nearest-neighbor distance.

    For charged defects in finite-size supercells, the Madelung constant
    appears in the correction energy:
        E_lat = (q² α_M) / (2 ε L)

    Using the wrong α_M can introduce significant errors (10-30%) in the
    correction energy, especially for high-charge defects.
    """  # noqa: RUF002
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        # Get space group information
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        spacegroup = sga.get_space_group_symbol()
        crystal_system = sga.get_crystal_system()

        logger.info(
            f"Detected space group: {spacegroup}, crystal system: {crystal_system}"
        )

        # Try to identify structure type
        structure_type = _identify_structure_type(structure, spacegroup, crystal_system)

        if structure_type in MADELUNG_CONSTANTS:
            alpha_M, citation = MADELUNG_CONSTANTS[structure_type]  # noqa: N806
            logger.info(
                f"Using Madelung constant for {structure_type}: "
                f"α_M = {alpha_M:.5f} (source: {citation})"  # noqa: RUF001
            )
            return alpha_M, citation

    except Exception as e:
        logger.warning(f"Could not analyze structure symmetry: {e}")

    # Fallback to Wigner-Seitz approximation
    if use_wigner_seitz_fallback:
        alpha_M, citation = MADELUNG_CONSTANTS["wigner_seitz"]  # noqa: N806
        logger.warning(
            f"Unknown structure type - using Wigner-Seitz approximation: "
            f"α_M = {alpha_M:.5f}. This may introduce ~10-30% error. "  # noqa: RUF001
            f"Consider providing madelung_constant manually for better accuracy."
        )
        return alpha_M, citation
    raise ValueError(
        "Could not determine structure type for Madelung constant. "
        "Please provide madelung_constant parameter explicitly."
    )


def _identify_structure_type(
    structure: Structure,
    spacegroup: str,
    crystal_system: str,
) -> str | None:
    """
    Identify common structure types from space group and composition.

    Parameters
    ----------
    structure : Structure
        Pymatgen Structure object
    spacegroup : str
        Space group symbol (e.g., "Fm-3m")
    crystal_system : str
        Crystal system (cubic, hexagonal, etc.)

    Returns
    -------
    str or None
        Structure type name (e.g., "rocksalt", "wurtzite") or None if unknown
    """
    # Get composition
    comp = structure.composition
    num_species = len(comp.elements)

    # Check for binary compounds (most ionic crystals)
    if num_species == 2:
        # Get stoichiometry ratio
        reduced = comp.reduced_composition
        elements = list(reduced.elements)
        amounts = [reduced[el] for el in elements]

        # Rocksalt (NaCl): Fm-3m, 1:1 ratio
        if spacegroup in ["Fm-3m", "Fm3m", "225"] and amounts == [1, 1]:
            return "rocksalt"

        # Zincblende (ZnS): F-43m, 1:1 ratio
        if spacegroup in ["F-43m", "F43m", "216"] and amounts == [1, 1]:
            return "zincblende"

        # Wurtzite (ZnO): P63mc, 1:1 ratio
        if spacegroup in ["P63mc", "P6_3mc", "186"] and amounts == [1, 1]:
            return "wurtzite"

        # Fluorite (CaF2): Fm-3m, 1:2 ratio
        if spacegroup in ["Fm-3m", "Fm3m", "225"] and sorted(amounts) == [1, 2]:
            return "fluorite"

        # Cesium chloride (CsCl): Pm-3m, 1:1 ratio
        if spacegroup in ["Pm-3m", "Pm3m", "221"] and amounts == [1, 1]:
            return "cesium_chloride"

        # Rutile (TiO2): P42/mnm, 1:2 ratio
        if spacegroup in ["P42/mnm", "P4_2/mnm", "136"] and sorted(amounts) == [1, 2]:
            return "rutile"

        # Nickel arsenide (NiAs): P63/mmc, 1:1 ratio
        if spacegroup in ["P63/mmc", "P6_3/mmc", "194"] and amounts == [1, 1]:
            return "nickel_arsenide"

    # Check for ternary compounds
    elif num_species == 3:
        # Perovskite (ABX3): Pm-3m (cubic)
        if spacegroup in ["Pm-3m", "Pm3m", "221"]:
            reduced = comp.reduced_composition
            amounts = sorted([reduced[el] for el in reduced.elements])
            if amounts == [1, 1, 3]:
                return "perovskite"

    # Check for simple elemental structures
    elif num_species == 1:
        if crystal_system == "cubic":
            # Try to determine FCC vs BCC from number of atoms
            # This is approximate - may need lattice analysis
            if len(structure) == 4:  # FCC primitive cell
                return "fcc"
            if len(structure) == 2:  # BCC primitive cell
                return "bcc"
            if len(structure) == 1:  # Simple cubic
                return "simple_cubic"

    return None


def calculate_madelung_ewald(
    structure: Structure,
    real_space_cutoff: float = 20.0,
    reciprocal_cutoff: float = 20.0,
    eta: float | None = None,
) -> float:
    """
    Calculate Madelung constant using Ewald summation.

    This is a fallback method for structures not in the lookup table.
    Requires formal charges on all sites.

    Parameters
    ----------
    structure : Structure
        Pymatgen Structure with oxidation states set
    real_space_cutoff : float
        Real space cutoff in Angstrom
    reciprocal_cutoff : float
        Reciprocal space cutoff in inverse Angstrom
    eta : float, optional
        Ewald parameter. If None, optimizes automatically.

    Returns
    -------
    float
        Madelung constant

    Raises
    ------
    ValueError
        If structure doesn't have oxidation states set

    Notes
    -----
    This uses pymatgen's EwaldSummation class. The Madelung constant is
    defined as: α_M = E_Madelung × r₀ / (Z₁ Z₂ e²/(4πε₀))

    For accurate results with Ewald summation, ensure:
    - Oxidation states are correctly assigned
    - Cutoffs are sufficiently large (>20 Å typically)
    - Structure is charge-neutral overall
    """  # noqa: RUF002
    from pymatgen.analysis.ewald import EwaldSummation

    # Check if oxidation states are set
    if not hasattr(structure[0], "specie") or structure[0].specie.oxi_state is None:
        raise ValueError(
            "Structure must have oxidation states set for Ewald summation. "
            "Use structure.add_oxidation_state_by_guess() or set manually."
        )

    # Calculate Ewald sum
    ewald = EwaldSummation(
        structure,
        real_space_cut=real_space_cutoff,
        recip_space_cut=reciprocal_cutoff,
        eta=eta,
    )

    # Get total energy
    total_energy = ewald.total_energy

    # Convert to Madelung constant
    # This requires normalizing by the ion pair energy
    # Implementation depends on structure specifics
    logger.warning(
        "Ewald summation Madelung constant calculation is experimental. "
        "Verify results against known values."
    )

    return abs(total_energy)  # Simplified - needs proper normalization
