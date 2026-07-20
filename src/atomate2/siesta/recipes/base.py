"""Base infrastructure for workflow recipes."""
# ruff: noqa: T201  print() is the intentional user-facing output of the recipe book

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


@dataclass
class MaterialAnalysis:
    """Results of automatic structure analysis for smart defaults."""

    # Basic properties
    formula: str
    num_atoms: int
    volume: float
    density: float

    # Electronic properties
    is_metal: bool
    has_heavy_elements: bool
    max_z: int  # Highest atomic number

    # Structural properties
    space_group: int
    crystal_system: str
    is_layered: bool
    has_surfaces: bool

    # Magnetic properties
    has_magnetic_elements: bool
    suggested_spin: bool

    # Recommended settings
    recommended_kpts: list[int]
    recommended_cutoff: str
    recommended_basis: str
    recommended_tier: str
    recommended_preset: str | None

    # Computational estimates
    estimated_time_hours: float
    estimated_memory_gb: float
    recommended_cores: int


class MaterialAnalyzer:
    """Analyze structure and recommend optimal SIESTA parameters."""

    # Element classification
    METALS: ClassVar[set[int]] = {
        3,
        4,
        11,
        12,
        13,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        55,
        56,
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
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
    }

    MAGNETIC_ELEMENTS: ClassVar[set[int]] = {
        24,
        25,
        26,
        27,
        28,  # Cr, Mn, Fe, Co, Ni
        42,
        43,
        44,
        45,  # Mo, Tc, Ru, Rh
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
        70,
        71,  # Lanthanides
    }

    HEAVY_ELEMENTS: ClassVar[set[int]] = set(range(37, 87))  # Rb to Fr

    @classmethod
    def analyze(cls, structure: Structure) -> MaterialAnalysis:
        """
        Analyze structure and return recommended parameters.

        Parameters
        ----------
        structure : Structure
            The structure to analyze.

        Returns
        -------
        MaterialAnalysis
            Comprehensive analysis with recommended settings.
        """
        # Basic properties
        formula = structure.composition.reduced_formula
        num_atoms = len(structure)
        volume = structure.volume
        density = structure.density

        # Get atomic numbers
        atomic_numbers = {el.Z for el in structure.composition.elements}
        max_z = max(atomic_numbers)

        # Electronic properties
        is_metal = any(z in cls.METALS for z in atomic_numbers)
        has_heavy_elements = any(z in cls.HEAVY_ELEMENTS for z in atomic_numbers)

        # Magnetic properties
        has_magnetic_elements = any(z in cls.MAGNETIC_ELEMENTS for z in atomic_numbers)
        suggested_spin = has_magnetic_elements

        # Structural analysis
        try:
            sga = SpacegroupAnalyzer(structure)
            space_group = sga.get_space_group_number()
            crystal_system = sga.get_crystal_system()
        except Exception:  # noqa: BLE001 symmetry analysis may fail; fall back to P1
            space_group = 1
            crystal_system = "triclinic"

        # Check if layered (c-axis much larger than a,b)
        lattice = structure.lattice
        abc = lattice.abc
        is_layered = abc[2] > 1.5 * max(abc[0], abc[1]) if len(abc) == 3 else False
        has_surfaces = is_layered

        # Recommend k-points based on system size and type
        kpts = cls._recommend_kpts(structure, is_metal)

        # Recommend mesh cutoff
        cutoff = cls._recommend_cutoff(has_heavy_elements, max_z)

        # Recommend basis size
        basis = cls._recommend_basis(num_atoms, is_metal)

        # Recommend tier and preset
        tier, preset = cls._recommend_tier_preset(
            is_metal, has_magnetic_elements, is_layered
        )

        # Estimate computational cost
        est_time, est_mem, cores = cls._estimate_cost(num_atoms, kpts, is_metal)

        return MaterialAnalysis(
            formula=formula,
            num_atoms=num_atoms,
            volume=volume,
            density=density,
            is_metal=is_metal,
            has_heavy_elements=has_heavy_elements,
            max_z=max_z,
            space_group=space_group,
            crystal_system=crystal_system,
            is_layered=is_layered,
            has_surfaces=has_surfaces,
            has_magnetic_elements=has_magnetic_elements,
            suggested_spin=suggested_spin,
            recommended_kpts=kpts,
            recommended_cutoff=cutoff,
            recommended_basis=basis,
            recommended_tier=tier,
            recommended_preset=preset,
            estimated_time_hours=est_time,
            estimated_memory_gb=est_mem,
            recommended_cores=cores,
        )

    @staticmethod
    def _recommend_kpts(structure: Structure, is_metal: bool) -> list[int]:
        """Recommend k-point mesh based on reciprocal lattice."""
        # Target ~0.03 Å^-1 spacing, denser for metals
        target_spacing = 0.025 if is_metal else 0.035

        reciprocal_lattice = structure.lattice.reciprocal_lattice
        kpts = []
        for vec in reciprocal_lattice.abc:
            k = max(1, int(np.ceil(vec / (2 * np.pi * target_spacing))))
            # Round to reasonable values
            if k <= 2:
                k = 2
            elif k <= 4:
                k = 4
            elif k <= 6:
                k = 6
            elif k <= 8:
                k = 8
            elif k <= 12:
                k = 12
            else:
                k = min(16, k)
            kpts.append(k)

        return kpts

    @staticmethod
    def _recommend_cutoff(has_heavy_elements: bool, max_z: int) -> str:
        """Recommend mesh cutoff based on elements."""
        if max_z > 50:  # Very heavy elements
            return "500 Ry"
        if has_heavy_elements:
            return "400 Ry"
        if max_z > 18:  # 3rd row and beyond
            return "350 Ry"
        # Light elements (H-Ar)
        return "300 Ry"

    @staticmethod
    def _recommend_basis(num_atoms: int, is_metal: bool) -> str:
        """Recommend basis size based on system size and type."""
        if num_atoms > 100:
            return "SZ" if not is_metal else "SZP"
        if num_atoms > 50:
            return "DZ" if not is_metal else "DZP"
        return "DZP"

    @staticmethod
    def _recommend_tier_preset(
        is_metal: bool, has_magnetic: bool, is_layered: bool
    ) -> tuple[str, str | None]:
        """Recommend tier and preset based on material properties."""
        if is_layered:
            # For layered/2D materials, use appropriate surface preset
            if has_magnetic:
                return "intermediate", "magnetic_2d"
            return (
                "intermediate",
                "surface_metal" if is_metal else "surface_semiconductor",
            )
        if is_metal and has_magnetic:
            return "intermediate", "magnetic_correlated"
        if is_metal:
            return "intermediate", "relax_bulk_metal"
        if has_magnetic:
            return "intermediate", None
        return "basic", "relax_standard"

    @staticmethod
    def _estimate_cost(
        num_atoms: int, kpts: list[int], is_metal: bool
    ) -> tuple[float, float, int]:
        """Estimate computational cost (time, memory, cores)."""
        # Very rough estimates based on experience
        nkpts = np.prod(kpts)

        # Base time per atom per k-point (minutes)
        base_time = 0.5 if not is_metal else 1.0

        # SCF iterations (more for metals)
        scf_iters = 50 if is_metal else 30

        # Estimated time in hours
        est_time = (num_atoms * nkpts * base_time * scf_iters) / 60

        # Memory estimate (GB) - roughly 100 MB per atom per k-point
        est_mem = (num_atoms * nkpts * 0.1) / 1024

        # Recommended cores
        if num_atoms > 100:
            cores = 16
        elif num_atoms > 50:
            cores = 8
        elif num_atoms > 20:
            cores = 4
        else:
            cores = 2

        return est_time, est_mem, cores


class RecipeBook:
    """
    Collection of high-level workflow recipes.

    This class provides a unified interface to all recipe functions,
    allowing users to access recipes through a single entry point.

    Example:
        >>> from atomate2.siesta.recipes import RecipeBook
        >>> flow = RecipeBook.complete_material_study(structure)
    """

    @staticmethod
    def analyze_structure(structure: Structure) -> MaterialAnalysis:
        """
        Analyze structure and get recommended parameters.

        Parameters
        ----------
        structure : Structure
            Structure to analyze.

        Returns
        -------
        MaterialAnalysis
            Comprehensive analysis with recommendations.
        """
        return MaterialAnalyzer.analyze(structure)

    @staticmethod
    def print_analysis(structure: Structure, detailed: bool = False) -> None:
        """
        Print structure analysis and recommendations.

        Parameters
        ----------
        structure : Structure
            Structure to analyze.
        detailed : bool
            If True, show computational cost estimates (time, memory, cores).
            Default: False (estimates hidden, as they are rough heuristics).

        Notes
        -----
        Computational estimates are very rough order-of-magnitude guesses based on
        simple heuristics. Actual time/memory can vary significantly based on:
        - Hardware speed and architecture
        - Basis set and cutoff settings
        - System complexity and convergence difficulty
        - Parallelization efficiency

        Use estimates only for rough planning, not for accurate resource allocation.
        """
        analysis = MaterialAnalyzer.analyze(structure)

        print("\n" + "=" * 70)
        print(f"Material Analysis: {analysis.formula}")
        print("=" * 70)

        print("\n📊 Basic Properties:")
        print(f"  - Formula: {analysis.formula}")
        print(f"  - Atoms: {analysis.num_atoms}")
        print(f"  - Volume: {analysis.volume:.2f} ų")
        print(f"  - Density: {analysis.density:.2f} g/cm³")

        print("\n🔬 Electronic Properties:")
        print(
            f"  - Type: {'Metal' if analysis.is_metal else 'Insulator/Semiconductor'}"
        )
        magnetic = "Yes" if analysis.has_magnetic_elements else "No"
        print(f"  - Magnetic elements: {magnetic}")
        print(f"  - Heavy elements: {'Yes' if analysis.has_heavy_elements else 'No'}")
        print(f"  - Max Z: {analysis.max_z}")

        print("\n🔮 Structural Properties:")
        print(f"  - Space group: {analysis.space_group}")
        print(f"  - Crystal system: {analysis.crystal_system}")
        print(f"  - Layered: {'Yes' if analysis.is_layered else 'No'}")

        print("\n⚙️ Recommended SIESTA Settings:")
        print(f"  - K-points: {analysis.recommended_kpts}")
        print(f"  - Mesh cutoff: {analysis.recommended_cutoff}")
        print(f"  - Basis size: {analysis.recommended_basis}")
        print(f"  - Tier: {analysis.recommended_tier}")
        if analysis.recommended_preset:
            print(f"  - Preset: {analysis.recommended_preset}")

        if detailed:
            print("\n💰 Computational Estimates (⚠️ Very Rough Heuristics):")
            print(f"  - Est. time: {analysis.estimated_time_hours:.1f} hours")
            print(f"  - Est. memory: {analysis.estimated_memory_gb:.1f} GB")
            print(f"  - Recommended cores: {analysis.recommended_cores}")
            print("  [Note: Order-of-magnitude guesses only, not accurate predictions]")

        print("\n" + "=" * 70 + "\n")


# Dynamic method attachment - methods are added in __init__.py after all modules load
# This avoids circular import issues while maintaining the RecipeBook interface
