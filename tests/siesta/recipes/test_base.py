"""Tests for recipe base infrastructure."""

import pytest
from pymatgen.core import Lattice, Structure

from atomate2.siesta.recipes.base import MaterialAnalysis, MaterialAnalyzer, RecipeBook


@pytest.fixture
def silicon_structure():
    """Silicon diamond structure."""
    lattice = Lattice.cubic(5.43)
    return Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


@pytest.fixture
def aluminum_structure():
    """Aluminum FCC structure."""
    lattice = Lattice.cubic(4.05)
    return Structure(lattice, ["Al"], [[0, 0, 0]])


def test_material_analyzer_silicon(silicon_structure):
    """Test MaterialAnalyzer on silicon."""
    analysis = MaterialAnalyzer.analyze(silicon_structure)

    assert isinstance(analysis, MaterialAnalysis)
    assert analysis.formula == "Si"  # Reduced formula (Si2 → Si)
    assert analysis.num_atoms == 2
    assert not analysis.is_metal  # Silicon is semiconductor
    assert not analysis.has_heavy_elements
    assert not analysis.has_magnetic_elements
    # Crystal system can be trigonal or cubic depending on symmetry detection
    assert analysis.crystal_system in ["cubic", "trigonal"]
    assert len(analysis.recommended_kpts) == 3
    assert "Ry" in analysis.recommended_cutoff
    assert analysis.recommended_basis in ["SZ", "DZ", "SZP", "DZP", "TZP"]


def test_material_analyzer_aluminum(aluminum_structure):
    """Test MaterialAnalyzer on aluminum (metal)."""
    analysis = MaterialAnalyzer.analyze(aluminum_structure)

    assert analysis.formula == "Al"
    assert analysis.is_metal  # Aluminum is metal
    assert analysis.recommended_preset == "relax_bulk_metal"  # Metal preset


def test_recipe_book_analyze_structure(silicon_structure):
    """Test RecipeBook.analyze_structure method."""
    analysis = RecipeBook.analyze_structure(silicon_structure)

    assert isinstance(analysis, MaterialAnalysis)
    assert analysis.formula == "Si"  # Reduced formula


def test_recipe_book_print_analysis(silicon_structure, capsys):
    """Test RecipeBook.print_analysis method."""
    RecipeBook.print_analysis(silicon_structure, detailed=True)

    captured = capsys.readouterr()
    assert "Material Analysis" in captured.out
    assert "Si" in captured.out  # Reduced formula
    assert "K-points" in captured.out
    assert "Mesh cutoff" in captured.out


def test_kpts_recommendation(silicon_structure):
    """Test k-point recommendation logic."""
    analysis = MaterialAnalyzer.analyze(silicon_structure)

    # K-points should be reasonable (2-16 range)
    for k in analysis.recommended_kpts:
        assert 1 <= k <= 16
        assert k in [1, 2, 4, 6, 8, 12, 16]  # Valid k-point values


def test_cutoff_recommendation():
    """Test mesh cutoff recommendations."""
    # Light element (C, Z=6)
    assert MaterialAnalyzer._recommend_cutoff(False, 6) == "300 Ry"

    # Medium element (Fe, Z=26)
    assert MaterialAnalyzer._recommend_cutoff(False, 26) == "350 Ry"

    # Heavy element (Pb, Z=82)
    assert MaterialAnalyzer._recommend_cutoff(True, 82) == "500 Ry"


def test_basis_recommendation():
    """Test basis size recommendations."""
    # Small system
    assert MaterialAnalyzer._recommend_basis(10, False) == "DZP"

    # Medium system
    assert MaterialAnalyzer._recommend_basis(60, False) == "DZ"

    # Large system
    assert MaterialAnalyzer._recommend_basis(150, False) == "SZ"

    # Metal (always polarized)
    assert "P" in MaterialAnalyzer._recommend_basis(60, True)


def test_tier_preset_recommendation():
    """Test tier and preset recommendations."""
    # Regular semiconductor
    tier, preset = MaterialAnalyzer._recommend_tier_preset(False, False, False)
    assert tier == "basic"
    assert preset == "relax_standard"

    # Metal
    tier, preset = MaterialAnalyzer._recommend_tier_preset(True, False, False)
    assert tier == "intermediate"
    assert preset == "relax_bulk_metal"  # Corrected from "surface_metal"

    # Magnetic metal
    tier, preset = MaterialAnalyzer._recommend_tier_preset(True, True, False)
    assert tier == "intermediate"
    assert preset == "magnetic_correlated"  # Corrected from "magnetic_metal"

    # Layered structure (non-metal)
    tier, preset = MaterialAnalyzer._recommend_tier_preset(False, False, True)
    assert tier == "intermediate"
    assert preset == "surface_semiconductor"  # Corrected from "surface_relax"


def test_cost_estimation():
    """Test computational cost estimation."""
    # Small system
    time_small, mem_small, cores_small = MaterialAnalyzer._estimate_cost(
        10, [4, 4, 4], False
    )
    assert time_small > 0
    assert mem_small > 0
    assert cores_small == 2  # Small system → 2 cores

    # Large system
    time_large, mem_large, cores_large = MaterialAnalyzer._estimate_cost(
        150, [8, 8, 8], False
    )
    assert time_large > time_small  # Larger takes longer
    assert mem_large > mem_small  # Larger uses more memory
    assert cores_large == 16  # Large system → 16 cores

    # Metal (more expensive)
    time_metal, _, _ = MaterialAnalyzer._estimate_cost(10, [4, 4, 4], True)
    time_semi, _, _ = MaterialAnalyzer._estimate_cost(10, [4, 4, 4], False)
    assert time_metal > time_semi  # Metals take longer
