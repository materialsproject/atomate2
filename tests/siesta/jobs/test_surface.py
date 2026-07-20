"""
Tests for surface energy calculation jobs.

These tests validate:
- calculate_surface_energies function (energy calculations, termination analysis)
- plot_surface_energies function (plotting and file generation)
- write_surface_energy_summary function (text report generation)
- Edge cases and error handling
"""

from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from atomate2.siesta.jobs.surface import (
    calculate_surface_energies,
    plot_surface_energies,
    write_surface_energy_summary,
)


@pytest.fixture
def si_bulk_structure():
    """Create Si bulk structure."""
    lattice = Lattice.cubic(5.43)
    structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    return structure


@pytest.fixture
def slab_structure_symmetric():
    """Create symmetric Si slab structure."""
    lattice = Lattice.from_parameters(5.43, 5.43, 20.0, 90, 90, 90)
    # 8 Si atoms in slab
    structure = Structure(
        lattice,
        ["Si"] * 8,
        [
            [0.0, 0.0, 0.25],
            [0.25, 0.25, 0.28],
            [0.0, 0.0, 0.35],
            [0.25, 0.25, 0.38],
            [0.0, 0.0, 0.62],
            [0.25, 0.25, 0.65],
            [0.0, 0.0, 0.72],
            [0.25, 0.25, 0.75],
        ],
    )
    return structure


@pytest.fixture
def slab_structure_asymmetric():
    """Create asymmetric Si slab structure."""
    lattice = Lattice.from_parameters(5.43, 5.43, 20.0, 90, 90, 90)
    # 6 Si atoms in slab
    structure = Structure(
        lattice,
        ["Si"] * 6,
        [
            [0.0, 0.0, 0.25],
            [0.25, 0.25, 0.28],
            [0.0, 0.0, 0.35],
            [0.25, 0.25, 0.38],
            [0.0, 0.0, 0.45],
            [0.25, 0.25, 0.48],
        ],
    )
    return structure


@pytest.fixture
def basic_slab_data(slab_structure_symmetric, slab_structure_asymmetric):
    """Create basic slab data for testing."""
    return [
        {
            "termination": "Si_term1",
            "slab_energy": -45.0,
            "slab_structure": slab_structure_symmetric,
            "metadata": {
                "bottom_composition": {"Si": 4},
                "top_composition": {"Si": 4},
                "is_symmetric": True,
                "z_position": 0.0,
            },
        },
        {
            "termination": "Si_term2",
            "slab_energy": -42.5,
            "slab_structure": slab_structure_asymmetric,
            "metadata": {
                "bottom_composition": {"Si": 3},
                "top_composition": {"Si": 3},
                "is_symmetric": False,
                "z_position": 0.5,
            },
        },
    ]


class TestCalculateSurfaceEnergies:
    """Tests for calculate_surface_energies function."""

    def test_calculate_surface_energies_basic(self, si_bulk_structure, basic_slab_data):
        """Test basic surface energy calculation using .original pattern."""
        bulk_energy = -10.8  # eV for 2 Si atoms
        miller_indices = (1, 1, 1)

        # Use .original to bypass @job decorator
        result = calculate_surface_energies.original(
            bulk_energy=bulk_energy,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=miller_indices,
            formula_units_per_cell=None,  # Auto-detect
        )

        # Check output structure
        assert isinstance(result, dict)
        assert "bulk_energy" in result
        assert "bulk_energy_per_atom" in result
        assert "miller_indices" in result
        assert "terminations" in result
        assert "lowest_termination" in result
        assert "n_terminations" in result
        assert "energy_spread" in result

        # Check values
        assert result["bulk_energy"] == bulk_energy
        assert result["bulk_energy_per_atom"] == bulk_energy / 2
        assert result["miller_indices"] == miller_indices
        assert result["n_terminations"] == 2
        # Si bulk has 2 atoms, reduced composition is Si1, so 2 formula units
        assert result["formula_units_per_cell"] == 2  # Auto-detected

    def test_calculate_surface_energies_termination_data(
        self, si_bulk_structure, basic_slab_data
    ):
        """Test termination-specific data."""
        result = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=(1, 1, 1),
        )

        # Check terminations list
        assert len(result["terminations"]) == 2

        for term in result["terminations"]:
            # Required fields
            assert "termination" in term
            assert "surface_energy" in term
            assert "surface_energy_Jm2" in term
            assert "slab_energy" in term
            assert "n_formula_units" in term
            assert "surface_area" in term
            assert "n_atoms" in term
            assert "thickness" in term
            assert "composition" in term
            assert "relative_energy" in term
            assert "is_lowest" in term

            # Optional metadata fields
            assert "bottom_composition" in term
            assert "top_composition" in term
            assert "is_symmetric" in term
            assert "z_position" in term

    def test_calculate_surface_energies_unit_conversion(
        self, si_bulk_structure, basic_slab_data
    ):
        """Test unit conversion eV/Ų to J/m²."""
        result = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=(1, 1, 1),
        )

        for term in result["terminations"]:
            # Check conversion factor (1 eV/Ų = 16.0218 J/m²)
            expected_Jm2 = term["surface_energy"] * 16.0218
            assert abs(term["surface_energy_Jm2"] - expected_Jm2) < 0.01

    def test_calculate_surface_energies_lowest_termination(
        self, si_bulk_structure, basic_slab_data
    ):
        """Test identification of lowest energy termination."""
        result = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=(1, 1, 1),
        )

        # Check lowest termination logic
        energies = [t["surface_energy"] for t in result["terminations"]]
        min_energy = min(energies)

        # Find termination marked as lowest
        lowest_terms = [t for t in result["terminations"] if t["is_lowest"]]
        assert len(lowest_terms) == 1
        assert lowest_terms[0]["surface_energy"] == min_energy

        # Check relative energies
        for term in result["terminations"]:
            expected_rel = term["surface_energy"] - min_energy
            assert abs(term["relative_energy"] - expected_rel) < 1e-6

    def test_calculate_surface_energies_energy_spread(
        self, si_bulk_structure, basic_slab_data
    ):
        """Test energy spread calculation."""
        result = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=(1, 1, 1),
        )

        energies = [t["surface_energy"] for t in result["terminations"]]
        expected_spread = max(energies) - min(energies)

        assert abs(result["energy_spread"] - expected_spread) < 1e-6

    def test_calculate_surface_energies_explicit_formula_units(
        self, si_bulk_structure, basic_slab_data
    ):
        """Test with explicit formula_units_per_cell."""
        result = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=(1, 1, 1),
            formula_units_per_cell=2,  # Explicit value
        )

        assert result["formula_units_per_cell"] == 2

    def test_calculate_surface_energies_surface_area_calculation(
        self, si_bulk_structure, basic_slab_data
    ):
        """Test surface area calculation (cross product of a and b vectors)."""
        result = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=(1, 1, 1),
        )

        for term in result["terminations"]:
            # Surface area should be positive
            assert term["surface_area"] > 0

            # For cubic lattice ~5.43 Å, area should be ~29.5 Ų
            assert 25 < term["surface_area"] < 35

    def test_calculate_surface_energies_thickness_calculation(
        self, si_bulk_structure, basic_slab_data
    ):
        """Test slab thickness calculation (max_z - min_z)."""
        result = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=(1, 1, 1),
        )

        for term in result["terminations"]:
            # Thickness should be positive and less than cell height
            assert term["thickness"] > 0
            slab_struct = next(
                s["slab_structure"]
                for s in basic_slab_data
                if s["termination"] == term["termination"]
            )
            cell_height = slab_struct.lattice.c
            assert term["thickness"] < cell_height

    def test_calculate_surface_energies_composition_tracking(
        self, si_bulk_structure, basic_slab_data
    ):
        """Test composition information is preserved."""
        result = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=(1, 1, 1),
        )

        for term in result["terminations"]:
            # Composition should be a dict
            assert isinstance(term["composition"], dict)

            # Number of atoms should match composition sum
            comp_sum = sum(term["composition"].values())
            assert term["n_atoms"] == comp_sum


class TestCalculateSurfaceEnergiesEdgeCases:
    """Test edge cases for calculate_surface_energies."""

    def test_single_termination(self, si_bulk_structure, slab_structure_symmetric):
        """Test with single termination."""
        slab_data = [
            {
                "termination": "only_term",
                "slab_energy": -45.0,
                "slab_structure": slab_structure_symmetric,
                "metadata": {},
            }
        ]

        result = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=slab_data,
            miller_indices=(1, 0, 0),
        )

        assert result["n_terminations"] == 1
        assert result["energy_spread"] == 0.0
        # numpy boolean converts to Python bool, so check boolean value
        assert result["terminations"][0]["is_lowest"] == True
        assert result["terminations"][0]["relative_energy"] == 0.0

    def test_empty_slab_data_raises_error(self, si_bulk_structure):
        """Test that empty slab data raises ValueError."""
        with pytest.raises(ValueError, match="No termination data to process"):
            calculate_surface_energies.original(
                bulk_energy=-10.8,
                bulk_structure=si_bulk_structure,
                slab_data=[],
                miller_indices=(1, 0, 0),
            )

    def test_multiple_terminations_same_energy(
        self, si_bulk_structure, slab_structure_symmetric
    ):
        """Test with multiple terminations having identical energies."""
        slab_data = [
            {
                "termination": "term1",
                "slab_energy": -45.0,
                "slab_structure": slab_structure_symmetric,
                "metadata": {},
            },
            {
                "termination": "term2",
                "slab_energy": -45.0,
                "slab_structure": slab_structure_symmetric,
                "metadata": {},
            },
        ]

        result = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=slab_data,
            miller_indices=(1, 1, 0),
        )

        # Both should be marked as lowest (within tolerance)
        lowest_count = sum(t["is_lowest"] for t in result["terminations"])
        assert lowest_count >= 1  # At least one marked as lowest
        assert result["energy_spread"] < 1e-5  # Essentially zero

    def test_missing_metadata_fields(self, si_bulk_structure, slab_structure_symmetric):
        """Test handling of missing metadata fields."""
        slab_data = [
            {
                "termination": "minimal_term",
                "slab_energy": -45.0,
                "slab_structure": slab_structure_symmetric,
                # No metadata field
            }
        ]

        result = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=slab_data,
            miller_indices=(1, 1, 1),
        )

        term = result["terminations"][0]
        # Should have empty dict defaults
        assert term["bottom_composition"] == {}
        assert term["top_composition"] == {}
        assert term["is_symmetric"] is False
        assert term["z_position"] == 0.0

    def test_different_miller_indices(self, si_bulk_structure, basic_slab_data):
        """Test different Miller indices."""
        miller_indices_list = [
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1),
            (2, 1, 0),
        ]

        for miller in miller_indices_list:
            result = calculate_surface_energies.original(
                bulk_energy=-10.8,
                bulk_structure=si_bulk_structure,
                slab_data=basic_slab_data,
                miller_indices=miller,
            )

            assert result["miller_indices"] == miller


class TestPlotSurfaceEnergies:
    """Tests for plot_surface_energies function."""

    def test_plot_surface_energies_creates_file(
        self, si_bulk_structure, basic_slab_data, tmp_path
    ):
        """Test that plotting creates output file."""
        # First calculate surface energies
        surface_doc = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=(1, 1, 1),
        )

        # Create plot
        result = plot_surface_energies.original(
            surface_doc=surface_doc,
            output_dir=str(tmp_path),
            filename="test_surface_plot.png",
        )

        # Check return value
        assert "plot_file" in result
        plot_path = Path(result["plot_file"])
        assert plot_path.exists()
        assert plot_path.suffix == ".png"
        assert plot_path.name == "test_surface_plot.png"

    def test_plot_surface_energies_custom_figsize(
        self, si_bulk_structure, basic_slab_data, tmp_path
    ):
        """Test plotting with custom figure size."""
        surface_doc = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=(1, 1, 1),
        )

        result = plot_surface_energies.original(
            surface_doc=surface_doc,
            output_dir=str(tmp_path),
            figsize=(20, 10),
            dpi=150,
        )

        assert Path(result["plot_file"]).exists()

    def test_plot_surface_energies_creates_directory(
        self, si_bulk_structure, basic_slab_data, tmp_path
    ):
        """Test that plotting creates output directory if it doesn't exist."""
        output_dir = tmp_path / "nonexistent" / "nested"

        surface_doc = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=(1, 1, 1),
        )

        result = plot_surface_energies.original(
            surface_doc=surface_doc,
            output_dir=str(output_dir),
        )

        assert Path(result["plot_file"]).exists()
        assert Path(result["plot_file"]).parent == output_dir

    def test_plot_surface_energies_single_termination(
        self, si_bulk_structure, slab_structure_symmetric, tmp_path
    ):
        """Test plotting with single termination."""
        slab_data = [
            {
                "termination": "only_term",
                "slab_energy": -45.0,
                "slab_structure": slab_structure_symmetric,
                "metadata": {},
            }
        ]

        surface_doc = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=slab_data,
            miller_indices=(1, 0, 0),
        )

        result = plot_surface_energies.original(
            surface_doc=surface_doc,
            output_dir=str(tmp_path),
        )

        assert Path(result["plot_file"]).exists()

    def test_plot_surface_energies_many_terminations(
        self, si_bulk_structure, slab_structure_symmetric, tmp_path
    ):
        """Test plotting with many terminations."""
        # Create 10 terminations with different energies
        slab_data = [
            {
                "termination": f"term_{i}",
                "slab_energy": -45.0 + i * 0.5,
                "slab_structure": slab_structure_symmetric,
                "metadata": {},
            }
            for i in range(10)
        ]

        surface_doc = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=slab_data,
            miller_indices=(1, 1, 0),
        )

        result = plot_surface_energies.original(
            surface_doc=surface_doc,
            output_dir=str(tmp_path),
        )

        assert Path(result["plot_file"]).exists()


class TestWriteSurfaceEnergySummary:
    """Tests for write_surface_energy_summary function."""

    def test_write_summary_creates_file(
        self, si_bulk_structure, basic_slab_data, tmp_path
    ):
        """Test that summary writing creates output file."""
        surface_doc = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=(1, 1, 1),
        )

        result = write_surface_energy_summary.original(
            surface_doc=surface_doc,
            output_dir=str(tmp_path),
            filename="test_summary.txt",
        )

        # Check return value
        assert "summary_file" in result
        summary_path = Path(result["summary_file"])
        assert summary_path.exists()
        assert summary_path.suffix == ".txt"
        assert summary_path.name == "test_summary.txt"

    def test_write_summary_content_structure(
        self, si_bulk_structure, basic_slab_data, tmp_path
    ):
        """Test summary file contains expected sections."""
        surface_doc = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=(1, 1, 1),
        )

        result = write_surface_energy_summary.original(
            surface_doc=surface_doc,
            output_dir=str(tmp_path),
        )

        content = Path(result["summary_file"]).read_text()

        # Check for expected sections
        assert "SURFACE ENERGY CALCULATION SUMMARY" in content
        assert "BULK PROPERTIES" in content
        assert "SURFACE INFORMATION" in content
        assert "TERMINATION ENERGIES" in content
        assert "DETAILED BREAKDOWN" in content
        assert "CONVERGENCE NOTES" in content
        assert "REFERENCES" in content

    def test_write_summary_includes_all_terminations(
        self, si_bulk_structure, basic_slab_data, tmp_path
    ):
        """Test summary includes all terminations."""
        surface_doc = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=(1, 1, 1),
        )

        result = write_surface_energy_summary.original(
            surface_doc=surface_doc,
            output_dir=str(tmp_path),
        )

        content = Path(result["summary_file"]).read_text()

        # Check that all terminations appear in summary
        for term in surface_doc["terminations"]:
            assert term["termination"] in content

    def test_write_summary_marks_lowest_termination(
        self, si_bulk_structure, basic_slab_data, tmp_path
    ):
        """Test summary marks lowest energy termination."""
        surface_doc = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=(1, 1, 1),
        )

        result = write_surface_energy_summary.original(
            surface_doc=surface_doc,
            output_dir=str(tmp_path),
        )

        content = Path(result["summary_file"]).read_text()

        # Check for lowest marker
        assert "LOWEST ✓" in content
        assert surface_doc["lowest_termination"] in content

    def test_write_summary_includes_miller_indices(
        self, si_bulk_structure, basic_slab_data, tmp_path
    ):
        """Test summary includes Miller indices."""
        miller_indices = (2, 1, 0)
        surface_doc = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=miller_indices,
        )

        result = write_surface_energy_summary.original(
            surface_doc=surface_doc,
            output_dir=str(tmp_path),
        )

        content = Path(result["summary_file"]).read_text()

        # Miller indices should appear in summary
        assert "(2 1 0)" in content

    def test_write_summary_creates_directory(
        self, si_bulk_structure, basic_slab_data, tmp_path
    ):
        """Test that summary writing creates output directory."""
        output_dir = tmp_path / "nested" / "directory"

        surface_doc = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=(1, 1, 1),
        )

        result = write_surface_energy_summary.original(
            surface_doc=surface_doc,
            output_dir=str(output_dir),
        )

        assert Path(result["summary_file"]).exists()
        assert Path(result["summary_file"]).parent == output_dir

    def test_write_summary_with_metadata(
        self, si_bulk_structure, basic_slab_data, tmp_path
    ):
        """Test summary includes metadata when present."""
        surface_doc = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=(1, 1, 1),
        )

        result = write_surface_energy_summary.original(
            surface_doc=surface_doc,
            output_dir=str(tmp_path),
        )

        content = Path(result["summary_file"]).read_text()

        # Check for composition data (should appear for at least one termination)
        assert "Bottom layer:" in content or "composition" in content.lower()

    def test_write_summary_single_termination(
        self, si_bulk_structure, slab_structure_symmetric, tmp_path
    ):
        """Test summary with single termination."""
        slab_data = [
            {
                "termination": "only_term",
                "slab_energy": -45.0,
                "slab_structure": slab_structure_symmetric,
                "metadata": {},
            }
        ]

        surface_doc = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=slab_data,
            miller_indices=(1, 0, 0),
        )

        result = write_surface_energy_summary.original(
            surface_doc=surface_doc,
            output_dir=str(tmp_path),
        )

        content = Path(result["summary_file"]).read_text()
        assert "Number of terminations:   1" in content


class TestSurfaceEnergyIntegration:
    """Integration tests combining multiple surface energy functions."""

    def test_complete_workflow(self, si_bulk_structure, basic_slab_data, tmp_path):
        """Test complete surface energy workflow."""
        # Step 1: Calculate energies
        surface_doc = calculate_surface_energies.original(
            bulk_energy=-10.8,
            bulk_structure=si_bulk_structure,
            slab_data=basic_slab_data,
            miller_indices=(1, 1, 1),
        )

        assert surface_doc is not None
        assert len(surface_doc["terminations"]) == 2

        # Step 2: Create plot
        plot_result = plot_surface_energies.original(
            surface_doc=surface_doc,
            output_dir=str(tmp_path),
            filename="workflow_plot.png",
        )

        assert Path(plot_result["plot_file"]).exists()

        # Step 3: Write summary
        summary_result = write_surface_energy_summary.original(
            surface_doc=surface_doc,
            output_dir=str(tmp_path),
            filename="workflow_summary.txt",
        )

        assert Path(summary_result["summary_file"]).exists()

        # Verify all outputs in same directory
        plot_path = Path(plot_result["plot_file"])
        summary_path = Path(summary_result["summary_file"])
        assert plot_path.parent == summary_path.parent == tmp_path

    def test_workflow_with_different_materials(
        self, slab_structure_symmetric, tmp_path
    ):
        """Test workflow with different material compositions."""
        # Create Al bulk structure
        al_lattice = Lattice.cubic(4.05)
        al_bulk = Structure(al_lattice, ["Al"], [[0, 0, 0]])

        # Create Al slab data
        al_slab_data = [
            {
                "termination": "Al_100",
                "slab_energy": -30.0,
                "slab_structure": slab_structure_symmetric,
                "metadata": {},
            }
        ]

        surface_doc = calculate_surface_energies.original(
            bulk_energy=-3.7,
            bulk_structure=al_bulk,
            slab_data=al_slab_data,
            miller_indices=(1, 0, 0),
        )

        # Should work with Al
        assert surface_doc["n_terminations"] == 1

        # Create outputs
        plot_result = plot_surface_energies.original(
            surface_doc=surface_doc, output_dir=str(tmp_path)
        )
        summary_result = write_surface_energy_summary.original(
            surface_doc=surface_doc, output_dir=str(tmp_path / "al")
        )

        assert Path(plot_result["plot_file"]).exists()
        assert Path(summary_result["summary_file"]).exists()
