"""
Tests for slab generation infrastructure (slab_generation.py).

These tests validate:
- Slab generation for single Miller index
- Slab generation for multiple Miller indices
- Termination detection and labeling
- Metadata collection
- SlabGenerator parameter passing
"""

import pytest
from pymatgen.core import Lattice, Structure

from atomate2.siesta.jobs.surface.slab_generation import (
    generate_slabs_for_all_miller_indices,
    generate_slabs_for_miller_index,
)


@pytest.fixture
def fcc_structure():
    """Create a simple FCC structure (like Al) for testing."""
    lattice = Lattice.cubic(4.05)
    return Structure(
        lattice,
        ["Al", "Al", "Al", "Al"],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
    )


@pytest.fixture
def cubic_structure():
    """Create a simple cubic structure for testing."""
    lattice = Lattice.cubic(5.0)
    return Structure(lattice, ["Si", "Si"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


class TestGenerateSlabsForMillerIndex:
    """Tests for generate_slabs_for_miller_index function."""

    def test_generate_100_slab(self, fcc_structure):
        """Test generating (100) surface slab."""
        result = generate_slabs_for_miller_index.original(
            structure=fcc_structure,
            miller_index=(1, 0, 0),
            min_slab_size=10.0,
            min_vacuum_size=15.0,
        )

        # Check basic return structure
        assert result["miller_index"] == (1, 0, 0)
        assert "slabs" in result
        assert "n_terminations" in result
        assert "termination_labels" in result
        assert "metadata" in result

        # Should have at least one slab
        assert result["n_terminations"] >= 1
        assert len(result["slabs"]) == result["n_terminations"]
        assert len(result["termination_labels"]) == result["n_terminations"]

    def test_generate_111_slab(self, fcc_structure):
        """Test generating (111) surface slab."""
        result = generate_slabs_for_miller_index.original(
            structure=fcc_structure,
            miller_index=(1, 1, 1),
            min_slab_size=10.0,
            min_vacuum_size=15.0,
        )

        assert result["miller_index"] == (1, 1, 1)
        assert result["n_terminations"] >= 1

    def test_generate_110_slab(self, fcc_structure):
        """Test generating (110) surface slab."""
        result = generate_slabs_for_miller_index.original(
            structure=fcc_structure,
            miller_index=(1, 1, 0),
            min_slab_size=10.0,
            min_vacuum_size=15.0,
        )

        assert result["miller_index"] == (1, 1, 0)
        assert result["n_terminations"] >= 1

    def test_slab_structure_is_valid(self, fcc_structure):
        """Test that generated slabs are valid Structure objects."""
        result = generate_slabs_for_miller_index.original(
            structure=fcc_structure,
            miller_index=(1, 0, 0),
            min_slab_size=10.0,
            min_vacuum_size=15.0,
        )

        for slab in result["slabs"]:
            assert isinstance(slab, Structure)
            assert len(slab) > 0  # Has atoms
            assert slab.lattice is not None

    def test_slab_has_vacuum(self, fcc_structure):
        """Test that generated slabs have vacuum layer."""
        min_vacuum = 15.0
        result = generate_slabs_for_miller_index.original(
            structure=fcc_structure,
            miller_index=(1, 0, 0),
            min_slab_size=10.0,
            min_vacuum_size=min_vacuum,
        )

        for slab in result["slabs"]:
            # Check c-axis is larger than just slab thickness
            # (should include vacuum)
            assert slab.lattice.c > 10.0  # min_slab_size

    def test_termination_labels_format(self, fcc_structure):
        """Test termination label format."""
        result = generate_slabs_for_miller_index.original(
            structure=fcc_structure,
            miller_index=(1, 0, 0),
            min_slab_size=10.0,
            min_vacuum_size=15.0,
        )

        for label in result["termination_labels"]:
            assert isinstance(label, str)
            assert "_term" in label  # Should contain _term
            assert "Al" in label  # Should contain element symbol

    def test_metadata_contents(self, fcc_structure):
        """Test metadata contains expected information."""
        result = generate_slabs_for_miller_index.original(
            structure=fcc_structure,
            miller_index=(1, 0, 0),
            min_slab_size=10.0,
            min_vacuum_size=15.0,
        )

        metadata = result["metadata"]
        assert "bulk_formula" in metadata
        assert "bulk_n_atoms" in metadata
        assert "min_slab_size" in metadata
        assert "min_vacuum_size" in metadata
        assert "symmetrize" in metadata
        assert "primitive" in metadata

        assert metadata["bulk_formula"] == "Al"
        assert metadata["bulk_n_atoms"] == 4
        assert metadata["min_slab_size"] == 10.0
        assert metadata["min_vacuum_size"] == 15.0

    def test_custom_slab_parameters(self, fcc_structure):
        """Test generating slabs with custom parameters."""
        result = generate_slabs_for_miller_index.original(
            structure=fcc_structure,
            miller_index=(1, 0, 0),
            min_slab_size=12.0,
            min_vacuum_size=20.0,
            symmetrize=True,
            lll_reduce=True,
            center_slab=False,
            primitive=False,
        )

        metadata = result["metadata"]
        assert metadata["min_slab_size"] == 12.0
        assert metadata["min_vacuum_size"] == 20.0
        assert metadata["symmetrize"] is True
        assert metadata["primitive"] is False

    def test_different_miller_indices(self, fcc_structure):
        """Test that different Miller indices produce different results."""
        result_100 = generate_slabs_for_miller_index.original(
            structure=fcc_structure,
            miller_index=(1, 0, 0),
        )

        result_111 = generate_slabs_for_miller_index.original(
            structure=fcc_structure,
            miller_index=(1, 1, 1),
        )

        # Miller indices should be different
        assert result_100["miller_index"] != result_111["miller_index"]

        # May have different number of terminations
        # (this is material-dependent)
        assert isinstance(result_100["n_terminations"], int)
        assert isinstance(result_111["n_terminations"], int)

    def test_small_slab_size(self, fcc_structure):
        """Test generating slabs with smaller slab size."""
        result = generate_slabs_for_miller_index.original(
            structure=fcc_structure,
            miller_index=(1, 0, 0),
            min_slab_size=7.0,  # Smaller slab
            min_vacuum_size=10.0,
        )

        assert result["n_terminations"] >= 1
        assert result["metadata"]["min_slab_size"] == 7.0


class TestGenerateSlabsForAllMillerIndices:
    """Tests for generate_slabs_for_all_miller_indices function."""

    def test_auto_generate_miller_indices(self, fcc_structure):
        """Test automatic Miller index generation."""
        result = generate_slabs_for_all_miller_indices.original(
            structure=fcc_structure,
            max_index=1,
            min_slab_size=10.0,
            min_vacuum_size=15.0,
        )

        # Check return structure
        assert "bulk_structure" in result
        assert "miller_indices" in result
        assert "slab_data" in result
        assert "n_miller_indices" in result

        # Should have generated some Miller indices
        assert result["n_miller_indices"] > 0
        assert len(result["miller_indices"]) == result["n_miller_indices"]

    def test_user_specified_miller_indices(self, fcc_structure):
        """Test with user-specified Miller indices."""
        miller_indices = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]

        result = generate_slabs_for_all_miller_indices.original(
            structure=fcc_structure,
            miller_indices=miller_indices,
            min_slab_size=10.0,
            min_vacuum_size=15.0,
        )

        assert result["n_miller_indices"] == 3
        assert result["miller_indices"] == miller_indices

        # Check slab_data has entries for all indices
        assert len(result["slab_data"]) == 3

    def test_slab_data_structure(self, fcc_structure):
        """Test structure of slab_data dictionary."""
        miller_indices = [(1, 0, 0), (1, 1, 1)]

        result = generate_slabs_for_all_miller_indices.original(
            structure=fcc_structure,
            miller_indices=miller_indices,
            min_slab_size=10.0,
            min_vacuum_size=15.0,
        )

        slab_data = result["slab_data"]

        # Keys should be string representations of Miller indices
        assert "(1, 0, 0)" in slab_data
        assert "(1, 1, 1)" in slab_data

        # slab_data contains Job objects (not executed results)
        # Just verify we have entries for all requested Miller indices
        assert len(slab_data) == 2

    def test_bulk_structure_stored(self, fcc_structure):
        """Test that bulk structure is stored in result."""
        result = generate_slabs_for_all_miller_indices.original(
            structure=fcc_structure,
            max_index=1,
        )

        stored_structure = result["bulk_structure"]
        assert isinstance(stored_structure, Structure)
        assert stored_structure.composition == fcc_structure.composition

    def test_single_miller_index_list(self, fcc_structure):
        """Test with single Miller index in list."""
        result = generate_slabs_for_all_miller_indices.original(
            structure=fcc_structure,
            miller_indices=[(1, 0, 0)],
        )

        assert result["n_miller_indices"] == 1
        assert len(result["slab_data"]) == 1

    def test_max_index_parameter(self, fcc_structure):
        """Test max_index parameter for auto-generation."""
        result_1 = generate_slabs_for_all_miller_indices.original(
            structure=fcc_structure,
            max_index=1,
        )

        result_2 = generate_slabs_for_all_miller_indices.original(
            structure=fcc_structure,
            max_index=2,
        )

        # Higher max_index should generate more or equal Miller indices
        assert result_2["n_miller_indices"] >= result_1["n_miller_indices"]

    def test_custom_slab_parameters_accepted(self, fcc_structure):
        """Test that custom slab parameters are accepted."""
        result = generate_slabs_for_all_miller_indices.original(
            structure=fcc_structure,
            miller_indices=[(1, 0, 0), (1, 1, 1)],
            min_slab_size=12.0,
            min_vacuum_size=18.0,
            symmetrize=True,
        )

        # Should complete without error
        assert result["n_miller_indices"] == 2
        assert len(result["slab_data"]) == 2

    def test_different_structures(self, fcc_structure, cubic_structure):
        """Test slab generation works with different structure types."""
        result_fcc = generate_slabs_for_all_miller_indices.original(
            structure=fcc_structure,
            miller_indices=[(1, 0, 0)],
        )

        result_cubic = generate_slabs_for_all_miller_indices.original(
            structure=cubic_structure,
            miller_indices=[(1, 0, 0)],
        )

        # Both should succeed and have slabs
        assert result_fcc["n_miller_indices"] == 1
        assert result_cubic["n_miller_indices"] == 1

        # Compositions should be different
        assert (
            result_fcc["bulk_structure"].composition
            != result_cubic["bulk_structure"].composition
        )
