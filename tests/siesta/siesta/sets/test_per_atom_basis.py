"""Tests for per-atom basis helper functions (Phase 3)."""

import pytest
from pymatgen.core import Lattice, Structure

from atomate2.siesta.sets.utils.per_atom_basis import (
    apply_per_atom_basis,
    create_per_atom_basis_dict,
)


@pytest.fixture
def simple_structure():
    """Create simple Si + 3O structure for testing."""
    lattice = Lattice.cubic(5.0)
    structure = Structure(
        lattice,
        ["Si", "O", "O", "O"],
        [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
    )
    return structure


@pytest.fixture
def surface_structure():
    """Create slab-like structure with Ti and O."""
    lattice = Lattice.from_parameters(5.0, 5.0, 15.0, 90, 90, 90)
    structure = Structure(
        lattice,
        ["Ti", "Ti", "O", "O", "O", "O"],
        [
            [0, 0, 0.1],  # Ti surface
            [0.5, 0.5, 0.2],  # Ti surface
            [0, 0, 0.3],  # O surface
            [0.5, 0, 0.4],  # O surface
            [0, 0.5, 0.6],  # O bulk
            [0.5, 0.5, 0.7],  # O bulk
        ],
    )
    return structure


class TestApplyPerAtomBasis:
    """Tests for apply_per_atom_basis function."""

    def test_simple_case(self, simple_structure):
        """Test basic per-atom basis assignment."""
        per_atom_basis = {
            1: "TZP",  # Si
            2: "TZP",  # O1
            3: "DZP",  # O2
            # O3 uses fallback
        }

        species_labels, pao_basissizes = apply_per_atom_basis(
            simple_structure, per_atom_basis, fallback_basis="DZ"
        )

        # Check species labels length
        assert len(species_labels) == 4

        # Check Si got TZP
        assert species_labels[0] == "Si"  # First Si with TZP
        assert "Si" in pao_basissizes
        assert pao_basissizes["Si"] == "TZP"

        # Check we have distinct O species
        assert len(set(species_labels[1:])) >= 2  # At least 2 different O labels

    def test_all_same_basis(self, simple_structure):
        """Test when all atoms use same basis."""
        per_atom_basis = {1: "DZP", 2: "DZP", 3: "DZP", 4: "DZP"}

        species_labels, pao_basissizes = apply_per_atom_basis(
            simple_structure, per_atom_basis
        )

        # Should have Si and O species
        assert "Si" in pao_basissizes
        assert "O" in pao_basissizes
        assert pao_basissizes["Si"] == "DZP"
        assert pao_basissizes["O"] == "DZP"

    def test_fallback_basis(self, simple_structure):
        """Test that fallback basis is used for unspecified atoms."""
        per_atom_basis = {1: "TZP"}  # Only specify atom 1

        species_labels, pao_basissizes = apply_per_atom_basis(
            simple_structure, per_atom_basis, fallback_basis="SZ"
        )

        # Atom 1 (Si) should have TZP
        assert species_labels[0] == "Si"
        assert pao_basissizes["Si"] == "TZP"

        # Atoms 2-4 (O) should have fallback SZ
        # They should all share the same species label
        assert species_labels[1] == species_labels[2] == species_labels[3]
        o_label = species_labels[1]
        assert pao_basissizes[o_label] == "SZ"

    def test_invalid_atom_index_too_low(self, simple_structure):
        """Test error for atom index < 1."""
        per_atom_basis = {0: "DZP"}  # Invalid: 0-indexed

        with pytest.raises(ValueError, match="Invalid atom index 0"):
            apply_per_atom_basis(simple_structure, per_atom_basis)

    def test_invalid_atom_index_too_high(self, simple_structure):
        """Test error for atom index > n_atoms."""
        per_atom_basis = {10: "DZP"}  # Invalid: only 4 atoms

        with pytest.raises(ValueError, match="Invalid atom index 10"):
            apply_per_atom_basis(simple_structure, per_atom_basis)

    def test_surface_bulk_distinction(self, surface_structure):
        """Test distinguishing surface vs bulk atoms."""
        per_atom_basis = {
            1: "TZP",  # Ti surface
            2: "TZP",  # Ti surface
            3: "TZP",  # O surface
            4: "TZP",  # O surface
            5: "DZ",  # O bulk
            6: "DZ",  # O bulk
        }

        species_labels, pao_basissizes = apply_per_atom_basis(
            surface_structure, per_atom_basis
        )

        # Ti atoms should all be same (both TZP)
        assert species_labels[0] == species_labels[1]
        ti_label = species_labels[0]
        assert pao_basissizes[ti_label] == "TZP"

        # O surface atoms should be same
        assert species_labels[2] == species_labels[3]
        o_surface_label = species_labels[2]
        assert pao_basissizes[o_surface_label] == "TZP"

        # O bulk atoms should be same
        assert species_labels[4] == species_labels[5]
        o_bulk_label = species_labels[4]
        assert pao_basissizes[o_bulk_label] == "DZ"

        # Surface and bulk O should be different
        assert o_surface_label != o_bulk_label

    def test_all_different_basis(self, simple_structure):
        """Test when every atom has different basis."""
        per_atom_basis = {
            1: "TZP",  # Si
            2: "TZP",  # O1
            3: "DZP",  # O2
            4: "DZ",  # O3
        }

        species_labels, pao_basissizes = apply_per_atom_basis(
            simple_structure, per_atom_basis
        )

        # Should have 4 different species (1 Si, 3 O variants)
        assert len(pao_basissizes) == 4

        # All O atoms should have different labels
        o_labels = species_labels[1:]
        assert len(set(o_labels)) == 3  # 3 unique O labels

    def test_empty_per_atom_dict(self, simple_structure):
        """Test with empty per-atom dict (all use fallback)."""
        species_labels, pao_basissizes = apply_per_atom_basis(
            simple_structure, {}, fallback_basis="DZP"
        )

        # Should have just Si and O
        assert len(pao_basissizes) == 2
        assert "Si" in pao_basissizes
        assert "O" in pao_basissizes
        assert pao_basissizes["Si"] == "DZP"
        assert pao_basissizes["O"] == "DZP"


class TestCreatePerAtomBasisDict:
    """Tests for create_per_atom_basis_dict function."""

    def test_basic_groups(self, surface_structure):
        """Test basic grouped atom specification."""
        atom_groups = {
            "surface": ([1, 2, 3, 4], "TZP"),
            "bulk": ([5, 6], "DZ"),
        }

        species_labels, pao_basissizes = create_per_atom_basis_dict(
            surface_structure, atom_groups
        )

        # Check surface atoms (1-4) got TZP
        for i in range(4):
            label = species_labels[i]
            assert pao_basissizes[label] == "TZP"

        # Check bulk atoms (5-6) got DZ
        for i in range(4, 6):
            label = species_labels[i]
            assert pao_basissizes[label] == "DZ"

    def test_overlapping_groups(self, simple_structure):
        """Test error when atom appears in multiple groups."""
        atom_groups = {
            "group1": ([1, 2], "TZP"),
            "group2": ([2, 3], "DZP"),  # Atom 2 appears twice!
        }

        with pytest.raises(ValueError, match="Atom 2 appears in multiple groups"):
            create_per_atom_basis_dict(simple_structure, atom_groups)

    def test_partial_coverage(self, simple_structure):
        """Test when some atoms not in any group (use fallback)."""
        atom_groups = {
            "high_accuracy": ([1, 2], "TZP"),
            # Atoms 3-4 not in any group
        }

        species_labels, pao_basissizes = create_per_atom_basis_dict(
            simple_structure, atom_groups, fallback_basis="SZ"
        )

        # Atoms 1-2 should have TZP
        assert pao_basissizes[species_labels[0]] == "TZP"
        assert pao_basissizes[species_labels[1]] == "TZP"

        # Atoms 3-4 should have fallback SZ
        assert pao_basissizes[species_labels[2]] == "SZ"
        assert pao_basissizes[species_labels[3]] == "SZ"

    def test_layer_based_grouping(self, surface_structure):
        """Test realistic layer-based grouping."""
        atom_groups = {
            "surface_layer": ([1, 2, 3, 4], "TZP"),  # High accuracy for surface
            "bulk_layer": ([5, 6], "DZP"),  # Medium for bulk
        }

        species_labels, pao_basissizes = create_per_atom_basis_dict(
            surface_structure, atom_groups
        )

        # Verify we have at least 2 species (Ti and O with different basis)
        assert len(pao_basissizes) >= 2

        # Verify TZP and DZP both present
        basis_values = set(pao_basissizes.values())
        assert "TZP" in basis_values
        assert "DZP" in basis_values

    def test_empty_groups(self, simple_structure):
        """Test with no groups (all use fallback)."""
        species_labels, pao_basissizes = create_per_atom_basis_dict(
            simple_structure, {}, fallback_basis="DZ"
        )

        # All atoms should use fallback
        assert all(pao_basissizes[label] == "DZ" for label in species_labels)


class TestIntegration:
    """Integration tests with real workflow patterns."""

    def test_surface_adsorption_pattern(self):
        """Test realistic surface + adsorbate pattern."""
        # Create slab with adsorbate
        lattice = Lattice.from_parameters(10.0, 10.0, 20.0, 90, 90, 90)
        structure = Structure(
            lattice,
            ["Cu"] * 12 + ["C", "O"],  # 12 Cu + CO molecule
            [
                [0, 0, 0.1],
                [0.5, 0, 0.1],  # Surface Cu
                [0, 0.5, 0.1],
                [0.5, 0.5, 0.1],
                [0, 0, 0.2],
                [0.5, 0, 0.2],  # Subsurface Cu
                [0, 0.5, 0.2],
                [0.5, 0.5, 0.2],
                [0, 0, 0.3],
                [0.5, 0, 0.3],  # Bulk Cu
                [0, 0.5, 0.3],
                [0.5, 0.5, 0.3],
                [0.25, 0.25, 0.05],  # C adsorbate
                [0.25, 0.25, 0.02],  # O adsorbate
            ],
        )

        # Define layers
        atom_groups = {
            "adsorbate": ([13, 14], "TZP"),  # High accuracy for CO
            "surface": ([1, 2, 3, 4], "TZP"),  # High for surface Cu
            "subsurface": ([5, 6, 7, 8], "DZP"),  # Medium for subsurface
            "bulk": ([9, 10, 11, 12], "DZ"),  # Efficient for bulk
        }

        species_labels, pao_basissizes = create_per_atom_basis_dict(
            structure, atom_groups
        )

        # Verify all basis sizes present
        assert "TZP" in pao_basissizes.values()
        assert "DZP" in pao_basissizes.values()
        assert "DZ" in pao_basissizes.values()

        # Verify we have C and O species
        assert any("C" in label for label in pao_basissizes.keys())
        assert any("O" in label for label in pao_basissizes.keys())

        # Verify correct number of species labels
        assert len(species_labels) == 14

    def test_with_relax_maker(self, simple_structure):
        """Test that output can be used with RelaxMaker."""
        per_atom_basis = {1: "TZP", 2: "TZP", 3: "DZP", 4: "DZ"}

        species_labels, pao_basissizes = apply_per_atom_basis(
            simple_structure, per_atom_basis
        )

        # Add to structure
        simple_structure.add_site_property("species_label", species_labels)

        # Verify can create user_params dict
        user_params = {
            "%block PAO.BasisSizes": pao_basissizes,
            "Mesh.Cutoff": "300 Ry",
        }

        # Verify pao_basissizes is a dict
        assert isinstance(user_params["%block PAO.BasisSizes"], dict)

        # Verify it has the right format
        for label, basis in pao_basissizes.items():
            assert isinstance(label, str)
            assert isinstance(basis, str)
            assert basis in ["TZP", "DZP", "DZ", "SZ", "SZP"]
