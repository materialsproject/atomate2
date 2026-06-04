"""Tests for PAO.Basis helper functions."""

import pytest

from atomate2.siesta.sets.utils.basis_builder import (
    PAOBasisSpecies,
    PAOShell,
    create_pao_basis,
)


class TestPAOShell:
    """Tests for PAOShell dataclass."""

    def test_simple_shell(self):
        """Test creation of simple shell without optional parameters."""
        shell = PAOShell(n=2, l=1, nzeta=2, rc=[5.0, 3.5])

        assert shell.n == 2
        assert shell.l == 1
        assert shell.nzeta == 2
        assert shell.rc == [5.0, 3.5]
        assert shell.l_symbol == "p"
        assert not shell.polarization

    def test_polarization_shell(self):
        """Test polarization orbital creation."""
        shell = PAOShell(l=2, nzeta=1, rc=[4.0], polarization=True)

        assert shell.l == 2
        assert shell.l_symbol == "d"
        assert shell.polarization
        assert shell.nzeta_pol == 1  # Auto-set

    def test_validation_l_range(self):
        """Test that l must be 0-3."""
        with pytest.raises(ValueError, match="l=4 must be 0-3"):
            PAOShell(l=4, nzeta=1, rc=[5.0])

        with pytest.raises(ValueError, match="l=-1 must be 0-3"):
            PAOShell(l=-1, nzeta=1, rc=[5.0])

    def test_validation_nzeta(self):
        """Test that nzeta must be >= 1."""
        with pytest.raises(ValueError, match="nzeta=0 must be >= 1"):
            PAOShell(l=0, nzeta=0, rc=[])

    def test_validation_rc_length(self):
        """Test that rc length must match nzeta."""
        with pytest.raises(ValueError, match="rc length.*must match nzeta"):
            PAOShell(l=0, nzeta=2, rc=[5.0])  # Only 1 rc for nzeta=2

        with pytest.raises(ValueError, match="rc length.*must match nzeta"):
            PAOShell(l=0, nzeta=1, rc=[5.0, 3.0])  # 2 rc for nzeta=1

    def test_validation_split_norm(self):
        """Test split_norm must be 0.0-1.0."""
        with pytest.raises(ValueError, match="split_norm.*must be between 0.0 and 1.0"):
            PAOShell(l=0, nzeta=1, rc=[5.0], split_norm_flag=True, split_norm=1.5)

        with pytest.raises(ValueError, match="split_norm.*must be between 0.0 and 1.0"):
            PAOShell(l=0, nzeta=1, rc=[5.0], split_norm_flag=True, split_norm=-0.1)

    def test_to_fdf_lines_simple(self):
        """Test FDF generation for simple shell."""
        shell = PAOShell(n=2, l=1, nzeta=2, rc=[5.0, 3.5])
        lines = shell.to_fdf_lines()

        assert len(lines) == 2
        assert lines[0] == "  n=2  1  2"
        assert lines[1] == "    5.0  3.5"

    def test_to_fdf_lines_polarization(self):
        """Test FDF generation with polarization flag."""
        shell = PAOShell(l=2, nzeta=1, rc=[4.0], polarization=True)
        lines = shell.to_fdf_lines()

        assert len(lines) == 2
        assert "P" in lines[0]
        assert lines[1] == "    4.0"

    def test_to_fdf_lines_split_norm(self):
        """Test FDF generation with split norm."""
        shell = PAOShell(
            n=2, l=0, nzeta=2, rc=[6.0, 0.0], split_norm_flag=True, split_norm=0.15
        )
        lines = shell.to_fdf_lines()

        assert "S" in lines[0]
        assert "0.15" in lines[0]

    def test_to_fdf_lines_soft_confinement(self):
        """Test FDF generation with soft confinement."""
        shell = PAOShell(
            n=2,
            l=0,
            nzeta=2,
            rc=[6.0, 0.0],
            soft_conf_flag=True,
            v0_soft=40.0,
            ri_soft=0.9,
        )
        lines = shell.to_fdf_lines()

        assert "E" in lines[0]
        assert "40.0" in lines[0]
        assert "0.9" in lines[0]

    def test_all_angular_momenta(self):
        """Test all angular momentum symbols."""
        assert PAOShell(l=0, nzeta=1, rc=[5.0]).l_symbol == "s"
        assert PAOShell(l=1, nzeta=1, rc=[5.0]).l_symbol == "p"
        assert PAOShell(l=2, nzeta=1, rc=[5.0]).l_symbol == "d"
        assert PAOShell(l=3, nzeta=1, rc=[5.0]).l_symbol == "f"


class TestPAOBasisSpecies:
    """Tests for PAOBasisSpecies dataclass."""

    def test_simple_species(self):
        """Test creation of species with shells."""
        shells = [
            PAOShell(n=2, l=0, nzeta=2, rc=[6.0, 0.0]),
            PAOShell(n=2, l=1, nzeta=2, rc=[7.0, 0.0]),
        ]
        species = PAOBasisSpecies(label="O", shells=shells)

        assert species.label == "O"
        assert len(species.shells) == 2

    def test_to_fdf_lines_basic(self):
        """Test FDF generation for basic species."""
        shells = [
            PAOShell(n=2, l=0, nzeta=2, rc=[6.0, 0.0]),
            PAOShell(n=2, l=1, nzeta=2, rc=[7.0, 0.0]),
        ]
        species = PAOBasisSpecies(label="O", shells=shells)
        lines = species.to_fdf_lines()

        # First line: O 2 (label and number of shells)
        assert lines[0] == "O  2"

        # Should have 5 lines total: 1 header + 2 shells * 2 lines each
        assert len(lines) == 5

    def test_to_fdf_lines_with_basis_type(self):
        """Test FDF generation with basis_type."""
        shells = [PAOShell(n=2, l=0, nzeta=1, rc=[5.0])]
        species = PAOBasisSpecies(label="O", shells=shells, basis_type="split")
        lines = species.to_fdf_lines()

        assert "split" in lines[0]

    def test_to_fdf_lines_with_ionic_charge(self):
        """Test FDF generation with ionic charge."""
        shells = [PAOShell(n=2, l=0, nzeta=1, rc=[5.0])]
        species = PAOBasisSpecies(label="O", shells=shells, ionic_charge=-2.0)
        lines = species.to_fdf_lines()

        assert "-2.0" in lines[0]

    def test_species_variant_labels(self):
        """Test species with variant labels."""
        shells = [PAOShell(n=2, l=0, nzeta=2, rc=[6.0, 0.0])]

        species_surface = PAOBasisSpecies(label="O_surface", shells=shells)
        species_bulk = PAOBasisSpecies(label="O_bulk", shells=shells)
        species_ghost = PAOBasisSpecies(label="O_ghost", shells=shells)

        assert species_surface.to_fdf_lines()[0].startswith("O_surface")
        assert species_bulk.to_fdf_lines()[0].startswith("O_bulk")
        assert species_ghost.to_fdf_lines()[0].startswith("O_ghost")


class TestCreatePAOBasis:
    """Tests for create_pao_basis helper function."""

    def test_simple_basis(self):
        """Test creation of simple basis for one species."""
        basis_spec = {
            "Si": {
                "shells": [
                    {"n": 3, "l": 0, "nzeta": 2, "rc": [5.0, 3.5]},
                    {"n": 3, "l": 1, "nzeta": 2, "rc": [5.5, 4.0]},
                ]
            }
        }

        lines = create_pao_basis(basis_spec)

        # Should have: 1 header + 2 shells * 2 lines = 5 lines
        assert len(lines) == 5
        assert lines[0] == "Si  2"

    def test_multiple_species(self):
        """Test basis with multiple species."""
        basis_spec = {
            "O_surface": {
                "shells": [
                    {"n": 2, "l": 0, "nzeta": 2, "rc": [6.0, 0.0]},
                    {"n": 2, "l": 1, "nzeta": 2, "rc": [7.0, 0.0]},
                ]
            },
            "O_bulk": {
                "shells": [
                    {"n": 2, "l": 0, "nzeta": 2, "rc": [4.5, 0.0]},
                    {"n": 2, "l": 1, "nzeta": 2, "rc": [5.5, 0.0]},
                ]
            },
        }

        lines = create_pao_basis(basis_spec)

        # Each species: 1 header + 2 shells * 2 lines = 5 lines per species = 10 total
        assert len(lines) == 10

        # Check species labels appear
        assert any("O_surface" in line for line in lines)
        assert any("O_bulk" in line for line in lines)

    def test_polarization_orbitals(self):
        """Test basis with polarization orbitals."""
        basis_spec = {
            "Si": {
                "shells": [
                    {"n": 3, "l": 0, "nzeta": 2, "rc": [5.0, 0.0]},
                    {"n": 3, "l": 1, "nzeta": 2, "rc": [5.5, 0.0]},
                    {"l": 2, "nzeta": 1, "rc": [4.5], "polarization": True},
                ]
            }
        }

        lines = create_pao_basis(basis_spec)

        # Should have polarization flag in one of the lines
        assert any("P" in line for line in lines)

    def test_all_optional_flags(self):
        """Test basis with all optional shell parameters."""
        basis_spec = {
            "Fe": {
                "shells": [
                    {
                        "n": 3,
                        "l": 2,
                        "nzeta": 2,
                        "rc": [5.0, 0.0],
                        "polarization": True,
                        "split_norm_flag": True,
                        "split_norm": 0.15,
                        "soft_conf_flag": True,
                        "v0_soft": 40.0,
                        "ri_soft": 0.9,
                    }
                ]
            }
        }

        lines = create_pao_basis(basis_spec)

        # Check that flags appear
        shell_line = lines[1]  # First shell definition line
        assert "P" in shell_line  # Polarization
        assert "S" in shell_line  # Split norm
        assert "E" in shell_line  # Soft confinement

    def test_with_basis_type(self):
        """Test basis specification with basis_type."""
        basis_spec = {
            "O": {
                "shells": [{"n": 2, "l": 0, "nzeta": 1, "rc": [5.0]}],
                "basis_type": "split",
            }
        }

        lines = create_pao_basis(basis_spec)
        assert "split" in lines[0]

    def test_with_ionic_charge(self):
        """Test basis specification with ionic charge."""
        basis_spec = {
            "O": {
                "shells": [{"n": 2, "l": 0, "nzeta": 1, "rc": [5.0]}],
                "ionic_charge": -2.0,
            }
        }

        lines = create_pao_basis(basis_spec)
        assert "-2.0" in lines[0]

    def test_ghost_atoms(self):
        """Test basis for ghost atoms."""
        basis_spec = {
            "O_ghost": {
                "shells": [
                    {"l": 0, "nzeta": 1, "rc": [3.0]},
                    {"l": 1, "nzeta": 1, "rc": [3.5]},
                ]
            }
        }

        lines = create_pao_basis(basis_spec)

        assert lines[0] == "O_ghost  2"
        # Ghost atoms typically have simpler basis (no n specified, SZ)

    def test_realistic_surface_example(self):
        """Test realistic surface calculation basis."""
        basis_spec = {
            "Ti_surface": {
                "shells": [
                    {"n": 3, "l": 2, "nzeta": 2, "rc": [6.0, 0.0]},
                    {"n": 4, "l": 0, "nzeta": 2, "rc": [7.0, 0.0]},
                    {"l": 3, "nzeta": 1, "rc": [5.0], "polarization": True},
                ]
            },
            "Ti_bulk": {
                "shells": [
                    {"n": 3, "l": 2, "nzeta": 2, "rc": [5.0, 0.0]},
                    {"n": 4, "l": 0, "nzeta": 2, "rc": [6.0, 0.0]},
                ]
            },
            "O_surface": {
                "shells": [
                    {"n": 2, "l": 0, "nzeta": 2, "rc": [6.0, 0.0]},
                    {
                        "n": 2,
                        "l": 1,
                        "nzeta": 2,
                        "rc": [7.0, 0.0],
                        "polarization": True,
                    },
                ]
            },
            "O_bulk": {
                "shells": [
                    {"n": 2, "l": 0, "nzeta": 2, "rc": [4.5, 0.0]},
                    {"n": 2, "l": 1, "nzeta": 2, "rc": [5.5, 0.0]},
                ]
            },
        }

        lines = create_pao_basis(basis_spec)

        # 4 species, each with 2-3 shells
        # Ti_surface: 1 + 3*2 = 7
        # Ti_bulk: 1 + 2*2 = 5
        # O_surface: 1 + 2*2 = 5
        # O_bulk: 1 + 2*2 = 5
        assert len(lines) == 22

        # Check all species present
        assert any("Ti_surface" in line for line in lines)
        assert any("Ti_bulk" in line for line in lines)
        assert any("O_surface" in line for line in lines)
        assert any("O_bulk" in line for line in lines)

    def test_empty_basis(self):
        """Test that empty basis spec returns empty list."""
        lines = create_pao_basis({})
        assert lines == []

    def test_pao_shell_objects_directly(self):
        """Test passing PAOShell objects directly (not dicts)."""
        shells = [
            PAOShell(n=2, l=0, nzeta=2, rc=[6.0, 0.0]),
            PAOShell(n=2, l=1, nzeta=2, rc=[7.0, 0.0]),
        ]

        basis_spec = {"O": {"shells": shells}}

        lines = create_pao_basis(basis_spec)

        assert len(lines) == 5  # 1 header + 2 shells * 2 lines
        assert lines[0] == "O  2"
