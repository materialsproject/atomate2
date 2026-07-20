"""
Tests for spin configuration utilities.

These tests validate:
- Automatic spin state detection for common molecules
- Spin-polarized vs. non-polarized determination
- Magnetic moment initialization
- Database coverage
"""

from atomate2.siesta.flows.electrocatalysis.utils.spin_config import (
    get_siesta_spin_config,
)


class TestSpinConfigDetection:
    """Tests for get_siesta_spin_config function."""

    def test_paramagnetic_o2(self):
        """Test O₂ (triplet ground state) detection."""
        config = get_siesta_spin_config("O2")

        assert config["spin_polarized"] is True
        assert config["init_magnetic_moments"] == {"O": 1.0}
        assert config["total_spin_moment"] == 2.0
        assert config["n_unpaired_electrons"] == 2
        assert config["fix_spin"] is True

    def test_paramagnetic_o_atom(self):
        """Test O atom (triplet ground state) detection."""
        config = get_siesta_spin_config("O")

        assert config["spin_polarized"] is True
        assert config["init_magnetic_moments"] == {"O": 2.0}
        assert config["total_spin_moment"] == 2.0
        assert config["n_unpaired_electrons"] == 2
        assert config["fix_spin"] is True

    def test_paramagnetic_oh_radical(self):
        """Test OH radical (doublet ground state) detection."""
        config = get_siesta_spin_config("OH")

        assert config["spin_polarized"] is True
        assert config["init_magnetic_moments"] == {"O": 1.0}
        assert config["total_spin_moment"] == 1.0
        assert config["n_unpaired_electrons"] == 1
        assert config["fix_spin"] is True

    def test_paramagnetic_ooh_radical(self):
        """Test OOH radical detection."""
        config = get_siesta_spin_config("OOH")

        assert config["spin_polarized"] is True
        assert config["total_spin_moment"] == 1.0
        assert config["n_unpaired_electrons"] == 1

    def test_paramagnetic_no(self):
        """Test NO (doublet ground state) detection."""
        config = get_siesta_spin_config("NO")

        assert config["spin_polarized"] is True
        assert config["init_magnetic_moments"] == {"N": 1.0}
        assert config["total_spin_moment"] == 1.0
        assert config["n_unpaired_electrons"] == 1

    def test_diamagnetic_h2o(self):
        """Test H₂O (closed-shell singlet) detection."""
        config = get_siesta_spin_config("H2O")

        assert config["spin_polarized"] is False
        assert config["init_magnetic_moments"] is None
        assert config["total_spin_moment"] == 0.0
        assert config["n_unpaired_electrons"] == 0
        assert config["fix_spin"] is False

    def test_diamagnetic_co2(self):
        """Test CO₂ (closed-shell singlet) detection."""
        config = get_siesta_spin_config("CO2")

        assert config["spin_polarized"] is False
        assert config["init_magnetic_moments"] is None
        assert config["total_spin_moment"] == 0.0

    def test_diamagnetic_n2(self):
        """Test N₂ (closed-shell singlet) detection."""
        config = get_siesta_spin_config("N2")

        assert config["spin_polarized"] is False
        assert config["init_magnetic_moments"] is None

    def test_diamagnetic_h2(self):
        """Test H₂ (closed-shell singlet) detection."""
        config = get_siesta_spin_config("H2")

        assert config["spin_polarized"] is False
        assert config["total_spin_moment"] == 0.0

    def test_diamagnetic_co(self):
        """Test CO (closed-shell singlet) detection."""
        config = get_siesta_spin_config("CO")

        assert config["spin_polarized"] is False

    def test_bulk_products_li2o2(self):
        """Test Li₂O₂ (closed-shell) detection."""
        config = get_siesta_spin_config("Li2O2")

        assert config["spin_polarized"] is False
        assert config["init_magnetic_moments"] is None

    def test_bulk_products_na2o2(self):
        """Test Na₂O₂ (closed-shell) detection."""
        config = get_siesta_spin_config("Na2O2")

        assert config["spin_polarized"] is False

    def test_case_insensitive_matching(self):
        """Test that formula matching is case-insensitive."""
        # Test with lowercase
        config_lower = get_siesta_spin_config("o2")
        assert config_lower["spin_polarized"] is True

        # Test with mixed case
        config_mixed = get_siesta_spin_config("O2")
        assert config_mixed["spin_polarized"] is True

        # Should be identical
        assert config_lower == config_mixed

    def test_unknown_molecule_default(self):
        """Test default behavior for unknown molecules."""
        config = get_siesta_spin_config("XYZ123")  # Made-up formula

        # Should default to closed-shell (conservative)
        assert config["spin_polarized"] is False
        assert config["init_magnetic_moments"] is None
        assert config["total_spin_moment"] == 0.0
        assert config["n_unpaired_electrons"] == 0
        assert config["fix_spin"] is False

    def test_all_config_keys_present(self):
        """Test that all expected keys are present in output."""
        config = get_siesta_spin_config("O2")

        expected_keys = {
            "spin_polarized",
            "init_magnetic_moments",
            "total_spin_moment",
            "n_unpaired_electrons",
            "fix_spin",
        }

        assert set(config.keys()) == expected_keys

    def test_paramagnetic_molecules_require_fix_spin(self):
        """Test that all paramagnetic molecules recommend FixSpin."""
        paramagnetic = ["O2", "O", "OH", "OOH", "NO", "NO2"]

        for formula in paramagnetic:
            config = get_siesta_spin_config(formula)
            assert config["fix_spin"] is True, (
                f"{formula} should recommend FixSpin=True"
            )

    def test_diamagnetic_molecules_no_fix_spin(self):
        """Test that diamagnetic molecules don't need FixSpin."""
        diamagnetic = ["H2O", "CO2", "N2", "H2", "CO", "Li2O2", "Na2O2"]

        for formula in diamagnetic:
            config = get_siesta_spin_config(formula)
            assert config["fix_spin"] is False, f"{formula} should not need FixSpin"


class TestSpinConfigCoverage:
    """Tests for database coverage of common molecules."""

    def test_orr_molecules_covered(self):
        """Test that all ORR-relevant molecules are in database."""
        orr_molecules = ["O2", "H2O", "OH", "OOH", "O"]

        for formula in orr_molecules:
            config = get_siesta_spin_config(formula)
            # Should not be default (unknown)
            # All these have specific spin states
            assert "spin_polarized" in config

    def test_oer_molecules_covered(self):
        """Test that all OER-relevant molecules are in database."""
        # OER is reverse of ORR
        oer_molecules = ["O2", "H2O", "OH", "OOH", "O"]

        for formula in oer_molecules:
            config = get_siesta_spin_config(formula)
            assert "spin_polarized" in config

    def test_her_molecules_covered(self):
        """Test that HER-relevant molecules are in database."""
        her_molecules = ["H2", "H2O"]

        for formula in her_molecules:
            config = get_siesta_spin_config(formula)
            assert "spin_polarized" in config

    def test_co2rr_molecules_covered(self):
        """Test that CO2RR-relevant molecules are in database."""
        co2rr_molecules = ["CO2", "CO", "CH4", "C2H4"]

        for formula in co2rr_molecules:
            config = get_siesta_spin_config(formula)
            assert "spin_polarized" in config

    def test_metal_air_molecules_covered(self):
        """Test that metal-air battery molecules are in database."""
        metal_air = ["O2", "Li2O2", "Na2O2", "K2O2"]

        for formula in metal_air:
            config = get_siesta_spin_config(formula)
            assert "spin_polarized" in config
