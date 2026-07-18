"""Tests for thermodynamic analysis (CHE model) in electrocatalysis."""

import pytest

from atomate2.siesta.flows.electrocatalysis.analysis.thermodynamics import (
    calculate_free_energy_corrections,
    calculate_reaction_free_energies,
    identify_rate_limiting_step,
)


class TestFreeEnergyCorrections:
    """Tests for calculate_free_energy_corrections function."""

    def test_standard_conditions(self):
        """Test corrections at standard conditions (298.15 K, 1 atm)."""
        corrections = calculate_free_energy_corrections()

        # Test standard values
        assert corrections["H2"] == 0.00  # Reference
        assert corrections["H2O"] == pytest.approx(0.67, abs=0.01)
        assert corrections["O2"] == pytest.approx(0.05, abs=0.01)
        assert corrections["CO2"] == pytest.approx(0.45, abs=0.01)
        assert corrections["CO"] == pytest.approx(0.13, abs=0.01)

    def test_all_molecules_present(self):
        """Test that all expected molecules have corrections."""
        corrections = calculate_free_energy_corrections()

        expected_molecules = [
            "H2",
            "H2O",
            "O2",
            "CO2",
            "CO",
            "CH4",
            "N2",
            "NH3",
            "C2H4",
            "O*",
            "OH*",
            "OOH*",
            "H*",
        ]

        for molecule in expected_molecules:
            assert molecule in corrections
            assert isinstance(corrections[molecule], float)

    def test_temperature_scaling(self):
        """Test temperature dependence of corrections."""
        # Standard temperature
        corr_298 = calculate_free_energy_corrections(temperature=298.15)

        # Elevated temperature
        corr_350 = calculate_free_energy_corrections(temperature=350.0)

        # H2O correction should scale with temperature
        # At higher T, -TS term is larger (more negative)
        # But total correction should scale roughly linearly
        scale_factor = 350.0 / 298.15
        assert corr_350["H2O"] == pytest.approx(
            corr_298["H2O"] * scale_factor, rel=0.01
        )

    def test_custom_corrections(self):
        """Test custom ZPE and entropy corrections."""
        custom_zpe = {"O2": 0.10, "H2O": 0.56}
        custom_entropy = {"O2": -0.05, "H2O": 0.11}

        corrections = calculate_free_energy_corrections(
            zpe_corrections=custom_zpe, entropy_corrections=custom_entropy
        )

        # O2: ZPE + entropy
        assert corrections["O2"] == pytest.approx(0.10 + (-0.05), abs=0.01)
        # H2O: ZPE + entropy
        assert corrections["H2O"] == pytest.approx(0.56 + 0.11, abs=0.01)


class TestReactionFreeEnergies:
    """Tests for calculate_reaction_free_energies function."""

    def test_simple_orr_pathway(self):
        """Test ORR pathway free energy calculation."""
        # Simplified ORR pathway (4 steps)
        pathway_steps = [
            {
                "label": "O2_ads",
                "energy": -500.0,
                "species": "O2",
                "n_H": 0,
                "n_e": 0,
            },
            {
                "label": "OOH*",
                "energy": -498.5,
                "species": "H",
                "n_H": 1,
                "n_e": 1,
            },
            {
                "label": "O*",
                "energy": -497.0,
                "species": "H2O",
                "n_H": 1,
                "n_e": 1,
            },
            {
                "label": "OH*",
                "energy": -496.0,
                "species": "H",
                "n_H": 1,
                "n_e": 1,
            },
        ]

        gas_energies = {"H2": -6.77, "H2O": -14.22, "O2": -9.86}
        clean_surf = -494.0

        result = calculate_reaction_free_energies(
            surface_name="test_surface",
            pathway_steps=pathway_steps,
            gas_phase_energies=gas_energies,
            clean_surface_energy=clean_surf,
            temperature=298.15,
            ph=0.0,
            potential=0.0,
        )

        # Check output structure
        assert "step_labels" in result
        assert "absolute_energies" in result
        assert "delta_E" in result
        assert "delta_G" in result
        assert "cumulative_G" in result
        assert "thermodynamic_overpotential" in result

        # Check correct number of steps
        assert len(result["step_labels"]) == 4
        assert len(result["delta_E"]) == 4
        assert len(result["delta_G"]) == 4

    def test_ph_dependence(self):
        """Test pH correction on free energies."""
        pathway_steps = [
            {
                "label": "Step1",
                "energy": -100.0,
                "species": "H",
                "n_H": 1,
                "n_e": 1,
            },
        ]

        gas_energies = {"H2": -6.77}
        clean_surf = -98.0

        # pH = 0 (acidic)
        result_acidic = calculate_reaction_free_energies(
            surface_name="test",
            pathway_steps=pathway_steps,
            gas_phase_energies=gas_energies,
            clean_surface_energy=clean_surf,
            ph=0.0,
        )

        # pH = 14 (alkaline)
        result_alkaline = calculate_reaction_free_energies(
            surface_name="test",
            pathway_steps=pathway_steps,
            gas_phase_energies=gas_energies,
            clean_surface_energy=clean_surf,
            ph=14.0,
        )

        # pH correction: ΔG_pH = -0.059 eV × pH at 298 K
        # Higher pH should increase ΔG for reduction (make it less favorable)
        assert result_alkaline["delta_G"][0] > result_acidic["delta_G"][0]

        # Difference should be ~0.059 × 14 ≈ 0.83 eV
        delta_ph_effect = result_alkaline["delta_G"][0] - result_acidic["delta_G"][0]
        assert delta_ph_effect == pytest.approx(0.059 * 14, rel=0.1)

    def test_potential_dependence(self):
        """Test electrode potential effect on free energies."""
        pathway_steps = [
            {
                "label": "Step1",
                "energy": -100.0,
                "species": "H",
                "n_H": 1,
                "n_e": 1,
            },
        ]

        gas_energies = {"H2": -6.77}
        clean_surf = -98.0

        # U = 0 V
        result_0V = calculate_reaction_free_energies(
            surface_name="test",
            pathway_steps=pathway_steps,
            gas_phase_energies=gas_energies,
            clean_surface_energy=clean_surf,
            potential=0.0,
        )

        # U = 1.0 V
        result_1V = calculate_reaction_free_energies(
            surface_name="test",
            pathway_steps=pathway_steps,
            gas_phase_energies=gas_energies,
            clean_surface_energy=clean_surf,
            potential=1.0,
        )

        # ΔG(U) = ΔG(U=0) - n_e × eU
        # At U = 1 V, reduction becomes 1 eV more favorable
        assert result_1V["delta_G"][0] == pytest.approx(
            result_0V["delta_G"][0] - 1.0, abs=0.01
        )


class TestRateLimitingStep:
    """Tests for identify_rate_limiting_step function."""

    def test_simple_rls_identification(self):
        """Test RLS identification from delta_G list."""
        delta_G = [0.45, 1.20, 0.80, 0.60]
        labels = ["Step1", "Step2", "Step3", "Step4"]

        rls = identify_rate_limiting_step(delta_G, labels)

        assert rls["rls_index"] == 1  # Second step (0-indexed)
        assert rls["rls_label"] == "Step2"
        assert rls["rls_delta_G"] == pytest.approx(1.20)

    def test_rls_without_labels(self):
        """Test RLS identification without step labels."""
        delta_G = [0.3, 0.8, 0.5]

        rls = identify_rate_limiting_step(delta_G)

        assert rls["rls_index"] == 1
        assert rls["rls_label"] is None
        assert rls["rls_delta_G"] == pytest.approx(0.8)

    def test_empty_delta_g(self):
        """Test RLS with empty delta_G list."""
        rls = identify_rate_limiting_step([])

        assert rls["rls_index"] == -1
        assert rls["rls_label"] is None
        assert rls["rls_delta_G"] == 0.0

    def test_all_negative_delta_g(self):
        """Test RLS when all steps are downhill."""
        delta_G = [-0.2, -0.5, -0.3, -0.1]

        rls = identify_rate_limiting_step(delta_G)

        # Should identify least negative (least favorable downhill step)
        assert rls["rls_index"] == 3
        assert rls["rls_delta_G"] == pytest.approx(-0.1)

    def test_thermoneutral_pathway(self):
        """Test ideal pathway with thermoneutral steps."""
        # Ideal ORR: each step is 1.23/4 = 0.3075 eV
        delta_G = [0.3075, 0.3075, 0.3075, 0.3075]

        rls = identify_rate_limiting_step(delta_G)

        # All steps are equal, RLS should be first one (index 0)
        assert rls["rls_index"] == 0
        assert rls["rls_delta_G"] == pytest.approx(0.3075)
