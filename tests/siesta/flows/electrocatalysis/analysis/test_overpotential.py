"""Tests for overpotential calculations in electrocatalysis."""

import pytest

from atomate2.siesta.flows.electrocatalysis.analysis.overpotential import (
    U_EQ_HER,
    U_EQ_OER,
    U_EQ_ORR,
    calculate_bifunctional_gap,
    calculate_her_overpotential,
    calculate_oer_overpotential,
    calculate_orr_overpotential,
)


class TestORROverpotential:
    """Tests for calculate_orr_overpotential function."""

    def test_typical_orr_overpotential(self):
        """Test ORR overpotential calculation with typical values."""
        # Typical ORR pathway with one large uphill step
        delta_G = [0.45, 1.20, 0.80, 0.60]

        result = calculate_orr_overpotential(delta_G)

        # Check output structure
        assert "eta_ORR" in result
        assert "U_onset" in result
        assert "U_equilibrium" in result
        assert "max_delta_G" in result
        assert "rls_index" in result

        # Overpotential should equal max uphill step
        assert result["eta_ORR"] == pytest.approx(1.20)
        assert result["max_delta_G"] == pytest.approx(1.20)
        assert result["rls_index"] == 1

        # U_onset = U_eq - η
        assert result["U_onset"] == pytest.approx(1.23 - 1.20, abs=0.01)
        assert result["U_equilibrium"] == U_EQ_ORR

    def test_ideal_orr_catalyst(self):
        """Test ideal ORR catalyst (all steps thermoneutral)."""
        # Ideal: each step is 1.23/4 = 0.3075 eV
        delta_G = [0.3075, 0.3075, 0.3075, 0.3075]

        result = calculate_orr_overpotential(delta_G)

        # Overpotential should be 0.3075 V (each step is the bottleneck)
        assert result["eta_ORR"] == pytest.approx(0.3075, abs=0.01)
        assert result["U_onset"] == pytest.approx(1.23 - 0.3075, abs=0.01)

    def test_perfect_catalyst(self):
        """Test perfect catalyst (zero overpotential)."""
        # All steps downhill
        delta_G = [0.0, 0.0, 0.0, 0.0]

        result = calculate_orr_overpotential(delta_G)

        assert result["eta_ORR"] == pytest.approx(0.0)
        assert result["U_onset"] == pytest.approx(1.23)

    def test_empty_delta_g(self):
        """Test with empty delta_G list."""
        result = calculate_orr_overpotential([])

        assert result["eta_ORR"] == 0.0
        assert result["U_onset"] == U_EQ_ORR
        assert result["max_delta_G"] == 0.0
        assert result["rls_index"] == -1

    def test_all_downhill_steps(self):
        """Test when all steps are downhill (exothermic)."""
        delta_G = [-0.3, -0.5, -0.2, -0.4]

        result = calculate_orr_overpotential(delta_G)

        # Maximum (least negative) is -0.2
        assert result["max_delta_G"] == pytest.approx(-0.2)
        # Overpotential = U_eq - (U_eq - max_delta_G) = max_delta_G
        # But max_delta_G is negative, so η = -(-0.2) = 0.2? No!
        # Actually: U_onset = U_eq - max_delta_G = 1.23 - (-0.2) = 1.43
        # η = U_eq - U_onset = 1.23 - 1.43 = -0.2 (negative overpotential!)
        # This means the reaction is spontaneous even without applied potential
        assert result["eta_ORR"] == pytest.approx(-0.2)


class TestOEROverpotential:
    """Tests for calculate_oer_overpotential function."""

    def test_typical_oer_overpotential(self):
        """Test OER overpotential calculation."""
        # OER pathway (reverse of ORR)
        delta_G = [0.60, 0.80, 1.20, 0.45]

        result = calculate_oer_overpotential(delta_G)

        # Check output structure
        assert "eta_OER" in result
        assert "U_onset" in result
        assert "U_equilibrium" in result
        assert "max_delta_G" in result
        assert "rls_index" in result

        # Overpotential should equal max uphill step
        assert result["eta_OER"] == pytest.approx(1.20)
        assert result["max_delta_G"] == pytest.approx(1.20)
        assert result["rls_index"] == 2

        # For OER: U_onset = U_eq + η
        assert result["U_onset"] == pytest.approx(1.23 + 1.20, abs=0.01)
        assert result["U_equilibrium"] == U_EQ_OER

    def test_ideal_oer_catalyst(self):
        """Test ideal OER catalyst."""
        # Ideal: each step is 1.23/4 = 0.3075 eV
        delta_G = [0.3075, 0.3075, 0.3075, 0.3075]

        result = calculate_oer_overpotential(delta_G)

        assert result["eta_OER"] == pytest.approx(0.3075, abs=0.01)
        assert result["U_onset"] == pytest.approx(1.23 + 0.3075, abs=0.01)

    def test_empty_delta_g(self):
        """Test with empty delta_G list."""
        result = calculate_oer_overpotential([])

        assert result["eta_OER"] == 0.0
        assert result["U_onset"] == U_EQ_OER
        assert result["max_delta_G"] == 0.0


class TestBifunctionalGap:
    """Tests for calculate_bifunctional_gap function."""

    def test_typical_bifunctional_gap(self):
        """Test bifunctional overpotential gap calculation."""
        delta_G_ORR = [0.45, 1.20, 0.80, 0.60]
        delta_G_OER = [0.60, 0.80, 1.20, 0.45]

        result = calculate_bifunctional_gap(delta_G_ORR, delta_G_OER)

        # Check output structure
        assert "eta_ORR" in result
        assert "eta_OER" in result
        assert "overpotential_gap" in result
        assert "U_ORR_onset" in result
        assert "U_OER_onset" in result
        assert "voltage_window" in result

        # Both pathways have max ΔG = 1.20 eV
        assert result["eta_ORR"] == pytest.approx(1.20)
        assert result["eta_OER"] == pytest.approx(1.20)

        # Gap = η_ORR + η_OER
        assert result["overpotential_gap"] == pytest.approx(2.40)

        # Voltage window
        U_ORR = 1.23 - 1.20  # 0.03 V
        U_OER = 1.23 + 1.20  # 2.43 V
        assert result["U_ORR_onset"] == pytest.approx(U_ORR, abs=0.01)
        assert result["U_OER_onset"] == pytest.approx(U_OER, abs=0.01)
        assert result["voltage_window"] == pytest.approx(U_OER - U_ORR, abs=0.01)

    def test_excellent_bifunctional_catalyst(self):
        """Test excellent bifunctional catalyst (gap < 0.4 V)."""
        # Very good catalyst with low overpotentials
        delta_G_ORR = [0.2, 0.2, 0.2, 0.2]
        delta_G_OER = [0.2, 0.2, 0.2, 0.2]

        result = calculate_bifunctional_gap(delta_G_ORR, delta_G_OER)

        assert result["eta_ORR"] == pytest.approx(0.2)
        assert result["eta_OER"] == pytest.approx(0.2)
        assert result["overpotential_gap"] == pytest.approx(0.4)

        # Excellent for metal-air batteries!

    def test_poor_bifunctional_catalyst(self):
        """Test poor bifunctional catalyst (gap > 1.0 V)."""
        delta_G_ORR = [0.5, 1.5, 0.5, 0.5]
        delta_G_OER = [0.5, 1.5, 0.5, 0.5]

        result = calculate_bifunctional_gap(delta_G_ORR, delta_G_OER)

        assert result["eta_ORR"] == pytest.approx(1.5)
        assert result["eta_OER"] == pytest.approx(1.5)
        assert result["overpotential_gap"] == pytest.approx(3.0)

        # Poor bifunctional performance


class TestHEROverpotential:
    """Tests for calculate_her_overpotential function."""

    def test_pt_like_catalyst(self):
        """Test Pt-like catalyst (nearly ideal, ΔG_H ≈ 0)."""
        delta_G_H = 0.05  # Very small, near-optimal

        result = calculate_her_overpotential(delta_G_H)

        assert "eta_HER" in result
        assert "delta_G_H" in result
        assert "U_equilibrium" in result

        assert result["eta_HER"] == pytest.approx(0.05)
        assert result["delta_G_H"] == pytest.approx(0.05)
        assert result["U_equilibrium"] == U_EQ_HER

    def test_weak_binding(self):
        """Test weak H binding (Au-like)."""
        delta_G_H = 0.50  # Positive: weak binding

        result = calculate_her_overpotential(delta_G_H)

        # Overpotential = |ΔG_H|
        assert result["eta_HER"] == pytest.approx(0.50)
        assert result["delta_G_H"] == pytest.approx(0.50)

    def test_strong_binding(self):
        """Test strong H binding (W-like)."""
        delta_G_H = -0.60  # Negative: strong binding

        result = calculate_her_overpotential(delta_G_H)

        # Overpotential = |ΔG_H|
        assert result["eta_HER"] == pytest.approx(0.60)
        assert result["delta_G_H"] == pytest.approx(-0.60)

    def test_perfect_her_catalyst(self):
        """Test perfect HER catalyst (ΔG_H = 0)."""
        delta_G_H = 0.0

        result = calculate_her_overpotential(delta_G_H)

        assert result["eta_HER"] == pytest.approx(0.0)
        assert result["delta_G_H"] == pytest.approx(0.0)

    def test_volcano_peak(self):
        """Test catalysts around volcano peak."""
        # Test range of ΔG_H values around peak
        delta_G_values = [-0.2, -0.1, 0.0, 0.1, 0.2]

        overpotentials = []
        for dG_H in delta_G_values:
            result = calculate_her_overpotential(dG_H)
            overpotentials.append(result["eta_HER"])

        # Minimum overpotential should be at ΔG_H = 0
        min_eta = min(overpotentials)
        assert min_eta == pytest.approx(0.0)
        assert overpotentials[2] == min_eta  # Index 2 is ΔG_H = 0
