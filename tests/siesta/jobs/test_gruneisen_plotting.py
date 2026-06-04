"""Tests for Grüneisen parameter plotting utilities."""

import pytest
import numpy as np
from pymatgen.core import Lattice, Structure


def get_si_structure():
    """Get a simple Si structure for testing."""
    return Structure(
        lattice=Lattice.cubic(5.43),
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )


def get_mock_gruneisen_doc():
    """Create a mock Grüneisen document for testing."""
    from pymatgen.phonon.gruneisen import GruneisenParameter

    structure = get_si_structure()

    # Create mock data
    n_qpoints = 10
    n_modes = 3 * len(structure)  # 3N modes

    frequencies = np.random.uniform(0, 20, (n_qpoints, n_modes))
    gruneisen = np.random.uniform(-1, 3, (n_qpoints, n_modes))

    # Create minimal GruneisenParameter object
    grun_param = GruneisenParameter(
        qpoints=np.random.rand(n_qpoints, 3),
        gruneisen=gruneisen,
        frequencies=frequencies,
        structure=structure,
    )

    return {
        "structure": structure,
        "code": "siesta",
        "gruneisen_parameter": grun_param,
        "gruneisen_band_structure": None,  # Would be GruneisenPhononBandStructureSymmLine
        "derived_properties": {
            "average_gruneisen": 2.0,
            "thermal_conductivity_slack": 50.0,
        },
        "phonon_runs_has_imaginary_modes": {
            "ground": False,
            "plus": False,
            "minus": False,
        },
    }


def test_calculate_thermal_expansion():
    """Test thermal expansion calculation."""
    from atomate2.siesta.jobs.phonon.gruneisen_plotting import (
        calculate_thermal_expansion,
    )

    gruneisen_doc = get_mock_gruneisen_doc()

    # Calculate thermal expansion with provided bulk modulus
    result = calculate_thermal_expansion.original(
        gruneisen_doc=gruneisen_doc,
        bulk_modulus=100.0,  # GPa
        temperature_range=(0, 500),
        n_points=51,
    )

    assert "temperatures" in result
    assert "alpha_v" in result
    assert "alpha_l" in result
    assert "bulk_modulus" in result
    assert result["bulk_modulus"] == 100.0

    # Check arrays have correct length
    assert len(result["temperatures"]) == 51
    assert len(result["alpha_v"]) == 51
    assert len(result["alpha_l"]) == 51

    # Check physical relationship: alpha_l = alpha_v / 3
    alpha_v = np.array(result["alpha_v"])
    alpha_l = np.array(result["alpha_l"])
    np.testing.assert_allclose(alpha_l, alpha_v / 3, rtol=1e-10)

    # Check non-negative at positive temperatures
    assert all(a >= 0 for a in result["alpha_v"][1:])


def test_calculate_thermal_expansion_no_bulk_modulus():
    """Test thermal expansion calculation without bulk modulus (estimated)."""
    from atomate2.siesta.jobs.phonon.gruneisen_plotting import (
        calculate_thermal_expansion,
    )

    gruneisen_doc = get_mock_gruneisen_doc()

    # Should estimate bulk modulus
    result = calculate_thermal_expansion.original(
        gruneisen_doc=gruneisen_doc,
        bulk_modulus=None,
        temperature_range=(0, 300),
        n_points=31,
    )

    assert "bulk_modulus" in result
    assert result["bulk_modulus"] > 0  # Should have estimated something


def test_write_gruneisen_summary(tmp_path):
    """Test Grüneisen summary text file generation."""
    from atomate2.siesta.jobs.phonon.gruneisen_plotting import write_gruneisen_summary

    gruneisen_doc = get_mock_gruneisen_doc()

    result = write_gruneisen_summary.original(
        gruneisen_doc=gruneisen_doc,
        output_dir=tmp_path,
        filename="test_summary.txt",
    )

    assert "summary_file" in result
    summary_file = tmp_path / "test_summary.txt"
    assert summary_file.exists()

    # Check content
    content = summary_file.read_text()
    assert "GRÜNEISEN PARAMETER" in content
    assert "STRUCTURE INFORMATION" in content
    assert "PHYSICAL INTERPRETATION" in content
    assert "THERMAL EXPANSION" in content
    assert "Si" in content  # Structure formula


@pytest.mark.skip(reason="Requires matplotlib and full pymatgen phonon objects")
def test_plot_gruneisen_vs_frequency(tmp_path):
    """Test Grüneisen vs frequency plot generation."""
    from atomate2.siesta.jobs.phonon.gruneisen_plotting import (
        plot_gruneisen_vs_frequency,
    )

    gruneisen_doc = get_mock_gruneisen_doc()

    result = plot_gruneisen_vs_frequency.original(
        gruneisen_doc=gruneisen_doc,
        output_dir=tmp_path,
        filename="test_gru_vs_freq.png",
    )

    assert "gruneisen_vs_freq_plot" in result
    if result["gruneisen_vs_freq_plot"] != "not_available":
        plot_file = tmp_path / "test_gru_vs_freq.png"
        assert plot_file.exists()


@pytest.mark.skip(reason="Requires matplotlib and full pymatgen phonon objects")
def test_plot_gruneisen_distribution(tmp_path):
    """Test Grüneisen distribution histogram generation."""
    from atomate2.siesta.jobs.phonon.gruneisen_plotting import (
        plot_gruneisen_distribution,
    )

    gruneisen_doc = get_mock_gruneisen_doc()

    result = plot_gruneisen_distribution.original(
        gruneisen_doc=gruneisen_doc,
        output_dir=tmp_path,
        filename="test_distribution.png",
        bins=30,
    )

    assert "gruneisen_dist_plot" in result
    if result["gruneisen_dist_plot"] != "not_available":
        plot_file = tmp_path / "test_distribution.png"
        assert plot_file.exists()


@pytest.mark.skip(reason="Requires matplotlib")
def test_plot_thermal_expansion(tmp_path):
    """Test thermal expansion plot generation."""
    from atomate2.siesta.jobs.phonon.gruneisen_plotting import (
        calculate_thermal_expansion,
        plot_thermal_expansion,
    )

    gruneisen_doc = get_mock_gruneisen_doc()

    # First calculate thermal expansion
    thermal_data = calculate_thermal_expansion.original(
        gruneisen_doc=gruneisen_doc,
        bulk_modulus=100.0,
        temperature_range=(0, 800),
        n_points=81,
    )

    # Then plot it
    result = plot_thermal_expansion.original(
        thermal_expansion_data=thermal_data,
        output_dir=tmp_path,
        filename="test_thermal_expansion.png",
    )

    assert "thermal_expansion_plot" in result
    plot_file = tmp_path / "test_thermal_expansion.png"
    assert plot_file.exists()


def test_gruneisen_doc_structure():
    """Test that mock Grüneisen document has expected structure."""
    doc = get_mock_gruneisen_doc()

    # Check required keys
    assert "structure" in doc
    assert "code" in doc
    assert "gruneisen_parameter" in doc
    assert "derived_properties" in doc

    # Check derived properties
    derived = doc["derived_properties"]
    assert "average_gruneisen" in derived
    assert "thermal_conductivity_slack" in derived
    assert derived["average_gruneisen"] > 0

    # Check Grüneisen parameter object
    grun_param = doc["gruneisen_parameter"]
    assert hasattr(grun_param, "frequencies")
    assert hasattr(grun_param, "gruneisen")
    assert grun_param.frequencies.shape == grun_param.gruneisen.shape


def test_thermal_expansion_physical_limits():
    """Test that thermal expansion follows physical expectations."""
    from atomate2.siesta.jobs.phonon.gruneisen_plotting import (
        calculate_thermal_expansion,
    )

    gruneisen_doc = get_mock_gruneisen_doc()
    gruneisen_doc["derived_properties"]["average_gruneisen"] = 2.0

    result = calculate_thermal_expansion.original(
        gruneisen_doc=gruneisen_doc,
        bulk_modulus=100.0,
        temperature_range=(0, 1000),
        n_points=101,
    )

    temps = np.array(result["temperatures"])
    alpha_v = np.array(result["alpha_v"])

    # At T=0, thermal expansion should be very small
    assert alpha_v[0] < 1e-10

    # Thermal expansion should increase with temperature (mostly)
    # Allow some noise in low temperature region
    high_temp_mask = temps > 100
    assert np.mean(np.diff(alpha_v[high_temp_mask])) > 0

    # Values should be reasonable (< 1e-4 K^-1 for most materials)
    assert all(a < 1e-3 for a in alpha_v)


def test_debye_temperature_estimation():
    """Test Debye temperature estimation from frequencies."""
    from atomate2.siesta.jobs.phonon.gruneisen_plotting import (
        calculate_thermal_expansion,
    )

    gruneisen_doc = get_mock_gruneisen_doc()

    result = calculate_thermal_expansion.original(
        gruneisen_doc=gruneisen_doc,
        bulk_modulus=100.0,
    )

    # Check Debye temperature is in reasonable range (typically 200-1000 K)
    assert "debye_temperature" in result
    theta_d = result["debye_temperature"]
    assert 50 < theta_d < 2000  # Very broad range for generic test


@pytest.mark.skip(reason="Requires matplotlib and GruneisenPhononBandStructureSymmLine")
def test_plot_gruneisen_band_structure(tmp_path):
    """Test Grüneisen band structure plot generation."""
    from atomate2.siesta.jobs.phonon.gruneisen_plotting import (
        plot_gruneisen_band_structure,
    )

    gruneisen_doc = get_mock_gruneisen_doc()

    # This test requires gruneisen_band_structure to be set
    # Would need a proper GruneisenPhononBandStructureSymmLine object
    result = plot_gruneisen_band_structure.original(
        gruneisen_doc=gruneisen_doc,
        output_dir=tmp_path,
        filename="test_band_structure.png",
    )

    assert "gruneisen_bands_plot" in result
    # Function should handle missing band structure gracefully
    if result["gruneisen_bands_plot"] != "not_available":
        plot_file = tmp_path / "test_band_structure.png"
        assert plot_file.exists()
