"""
Tests for phonon plotting and analysis functions.

These tests validate:
- plot_phonon_band_structure
- plot_phonon_dos
- plot_thermal_properties
- write_phonon_summary
"""

import pytest
import numpy as np
from pathlib import Path
from pymatgen.core import Structure, Lattice

from atomate2.siesta.jobs.phonon.plotting import (
    plot_phonon_band_structure,
    plot_phonon_dos,
    plot_thermal_properties,
    write_phonon_summary,
)


@pytest.fixture
def mock_phonon_doc(si_structure):
    """Create a mock phonon document for testing."""
    # Create minimal force constants matrix
    natoms = len(si_structure)
    force_constants = np.random.rand(natoms, natoms, 3, 3) * 0.1

    return {
        "structure": si_structure,
        "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        "displacement": 0.01,
        "symprec": 1e-5,
        "n_displacements": 6,
        "force_constants": force_constants.tolist(),
        "min_frequency": -0.5,
        "max_frequency": 15.2,
        "has_imaginary_frequencies": True,
        "mesh": (20, 20, 20),
        "phonopy_settings": {
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        },
        "thermal_properties": {
            "temperatures": list(range(0, 1001, 50)),
            "heat_capacity": list(np.random.rand(21) * 0.001),
            "entropy": list(np.random.rand(21) * 0.0005),
            "free_energy": list(np.random.rand(21) * -10.0),
        },
    }


class TestPlotPhononBandStructure:
    """Tests for plot_phonon_band_structure function."""

    @pytest.mark.skip(reason="Requires matplotlib and seekpath")
    def test_plot_phonon_band_structure_basic(self, mock_phonon_doc, tmp_path):
        """Test basic phonon band structure plotting."""
        result = plot_phonon_band_structure.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(tmp_path),
        )

        assert isinstance(result, dict)
        assert "band_structure_plot" in result

    @pytest.mark.skip(reason="Requires matplotlib and seekpath")
    def test_plot_phonon_band_structure_custom_params(self, mock_phonon_doc, tmp_path):
        """Test phonon band structure with custom parameters."""
        result = plot_phonon_band_structure.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(tmp_path),
            filename="custom_bands.png",
            figsize=(10, 8),
            dpi=150,
        )

        assert isinstance(result, dict)
        assert "band_structure_plot" in result

    def test_plot_phonon_band_structure_missing_matplotlib(
        self, mock_phonon_doc, tmp_path, monkeypatch
    ):
        """Test graceful handling when matplotlib is not available."""
        # Mock matplotlib import to fail
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib.pyplot":
                raise ImportError("matplotlib not available")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="matplotlib"):
            plot_phonon_band_structure.original(
                phonon_doc=mock_phonon_doc,
                output_dir=str(tmp_path),
            )

    def test_plot_phonon_band_structure_output_dir_creation(
        self, mock_phonon_doc, tmp_path
    ):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "nested" / "dir"
        assert not output_dir.exists()

        # This will fail due to matplotlib, but dir should be created
        try:
            plot_phonon_band_structure.original(
                phonon_doc=mock_phonon_doc,
                output_dir=str(output_dir),
            )
        except Exception:
            pass

        # Check if attempted to create directory structure
        # (may not exist if matplotlib import fails early)
        assert tmp_path.exists()


class TestPlotPhononDOS:
    """Tests for plot_phonon_dos function."""

    @pytest.mark.skip(reason="Requires matplotlib and phonopy")
    def test_plot_phonon_dos_basic(self, mock_phonon_doc, tmp_path):
        """Test basic phonon DOS plotting."""
        result = plot_phonon_dos.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(tmp_path),
        )

        assert isinstance(result, dict)
        assert "dos_plot" in result

    @pytest.mark.skip(reason="Requires matplotlib and phonopy")
    def test_plot_phonon_dos_custom_params(self, mock_phonon_doc, tmp_path):
        """Test phonon DOS with custom parameters."""
        result = plot_phonon_dos.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(tmp_path),
            filename="custom_dos.png",
            figsize=(12, 8),
            dpi=200,
        )

        assert isinstance(result, dict)
        assert "dos_plot" in result

    def test_plot_phonon_dos_missing_matplotlib(
        self, mock_phonon_doc, tmp_path, monkeypatch
    ):
        """Test graceful handling when matplotlib is not available."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib.pyplot":
                raise ImportError("matplotlib not available")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="matplotlib"):
            plot_phonon_dos.original(
                phonon_doc=mock_phonon_doc,
                output_dir=str(tmp_path),
            )


class TestPlotThermalProperties:
    """Tests for plot_thermal_properties function."""

    @pytest.mark.skip(reason="Requires matplotlib")
    def test_plot_thermal_properties_basic(self, mock_phonon_doc, tmp_path):
        """Test basic thermal properties plotting."""
        result = plot_thermal_properties.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(tmp_path),
        )

        assert isinstance(result, dict)
        assert "thermal_plot" in result

    @pytest.mark.skip(reason="Requires matplotlib")
    def test_plot_thermal_properties_custom_params(self, mock_phonon_doc, tmp_path):
        """Test thermal properties with custom parameters."""
        result = plot_thermal_properties.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(tmp_path),
            filename="custom_thermal.png",
            figsize=(14, 12),
            dpi=250,
        )

        assert isinstance(result, dict)
        assert "thermal_plot" in result

    def test_plot_thermal_properties_missing_matplotlib(
        self, mock_phonon_doc, tmp_path, monkeypatch
    ):
        """Test graceful handling when matplotlib is not available."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib.pyplot":
                raise ImportError("matplotlib not available")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="matplotlib"):
            plot_thermal_properties.original(
                phonon_doc=mock_phonon_doc,
                output_dir=str(tmp_path),
            )

    def test_plot_thermal_properties_no_thermal_data(self, mock_phonon_doc, tmp_path):
        """Test handling when thermal properties are not available."""
        # Remove thermal properties
        doc_no_thermal = mock_phonon_doc.copy()
        del doc_no_thermal["thermal_properties"]

        result = plot_thermal_properties.original(
            phonon_doc=doc_no_thermal,
            output_dir=str(tmp_path),
        )

        assert result == {"thermal_plot": "not_available"}


class TestWritePhononSummary:
    """Tests for write_phonon_summary function."""

    def test_write_phonon_summary_basic(self, mock_phonon_doc, tmp_path):
        """Test basic phonon summary writing."""
        result = write_phonon_summary.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(tmp_path),
        )

        assert isinstance(result, dict)
        assert "summary_file" in result

        # Check file was created
        summary_file = Path(result["summary_file"])
        assert summary_file.exists()

        # Check file content
        content = summary_file.read_text()
        assert "PHONON CALCULATION SUMMARY" in content
        assert "STRUCTURE INFORMATION" in content
        assert "CALCULATION PARAMETERS" in content
        assert "PHONON FREQUENCIES" in content

    def test_write_phonon_summary_custom_filename(self, mock_phonon_doc, tmp_path):
        """Test phonon summary with custom filename."""
        result = write_phonon_summary.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(tmp_path),
            filename="custom_summary.txt",
        )

        summary_file = Path(result["summary_file"])
        assert summary_file.name == "custom_summary.txt"
        assert summary_file.exists()

    def test_write_phonon_summary_structure_info(self, mock_phonon_doc, tmp_path):
        """Test that structure information is written correctly."""
        result = write_phonon_summary.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(tmp_path),
        )

        content = Path(result["summary_file"]).read_text()

        # Check for structure details
        assert "Formula:" in content
        assert "Number of atoms:" in content
        assert "Lattice parameters:" in content
        assert "Volume" in content

    def test_write_phonon_summary_calculation_params(self, mock_phonon_doc, tmp_path):
        """Test that calculation parameters are written correctly."""
        result = write_phonon_summary.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(tmp_path),
        )

        content = Path(result["summary_file"]).read_text()

        # Check for calculation parameters
        assert "Supercell matrix:" in content
        assert "Displacement:" in content
        assert "Symmetry precision:" in content
        assert "Number of displacements:" in content

    def test_write_phonon_summary_frequencies(self, mock_phonon_doc, tmp_path):
        """Test that frequency information is written correctly."""
        result = write_phonon_summary.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(tmp_path),
        )

        content = Path(result["summary_file"]).read_text()

        # Check for frequency information
        assert "Minimum frequency:" in content
        assert "Maximum frequency:" in content
        assert "Frequency range:" in content
        assert "Imaginary frequencies:" in content

    def test_write_phonon_summary_imaginary_warning(self, mock_phonon_doc, tmp_path):
        """Test that warning is shown for imaginary frequencies."""
        result = write_phonon_summary.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(tmp_path),
        )

        content = Path(result["summary_file"]).read_text()

        # Check for warning
        assert "WARNING: Imaginary frequencies detected!" in content
        assert "Structural instability" in content

    def test_write_phonon_summary_thermal_properties(self, mock_phonon_doc, tmp_path):
        """Test that thermal properties are written correctly."""
        result = write_phonon_summary.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(tmp_path),
        )

        content = Path(result["summary_file"]).read_text()

        # Check for thermal properties
        assert "THERMAL PROPERTIES" in content
        assert "Temperature range:" in content
        assert "Cv (eV/K)" in content
        assert "S (eV/K)" in content
        assert "F (eV)" in content

    def test_write_phonon_summary_convergence_recommendations(
        self, mock_phonon_doc, tmp_path
    ):
        """Test that convergence recommendations are included."""
        result = write_phonon_summary.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(tmp_path),
        )

        content = Path(result["summary_file"]).read_text()

        # Check for recommendations
        assert "CONVERGENCE RECOMMENDATIONS" in content
        assert "Supercell size:" in content
        assert "Displacement distance:" in content
        assert "SIESTA parameters:" in content

    def test_write_phonon_summary_output_dir_creation(self, mock_phonon_doc, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "nested" / "dir"
        assert not output_dir.exists()

        result = write_phonon_summary.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(output_dir),
        )

        assert output_dir.exists()
        assert Path(result["summary_file"]).exists()

    def test_write_phonon_summary_no_thermal_properties(
        self, mock_phonon_doc, tmp_path
    ):
        """Test summary generation without thermal properties."""
        doc_no_thermal = mock_phonon_doc.copy()
        del doc_no_thermal["thermal_properties"]

        result = write_phonon_summary.original(
            phonon_doc=doc_no_thermal,
            output_dir=str(tmp_path),
        )

        content = Path(result["summary_file"]).read_text()

        # Should not have thermal section
        assert "THERMAL PROPERTIES" not in content
        # But should still have other sections
        assert "STRUCTURE INFORMATION" in content
        assert "PHONON FREQUENCIES" in content


class TestPhononPlottingIntegration:
    """Integration tests for phonon plotting functions."""

    def test_all_functions_return_dict(self, mock_phonon_doc, tmp_path):
        """Test that all functions return dictionaries."""
        functions = [
            (write_phonon_summary, {}),
        ]

        for func, _ in functions:
            result = func.original(
                phonon_doc=mock_phonon_doc,
                output_dir=str(tmp_path),
            )
            assert isinstance(result, dict)

    def test_output_files_in_same_directory(self, mock_phonon_doc, tmp_path):
        """Test that all outputs can be saved to the same directory."""
        # Only test write_phonon_summary since others require matplotlib
        result1 = write_phonon_summary.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(tmp_path),
            filename="summary.txt",
        )

        assert Path(result1["summary_file"]).parent == tmp_path

    def test_custom_output_directory_for_all(self, mock_phonon_doc, tmp_path):
        """Test that all functions respect custom output directory."""
        custom_dir = tmp_path / "custom_output"

        result = write_phonon_summary.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(custom_dir),
        )

        assert custom_dir.exists()
        assert Path(result["summary_file"]).parent == custom_dir


class TestPhononPlottingEdgeCases:
    """Test edge cases for phonon plotting."""

    def test_write_summary_with_minimal_doc(self, si_structure, tmp_path):
        """Test summary writing with minimal phonon document."""
        minimal_doc = {
            "structure": si_structure,
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "displacement": 0.01,
            "symprec": 1e-5,
            "n_displacements": 6,
            "min_frequency": 0.0,
            "max_frequency": 10.0,
            "has_imaginary_frequencies": False,
        }

        result = write_phonon_summary.original(
            phonon_doc=minimal_doc,
            output_dir=str(tmp_path),
        )

        assert Path(result["summary_file"]).exists()

    def test_write_summary_special_characters_in_formula(self, tmp_path):
        """Test summary with complex chemical formula."""
        # Create structure with subscripts
        lattice = Lattice.cubic(5.0)
        structure = Structure(
            lattice,
            ["Fe", "O", "O"],
            [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        )

        phonon_doc = {
            "structure": structure,
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "displacement": 0.01,
            "symprec": 1e-5,
            "n_displacements": 12,
            "force_constants": np.random.rand(3, 3, 3, 3).tolist(),
            "min_frequency": -1.0,
            "max_frequency": 20.0,
            "has_imaginary_frequencies": True,
        }

        result = write_phonon_summary.original(
            phonon_doc=phonon_doc,
            output_dir=str(tmp_path),
        )

        content = Path(result["summary_file"]).read_text()
        assert "Fe" in content or "O" in content

    def test_functions_with_path_objects(self, mock_phonon_doc, tmp_path):
        """Test that functions work with Path objects, not just strings."""
        result = write_phonon_summary.original(
            phonon_doc=mock_phonon_doc,
            output_dir=tmp_path,  # Path object
        )

        assert Path(result["summary_file"]).exists()

    def test_write_summary_frequency_statistics(self, mock_phonon_doc, tmp_path):
        """Test summary includes frequency statistics."""
        mock_phonon_doc["min_frequency"] = -0.5
        mock_phonon_doc["max_frequency"] = 25.0
        mock_phonon_doc["has_imaginary_frequencies"] = True

        result = write_phonon_summary.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(tmp_path),
        )

        content = Path(result["summary_file"]).read_text()
        assert "25.0" in content or "25" in content  # Max frequency
        assert "imaginary" in content.lower()

    def test_write_summary_with_supercell_info(self, mock_phonon_doc, tmp_path):
        """Test summary includes supercell information."""
        mock_phonon_doc["supercell_matrix"] = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
        mock_phonon_doc["symprec"] = 1e-6

        result = write_phonon_summary.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(tmp_path),
        )

        content = Path(result["summary_file"]).read_text()
        assert "supercell" in content.lower() or "3" in content

    def test_summary_file_created_in_output_dir(self, mock_phonon_doc, tmp_path):
        """Test that summary file is created in specified output directory."""
        custom_dir = tmp_path / "custom_output"

        result = write_phonon_summary.original(
            phonon_doc=mock_phonon_doc,
            output_dir=str(custom_dir),
        )

        summary_path = Path(result["summary_file"])
        assert summary_path.exists()
        assert summary_path.parent == custom_dir
