"""Tests for plot_pseudopotential CLI module."""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from click.testing import CliRunner

from atomate2.siesta.cli.pseudo.plot_pseudopotential import (
    parse_psml,
    plot_wavefunctions,
    plot_potentials,
    plot_3d_potential,
    plot_occupation_map,
    plot_density,
    plot_pseudopotential,
)


@pytest.fixture
def runner():
    """Create Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def minimal_psml_xml():
    """Create minimal valid PSML XML content."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<psml:psml xmlns:psml="http://esl.cecam.org/PSML/ns/1.1" version="1.1">
    <psml:pseudo-atom-spec atomic-label="Si"/>
    <psml:grid>
        <psml:grid-data>
            0.0 0.5 1.0 1.5 2.0 2.5 3.0
        </psml:grid-data>
    </psml:grid>
    <psml:valence-configuration>
        <psml:shell n="3" l="s" occupation="2.0"/>
        <psml:shell n="3" l="p" occupation="2.0"/>
    </psml:valence-configuration>
    <psml:nonlocal-projectors>
        <psml:proj l="s" seq="1">
            <psml:radfunc>
                <psml:data>0.0 0.1 0.2 0.3 0.4 0.5 0.6</psml:data>
            </psml:radfunc>
        </psml:proj>
        <psml:proj l="p" seq="1">
            <psml:radfunc>
                <psml:data>0.0 0.15 0.25 0.35 0.45 0.55 0.65</psml:data>
            </psml:radfunc>
        </psml:proj>
    </psml:nonlocal-projectors>
    <psml:local-potential>
        <psml:radfunc>
            <psml:data>-1.0 -0.8 -0.6 -0.4 -0.2 -0.1 0.0</psml:data>
        </psml:radfunc>
    </psml:local-potential>
</psml:psml>"""


@pytest.fixture
def psml_with_semilocal():
    """Create PSML XML with semilocal potentials."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<psml:psml xmlns:psml="http://esl.cecam.org/PSML/ns/1.1" version="1.1">
    <psml:pseudo-atom-spec atomic-label="Fe"/>
    <psml:grid>
        <psml:grid-data>
            0.0 1.0 2.0 3.0 4.0
        </psml:grid-data>
    </psml:grid>
    <psml:valence-configuration>
        <psml:shell n="3" l="d" occupation="6.0"/>
        <psml:shell n="4" l="s" occupation="2.0"/>
    </psml:valence-configuration>
    <psml:nonlocal-projectors>
        <psml:proj l="d" seq="1">
            <psml:radfunc>
                <psml:data>0.0 0.2 0.4 0.6 0.8</psml:data>
            </psml:radfunc>
        </psml:proj>
    </psml:nonlocal-projectors>
    <psml:local-potential>
        <psml:radfunc>
            <psml:data>-2.0 -1.5 -1.0 -0.5 0.0</psml:data>
        </psml:radfunc>
    </psml:local-potential>
    <psml:semilocal-potentials>
        <psml:slps l="s" n="4">
            <psml:radfunc>
                <psml:data>-1.5 -1.0 -0.5 -0.2 0.0</psml:data>
            </psml:radfunc>
        </psml:slps>
        <psml:slps l="d" n="3">
            <psml:radfunc>
                <psml:data>-1.8 -1.2 -0.6 -0.3 0.0</psml:data>
            </psml:radfunc>
        </psml:slps>
    </psml:semilocal-potentials>
</psml:psml>"""


@pytest.fixture
def invalid_psml_xml():
    """Create invalid PSML XML (not well-formed)."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<psml:psml xmlns:psml="http://esl.cecam.org/PSML/ns/1.1" version="1.1">
    <psml:pseudo-atom-spec atomic-label="Si"
    <!-- Missing closing tag -->
</psml:psml>"""


@pytest.fixture
def psml_missing_grid():
    """Create PSML XML without grid element."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<psml:psml xmlns:psml="http://esl.cecam.org/PSML/ns/1.1" version="1.1">
    <psml:pseudo-atom-spec atomic-label="Si"/>
    <psml:valence-configuration>
        <psml:shell n="3" l="s" occupation="2.0"/>
    </psml:valence-configuration>
</psml:psml>"""


class TestParsePsml:
    """Test parse_psml function."""

    def test_parse_minimal_psml(self, tmp_path, minimal_psml_xml):
        """Test parsing minimal valid PSML file."""
        psml_file = tmp_path / "Si.psml"
        psml_file.write_text(minimal_psml_xml)

        (
            radial_grid,
            valence_config,
            wavefunctions,
            potentials,
            element_name,
        ) = parse_psml(str(psml_file))

        assert len(radial_grid) == 7
        assert element_name == "Si"
        assert len(valence_config) == 2
        assert valence_config[0]["n"] == 3
        assert valence_config[0]["l"] == 0  # s = 0
        assert valence_config[1]["l"] == 1  # p = 1
        assert len(wavefunctions) == 2
        assert len(potentials) >= 1  # At least local potential

    def test_parse_psml_with_semilocal(self, tmp_path, psml_with_semilocal):
        """Test parsing PSML with semilocal potentials."""
        psml_file = tmp_path / "Fe.psml"
        psml_file.write_text(psml_with_semilocal)

        (
            radial_grid,
            valence_config,
            wavefunctions,
            potentials,
            element_name,
        ) = parse_psml(str(psml_file))

        assert element_name == "Fe"
        assert len(radial_grid) == 5
        assert len(valence_config) == 2
        # Should have local + 2 semilocal potentials
        assert len(potentials) == 3

    def test_parse_invalid_xml(self, tmp_path, invalid_psml_xml):
        """Test parsing invalid XML raises ValueError."""
        psml_file = tmp_path / "invalid.psml"
        psml_file.write_text(invalid_psml_xml)

        with pytest.raises(ValueError, match="Failed to parse PSML file as XML"):
            parse_psml(str(psml_file))

    def test_parse_missing_grid(self, tmp_path, psml_missing_grid):
        """Test parsing PSML without grid raises ValueError."""
        psml_file = tmp_path / "no_grid.psml"
        psml_file.write_text(psml_missing_grid)

        with pytest.raises(ValueError, match="No <grid> element found"):
            parse_psml(str(psml_file))

    def test_parse_missing_valence_config(self, tmp_path):
        """Test parsing PSML without valence configuration."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<psml:psml xmlns:psml="http://esl.cecam.org/PSML/ns/1.1" version="1.1">
    <psml:pseudo-atom-spec atomic-label="H"/>
    <psml:grid>
        <psml:grid-data>0.0 1.0 2.0</psml:grid-data>
    </psml:grid>
    <psml:local-potential>
        <psml:radfunc>
            <psml:data>-1.0 -0.5 0.0</psml:data>
        </psml:radfunc>
    </psml:local-potential>
</psml:psml>"""
        psml_file = tmp_path / "H.psml"
        psml_file.write_text(xml)

        (
            radial_grid,
            valence_config,
            wavefunctions,
            potentials,
            element_name,
        ) = parse_psml(str(psml_file))

        assert len(valence_config) == 0  # No valence configuration
        assert len(potentials) >= 1  # Still has local potential


@patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.plt")
class TestPlottingFunctions:
    """Test plotting functions with mocked matplotlib."""

    def test_plot_wavefunctions(self, mock_plt):
        """Test plot_wavefunctions function."""
        radial_grid = np.array([0.0, 1.0, 2.0, 3.0])
        wavefunctions = [
            {"n": 3, "l": 0, "data": np.array([0.0, 0.1, 0.2, 0.3])},
            {"n": 3, "l": 1, "data": np.array([0.0, 0.15, 0.25, 0.35])},
        ]

        plot_wavefunctions(radial_grid, wavefunctions, "test_wf.png", "Si", r_max=2.5)

        # Verify matplotlib methods were called
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called()
        mock_plt.savefig.assert_called_once_with("test_wf.png", dpi=300)
        mock_plt.close.assert_called_once()

    def test_plot_potentials(self, mock_plt):
        """Test plot_potentials function."""
        radial_grid = np.array([0.0, 1.0, 2.0, 3.0])
        potentials = [
            {"l": None, "data": np.array([-1.0, -0.5, -0.2, 0.0])},
            {"n": 3, "l": 0, "data": np.array([-0.8, -0.4, -0.1, 0.0])},
        ]

        # Mock the subplot creation
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        plot_potentials(radial_grid, potentials, "test_pot.png", "Si")

        mock_plt.figure.assert_called_once()
        mock_fig.add_subplot.assert_called_once()
        mock_plt.savefig.assert_called_once_with("test_pot.png", dpi=300)

    def test_plot_3d_potential(self, mock_plt):
        """Test plot_3d_potential function."""
        radial_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        potentials = [
            {"l": None, "data": np.array([-2.0, -1.5, -1.0, -0.5, 0.0])},
        ]

        # Mock 3D plotting
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        plot_3d_potential(radial_grid, potentials, "test_3d.png", "Si", r_max=3.0)

        mock_plt.figure.assert_called_once()
        mock_plt.savefig.assert_called_once_with("test_3d.png", dpi=300)

    def test_plot_occupation_map(self, mock_plt):
        """Test plot_occupation_map function."""
        valence_config = [
            {"n": 3, "l": 0, "occupation": 2.0},
            {"n": 3, "l": 1, "occupation": 2.0},
        ]

        # Mock seaborn heatmap
        with patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.sns") as mock_sns:
            plot_occupation_map(valence_config, "test_occ.png", "Si")

            mock_plt.figure.assert_called_once()
            mock_sns.heatmap.assert_called_once()
            mock_plt.savefig.assert_called_once_with("test_occ.png", dpi=300)

    def test_plot_density(self, mock_plt):
        """Test plot_density function."""
        radial_grid = np.array([0.0, 1.0, 2.0, 3.0])
        wavefunctions = [
            {"n": 3, "l": 0, "data": np.array([0.0, 0.1, 0.2, 0.3])},
        ]

        plot_density(radial_grid, wavefunctions, "test_dens.png", "Si", r_max=2.5)

        mock_plt.figure.assert_called_once()
        mock_plt.fill_between.assert_called()
        mock_plt.savefig.assert_called_once_with("test_dens.png", dpi=300)


class TestPlotPseudopotentialCLI:
    """Test plot_pseudopotential CLI command."""

    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.plot_wavefunctions")
    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.parse_psml")
    def test_cli_plot_wavefunctions_only(
        self, mock_parse, mock_plot_wf, runner, tmp_path
    ):
        """Test CLI with wavefunctions plot type."""
        # Create dummy PSML file
        psml_file = tmp_path / "Si.psml"
        psml_file.write_text("dummy content")

        # Mock parse_psml return
        mock_parse.return_value = (
            np.array([0.0, 1.0, 2.0]),  # radial_grid
            [],  # valence_config
            [{"n": 3, "l": 0, "data": np.array([0.0, 0.1, 0.2])}],  # wavefunctions
            [{"l": None, "data": np.array([-1.0, -0.5, 0.0])}],  # potentials
            "Si",  # element_name
        )

        result = runner.invoke(
            plot_pseudopotential,
            [
                str(psml_file),
                "--plot-type",
                "wavefunctions",
                "--output-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0
        mock_parse.assert_called_once()
        mock_plot_wf.assert_called_once()

    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.plot_potentials")
    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.parse_psml")
    def test_cli_plot_potentials_only(
        self, mock_parse, mock_plot_pot, runner, tmp_path
    ):
        """Test CLI with potentials plot type."""
        psml_file = tmp_path / "Si.psml"
        psml_file.write_text("dummy content")

        mock_parse.return_value = (
            np.array([0.0, 1.0, 2.0]),
            [],
            [],
            [{"l": None, "data": np.array([-1.0, -0.5, 0.0])}],
            "Si",
        )

        result = runner.invoke(
            plot_pseudopotential,
            [str(psml_file), "--plot-type", "potentials"],
        )

        assert result.exit_code == 0
        mock_plot_pot.assert_called_once()

    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.plot_3d_potential")
    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.parse_psml")
    def test_cli_plot_3d_potential_only(
        self, mock_parse, mock_plot_3d, runner, tmp_path
    ):
        """Test CLI with 3d-potential plot type."""
        psml_file = tmp_path / "Si.psml"
        psml_file.write_text("dummy content")

        mock_parse.return_value = (
            np.array([0.0, 1.0, 2.0]),
            [],
            [],
            [{"l": None, "data": np.array([-1.0, -0.5, 0.0])}],
            "Si",
        )

        result = runner.invoke(
            plot_pseudopotential,
            [str(psml_file), "--plot-type", "3d-potential"],
        )

        assert result.exit_code == 0
        mock_plot_3d.assert_called_once()

    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.plot_occupation_map")
    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.parse_psml")
    def test_cli_plot_occupation_only(
        self, mock_parse, mock_plot_occ, runner, tmp_path
    ):
        """Test CLI with occupation plot type."""
        psml_file = tmp_path / "Si.psml"
        psml_file.write_text("dummy content")

        mock_parse.return_value = (
            np.array([0.0, 1.0, 2.0]),
            [{"n": 3, "l": 0, "occupation": 2.0}],
            [],
            [],
            "Si",
        )

        result = runner.invoke(
            plot_pseudopotential,
            [str(psml_file), "--plot-type", "occupation"],
        )

        assert result.exit_code == 0
        mock_plot_occ.assert_called_once()

    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.plot_density")
    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.parse_psml")
    def test_cli_plot_density_only(self, mock_parse, mock_plot_dens, runner, tmp_path):
        """Test CLI with density plot type."""
        psml_file = tmp_path / "Si.psml"
        psml_file.write_text("dummy content")

        mock_parse.return_value = (
            np.array([0.0, 1.0, 2.0]),
            [],
            [{"n": 3, "l": 0, "data": np.array([0.0, 0.1, 0.2])}],
            [],
            "Si",
        )

        result = runner.invoke(
            plot_pseudopotential,
            [str(psml_file), "--plot-type", "density"],
        )

        assert result.exit_code == 0
        mock_plot_dens.assert_called_once()

    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.plot_wavefunctions")
    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.plot_potentials")
    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.plot_3d_potential")
    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.plot_occupation_map")
    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.plot_density")
    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.parse_psml")
    def test_cli_plot_all(
        self,
        mock_parse,
        mock_dens,
        mock_occ,
        mock_3d,
        mock_pot,
        mock_wf,
        runner,
        tmp_path,
    ):
        """Test CLI with 'all' plot type."""
        psml_file = tmp_path / "Si.psml"
        psml_file.write_text("dummy content")

        mock_parse.return_value = (
            np.array([0.0, 1.0, 2.0]),
            [{"n": 3, "l": 0, "occupation": 2.0}],
            [{"n": 3, "l": 0, "data": np.array([0.0, 0.1, 0.2])}],
            [{"l": None, "data": np.array([-1.0, -0.5, 0.0])}],
            "Si",
        )

        result = runner.invoke(
            plot_pseudopotential,
            [str(psml_file), "--plot-type", "all"],
        )

        assert result.exit_code == 0
        # All plotting functions should be called
        mock_wf.assert_called_once()
        mock_pot.assert_called_once()
        mock_3d.assert_called_once()
        mock_occ.assert_called_once()
        mock_dens.assert_called_once()

    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.parse_psml")
    def test_cli_with_r_plot_option(self, mock_parse, runner, tmp_path):
        """Test CLI with --r-plot option."""
        psml_file = tmp_path / "Si.psml"
        psml_file.write_text("dummy content")

        mock_parse.return_value = (
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            [],
            [{"n": 3, "l": 0, "data": np.array([0.0, 0.1, 0.2, 0.3, 0.4])}],
            [{"l": None, "data": np.array([-2.0, -1.5, -1.0, -0.5, 0.0])}],
            "Si",
        )

        with patch(
            "atomate2.siesta.cli.pseudo.plot_pseudopotential.plot_wavefunctions"
        ) as mock_wf:
            result = runner.invoke(
                plot_pseudopotential,
                [str(psml_file), "--plot-type", "wavefunctions", "--r-plot", "2.5"],
            )

            assert result.exit_code == 0
            # Check that r_max was passed to plotting function
            call_args = mock_wf.call_args
            assert call_args[1]["r_max"] == 2.5

    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.parse_psml")
    def test_cli_missing_wavefunctions_warning(self, mock_parse, runner, tmp_path):
        """Test CLI warns when wavefunctions are missing."""
        psml_file = tmp_path / "Si.psml"
        psml_file.write_text("dummy content")

        # Return empty wavefunctions
        mock_parse.return_value = (
            np.array([0.0, 1.0, 2.0]),
            [],
            [],  # No wavefunctions
            [{"l": None, "data": np.array([-1.0, -0.5, 0.0])}],
            "Si",
        )

        result = runner.invoke(
            plot_pseudopotential,
            [str(psml_file), "--plot-type", "wavefunctions"],
        )

        assert result.exit_code == 0
        assert "No wavefunctions" in result.output

    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.parse_psml")
    def test_cli_missing_potentials_warning(self, mock_parse, runner, tmp_path):
        """Test CLI warns when potentials are missing."""
        psml_file = tmp_path / "Si.psml"
        psml_file.write_text("dummy content")

        # Return empty potentials
        mock_parse.return_value = (
            np.array([0.0, 1.0, 2.0]),
            [],
            [{"n": 3, "l": 0, "data": np.array([0.0, 0.1, 0.2])}],
            [],  # No potentials
            "Si",
        )

        result = runner.invoke(
            plot_pseudopotential,
            [str(psml_file), "--plot-type", "potentials"],
        )

        assert result.exit_code == 0
        assert "No potentials found" in result.output

    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.parse_psml")
    def test_cli_missing_valence_config_warning(self, mock_parse, runner, tmp_path):
        """Test CLI warns when valence config is missing."""
        psml_file = tmp_path / "Si.psml"
        psml_file.write_text("dummy content")

        # Return empty valence_config
        mock_parse.return_value = (
            np.array([0.0, 1.0, 2.0]),
            [],  # No valence config
            [],
            [],
            "Si",
        )

        result = runner.invoke(
            plot_pseudopotential,
            [str(psml_file), "--plot-type", "occupation"],
        )

        assert result.exit_code == 0
        assert "No valence configuration found" in result.output

    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.parse_psml")
    def test_cli_parse_error(self, mock_parse, runner, tmp_path):
        """Test CLI handles parse errors."""
        psml_file = tmp_path / "bad.psml"
        psml_file.write_text("bad content")

        mock_parse.side_effect = ValueError("Failed to parse PSML file")

        result = runner.invoke(
            plot_pseudopotential,
            [str(psml_file)],
        )

        assert result.exit_code != 0
        assert "Error" in result.output

    @patch("atomate2.siesta.cli.pseudo.plot_pseudopotential.parse_psml")
    def test_cli_empty_radial_grid_error(self, mock_parse, runner, tmp_path):
        """Test CLI handles empty radial grid."""
        psml_file = tmp_path / "Si.psml"
        psml_file.write_text("dummy content")

        # Return empty radial grid
        mock_parse.return_value = (
            np.array([]),  # Empty radial grid
            [],
            [],
            [],
            "Si",
        )

        result = runner.invoke(
            plot_pseudopotential,
            [str(psml_file)],
        )

        assert result.exit_code != 0
        assert "No radial grid data found" in result.output

    def test_cli_nonexistent_file(self, runner):
        """Test CLI with nonexistent file."""
        result = runner.invoke(
            plot_pseudopotential,
            ["/nonexistent/file.psml"],
        )

        # Click should handle this with proper error
        assert result.exit_code != 0
