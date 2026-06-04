"""Tests for siesta_pseudos CLI module."""

from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from atomate2.siesta.cli.pseudo.siesta_pseudos import (
    cli,
    get_local_pseudo_path,
    download_and_extract_pseudo,
    PSEUDOS,
)


@pytest.fixture
def runner():
    """Create Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_pseudo_dir(tmp_path, monkeypatch):
    """Create a temporary pseudo directory."""
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    monkeypatch.setattr(
        "atomate2.siesta.cli.pseudo.siesta_pseudos.PSEUDO_DIR", str(pseudo_dir)
    )
    return pseudo_dir


class TestGetLocalPseudoPath:
    """Test get_local_pseudo_path function."""

    def test_get_local_pseudo_path_exists(self, tmp_path, monkeypatch):
        """Test finding an existing local pseudo file."""
        # Mock the script directory path
        mock_script_dir = tmp_path / "src" / "atomate2" / "siesta" / "cli"
        mock_script_dir.mkdir(parents=True)

        # Create pseudos directory at project root
        project_dir = tmp_path
        pseudos_dir = project_dir / "pseudos"
        pseudos_dir.mkdir()

        # Create a pseudo file (filename comes from the PSEUDOS entry's local_path)
        pseudo_file = (
            pseudos_dir / PSEUDOS["ONCVPSP-PBEsol-FR-PDv0.4-Standard"]["local_path"]
        )
        pseudo_file.write_text("dummy content")

        # Patch __file__ to return our mock path
        with patch(
            "atomate2.siesta.cli.pseudo.siesta_pseudos.__file__",
            str(mock_script_dir / "siesta_pseudos.py"),
        ):
            result = get_local_pseudo_path("ONCVPSP-PBEsol-FR-PDv0.4-Standard")

        assert result == str(pseudo_file)

    def test_get_local_pseudo_path_not_found(self, tmp_path, monkeypatch):
        """Test when local pseudo file doesn't exist."""
        # Mock the script directory path
        mock_script_dir = tmp_path / "src" / "atomate2" / "siesta" / "cli"
        mock_script_dir.mkdir(parents=True)

        # Don't create pseudos directory - should return None
        with patch(
            "atomate2.siesta.cli.pseudo.siesta_pseudos.__file__",
            str(mock_script_dir / "siesta_pseudos.py"),
        ):
            result = get_local_pseudo_path("ONCVPSP-PBEsol-FR-PDv0.4-Standard")

        assert result is None


class TestDownloadAndExtractPseudo:
    """Test download_and_extract_pseudo function."""

    @patch("atomate2.siesta.cli.pseudo.siesta_pseudos.tarfile.open")
    @patch("atomate2.siesta.cli.pseudo.siesta_pseudos.shutil.copy")
    @patch("atomate2.siesta.cli.pseudo.siesta_pseudos.get_local_pseudo_path")
    def test_download_with_local_file(
        self, mock_get_local, mock_copy, mock_tarfile, mock_pseudo_dir
    ):
        """Test using local file when available."""
        # Mock local pseudo path exists
        local_file = "/fake/local/path/pseudo.tgz"
        mock_get_local.return_value = local_file

        # Mock tarfile extraction
        mock_tar = MagicMock()
        mock_tarfile.return_value.__enter__.return_value = mock_tar
        mock_tar.getmembers.return_value = []

        download_and_extract_pseudo(
            "nc-fr-04_pbe_standard_psml.tgz",
            "ONCVPSP-PBEsol-FR-PDv0.4-Standard",
            local_only=False,
        )

        # Verify local file was used
        mock_get_local.assert_called_once()
        mock_copy.assert_called_once()

    @patch("atomate2.siesta.cli.pseudo.siesta_pseudos.get_local_pseudo_path")
    def test_download_local_only_not_found(
        self, mock_get_local, mock_pseudo_dir, capsys
    ):
        """Test local-only mode when local file doesn't exist."""
        # Mock local pseudo path doesn't exist
        mock_get_local.return_value = None

        download_and_extract_pseudo(
            "nc-fr-04_pbe_standard_psml.tgz",
            "ONCVPSP-PBEsol-FR-PDv0.4-Standard",
            local_only=True,
        )

        # Function should return early without downloading
        mock_get_local.assert_called_once()

    @patch("atomate2.siesta.cli.pseudo.siesta_pseudos.tarfile.open")
    @patch("atomate2.siesta.cli.pseudo.siesta_pseudos.requests.get")
    @patch("atomate2.siesta.cli.pseudo.siesta_pseudos.get_local_pseudo_path")
    def test_download_from_url(
        self, mock_get_local, mock_requests, mock_tarfile, mock_pseudo_dir
    ):
        """Test downloading from URL when local file not found."""
        # Mock local pseudo path doesn't exist
        mock_get_local.return_value = None

        # Mock successful HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake tarball content"
        mock_requests.return_value = mock_response

        # Mock tarfile extraction
        mock_tar = MagicMock()
        mock_tarfile.return_value.__enter__.return_value = mock_tar
        mock_tar.getmembers.return_value = []

        download_and_extract_pseudo(
            "nc-fr-04_pbe_standard_psml.tgz",
            "ONCVPSP-PBEsol-FR-PDv0.4-Standard",
            local_only=False,
        )

        # Verify HTTP request was made
        mock_requests.assert_called_once()

    @patch("atomate2.siesta.cli.pseudo.siesta_pseudos.requests.get")
    @patch("atomate2.siesta.cli.pseudo.siesta_pseudos.get_local_pseudo_path")
    def test_download_http_failure(
        self, mock_get_local, mock_requests, mock_pseudo_dir
    ):
        """Test handling HTTP download failure."""
        # Mock local pseudo path doesn't exist
        mock_get_local.return_value = None

        # Mock failed HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_requests.return_value = mock_response

        download_and_extract_pseudo(
            "nc-fr-04_pbe_standard_psml.tgz",
            "ONCVPSP-PBEsol-FR-PDv0.4-Standard",
            local_only=False,
        )

        # Function should handle error and return
        mock_requests.assert_called_once()


class TestAvailableCommand:
    """Test 'available' CLI command."""

    def test_available_shows_all_pseudos(self, runner):
        """Test that available command shows all pseudopotentials."""
        result = runner.invoke(cli, ["available"])

        assert result.exit_code == 0
        # Rich may truncate names in table - check for key components
        assert "List of available pseudopotential" in result.output
        assert "PBEsol" in result.output
        assert "PBE" in result.output
        # Check table shows ONCVPSP entries
        assert "ONCVPSP" in result.output


class TestListCommand:
    """Test 'list' CLI command."""

    def test_list_no_pseudo_dir(self, runner, tmp_path, monkeypatch):
        """Test list when pseudo directory doesn't exist."""
        # Set PSEUDO_DIR to non-existent path
        fake_dir = tmp_path / "nonexistent"
        monkeypatch.setattr(
            "atomate2.siesta.cli.pseudo.siesta_pseudos.PSEUDO_DIR", str(fake_dir)
        )

        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "Could not find" in result.output or "not found" in result.output.lower()

    def test_list_with_installed_pseudos(self, runner, mock_pseudo_dir):
        """Test list with some pseudos installed."""
        # Create a pseudo folder
        pseudo_folder = mock_pseudo_dir / "ONCVPSP-PBEsol-FR-PDv0.4-Standard"
        pseudo_folder.mkdir()

        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        # Should show the pseudo table
        assert "pseudopotential" in result.output.lower()


class TestInstallCommand:
    """Test 'install' CLI command."""

    @patch("atomate2.siesta.cli.pseudo.siesta_pseudos.download_and_extract_pseudo")
    def test_install_valid_pseudo(self, mock_download, runner, mock_pseudo_dir):
        """Test installing a valid pseudopotential."""
        result = runner.invoke(cli, ["install", "ONCVPSP-PBEsol-FR-PDv0.4-Standard"])

        assert result.exit_code == 0
        mock_download.assert_called_once()

    def test_install_invalid_pseudo(self, runner, mock_pseudo_dir):
        """Test installing an invalid pseudopotential name."""
        result = runner.invoke(cli, ["install", "NonExistent-Pseudo"])

        assert result.exit_code == 0
        assert "not found" in result.output

    @patch("atomate2.siesta.cli.pseudo.siesta_pseudos.download_and_extract_pseudo")
    def test_install_with_local_only_flag(self, mock_download, runner, mock_pseudo_dir):
        """Test install with --local-only flag."""
        result = runner.invoke(
            cli, ["install", "ONCVPSP-PBEsol-FR-PDv0.4-Standard", "--local-only"]
        )

        assert result.exit_code == 0
        # Check that local_only=True was passed
        call_args = mock_download.call_args
        assert call_args[1]["local_only"] is True


class TestUninstallCommand:
    """Test 'uninstall' CLI command."""

    def test_uninstall_invalid_pseudo(self, runner, mock_pseudo_dir):
        """Test uninstalling an invalid pseudopotential name."""
        result = runner.invoke(cli, ["uninstall", "NonExistent-Pseudo"])

        assert result.exit_code == 0
        assert "not found" in result.output

    def test_uninstall_not_installed(self, runner, mock_pseudo_dir):
        """Test uninstalling a pseudo that's not installed."""
        result = runner.invoke(
            cli, ["uninstall", "ONCVPSP-PBEsol-FR-PDv0.4-Standard", "--force"]
        )

        assert result.exit_code == 0
        assert "not installed" in result.output

    @patch("atomate2.siesta.cli.pseudo.siesta_pseudos.shutil.rmtree")
    def test_uninstall_with_force(self, mock_rmtree, runner, mock_pseudo_dir):
        """Test uninstalling with --force flag."""
        # Create pseudo folder
        pseudo_folder = mock_pseudo_dir / "ONCVPSP-PBEsol-FR-PDv0.4-Standard"
        pseudo_folder.mkdir()

        result = runner.invoke(
            cli, ["uninstall", "ONCVPSP-PBEsol-FR-PDv0.4-Standard", "--force"]
        )

        assert result.exit_code == 0
        mock_rmtree.assert_called_once()

    def test_uninstall_without_force_abort(self, runner, mock_pseudo_dir):
        """Test uninstalling without --force requires confirmation."""
        # Create pseudo folder
        pseudo_folder = mock_pseudo_dir / "ONCVPSP-PBEsol-FR-PDv0.4-Standard"
        pseudo_folder.mkdir()

        # Simulate user aborting confirmation
        result = runner.invoke(
            cli, ["uninstall", "ONCVPSP-PBEsol-FR-PDv0.4-Standard"], input="n\n"
        )

        # Should abort
        assert result.exit_code != 0
        # Folder should still exist
        assert pseudo_folder.exists()


class TestShowCommand:
    """Test 'show' CLI command."""

    def test_show_valid_pseudo(self, runner, mock_pseudo_dir):
        """Test showing info for a valid pseudopotential."""
        result = runner.invoke(cli, ["show", "ONCVPSP-PBEsol-FR-PDv0.4-Standard"])

        assert result.exit_code == 0
        assert "Information about" in result.output
        assert "XC Functional" in result.output or "PBEsol" in result.output

    def test_show_invalid_pseudo(self, runner, mock_pseudo_dir):
        """Test showing info for an invalid pseudopotential."""
        result = runner.invoke(cli, ["show", "NonExistent-Pseudo"])

        assert result.exit_code == 0
        assert "not found" in result.output

    def test_show_installed_pseudo(self, runner, mock_pseudo_dir):
        """Test showing info for an installed pseudopotential."""
        # Create pseudo folder
        pseudo_folder = mock_pseudo_dir / "ONCVPSP-PBEsol-FR-PDv0.4-Standard"
        pseudo_folder.mkdir()

        result = runner.invoke(cli, ["show", "ONCVPSP-PBEsol-FR-PDv0.4-Standard"])

        assert result.exit_code == 0
        assert (
            "Installed Path" in result.output or str(mock_pseudo_dir) in result.output
        )


class TestElementCommand:
    """Test 'element' CLI command."""

    def test_element_found(self, runner):
        """Test finding pseudos for an element that exists."""
        result = runner.invoke(cli, ["element", "Si"])

        assert result.exit_code == 0
        assert "Pseudos found" in result.output
        assert "Si" in result.output

    def test_element_not_found(self, runner):
        """Test finding pseudos for an element that doesn't exist."""
        result = runner.invoke(cli, ["element", "Unobtainium"])

        assert result.exit_code == 0
        assert "No pseudos found" in result.output

    def test_element_case_sensitive(self, runner):
        """Test element search is case-sensitive."""
        # Should find Si (capital S, lowercase i)
        result = runner.invoke(cli, ["element", "Si"])
        assert result.exit_code == 0
        assert "Pseudos found" in result.output


class TestPlotCommand:
    """Test 'plot' CLI command."""

    def test_plot_invalid_pseudo(self, runner, mock_pseudo_dir):
        """Test plotting with invalid pseudopotential name."""
        result = runner.invoke(cli, ["plot", "NonExistent-Pseudo", "Si"])

        assert result.exit_code == 0
        assert "not found" in result.output

    def test_plot_pseudo_not_installed(self, runner, mock_pseudo_dir):
        """Test plotting when pseudo is not installed."""
        result = runner.invoke(cli, ["plot", "ONCVPSP-PBEsol-FR-PDv0.4-Standard", "Si"])

        assert result.exit_code == 0
        assert "not installed" in result.output

    @patch("atomate2.siesta.cli.pseudo.siesta_pseudos.parse_psml")
    def test_plot_psml_file_not_found(self, mock_parse, runner, mock_pseudo_dir):
        """Test plotting when PSML file for element doesn't exist."""
        # Create pseudo folder but not the element file
        pseudo_folder = mock_pseudo_dir / "ONCVPSP-PBEsol-FR-PDv0.4-Standard"
        pseudo_folder.mkdir()

        result = runner.invoke(cli, ["plot", "ONCVPSP-PBEsol-FR-PDv0.4-Standard", "Si"])

        assert result.exit_code == 0
        assert "not found" in result.output

    @patch("atomate2.siesta.cli.pseudo.siesta_pseudos.plot_wavefunctions")
    @patch("atomate2.siesta.cli.pseudo.siesta_pseudos.parse_psml")
    def test_plot_wavefunctions_success(
        self, mock_parse, mock_plot_wf, runner, mock_pseudo_dir
    ):
        """Test successful plotting of wavefunctions."""
        import numpy as np

        # Create pseudo folder and element file
        pseudo_folder = mock_pseudo_dir / "ONCVPSP-PBEsol-FR-PDv0.4-Standard"
        pseudo_folder.mkdir()
        psml_file = pseudo_folder / "Si.psml"
        psml_file.write_text("dummy content")

        # Mock parse_psml return
        mock_parse.return_value = (
            np.array([0.0, 1.0, 2.0]),  # radial_grid
            [{"n": 3, "l": 0, "occupation": 2.0}],  # valence_config
            [{"n": 3, "l": 0, "data": np.array([0.0, 0.1, 0.2])}],  # wavefunctions
            [{"l": None, "data": np.array([-1.0, -0.5, 0.0])}],  # potentials
            "Si",  # element_name
        )

        result = runner.invoke(
            cli,
            [
                "plot",
                "ONCVPSP-PBEsol-FR-PDv0.4-Standard",
                "Si",
                "--plot-type",
                "wavefunctions",
            ],
        )

        assert result.exit_code == 0
        mock_parse.assert_called_once()
        mock_plot_wf.assert_called_once()

    @patch("atomate2.siesta.cli.pseudo.siesta_pseudos.parse_psml")
    def test_plot_empty_radial_grid(self, mock_parse, runner, mock_pseudo_dir):
        """Test plotting with empty radial grid."""
        import numpy as np

        # Create pseudo folder and element file
        pseudo_folder = mock_pseudo_dir / "ONCVPSP-PBEsol-FR-PDv0.4-Standard"
        pseudo_folder.mkdir()
        psml_file = pseudo_folder / "Si.psml"
        psml_file.write_text("dummy content")

        # Mock parse_psml return with empty radial grid
        mock_parse.return_value = (
            np.array([]),  # Empty radial_grid
            [],
            [],
            [],
            "Si",
        )

        result = runner.invoke(cli, ["plot", "ONCVPSP-PBEsol-FR-PDv0.4-Standard", "Si"])

        assert result.exit_code == 0
        assert "No radial grid data found" in result.output

    @patch("atomate2.siesta.cli.pseudo.siesta_pseudos.parse_psml")
    def test_plot_parse_error(self, mock_parse, runner, mock_pseudo_dir):
        """Test plotting when parsing fails."""
        # Create pseudo folder and element file
        pseudo_folder = mock_pseudo_dir / "ONCVPSP-PBEsol-FR-PDv0.4-Standard"
        pseudo_folder.mkdir()
        psml_file = pseudo_folder / "Si.psml"
        psml_file.write_text("dummy content")

        # Mock parse_psml to raise ValueError
        mock_parse.side_effect = ValueError("Failed to parse PSML file")

        result = runner.invoke(cli, ["plot", "ONCVPSP-PBEsol-FR-PDv0.4-Standard", "Si"])

        assert result.exit_code != 0
        assert "Error" in result.output


class TestCLIGroup:
    """Test CLI group functionality."""

    def test_cli_help(self, runner):
        """Test CLI help output."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "pseudopotential management" in result.output.lower()
        assert "available" in result.output
        assert "list" in result.output
        assert "install" in result.output

    def test_cli_version(self, runner):
        """Test CLI version option."""
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "version" in result.output.lower() or "0.1.0" in result.output


class TestPSEUDOSData:
    """Test PSEUDOS data structure."""

    def test_pseudos_not_empty(self):
        """Test that PSEUDOS dictionary is not empty."""
        assert len(PSEUDOS) > 0

    def test_pseudos_have_required_keys(self):
        """Test that each pseudo has required keys."""
        required_keys = [
            "filename",
            "local_path",
            "xc_name",
            "relativity_type",
            "version",
            "elements",
            "url",
        ]

        for name, data in PSEUDOS.items():
            for key in required_keys:
                assert key in data, f"Pseudo '{name}' missing key '{key}'"

    def test_pseudos_elements_are_sets(self):
        """Test that elements are stored as sets."""
        for name, data in PSEUDOS.items():
            assert isinstance(
                data["elements"], set
            ), f"Pseudo '{name}' elements should be a set"

    def test_pseudos_have_valid_elements(self):
        """Test that element symbols are valid."""
        for name, data in PSEUDOS.items():
            for element in data["elements"]:
                # Element symbols should be 1-2 characters, start with capital
                assert 1 <= len(element) <= 2
                assert element[0].isupper()
