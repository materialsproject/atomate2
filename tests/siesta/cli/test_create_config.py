"""Tests for create_config CLI module."""

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from atomate2.siesta.cli.config import cli, ensure_config_file


@pytest.fixture
def runner():
    """Create Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary directory for config files."""
    return tmp_path / "config"


class TestEnsureConfigFile:
    """Test ensure_config_file function."""

    def test_create_new_config_file(self, temp_config_dir):
        """Test creating a new config file."""
        config_path = ensure_config_file(
            file_name="test-config.yaml",
            config_dir=str(temp_config_dir),
            siesta_cmd="siesta < siesta.fdf",
            siesta_pp_path="/path/to/pseudos",
            flos_path="/path/to/flos",
            optical_input_cmd="optical_input < siesta.EPSIMG",
            optical_cmd="optical < siesta.EPSIMG",
            show_banner=True,
        )

        assert config_path is not None
        assert config_path.exists()
        assert config_path.name == "test-config.yaml"

        # Check file contents
        content = config_path.read_text()
        assert 'SIESTA_CMD: "siesta < siesta.fdf"' in content
        assert 'SIESTA_PP_PATH: "/path/to/pseudos"' in content
        assert 'FLOS_PATH: "/path/to/flos"' in content
        assert "SIESTA_SHOW_BANNER: True" in content

    def test_existing_config_file_raises_error(self, temp_config_dir):
        """Test that existing config file raises FileExistsError."""
        # Create config file first
        temp_config_dir.mkdir(parents=True, exist_ok=True)
        config_path = temp_config_dir / "existing-config.yaml"
        config_path.write_text("EXISTING: content")

        # Try to create again - should raise error
        result = ensure_config_file(
            file_name="existing-config.yaml",
            config_dir=str(temp_config_dir),
            siesta_cmd="siesta",
            siesta_pp_path="/path",
            flos_path="/flos",
            optical_input_cmd="opt_input",
            optical_cmd="optical",
            show_banner=False,
        )

        # Should return None due to exception handling
        assert result is None

    def test_show_banner_false(self, temp_config_dir):
        """Test creating config with show_banner=False."""
        config_path = ensure_config_file(
            file_name="no-banner.yaml",
            config_dir=str(temp_config_dir),
            siesta_cmd="siesta",
            siesta_pp_path="/pseudos",
            flos_path="/flos",
            optical_input_cmd="opt_input",
            optical_cmd="optical",
            show_banner=False,
        )

        assert config_path is not None
        content = config_path.read_text()
        assert "SIESTA_SHOW_BANNER: False" in content

    def test_creates_parent_directories(self, temp_config_dir):
        """Test that parent directories are created if they don't exist."""
        nested_dir = temp_config_dir / "nested" / "deep" / "path"

        config_path = ensure_config_file(
            file_name="nested-config.yaml",
            config_dir=str(nested_dir),
            siesta_cmd="siesta",
            siesta_pp_path="/pseudos",
            flos_path="/flos",
            optical_input_cmd="opt_input",
            optical_cmd="optical",
            show_banner=True,
        )

        assert config_path is not None
        assert nested_dir.exists()
        assert config_path.exists()


class TestCreateCommand:
    """Test 'create' CLI command."""

    def test_create_default_config(self, runner, tmp_path):
        """Test creating config with default parameters."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["create"])

            assert result.exit_code == 0
            assert (
                "Created SIESTA configuration file" in result.output
                or "already exists" in result.output
            )

            # Check that default file was created
            config_path = Path.cwd() / "atomate2siesta-local.yaml"
            if config_path.exists():
                content = config_path.read_text()
                assert "SIESTA_CMD" in content
                assert "SIESTA_PP_PATH" in content

    def test_create_custom_filename(self, runner, tmp_path):
        """Test creating config with custom filename."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["create", "--file-name", "custom-config.yaml"])

            assert result.exit_code == 0
            config_path = Path.cwd() / "custom-config.yaml"
            assert config_path.exists()

    def test_create_custom_output_dir(self, runner, tmp_path):
        """Test creating config in custom directory."""
        output_dir = tmp_path / "custom_output"
        output_dir.mkdir()

        result = runner.invoke(cli, ["create", "--output-dir", str(output_dir)])

        assert result.exit_code == 0
        # Rich may format output across multiple lines - check key parts
        assert "Using output directory" in result.output
        assert "custom_output" in result.output

        config_path = output_dir / ".atomate2siesta-local.yaml"
        assert config_path.exists()

    def test_create_custom_siesta_cmd(self, runner, tmp_path):
        """Test creating config with custom SIESTA command."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli,
                [
                    "create",
                    "--file-name",
                    "custom-cmd.yaml",
                    "--siesta-cmd",
                    "mpirun -np 4 siesta",
                ],
            )

            assert result.exit_code == 0
            config_path = Path.cwd() / "custom-cmd.yaml"
            content = config_path.read_text()
            assert 'SIESTA_CMD: "mpirun -np 4 siesta"' in content

    def test_create_custom_paths(self, runner, tmp_path):
        """Test creating config with custom paths."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli,
                [
                    "create",
                    "--file-name",
                    "custom-paths.yaml",
                    "--siesta-pp-path",
                    "/custom/pseudos",
                    "--flos-path",
                    "/custom/flos",
                ],
            )

            assert result.exit_code == 0
            config_path = Path.cwd() / "custom-paths.yaml"
            content = config_path.read_text()
            assert 'SIESTA_PP_PATH: "/custom/pseudos"' in content
            assert 'FLOS_PATH: "/custom/flos"' in content

    def test_create_no_banner(self, runner, tmp_path):
        """Test creating config with banner disabled."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli, ["create", "--file-name", "no-banner.yaml", "--no-show-banner"]
            )

            assert result.exit_code == 0
            config_path = Path.cwd() / "no-banner.yaml"
            content = config_path.read_text()
            assert "SIESTA_SHOW_BANNER: False" in content

    def test_create_export_command_shown(self, runner, tmp_path):
        """Test that export command is shown to user."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["create", "--file-name", "export-test.yaml"])

            assert result.exit_code == 0
            # Rich may split text across lines - check for both parts
            assert "export" in result.output
            assert "ATOMATE2_CONFIG_FILE" in result.output
            assert "export-test.yaml" in result.output


class TestSetCommand:
    """Test 'set' CLI command."""

    def test_set_absolute_path(self, runner, tmp_path):
        """Test setting config with absolute path."""
        # Create a config file
        config_file = tmp_path / "test-config.yaml"
        config_file.write_text("SIESTA_CMD: test")

        result = runner.invoke(cli, ["set", str(config_file)])

        assert result.exit_code == 0
        # Rich may split text across lines - check for both parts
        assert "export" in result.output
        assert "ATOMATE2_CONFIG_FILE" in result.output
        assert "test-config.yaml" in result.output

    def test_set_relative_path(self, runner, tmp_path):
        """Test setting config with relative path."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config file in current directory
            config_file = Path.cwd() / "relative-config.yaml"
            config_file.write_text("SIESTA_CMD: test")

            result = runner.invoke(cli, ["set", "relative-config.yaml"])

            assert result.exit_code == 0
            # Rich may split text across lines - check for both parts
            assert "export" in result.output
            assert "ATOMATE2_CONFIG_FILE" in result.output

    def test_set_nonexistent_file(self, runner, tmp_path):
        """Test setting nonexistent config file."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["set", "nonexistent.yaml"])

            assert result.exit_code == 0  # Click doesn't exit with error
            assert "does not exist" in result.output

    def test_set_unreadable_file(self, runner, tmp_path, monkeypatch):
        """Test setting unreadable config file."""
        # Create a config file
        config_file = tmp_path / "unreadable.yaml"
        config_file.write_text("SIESTA_CMD: test")

        # Mock os.access to return False (file not readable)
        def mock_access(path, mode):
            if str(path) == str(config_file) and mode == os.R_OK:
                return False
            return True

        with monkeypatch.context() as m:
            m.setattr(os, "access", mock_access)
            result = runner.invoke(cli, ["set", str(config_file)])

            assert result.exit_code == 0
            assert "not readable" in result.output


class TestCLIGroup:
    """Test CLI group and general functionality."""

    def test_cli_group_help(self, runner):
        """Test CLI help output."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "CLI for Atomate2 SIESTA configuration" in result.output
        assert "create" in result.output
        assert "set" in result.output

    def test_create_help(self, runner):
        """Test create command help."""
        result = runner.invoke(cli, ["create", "--help"])

        assert result.exit_code == 0
        assert "--file-name" in result.output
        assert "--output-dir" in result.output
        assert "--siesta-cmd" in result.output
        assert "--show-banner" in result.output

    def test_set_help(self, runner):
        """Test set command help."""
        result = runner.invoke(cli, ["set", "--help"])

        assert result.exit_code == 0
        assert "file_path" in result.output.lower()
        assert "ATOMATE2_CONFIG_FILE" in result.output
