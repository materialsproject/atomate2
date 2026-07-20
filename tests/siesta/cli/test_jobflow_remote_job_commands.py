"""Tests for jobflow-remote job management commands.

This module tests the job inspect, recreate, and modify-db commands
for FDF parameter modification.
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from atomate2.siesta.cli.jobflow_remote.main import cli
from atomate2.siesta.cli.jobflow_remote.parameter_modifier import (
    merge_parameters,
    parse_parameter_string,
    validate_fdf_parameter,
)


class TestParameterParsing:
    """Tests for parameter string parsing."""

    def test_parse_simple_string(self):
        """Test parsing simple string value."""
        result = parse_parameter_string("SystemName=TestSystem")
        assert result == ("SystemName", "TestSystem")

    def test_parse_integer(self):
        """Test parsing integer value."""
        result = parse_parameter_string("MaxSCFIterations=100")
        assert result == ("MaxSCFIterations", 100)

    def test_parse_float(self):
        """Test parsing float value."""
        result = parse_parameter_string("DM.Tolerance=1e-4")
        assert result == ("DM.Tolerance", 1e-4)

    def test_parse_list(self):
        """Test parsing list value."""
        result = parse_parameter_string("kpts=[4,4,4]")
        assert result == ("kpts", [4, 4, 4])

    def test_parse_boolean_true(self):
        """Test parsing boolean true."""
        result = parse_parameter_string("SaveHS=true")
        assert result == ("SaveHS", True)

    def test_parse_boolean_false(self):
        """Test parsing boolean false."""
        result = parse_parameter_string("WriteKpoints=false")
        assert result == ("WriteKpoints", False)

    def test_parse_string_with_units(self):
        """Test parsing string with units."""
        result = parse_parameter_string("Mesh.Cutoff=300 Ry")
        assert result == ("Mesh.Cutoff", "300 Ry")

    def test_parse_invalid_format(self):
        """Test parsing invalid format without equals."""
        result = parse_parameter_string("InvalidParameter")
        assert result is None


class TestParameterValidation:
    """Tests for FDF parameter validation."""

    def test_validate_kpts_valid(self):
        """Test validation of valid kpts."""
        is_valid, msg = validate_fdf_parameter("kpts", [4, 4, 4])
        assert is_valid is True
        assert msg == ""

    def test_validate_kpts_invalid_length(self):
        """Test validation of kpts with wrong length."""
        is_valid, msg = validate_fdf_parameter("kpts", [4, 4])
        assert is_valid is False
        assert "3 integers" in msg

    def test_validate_kpts_invalid_type(self):
        """Test validation of kpts with wrong type."""
        is_valid, msg = validate_fdf_parameter("kpts", "4 4 4")
        assert is_valid is False

    def test_validate_spin_valid(self):
        """Test validation of valid Spin parameter."""
        is_valid, msg = validate_fdf_parameter("Spin", "polarized")
        assert is_valid is True

    def test_validate_spin_invalid(self):
        """Test validation of invalid Spin parameter."""
        is_valid, msg = validate_fdf_parameter("Spin", "invalid_value")
        assert is_valid is False
        assert "polarized" in msg

    def test_validate_mesh_cutoff_with_units(self):
        """Test validation of Mesh.Cutoff with units."""
        is_valid, msg = validate_fdf_parameter("Mesh.Cutoff", "300 Ry")
        assert is_valid is True

    def test_validate_mesh_cutoff_without_units(self):
        """Test validation of Mesh.Cutoff without units."""
        is_valid, msg = validate_fdf_parameter("Mesh.Cutoff", "300")
        assert is_valid is False
        assert "units" in msg

    def test_validate_common_typo(self):
        """Test detection of common typo."""
        is_valid, msg = validate_fdf_parameter("MeshCutoff", "300 Ry")
        assert is_valid is False
        # Unregistered typo is rejected with a suggestion of the registered
        # parameter (registry stores names lowercase).
        assert "mesh.cutoff" in msg.lower()


class TestParameterMerging:
    """Tests for parameter merging."""

    def test_merge_empty_base(self):
        """Test merging into empty base."""
        base = {}
        override = {"kpts": [4, 4, 4]}
        result = merge_parameters(base, override)
        assert result == {"kpts": [4, 4, 4]}

    def test_merge_override_existing(self):
        """Test overriding existing parameter."""
        base = {"kpts": [2, 2, 2], "Mesh.Cutoff": "200 Ry"}
        override = {"kpts": [4, 4, 4]}
        result = merge_parameters(base, override)
        assert result["kpts"] == [4, 4, 4]
        assert result["Mesh.Cutoff"] == "200 Ry"

    def test_merge_add_new(self):
        """Test adding new parameter."""
        base = {"kpts": [2, 2, 2]}
        override = {"Mesh.Cutoff": "300 Ry"}
        result = merge_parameters(base, override)
        assert len(result) == 2
        assert result["kpts"] == [2, 2, 2]
        assert result["Mesh.Cutoff"] == "300 Ry"


class TestJobInspectCommand:
    """Tests for job inspect command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("atomate2.siesta.cli.jobflow_remote.inspect.get_job_info_from_jf")
    def test_inspect_basic(self, mock_get_job_info):
        """Test basic job inspection."""
        mock_get_job_info.return_value = {
            "db_id": "70",
            "name": "RelaxJob",
            "state": "COMPLETED",
            "worker": "local_worker",
            "uuid": "test-uuid-1234",
        }

        result = self.runner.invoke(cli, ["-p", "test", "job", "inspect", "70"])

        assert result.exit_code == 0
        assert "RelaxJob" in result.output
        assert "COMPLETED" in result.output

    @patch("atomate2.siesta.cli.jobflow_remote.inspect.get_job_info_from_jf")
    def test_inspect_job_not_found(self, mock_get_job_info):
        """Test inspection of non-existent job."""
        mock_get_job_info.return_value = None

        result = self.runner.invoke(cli, ["-p", "test", "job", "inspect", "999"])

        assert result.exit_code == 1


class TestJobRecreateCommand:
    """Tests for job recreate command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_recreate_no_modifications(self):
        """Test recreate without modifications specified."""
        result = self.runner.invoke(cli, ["-p", "test", "job", "recreate", "70"])

        assert result.exit_code == 1
        assert "No modifications specified" in result.output

    @patch("atomate2.siesta.cli.jobflow_remote.recreate.get_job_details_from_db")
    @patch("atomate2.siesta.cli.jobflow_remote.recreate.extract_fdf_parameters")
    def test_recreate_with_modifications(self, mock_extract, mock_get_job, tmp_path):
        """Test recreate with valid modifications."""
        # Mock job document
        mock_get_job.return_value = {
            "db_id": 70,
            "job": {"name": "RelaxJob", "function_kwargs": {}},
        }

        # Mock original parameters
        mock_extract.return_value = {
            "kpts": [2, 2, 2],
            "Mesh.Cutoff": "200 Ry",
        }

        # Run with modification
        output_file = tmp_path / "test_recreate.py"
        result = self.runner.invoke(
            cli,
            [
                "-p",
                "test",
                "job",
                "recreate",
                "70",
                "-m",
                "kpts=[4,4,4]",
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        # Check script content
        content = output_file.read_text()
        assert "kpts" in content
        assert "[4, 4, 4]" in content

    @patch("atomate2.siesta.cli.jobflow_remote.recreate.get_job_details_from_db")
    @patch("atomate2.siesta.cli.jobflow_remote.recreate.extract_fdf_parameters")
    def test_recreate_preview_only(self, mock_extract, mock_get_job):
        """Test recreate with preview-only mode."""
        mock_get_job.return_value = {
            "db_id": 70,
            "job": {"name": "RelaxJob", "function_kwargs": {}},
        }

        mock_extract.return_value = {"kpts": [2, 2, 2]}

        result = self.runner.invoke(
            cli,
            [
                "-p",
                "test",
                "job",
                "recreate",
                "70",
                "-m",
                "kpts=[4,4,4]",
                "--preview-only",
            ],
        )

        assert result.exit_code == 0
        assert "Preview mode" in result.output

    def test_recreate_invalid_parameter(self):
        """Test recreate with invalid parameter format."""
        result = self.runner.invoke(
            cli,
            ["-p", "test", "job", "recreate", "70", "-m", "invalid_param"],
        )

        assert result.exit_code == 1


class TestJobModifyDbCommand:
    """Tests for job modify-db command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_modify_db_no_parameters(self):
        """Test modify-db without parameters."""
        result = self.runner.invoke(
            cli, ["-p", "test", "job", "modify-db", "70", "--force"]
        )

        assert result.exit_code == 1
        assert "No modifications specified" in result.output

    @patch("atomate2.siesta.cli.jobflow_remote.modify_db.get_job_details_from_db")
    def test_modify_db_requires_confirmation(self, mock_get_job):
        """Test that modify-db requires confirmation without --force."""
        mock_get_job.return_value = {
            "db_id": 70,
            "job": {"name": "RelaxJob", "function_kwargs": {}},
        }

        # Without --force, should abort when user says no
        result = self.runner.invoke(
            cli,
            ["-p", "test", "job", "modify-db", "70", "--param", "kpts=[4,4,4]"],
            input="n\n",  # User says no to confirmation
        )

        assert "WARNING" in result.output or "DANGER" in result.output

    @patch("atomate2.siesta.cli.jobflow_remote.modify_db.get_job_details_from_db")
    @patch("atomate2.siesta.cli.jobflow_remote.modify_db.extract_fdf_parameters")
    @patch("atomate2.siesta.cli.jobflow_remote.modify_db._modify_job_in_database")
    def test_modify_db_with_force(self, mock_modify, mock_extract, mock_get_job):
        """Test modify-db with --force flag."""
        mock_get_job.return_value = {
            "db_id": 70,
            "job": {"name": "RelaxJob", "function_kwargs": {}},
        }

        mock_extract.return_value = {"kpts": [2, 2, 2]}
        mock_modify.return_value = True

        result = self.runner.invoke(
            cli,
            [
                "-p",
                "test",
                "job",
                "modify-db",
                "70",
                "--param",
                "kpts=[4,4,4]",
                "--force",
            ],
        )

        # Should process without prompts
        assert "WARNING" in result.output or "DANGER" in result.output


class TestCLIIntegration:
    """Integration tests for CLI structure."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_job_group_exists(self):
        """Test that job command group exists."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "job" in result.output

    def test_job_commands_listed(self):
        """Test that job subcommands are listed."""
        result = self.runner.invoke(cli, ["-p", "test", "job", "--help"])
        assert result.exit_code == 0
        assert "inspect" in result.output
        assert "modify-db" in result.output
        assert "update-resources" in result.output

    def test_project_name_context_passing(self):
        """Test that -p flag is properly passed to subcommands."""
        result = self.runner.invoke(cli, ["-p", "my_project", "job", "--help"])
        assert result.exit_code == 0
