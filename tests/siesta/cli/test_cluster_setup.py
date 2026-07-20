"""Tests for cluster setup CLI tool.

This module tests the atomate2siesta-cluster command-line interface for remote
cluster setup with conda environments for jobflow-remote.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

from click.testing import CliRunner

from atomate2.siesta.cli.cluster import cli
from atomate2.siesta.cli.cluster.proxy_utils import (
    add_proxy_to_remote_bashrc,
    configure_proxy_on_remote,
    detect_proxy_on_remote,
    get_squid_status,
    is_proxy_error,
    is_squid_installed,
    is_squid_running,
    show_proxy_error_help,
    start_squid,
    stop_squid,
)
from atomate2.siesta.cli.cluster.ssh_utils import (
    cleanup_ssh_tunnel,
    create_ssh_reverse_tunnel,
    create_ssh_tunnel,
    run_ssh_command,
    show_verbose_output,
)


class TestShowVerboseOutput:
    """Tests for show_verbose_output function."""

    def test_verbose_disabled(self, capsys):
        """Test that output is not shown when verbose is False."""
        show_verbose_output("stdout content", "stderr content", verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_stdout(self, capsys):
        """Test showing stdout in verbose mode."""
        show_verbose_output("line1\nline2", "", verbose=True)
        captured = capsys.readouterr()
        assert "stdout:" in captured.out
        assert "line1" in captured.out
        assert "line2" in captured.out

    def test_verbose_stderr(self, capsys):
        """Test showing stderr in verbose mode."""
        show_verbose_output("", "error1\nerror2", verbose=True)
        captured = capsys.readouterr()
        assert "stderr:" in captured.out
        assert "error1" in captured.out
        assert "error2" in captured.out

    def test_verbose_empty_strings(self, capsys):
        """Test handling of empty stdout/stderr."""
        show_verbose_output("", "", verbose=True)
        captured = capsys.readouterr()
        # Should not show anything for empty strings
        assert "stdout:" not in captured.out
        assert "stderr:" not in captured.out


class TestRunSSHCommand:
    """Tests for run_ssh_command function."""

    @patch("atomate2.siesta.cli.cluster.ssh_utils.subprocess.run")
    def test_basic_ssh_command(self, mock_run):
        """Test basic SSH command execution."""
        mock_run.return_value = Mock(returncode=0, stdout="output", stderr="")

        returncode, stdout, stderr = run_ssh_command("host.edu", "user", "echo test")

        assert returncode == 0
        assert stdout == "output"
        assert stderr == ""
        mock_run.assert_called_once()

    @patch("atomate2.siesta.cli.cluster.ssh_utils.subprocess.run")
    def test_ssh_with_identity_file(self, mock_run):
        """Test SSH command with identity file."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        run_ssh_command("host.edu", "user", "echo test", identity_file="/path/to/key")

        # Check that identity file was added to SSH command
        call_args = mock_run.call_args[0][0]
        assert "-i" in call_args
        assert "/path/to/key" in call_args

    @patch("atomate2.siesta.cli.cluster.ssh_utils.subprocess.run")
    def test_ssh_with_password(self, mock_run):
        """Test SSH command with password (sshpass)."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        run_ssh_command("host.edu", "user", "echo test", password="secret")

        # Check that sshpass was used
        call_args = mock_run.call_args[0][0]
        assert "sshpass" in call_args[0]

    @patch("atomate2.siesta.cli.cluster.ssh_utils.subprocess.run")
    def test_ssh_config_mode(self, mock_run):
        """Test SSH command with SSH config alias."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        run_ssh_command("myalias", None, "echo test", use_ssh_config=True)

        # Check that user@host format was not used
        call_args = mock_run.call_args[0][0]
        ssh_destination = call_args[-2]  # Second to last argument
        assert "@" not in ssh_destination
        assert ssh_destination == "myalias"

    @patch("atomate2.siesta.cli.cluster.ssh_utils.subprocess.run")
    def test_ssh_timeout(self, mock_run):
        """Test SSH command timeout handling."""
        from subprocess import TimeoutExpired

        mock_run.side_effect = TimeoutExpired("ssh", 300)

        returncode, stdout, stderr = run_ssh_command("host.edu", "user", "sleep 1000")

        assert returncode == 1
        assert "timed out" in stderr

    @patch("atomate2.siesta.cli.cluster.ssh_utils.subprocess.run")
    def test_sshpass_not_found(self, mock_run):
        """Test error when sshpass is not installed."""
        mock_run.side_effect = FileNotFoundError("sshpass: command not found")

        returncode, stdout, stderr = run_ssh_command(
            "host.edu", "user", "echo test", password="secret"
        )

        assert returncode == 1
        assert "sshpass not found" in stderr


class TestSSHTunnelFunctions:
    """Tests for SSH tunnel creation and management."""

    @patch("atomate2.siesta.cli.cluster.ssh_utils.subprocess.run")
    def test_create_ssh_tunnel_success(self, mock_run):
        """Test successful SSH tunnel creation."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        result = create_ssh_tunnel("host.edu", "user", 3129)

        assert result is True
        # Check that -D flag was used for SOCKS proxy
        call_args = mock_run.call_args[0][0]
        assert "-D" in call_args
        assert "3129" in call_args

    @patch("atomate2.siesta.cli.cluster.ssh_utils.subprocess.run")
    def test_create_ssh_tunnel_failure(self, mock_run):
        """Test SSH tunnel creation failure."""
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="Port in use")

        result = create_ssh_tunnel("host.edu", "user", 3129)

        assert result is None

    @patch("atomate2.siesta.cli.cluster.ssh_utils.subprocess.run")
    def test_create_reverse_tunnel_success(self, mock_run):
        """Test successful reverse SSH tunnel creation."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        result = create_ssh_reverse_tunnel("host.edu", "user", 3129, None, False)

        assert result is True
        # Check that -R flag was used for reverse tunnel
        call_args = mock_run.call_args[0][0]
        assert "-R" in call_args
        assert "3129:localhost:3129" in call_args

    @patch("atomate2.siesta.cli.cluster.ssh_utils.subprocess.run")
    def test_cleanup_ssh_tunnel(self, mock_run):
        """Test SSH tunnel cleanup."""
        # Mock pgrep and kill commands
        mock_run.side_effect = [
            Mock(returncode=0, stdout="12345\n67890\n"),  # pgrep -f ssh.*-D.*3129
            Mock(returncode=0),  # kill 12345
            Mock(returncode=0),  # kill 67890
            Mock(returncode=1, stdout=""),  # pgrep -f ssh.*-R.*3129 (no results)
        ]

        cleanup_ssh_tunnel(3129)

        # Should have called pgrep twice (for -D and -R patterns)
        assert mock_run.call_count >= 2


class TestSquidFunctions:
    """Tests for Squid HTTP proxy management."""

    @patch("atomate2.siesta.cli.cluster.proxy_utils.subprocess.run")
    def test_is_squid_installed_true(self, mock_run):
        """Test detecting installed squid."""
        mock_run.return_value = Mock(returncode=0)

        result = is_squid_installed()

        assert result is True
        mock_run.assert_called_once()
        assert "which" in mock_run.call_args[0][0][0]

    @patch("atomate2.siesta.cli.cluster.proxy_utils.subprocess.run")
    def test_is_squid_installed_false(self, mock_run):
        """Test detecting missing squid."""
        mock_run.return_value = Mock(returncode=1)

        result = is_squid_installed()

        assert result is False

    @patch("atomate2.siesta.cli.cluster.proxy_utils.subprocess.run")
    def test_is_squid_running_true(self, mock_run):
        """Test detecting running squid."""
        mock_run.return_value = Mock(returncode=0, stdout="squid listening on port")

        result = is_squid_running(3129)

        assert result is True

    @patch("atomate2.siesta.cli.cluster.proxy_utils.subprocess.run")
    def test_is_squid_running_false(self, mock_run):
        """Test detecting stopped squid."""
        mock_run.return_value = Mock(returncode=1, stdout="")

        result = is_squid_running(3129)

        assert result is False

    @patch("atomate2.siesta.cli.cluster.proxy_utils.subprocess.run")
    def test_stop_squid_success(self, mock_run):
        """Test stopping squid successfully."""
        mock_run.return_value = Mock(returncode=0)

        result = stop_squid()

        assert result is True

    @patch("atomate2.siesta.cli.cluster.proxy_utils.subprocess.run")
    def test_stop_squid_not_running(self, mock_run):
        """Test stopping squid when not running."""
        # First call (squid -k shutdown) fails, second call (killall) fails too
        mock_run.side_effect = [
            Mock(returncode=1),  # squid -k shutdown
            Mock(returncode=1),  # killall squid
        ]

        result = stop_squid()

        assert result is False

    def test_get_squid_status_not_installed(self):
        """Test getting squid status when not installed."""
        with patch(
            "atomate2.siesta.cli.cluster.proxy_utils.is_squid_installed",
            return_value=False,
        ):
            with patch(
                "atomate2.siesta.cli.cluster.proxy_utils.is_squid_running",
                return_value=False,
            ):
                status = get_squid_status(3129)

                assert status["installed"] is False
                assert status["running"] is False
                assert status["port"] == 3129
                assert status["proxy_url"] is None

    def test_get_squid_status_running(self):
        """Test getting squid status when running."""
        with patch(
            "atomate2.siesta.cli.cluster.proxy_utils.is_squid_installed",
            return_value=True,
        ):
            with patch(
                "atomate2.siesta.cli.cluster.proxy_utils.is_squid_running",
                return_value=True,
            ):
                status = get_squid_status(3129)

                assert status["installed"] is True
                assert status["running"] is True
                assert status["proxy_url"] == "http://127.0.0.1:3129"


class TestProxyFunctions:
    """Tests for proxy configuration and detection."""

    @patch("atomate2.siesta.cli.cluster.proxy_utils.run_ssh_command")
    def test_detect_proxy_found(self, mock_ssh):
        """Test detecting proxy on remote cluster."""
        mock_ssh.return_value = (0, "http://proxy.cluster.edu:8080\n", "")

        proxy_url = detect_proxy_on_remote("host", "user", None, None, False)

        assert proxy_url == "http://proxy.cluster.edu:8080"

    @patch("atomate2.siesta.cli.cluster.proxy_utils.run_ssh_command")
    def test_detect_proxy_not_found(self, mock_ssh):
        """Test when no proxy is detected."""
        mock_ssh.return_value = (0, "", "")

        proxy_url = detect_proxy_on_remote("host", "user", None, None, False)

        assert proxy_url is None

    @patch("atomate2.siesta.cli.cluster.proxy_utils.run_ssh_command")
    def test_configure_proxy_success(self, mock_ssh):
        """Test successful proxy configuration."""
        mock_ssh.return_value = (0, "", "")

        result = configure_proxy_on_remote(
            "host", "user", "http://proxy:8080", None, None, False, False
        )

        assert result is True
        # Should have created both .condarc and pip.conf
        assert mock_ssh.call_count == 2

    @patch("atomate2.siesta.cli.cluster.proxy_utils.run_ssh_command")
    def test_configure_proxy_failure(self, mock_ssh):
        """Test proxy configuration failure."""
        mock_ssh.return_value = (1, "", "Permission denied")

        result = configure_proxy_on_remote(
            "host", "user", "http://proxy:8080", None, None, False, False
        )

        assert result is False

    @patch("atomate2.siesta.cli.cluster.proxy_utils.run_ssh_command")
    def test_add_proxy_to_bashrc_new(self, mock_ssh):
        """Test adding proxy exports to bashrc (first time)."""
        # First call checks if proxy exists, second adds it
        mock_ssh.side_effect = [
            (0, "not_exists", ""),  # Check command
            (0, "", ""),  # Append command
        ]

        result = add_proxy_to_remote_bashrc(
            "host", "user", "http://127.0.0.1:3129", None, None, False, False
        )

        assert result is True

    @patch("atomate2.siesta.cli.cluster.proxy_utils.run_ssh_command")
    def test_add_proxy_to_bashrc_existing(self, mock_ssh):
        """Test adding proxy exports when already exist."""
        mock_ssh.return_value = (0, "exists", "")

        result = add_proxy_to_remote_bashrc(
            "host", "user", "http://127.0.0.1:3129", None, None, False, False
        )

        assert result is True
        # Should only check, not append
        assert mock_ssh.call_count == 1

    def test_is_proxy_error_true(self):
        """Test detecting proxy-related errors."""
        error_messages = [
            "CondaHTTPError: HTTP 000 CONNECTION FAILED",
            "Connection refused",
            "Max retries exceeded",
            "repo.anaconda.com blocked",
        ]

        for error in error_messages:
            assert is_proxy_error(error) is True

    def test_is_proxy_error_false(self):
        """Test non-proxy errors."""
        error = "ValueError: invalid input"
        assert is_proxy_error(error) is False

    def test_show_proxy_error_help(self, capsys):
        """Test proxy error help message."""
        show_proxy_error_help("http://proxy:8080")

        captured = capsys.readouterr()
        assert "Internet connection failed" in captured.out
        assert "proxy" in captured.out.lower()


class TestCLICommands:
    """Tests for Click CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_info_command(self):
        """Test info command."""
        result = self.runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert "Remote Cluster Setup" in result.output
        assert "Commands:" in result.output
        assert "Examples:" in result.output

    @patch("atomate2.siesta.cli.cluster.commands.is_squid_installed")
    @patch("atomate2.siesta.cli.cluster.commands.is_squid_running")
    def test_squid_status_command(self, mock_running, mock_installed):
        """Test squid status command."""
        mock_installed.return_value = True
        mock_running.return_value = True

        result = self.runner.invoke(cli, ["squid", "status"])

        assert result.exit_code == 0
        assert "Squid Status" in result.output
        assert "Installed" in result.output

    @patch("atomate2.siesta.cli.cluster.commands.is_squid_installed")
    def test_squid_install_already_installed(self, mock_installed):
        """Test squid install when already installed."""
        mock_installed.return_value = True

        with patch("atomate2.siesta.cli.cluster.commands.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0, stdout="Squid Cache Version 5.9\n"
            )

            result = self.runner.invoke(cli, ["squid", "install"])

            assert result.exit_code == 0
            assert "already installed" in result.output

    @patch("atomate2.siesta.cli.cluster.commands.is_squid_installed")
    @patch("atomate2.siesta.cli.cluster.commands.start_squid")
    def test_squid_start_command(self, mock_start, mock_installed):
        """Test squid start command."""
        mock_installed.return_value = True
        mock_start.return_value = True

        result = self.runner.invoke(cli, ["squid", "start"])

        assert result.exit_code == 0
        assert "Squid is running" in result.output

    @patch("atomate2.siesta.cli.cluster.commands.stop_squid")
    def test_squid_stop_command(self, mock_stop):
        """Test squid stop command."""
        mock_stop.return_value = True

        result = self.runner.invoke(cli, ["squid", "stop"])

        assert result.exit_code == 0
        assert "stopped" in result.output

    @patch("atomate2.siesta.cli.cluster.commands.stop_squid")
    @patch("atomate2.siesta.cli.cluster.commands.start_squid")
    def test_squid_restart_command(self, mock_start, mock_stop):
        """Test squid restart command."""
        mock_stop.return_value = True
        mock_start.return_value = True

        result = self.runner.invoke(cli, ["squid", "restart"])

        assert result.exit_code == 0
        assert "restarted" in result.output

    def test_squid_start_custom_port(self):
        """Test squid start with custom port."""
        with patch("atomate2.siesta.cli.cluster.commands.start_squid") as mock_start:
            mock_start.return_value = True

            result = self.runner.invoke(cli, ["squid", "start", "--port", "8080"])

            assert result.exit_code == 0
            mock_start.assert_called_with(8080, remove_old_config=False)

    def test_status_command_missing_host(self):
        """Test status command requires --host."""
        result = self.runner.invoke(cli, ["status"])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "Error" in result.output

    @patch("atomate2.siesta.cli.cluster.commands.run_ssh_command")
    @patch("atomate2.siesta.cli.cluster.commands.getpass.getuser")
    def test_status_command_connection_failure(self, mock_getuser, mock_ssh):
        """Test status command with SSH connection failure."""
        mock_getuser.return_value = "testuser"
        mock_ssh.return_value = (1, "", "Connection refused")

        result = self.runner.invoke(cli, ["status", "--host", "test.edu"])

        assert result.exit_code == 1
        assert "SSH connection failed" in result.output


class TestStartSquidFunction:
    """Tests for start_squid function with file creation."""

    @patch("atomate2.siesta.cli.cluster.proxy_utils.is_squid_running")
    @patch("atomate2.siesta.cli.cluster.proxy_utils.is_squid_installed")
    @patch("atomate2.siesta.cli.cluster.proxy_utils.subprocess.run")
    def test_start_squid_success(
        self, mock_run, mock_installed, mock_running, tmp_path
    ):
        """Test starting squid with temporary config file."""

        mock_installed.return_value = True
        mock_running.side_effect = [
            False,
            True,
        ]  # Not running, then running after start
        mock_run.return_value = Mock(returncode=0)

        # Mock the config directory to use tmp_path and time.sleep
        with patch("os.path.expanduser") as mock_expand:
            with patch("time.sleep"):  # Mock time.sleep from time module
                config_dir = tmp_path / ".atomate2siesta-cluster"
                config_dir.mkdir()
                mock_expand.return_value = str(tmp_path / ".atomate2siesta-cluster")

                result = start_squid(3129, remove_old_config=False)

                assert result is True
                # Check that config file was created
                config_file = config_dir / "squid.conf"
                assert config_file.exists()
                content = config_file.read_text()
                assert "http_port 127.0.0.1:3129" in content

    @patch("atomate2.siesta.cli.cluster.proxy_utils.is_squid_installed")
    def test_start_squid_not_installed(self, mock_installed):
        """Test start_squid when squid is not installed."""
        mock_installed.return_value = False

        result = start_squid(3129)

        assert result is False

    @patch("atomate2.siesta.cli.cluster.proxy_utils.is_squid_running")
    @patch("atomate2.siesta.cli.cluster.proxy_utils.is_squid_installed")
    def test_start_squid_already_running(self, mock_installed, mock_running):
        """Test start_squid when squid is already running."""
        mock_installed.return_value = True
        mock_running.return_value = True

        result = start_squid(3129)

        assert result is True


class TestSetupCommandIntegration:
    """Integration tests for setup command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_setup_missing_host(self):
        """Test that setup command requires --host."""
        result = self.runner.invoke(cli, ["setup"])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "--host" in result.output

    @patch("atomate2.siesta.cli.cluster.commands.getpass.getuser")
    @patch("atomate2.siesta.cli.cluster.commands.Confirm.ask")
    def test_setup_user_cancels(self, mock_confirm, mock_getuser):
        """Test setup when user cancels at confirmation."""
        mock_getuser.return_value = "testuser"
        mock_confirm.return_value = False  # User says no

        result = self.runner.invoke(cli, ["setup", "--host", "test.edu"])

        assert result.exit_code == 0
        assert "cancelled" in result.output.lower()
