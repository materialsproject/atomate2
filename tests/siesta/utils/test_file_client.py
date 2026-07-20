"""
Tests for file_client.py (remote and local file operations).

These tests validate:
- FileClient initialization and connection management
- Local file operations (host=None)
- Remote file operations (mocked SSH/SFTP)
- File copying scenarios
- Compression/decompression
- Context manager support
- Utility functions and decorators
"""

import warnings
from unittest.mock import MagicMock, patch

import pytest

from atomate2.siesta.utils.file_client import (
    FileClient,
    auto_fileclient,
    get_ssh_connection,
)


class TestFileClientInit:
    """Tests for FileClient initialization."""

    def test_file_client_creation(self):
        """Test creating FileClient with default parameters."""
        client = FileClient()

        assert client.key_filename == "~/.ssh/id_rsa"
        assert client.config_filename == "~/.ssh/config"
        assert client.connections == {}

    def test_file_client_custom_parameters(self):
        """Test creating FileClient with custom parameters."""
        client = FileClient(
            key_filename="~/.ssh/custom_key", config_filename="~/.ssh/custom_config"
        )

        assert client.key_filename == "~/.ssh/custom_key"
        assert client.config_filename == "~/.ssh/custom_config"

    def test_file_client_connections_dict(self):
        """Test that connections dictionary is initialized empty."""
        client = FileClient()

        assert isinstance(client.connections, dict)
        assert len(client.connections) == 0


class TestFileClientConnection:
    """Tests for FileClient connection management."""

    @patch("atomate2.siesta.utils.file_client.get_ssh_connection")
    def test_connect_new_host(self, mock_get_ssh):
        """Test connecting to a new host."""
        client = FileClient()
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp
        mock_get_ssh.return_value = mock_ssh

        client.connect("testhost")

        assert "testhost" in client.connections
        assert client.connections["testhost"]["ssh"] == mock_ssh
        assert client.connections["testhost"]["sftp"] == mock_sftp
        mock_get_ssh.assert_called_once()

    @patch("atomate2.siesta.utils.file_client.get_ssh_connection")
    def test_connect_existing_host(self, mock_get_ssh):
        """Test connecting to an already connected host."""
        client = FileClient()
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp
        mock_get_ssh.return_value = mock_ssh

        client.connect("testhost")
        client.connect("testhost")  # Should not reconnect

        # Should only call get_ssh_connection once
        mock_get_ssh.assert_called_once()

    @patch("atomate2.siesta.utils.file_client.get_ssh_connection")
    def test_connect_with_username(self, mock_get_ssh):
        """Test connecting with username@host format."""
        client = FileClient()
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp
        mock_get_ssh.return_value = mock_ssh

        client.connect("user@testhost")

        # Should extract username and hostname
        assert "user@testhost" in client.connections
        mock_get_ssh.assert_called_once()

    @patch("atomate2.siesta.utils.file_client.get_ssh_connection")
    def test_get_ssh(self, mock_get_ssh):
        """Test getting SSH connection."""
        client = FileClient()
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp
        mock_get_ssh.return_value = mock_ssh

        ssh = client.get_ssh("testhost")

        assert ssh == mock_ssh
        assert "testhost" in client.connections

    @patch("atomate2.siesta.utils.file_client.get_ssh_connection")
    def test_get_sftp(self, mock_get_ssh):
        """Test getting SFTP connection."""
        client = FileClient()
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp
        mock_get_ssh.return_value = mock_ssh

        sftp = client.get_sftp("testhost")

        assert sftp == mock_sftp
        assert "testhost" in client.connections


class TestFileClientLocalOperations:
    """Tests for FileClient local file operations (host=None)."""

    def test_exists_local_file(self, tmp_path):
        """Test checking existence of local file."""
        client = FileClient()
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        assert client.exists(test_file, host=None) is True

    def test_exists_local_missing(self, tmp_path):
        """Test checking existence of missing local file."""
        client = FileClient()
        test_file = tmp_path / "missing.txt"

        assert client.exists(test_file, host=None) is False

    def test_is_file_local(self, tmp_path):
        """Test checking if local path is file."""
        client = FileClient()
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        assert client.is_file(test_file, host=None) is True

    def test_is_file_local_directory(self, tmp_path):
        """Test checking if local directory is not a file."""
        client = FileClient()

        assert client.is_file(tmp_path, host=None) is False

    def test_is_dir_local(self, tmp_path):
        """Test checking if local path is directory."""
        client = FileClient()

        assert client.is_dir(tmp_path, host=None) is True

    def test_is_dir_local_file(self, tmp_path):
        """Test checking if local file is not a directory."""
        client = FileClient()
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        assert client.is_dir(test_file, host=None) is False

    def test_listdir_local(self, tmp_path):
        """Test listing local directory."""
        client = FileClient()
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt").write_text("content2")
        (tmp_path / "subdir").mkdir()

        listing = client.listdir(tmp_path, host=None)

        assert len(listing) == 3
        names = [p.name for p in listing]
        assert "file1.txt" in names
        assert "file2.txt" in names
        assert "subdir" in names

    def test_abspath_local(self, tmp_path):
        """Test getting absolute path locally."""
        client = FileClient()
        test_file = tmp_path / "test.txt"

        abspath = client.abspath(test_file, host=None)

        assert abspath.is_absolute()
        assert str(test_file) in str(abspath)

    def test_remove_local(self, tmp_path):
        """Test removing local file."""
        client = FileClient()
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        client.remove(test_file, host=None)

        assert not test_file.exists()

    def test_rename_local(self, tmp_path):
        """Test renaming local file."""
        client = FileClient()
        old_file = tmp_path / "old.txt"
        new_file = tmp_path / "new.txt"
        old_file.write_text("content")

        client.rename(old_file, new_file, host=None)

        assert not old_file.exists()
        assert new_file.exists()
        assert new_file.read_text() == "content"

    def test_glob_local(self, tmp_path):
        """Test globbing local files."""
        client = FileClient()
        (tmp_path / "test1.txt").write_text("content1")
        (tmp_path / "test2.txt").write_text("content2")
        (tmp_path / "other.dat").write_text("content3")

        results = client.glob(str(tmp_path / "*.txt"), host=None)

        assert len(results) == 2
        names = [p.name for p in results]
        assert "test1.txt" in names
        assert "test2.txt" in names


class TestFileClientRemoteOperations:
    """Tests for FileClient remote file operations (mocked)."""

    @patch("atomate2.siesta.utils.file_client.get_ssh_connection")
    def test_exists_remote_file(self, mock_get_ssh):
        """Test checking existence of remote file."""
        client = FileClient()
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp
        mock_ssh.exec_command.return_value = (None, MagicMock(), None)
        mock_sftp.stat.return_value = MagicMock()
        mock_get_ssh.return_value = mock_ssh

        # Mock exec_command for abspath
        stdout = MagicMock()
        stdout.__iter__ = lambda self: iter(["/remote/path/file.txt\n"])
        mock_ssh.exec_command.return_value = (None, stdout, None)

        result = client.exists("/remote/path/file.txt", host="testhost")

        assert result is True

    @patch("atomate2.siesta.utils.file_client.get_ssh_connection")
    def test_exists_remote_missing(self, mock_get_ssh):
        """Test checking existence of missing remote file."""
        client = FileClient()
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp
        mock_sftp.stat.side_effect = FileNotFoundError()
        mock_get_ssh.return_value = mock_ssh

        # Mock exec_command for abspath
        stdout = MagicMock()
        stdout.__iter__ = lambda self: iter(["/remote/path/missing.txt\n"])
        mock_ssh.exec_command.return_value = (None, stdout, None)

        result = client.exists("/remote/path/missing.txt", host="testhost")

        assert result is False

    @patch("atomate2.siesta.utils.file_client.get_ssh_connection")
    def test_is_file_remote(self, mock_get_ssh):
        """Test checking if remote path is file."""
        client = FileClient()
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp

        # Mock lstat to return regular file mode
        mock_stat = MagicMock()
        mock_stat.st_mode = 0o100644  # Regular file
        mock_sftp.lstat.return_value = mock_stat
        mock_get_ssh.return_value = mock_ssh

        # Mock exec_command for abspath
        stdout = MagicMock()
        stdout.__iter__ = lambda self: iter(["/remote/path/file.txt\n"])
        mock_ssh.exec_command.return_value = (None, stdout, None)

        result = client.is_file("/remote/path/file.txt", host="testhost")

        assert result is True

    @patch("atomate2.siesta.utils.file_client.get_ssh_connection")
    def test_is_dir_remote(self, mock_get_ssh):
        """Test checking if remote path is directory."""
        client = FileClient()
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp

        # Mock lstat to return directory mode
        mock_stat = MagicMock()
        mock_stat.st_mode = 0o040755  # Directory
        mock_sftp.lstat.return_value = mock_stat
        mock_get_ssh.return_value = mock_ssh

        # Mock exec_command for abspath
        stdout = MagicMock()
        stdout.__iter__ = lambda self: iter(["/remote/path\n"])
        mock_ssh.exec_command.return_value = (None, stdout, None)

        result = client.is_dir("/remote/path", host="testhost")

        assert result is True

    @patch("atomate2.siesta.utils.file_client.get_ssh_connection")
    def test_listdir_remote(self, mock_get_ssh):
        """Test listing remote directory."""
        client = FileClient()
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp
        mock_sftp.listdir.return_value = ["file1.txt", "file2.txt", "subdir"]
        mock_get_ssh.return_value = mock_ssh

        # Mock exec_command for abspath
        stdout = MagicMock()
        stdout.__iter__ = lambda self: iter(["/remote/path\n"])
        mock_ssh.exec_command.return_value = (None, stdout, None)

        listing = client.listdir("/remote/path", host="testhost")

        assert len(listing) == 3
        names = [p.name for p in listing]
        assert "file1.txt" in names
        assert "file2.txt" in names


class TestFileClientCopyOperations:
    """Tests for FileClient copy operations."""

    def test_copy_local_to_local(self, tmp_path):
        """Test copying between local paths."""
        client = FileClient()
        src = tmp_path / "src.txt"
        dest = tmp_path / "dest.txt"
        src.write_text("content")

        client.copy(src, dest, src_host=None, dest_host=None)

        assert dest.exists()
        assert dest.read_text() == "content"

    @patch("atomate2.siesta.utils.file_client.get_ssh_connection")
    def test_copy_remote_to_local(self, mock_get_ssh, tmp_path):
        """Test copying from remote to local."""
        client = FileClient()
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp
        mock_get_ssh.return_value = mock_ssh

        # Mock exec_command for abspath
        stdout = MagicMock()
        stdout.__iter__ = lambda self: iter(["/remote/src.txt\n"])
        mock_ssh.exec_command.return_value = (None, stdout, None)

        dest = tmp_path / "dest.txt"
        client.copy("/remote/src.txt", dest, src_host="testhost", dest_host=None)

        mock_sftp.get.assert_called_once()

    @patch("atomate2.siesta.utils.file_client.get_ssh_connection")
    def test_copy_local_to_remote(self, mock_get_ssh, tmp_path):
        """Test copying from local to remote."""
        client = FileClient()
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp
        mock_get_ssh.return_value = mock_ssh

        # Mock exec_command for abspath
        stdout = MagicMock()
        stdout.__iter__ = lambda self: iter(["/remote/dest.txt\n"])
        mock_ssh.exec_command.return_value = (None, stdout, None)

        src = tmp_path / "src.txt"
        src.write_text("content")

        client.copy(src, "/remote/dest.txt", src_host=None, dest_host="testhost")

        mock_sftp.put.assert_called_once()

    @patch("atomate2.siesta.utils.file_client.get_ssh_connection")
    def test_copy_remote_to_remote_same_host(self, mock_get_ssh):
        """Test copying between paths on same remote host."""
        client = FileClient()
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp

        # Mock exec_command for both abspath calls and cp command
        stdout = MagicMock()
        stdout.__iter__ = lambda self: iter(["/remote/src.txt\n"])
        stderr = MagicMock()
        stderr.readlines.return_value = []
        mock_ssh.exec_command.return_value = (None, stdout, stderr)
        mock_get_ssh.return_value = mock_ssh

        client.copy(
            "/remote/src.txt",
            "/remote/dest.txt",
            src_host="testhost",
            dest_host="testhost",
        )

        # Should use SSH exec_command with cp
        assert mock_ssh.exec_command.called

    @patch("atomate2.siesta.utils.file_client.get_ssh_connection")
    def test_copy_remote_to_remote_different_hosts(self, mock_get_ssh):
        """Test that copying between different remote hosts raises error."""
        client = FileClient()
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp
        mock_get_ssh.return_value = mock_ssh

        # Mock exec_command for abspath calls
        stdout = MagicMock()
        stdout.__iter__ = lambda self: iter(["/remote/src.txt\n"])
        mock_ssh.exec_command.return_value = (None, stdout, None)

        with pytest.raises(ValueError, match="not supported"):
            client.copy(
                "/remote/src.txt",
                "/remote/dest.txt",
                src_host="host1",
                dest_host="host2",
            )


class TestFileClientLink:
    """Tests for FileClient link operations."""

    def test_link_creates_symlink(self, tmp_path):
        """Test creating a symbolic link."""
        client = FileClient()
        src = tmp_path / "src.txt"
        dest = tmp_path / "link.txt"
        src.write_text("content")

        client.link(src, dest)

        assert dest.exists()
        assert dest.is_symlink()
        assert dest.read_text() == "content"

    def test_link_replaces_existing(self, tmp_path):
        """Test that link replaces existing symlink."""
        client = FileClient()
        src1 = tmp_path / "src1.txt"
        src2 = tmp_path / "src2.txt"
        dest = tmp_path / "link.txt"

        src1.write_text("content1")
        src2.write_text("content2")

        # Create initial link
        client.link(src1, dest)
        assert dest.read_text() == "content1"

        # Replace with new link
        client.link(src2, dest)
        assert dest.read_text() == "content2"


class TestFileClientCompression:
    """Tests for FileClient compression operations."""

    def test_gzip_local_file(self, tmp_path):
        """Test gzipping a local file."""
        client = FileClient()
        test_file = tmp_path / "test.txt"
        test_file.write_text("content to compress")

        client.gzip(test_file, host=None)

        assert not test_file.exists()
        assert (tmp_path / "test.txt.gz").exists()

    def test_gzip_already_gzipped(self, tmp_path):
        """Test gzipping an already gzipped file."""
        client = FileClient()
        test_file = tmp_path / "test.txt.gz"
        test_file.write_text("already compressed")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            client.gzip(test_file, host=None)
            assert len(w) == 1
            assert "already gzipped" in str(w[0].message)

    def test_gzip_directory(self, tmp_path):
        """Test gzipping a directory (should warn and skip)."""
        client = FileClient()
        test_dir = tmp_path / "testdir"
        test_dir.mkdir()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            client.gzip(test_dir, host=None)
            assert len(w) == 1
            assert "directory" in str(w[0].message)

    def test_gzip_force_overwrite(self, tmp_path):
        """Test gzipping with force=True to overwrite."""
        client = FileClient()
        test_file = tmp_path / "test.txt"
        test_gz = tmp_path / "test.txt.gz"

        test_file.write_text("content")
        test_gz.write_text("old gz content")

        client.gzip(test_file, host=None, force=True)

        assert test_gz.exists()
        assert not test_file.exists()

    def test_gzip_force_skip(self, tmp_path):
        """Test gzipping with force='skip'."""
        client = FileClient()
        test_file = tmp_path / "test.txt"
        test_gz = tmp_path / "test.txt.gz"

        test_file.write_text("content")
        test_gz.write_text("old gz content")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            client.gzip(test_file, host=None, force="skip")
            assert len(w) == 1
            assert "already exists" in str(w[0].message)

        assert test_file.exists()  # Should not remove original

    def test_gzip_force_raise(self, tmp_path):
        """Test gzipping with force=False raises error."""
        client = FileClient()
        test_file = tmp_path / "test.txt"
        test_gz = tmp_path / "test.txt.gz"

        test_file.write_text("content")
        test_gz.write_text("old gz content")

        with pytest.raises(FileExistsError):
            client.gzip(test_file, host=None, force=False)

    def test_gunzip_local_file(self, tmp_path):
        """Test gunzipping a local file."""
        import gzip

        client = FileClient()
        test_file = tmp_path / "test.txt"
        test_gz = tmp_path / "test.txt.gz"

        # Create a gzipped file
        with gzip.open(test_gz, "wt") as f:
            f.write("compressed content")

        client.gunzip(test_gz, host=None)

        assert not test_gz.exists()
        assert test_file.exists()
        assert "compressed content" in test_file.read_text()

    def test_gunzip_not_gzipped(self, tmp_path):
        """Test gunzipping a non-gzipped file."""
        client = FileClient()
        test_file = tmp_path / "test.txt"
        test_file.write_text("not compressed")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            client.gunzip(test_file, host=None)
            assert len(w) == 1
            assert "not gzipped" in str(w[0].message)

    def test_gunzip_force_skip(self, tmp_path):
        """Test gunzipping with force='skip'."""
        import gzip

        client = FileClient()
        test_file = tmp_path / "test.txt"
        test_gz = tmp_path / "test.txt.gz"

        test_file.write_text("existing content")
        with gzip.open(test_gz, "wt") as f:
            f.write("compressed content")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            client.gunzip(test_gz, host=None, force="skip")
            assert len(w) == 1
            assert "already exists" in str(w[0].message)


class TestFileClientContextManager:
    """Tests for FileClient context manager support."""

    def test_context_manager_enter_exit(self):
        """Test using FileClient as context manager."""
        with FileClient() as client:
            assert isinstance(client, FileClient)
            assert client.connections == {}

    @patch("atomate2.siesta.utils.file_client.get_ssh_connection")
    def test_context_manager_closes_connections(self, mock_get_ssh):
        """Test that context manager closes connections on exit."""
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp
        mock_get_ssh.return_value = mock_ssh

        with FileClient() as client:
            client.connect("testhost")
            assert len(client.connections) == 1

        # After exiting context, connections should be closed
        mock_ssh.close.assert_called_once()
        mock_sftp.close.assert_called_once()


class TestGetSSHConnection:
    """Tests for get_ssh_connection function."""

    @patch("atomate2.siesta.utils.file_client.paramiko.SSHClient")
    @patch("atomate2.siesta.utils.file_client.Path")
    def test_get_ssh_connection_basic(self, mock_path_class, mock_ssh_client):
        """Test basic SSH connection."""
        # Mock key file exists
        mock_key_path = MagicMock()
        mock_key_path.exists.return_value = True
        mock_key_path.expanduser.return_value = mock_key_path

        # Mock config file doesn't exist
        mock_config_path = MagicMock()
        mock_config_path.exists.return_value = False
        mock_config_path.expanduser.return_value = mock_config_path

        # Use a function to return different mocks based on the argument
        def path_side_effect(arg):
            if "id_rsa" in str(arg):
                return mock_key_path
            return mock_config_path

        mock_path_class.side_effect = path_side_effect

        mock_client = MagicMock()
        mock_ssh_client.return_value = mock_client

        result = get_ssh_connection(
            "user", "hostname", "~/.ssh/id_rsa", "~/.ssh/config"
        )

        assert result == mock_client
        mock_client.set_missing_host_key_policy.assert_called_once()
        mock_client.connect.assert_called_once()

    @patch("atomate2.siesta.utils.file_client.Path")
    def test_get_ssh_connection_missing_key(self, mock_path_class):
        """Test SSH connection with missing key file."""
        mock_key_path = MagicMock()
        mock_key_path.exists.return_value = False
        mock_key_path.expanduser.return_value = mock_key_path
        mock_path_class.return_value = mock_key_path

        with pytest.raises(ValueError, match="Cannot locate private key"):
            get_ssh_connection("user", "hostname", "~/.ssh/missing_key")


class TestAutoFileClient:
    """Tests for auto_fileclient decorator."""

    def test_auto_fileclient_without_parens(self):
        """Test decorator without parentheses."""

        @auto_fileclient
        def test_func(file_client=None):
            return file_client

        result = test_func()
        assert isinstance(result, FileClient)

    def test_auto_fileclient_with_parens(self):
        """Test decorator with parentheses."""

        @auto_fileclient()
        def test_func(file_client=None):
            return file_client

        result = test_func()
        assert isinstance(result, FileClient)

    def test_auto_fileclient_with_custom_client(self):
        """Test decorator with custom file client."""
        custom_client = FileClient()

        @auto_fileclient
        def test_func(file_client=None):
            return file_client

        result = test_func(file_client=custom_client)
        assert result is custom_client

    def test_auto_fileclient_passes_args_kwargs(self):
        """Test that decorator passes through args and kwargs."""

        @auto_fileclient
        def test_func(arg1, arg2, file_client=None, kwarg1=None):
            return (arg1, arg2, kwarg1, file_client)

        result = test_func("a", "b", kwarg1="c")

        assert result[0] == "a"
        assert result[1] == "b"
        assert result[2] == "c"
        assert isinstance(result[3], FileClient)


class TestFileClientEdgeCases:
    """Test edge cases for FileClient."""

    def test_close_with_no_connections(self):
        """Test closing client with no connections."""
        client = FileClient()
        client.close()  # Should not raise error

        assert client.connections == {}

    @patch("atomate2.siesta.utils.file_client.get_ssh_connection")
    def test_close_with_connections(self, mock_get_ssh):
        """Test closing client with active connections."""
        client = FileClient()
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp
        mock_get_ssh.return_value = mock_ssh

        client.connect("testhost")
        client.close()

        mock_ssh.close.assert_called_once()
        mock_sftp.close.assert_called_once()
        assert client.connections == {}

    def test_exists_missing_file_returns_false(self, tmp_path):
        """Test that exists returns False for missing file."""
        client = FileClient()
        result = client.exists(tmp_path / "nonexistent.txt", host=None)

        assert result is False

    def test_is_file_missing_returns_false(self, tmp_path):
        """Test that is_file returns False for missing file."""
        client = FileClient()
        result = client.is_file(tmp_path / "nonexistent.txt", host=None)

        assert result is False

    def test_is_dir_missing_returns_false(self, tmp_path):
        """Test that is_dir returns False for missing directory."""
        client = FileClient()
        result = client.is_dir(tmp_path / "nonexistent", host=None)

        assert result is False

    @patch("atomate2.siesta.utils.file_client.get_ssh_connection")
    def test_is_file_remote_missing(self, mock_get_ssh):
        """Test is_file with missing remote file."""
        client = FileClient()
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp
        mock_sftp.lstat.side_effect = FileNotFoundError()
        mock_get_ssh.return_value = mock_ssh

        # Mock exec_command for abspath
        stdout = MagicMock()
        stdout.__iter__ = lambda self: iter(["/remote/missing.txt\n"])
        mock_ssh.exec_command.return_value = (None, stdout, None)

        result = client.is_file("/remote/missing.txt", host="testhost")

        assert result is False

    @patch("atomate2.siesta.utils.file_client.get_ssh_connection")
    def test_is_dir_remote_missing(self, mock_get_ssh):
        """Test is_dir with missing remote directory."""
        client = FileClient()
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp
        mock_sftp.lstat.side_effect = FileNotFoundError()
        mock_get_ssh.return_value = mock_ssh

        # Mock exec_command for abspath
        stdout = MagicMock()
        stdout.__iter__ = lambda self: iter(["/remote/missing\n"])
        mock_ssh.exec_command.return_value = (None, stdout, None)

        result = client.is_dir("/remote/missing", host="testhost")

        assert result is False

    def test_gzip_invalid_force_value(self, tmp_path):
        """Test gzip with invalid force value."""
        client = FileClient()
        test_file = tmp_path / "test.txt"
        test_gz = tmp_path / "test.txt.gz"

        test_file.write_text("content")
        test_gz.write_text("existing")

        with pytest.raises(ValueError, match="Invalid value for force"):
            client.gzip(test_file, host=None, force="invalid")

    def test_gunzip_invalid_force_value(self, tmp_path):
        """Test gunzip with invalid force value."""
        import gzip

        client = FileClient()
        test_file = tmp_path / "test.txt"
        test_gz = tmp_path / "test.txt.gz"

        test_file.write_text("existing")
        with gzip.open(test_gz, "wt") as f:
            f.write("compressed")

        with pytest.raises(ValueError, match="Invalid value for force"):
            client.gunzip(test_gz, host=None, force="invalid")
