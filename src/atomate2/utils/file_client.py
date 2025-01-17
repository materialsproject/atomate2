"""Tools for remote file IO using paramiko."""

from __future__ import annotations

import errno
import os
import shutil
import stat
import warnings
from functools import wraps
from glob import glob
from gzip import GzipFile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import paramiko
from monty.io import zopen
from paramiko import SFTPClient, SSHClient

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType


class FileClient:
    """
    Tool for performing operations on files.

    The client is agnostic of whether file operations are happening locally or via SSH.
    All operations have a ``host`` parameter that specifies which remote host the
    operation should be performed on.

    .. Note::
        To use abbreviated host names without user information, the FileClient requires
        the appropriate configuration to be defined in the ssh config file.

    Parameters
    ----------
    key_filename : str or Path
        Path to private key file (for remote connections only).
    config_filename : str or Path
        Path to OpenSSH config file defining host connection settings.
    """

    def __init__(
        self,
        key_filename: str | Path = "~/.ssh/id_rsa",
        config_filename: str | Path = "~/.ssh/config",
    ) -> None:
        self.key_filename = key_filename
        self.config_filename = config_filename

        self.connections: dict[str, dict[str, Any]] = {}

    def connect(self, host: str) -> None:
        """
        Connect to a remote host.

        Parameters
        ----------
        host : str
            A remote host filesystem. Supports using hosts defined in the ssh
            config file. The host can be specified as either "username@remote_host" or
            just "remote_host" in which case the username will be inferred from the
            current user.
        """
        if host in self.connections:
            return

        if "@" in host:
            username, hostname = host.split("@", 1)
        else:
            username = None  # paramiko sets default username
            hostname = host

        ssh = get_ssh_connection(
            username, hostname, self.key_filename, self.config_filename
        )
        self.connections[host] = {"ssh": ssh, "sftp": ssh.open_sftp()}

    def get_ssh(self, host: str) -> SSHClient:
        """Get an SSH connection to a host.

        Parameters
        ----------
        host : str
            A remote host filesystem. Supports using hosts defined in the ssh
            config file. The host can be specified as either "username@remote_host" or
            just "remote_host" in which case the username will be inferred from the
            current user.

        Returns
        -------
        .SSHClient
            An ssh client to the host.
        """
        if host not in self.connections:
            self.connect(host)
        return self.connections[host]["ssh"]

    def get_sftp(self, host: str) -> SFTPClient:
        """Get an SFTP connection to a host.

        Parameters
        ----------
        host : str
            A remote host filesystem. Supports using hosts defined in the ssh
            config file. The host can be specified as either "username@remote_host" or
            just "remote_host" in which case the username will be inferred from the
            current user.

        Returns
        -------
        .SFTPClient
            An sftp client to the host.
        """
        if host not in self.connections:
            self.connect(host)
        return self.connections[host]["sftp"]

    def exists(self, path: str | Path, host: str | None = None) -> bool:
        """
        Check whether a file exists.

        Parameters
        ----------
        path : str or Path
            A path to check existence of.
        host : str or None
            A remote file system host on which to perform file operations.

        Returns
        -------
        bool
            Whether the file exists.
        """
        if host is None:
            return Path(path).exists()
        path = str(self.abspath(path, host=host))
        try:
            self.get_sftp(host).stat(path)
        except FileNotFoundError:
            return False
        return True

    def is_file(self, path: str | Path, host: str | None = None) -> bool:
        """Whether a path is a file.

        Parameters
        ----------
        path : str or Path
            A path.
        host : str or None
            A remote file system host on which to perform file operations.

        Returns
        -------
        bool
            Whether the path is a file.
        """
        if host is None:
            return Path(path).is_file()
        path = str(self.abspath(path, host=host))
        try:
            return stat.S_ISREG(self.get_sftp(host).lstat(path).st_mode)
        except FileNotFoundError:
            return False

    def is_dir(self, path: str | Path, host: str | None = None) -> bool:
        """Whether a path is a directory.

        Parameters
        ----------
        path : str or Path
            A path.
        host : str or None
            A remote file system host on which to perform file operations.

        Returns
        -------
        bool
            Whether the path is a directory.
        """
        if host is None:
            return Path(path).is_dir()
        path = str(self.abspath(path, host=host))
        try:
            return stat.S_ISDIR(self.get_sftp(host).lstat(path).st_mode)
        except FileNotFoundError:
            return False

    def listdir(self, path: str | Path, host: str | None = None) -> list[Path]:
        """Get the directory listing.

        Parameters
        ----------
        path : str or Path
            Full path to the directory.
        host : str or None
            A remote file system host on which to perform file operations.

        Returns
        -------
        list of Path
            List of filenames and directories.
        """
        if host is None:
            path = self.abspath(path, host=host)
            return [p.relative_to(path) for p in Path(path).iterdir()]

        path = str(self.abspath(path, host=host))
        return [Path(p) for p in self.get_sftp(host).listdir(path)]

    def copy(
        self,
        src_filename: str | Path,
        dest_filename: str | Path,
        src_host: str | None = None,
        dest_host: str | None = None,
    ) -> None:
        """
        Copy a file from source to destination.

        Parameters
        ----------
        src_filename : str or Path
            Full path to source file.
        dest_filename : str or Path
            Full path to destination file.
        src_host : str or None
            A remote file system host for the source file.
        dest_host : str or None
            A remote file system host for the destination file.
        """
        src_filename = self.abspath(src_filename, host=src_host)
        dest_filename = self.abspath(dest_filename, host=dest_host)

        if src_host is None and dest_host is None:
            # copying on local machine
            shutil.copy2(src_filename, dest_filename)
        elif src_host is not None and dest_host is None:
            # copying from remote to local
            self.get_sftp(src_host).get(str(src_filename), str(dest_filename))
        elif src_host is None and dest_host is not None:
            # copying from local to remote
            self.get_sftp(dest_host).put(str(src_filename), str(dest_filename))
        elif src_host == dest_host:
            # copying between the same remote machine.
            ssh = self.get_ssh(src_host)
            _, _, stderr = ssh.exec_command(f"cp {src_filename} {dest_filename}")
            if len(stderr.readlines()) > 0:
                warnings.warn(f"Copy command gave error: {stderr}", stacklevel=2)
        else:
            # copying between two remote hosts; this is a pain and it is unlikely anyone
            # will want to do it.
            raise ValueError(
                "Copying between two different remote hosts is not supported."
            )

    def link(
        self,
        src_filename: str | Path,
        dest_filename: str | Path,
    ) -> None:
        """
        Link a file from source to destination.

        Parameters
        ----------
        src_filename : str or Path
            Full path to source file.
        dest_filename : str or Path
            Full path to destination file.
        """
        try:
            os.symlink(src_filename, dest_filename)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                os.remove(dest_filename)
                os.symlink(src_filename, dest_filename)
            else:
                raise

    def remove(self, path: str | Path, host: str | None = None) -> None:
        """
        Remove a file (does not work on directories).

        Parameters
        ----------
        path : str or Path
            Path to a file.
        host : str or None
            A remote file system host on which to perform file operations.
        """
        if host is None:
            Path(path).unlink()
        else:
            path = str(self.abspath(host, host=host))
            self.get_sftp(host).unlink(path)

    def rename(
        self,
        old_path: str | Path,
        new_path: str | Path,
        host: str | None = None,
    ) -> None:
        """
        Rename (move) a file.

        Parameters
        ----------
        old_path : str or Path
            Path to an existing file.
        new_path : str or Path
            Requested path to new file.
        host : str or None
            A remote file system host on which to perform file operations.
        """
        if host is None:
            Path(old_path).rename(new_path)
        else:
            old_path = str(self.abspath(old_path, host=host))
            new_path = str(self.abspath(new_path, host=host))
            self.get_sftp(host).rename(old_path, new_path)

    def abspath(self, path: str | Path, host: str | None = None) -> Path:
        """Get the absolute path.

        Parameters
        ----------
        path : str or Path
            A path to a file or directory.
        host : str or None
            A remote file system host on which to perform file operations.

        Returns
        -------
        Path
            The absolute file path.
        """
        if host is None:
            return Path(path).absolute()
        ssh = self.get_ssh(host)
        _, stdout, _ = ssh.exec_command(f"readlink -f {path}")
        return Path(next(o.split("\n")[0] for o in stdout))

    def glob(self, path: str | Path, host: str | None = None) -> list[Path]:
        """
        Glob files and folders.

        Parameters
        ----------
        path : str or Path
            A path to glob.
        host : str or None
            A remote file system host on which to perform file operations.

        Returns
        -------
        list[Path]
            A list of globs files and directories.
        """
        if host is None:
            files = glob(str(path))
        else:
            ssh = self.get_ssh(host)
            _, stdout, _ = ssh.exec_command(f"readlink -f {path}")
            files = [o.split("\n")[0] for o in stdout]

        return [Path(f) for f in files]

    def gzip(
        self,
        path: str | Path,
        host: str | None = None,
        compresslevel: int = 6,
        force: bool | str = False,
    ) -> None:
        """
        Gzip a file.

        Parameters
        ----------
        path : str or Path
            Path to a file to gzip.
        host : str or None
            A remote file system host on which to perform file operations.
        compresslevel : bool
            Level of compression, 1-9. 9 is default for GzipFile, 6 is default for gzip.
        force : bool
            How to handle writing a gzipped file if it already exists. Accepts
            either a string or bool:

            - `"force"` or `True`: Overwrite gzipped file if it already exists.
            - `"raise"` or `False`: Raise an error if file already exists.
            - `"skip"` Skip file if it already exists.
        """
        path = self.abspath(path, host=host)
        path_gz = path.parent / f"{path.name}.gz"

        if str(path).lower().endswith("gz"):
            warnings.warn(f"{path} is already gzipped, skipping...", stacklevel=1)
            return

        if self.is_dir(path, host=host):
            warnings.warn(f"{path} is a directory, skipping...", stacklevel=1)
            return

        if self.exists(path_gz, host=host):
            if force is False or force == "raise":
                raise FileExistsError(f"{path_gz} file already exists")
            if force is True or force == "force":
                pass
            elif force == "skip":
                warnings.warn(
                    f"{path_gz} file already exists, skipping...", stacklevel=2
                )
                return
            else:
                raise ValueError(
                    f"Invalid value for force: {force} "
                    "(must be True, False, 'raise', 'force', or 'skip'))"
                )

        if host is None:
            with (
                open(path, "rb") as f_in,
                GzipFile(path_gz, "wb", compresslevel=compresslevel) as f_out,
            ):
                shutil.copyfileobj(f_in, f_out)
            shutil.copystat(path, path_gz)
            path.unlink()
        else:
            ssh = self.get_ssh(host)
            _, _stdout, _ = ssh.exec_command(f"gzip -f {path!s}")

    def gunzip(
        self,
        path: str | Path,
        host: str | None = None,
        force: bool | str = False,
    ) -> None:
        """
        Ungzip a file.

        Parameters
        ----------
        path : str or Path
            Path to a file to gzip.
        host : str or None
            A remote file system host on which to perform file operations.
        force : bool
            How to handle writing a non-gzipped file if it already exists. Accepts
            either a string or bool:

            - `"force"` or `True`: Overwrite non-gzipped file if it already exists.
            - `"raise"` or `False`: Raise an error if file already exists.
            - `"skip"` Skip file if it already exists.
        """
        path = self.abspath(path, host=host)
        path_nongz = path.with_suffix("")

        if not str(path).lower().endswith("gz"):
            warnings.warn(f"{path} is not gzipped, skipping...", stacklevel=2)
            return

        if self.exists(path_nongz, host=host):
            if force is False or force == "raise":
                raise FileExistsError(f"{path_nongz} file already exists")
            if force is True or force == "force":
                pass
            elif force == "skip":
                warnings.warn(
                    f"{path_nongz} file already exists, skipping...", stacklevel=2
                )
                return
            else:
                raise ValueError(
                    f"Invalid value for force: {force} "
                    "(must be True, False, 'raise', 'force', or 'skip'))"
                )

        if host is None:
            with open(path_nongz, "wb") as f_out, zopen(path, "rb") as f_in:
                f_out.writelines(f_in)
            path.unlink()
        else:
            ssh = self.get_ssh(host)
            _stdin, _stdout, _stderr = ssh.exec_command(f"gunzip -f {path!s}")

    def close(self) -> None:
        """Close all connections."""
        for connection in self.connections.values():
            connection["ssh"].close()
            connection["sftp"].close()
        self.connections = {}

    def __enter__(self) -> FileClient:  # noqa: PYI034
        """Support for "with" context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Support for "with" context."""
        self.close()


def get_ssh_connection(
    username: str | None,
    hostname: str,
    key_filename: str | Path,
    config_filename: str | Path | None = None,
) -> SSHClient:
    """
    Connect to a remote host via paramiko.

    If the host key is not present it will be added automatically.

    Parameters
    ----------
    username : str or None
        The username. If ``None``, the current logged in username will be used.
    hostname : str
        The host name. Supports host aliases defined in the ssh config file.
    key_filename : str or Path
        Path to private key file.
    config_filename : str or Path or None
        Path to OpenSSH config file.

    Returns
    -------
    .SSHClient
        An ssh connection to the host.
    """
    key_filename = Path(key_filename).expanduser()
    if not key_filename.exists():
        raise ValueError(f"Cannot locate private key file: {key_filename}")

    config: dict[str, Any] = {"hostname": hostname, "username": username}
    config_filename = Path(config_filename).expanduser()
    if Path(config_filename).exists():
        # try reading ssh config file
        ssh_config = paramiko.SSHConfig().from_path(str(config_filename))

        host_config = ssh_config.lookup(hostname)  # type: ignore[attr-defined]
        for key in ("hostname", "user", "port"):
            if key in host_config:
                config[key.replace("user", "username")] = host_config[key]

        if "proxycommand" in host_config:
            config["sock"] = paramiko.ProxyCommand(host_config["proxycommand"])

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(**config)
    return client


def auto_fileclient(method: Callable | None = None) -> Callable:
    """
    Automatically pass a FileClient to the function if not already present in kwargs.

    This decorator should only be applied to functions with a ``file_client`` keyword
    argument. If a custom file client is not supplied when the function is called, it
    will automatically create a new FileClient, add it to the function arguments and
    close the file client connects at the end of the function.

    Parameters
    ----------
    method : callable or None
        A function to wrap. This should not be specified directly and is implied
        by the decorator.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def gen_file_client(*args, **kwargs) -> Any:
            file_client = kwargs.get("file_client")
            if file_client is None:
                with FileClient() as file_client:
                    kwargs["file_client"] = file_client
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return gen_file_client

    # See if we're being called as @auto_fileclient or @auto_fileclient().
    if method is None:
        # We're called with parens.
        return decorator

    # We're called as @auto_fileclient without parens.
    return decorator(method)
