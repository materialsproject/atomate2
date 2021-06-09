"""Tools for remote file IO using paramiko."""

from __future__ import annotations

import typing
import warnings

if typing.TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Callable, Dict, List, Optional, Union

    from paramiko import SFTPClient, SSHClient

__all__ = ["FileClient", "auto_fileclient"]


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
    hosts
        A list of remote host filesystems. Supports using hosts defined in the ssh
        config file. Hosts can be specified as either "username@remote_host" or just
        "remote_host" in which case the username will be inferred from the current
        user.
    key_filename
        Path to private key file (for remote connections only).
    config_filename
        Path to OpenSSH config file defining host connection settings.
    """

    def __init__(
        self,
        key_filename: Union[str, Path] = "~/.ssh/id_rsa",
        config_filename: Union[str, Path] = "~/.ssh/config",
    ):
        self.key_filename = key_filename
        self.config_filename = config_filename

        self.connections: Dict[str, Dict[str, Any]] = {}

    def connect(self, host):
        """
        Connect to a remote host.

        Parameters
        ----------
        host
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

    def get_ssh(self, host) -> SSHClient:
        """
        Get an SSH connection to a host.

        Parameters
        ----------
        host
            A remote host filesystem. Supports using hosts defined in the ssh
            config file. The host can be specified as either "username@remote_host" or
            just "remote_host" in which case the username will be inferred from the
            current user.

        Returns
        -------
        SSHClient
            An ssh client to the host.
        """
        if host not in self.connections:
            self.connect(host)
        return self.connections[host]["ssh"]

    def get_sftp(self, host) -> SFTPClient:
        """
        Get an SFTP connection to a host.

        Parameters
        ----------
        host
            A remote host filesystem. Supports using hosts defined in the ssh
            config file. The host can be specified as either "username@remote_host" or
            just "remote_host" in which case the username will be inferred from the
            current user.

        Returns
        -------
        SFTPClient
            An sftp client to the host.
        """
        if host not in self.connections:
            self.connect(host)
        return self.connections[host]["sftp"]

    def exists(self, path: Union[str, Path], host: Optional[str] = None) -> bool:
        """
        Check whether a file exists.

        Parameters
        ----------
        path
            A path to check existence of.
        host
            A remote file system host on which to perform file operations.

        Returns
        -------
        bool
            Whether the file exists.
        """
        from pathlib import Path

        if host is None:
            return Path(path).exists()
        else:
            path = str(self.abspath(path, host=host))
            try:
                self.get_sftp(host).stat(path)
                return True
            except FileNotFoundError:
                return False

    def is_file(self, path: Union[str, Path], host: Optional[str] = None) -> bool:
        """
        Whether a path is a file.

        Parameters
        ----------
        path
            A path.
        host
            A remote file system host on which to perform file operations.

        Returns
        -------
        bool
            Whether the path is a file.
        """
        from pathlib import Path

        if host is None:
            return Path(path).is_file()
        else:
            import stat

            path = str(self.abspath(path, host=host))
            try:
                return stat.S_ISREG(self.get_sftp(host).lstat(path).st_mode)
            except FileNotFoundError:
                return False

    def is_dir(self, path: Union[str, Path], host: Optional[str] = None) -> bool:
        """
        Whether a path is a directory.

        Parameters
        ----------
        path
            A path.
        host
            A remote file system host on which to perform file operations.

        Returns
        -------
        bool
            Whether the path is a directory.
        """
        from pathlib import Path

        if host is None:
            return Path(path).is_dir()
        else:
            import stat

            path = str(self.abspath(path, host=host))
            try:
                return stat.S_ISDIR(self.get_sftp(host).lstat(path).st_mode)
            except FileNotFoundError:
                return False

    def listdir(self, path: Union[str, Path], host: Optional[str] = None) -> List[Path]:
        """
        Get the directory listing.

        Parameters
        ----------
        path
            Full path to the directory.
        host
            A remote file system host on which to perform file operations.

        Returns
        -------
        list[Path]
            List of filenames and directories.
        """
        from pathlib import Path

        if host is None:
            return list(Path(path).iterdir())
        else:
            path = str(self.abspath(path, host=host))
            return [Path(p) for p in self.get_sftp(host).listdir(path)]

    def copy(
        self,
        src_filename: Union[str, Path],
        dest_filename: Union[str, Path],
        src_host: Optional[str] = None,
        dest_host: Optional[str] = None,
    ):
        """
        Copy a file from source to destination.

        Parameters
        ----------
        src_filename
            Full path to source file.
        dest_filename
            Full path to destination file.
        src_host
            A remote file system host for the source file.
        dest_host
            A remote file system host for the destination file.
        """
        import shutil

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
                warnings.warn(f"Copy command gave error: {stderr}")
        else:
            # copying between two remote hosts; this is a pain and it is unlikely anyone
            # will want to do it.
            raise ValueError(
                "Copying between two different remote hosts is not supported."
            )

    def remove(self, path: Union[str, Path], host: Optional[str] = None):
        """
        Remove a file (does not work on directories).

        Parameters
        ----------
        path
            Path to a file.
        host
            A remote file system host on which to perform file operations.
        """
        if host is None:
            Path(path).unlink()
        else:
            path = str(self.abspath(host, host=host))
            self.get_sftp(host).unlink(path)

    def rename(
        self,
        old_path: Union[str, Path],
        new_path: Union[str, Path],
        host: Optional[str] = None,
    ):
        """
        Rename (move) a file.

        Parameters
        ----------
        old_path
            Path to an existing file.
        new_path
            Requested path to new file.
        host
            A remote file system host on which to perform file operations.
        """
        if host is None:
            Path(old_path).rename(new_path)
        else:
            old_path = str(self.abspath(old_path, host=host))
            new_path = str(self.abspath(new_path, host=host))
            self.get_sftp(host).rename(old_path, new_path)

    def abspath(self, path: Union[str, Path], host: Optional[str] = None) -> Path:
        """
        Get the absolute path.

        Parameters
        ----------
        path
            A path to a file or directory.
        host
            A remote file system host on which to perform file operations.

        Returns
        -------
        Path
            The absolute file path.
        """
        from pathlib import Path

        if host is None:
            return Path(path).absolute()
        else:
            ssh = self.get_ssh(host)
            _, stdout, _ = ssh.exec_command(f"readlink -f {path}")
            return Path([o.split("\n")[0] for o in stdout][0])

    def glob(self, path: Union[str, Path], host: Optional[str] = None) -> List[Path]:
        """
        Glob files and folders.

        Parameters
        ----------
        path
            A path to glob.
        host
            A remote file system host on which to perform file operations.

        Returns
        -------
        list[Path]
            A list of globs files and directories.
        """
        from glob import glob
        from pathlib import Path

        if host is None:
            files = glob(str(path))
        else:
            ssh = self.get_ssh(host)
            _, stdout, _ = ssh.exec_command(f"readlink -f {path}")
            files = [o.split("\n")[0] for o in stdout]

        return [Path(f) for f in files]

    def gzip(
        self,
        path: Union[str, Path],
        host: Optional[str] = None,
        compresslevel: int = 6,
        force: bool = False,
    ):
        """
        Gzip a file.

        Parameters
        ----------
        path
            Path to a file to gzip.
        host
            A remote file system host on which to perform file operations.
        compresslevel
            Level of compression, 1-9. 9 is default for GzipFile, 6 is default for gzip.
        force
            Overwrite gzipped file if it already exists.
        """
        import shutil
        from gzip import GzipFile

        path = self.abspath(path, host=host)
        path_gz = path.with_suffix(".gz")

        if str(path).lower().endswith("gz"):
            warnings.warn(f"{path} is already gzipped, skipping...")
            return None

        if self.is_dir(path, host=host):
            warnings.warn(f"{path} is a directory, skipping...")
            return None

        if self.exists(path_gz, host=host) and not force:
            raise FileExistsError(f"{path_gz} file already exists.")

        if host is None:
            with open(path, "rb") as f_in, GzipFile(
                path_gz, "wb", compresslevel=compresslevel
            ) as f_out:
                shutil.copyfileobj(f_in, f_out)
            shutil.copystat(path, path_gz)
            path.unlink()
        else:
            ssh = self.get_ssh(host)
            _, stdout, _ = ssh.exec_command(f"gzip -f {str(path)}")

    def gunzip(
        self,
        path: Union[str, Path],
        host: Optional[str] = None,
        force: bool = False,
    ):
        """
        Ungzip a file.

        Parameters
        ----------
        path
            Path to a file to gzip.
        host
            A remote file system host on which to perform file operations.
        force
            Overwrite non-gzipped file if it already exists.
        """
        from monty.io import zopen

        path = self.abspath(path, host=host)
        path_nongz = path.stem

        if not str(path).lower().endswith("gz"):
            warnings.warn(f"{path} is not gzipped, skipping...")
            return None

        if self.exists(path_nongz, host=host) and not force:
            raise FileExistsError(f"{path_nongz} file already exists")

        if host is None:
            with open(path_nongz, "wb") as f_out, zopen(path, "rb") as f_in:
                f_out.writelines(f_in)
            path.unlink()
        else:
            ssh = self.get_ssh(host)
            _, stdout, _ = ssh.exec_command(f"gunzip -f {str(path)}")

    def close(self):
        """Close all connections."""
        for connection in self.connections.values():
            connection["ssh"].close()
            connection["sftp"].close()
        self.connections = {}

    def __enter__(self):
        """Support for "with" context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for "with" context."""
        self.close()


def get_ssh_connection(
    username: Optional[str],
    hostname: str,
    key_filename: Union[str, Path],
    config_filename: Optional[Union[str, Path]] = None,
) -> SSHClient:
    """
    Connect to a remote host via paramiko.

    If the host key is not present it will be added automatically.

    Parameters
    ----------
    username
        The username. If ``None``, the current logged in username will be used.
    hostname
        The host name. Supports host aliases defined in the ssh config file.
    key_filename
        Path to private key file.
    config_filename
        Path to OpenSSH config file.

    Returns
    -------
    SSHClient
        An ssh connection to the host.
    """
    from pathlib import Path

    import paramiko

    key_filename = Path(key_filename).expanduser()
    if not key_filename.exists():
        raise ValueError(f"Cannot locate private key file: {key_filename}")

    config: Dict[str, Any] = {"hostname": hostname, "username": username}
    config_filename = Path(config_filename).expanduser()
    if Path(config_filename).exists():
        # try reading ssh config file
        ssh_config = paramiko.SSHConfig().from_path(str(config_filename))

        host_config = ssh_config.lookup(hostname)
        for k in ("hostname", "user", "port"):
            if k in host_config:
                config[k.replace("user", "username")] = host_config[k]

        if "proxycommand" in host_config:
            config["sock"] = paramiko.ProxyCommand(host_config["proxycommand"])

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(**config)
    return client


def auto_fileclient(method: Optional[Callable] = None):
    """
    Automatically pass a FileClient to the function if not already present in kwargs.

    This decorator should only be applied to functions with a ``file_client`` keyword
    argument. If a custom file client is not supplied when the function is called, it
    will automatically create a new FileClient, add it to the function arguments and
    close the file client connects at the end of the function.

    Parameters
    ----------
    method
        A function to wrap. This should not be specified directly and is implied
        by the decorator.
    """

    def decorator(func):
        from functools import wraps

        @wraps(func)
        def gen_fileclient(*args, **kwargs):
            file_client = kwargs.get("file_client", None)
            if file_client is None:
                with FileClient() as file_client:
                    kwargs["file_client"] = file_client
                    return func(*args, **kwargs)

        return gen_fileclient

    # See if we're being called as @auto_fileclient or @auto_fileclient().
    if method is None:
        # We're called with parens.
        return decorator

    # We're called as @auto_fileclient without parens.
    return decorator(method)
