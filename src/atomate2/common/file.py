"""Common functions for operations on files."""

from __future__ import annotations

import typing
from fnmatch import fnmatch
from pathlib import Path

from atomate2.utils.file_client import auto_fileclient

if typing.TYPE_CHECKING:
    from typing import Dict, List, Optional, Union

    from atomate2.utils.file_client import FileClient

__all__ = ["copy_files", "delete_files", "rename_files", "gzip_files", "gunzip_files"]


@auto_fileclient
def copy_files(
    src_dir: Union[str, Path],
    dest_dir: Union[str, Path] = None,
    src_host: str = None,
    include_files: List[Union[str, Path]] = None,
    exclude_files: List[Union[str, Path]] = None,
    suffix: str = "",
    allow_missing: bool = False,
    file_client: FileClient = None,
):
    """
    Copy files between source and destination folders.

    Parameters
    ----------
    src_dir
        The source directory.
    dest_dir
        The destination directory.
    src_host
        The source hostname used to specify a remote filesystem. Can be given as
        either "username@remote_host" or just "remote_host" in which case the username
        will be inferred from the current user. If ``None``, the local filesystem will
        be used as the source.
    include_files
        Filenames to include as a list of str or Path objects given relative to src_dir.
        Glob file paths are supported, e.g. "*.dat". If ``None``, all files in the
        source directory will be copied.
    exclude_files
        Filenames to exclude. Supports glob file matching, e.g., "*.dat".
    suffix
        A suffix to add to copied files. For example ".original".
    allow_missing
        Whether to error if a file in "include_files" is not present in the source
        directory.
    file_client
        A file client to use for performing file operations.
    """
    src_dir = file_client.abspath(src_dir, host=src_host)
    if dest_dir is None:
        dest_dir = Path.cwd()

    files = find_and_filter_files(
        file_client, src_dir, include_files, exclude_files, src_host
    )

    for file in files:
        from_file = src_dir / file
        to_file = (dest_dir / file).with_suffix(suffix)
        try:
            file_client.copy(from_file, to_file, src_host=src_host)
        except FileNotFoundError:
            if not allow_missing:
                raise


@auto_fileclient
def delete_files(
    directory: Union[str, Path] = None,
    host: str = None,
    include_files: List[Union[str, Path]] = None,
    exclude_files: List[Union[str, Path]] = None,
    allow_missing: bool = False,
    file_client: FileClient = None,
):
    """
    Delete files in a directory.

    Parameters
    ----------
    directory
        Directory in which to delete files. If ``None``, the current directory will be
        used (or home folder if specifying a remote host).
    host
        The hostname used to specify a remote filesystem. Can be given as either
        "username@remote_host" or just "remote_host" in which case the username will be
        inferred from the current user. If ``None``, the local filesystem will be used.
    include_files
        Filenames to include as a list of str or Path objects given relative to
        directory. Glob file paths are supported, e.g. "*.dat". If ``None``, all files
        in the directory will be deleted.
    exclude_files
        Filenames to exclude. Supports glob file matching, e.g., "*.dat".
    allow_missing
        Whether to error if a file in "include_files" is not present in the directory.
    file_client
        A file client to use for performing file operations.
    """
    if directory is None:
        directory = Path.cwd() if host is None else Path("~/")
    directory = file_client.abspath(directory, host=host)

    files = find_and_filter_files(
        file_client, directory, include_files, exclude_files, host
    )

    for file in files:
        file = directory / file
        try:
            file_client.remove(file, host=host)
        except FileNotFoundError:
            if not allow_missing:
                raise


@auto_fileclient
def rename_files(
    filenames: Dict[Union[str, Path], Union[str, Path]],
    directory: Union[str, Path] = None,
    host: str = None,
    allow_missing: bool = False,
    file_client: Optional[FileClient] = None,
):
    """
    Delete files in a directory.

    Parameters
    ----------
    filenames
        Files to rename. Given as a dictionary of ``{old_name: new_name}``. File names
        should be given relative to ``directory``. Glob matches are not supported.
    directory
        Directory in which to rename files. If ``None``, the current directory will be
        used (or home folder if specifying a remote host).
    host
        The hostname used to specify a remote filesystem. Can be given as either
        "username@remote_host" or just "remote_host" in which case the username will be
        inferred from the current user. If ``None``, the local filesystem will be used.
    allow_missing
        Whether to error if a file in "include_files" is not present in the directory.
    file_client
        A file client to use for performing file operations.
    """
    if directory is None:
        directory = Path.cwd() if host is None else Path("~/")
    directory = file_client.abspath(directory, host=host)

    for old_filename, new_filename in filenames.items():
        try:
            file_client.rename(
                directory / old_filename, directory / new_filename, host=host
            )
        except FileNotFoundError:
            if not allow_missing:
                raise


@auto_fileclient
def gzip_files(
    directory: Union[str, Path] = None,
    host: str = None,
    include_files: List[Union[str, Path]] = None,
    exclude_files: List[Union[str, Path]] = None,
    allow_missing: bool = False,
    force: bool = False,
    file_client: FileClient = None,
):
    """
    Gzip files in a directory.

    Parameters
    ----------
    directory
        Directory in which to gzip files. If ``None``, the current directory will be
        used (or home folder if specifying a remote host).
    host
        The hostname used to specify a remote filesystem. Can be given as either
        "username@remote_host" or just "remote_host" in which case the username will be
        inferred from the current user. If ``None``, the local filesystem will be used.
    include_files
        Filenames to include as a list of str or Path objects given relative to
        directory. Glob file paths are supported, e.g. "*.dat". If ``None``, all files
        in the directory will be gzipped.
    exclude_files
        Filenames to exclude. Supports glob file matching, e.g., "*.dat".
    allow_missing
        Whether to error if a file in "include_files" is not present in the directory.
    force
        Whether to overwrite files if they exist.
    file_client
        A file client to use for performing file operations.
    """
    if directory is None:
        directory = Path.cwd() if host is None else Path("~/")
    directory = file_client.abspath(directory, host=host)

    exclude_files = [] if exclude_files is None else list(exclude_files)
    exclude_files += ["*.gz", "*.GZ"]  # exclude files that are already gzipped
    files = find_and_filter_files(
        file_client, directory, include_files, exclude_files, host
    )

    for file in files:
        try:
            file_client.gzip(directory / file, host=host, force=force)
        except FileNotFoundError:
            if not allow_missing:
                raise


@auto_fileclient
def gunzip_files(
    directory: Union[str, Path] = None,
    host: str = None,
    include_files: List[Union[str, Path]] = None,
    exclude_files: List[Union[str, Path]] = None,
    allow_missing: bool = False,
    force: bool = False,
    file_client: FileClient = None,
):
    """
    Gunzip files in a directory.

    Parameters
    ----------
    directory
        Directory in which to gunzip files. If ``None``, the current directory will be
        used (or home folder if specifying a remote host).
    host
        The hostname used to specify a remote filesystem. Can be given as either
        "username@remote_host" or just "remote_host" in which case the username will be
        inferred from the current user. If ``None``, the local filesystem will be used.
    include_files
        Filenames to include as a list of str or Path objects given relative to
        directory. Glob file paths are supported, e.g. "*.dat". If ``None``, all gzipped
        files in the directory will be gunzipped.
    exclude_files
        Filenames to exclude. Supports glob file matching, e.g., "*.dat".
    allow_missing
        Whether to error if a file in "include_files" is not present in the directory.
    force
        Whether to overwrite files if they exist.
    file_client
        A file client to use for performing file operations.
    """
    if directory is None:
        directory = Path.cwd() if host is None else Path("~/")
    directory = file_client.abspath(directory, host=host)

    include_files = ["*.gz"] if include_files is None else include_files
    files = find_and_filter_files(
        file_client, directory, include_files, exclude_files, host
    )

    for file in files:
        try:
            file_client.gunzip(directory / file, host=host, force=force)
        except FileNotFoundError:
            if not allow_missing:
                raise


def find_and_filter_files(
    file_client: FileClient,
    directory: Union[str, Path],
    include_files: Optional[List[Union[str, Path]]],
    exclude_files: Optional[List[Union[str, Path]]],
    host: Optional[str],
) -> List[Path]:
    """
    Find and filter files.

    Parameters
    ----------
    file_client
        A file client.
    directory
        A directory in which to find files.
    include_files
        Filenames to include as a list of str or Path objects given relative to
        directory. Glob file paths are supported, e.g. "*.dat". If ``None``, all files
        in the source directory will be returned.
    exclude_files
        Filenames to exclude. Supports glob file matching, e.g., "*.dat".
    host
        A hostname used to specify a remote filesystem. Can be given as either
        "username@remote_host" or just "remote_host" in which case the username will be
        inferred from the current user. If ``None``, the local filesystem will be used.

    Returns
    -------
    list[Path]
        A list of file paths.
    """
    directory = file_client.abspath(directory, host=host)
    exclude_files = [] if exclude_files is None else exclude_files

    if include_files is None:
        files = file_client.listdir(directory, host=host)
        files = [f for f in files if file_client.is_file(f, host=host)]
    else:
        files = []
        for file in include_files:
            # expand any glob matches
            globbed_files = file_client.glob(directory / file, host=host)

            if len(globbed_files) > 0:
                # Need to get the path relative to directory
                globbed_files = [p.relative_to(directory) for p in globbed_files]
                files.extend(globbed_files)
            else:
                # no matches, only add the original file to be dealt with later
                files.append(Path(file))

    filtered_files = []
    for file in files:
        matches = [fnmatch(str(file), str(ex)) for ex in exclude_files]
        if not any(matches):
            filtered_files.append(file)

    return filtered_files
