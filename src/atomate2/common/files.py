"""Common functions for operations on files."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from atomate2.utils.file_client import FileClient, auto_fileclient


@auto_fileclient
def copy_files(
    src_dir: str | Path,
    dest_dir: str | Path | None = None,
    src_host: str | None = None,
    include_files: list[str | Path] | None = None,
    exclude_files: list[str | Path] | None = None,
    suffix: str = "",
    prefix: str = "",
    allow_missing: bool = False,
    file_client: FileClient | None = None,
    link_files: bool = False,
) -> None:
    r"""
    Copy files between source and destination folders.

    Parameters
    ----------
    src_dir : str or Path
        The source directory.
    dest_dir : str or Path or None
        The destination directory.
    src_host : str or None
        The source hostname used to specify a remote filesystem. Can be given as
        either "usernamehost" or just "host" in which case the username
        will be inferred from the current user. If ``None``, the local filesystem will
        be used as the source.
    include_files : None or list of (str or .Path)
        Filenames to include as a list of str or Path objects given relative to
        ``src_dir``. Glob file paths are supported, e.g. "\*.dat". If ``None``, all
        files in the source directory will be copied.
    exclude_files : None or list of (str or .Path)
        Filenames to exclude. Supports glob file matching, e.g., "\*.dat".
    suffix : str
        A suffix to add to copied files. For example ".original".
    prefix : str
        A prefix to add to copied files. For example "original.".
    allow_missing : bool
        Whether to error if a file in ``include_files`` is not present in the source
        directory.
    file_client : .FileClient
        A file client to use for performing file operations.
    link_files : bool
        Whether to link the files instead of copying them. This option will raise an
        error if it is used in combination with a file_client.
    """
    src_dir = file_client.abspath(src_dir, host=src_host)
    if dest_dir is None:
        dest_dir = Path.cwd()

    files = find_and_filter_files(
        file_client, src_dir, include_files, exclude_files, src_host
    )

    for file in files:
        from_file = src_dir / file
        to_file = Path(file.parent) / f"{prefix}{file.name}"
        to_file = (dest_dir / to_file).with_suffix(file.suffix + suffix)
        try:
            if link_files and src_host is None:
                file_client.link(from_file, to_file)
            else:
                file_client.copy(from_file, to_file, src_host=src_host)
        except FileNotFoundError:
            if not allow_missing:
                raise


@auto_fileclient
def delete_files(
    directory: str | Path | None = None,
    host: str | None = None,
    include_files: list[str | Path] | None = None,
    exclude_files: list[str | Path] | None = None,
    allow_missing: bool = False,
    file_client: FileClient | None = None,
) -> None:
    r"""
    Delete files in a directory.

    Parameters
    ----------
    directory : str or Path or None
        Directory in which to delete files. If ``None``, the current directory will be
        used (or home folder if specifying a remote host).
    host : str or None
        The hostname used to specify a remote filesystem. Can be given as either
        "username@host" or just "host" in which case the username will be
        inferred from the current user. If ``None``, the local filesystem will be used.
    include_files : None or list of (str or .Path)
        Filenames to include as a list of str or Path objects given relative to
        directory. Glob file paths are supported, e.g. "\*.dat". If ``None``, all files
        in the directory will be deleted.
    exclude_files : None or list of (str or .Path)
        Filenames to exclude. Supports glob file matching, e.g., "\*.dat".
    allow_missing : bool
        Whether to error if a file in ``include_files`` is not present in the directory.
    file_client : .FileClient
        A file client to use for performing file operations.
    """
    if directory is None:
        directory = Path.cwd() if host is None else Path("~/")
    directory = file_client.abspath(directory, host=host)

    files = find_and_filter_files(
        file_client, directory, include_files, exclude_files, host
    )

    for file in files:
        try:
            file_client.remove(directory / file, host=host)
        except FileNotFoundError:
            if not allow_missing:
                raise


@auto_fileclient
def rename_files(
    filenames: dict[str | Path, str | Path],
    directory: str | Path | None = None,
    host: str | None = None,
    allow_missing: bool = False,
    file_client: FileClient | None = None,
) -> None:
    """
    Delete files in a directory.

    Parameters
    ----------
    filenames : dict
        Files to rename. Given as a dictionary of ``{old_name: new_name}``. File names
        should be given relative to ``directory``. Glob matches are not supported.
    directory : str or Path or None
        Directory in which to rename files. If ``None``, the current directory will be
        used (or home folder if specifying a remote host).
    host : str or None
        The hostname used to specify a remote filesystem. Can be given as either
        "username@host" or just "host" in which case the username will be
        inferred from the current user. If ``None``, the local filesystem will be used.
    allow_missing : bool
        Whether to error if a file in ``include_files`` is not present in the directory.
    file_client : .FileClient
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
    directory: str | Path | None = None,
    host: str | None = None,
    include_files: list[str | Path] | None = None,
    exclude_files: list[str | Path] | None = None,
    allow_missing: bool = False,
    force: bool = False,
    file_client: FileClient = None,
) -> None:
    r"""
    Gzip files in a directory.

    Parameters
    ----------
    directory : str or Path or None
        Directory in which to gzip files. If ``None``, the current directory will be
        used (or home folder if specifying a remote host).
    host : str or None
        The hostname used to specify a remote filesystem. Can be given as either
        "username@host" or just "host" in which case the username will be
        inferred from the current user. If ``None``, the local filesystem will be used.
    include_files : None or list of (str or .Path)
        Filenames to include as a list of str or Path objects given relative to
        directory. Glob file paths are supported, e.g. "\*.dat". If ``None``, all files
        in the directory will be gzipped.
    exclude_files : None or list of (str or .Path)
        Filenames to exclude. Supports glob file matching, e.g., "\*.dat".
    allow_missing : bool
        Whether to error if a file in ``include_files`` is not present in the directory.
    force : bool
        Whether to overwrite files if they exist.
    file_client : .FileClient
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
    directory: str | Path | None = None,
    host: str | None = None,
    include_files: list[str | Path] | None = None,
    exclude_files: list[str | Path] | None = None,
    allow_missing: bool = False,
    force: bool = False,
    file_client: FileClient | None = None,
) -> None:
    r"""
    Gunzip files in a directory.

    Parameters
    ----------
    directory : str or Path or None
        Directory in which to gunzip files. If ``None``, the current directory will be
        used (or home folder if specifying a remote host).
    host : str or None
        The hostname used to specify a remote filesystem. Can be given as either
        "username@host" or just "host" in which case the username will be
        inferred from the current user. If ``None``, the local filesystem will be used.
    include_files : None or list of (str or .Path)
        Filenames to include as a list of str or Path objects given relative to
        directory. Glob file paths are supported, e.g. "\*.dat". If ``None``, all
        gzipped files in the directory will be gunzipped.
    exclude_files : None or list of (str or .Path)
        Filenames to exclude. Supports glob file matching, e.g., "\*.dat".
    allow_missing : bool
        Whether to error if a file in ``include_files`` is not present in the directory.
    force : bool
        Whether to overwrite files if they exist.
    file_client : .FileClient
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
    directory: str | Path,
    include_files: list[str | Path] | None,
    exclude_files: list[str | Path] | None,
    host: str | None,
) -> list[Path]:
    r"""
    Find and filter files.

    Parameters
    ----------
    file_client : .FileClient
        A file client.
    directory : str or Path
        A directory in which to find files.
    include_files : None or list of (str or .Path)
        Filenames to include as a list of str or Path objects given relative to
        directory. Glob file paths are supported, e.g. "\*.dat". If ``None``, all files
        in the source directory will be returned.
    exclude_files : None or list of (str or .Path)
        Filenames to exclude. Supports glob file matching, e.g., "\*.dat".
    host : str or None
        A hostname used to specify a remote filesystem. Can be given as either
        "username@host" or just "host" in which case the username will be
        inferred from the current user. If ``None``, the local filesystem will be used.

    Returns
    -------
    list of Path
        A list of file paths.
    """
    directory = file_client.abspath(directory, host=host)
    exclude_files = [] if exclude_files is None else exclude_files

    if include_files is None:
        files = file_client.listdir(directory, host=host)
        files = [f for f in files if file_client.is_file(directory / f, host=host)]
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


def get_zfile(
    directory_listing: list[Path],
    base_name: str,
    allow_missing: bool = False,
) -> Path | None:
    """
    Find gzipped or non-gzipped versions of a file in a directory listing.

    Parameters
    ----------
    directory_listing : list of Path
        A list of files in a directory.
    base_name : str
        The base name of file to find.
    allow_missing : bool
        Whether to error if no version of the file (gzipped or un-gzipped) can be found.

    Returns
    -------
    Path or None
        A path to the matched file. If ``allow_missing=True`` and the file cannot be
        found, then ``None`` will be returned.
    """
    for file in directory_listing:
        if file.name in (base_name, f"{base_name}.gz", f"{base_name}.GZ"):
            return file

    if allow_missing:
        return None

    raise FileNotFoundError(f"Could not find {base_name} or {base_name}.gz file.")


def gzip_output_folder(
    directory: str | Path, setting: bool | str, files_list: list[str]
) -> None:
    """
    Zip the content of the output folder based on the specific code setting.

    Parameters
    ----------
    directory: str or Path or None
        Directory in which to gzip files.
    setting: bool or str
        the setting determining which files to zip. If True all the files in
        the directory will be zipped, if "atomate" only the files in
        files_list, if False no file will be zipped.
    files_list: list of str
        list of files to be zipped in case setting is "atomate"
    """
    if setting == "atomate":
        gzip_files(
            directory=directory,
            include_files=files_list,
            allow_missing=True,
            force=True,
        )
    elif setting:
        gzip_files(directory=directory, force=True)
