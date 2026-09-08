"""Class to store the raw data of ABINIT files."""

import base64
import os
from pathlib import Path

from monty.json import MSONable
from typing_extensions import Self

__all__ = ["AbinitStoredFile"]


class AbinitStoredFile(MSONable):
    """Store raw data from ABINIT output files.

    This class wraps ABINIT file data (as string or bytes) along with the
    source file path, providing serialization and deserialization capabilities.

    Attributes
    ----------
    data : str or bytes
        The raw file data.
    source_filepath : str or Path
        The absolute path to the source file.
    """

    def __init__(self, data: str | bytes, source_filepath: str | Path) -> None:
        """
        Initialize an AbinitStoredFile.

        Parameters
        ----------
        data : str or bytes
            The raw file data to store.
        source_filepath : str or Path
            The path to the source file.
        """
        self.data = data
        self.source_filepath = source_filepath

    def as_dict(self) -> dict:
        """
        Return a dictionary representation of the stored file.

        Returns
        -------
        dict
            Dictionary with data (base64-encoded if bytes), data type,
            and source filepath.
        """
        data = (
            base64.b64encode(self.data).decode("ascii")
            if isinstance(self.data, bytes)
            else self.data
        )

        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "data": data,
            "data_type": self.data_type,
            "source_filepath": self.source_filepath,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """
        Create an AbinitStoredFile from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary containing data, data_type, and source_filepath.

        Returns
        -------
        AbinitStoredFile
            An AbinitStoredFile instance.
        """
        data_type = d["data_type"]
        data = base64.b64decode(d["data"]) if data_type == "bytes" else d["data"]
        return cls(data=data, source_filepath=d["source_filepath"])

    @classmethod
    def from_file(cls, filepath: str | Path, data_type: str | type) -> Self:
        """
        Create an AbinitStoredFile from a file on disk.

        Parameters
        ----------
        filepath : str or Path
            Path to the file to read.
        data_type : str or type
            Type of data to read. Can be "bytes", bytes, "str", or str.

        Returns
        -------
        AbinitStoredFile
            An AbinitStoredFile instance containing the file data.

        Raises
        ------
        TypeError
            If data_type is not bytes or str.
        """
        source_filepath = os.path.abspath(filepath)
        if data_type in {"bytes", bytes}:
            read_type = "rb"
        elif data_type in {"str", str}:
            read_type = "r"
        else:
            raise TypeError("data_type should be either bytes or str.")
        with open(source_filepath, read_type) as f:
            data = f.read()

        return cls(data=data, source_filepath=source_filepath)

    @property
    def data_type(self) -> str:
        """Return the type of the data."""
        return type(self.data).__name__

    @property
    def filename(self) -> str:
        """Return the name of the source file."""
        return os.path.basename(self.source_filepath)

    @property
    def extension(self) -> str:
        """Return the extension of the source file."""
        return str(self.source_filepath).split("_")[-1]

    def write(self, filepath: str | Path | None = None) -> None:
        """
        Write the data to a file.

        Parameters
        ----------
        filepath : str or Path or None
            Path to the output file. If None, uses the original filename.
            Default is None.

        Raises
        ------
        TypeError
            If the data type is not supported for writing.
        """
        filepath = filepath or self.filename
        if self.data_type == "bytes":
            write_type = "wb"
        elif self.data_type == "str":
            write_type = "w"
        else:
            raise TypeError(f"The data type {self.data_type} is not supported.")
        with open(filepath, write_type) as f:
            f.write(self.data)
