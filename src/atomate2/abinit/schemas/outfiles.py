"""Class to store the raw data of Abinit files."""

import base64
import os
from pathlib import Path
from typing import Optional, Union

from monty.json import MSONable
from typing_extensions import Self


class AbinitStoredFile(MSONable):
    """Wrap a file to store its raw data."""

    def __init__(
        self, data: Union[str, bytes], source_filepath: Union[str, Path]
    ) -> None:
        self.data = data
        self.source_filepath = source_filepath

    def as_dict(self) -> dict:
        """Return a dict of the stored file."""
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
        """Create an AbinitStoredFile from a dict."""
        data_type = d["data_type"]
        data = base64.b64decode(d["data"]) if data_type == "bytes" else d["data"]
        return cls(data=data, source_filepath=d["source_filepath"])

    @classmethod
    def from_file(cls, filepath: Union[str, Path], data_type: Union[str, type]) -> Self:
        """Create an AbinitStoredFile from the original file."""
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

    def write(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """Write the data into a file."""
        filepath = filepath or self.filename
        if self.data_type == "bytes":
            write_type = "wb"
        elif self.data_type == "str":
            write_type = "w"
        else:
            raise TypeError(f"The data type {self.data_type} is not supported.")
        with open(filepath, write_type) as f:
            f.write(self.data)
