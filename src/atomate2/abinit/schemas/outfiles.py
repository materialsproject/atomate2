import os
import base64
from monty.json import MSONable
from typing import TYPE_CHECKING
from pathlib import Path
from typing_extensions import Self

class AbinitStoredFile(MSONable):
    """Wrap a file to store its raw data."""

    def __init__(self, data: str | bytes, source_filepath: str | Path) -> None:
        self.data = data
        self.source_filepath = source_filepath

    def as_dict(self) -> dict:
        data = base64.b64encode(self.data).decode('ascii') if self.data_type == "bytes" else self.data
        
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "data": data,
            "data_type": self.data_type,
            "source_filepath": self.source_filepath,
            }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        data_type = d["data_type"]
        data = base64.b64decode(d["data"]) if data_type == "bytes" else d["data"]
        return cls(data=data, source_filepath=d["source_filepath"])

    @classmethod
    def from_file(cls, filepath: str | Path, data_type: str | type) -> Self:
        source_filepath = os.path.abspath(filepath)
        if data_type=="bytes" or data_type==bytes:
            read_type = 'rb'
        elif data_type=="str" or data_type==str:
            read_type = 'r'
        else:
            raise TypeError("data_type should be either bytes or str.")
        with open(source_filepath, read_type) as f:
            data = f.read()
        
        return cls(data=data, source_filepath=source_filepath)

    @property
    def data_type(self) -> str:
        return type(self.data).__name__

    @property
    def filename(self) -> str:
        return os.path.basename(self.source_filepath)

    @property
    def extension(self) -> str:
        return self.source_filepath.split("_")[-1]

    def write(self, filepath: str | Path=None) -> None:
        filepath = filepath or self.filename
        out_type = self.data_type[0]
        if self.data_type == 'bytes':
            write_type = 'wb'
        elif self.data_type == "str":
            write_type = 'w'
        else:
            raise TypeError("The data type is neither bytes nor string,\
                            does it make sense to write it into a file?")
        with open(filepath, write_type) as f:
            f.write(self.data)
