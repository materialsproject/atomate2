import math
import os
from dataclasses import dataclass
from functools import wraps
from atomate2.jdftx.io.JDFTXOutfileSlice import JDFTXOutfileSlice
from dataclasses import dataclass
from typing import List, Optional
from monty.io import zopen
from pathlib import Path


class ClassPrintFormatter():

    def __str__(self) -> str:
        """Generic means of printing class to command line in readable format"""
        return (
            str(self.__class__)
            + "\n"
            + "\n".join(
                str(item) + " = " + str(self.__dict__[item])
                for item in sorted(self.__dict__)
            )
        )


def check_file_exists(func):
    """Check if file exists (and continue normally) or raise an exception if it does not"""

    @wraps(func)
    def wrapper(filename):
        filename = Path(filename)
        if not filename.is_file():
            raise OSError(f"'{filename}' file doesn't exist!")
        return func(filename)

    return wrapper


@check_file_exists
def read_file(file_name: str) -> list[str]:
    '''
    Read file into a list of str

    Parameters
    ----------
    filename: Path or str
        name of file to read

    Returns
    -------
    text: list[str]
        list of strings from file
    '''
    with zopen(file_name, 'r') as f:
        text = f.readlines()
    return text


def get_start_lines(text: list[str], start_key: Optional[str]="*************** JDFTx", add_end: Optional[bool]=False) -> list[int]:
    '''
    Get the line numbers corresponding to the beginning of seperate JDFTx calculations
    (in case of multiple calculations appending the same out file)

    Args:
        text: output of read_file for out file
    '''
    start_lines = []
    for i, line in enumerate(text):
        if start_key in line:
            start_lines.append(i)
    if add_end:
        start_lines.append(i)
    return start_lines

def read_outfile_slices(file_name: str) -> list[list[str]]:
    '''
    Read slice of out file into a list of str

    Parameters
    ----------
    filename: Path or str
        name of file to read
    out_slice_idx: int
        index of slice to read from file

    Returns
    -------
    texts: list[list[str]]
        list of out file slices (individual calls of JDFTx)
    '''
    _text = read_file(file_name)
    start_lines = get_start_lines(_text, add_end=True)
    texts = []
    for i in range(len(start_lines)-1):
        text = _text[start_lines[i]:start_lines[i+1]]
        texts.append(text)
    return texts


@dataclass
class JDFTXOutfile(List[JDFTXOutfileSlice], ClassPrintFormatter):
    '''
    A class to read and process a JDFTx out file
    '''

    @classmethod
    def from_file(cls, file_path: str):
        texts = read_outfile_slices(file_path)
        instance = cls()
        for text in texts:
            instance.append(JDFTXOutfileSlice.from_out_slice(text))
        return instance

    def __getattr__(self, name):
        if len(self):
            return getattr(self[-1], name)
        else:
            try:
                return super().__getattr__(name)
            except AttributeError:
                if self:
                    return getattr(self[-1], name)
                raise AttributeError(f"'JDFTXOutfile' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Do we want this? I don't imagine this class object should be modified
        if name in self.__annotations__:
            super().__setattr__(name, value)
        elif self:
            setattr(self[-1], name, value)
        else:
            raise AttributeError(f"'JDFTXOutfile' object has no attribute '{name}'")
