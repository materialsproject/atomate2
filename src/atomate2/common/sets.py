"""Module defining an abstract interface for software inputs."""

import abc
from pathlib import Path
from typing import Union

from monty.json import MSONable


class InputSet(MSONable):
    """
    Abstract base class for code input sets.

    InputSet classes serve as containers for all calculation input data.

    All InputSet subclasses must implement a ``write_inputs`` method. Implementing the
    ``from_files`` method and ``is_valid`` property are optional.
    """

    @abc.abstractmethod
    def write_input(
        self,
        directory: Union[str, Path],
        make_dir: bool = True,
        overwrite: bool = True,
    ):
        """
        Write inputs to files.

        Parameters
        ----------
        directory
            Directory to write input files to.
        make_dir
            Whether to create the directory if it does not already exist.
        overwrite
            Whether to overwrite an input file if it already exists.
        """

    @classmethod
    def from_directory(cls, directory: Union[str, Path]):
        """
        Construct an InputSet from a directory containing one or more files.

        Parameters
        ----------
        directory
            Directory to read input files from.
        """
        raise NotImplementedError(f"from_files has not been implemented in {cls}")

    @property
    def is_valid(self) -> bool:
        """
        Verify the validity of an input set.

        Can be as simple or as complex as desired.

        Will raise a NotImplementedError unless overloaded by the inheriting class.
        """
        raise NotImplementedError(
            f"is_valid has not been implemented in {self.__class__}"
        )


class InputSetGenerator(MSONable):
    """
    A generator for an input set.

    It contains settings or sets of instructions for how to create InputSets from
    coordinates/structures or a previous calculation directory.
    """

    @staticmethod
    @abc.abstractmethod
    def get_input_set(*args, **kwargs) -> InputSet:
        """Generate an InputSet object."""
