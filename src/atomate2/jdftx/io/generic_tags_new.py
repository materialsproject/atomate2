"""
Module for generic tags used in JDFTx input file generation.

This module contains classes and functions for handling various types of tags
and their representations in JDFTx input files.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Dict, Union

import numpy as np

__author__ = "Jacob Clary"


def flatten_list(tag: str, list_of_lists: List[Any]) -> List[Any]:
    """Flatten a list of lists into a single list."""
    if not isinstance(list_of_lists, list):
        raise TypeError(f"{tag}: You must provide a list to flatten_list()!")
    while any(isinstance(x, list) for x in list_of_lists):
        list_of_lists = [item for sublist in list_of_lists for item in sublist]
    return list_of_lists


class ClassPrintFormatter:
    """Mixin class for formatting class string representation."""

    def __str__(self) -> str:
        """Format class for readable command line output."""
        return (
            f"{self.__class__}\n"
            + "\n".join(
                f"{item} = {self.__dict__[item]}"
                for item in sorted(self.__dict__)
            )
        )


@dataclass(kw_only=True)
class AbstractTag(ClassPrintFormatter, ABC):
    """Abstract base class for all tags."""

    multiline_tag: bool = False
    can_repeat: bool = False
    write_tagname: bool = True
    write_value: bool = True
    optional: bool = True
    defer_until_struc: bool = False
    _is_tag_container: bool = False
    allow_list_representation: bool = False

    @abstractmethod
    def validate_value_type(self, tag: str, value: Any, try_auto_type_fix: bool = False) -> bool:
        """Validate the type of the value for this tag."""

    def _validate_value_type(
        self, type_check: type, tag: str, value: Any, try_auto_type_fix: bool = False
    ) -> tuple:
        if self.can_repeat:
            self._validate_repeat(tag, value)
            is_valid = all(isinstance(x, type_check) for x in value)
        else:
            is_valid = isinstance(value, type_check)

        if not is_valid and try_auto_type_fix:
            try:
                if self.can_repeat:
                    value = [self.read(tag, str(x)) for x in value]
                else:
                    value = self.read(tag, str(value))
                is_valid = self._validate_value_type(type_check, tag, value)
            except Exception:
                print(f"Warning: Could not fix the typing for {tag} {value}!")
        return tag, is_valid, value

    def _validate_repeat(self, tag: str, value: Any) -> None:
        if not isinstance(value, list):
            raise TypeError(f"The {tag} tag can repeat but is not a list: {value}")

    @abstractmethod
    def read(self, tag: str, value_str: str) -> Any:
        """Read and parse the value string for this tag."""

    @abstractmethod
    def write(self, tag: str, value: Any) -> str:
        """Write the tag and its value as a string."""

    def _write(self, tag: str, value: Any, multiline_override: bool = False) -> str:
        tag_str = f"{tag} " if self.write_tagname else ""
        if self.multiline_tag or multiline_override:
            tag_str += "\\\n"
        if self.write_value:
            tag_str += f"{value} "
        return tag_str

    def _get_token_len(self) -> int:
        return int(self.write_tagname) + int(self.write_value)

# ... [rest of the code remains the same] ...
