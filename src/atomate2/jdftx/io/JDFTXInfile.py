"""
Classes for reading/manipulating/writing JDFTx input files.
All major JDFTx input files.
"""

from __future__ import annotations

import itertools
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.constants as const
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Structure
from pymatgen.util.io_utils import clean_lines

from atomate2.jdftx.io.generic_tags import flatten_list
from atomate2.jdftx.io.JDFTXInfile_master_format import (
    __PHONON_TAGS__,
    __TAG_LIST__,
    __WANNIER_TAGS__,
    MASTER_TAG_LIST,
    get_tag_object,
)

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike
    from pymatgen.util.typing import PathLike
    from typing_extensions import Self

__author__ = "Jacob Clary"


class JDFTXInfile(dict, MSONable):
    """
    JDFTxInfile object for reading and writing JDFTx input files.
    Essentially a dictionary with some helper functions.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Create a JDFTXInfile object.

        Args:
            params (dict): Input parameters as a dictionary.
        """
        super().__init__()
        if params is not None:
            self.update(params)

    def __str__(self) -> str:
        """Str representation of dict"""
        return "".join([line + "\n" for line in self.get_text_list()])

    def __add__(self, other: Self) -> Self:
        """
        Add all the values of another JDFTXInfile object to this object.
        Facilitate the use of "standard" JDFTXInfiles.
        """
        params: dict[str, Any] = dict(self.items())
        for key, val in other.items():
            if key in self and val != self[key]:
                raise ValueError(
                    f"JDFTXInfiles have conflicting values for {key}: {self[key]} != {val}"
                )
            params[key] = val
        return type(self)(params)

    def as_dict(self, sort_tags: bool = True, skip_module_keys: bool = False) -> dict:
        """MSONable dict."""
        params = dict(self)
        if sort_tags:
            params = {tag: params[tag] for tag in __TAG_LIST__ if tag in params}
        if not skip_module_keys:
            params["@module"] = type(self).__module__
            params["@class"] = type(self).__name__
        return params

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> Self:
        """
        Args:
            dct (dict): Serialized JDFTXInfile

        Returns
        -------
            JDFTXInfile
        """
        temp = cls({k: v for k, v in dct.items() if k not in ("@module", "@class")})
        # since users can provide arbitrary tags and values, need to do some validation
        #    (could do more later)
        # passing through the list -> dict representation ensures that tags pass through
        # a conversion to string and then through all .read() methods (happens during
        # list->dict conversion) to help ensure correct formatting
        # the list representation is easier to look at so convert back at the end
        temp = cls.get_dict_representation(cls.get_list_representation(temp))
        return cls.get_list_representation(temp)

    def copy(self) -> Self:
        return type(self)(self)

    def get_text_list(self) -> str:
        """Get a list of strings representation of the JDFTXInfile"""
        self_as_dict = self.get_dict_representation(self)

        text = []
        for tag_group in MASTER_TAG_LIST:
            added_tag_in_group = False
            for tag in MASTER_TAG_LIST[tag_group]:
                if tag not in self:
                    continue
                if tag in __WANNIER_TAGS__:
                    raise ValueError("Wannier functionality has not been added!")

                added_tag_in_group = True
                tag_object = MASTER_TAG_LIST[tag_group][tag]
                if tag_object.can_repeat and isinstance(self_as_dict[tag], list):
                    # if a tag_object.can_repeat, it is assumed that self[tag] is a list
                    #    the 2nd condition ensures this
                    # if it is not a list, then the tag will still be printed by the else
                    #    this could be relevant if someone manually sets the tag the can repeat's value to a non-list
                    for entry in self_as_dict[tag]:
                        text.append(tag_object.write(tag, entry))
                else:
                    text.append(tag_object.write(tag, self_as_dict[tag]))

            if added_tag_in_group:
                text.append("")
        return text

    def write_file(self, filename: PathLike) -> None:
        """Write JDFTXInfile to a file.

        Args:
            filename (str): filename to write to.
        """
        with zopen(filename, mode="wt") as file:
            file.write(str(self))

    @classmethod
    def from_file(
        cls,
        filename: PathLike,
        dont_require_structure: bool = False,
        sort_tags: bool = True,
    ) -> Self:
        """Read an JDFTXInfile object from a file.

        Args:
            filename (str): Filename for file

        Returns
        -------
            JDFTXInfile object
        """
        with zopen(filename, mode="rt") as file:
            return cls.from_str(
                file.read(),
                dont_require_structure=dont_require_structure,
                sort_tags=sort_tags,
            )

    @staticmethod
    def _preprocess_line(line):
        line = line.strip().split(maxsplit=1)
        tag: str = line[0].strip()
        if tag in __PHONON_TAGS__:
            raise ValueError("Phonon functionality has not been added!")
        if tag in __WANNIER_TAGS__:
            raise ValueError("Wannier functionality has not been added!")
        if tag not in __TAG_LIST__:
            raise ValueError(
                f"The {tag} tag in {line} is not in MASTER_TAG_LIST and is not a comment, something is wrong with this input data!"
            )
        tag_object = get_tag_object(tag)

        if len(line) == 2:
            value: Any = line[1].strip()
        elif len(line) == 1:
            value = (
                ""  # exception for tags where only tagname is used, e.g. dump-only tag
            )
        else:
            raise ValueError(
                f"The len(line.split(maxsplit=1)) of {line} should never not be 1 or 2"
            )

        return tag_object, tag, value

    @staticmethod
    def _store_value(params, tag_object, tag, value):
        if tag_object.can_repeat:  # store tags that can repeat in a list
            if tag not in params:
                params[tag] = []
            params[tag].append(value)
        else:
            if tag in params:
                raise ValueError(
                    f"The '{tag}' tag appears multiple times in this input when it should not!"
                )
            params[tag] = value
        return params

    @staticmethod
    def _gather_tags(lines):
        # gather all tags broken across lines into single string for processing later
        total_tag = ""
        gathered_string = []
        for line in lines:
            if line[-1] == "\\":  # then tag is continued on next line
                total_tag += (
                    line[:-1].strip() + " "
                )  # remove \ and any extra whitespace
            elif total_tag:  # then finished with line continuations
                total_tag += line
                gathered_string.append(total_tag)
                total_tag = ""
            else:  # then append line like normal
                gathered_string.append(line)
        return gathered_string

    @property
    def structure(self):
        """
        return a pymatgen Structure object
        """
        jdftstructure = self.to_pmg_structure()
        structure = jdftstructure.structure
        return structure

    @classmethod
    def from_str(
        cls, string: str, dont_require_structure: bool = False, sort_tags: bool = True
    ) -> Self:
        """Read an JDFTXInfile object from a string.

        Args:
            string (str): JDFTXInfile string

        Returns
        -------
            JDFTXInfile object
        """
        lines: list[str] = list(clean_lines(string.splitlines()))
        lines = cls._gather_tags(lines)

        params: dict[str, Any] = {}
        # process all tag value lines using specified tag formats in MASTER_TAG_LIST
        for line in lines:
            tag_object, tag, value = cls._preprocess_line(line)
            processed_value = tag_object.read(tag, value)
            params = cls._store_value(
                params, tag_object, tag, processed_value
            )  # this will change with tag categories

        if "include" in params:
            for filename in params["include"]:
                params.update(cls.from_file(filename, dont_require_structure=True))
            del params["include"]

        if (
            not dont_require_structure
            and "lattice" not in params
            and "ion" not in params
            and "ion-species" not in params
        ):
            raise ValueError("This input file is missing required structure tags")

        # if 'lattice' in params and 'ion' in params:  #skips if reading a partial input file added using the include tag
        #     structure = cls.to_pmg_structure(cls(params))

        #     for tag, value in params.items():         #this will change with tag categories
        #         #now correct the processing of tags that need to know the number and species of atoms in the system
        #         #in order to parse their values, e.g. initial-magnetic-moments tag
        #         tag_object = get_tag_object(tag)
        #         if tag_object.defer_until_struc:
        #             corrected_value = tag_object.read_with_structure(tag, value, structure)
        #             params[tag] = corrected_value

        if sort_tags:
            params = {tag: params[tag] for tag in __TAG_LIST__ if tag in params}
        return cls(params)

    @classmethod
    def to_JDFTXStructure(cls, JDFTXInfile, sort_structure: bool = False):
        """Converts JDFTx lattice, lattice-scale, ion tags into JDFTXStructure, with Pymatgen structure as attribute"""
        # use dict representation so it's easy to get the right column for moveScale, rather than checking for velocities
        JDFTXInfile = cls.get_dict_representation(JDFTXInfile)
        return JDFTXStructure._from_JDFTXInfile(
            JDFTXInfile, sort_structure=sort_structure
        )

    @classmethod
    def to_pmg_structure(cls, JDFTXInfile, sort_structure: bool = False):
        """Converts JDFTx lattice, lattice-scale, ion tags into Pymatgen structure"""
        # use dict representation so it's easy to get the right column for moveScale, rather than checking for velocities
        JDFTXInfile = cls.get_dict_representation(JDFTXInfile)
        return JDFTXStructure._from_JDFTXInfile(
            JDFTXInfile, sort_structure=sort_structure
        ).structure

    @staticmethod
    def _needs_conversion(conversion, value):
        # value will be in one of these formats:
        #  dict-to-list:
        #    dict
        #    list[dicts] (repeat tags in dict representation)
        #  list-to-dict:
        #    list
        #    list[lists] (repeat tags in list representation or lattice in list representation)

        if conversion == "list-to-dict":
            flag = False
        elif conversion == "dict-to-list":
            flag = True

        if isinstance(value, dict) or all(
            [isinstance(x, dict) for x in value]
        ):  # value is like {'subtag': 'subtag_value'}
            return flag
        else:
            return not flag

    @classmethod
    def get_list_representation(cls, JDFTXInfile):
        reformatted_params = deepcopy(JDFTXInfile.as_dict(skip_module_keys=True))
        # rest of code assumes lists are lists and not np.arrays
        reformatted_params = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in reformatted_params.items()
        }
        for tag, value in reformatted_params.items():
            tag_object = get_tag_object(tag)
            if tag_object.allow_list_representation and tag_object._is_tag_container:
                if cls._needs_conversion("dict-to-list", value):
                    reformatted_params.update(
                        {tag: tag_object.get_list_representation(tag, value)}
                    )
        return cls(reformatted_params)

    @classmethod
    def get_dict_representation(cls, JDFTXInfile):
        reformatted_params = deepcopy(JDFTXInfile.as_dict(skip_module_keys=True))
        # rest of code assumes lists are lists and not np.arrays
        reformatted_params = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in reformatted_params.items()
        }
        for tag, value in reformatted_params.items():
            tag_object = get_tag_object(tag)
            if tag_object.allow_list_representation and tag_object._is_tag_container:
                if cls._needs_conversion("list-to-dict", value):
                    reformatted_params.update(
                        {tag: tag_object.get_dict_representation(tag, value)}
                    )
        return cls(reformatted_params)

    def validate_tags(
        self,
        try_auto_type_fix: bool = False,
        error_on_failed_fix: bool = True,
        return_list_rep: bool = False,
    ):
        for tag in self:
            tag_object = get_tag_object(tag)
            checked_tags, is_tag_valid, value = tag_object.validate_value_type(
                tag, self[tag], try_auto_type_fix=try_auto_type_fix
            )
            if isinstance(is_tag_valid, list):
                checked_tags = flatten_list(tag, checked_tags)
                is_tag_valid = flatten_list(tag, is_tag_valid)
                should_warn = not all(is_tag_valid)
            else:
                should_warn = not is_tag_valid

            if return_list_rep:
                if (
                    tag_object.allow_list_representation
                ):  # converts newly formatted tag into list repr
                    value = tag_object.get_list_representation(tag, value)
            if error_on_failed_fix and should_warn and try_auto_type_fix:
                raise ValueError(
                    f"The {tag} tag with value:\n{self[tag]}\ncould not be fixed!"
                )
            if try_auto_type_fix and is_tag_valid:
                self.update({tag: value})
            if should_warn:
                warnmsg = f"The {tag} tag with value:\n{self[tag]}\nhas incorrect typing!\n    Subtag IsValid?\n"
                for i in range(len(checked_tags)):
                    warnmsg += f"    {checked_tags[i]} {is_tag_valid[i]}\n"
                warnings.warn(warnmsg)


@dataclass
class JDFTXStructure(MSONable):
    """Object for representing the data in JDFTXStructure tags

    Attributes
    ----------
        structure: Associated Structure.
        selective_dynamics: Selective dynamics attribute for each site if available.
            A Nx1 array of booleans.
        sort_structure (bool, optional): Whether to sort the structure. Useful if species
            are not grouped properly together. Defaults to False.
    """

    structure: Structure
    selective_dynamics: ArrayLike | None = None
    sort_structure: bool = False

    def __post_init__(self):
        if self.structure.is_ordered:
            site_properties = {}
            if self.selective_dynamics is not None:
                selective_dynamics = np.array(self.selective_dynamics)
                if not selective_dynamics.all():
                    site_properties["selective_dynamics"] = selective_dynamics

            # create new copy of structure so can add selective dynamics and sort atoms if needed
            structure = Structure.from_sites(self.structure)
            self.structure = structure.copy(site_properties=site_properties)
            if self.sort_structure:
                self.structure = self.structure.get_sorted_structure()
        else:
            raise ValueError(
                "Disordered structure with partial occupancies cannot be converted into JDFTXStructure!"
            )

    def __repr__(self) -> str:
        return self.get_str()

    def __str__(self) -> str:
        """String representation of Poscar file."""
        return self.get_str()

    @property
    def natoms(self) -> list[int]:
        """Sequence of number of sites of each type associated with JDFTXStructure"""
        syms: list[str] = [site.specie.symbol for site in self.structure]
        return [len(tuple(a[1])) for a in itertools.groupby(syms)]

    @classmethod
    def from_str(cls, data: str):
        return cls.from_JDFTXInfile(JDFTXInfile.from_str(data))

    @classmethod
    def from_file(cls, filename: str):
        return cls._from_JDFTXInfile(JDFTXInfile.from_file(filename))

    @classmethod
    def _from_JDFTXInfile(cls, JDFTXInfile, sort_structure: bool = False):
        lattice = np.array([JDFTXInfile["lattice"][x] for x in JDFTXInfile["lattice"]])
        if "latt-scale" in JDFTXInfile:
            latt_scale = np.array(
                [[JDFTXInfile["latt-scale"][x] for x in ["s0", "s1", "s2"]]]
            )
            lattice *= latt_scale
        lattice = lattice.T  # convert to row vector format
        lattice *= (
            const.value("Bohr radius") * 10**10
        )  # Bohr radius in Ang; convert to Ang

        atomic_symbols = [x["species-id"] for x in JDFTXInfile["ion"]]
        coords = np.array([[x["x0"], x["x1"], x["x2"]] for x in JDFTXInfile["ion"]])
        selective_dynamics = np.array([x["moveScale"] for x in JDFTXInfile["ion"]])

        coords_are_cartesian = False  # is default for JDFTx
        if "coords-type" in JDFTXInfile:
            coords_are_cartesian = JDFTXInfile["coords-type"] == "Cartesian"

        struct = Structure(
            lattice,
            atomic_symbols,
            coords,
            to_unit_cell=False,
            validate_proximity=False,
            coords_are_cartesian=coords_are_cartesian,
        )
        return cls(struct, selective_dynamics, sort_structure=sort_structure)

    def get_str(self, in_cart_coords: bool = False) -> str:
        """Return a string to be written as JDFTXInfile tags. Allows extra options as
        compared to calling str(JDFTXStructure) directly

        Args:
            in_cart_coords (bool): Whether coordinates are output in direct or Cartesian

        Returns
        -------
            str: representation of JDFTXInfile structure tags
        """
        JDFTX_tagdict = {}

        lattice = np.copy(self.structure.lattice.matrix)
        lattice = lattice.T  # transpose to get into column-vector format
        lattice /= (
            const.value("Bohr radius") * 10**10
        )  # Bohr radius in Ang; convert to Bohr

        JDFTX_tagdict["lattice"] = lattice
        JDFTX_tagdict["ion"] = []
        for i, site in enumerate(self.structure):
            coords = site.coords if in_cart_coords else site.frac_coords
            if self.selective_dynamics is not None:
                sd = self.selective_dynamics[i]
            else:
                sd = 1
            JDFTX_tagdict["ion"].append([site.label, *coords, sd])

        return str(JDFTXInfile.from_dict(JDFTX_tagdict))

    def write_file(self, filename: PathLike, **kwargs) -> None:
        """Write JDFTXStructure to a file. The supported kwargs are the same as those for
        the JDFTXStructure.get_str method and are passed through directly.
        """
        with zopen(filename, mode="wt") as file:
            file.write(self.get_str(**kwargs))

    def as_dict(self) -> dict:
        """MSONable dict."""
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "structure": self.structure.as_dict(),
            "selective_dynamics": np.array(self.selective_dynamics).tolist(),
        }

    @classmethod
    def from_dict(cls, params: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns
        -------
            JDFTXStructure
        """
        return cls(
            Structure.from_dict(params["structure"]),
            selective_dynamics=params["selective_dynamics"],
        )
