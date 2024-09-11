from secrets import token_bytes
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List

import numpy as np

__author__ = "Jacob Clary"


# def flatten_list(tag: str, list_of_lists: List[Any]) -> List[Any]:
#     """Flattens list of lists into a single list, then stops"""
#     if not isinstance(list_of_lists, list):
#         raise ValueError(f"{tag}: You must provide a list to flatten_list()!")
#     while any([isinstance(x, list) for x in list_of_lists]):
#         list_of_lists = sum(list_of_lists, [])
#     return list_of_lists

def flatten_list(tag: str, list_of_lists: List[Any]) -> List[Any]:
    # Ben: I don't know what "then stops" means but I think this is how this
    # function should work.
    """Flattens list of lists into a single list, then stops"""
    if not isinstance(list_of_lists, list):
        raise ValueError(f"{tag}: You must provide a list to flatten_list()!")
    flist = []
    for v in list_of_lists:
        if isinstance(v, list):
            flist.extend(flatten_list(tag, v))
        else:
            flist.append(v)
    return flist


class ClassPrintFormatter:
    """Generic means of printing class to command line in readable format"""

    def __str__(self) -> str:
        return f"{self.__class__}\n" + "\n".join(
            f"{item} = {self.__dict__[item]}" for item in sorted(self.__dict__)
        )


@dataclass(kw_only=True)
class AbstractTag(ClassPrintFormatter, ABC):
    """Abstract base class for all tags."""

    multiline_tag: bool = False  # set to True if what to print tags across multiple lines, typically like electronic-minimize
    can_repeat: bool = (
        False  # set to True for tags that can appear on multiple lines, like ion
    )
    write_tagname: bool = (
        True  # set to False to not print the tagname, like for subtags of elec-cutoff
    )
    write_value: bool = (
        True  # set to False to not print any value, like for dump-interval
    )
    optional: bool = True  # set to False if tag (usually a subtag of a TagContainer) must be set for the JDFTXInfile to be valid
    # the lattice, ion, and ion-species are the main tags that are not optional
    defer_until_struc: bool = False
    _is_tag_container: bool = False
    allow_list_representation: bool = (
        False  # if True, allow this tag to exist as a list or list of lists
    )

    @abstractmethod
    def validate_value_type(
        self, tag: str, value: Any, try_auto_type_fix: bool = False
    ) -> bool:
        """Validate the type of the value for this tag."""

    def _validate_value_type(
        self, type_check: type, tag: str, value: Any, try_auto_type_fix: bool = False
    ) -> tuple:
        if self.can_repeat:
            self._validate_repeat(tag, value)
            is_valid = all([isinstance(x, type_check) for x in value])
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
                warnings.warn(
                    f"Could not fix the typing for {tag} {value}!", stacklevel=2
                )
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


"""
TODO:
fix dump-name and density-of-states tags

check that all ions either have or lack velocities
add validation of which tags require/forbid presence of other tags according to JDFTx docs?

choose how DeferredTags inherit from TagContainer?? same functionality once process the values "for real"

#possible TODO: add defaults like JDFTx does

MISC TODO:
    note which tags I've enforced a mandatory formatting,
        1. dump-name allows only 1 format
        2. debug requires 1 entry per line
        3. dump requires 1 entry per line

"""


@dataclass(kw_only=True)
class BoolTag(AbstractTag):
    _TF_options = {
        "read": {"yes": True, "no": False},
        "write": {True: "yes", False: "no"},
    }

    def validate_value_type(
        self, tag: str, value: Any, try_auto_type_fix: bool = False
    ) -> tuple:
        return self._validate_value_type(
            bool, tag, value, try_auto_type_fix=try_auto_type_fix
        )

    def read(self, tag: str, value: str) -> bool:
        if len(value.split()) > 1:
            raise ValueError(f"'{value}' for {tag} should not have a space in it!")
        try:
            if not self.write_value:
                # accounts for exceptions where only the tagname is used, e.g. dump-only or dump-fermi-density (sometimes) tags
                if not value:  # then the string '' was passed in because no value was provided but the tag was present
                    value = "yes"
                else:
                    raise ValueError(
                        f"The value '{value}' was provided to {tag}, it is not acting like a boolean"
                    )
            return self._TF_options["read"][value]
        except:
            raise ValueError(f"Could not set '{value}' as True/False for {tag}!")

    def write(self, tag: str, value: Any) -> str:
        value = self._TF_options["write"][value]
        return self._write(tag, value)

    def get_token_len(self) -> int:
        return self._get_token_len()


@dataclass(kw_only=True)
class StrTag(AbstractTag):
    options: list = None

    def validate_value_type(self, tag, value, try_auto_type_fix: bool = False) -> tuple:
        return self._validate_value_type(
            str, tag, value, try_auto_type_fix=try_auto_type_fix
        )

    def read(self, tag: str, value: str) -> str:
        if len(value.split()) > 1:
            raise ValueError(f"'{value}' for {tag} should not have a space in it!")
        try:
            value = str(value)
        except:
            raise ValueError(f"Could not set '{value}' to a str for {tag}!")

        if self.options is None or value in self.options:
            return value
        else:
            raise ValueError(
                f"The '{value}' string must be one of {self.options} for {tag}"
            )

    def write(self, tag: str, value) -> str:
        return self._write(tag, value)

    def get_token_len(self) -> int:
        return self._get_token_len()


@dataclass(kw_only=True)
class IntTag(AbstractTag):
    def validate_value_type(self, tag, value, try_auto_type_fix: bool = False) -> tuple:
        return self._validate_value_type(
            int, tag, value, try_auto_type_fix=try_auto_type_fix
        )

    def read(self, tag: str, value: str) -> int:
        if len(value.split()) > 1:
            raise ValueError(f"'{value}' for {tag} should not have a space in it!")
        try:
            return int(float(value))
        except:
            raise ValueError(f"Could not set '{value}' to a int for {tag}!")

    def write(self, tag: str, value) -> str:
        return self._write(tag, value)

    def get_token_len(self) -> int:
        return self._get_token_len()


@dataclass(kw_only=True)
class FloatTag(AbstractTag):
    prec: int = None

    def validate_value_type(self, tag, value, try_auto_type_fix: bool = False) -> tuple:
        return self._validate_value_type(
            float, tag, value, try_auto_type_fix=try_auto_type_fix
        )

    def read(self, tag: str, value: str) -> float:
        if len(value.split()) > 1:
            raise ValueError(f"'{value}' for {tag} should not have a space in it!")
        try:
            return float(value)  # can accept np.nan
        except:
            raise ValueError(f"Could not set '{value}' to a float for {tag}!")

    def write(self, tag: str, value) -> str:
        # pre-convert to string: self.prec+3 is minimum room for: - sign, 1 integer left of decimal, decimal, and precision
        # larger numbers auto add places to left of decimal
        if self.prec is not None:
            value = f"{value:{self.prec+3}.{self.prec}f}"
        return self._write(tag, value)

    def get_token_len(self) -> int:
        return self._get_token_len()


@dataclass(kw_only=True)
class TagContainer(AbstractTag):
    _is_tag_container: bool = True  # used to ensure only TagContainers are converted between list and dict representations
    subtags: dict = None
    linebreak_Nth_entry: int = (
        None  # handles special formatting for matrix tags, e.g. lattice tag
    )

    def _validate_single_entry(self, value, try_auto_type_fix: bool = False):
        if not isinstance(value, dict):
            raise ValueError(f"This tag should be a dict: {value}")

        tags_checked = []
        types_checks = []
        updated_value = deepcopy(value)
        for subtag, subtag_value in value.items():
            subtag_object = self.subtags[subtag]
            tags, checks, subtag_value = subtag_object.validate_value_type(
                subtag, subtag_value, try_auto_type_fix=try_auto_type_fix
            )
            if try_auto_type_fix:
                updated_value[subtag] = subtag_value
            if isinstance(checks, list):
                tags_checked.extend(tags)
                types_checks.extend(checks)
            else:
                tags_checked.append(tags)
                types_checks.append(checks)
        return tags_checked, types_checks, updated_value

    def validate_value_type(
        self, tag, value, try_auto_type_fix: bool = False
    ) -> list[list[Any]]:
        value = self.get_dict_representation(tag, value)
        if self.can_repeat:
            self._validate_repeat(tag, value)
            results = [
                self._validate_single_entry(x, try_auto_type_fix=try_auto_type_fix)
                for x in value
            ]
            return [list(x) for x in list(zip(*results))]
        else:
            return self._validate_single_entry(
                value, try_auto_type_fix=try_auto_type_fix
            )

    def read(self, tag: str, value: str) -> dict:
        value = value.split()
        if tag == "ion":
            special_constraints = [
                x in ["HyperPlane", "Linear", "None", "Planar"] for x in value
            ]
            if any(special_constraints):
                value = value[: special_constraints.index(True)]
                warnings.warn(
                    "Found special constraints reading an 'ion' tag, these were dropped; reading them has not been implemented!"
                )

        tempdict = {}  # temporarily store read tags out of order they are processed

        for subtag, subtag_type in self.subtags.items():
            if subtag == "poleEl":
                print("here")
            # every subtag with write_tagname=True in a TagContainer has a fixed length and can be immediately read in this loop if it is present
            if subtag in value:  # this subtag is present in the value string
                # Ben: At this point, the subtag is a string, and subtag_type is tag object with a can_repeat class variable.
                # If I am following this right, even if the subtag can repeat, its value will only be
                # fetched for the first time it appears, and the rest will be ignored.
                # Testing fix below.
                subtag_count = value.count(subtag)
                if not subtag_type.can_repeat:
                    if subtag_count > 1:
                        raise ValueError(
                            f"Subtag {subtag} is not allowed to repeat but appears more than once in {tag}'s value {value}"
                        )
                    else:
                        idx_start = value.index(subtag)
                        token_len = subtag_type.get_token_len()
                        idx_end = idx_start + token_len
                        subtag_value = " ".join(
                            value[(idx_start + 1) : idx_end]
                        )  # add 1 so the subtag value string excludes the subtagname
                        tempdict[subtag] = subtag_type.read(subtag, subtag_value)
                        del value[idx_start:idx_end]
                else:
                    tempdict[subtag] = []
                    for i in range(subtag_count):
                        idx_start = value.index(subtag)
                        idx_end = idx_start + subtag_type.get_token_len()
                        subtag_value = " ".join(
                            value[(idx_start + 1) : idx_end]
                        )  # add 1 so the subtag value string excludes the subtagname
                        tempdict[subtag].append(subtag_type.read(subtag, subtag_value))
                        del value[idx_start:idx_end]
                # idx_start = value.index(subtag)
                # idx_end = idx_start + subtag_type.get_token_len()
                # subtag_value = " ".join(
                #     value[(idx_start + 1) : idx_end]
                # )  # add 1 so the subtag value string excludes the subtagname
                # tempdict[subtag] = subtag_type.read(subtag, subtag_value)
                # del value[idx_start:idx_end]

        for subtag, subtag_type in self.subtags.items():
            # now try to populate remaining subtags that do not use a keyword in order of appearance
            # since all subtags in JDFTx that are TagContainers use a keyword to start their field, we know that
            #    any subtags processed here are only populated with a single token
            if len(value) == 0:
                break
            if (
                subtag in tempdict or subtag_type.write_tagname
            ):  # this tag has already been read or requires a tagname keyword to be present
                continue
            # note that this next line breaks if the JDFTx dump-name formatting is allowing
            # dump-name would have nested repeating TagContainers, which each need 2 values
            # you could check for which nonoptional args the TagContainers need and provide those but that's not general
            # you really need to be passing the entire value string for parsing, but that changes the return args
            tempdict[subtag] = subtag_type.read(subtag, value[0])
            del value[0]

        # reorder all tags to match order of __MASTER_TAG_LIST__ and do coarse-grained validation of read
        subdict = {x: tempdict[x] for x in self.subtags if x in tempdict}
        for subtag, subtag_type in self.subtags.items():
            if not subtag_type.optional and subtag not in subdict:
                raise ValueError(
                    f"The {subtag} tag is not optional but was not populated during the read!"
                )
        if len(value) > 0:
            raise ValueError(
                f"Something is wrong in the JDFTXInfile formatting, some values were not processed: {value}"
            )
        return subdict

    def write(self, tag: str, value):
        if not isinstance(value, dict):
            raise ValueError(
                f"value = {value}\nThe value to the {tag} write method must be a dict since it is a TagContainer!"
            )

        final_value = ""
        indent = "    "
        count = 0
        for subtag, subvalue in value.items():
            count += 1

            if self.subtags[subtag].can_repeat and isinstance(subvalue, list):
                # if a subtag.can_repeat, it is assumed that subvalue is a list
                #    the 2nd condition ensures this
                # if it is not a list, then the tag will still be printed by the else
                #    this could be relevant if someone manually sets the tag the can repeat's value to a non-list
                print_str = [
                    self.subtags[subtag].write(subtag, entry) for entry in subvalue
                ]
                print_str = " ".join(print_str)
            else:
                print_str = self.subtags[subtag].write(subtag, subvalue)

            if self.multiline_tag:
                final_value += f"{indent}{print_str}\\\n"
            elif self.linebreak_Nth_entry is not None:
                # handles special formatting with extra linebreak, e.g. for lattice tag
                i_column = count % self.linebreak_Nth_entry
                if i_column == 1:
                    final_value += f"{indent}{print_str}"
                elif i_column == 0:
                    final_value += f"{print_str}\\\n"
                else:
                    final_value += f"{print_str}"
            else:
                final_value += f"{print_str}"
        if (
            self.multiline_tag or self.linebreak_Nth_entry is not None
        ):  # handles special formatting for lattice tag
            final_value = final_value[:-2]  # exclude final \\n from final print call

        return self._write(tag, final_value, self.linebreak_Nth_entry is not None)

    def get_token_len(self) -> int:
        min_token_len = int(self.write_tagname)  # length of value subtags added next
        for subtag, subtag_type in self.subtags.items():
            subtag_token_len = (
                subtag_type.get_token_len()
            )  # recursive for nested TagContainers
            if (
                not subtag_type.optional
            ):  # TagContainers could be longer with optional subtags included
                min_token_len += subtag_token_len
        return min_token_len

    def check_representation(self, tag, value):
        if not self.allow_list_representation:
            return "dict"
        value_list = self.get_list_representation(tag, value)
        value_dict = self.get_dict_representation(tag, value)
        if value == value_list:
            return "list"
        elif value == value_dict:
            return "dict"
        else:
            raise ValueError(
                "Could not determine TagContainer representation, something is wrong"
            )

    def _make_list(self, value):
        value_list = []
        for subtag, subtag_value in value.items():
            subtag_type = self.subtags[subtag]
            if subtag_type.allow_list_representation:
                # this block deals with making list representations of any nested TagContainers
                if not isinstance(value[subtag], dict):
                    raise ValueError(
                        f"The subtag {subtag} is not a dict: '{value[subtag]}', so could not be converted"
                    )
                subtag_value = subtag_type.get_list_representation(
                    subtag, value[subtag]
                )  # recursive list generation

                if subtag_type.write_tagname:  # needed to write 'v' subtag in 'ion' tag
                    value_list.append(subtag)
                value_list.extend(subtag_value)
            elif not subtag_type.allow_list_representation and isinstance(
                value[subtag], dict
            ):
                # this triggers if someone sets this tag using mixed dict/list representations
                warnings.warn(f"The {subtag} subtag does not allow list representation with a value {value[subtag]}.\n \
                              I added the dict to the list. Is this correct? You will not be able to convert back!")
                value_list.append(value[subtag])
            else:
                # the subtag is simply of form {'subtag': subtag_value} and now adds concrete values to the list
                value_list.append(value[subtag])

        # return list of lists for tags in matrix format, e.g. lattice tag
        if (Ncol := self.linebreak_Nth_entry) is not None:
            Nrow = int(len(value_list) / Ncol)
            value_list = [
                [value_list[row * Ncol + col] for col in range(Ncol)]
                for row in range(Nrow)
            ]
        return value_list

    def get_list_representation(self, tag: str, value: dict) -> list:
        # convert dict representation into list representation by writing (nested) dicts into list or list of lists
        # there are 4 types of TagContainers in the list representation:
        # can_repeat: list of bool/str/int/float (ion-species)
        # can_repeat: list of lists (ion)
        # cannot repeat: list of bool/str/int/float (elec-cutoff)
        # cannot repeat: list of lists (lattice)
        if self.can_repeat:
            if all([isinstance(entry, list) for entry in value]):
                return value  # no conversion needed
            if any([not isinstance(entry, dict) for entry in value]):
                raise ValueError(f"The {tag} tag set to {value} must be a list of dict")
            tag_as_list = [self._make_list(entry) for entry in value]
        else:
            tag_as_list = self._make_list(value)
        return tag_as_list

    @staticmethod
    def _check_for_mixed_nesting(tag, value):
        if any([isinstance(x, (dict, list)) for x in value]):
            raise ValueError(
                f"{tag} with {value} cannot have nested lists/dicts mixed with bool/str/int/floats!"
            )

    def _make_dict(self, tag, value):
        # Ben: Is this supposed to create a dictionary? This creates a string without any dictionary indications
        value = flatten_list(tag, value)
        self._check_for_mixed_nesting(tag, value)
        return " ".join([str(x) for x in value])

    def get_dict_representation(self, tag: str, value: list) -> dict:
        # convert list or list of lists representation into string the TagContainer can process back into (nested) dict
        if self.can_repeat:  # repeated tags must be in same format
            if len(set([len(x) for x in value])) > 1:
                raise ValueError(
                    f"The values for {tag} {value} provided in a list of lists have different lengths"
                )
        value = value.tolist() if isinstance(value, np.ndarray) else value

        # there are 4 types of TagContainers in the list representation:
        # can_repeat: list of bool/str/int/float (ion-species)
        # can_repeat: list of lists (ion)
        # cannot repeat: list of bool/str/int/float (elec-cutoff)
        # cannot repeat: list of lists (lattice)

        # the .read() method automatically handles regenerating any nesting because is just like reading a file
        if self.can_repeat:
            if all([isinstance(entry, dict) for entry in value]):
                return value  # no conversion needed
            string_value = [self._make_dict(tag, entry) for entry in value]
            return [self.read(tag, entry) for entry in string_value]
        else:
            if isinstance(value, dict):
                return value  # no conversion needed
            string_value = self._make_dict(tag, value)
            return self.read(tag, string_value)
        


    

@dataclass(kw_only=True)
class StructureDeferredTagContainer(TagContainer):
    """
    This tag class accommodates tags that can have complicated values that depend on
    the number and species of atoms present. The species labels do not necessarily have
    to be elements, but just match the species given in the ion/ion-species tag(s). We
    will use the set of labels provided by the ion tag(s) because that is a well-defined
    token, while it may not be explicitly defined in ion-species.

    Relevant tags: add-U, initial-magnetic-moments, initial-oxidation-states, set-atomic-radius, setVDW
    """

    defer_until_struc: bool = True

    def read(self, tag: str, value: str, structure=None):
        raise NotImplementedError

        """This method is similar to StrTag.read(), but with less validation because usually will
        get a string like 'Fe 2.0 2.5 Ni 1.0 1.1' as the value to process later

        If this method is called separately from the JDFTXInfile processing methods, a Pymatgen
        structure may be provided directly
        """
        try:
            value = str(value)
        except:
            raise ValueError(f"Could not set '{value}' to a str for {tag}!")

        if structure is not None:
            value = self.read_with_structure(tag, value, structure)
        return value

    def read_with_structure(self, tag: str, value: str, structure):
        raise NotImplementedError

        """Fully process the value string using data from the Pymatgen structure"""
        return self._TC_read(tag, value, structure)


@dataclass(kw_only=True)
class MultiformatTag(AbstractTag):
    """
    This tag class should be used for tags that could have different types of input values given to them
    or tags where different subtag options directly impact how many expected arguments are provided
    e.g. the coulomb-truncation or van-der-waals tags

    This class should not be used for tags with simply some combination of mandatory and optional args
    because the TagContainer class can handle those cases by itself
    """

    format_options: list = None

    def validate_value_type(self, tag, value, try_auto_type_fix: bool = False) -> bool:
        format_index, value = self._determine_format_option(
            tag, value, try_auto_type_fix=try_auto_type_fix
        )
        is_valid = format_index is not None
        return tag, is_valid, value

    def read(self, tag: str, value: str):
        problem_log = []
        for i, trial_format in enumerate(self.format_options):
            try:
                return trial_format.read(tag, value)
            except Exception as e:
                problem_log.append(e)
        errormsg = f"No valid read format for '{tag} {value}' tag\nAdd option to format_options or double-check the value string and retry!\n\n"
        errormsg += "Here is the log of errors for each known formatting option:\n"
        errormsg += "\n".join(
            [f"Format {x}: {problem_log[x]}" for x in range(len(problem_log))]
        )
        raise ValueError(errormsg)

    def _determine_format_option(self, tag, value, try_auto_type_fix: bool = False):
        for i, format_option in enumerate(self.format_options):
            try:
                # print(i, tag, value, format_option)
                _, is_tag_valid, value = format_option.validate_value_type(
                    tag, value, try_auto_type_fix=try_auto_type_fix
                )
                if isinstance(is_tag_valid, list):
                    is_tag_valid = flatten_list(tag, is_tag_valid)
                    if not all(is_tag_valid):
                        raise ValueError(
                            f"{tag} option {i} is not it: validation failed"
                        )
                elif not is_tag_valid:
                    raise ValueError(f"{tag} option {i} is not it: validation failed")

                # print('PASSED!', tag, value, 'option', i)
                return i, value
            except:
                pass
                # print(f'{tag} option {i} is not it')
        raise ValueError(
            f"The format for {tag} for:\n{value}\ncould not be determined from the available options! Check your inputs and/or MASTER_TAG_LIST!"
        )

    def write(self, tag: str, value) -> str:
        format_index, _ = self._determine_format_option(tag, value)
        # print(f'using index of {format_index}')
        # Ben: Changing _write to write, using _write seem to shoot you straight
        # to the floor level definition, and completely messes up all the calls
        # to subtags for how they're supposed to be printed and just prints
        # a dictionary instead.
        # Ben: Update: this fixes it.
        return self.format_options[format_index].write(tag, value)
        # return self.format_options[format_index]._write(tag, value)
    

@dataclass
class BoolTagContainer(TagContainer):

    def read(self, tag:str, value: str) -> dict:
        value = value.split()
        tempdict = {}
        for subtag, subtag_type in self.subtags.items():
            if subtag in value:
                idx_start = value.index(subtag)
                idx_end = idx_start + subtag_type.get_token_len()
                subtag_value = " ".join(value[(idx_start + 1) : idx_end])
                tempdict[subtag] = subtag_type.read(subtag, subtag_value)
                del value[idx_start:idx_end]
        subdict = {x: tempdict[x] for x in self.subtags if x in tempdict}
        for subtag, subtag_type in self.subtags.items():
            if not subtag_type.optional and subtag not in subdict:
                raise ValueError(
                    f"The {subtag} tag is not optional but was not populated during the read!"
                )
        if len(value) > 0:
            raise ValueError(
                f"Something is wrong in the JDFTXInfile formatting, some values were not processed: {value}"
            )
        return subdict

@dataclass
class DumpTagContainer(TagContainer):

    def read(self, tag: str, value: str) -> dict:
        value = value.split()
        tempdict = {} 
        # Each subtag is a freq, which will be a BoolTagContainer
        for subtag, subtag_type in self.subtags.items():
            if subtag in value:
                idx_start = value.index(subtag)
                subtag_value = " ".join(value[(idx_start + 1):])
                tempdict[subtag] = subtag_type.read(subtag, subtag_value)
                del value[idx_start:]
        # reorder all tags to match order of __MASTER_TAG_LIST__ and do coarse-grained validation of read
        subdict = {x: tempdict[x] for x in self.subtags if x in tempdict}
        for subtag, subtag_type in self.subtags.items():
            if not subtag_type.optional and subtag not in subdict:
                raise ValueError(
                    f"The {subtag} tag is not optional but was not populated during the read!"
                )
        if len(value) > 0:
            raise ValueError(
                f"Something is wrong in the JDFTXInfile formatting, some values were not processed: {value}"
            )
        return subdict