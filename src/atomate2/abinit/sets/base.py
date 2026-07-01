"""Module defining base ABINIT input set and generator."""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from abipy.abio.inputs import AbinitInput, MultiDataset
from abipy.flowtk.psrepos import get_repo_from_name
from abipy.flowtk.utils import Directory, irdvars_for_ext
from monty.json import MontyEncoder, jsanitize
from pymatgen.io.abinit.abiobjects import KSampling, KSamplingModes
from pymatgen.io.abinit.pseudos import Pseudo, PseudoTable
from pymatgen.io.core import InputGenerator, InputSet
from pymatgen.io.vasp import Kpoints
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath

from atomate2 import SETTINGS
from atomate2.abinit.files import fname2ext, load_abinit_input, out_to_in
from atomate2.abinit.utils.common import (
    INDATA_PREFIX,
    INDATAFILE_PREFIX,
    INDIR_NAME,
    INPUT_FILE_NAME,
    OUTDATA_PREFIX,
    OUTDATAFILE_PREFIX,
    OUTDIR_NAME,
    TMPDATA_PREFIX,
    TMPDIR_NAME,
    InitializationError,
    get_final_structure,
)
from atomate2.utils.path import strip_hostname

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from pymatgen.core.structure import Structure


logger = logging.getLogger(__name__)

__all__ = [
    "AbinitInputGenerator",
    "AbinitInputSet",
    "AbinitMixinInputGenerator",
    "as_pseudo_table",
    "get_ksampling",
    "set_workdir",
]


@dataclass
class AbinitMixinInputGenerator(InputGenerator):
    """
    Base class to generate input sets for ABINIT and related utilities.

    Attributes
    ----------
    calc_type : str
        A short description of the calculation type.
    prev_outputs_deps : str or tuple or None
        Defines the files that need to be linked from previous calculations.
        The format is a tuple where each element is a list of "|" separated
        run levels (as defined in the AbinitInput object) followed by a colon
        and a list of "|" separated extensions of files that need to be linked.
        The run level defines the type of calculations from which the file can
        be linked. An example is (f"{NSCF}:WFK",).
    """

    calc_type: str
    prev_outputs_deps: str | tuple | None

    @staticmethod
    def check_format_prev_dirs(
        prev_dirs: str | tuple | list | Path | None,
    ) -> list[str] | None:
        """
        Check and format previous directories for restart or dependency.

        Converts various input formats (string, Path, list, tuple) into a
        standardized list of directory paths as strings.

        Parameters
        ----------
        prev_dirs : str or tuple or list or Path or None
            Previous directory/directories in various formats. Can be a single
            path (str or Path) or a collection of paths (list or tuple).

        Returns
        -------
        list[str] or None
            List of directory paths as strings, or None if input is None.
        """
        if prev_dirs is None:
            return None
        if isinstance(prev_dirs, str | Path):
            return [str(prev_dirs)]
        return [str(prev_dir) for prev_dir in prev_dirs]

    def resolve_deps(
        self, prev_dirs: list[str], deps: tuple[str, ...], check_runlevel: bool = True
    ) -> tuple[dict[str, int], list[tuple[str, str]]]:
        """
        Resolve dependencies from previous calculations.

        Links required files from previous calculations by determining which
        files match the specified run levels and extensions.

        Parameters
        ----------
        prev_dirs : list[str]
            List of previous calculation directories.
        deps : tuple[str, ...]
            Tuple of dependency specifications. Each element is a string in the
            format "runlevel1|runlevel2:ext1|ext2" defining compatible run levels
            and file extensions.
        check_runlevel : bool
            Whether to check if the previous calculation's run level matches the
            dependency specification. Default is True.

        Returns
        -------
        tuple[dict[str, int], list[tuple[str, str]]]
            A tuple containing:
            - Dictionary mapping ird variable names to their values
            - List of (output_filepath, input_filename) tuples for files to link
        """
        input_files = []
        deps_irdvars = {}
        for prev_dir in prev_dirs:
            abinit_input = None
            if check_runlevel:
                abinit_input = load_abinit_input(prev_dir)
            for dep in deps:
                runlevel = set(dep.split(":")[0].split("|"))
                exts = list(dep.split(":")[1].split("|"))
                if not check_runlevel or (
                    abinit_input and runlevel.intersection(abinit_input.runlevel)
                ):
                    irdvars, inp_files = self.resolve_dep_exts(
                        prev_dir=prev_dir, exts=exts
                    )
                    input_files.extend(inp_files)
                    deps_irdvars.update(irdvars)
        return deps_irdvars, input_files

    @staticmethod
    def _get_in_file_name(out_filepath: str) -> str:
        """
        Convert an output file path to its corresponding input file name.

        Replaces the output file prefix with the input file prefix to generate
        the appropriate input file name for linking.

        Parameters
        ----------
        out_filepath : str
            Path to the output file.

        Returns
        -------
        str
            The input file name (basename only) with the appropriate prefix.
        """
        in_file = os.path.basename(out_filepath)
        in_file = in_file.replace(OUTDATAFILE_PREFIX, INDATAFILE_PREFIX, 1)

        return os.path.basename(in_file)

    @staticmethod
    def resolve_dep_exts(
        prev_dir: str, exts: list[str]
    ) -> tuple[dict[str, int], list[tuple[str, str]]]:
        """
        Resolve dependency files for specified extensions from a previous calculation.

        Searches for required output files in a previous calculation directory and
        determines the appropriate input file mappings and ird variables needed to
        link them.

        Parameters
        ----------
        prev_dir : str
            Path to the previous calculation directory.
        exts : list[str]
            List of file extensions to search for, in priority order (e.g.,
            ["WFK", "DEN"]). The first matching extension found will be used.

        Returns
        -------
        tuple[dict[str, int], list[tuple[str, str]]]
            A tuple containing:
            - Dictionary mapping ird variable names to their values
            - List of (output_filepath, input_filename) tuples for files to link

        Raises
        ------
        InitializationError
            If none of the specified extensions can be found in the previous
            calculation directory.
        """
        prev_dir = strip_hostname(prev_dir)  # TODO: to FileCLient?
        prev_outdir = Directory(os.path.join(prev_dir, OUTDIR_NAME))
        inp_files = []

        # Currently, a single previous job maps to one dependency, unless
        # "restart_from_deps" or "prev_outputs_deps" are tuples of multiple items.
        # However, these are currently tuples of ONE item (except for DDE, DTE,
        # and Phonons), which can have multiple
        # run_levels and extensions. If a previous job's run_level matches any of the
        # specified run_levels, the first matching extension in its output files is
        # used (e.g., WFK, then DEN if WFK is missing).
        #
        # Example: ("scf|relax|md:WFK|DEN",) means:
        #   - If the previous job is scf/relax/md, its outputs are considered.
        #   - WFK is prioritized; if missing, DEN is used.
        #
        # In light of this behavior, the use of break makes sense right now
        for ext in exts:
            # TODO: how to check that we have the files we need ?
            #  Should we raise if don't find at least one file for a given extension ?
            if ext in ("1WF", "1DEN", "DDK"):
                # Special treatment for 1WF and 1DEN files
                if ext in ["1WF", "DDK"]:
                    files = prev_outdir.find_1wf_files()
                elif ext == "1DEN":
                    files = prev_outdir.find_1den_files()
                else:
                    raise RuntimeError("Should not occur.")
                if files is not None:
                    inp_files = [
                        (f.path, AbinitMixinInputGenerator._get_in_file_name(f.path))
                        for f in files
                    ]
                    irdvars = irdvars_for_ext(ext)
                    break
            elif ext == "DEN":
                # Special treatment for DEN files
                # In case of relaxations or MD, there may be several TIM?_DEN files
                # First look for the standard out_DEN file.
                # If not found, look for the last TIM?_DEN file.
                out_den = prev_outdir.path_in(f"{OUTDATAFILE_PREFIX}_DEN")
                if os.path.exists(out_den):
                    irdvars = irdvars_for_ext("DEN")
                    inp_files.append(
                        (out_den, AbinitMixinInputGenerator._get_in_file_name(out_den))
                    )
                    break
                last_timden = prev_outdir.find_last_timden_file()
                if last_timden is not None:
                    if last_timden.path.endswith(".nc"):
                        in_file_name = f"{INDATAFILE_PREFIX}_DEN.nc"
                    else:
                        in_file_name = f"{INDATAFILE_PREFIX}_DEN"
                    inp_files.append((last_timden.path, in_file_name))
                    irdvars = irdvars_for_ext("DEN")
                    break
            else:
                out_file = prev_outdir.has_abiext(ext)
                irdvars = irdvars_for_ext(ext)
                if out_file:
                    inp_files.append(
                        (
                            out_file,
                            AbinitMixinInputGenerator._get_in_file_name(out_file),
                        )
                    )
                    break
        else:
            msg = f"Cannot find {' or '.join(exts)} file to restart from."
            logger.error(msg)
            raise InitializationError(msg)
        return irdvars, inp_files


class AbinitInputSet(InputSet):
    """
    A class to represent a set of ABINIT inputs.

    Attributes
    ----------
    input_files : list[tuple[str, str]] or None
        List of input files needed for the calculation. Each tuple contains
        (output_filepath, input_filename). The corresponding file reading
        variables (ird***) should be present in the abinit_input.
    link_files : bool
        Whether to create symbolic links for input files. Default is True.
    """

    def __init__(
        self,
        abinit_input: AbinitInput,
        input_files: list[tuple[str, str]] | None = None,
        link_files: bool = True,
    ) -> None:
        """
        Initialize an AbinitInputSet.

        Parameters
        ----------
        abinit_input : AbinitInput
            An AbinitInput object containing the calculation parameters.
        input_files : list[tuple[str, str]] or None
            List of (output_filepath, input_filename) tuples for files to link.
            Default is None.
        link_files : bool
            Whether to create symbolic links for the input files.
            Default is True.
        """
        self.input_files = input_files
        self.link_files = link_files
        super().__init__(
            inputs={
                INPUT_FILE_NAME: abinit_input,
                "abinit_input.json": json.dumps(
                    abinit_input.as_dict(), cls=MontyEncoder
                ),
            }
        )

    def write_input(
        self,
        directory: str | Path,
        make_dir: bool = True,
        overwrite: bool = True,
        zip_inputs: bool = False,
    ) -> None:
        """
        Write ABINIT input files to a directory.

        Creates the necessary directory structure (indir, outdir, tmpdir) and
        writes input files, optionally creating symbolic links to files from
        previous calculations.

        Parameters
        ----------
        directory : str or Path
            The directory to write the input files to.
        make_dir : bool
            Whether to create the directory if it does not exist.
            Default is True.
        overwrite : bool
            Whether to overwrite existing files. Default is True.
        zip_inputs : bool
            Whether to zip the input files. Default is False.

        Notes
        -----
        The zip_inputs functionality may not be fully compatible with ABINIT
        workflows as the input set creates symbolic links to previous calculation
        files and sets up specific directory structures (indir, outdir, tmpdir).
        """
        # TODO: Verify whether zip_inputs should be supported given that ABINIT
        # input sets create symbolic links and directory structures
        self.inputs["abinit_input.json"] = json.dumps(
            jsanitize(self.abinit_input.as_dict())
        )
        super().write_input(
            directory=directory,
            make_dir=make_dir,
            overwrite=overwrite,
            zip_inputs=zip_inputs,
        )
        del self.inputs["abinit_input.json"]
        indir, _outdir, _tmpdir = set_workdir(workdir=directory)

        if self.input_files:
            out_to_in(
                out_files=self.input_files,
                indir=indir.path,
                link_files=self.link_files,
            )

    def validate(self) -> bool:
        """
        Validate the input set.

        Checks that all files in the input directory have their corresponding
        ird variables properly set in the ABINIT input.

        Returns
        -------
        bool
            True if all input files have valid ird variable configurations,
            False otherwise.
        """
        if not self.input_files:
            return True
        for _out_filepath, in_file in self.input_files:
            ext = fname2ext(in_file)
            if ext is None:
                return False
            irdvars = irdvars_for_ext(ext)
            # Need to consider irdddk to read 1WF files
            if ext == "1WF":
                irdvars["irdddk"] = 1
            for irdvar, irdval in irdvars.items():
                if irdvar in self.abinit_input and self.abinit_input[irdvar] == irdval:
                    break
            else:
                return False
        return True

    @property
    def abinit_input(self) -> AbinitInput:
        """
        Get the AbinitInput object.

        Returns
        -------
        AbinitInput
            The ABINIT input object containing calculation parameters.
        """
        return self[INPUT_FILE_NAME]

    def set_vars(self, *args, **kwargs) -> dict:
        """
        Set the values of ABINIT variables.

        Sets ABINIT variables in the AbinitInput object. Variables can be
        passed as a dictionary or as keyword arguments, or a combination
        of both.

        Parameters
        ----------
        *args
            Dictionary mapping ABINIT variable names to their values.
        **kwargs
            ABINIT variables and their values as keyword arguments.

        Returns
        -------
        dict
            Dictionary with the variables that have been added.
        """
        return self.abinit_input.set_vars(*args, **kwargs)

    def remove_vars(self, keys: Iterable[str] | str, strict: bool = True) -> dict:
        """
        Remove ABINIT variables from the input.

        Removes the specified ABINIT variables from the AbinitInput object.

        Parameters
        ----------
        keys : Iterable[str] or str
            String or iterable of strings with the names of the ABINIT
            variables to be removed.
        strict : bool
            Whether to raise a KeyError if one of the ABINIT variables
            to be removed is not present. Default is True.

        Returns
        -------
        dict
            Dictionary with the variables that have been removed.
        """
        return self.abinit_input.remove_vars(keys=keys, strict=strict)

    def runlevel(self) -> set[str]:
        """
        Get the set of strings defining the calculation type.

        Returns
        -------
        set[str]
            Set of run level strings (e.g., {'scf'}, {'nscf'}, {'relax'}).
        """
        return self.abinit_input.runlevel

    def set_structure(self, structure: Any) -> Structure:
        """
        Set the structure for this input set.

        Forwards the structure setting to the AbinitInput object.

        Parameters
        ----------
        structure : Any
            A pymatgen Structure object or compatible structure representation.

        Returns
        -------
        Structure
            The structure that was set.
        """
        return self.abinit_input.set_structure(structure)

    def deepcopy(self) -> AbinitInputSet:
        """
        Create a deep copy of the input set.

        Returns
        -------
        AbinitInputSet
            A deep copy of this AbinitInputSet object.
        """
        return copy.deepcopy(self)


def as_pseudo_table(pseudos: str | Sequence[Pseudo]) -> PseudoTable:
    """
    Get pseudopotentials as a PseudoTable object.

    Converts various pseudopotential input formats into a standardized
    PseudoTable object.

    Parameters
    ----------
    pseudos : str or Sequence[Pseudo]
        Pseudopotentials specified as:
        - A single pseudopotential file path (str)
        - A string in format "repository:table" (e.g., "ONCVPSP-PBE:standard")
        - A sequence of Pseudo objects

    Returns
    -------
    PseudoTable
        Table of pseudopotentials.

    Raises
    ------
    RuntimeError
        If the specified pseudo repository is not installed.
    """
    if isinstance(pseudos, str):
        # In case a single path to a pseudopotential file has been passed
        if os.path.isfile(pseudos):
            return PseudoTable(pseudos)
        pseudo_repo_name, table_name = pseudos.rsplit(":", 1)
        repo = get_repo_from_name(pseudo_repo_name)
        if not repo.is_installed():
            msg = (
                f"Pseudo repository {pseudo_repo_name} is not installed "
                f"in {repo.dirpath}. "
                f"Use abips.py to install it."
            )
            raise RuntimeError(msg)
        return repo.get_pseudos(table_name)
    return PseudoTable(pseudos)


@dataclass
class AbinitInputGenerator(AbinitMixinInputGenerator):
    """
    A class to generate ABINIT input sets.

    Attributes
    ----------
    factory : Callable or None
        A callable that generates an AbinitInput or MultiDataset object.
        Default is None.
    calc_type : str
        A short description of the calculation type.
        Default is "abinit_calculation".
    pseudos : str or list[str] or PseudoTable or None
        Define the pseudopotentials that should be used for the calculation.
        Can be an instance of a PseudoTable, a list of strings with the paths of
        the pseudopotential files, or a string with the name of a PseudoDojo table
        (https://github.com/PseudoDojo/) followed by the accuracy of the pseudos
        in that table, separated by a colon. This requires that the PseudoTable
        is installed in the system.
        Set to None if no pseudopotentials should be set, as coming from a previous
        AbinitInput.
        Default is "ONCVPSP-PBE-SR-PDv0.4:standard".
    factory_kwargs : dict
        A dictionary to customize the values for the arguments of the factory
        function. Default is an empty dict.
    user_abinit_settings : dict
        A dictionary that allows to set any ABINIT variable in the AbinitInput
        after it has been generated from the factory function. This will override
        any value or default previously set. Set a value to None to remove it
        from the input. Default is an empty dict.
    user_kpoints_settings : dict or KSampling
        Allow user to override k-points setting by supplying a dict. E.g.,
        ``{"reciprocal_density": 1000}``. User can also supply a KSampling object.
        Default is an empty dict.
    restart_from_deps : tuple or None
        Defines the files that need to be linked from previous calculations in
        case of restart. The format is a tuple where each element is a list of
        "|" separated run levels (as defined in the AbinitInput object) followed
        by a colon and a list of "|" separated extensions of files that need to
        be linked. The run level defines the type of calculations from which the
        file can be linked. An example is (f"{NSCF}:WFK",).
        Default is None.
    prev_outputs_deps : tuple or None
        Defines the files that need to be linked from previous calculations and
        are required for the execution of the current calculation.
        The format is a tuple where each element is a list of "|" separated
        run levels (as defined in the AbinitInput object) followed by a colon and
        a list of "|" separated extensions of files that need to be linked.
        The run level defines the type of calculations from which the file can
        be linked. An example is (f"{NSCF}:WFK",).
        Default is None.
    factory_prev_inputs_kwargs : dict or None
        A dictionary defining the source of one or more previous AbinitInput
        objects in case they are required by a factory to build a new AbinitInput.
        The key should match the name of the argument of the factory function
        and the value should be a tuple with the run levels of the compatible
        types of AbinitInput that can be used.
        Default is None.
    force_gamma : bool
        Force gamma centered k-point generation. Default is True.
    symprec : float
        Tolerance for symmetry finding, used for line mode band structure k-points.
        Default is SETTINGS.SYMPREC.
    """

    factory: Callable | None = None
    calc_type: str = "abinit_calculation"
    pseudos: str | list[str] | PseudoTable | None = "ONCVPSP-PBE-SR-PDv0.4:standard"
    factory_kwargs: dict = field(default_factory=dict)
    user_abinit_settings: dict = field(default_factory=dict)
    user_kpoints_settings: dict | KSampling = field(default_factory=dict)
    restart_from_deps: tuple | None = None
    prev_outputs_deps: tuple | None = None
    factory_prev_inputs_kwargs: dict | None = None
    force_gamma: bool = True
    symprec: float = SETTINGS.SYMPREC

    def get_input_set(
        self,
        structure: Structure | None = None,
        restart_from: str | tuple | list | Path | None = None,
        prev_outputs: str | tuple | list | Path | None = None,
    ) -> AbinitInputSet:
        """
        Generate an AbinitInputSet object.

        This method creates an ABINIT input set either from a structure or by
        restarting from a previous calculation. It handles dependency resolution
        and links required files from previous calculations.

        Parameters
        ----------
        structure : Structure or None
            Pymatgen Structure object. Required if not restarting from a
            previous calculation. Default is None.
        restart_from : str or Path or list or tuple or None
            Directory or list/tuple of 1 directory to restart from. If provided,
            the structure and input from this calculation will be used.
            Default is None.
        prev_outputs : str or Path or list or tuple or None
            Directory or list/tuple of directories needed as dependencies for the
            AbinitInputSet generated. Files from these directories will be linked
            as required by prev_outputs_deps. Default is None.

        Returns
        -------
        AbinitInputSet
            An ABINIT input set ready to be written and executed.

        Notes
        -----
        This method assumes there is an abinit_input.json file in each directory
        containing the AbinitInput object used to execute ABINIT.
        """
        # Get the pseudos as a PseudoTable
        pseudos = as_pseudo_table(self.pseudos) if self.pseudos else None

        restart_from = self.check_format_prev_dirs(restart_from)
        prev_outputs = self.check_format_prev_dirs(prev_outputs)

        all_irdvars = {}
        input_files = []
        if restart_from is not None:
            # Use the previous ABINIT input
            abinit_input = load_abinit_input(restart_from[0])
            # Update the ABINIT input with the final structure from restart
            structure = get_final_structure(restart_from[0])
            abinit_input.set_structure(structure=structure)
            # Files for restart (e.g., continue a not yet converged
            # scf/nscf/relax calculation)
            irdvars, files = self.resolve_deps(
                restart_from, deps=self.restart_from_deps
            )
            all_irdvars.update(irdvars)
            input_files.extend(files)
        else:
            if prev_outputs is not None and not self.prev_outputs_deps:
                raise RuntimeError(
                    f"Previous outputs not allowed for {type(self).__name__}."
                )
            abinit_input = self.get_abinit_input(
                structure=structure,
                pseudos=pseudos,
                prev_outputs=prev_outputs,
            )
        # Reset ird variables to avoid conflicts with previously set values
        abinit_input.pop_irdvars()

        # Files that are dependencies (e.g., band structure calculations
        # need the density)
        if prev_outputs:
            irdvars, files = self.resolve_deps(prev_outputs, self.prev_outputs_deps)
            all_irdvars.update(irdvars)
            input_files.extend(files)

        # Set ird variables for file dependencies and user-specified settings
        abinit_input.set_vars(all_irdvars)
        abinit_input.set_vars(self.user_abinit_settings)

        abinit_input["indata_prefix"] = (f'"{INDATA_PREFIX}"',)
        abinit_input["outdata_prefix"] = (f'"{OUTDATA_PREFIX}"',)
        abinit_input["tmpdata_prefix"] = (f'"{TMPDATA_PREFIX}"',)

        # Note: link_files is currently hardcoded to True. Consider making this
        # configurable through the generator or input set parameters.
        return AbinitInputSet(
            abinit_input=abinit_input,
            input_files=input_files,
            link_files=True,
        )

    def resolve_prev_inputs(
        self, prev_dirs: list[str], prev_inputs_kwargs: dict[str, Any]
    ) -> dict[str, AbinitInput]:
        """
        Find suitable ABINIT inputs from previous calculation outputs.

        Retrieves ABINIT inputs from previous calculations that match the
        specified run levels. Also updates each input with the final structure
        from its corresponding calculation.

        Parameters
        ----------
        prev_dirs : list[str]
            List of previous calculation directories as strings.
        prev_inputs_kwargs : dict[str, Any]
            Dictionary where keys are factory argument names and values are
            sets of compatible run levels for matching previous inputs.

        Returns
        -------
        dict[str, AbinitInput]
            Dictionary mapping factory argument names to their corresponding
            AbinitInput objects from previous calculations.

        Raises
        ------
        RuntimeError
            If multiple previous inputs match the requirements for a single
            factory argument, or if the number of found inputs does not match
            the number of required inputs.

        Notes
        -----
        This method assumes that prev_dirs is in the correct format, i.e.,
        a list of directories as str or Path.
        """
        abinit_inputs = {}
        for prev_d in prev_dirs:
            # Note: Consider using FileClient for remote file handling
            prev_dir = strip_hostname(prev_d)
            abinit_input = load_abinit_input(prev_dir)
            for var_name, run_levels in prev_inputs_kwargs.items():
                if abinit_input.runlevel and abinit_input.runlevel.intersection(
                    run_levels
                ):
                    if var_name in abinit_inputs:
                        msg = (
                            "Multiple previous inputs match the "
                            "requirements as inputs for the factory"
                        )
                        raise RuntimeError(msg)
                    final_structure = get_final_structure(prev_dir)
                    abinit_input.set_structure(final_structure)
                    abinit_inputs[var_name] = abinit_input

        n_found = len(abinit_inputs)
        n_required = len(self.factory_prev_inputs_kwargs)
        if n_found != n_required:
            raise RuntimeError(
                f"Should have exactly {n_required} previous output. Found {n_found}"
            )

        return abinit_inputs

    def get_abinit_input(
        self,
        structure: Structure | None = None,
        pseudos: PseudoTable | None = None,
        prev_outputs: list[str] | None = None,
        abinit_settings: dict | None = None,
        factory_kwargs: dict | None = None,
        kpoints_settings: dict | KSampling | None = None,
        input_index: int | None = None,
    ) -> AbinitInput:
        """
        Generate an AbinitInput for the input set.

        Uses the defined factory function and additional parameters from user
        settings and subclasses to construct a complete ABINIT input.

        Parameters
        ----------
        structure : Structure or None
            A pymatgen Structure object. Default is None.
        pseudos : PseudoTable or None
            A pseudopotential table. Default is None.
        prev_outputs : list[str] or None
            A list of previous output directories. Default is None.
        abinit_settings : dict or None
            A dictionary with additional ABINIT keywords to set. Default is None.
        factory_kwargs : dict or None
            A dictionary with additional factory keywords to set. Default is None.
        kpoints_settings : dict or KSampling or None
            A dictionary or a KSampling object with additional settings
            for the k-points. Default is None.
        input_index : int or None
            The index to be used to select the AbinitInput in case a factory
            returns a MultiDataset. Default is None.

        Returns
        -------
        AbinitInput
            An AbinitInput object ready to be used in an AbinitInputSet.
        """
        total_factory_kwargs = dict(self.factory_kwargs) if self.factory_kwargs else {}
        if self.factory_prev_inputs_kwargs:
            if not prev_outputs:
                raise RuntimeError(
                    f"No previous_outputs. Required for {type(self).__name__}."
                )

            # Note: Consider supporting structure parameter when
            # factory_prev_inputs_kwargs is present for advanced use cases.
            if structure is not None:
                raise RuntimeError(
                    "Structure not supported if factory_prev_inputs_kwargs is defined"
                )

            abinit_inputs = self.resolve_prev_inputs(
                prev_outputs, self.factory_prev_inputs_kwargs
            )
            total_factory_kwargs.update(abinit_inputs)

        elif structure is None:
            msg = (
                f"Structure is mandatory for {type(self).__name__} "
                f"generation since no previous output is used."
            )
            raise RuntimeError(msg)

        if not self.prev_outputs_deps and prev_outputs:
            msg = (
                f"Previous outputs not allowed for {type(self).__name__}. "
                "Consider if restart_from argument of get_input_set method "
                "can fit your needs instead."
            )
            raise RuntimeError(msg)

        if structure:
            total_factory_kwargs["structure"] = structure
        if pseudos:
            total_factory_kwargs["pseudos"] = pseudos
        if factory_kwargs:
            total_factory_kwargs.update(factory_kwargs)

        generated_input = self.factory(**total_factory_kwargs)

        if input_index is not None:
            generated_input = generated_input[input_index]

        self._set_kpt_vars(generated_input, kpoints_settings)

        if abinit_settings:
            generated_input.set_vars(abinit_settings)
        if self.user_abinit_settings:
            generated_input.set_vars(self.user_abinit_settings)

        # Remove variables with None values to avoid issues when checking
        # if values are present in the input
        self._clean_none(generated_input)

        return generated_input

    def _set_kpt_vars(
        self,
        abinit_input: AbinitInput | MultiDataset,
        kpoints_settings: dict | KSampling | None,
    ) -> None:
        """
        Update the k-points variables according to the provided settings.

        Removes all existing k-point related variables from the input and sets
        new ones based on the provided k-points settings.

        Parameters
        ----------
        abinit_input : AbinitInput or MultiDataset
            An AbinitInput or MultiDataset object to be updated with new
            k-point variables.
        kpoints_settings : dict or KSampling or None
            The k-points settings to apply. Can be a dictionary with k-point
            configuration options or a KSampling object. If None, uses default
            settings.

        Returns
        -------
        None
            This method modifies the abinit_input in place.
        """
        ksampling = self._get_kpoints(abinit_input.structure, kpoints_settings)
        if ksampling:
            kpt_related_vars = [
                "kpt",
                "kptbounds",
                "kptnrm",
                "kptns",
                "kptns_hf",
                "kptopt",
                "kptrlatt",
                "kptrlen",
                "ndivk",
                "ndivsm",
                "ngkpt",
                "nkpath",
                "nkpt",
                "nshiftk",
                "shiftk",
                "wtk",
            ]
            abinit_input.pop_vars(kpt_related_vars)
            abinit_input.set_vars(**ksampling.abivars)

    @staticmethod
    def _clean_none(abinit_input: AbinitInput | MultiDataset) -> None:
        """
        Remove variables with None values from the AbinitInput.

        Iterates through all variables in the input and removes any that are
        set to None. This prevents issues when checking if values are present
        in the input.

        Parameters
        ----------
        abinit_input : AbinitInput or MultiDataset
            An AbinitInput or MultiDataset object to modify.

        Returns
        -------
        None
            This method modifies the abinit_input in place.
        """
        if not isinstance(abinit_input, MultiDataset):
            abinit_input = [abinit_input]

        for ai in abinit_input:
            for k, v in list(ai.items()):
                if v is None:
                    ai.remove_vars(k)

    def _get_kpoints(
        self,
        structure: Structure,
        kpoints_updates: dict[str, Any] | None,
    ) -> KSampling | None:
        """
        Generate a KSampling object based on k-points settings.

        Creates a k-points sampling configuration by combining user settings
        and any provided updates. Returns None if no k-points settings are
        specified.

        Parameters
        ----------
        structure : Structure
            Pymatgen Structure object for which to generate k-points.
        kpoints_updates : dict[str, Any] or None
            Dictionary with k-point configuration updates to apply. If None,
            only user_kpoints_settings will be used.

        Returns
        -------
        KSampling or None
            A KSampling object with the k-points configuration, or None if
            no k-points settings are specified.
        """
        kpoints_updates = {} if kpoints_updates is None else kpoints_updates
        if self.user_kpoints_settings == {} and not kpoints_updates:
            return None

        return get_ksampling(
            structure=structure,
            kpoints_updates=kpoints_updates,
            user_kpoints_settings=self.user_kpoints_settings,
            force_gamma=self.force_gamma,
            symprec=self.symprec,
        )


def _combine_kpoints(*kpoints_objects: KSampling) -> KSampling:
    """
    Combine multiple KSampling objects into a single KSampling object.

    Merges k-points and weights from multiple KSampling objects into one
    combined object. All input objects must use automatic mode.

    Parameters
    ----------
    *kpoints_objects : KSampling
        Variable number of KSampling objects to combine. None values are
        automatically filtered out.

    Returns
    -------
    KSampling
        A single KSampling object containing all k-points and weights from
        the input objects.

    Raises
    ------
    ValueError
        If any of the input KSampling objects does not have mode set to
        KSamplingModes.automatic.
    """
    kpoints = []
    weights = []

    for kpoints_object in filter(None, kpoints_objects):
        if not kpoints_object.mode == KSamplingModes.automatic:
            raise ValueError(
                "Can only combine k-points with mode=KSamplingModes.automatic"
            )

        weights.append(kpoints_object.kpts_weights)
        kpoints.append(kpoints_object.kpts)

    weights = np.concatenate(weights).tolist()
    kpoints = np.concatenate(kpoints)
    return KSampling(
        mode=KSamplingModes.automatic,
        num_kpts=len(kpoints),
        kpts=kpoints,
        kpts_weights=weights,
        comment="Combined k-points",
    )


def get_ksampling(
    structure: Structure,
    kpoints_updates: dict[str, Any] | None = None,
    user_kpoints_settings: dict | KSampling | None = None,
    force_gamma: bool = True,
    symprec: float = SETTINGS.SYMPREC,
) -> KSampling:
    """
    Generate a KSampling object for k-points configuration.

    Creates a k-points sampling configuration from user settings or updates.
    Supports various generation modes including line density along high
    symmetry paths, grid density, and reciprocal density with optional
    explicit k-point lists.

    Parameters
    ----------
    structure : Structure
        Pymatgen Structure object for which to generate k-points.
    kpoints_updates : dict[str, Any] or None
        Dictionary with k-point configuration updates. Default is None.
    user_kpoints_settings : dict or KSampling or None
        User k-points settings. Can be a dictionary with configuration
        options or a KSampling object. Default is None.
    force_gamma : bool
        Force gamma-centered k-point generation. Default is True.
    symprec : float
        Symmetry precision for determining irreducible k-points.
        Default is SETTINGS.SYMPREC.

    Returns
    -------
    KSampling
        A KSampling object with the k-points configuration.

    Raises
    ------
    ValueError
        If no k-point settings are defined or if k-point generation fails.
    """
    # Use user setting if provided, otherwise use updates
    if user_kpoints_settings != {}:
        kconfig = copy.deepcopy(user_kpoints_settings)
    elif kpoints_updates:
        kconfig = kpoints_updates
    else:
        raise ValueError("No k-point settings defined")

    if isinstance(kconfig, KSampling):
        return kconfig

    explicit = (
        kconfig.get("explicit")
        or len(kconfig.get("added_kpoints", [])) > 0
        or "zero_weighted_reciprocal_density" in kconfig
        or "zero_weighted_line_density" in kconfig
    )

    base_kpoints = None
    if kconfig.get("line_density"):
        # Generate k-points along high symmetry lines
        kpath = HighSymmKpath(structure, **kconfig.get("kpath_kwargs", {}))
        frac_k_points, _k_points_labels = kpath.get_kpoints(
            line_density=kconfig["line_density"], coords_are_cartesian=False
        )
        base_kpoints = KSampling(
            mode=KSamplingModes.automatic,
            num_kpts=len(frac_k_points),
            kpts=frac_k_points,
            kpts_weights=[1] * len(frac_k_points),
            comment="Non SCF run along symmetry lines",
        )
    elif kconfig.get("grid_density") or kconfig.get("reciprocal_density"):
        # Generate regular weighted k-point grid
        if kconfig.get("grid_density"):
            vasp_kpoints = Kpoints.automatic_density(
                structure, int(kconfig["grid_density"]), force_gamma
            )
            base_kpoints = KSampling(
                mode=KSamplingModes.monkhorst,
                num_kpts=0,
                kpts=vasp_kpoints.kpts,
                kpt_shifts=vasp_kpoints.kpts_shift,
                comment=vasp_kpoints.comment,
            )
        elif kconfig.get("reciprocal_density"):
            vasp_kpoints = Kpoints.automatic_density_by_vol(
                structure, kconfig["reciprocal_density"], force_gamma
            )
            base_kpoints = KSampling(
                mode=KSamplingModes.monkhorst,
                num_kpts=0,
                kpts=vasp_kpoints.kpts,
                kpt_shifts=vasp_kpoints.kpts_shift,
                comment=vasp_kpoints.comment,
            )
        if explicit:
            sga = SpacegroupAnalyzer(structure, symprec=symprec)
            mesh = sga.get_ir_reciprocal_mesh(base_kpoints.kpts[0])
            base_kpoints = KSampling(
                mode=KSamplingModes.automatic,
                num_kpts=len(mesh),
                kpts=[i[0] for i in mesh],
                kpts_weights=[i[1] for i in mesh],
                comment="Uniform grid",
            )
        else:
            # Return base k-points if no additional options specified
            return base_kpoints

    added_kpoints = None
    if kconfig.get("added_kpoints"):
        added_kpoints = KSampling(
            mode=KSamplingModes.automatic,
            num_kpts=len(kconfig.get("added_kpoints")),
            kpts=kconfig.get("added_kpoints"),
            kpts_weights=[0] * len(kconfig.get("added_kpoints")),
            comment="Specified k-points only",
        )

    if base_kpoints and not added_kpoints:
        return base_kpoints
    if added_kpoints and not base_kpoints:
        return added_kpoints

    # Validate that at least one k-point source exists
    if not (base_kpoints or added_kpoints):
        raise ValueError("Invalid k-point generation algorithm.")

    return _combine_kpoints(base_kpoints, added_kpoints)


def set_workdir(workdir: Path | str) -> tuple[Directory, Directory, Directory]:
    """
    Set up the working directory for an ABINIT calculation.

    Creates the necessary directory structure including standard input,
    output, and temporary directories.

    Parameters
    ----------
    workdir : Path or str
        Path to the working directory to set up.

    Returns
    -------
    tuple[Directory, Directory, Directory]
        A tuple containing (indir, outdir, tmpdir) Directory objects for
        input, output, and temporary data respectively.
    """
    workdir = os.path.abspath(workdir)

    # Directories with input, output, and temporary data
    indir = Directory(os.path.join(workdir, INDIR_NAME))
    outdir = Directory(os.path.join(workdir, OUTDIR_NAME))
    tmpdir = Directory(os.path.join(workdir, TMPDIR_NAME))

    # Create directories for input, output, and temporary data
    indir.makedirs()
    outdir.makedirs()
    tmpdir.makedirs()

    return indir, outdir, tmpdir
