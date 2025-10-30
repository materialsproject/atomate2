"""Module defining base ANADDB input set and generator."""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from abipy.abio.inputs import AnaddbInput
from abipy.dfpt.ddb import DdbFile
from monty.json import MontyEncoder, jsanitize
from pymatgen.io.core import InputSet

from atomate2.abinit.files import out_to_in
from atomate2.abinit.sets.base import AbinitMixinInputGenerator, set_workdir
from atomate2.abinit.utils.common import ANADDB_INPUT_FILE_NAME, INDATA_PREFIX

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from pymatgen.core.structure import Structure


logger = logging.getLogger(__name__)

__all__ = [
    "AnaddbDfptDteInputGenerator",
    "AnaddbInputGenerator",
    "AnaddbInputSet",
    "AnaddbPhbandsDOSInputGenerator",
    "anaddbinp_dfpt_dte",
    "anaddbinp_phbands_dos",
]


class AnaddbInputSet(InputSet):
    """
    A class to represent a set of ANADDB inputs.

    Parameters
    ----------
    anaddb_input : AnaddbInput
        An AnaddbInput object.
    input_files : Iterable[tuple[str, str]] or None
        A list of (source_path, dest_name) tuples for input files needed
        for the calculation (e.g., out_DDB). Default is None.
    link_files : bool
        Whether to link files (True) or copy them (False). Default is True.
    """

    def __init__(
        self,
        anaddb_input: AnaddbInput,
        input_files: Iterable[tuple[str, str]] | None = None,
        link_files: bool = True,
    ) -> None:
        self.input_files = input_files
        self.link_files = link_files
        super().__init__(
            inputs={
                ANADDB_INPUT_FILE_NAME: anaddb_input,
                "anaddb_input.json": json.dumps(
                    anaddb_input.as_dict(), cls=MontyEncoder
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
        Write ANADDB input files to a directory.

        Parameters
        ----------
        directory : str or Path
            Directory to write input files to.
        make_dir : bool
            Whether to create the directory if it does not exist. Default is True.
        overwrite : bool
            Whether to overwrite existing files. Default is True.
        zip_inputs : bool
            Whether to zip the input files. Default is False.
        """
        # Note: zipping may not be practical for ABINIT/ANADDB as the input set
        # manages links to previous files and sets up indir, outdir, and tmpdir
        self.inputs["anaddb_input.json"] = json.dumps(
            jsanitize(self.anaddb_input.as_dict())
        )
        super().write_input(
            directory=directory,
            make_dir=make_dir,
            overwrite=overwrite,
            zip_inputs=zip_inputs,
        )
        del self.inputs["anaddb_input.json"]
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

        Checks that required input files exist and include a DDB file.

        Returns
        -------
        bool
            True if validation passes, False otherwise.
        """
        if not self.input_files:
            return False
        for _out_filepath, in_file in self.input_files:
            if not os.path.isfile(_out_filepath) or in_file != "in_DDB":
                return False
        return True

    @property
    def anaddb_input(self) -> AnaddbInput:
        """
        Get the AnaddbInput object.

        Returns
        -------
        AnaddbInput
            The AnaddbInput object for this input set.
        """
        return self[ANADDB_INPUT_FILE_NAME]

    def set_vars(self, *args, **kwargs) -> dict:
        """
        Set the values of ANADDB variables.

        Sets ANADDB variables in the abipy AnaddbInput object. Variables can be
        passed as a dictionary or as keyword arguments, or a combination of both.

        Parameters
        ----------
        *args
            Positional arguments, typically a dictionary of variable names and values.
        **kwargs
            Keyword arguments where keys are variable names and values are the
            corresponding values to set.

        Returns
        -------
        dict
            Dictionary with the variables that have been added.
        """
        return self.anaddb_input.set_vars(*args, **kwargs)

    def remove_vars(self, keys: Iterable[str] | str, strict: bool = True) -> dict:
        """
        Remove ANADDB variables from the input.

        Removes the specified ANADDB variables from the abipy AnaddbInput object.

        Parameters
        ----------
        keys : str or Iterable[str]
            String or iterable of strings with the names of the ANADDB variables
            to remove.
        strict : bool
            Whether to raise a KeyError if a variable to be removed is not present.
            Default is True.

        Returns
        -------
        dict
            Dictionary with the variables that have been removed.
        """
        return self.anaddb_input.remove_vars(keys=keys, strict=strict)

    def set_structure(self, structure: Structure) -> Structure:
        """
        Set the structure for this input set.

        This forwards the structure setting to the abipy AnaddbInput object.

        Parameters
        ----------
        structure : Structure
            Pymatgen Structure object to set.

        Returns
        -------
        Structure
            The structure that was set.
        """
        return self.anaddb_input.set_structure(structure)

    def deepcopy(self) -> AnaddbInputSet:
        """
        Create a deep copy of the input set.

        Returns
        -------
        AnaddbInputSet
            A deep copy of this input set.
        """
        return copy.deepcopy(self)


@dataclass
class AnaddbInputGenerator(AbinitMixinInputGenerator):
    """
    Generator for ANADDB input sets.

    This class generates ANADDB input sets using a factory function and handles
    dependencies on previous calculations (typically DDB files from DFPT runs).

    Parameters
    ----------
    factory : Callable or None
        Factory function to generate the AnaddbInput. Default is None.
    calc_type : str
        Short description of the calculation type. Default is "anaddb".
    factory_kwargs : dict
        Dictionary of additional keyword arguments passed to the factory function.
        Default is an empty dict.
    user_abinit_settings : dict
        Dictionary to override any ANADDB variable after generation from the
        factory function. Set a value to None to remove it from the input.
        Default is an empty dict.
    prev_outputs_deps : tuple or None
        Defines files that need to be linked from previous calculations.
        Format is a tuple where each element is "|"-separated run levels
        followed by a colon and "|"-separated file extensions.
        Example: ("MRGDDB:DDB",). Default is ("MRGDDB:DDB",).
    """

    factory: Callable | None = None
    calc_type: str = "anaddb"
    factory_kwargs: dict = field(default_factory=dict)
    user_abinit_settings: dict = field(default_factory=dict)
    prev_outputs_deps: tuple | None = field(default_factory=lambda: ("MRGDDB:DDB",))

    def get_input_set(
        self,
        structure: Structure,
        prev_outputs: str | tuple | list | Path | None = None,
    ) -> AnaddbInputSet:
        """
        Generate an AnaddbInputSet object.

        Assumes prev_outputs is a directory or list of directories. Each directory
        should contain an abinit_input.json file with the AbinitInput object used
        to execute ABINIT.

        Parameters
        ----------
        structure : Structure
            Pymatgen Structure object.
        prev_outputs : str or Path or list or tuple or None
            Directory or list/tuple of directories (as str or Path) needed as
            dependencies for the AnaddbInputSet. Default is None.

        Returns
        -------
        AnaddbInputSet
            The generated AnaddbInputSet object.
        """
        prev_outputs = self.check_format_prev_dirs(prev_outputs)

        input_files = []
        if prev_outputs is not None and not self.prev_outputs_deps:
            raise RuntimeError(
                f"Previous outputs not allowed for {self.__class__.__name__}."
            )
        _irdvars, files = self.resolve_deps(
            prev_outputs, self.prev_outputs_deps, check_runlevel=False
        )
        input_files.extend(files)
        anaddb_input = self.get_anaddb_input(
            structure=structure,
            prev_outputs=prev_outputs,
            input_files=input_files,
        )

        # Set the DDB file path for ANADDB to read
        anaddb_input.set_vars({"ddb_filepath": f'"{INDATA_PREFIX}_DDB"'})

        return AnaddbInputSet(
            anaddb_input=anaddb_input,
            input_files=input_files,
        )

    def get_anaddb_input(
        self,
        structure: Structure | None = None,
        prev_outputs: list[str] | None = None,
        abinit_settings: dict | None = None,
        factory_kwargs: dict | None = None,
        input_files: list | None = None,
    ) -> AnaddbInput:
        """
        Generate the AnaddbInput for the input set.

        Uses the defined factory function and additional parameters from user
        and subclasses.

        Parameters
        ----------
        structure : Structure or None
            Pymatgen Structure object. Default is None.
        prev_outputs : list[str] or None
            List of previous output directories. Default is None.
        abinit_settings : dict or None
            Dictionary with additional ANADDB settings to set. Default is None.
        factory_kwargs : dict or None
            Dictionary with additional factory keywords to set. Default is None.
        input_files : list or None
            List of input files. Default is None.

        Returns
        -------
        AnaddbInput
            The generated AnaddbInput object.
        """
        total_factory_kwargs = dict(self.factory_kwargs) if self.factory_kwargs else {}

        if not self.prev_outputs_deps and prev_outputs:
            msg = f"Previous outputs not allowed for {self.__class__.__name__} "
            raise RuntimeError(msg)

        if structure:
            total_factory_kwargs["structure"] = structure
        if factory_kwargs:
            total_factory_kwargs.update(factory_kwargs)

        generated_input = self.factory(**total_factory_kwargs)

        if abinit_settings:
            generated_input.set_vars(abinit_settings)
        if self.user_abinit_settings:
            generated_input.set_vars(self.user_abinit_settings)

        return generated_input


def anaddbinp_dfpt_dte(
    structure: Structure, anaddb_kwargs: dict | None = None
) -> AnaddbInput:
    """
    Generate ANADDB input to compute the static SHG tensor from DTE.

    Parameters
    ----------
    structure : Structure
        Pymatgen Structure object.
    anaddb_kwargs : dict or None
        Dictionary with additional ANADDB keywords to set. Default is None.

    Returns
    -------
    AnaddbInput
        The generated AnaddbInput object for DTE analysis.
    """
    return AnaddbInput.dfpt(structure=structure, dte=True, anaddb_kwargs=anaddb_kwargs)


@dataclass
class AnaddbDfptDteInputGenerator(AnaddbInputGenerator):
    """
    Generator for ANADDB inputs to compute the static SHG tensor.

    This class generates ANADDB input files for post-processing DFPT calculations
    to extract the static second harmonic generation (SHG) tensor from DTE data.

    Parameters
    ----------
    factory : Callable
        Callable to generate the AnaddbInput for DTE DFPT. Default is
        anaddbinp_dfpt_dte.
    """

    factory: Callable = anaddbinp_dfpt_dte


def anaddbinp_phbands_dos(
    structure: Structure,
    **kwargs,
) -> AnaddbInput:
    """
    Generate ANADDB input for phonon band structure and DOS calculations.

    Parameters
    ----------
    structure : Structure
        Pymatgen Structure object.
    **kwargs
        Additional keyword arguments passed to AnaddbInput.phbands_and_dos.

    Returns
    -------
    AnaddbInput
        The generated AnaddbInput object for phonon bands and DOS.
    """
    return AnaddbInput.phbands_and_dos(structure=structure, **kwargs)


@dataclass
class AnaddbPhbandsDOSInputGenerator(AnaddbInputGenerator):
    """
    Generator for ANADDB inputs for phonon band structure and DOS calculations.

    This class generates ANADDB input files for computing phonon band structures
    and density of states (DOS) from DFPT calculations.

    Parameters
    ----------
    factory : Callable
        Callable to generate the AnaddbInput for phonon bands and DOS. Default is
        anaddbinp_phbands_dos.
    factory_kwargs : dict
        Dictionary with factory keywords. Default includes {"nqsmall": 15}.
    """

    factory: Callable = anaddbinp_phbands_dos
    factory_kwargs: dict = field(default_factory=lambda: {"nqsmall": 15})

    def get_anaddb_input(
        self,
        structure: Structure | None = None,
        prev_outputs: list[str] | None = None,
        abinit_settings: dict | None = None,
        factory_kwargs: dict | None = None,
        input_files: list | None = None,
    ) -> AnaddbInput:
        """
        Generate AnaddbInput, determining ngqpt from the DFPT DDB file.

        Parameters
        ----------
        structure : Structure or None
            Pymatgen Structure object. Default is None.
        prev_outputs : list[str] or None
            List of previous output directories. Default is None.
        abinit_settings : dict or None
            Dictionary with additional ANADDB settings. Default is None.
        factory_kwargs : dict or None
            Dictionary with additional factory keywords. Default is None.
        input_files : list or None
            List of input files. Default is None.

        Returns
        -------
        AnaddbInput
            The generated AnaddbInput object with ngqpt automatically determined
            from the DDB file if not explicitly provided.
        """
        factory_kwargs = factory_kwargs or {}

        if "ngqpt" not in factory_kwargs and "ngqpt" not in self.factory_kwargs:
            # Resolve dependencies to access the DDB file
            _irdvars, _files = self.resolve_deps(
                prev_outputs, self.prev_outputs_deps, check_runlevel=False
            )
            for infile in input_files:
                if infile[0].endswith("_DDB"):
                    ddb = DdbFile(infile[0])
                    factory_kwargs["ngqpt"] = ddb.guessed_ngqpt
                    break
            else:
                raise RuntimeError("Could not determine the DDB file to read ngqpt")

        return super().get_anaddb_input(
            structure=structure,
            prev_outputs=prev_outputs,
            abinit_settings=abinit_settings,
            factory_kwargs=factory_kwargs,
        )
