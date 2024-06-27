"""Module defining base anaddb input set and generator."""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from abipy.abio.inputs import AbinitInput, MultiDataset, AnaddbInput
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
    ANADDB_INPUT_FILE_NAME,
    OUTDATA_PREFIX,
    OUTDATAFILE_PREFIX,
    OUTDIR_NAME,
    TMPDATA_PREFIX,
    TMPDIR_NAME,
    InitializationError,
    get_final_structure,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pymatgen.core.structure import Structure


logger = logging.getLogger(__name__)


class AnaddbInputSet(InputSet):
    """
    A class to represent a set of Anaddb inputs.

    Parameters
    ----------
    anaddb_input
        An AnaddbInput object.
    input_file
        A list with one input file (out_DDB) needed for the calculation.
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
        """Write Anaddb input files to a directory."""
        # TODO: do we allow zipping ? not sure if it really makes sense for abinit as
        #  the abinit input set also sets up links to previous files, sets up the
        #  indir, outdir and tmpdir, ...
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
        indir, _outdir, _tmpdir = self.set_workdir(workdir=directory)

        if self.input_files:
            out_to_in(
                out_files=self.input_files,
                indir=indir.path,
                link_files=self.link_files,
            )

    def validate(self) -> bool:
        """Validate the input set.

        Check that the input file exists and is a DDB file.
        """
        if not self.input_files:
            return False
        for _out_filepath, in_file in self.input_files:
            if not os.path.isfile(_out_filepath) or in_file != "in_DDB":
                return False
        return True

    @property
    def anaddb_input(self) -> AnaddbInput:
        """Get the AnaddbInput object."""
        return self[ANADDB_INPUT_FILE_NAME]

    @staticmethod
    def set_workdir(workdir: Path | str) -> tuple[Directory, Directory, Directory]:
        """Set up the working directory.

        This also sets up and creates standard input, output and temporary directories.
        """
        workdir = os.path.abspath(workdir)

        # Directories with input|output|temporary data.
        indir = Directory(os.path.join(workdir, INDIR_NAME))
        outdir = Directory(os.path.join(workdir, OUTDIR_NAME))
        tmpdir = Directory(os.path.join(workdir, TMPDIR_NAME))

        # Create dirs for input, output and tmp data.
        indir.makedirs()
        outdir.makedirs()
        tmpdir.makedirs()

        return indir, outdir, tmpdir

    def set_vars(self, *args, **kwargs) -> dict:
        """Set the values of anaddb variables.

        This sets the anaddb variables in the abipy AnaddbInput object.

        One can pass a dictionary mapping the anaddb variables to their values or
        the anaddb variables as keyword arguments. A combination of the two
        options is also allowed.

        Returns
        -------
        dict
            dictionary with the variables that have been added.
        """
        return self.anaddb_input.set_vars(*args, **kwargs)

    def remove_vars(self, keys: Iterable[str] | str, strict: bool = True) -> dict:
        """Remove the anaddb variables listed in keys.

        This removes the anaddb variables from the abipy AnaddbInput object.

        Parameters
        ----------
        keys
            string or list of strings with the names of the anaddb variables
            to be removed.
        strict
            whether to raise a KeyError if one of the anaddb variables to be
            removed is not present.

        Returns
        -------
        dict
            dictionary with the variables that have been removed.
        """
        return self.anaddb_input.remove_vars(keys=keys, strict=strict)


    def set_structure(self, structure: Any) -> Structure:
        """Set the structure for this input set.

        This basically forwards the setting of the structure to the abipy
        AnaddbInput object.
        """
        return self.anaddb_input.set_structure(structure)

    def deepcopy(self) -> AnaddbInputSet:
        """Deep copy of the input set."""
        return copy.deepcopy(self)


@dataclass
class AnaddbInputGenerator(AbiBroadInputGenerator):
    """
    A class to generate Anaddb input sets.

    Parameters
    ----------
    calc_type
        A short description of the calculation type
    user_anaddb_settings
        A dictionary that allows to set any Abinit variable in the AbinitInput
        after it has been generated from the factory function. This will override
        any value or default previously set. Set a value to None to remove it
        from the input.
    user_kpoints_settings
        Allow user to override kpoints setting by supplying a dict. E.g.,
        ``{"reciprocal_density": 1000}``. User can also supply a KSampling object.
    restart_from_deps:
        Defines the files that needs to be linked from previous calculations in
        case of restart. The format is a tuple where each element is a list of
        "|" separated run levels (as defined in the AbinitInput object) followed
        by a colon and a list of "|" list of extensions of files that needs to
        be linked. The runlevel defines the type of calculations from which the
        file can be linked. An example is (f"{NSCF}:WFK",).
    prev_outputs_deps
        Defines the files that needs to be linked from previous calculations and
        are required for the execution of the current calculation.
        The format is a tuple where each element is a list of  "|" separated
        run levels (as defined in the AbinitInput object) followed by a colon and
        a list of "|" list of extensions of files that needs to be linked.
        The runlevel defines the type of calculations from which the file can
        be linked. An example is (f"{NSCF}:WFK",).
    factory_prev_inputs_kwargs
        A dictionary defining the source of the of one or more previous
        AbinitInput in case they are required by a factory to build a new
        AbinitInput. The key should match the name of the argument of the factory
        function and the value should be a tuple with the runlevels of the
        compatible types of AbinitInput that can be used.
    force_gamma
        Force gamma centered kpoint generation.
    symprec
        Tolerance for symmetry finding, used for line mode band structure k-points.
    """

    factory: Callable | None = None
    calc_type: str = "abinit_calculation"
    pseudos: str | list[str] | PseudoTable | None = "ONCVPSP-PBE-SR-PDv0.4:standard"
    factory_kwargs: dict = field(default_factory=dict)
    user_abinit_settings: dict = field(default_factory=dict)
    user_kpoints_settings: dict | KSampling = field(default_factory=dict)
    restart_from_deps: str | tuple | None = None
    prev_outputs_deps: str | tuple | None = None
    factory_prev_inputs_kwargs: dict | None = None
    force_gamma: bool = True
    symprec: float = SETTINGS.SYMPREC

    def get_input_set(
        self,
        structure: Structure = None,
        restart_from: str | tuple | list | Path | None = None,
        prev_outputs: str | tuple | list | Path | None = None,
    ) -> AbinitInputSet:
        """Generate an AbinitInputSet object.

        Here we assume that restart_from is a directory and prev_outputs is
        a list of directories. We also assume there is an abinit_input.json file
        in each of these directories containing the AbinitInput object used to
        execute abinit.

        Parameters
        ----------
        structure : Structure
            Pymatgen Structure object.
        restart_from : str or Path or list or tuple
            Directory (as a str or Path) or list/tuple of 1 directory (as a str
            or Path) to restart from.
        prev_outputs : str or Path or list or tuple
            Directory (as a str or Path) or list/tuple of directories (as a str
            or Path) needed as dependencies for the AbinitInputSet generated.
        """
        # Get the pseudos as a PseudoTable
        pseudos = as_pseudo_table(self.pseudos) if self.pseudos else None

        restart_from = self.check_format_prev_dirs(restart_from)
        prev_outputs = self.check_format_prev_dirs(prev_outputs)

        all_irdvars = {}
        input_files = []
        if restart_from is not None:
            # Use the previous abinit input
            abinit_input = load_abinit_input(restart_from[0])
            # Update with the abinit input with the final structure
            structure = get_final_structure(restart_from[0])
            abinit_input.set_structure(structure=structure)
            # Files for restart (e.g. continue a not yet converged
            # scf/nscf/relax calculation)
            irdvars, files = self.resolve_deps(
                restart_from, deps=self.restart_from_deps
            )
            all_irdvars.update(irdvars)
            input_files.extend(files)
        else:
            if prev_outputs is not None and not self.prev_outputs_deps:
                raise RuntimeError(
                    f"Previous outputs not allowed for {self.__class__.__name__}."
                )
            abinit_input = self.get_abinit_input(
                structure=structure,
                pseudos=pseudos,
                prev_outputs=prev_outputs,
            )
        # Always reset the ird variables.
        abinit_input.pop_irdvars()

        # Files that are dependencies (e.g. band structure calculations
        # need the density).
        if prev_outputs:
            irdvars, files = self.resolve_deps(prev_outputs, self.prev_outputs_deps)
            all_irdvars.update(irdvars)
            input_files.extend(files)

        # Set ird variables and extra variables.
        abinit_input.set_vars(all_irdvars)
        abinit_input.set_vars(self.user_abinit_settings)

        abinit_input["indata_prefix"] = (f'"{INDATA_PREFIX}"',)
        abinit_input["outdata_prefix"] = (f'"{OUTDATA_PREFIX}"',)
        abinit_input["tmpdata_prefix"] = (f'"{TMPDATA_PREFIX}"',)

        # TODO: where/how do we set up/pass down link_files ?
        return AbinitInputSet(
            abinit_input=abinit_input,
            input_files=input_files,
            link_files=True,
        )


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
        Generate the AbinitInput for the input set.

        Uses the defined factory function and additional parameters from user
        and subclasses.

        Parameters
        ----------
        structure
            A structure.
        pseudos
            A pseudopotential table.
        prev_outputs
            A list of previous output directories.
        abinit_settings
            A dictionary with additional abinit keywords to set.
        factory_kwargs
            A dictionary with additional factory keywords to set.
        kpoints_settings
            A dictionary or a KSampling object with additional settings
            for the k-points.
        input_index
            The index to be used to select the AbinitInput in case a factory
            returns a MultiDataset.

        Returns
        -------
            An AbinitInput
        """
        total_factory_kwargs = dict(self.factory_kwargs) if self.factory_kwargs else {}
        if self.factory_prev_inputs_kwargs:
            if not prev_outputs:
                raise RuntimeError(
                    f"No previous_outputs. Required for {self.__class__.__name__}."
                )

            # TODO consider cases where structure might be defined even if
            # factory_prev_inputs_kwargs is present.
            if structure is not None:
                raise RuntimeError(
                    "Structure not supported if factory_prev_inputs_kwargs is defined"
                )

            abinit_inputs = self.resolve_prev_inputs(
                prev_outputs, self.factory_prev_inputs_kwargs
            )
            total_factory_kwargs.update(abinit_inputs)

        else:
            # TODO check if this should be removed or the check be improved
            if structure is None:
                msg = (
                    f"Structure is mandatory for {self.__class__.__name__} "
                    f"generation since no previous output is used."
                )
                raise RuntimeError(msg)

        if not self.prev_outputs_deps and prev_outputs:
            msg = (
                f"Previous outputs not allowed for {self.__class__.__name__} "
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

        # remove the None values. They will not be printed in the input file
        # but can cause issues when checking if the values are present in the input.
        self._clean_none(generated_input)

        return generated_input

