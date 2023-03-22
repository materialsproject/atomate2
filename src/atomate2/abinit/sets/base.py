"""Module defining base abinit input set and generator."""
import copy
import json
import logging
import os
from collections import namedtuple
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

from abipy.abio.inputs import AbinitInput
from abipy.flowtk.psrepos import get_repo_from_name
from abipy.flowtk.utils import Directory, irdvars_for_ext
from monty.json import MontyEncoder
from pymatgen.core.structure import Structure
from pymatgen.io.abinit.abiobjects import KSampling
from pymatgen.io.abinit.pseudos import PseudoTable
from pymatgen.io.core import InputGenerator, InputSet

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

__all__ = ["AbinitInputSet", "AbinitInputGenerator"]

logger = logging.getLogger(__name__)


class AbinitInputSet(InputSet):
    """
    A class to represent a set of Abinit inputs.

    Parameters
    ----------
    abinit_input
        An AbinitInput object.
    input_files
        A list of input files needed for the calculation. The corresponding
        file reading variables (ird***) should be present in the abinit_input.
    """

    def __init__(
        self,
        abinit_input: AbinitInput,
        input_files: Optional[Iterable[Tuple[str, str]]] = None,
        link_files: bool = True,
    ):
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
        directory: Union[str, Path],
        make_dir: bool = True,
        overwrite: bool = True,
        zip_inputs: bool = False,
    ):
        """Write Abinit input files to a directory."""
        # TODO: do we allow zipping ? not sure if it really makes sense for abinit as
        #  the abinit input set also sets up links to previous files, sets up the
        #  indir, outdir and tmpdir, ...
        self.inputs["abinit_input.json"] = json.dumps(self.abinit_input.as_dict())
        super().write_input(
            directory=directory,
            make_dir=make_dir,
            overwrite=overwrite,
            zip_inputs=zip_inputs,
        )
        del self.inputs["abinit_input.json"]
        indir, outdir, tmpdir = self.set_workdir(workdir=directory)

        if self.input_files:
            out_to_in(
                out_files=self.input_files,
                indir=indir.path,
                link_files=self.link_files,
            )

    def validate(self) -> bool:
        """Validate the input set. Check that all files in the input directory
        have their corresponding ird variables."""
        if not self.input_files:
            return True
        for _out_filepath, in_file in self.input_files:
            ext = fname2ext(in_file)
            if ext is None:
                return False
            irdvars = irdvars_for_ext(ext)
            for irdvar, irdval in irdvars.items():
                if irdvar not in self.abinit_input:
                    return False
                if self.abinit_input[irdvar] != irdval:
                    return False
        return True

    @property
    def abinit_input(self):
        """Get the AbinitInput object."""
        return self[INPUT_FILE_NAME]

    @staticmethod
    def set_workdir(workdir):
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
        """Set the values of abinit variables.

        This sets the abinit variables in the abipy AbinitInput object.

        One can pass a dictionary mapping the abinit variables to their values or
        the abinit variables as keyword arguments. A combination of the two
        options is also allowed.

        Returns
        -------
        dict
            dictionary with the variables that have been added.
        """
        return self.abinit_input.set_vars(*args, **kwargs)

    def remove_vars(self, keys: Union[Iterable[str], str], strict: bool = True) -> dict:
        """Remove the abinit variables listed in keys.

        This removes the abinit variables from the abipy AbinitInput object.

        Parameters
        ----------
        keys
            string or list of strings with the names of the abinit variables
            to be removed.
        strict
            whether to raise a KeyError if one of the abinit variables to be
            removed is not present.

        Returns
        -------
        dict
            dictionary with the variables that have been removed.
        """
        return self.abinit_input.remove_vars(keys=keys, strict=strict)

    def runlevel(self):
        """Get the set of strings defining the calculation type."""
        return self.abinit_input.runlevel

    def set_structure(self, structure: Any) -> Structure:
        """Set the structure for this input set.

        This basically forwards the setting of the structure to the abipy
        AbinitInput object.
        """
        return self.abinit_input.set_structure(structure)

    def deepcopy(self):
        """Deep copy of the input set."""
        return copy.deepcopy(self)


def as_pseudo_table(pseudos):
    """Get the pseudos as a PseudoTable object.

    Parameters
    ----------
    pseudos
        Pseudopotentials as a list of pseudopotentials files, a single
        pseudopotential file, a string representing a pseudo repository.

    Returns
    -------
    PseudoTable
        Table of pseudopotentials.
    """
    # get the PseudoTable from the PseudoRepo
    if isinstance(pseudos, str):
        # in case a single path to a pseudopotential file has been passed
        if os.path.isfile(pseudos):
            return PseudoTable(pseudos)
        else:
            pseudo_repo_name, table_name = pseudos.rsplit(":", 1)
            repo = get_repo_from_name(pseudo_repo_name)
            if not repo.is_installed():
                msg = (
                    f"Pseudo repository {pseudo_repo_name} is not installed in {repo.dirpath}) "
                    f"Use abips.py to install it."
                )
                raise RuntimeError(msg)
            return repo.get_pseudos(table_name)
    return PseudoTable(pseudos)


def get_extra_abivars(extra_abivars, extra_mod):
    """Update and get extra_abivars.

    Parameters
    ----------
    extra_abivars
        Current additional abinit variables.
    extra_mod
        Modifications to the additional abinit variables. To remove one
        of the additional variables, set the variable to None.

    Returns
    -------
    dict
        The updated additional abinit variables.
    """
    extra_abivars.update(extra_mod)
    # Remove additional variables when their value is set to None
    return {k: v for k, v in extra_abivars.items() if v is not None}


PrevOutput = namedtuple("PrevOutput", "dirname exts")


@dataclass
class AbinitInputGenerator(InputGenerator):
    """A class to generate Abinit input sets."""

    calc_type: str = "abinit_calculation"

    pseudos: Union[str, List[str], PseudoTable] = "ONCVPSP-PBE-SR-PDv0.4:standard"

    extra_abivars: dict = field(default_factory=dict)

    restart_from_deps: Optional[Union[str, tuple]] = None
    prev_outputs_deps: Optional[Union[str, tuple]] = None

    @classmethod
    def from_prev_generator(cls, prev_input_generator, **kwargs):
        # Get the calc_type (current input generator or user-specified through kwargs)
        try:
            calc_type = kwargs.pop("calc_type")
        except KeyError:
            calc_type = cls.calc_type
        # Do not allow to change pseudopotentials
        if "pseudos" in kwargs:
            raise RuntimeError("Cannot change pseudos.")
        pseudos = prev_input_generator.pseudos
        # Get the additional abinit variables
        extra_abivars = prev_input_generator.extra_abivars or {}
        if "extra_abivars" in kwargs:
            extra_abivars = get_extra_abivars(
                extra_abivars=extra_abivars, extra_mod=kwargs.pop("extra_abivars")
            )
        # Update the parameters
        params = cls.get_params(
            instance_or_class=cls, kwargs=kwargs, prev_gen=prev_input_generator
        )
        return cls(
            calc_type=calc_type, pseudos=pseudos, extra_abivars=extra_abivars, **params
        )

    @staticmethod
    def get_params(instance_or_class, kwargs, prev_gen=None):
        """Get the parameters to generate the AbinitInputSet.

        It loops over all the generator's fields and gets the value of each parameter
        from the keyword arguments if it is there, then from the previous generator
        if it is provided and it has the attribute, then from the instance or class.

        Parameters
        ----------
        instance_or_class
        kwargs
        prev_gen

        Returns
        -------

        """
        params = {}
        for fld in fields(instance_or_class):
            param = fld.name
            if param in [
                "calc_type",
                "pseudos",
                "extra_abivars",
                "restart_from_deps",
                "prev_outputs_deps",
            ]:
                continue
            if param in kwargs:
                val = kwargs[param]
            elif prev_gen is not None and hasattr(prev_gen, param):
                val = prev_gen.__getattribute__(param)
            else:
                val = instance_or_class.__getattribute__(param)
            params[param] = val
        return params

    def get_input_set(  # type: ignore
        self,
        structure=None,
        restart_from=None,
        prev_outputs=None,
        **kwargs,
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
        # pseudos = self.pseudos
        # if "pseudos" in kwargs:
        #     pseudos = kwargs.pop("pseudos")
        pseudos = as_pseudo_table(self.pseudos)
        # extra_abivars = self.extra_abivars or {}
        # if "extra_abivars" in kwargs:
        #     extra_mod = kwargs.pop("extra_abivars")
        #     extra_abivars.update(extra_mod)
        #     # Remove additional variables when their value is set to None
        #     extra_abivars = {k: v for k, v in extra_abivars.items() if v is not None}

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
            params = self.get_params(
                instance_or_class=self, kwargs=kwargs, prev_gen=None
            )
            if prev_outputs is not None and not self.prev_outputs_deps:
                raise RuntimeError(
                    f"Previous outputs not allowed for {self.__class__.__name__}."
                )
            abinit_input = self.get_abinit_input(
                structure=structure,
                pseudos=pseudos,
                prev_outputs=prev_outputs,
                **params,
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
        abinit_input.set_vars(self.extra_abivars)

        if restart_from is not None:
            self.on_restart(abinit_input=abinit_input)

        abinit_input["indata_prefix"] = (f'"{INDATA_PREFIX}"',)
        abinit_input["outdata_prefix"] = (f'"{OUTDATA_PREFIX}"',)
        abinit_input["tmpdata_prefix"] = (f'"{TMPDATA_PREFIX}"',)

        # TODO: where/how do we set up/pass down link_files ?
        return AbinitInputSet(
            abinit_input=abinit_input,
            input_files=input_files,
            link_files=True,
        )

    def check_format_prev_dirs(self, prev_dirs):
        """Check and format the prev_dirs (restart or dependency)."""
        if prev_dirs is None:
            return None
        if isinstance(prev_dirs, (str, Path)):
            return [str(prev_dirs)]
        return [str(prev_dir) for prev_dir in prev_dirs]

    def resolve_deps(self, prev_dirs, deps, check_runlevel=True):
        """Resolve dependencies.

        This method assumes that prev_dirs is in the correct format, i.e.
        a list of directories as str or Path.
        """
        input_files = []
        deps_irdvars = {}
        for prev_dir in prev_dirs:
            if check_runlevel:
                abinit_input = load_abinit_input(prev_dir)
            for dep in deps:
                runlevel = set(dep.split(":")[0].split("|"))
                exts = tuple(dep.split(":")[1].split("|"))
                if not check_runlevel or runlevel.intersection(abinit_input.runlevel):
                    irdvars, inp_files = self.resolve_dep_exts(
                        prev_dir=prev_dir, exts=exts
                    )
                    input_files.extend(inp_files)
                    deps_irdvars.update(irdvars)

        return deps_irdvars, input_files

    @staticmethod
    def _get_in_file_name(out_filepath):
        in_file = os.path.basename(out_filepath)
        in_file = in_file.replace(OUTDATAFILE_PREFIX, INDATAFILE_PREFIX, 1)
        in_file = os.path.basename(in_file).replace("WFQ", "WFK", 1)
        return in_file

    @staticmethod
    def resolve_dep_exts(prev_dir, exts):
        """Return irdvars and corresponding file for a given dependency.

        This method assumes that prev_dir is in the correct format,
        i.e. a directory as a str or Path.
        """
        prev_outdir = Directory(os.path.join(prev_dir, OUTDIR_NAME))
        inp_files = []

        for ext in exts:
            # TODO: how to check that we have the files we need ?
            #  Should we raise if don't find at least one file for a given extension ?
            if ext in ("1WF", "1DEN"):
                # Special treatment for 1WF and 1DEN files
                if ext == "1WF":
                    files = prev_outdir.find_1wf_files()
                elif ext == "1DEN":
                    files = prev_outdir.find_1den_files()
                else:
                    raise RuntimeError("Should not occur.")
                if files is not None:
                    inp_files = (
                        (f.path, AbinitInputGenerator._get_in_file_name(f.path))
                        for f in files
                    )
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
                        (out_den, AbinitInputGenerator._get_in_file_name(out_den))
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
                        (out_file, AbinitInputGenerator._get_in_file_name(out_file))
                    )
                    break
        else:
            msg = f"Cannot find {' or '.join(exts)} file to restart from."
            logger.error(msg)
            raise InitializationError(msg)
        return irdvars, inp_files

    def get_abinit_input(
        self, structure=None, pseudos=None, prev_outputs=None, **params
    ):
        """Get AbinitInput object."""
        raise NotImplementedError

    def _get_shift_mode(self, shifts):
        if isinstance(shifts, str):
            return shifts
        else:
            return "Gamma"  # Dummy shift mode, shifts will be overwritten

    @staticmethod
    def _set_shifts_kpoints(abinit_input, structure, kppa, shifts):
        if not isinstance(shifts, str):
            ksampling = KSampling.automatic_density(
                structure, kppa, chksymbreak=0, shifts=shifts
            )
            abinit_input.set_vars(ksampling.to_abivars())

    def on_restart(self, abinit_input):
        """Perform updates of AbinitInput upon restart of a previous calculation."""
