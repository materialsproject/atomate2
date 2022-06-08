"""Module defining base abinit input set and generator."""
import copy
import json
import logging
import os
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable, List, Optional, Union

from abipy.abio.input_tags import ION_RELAX, IONCELL_RELAX, MOLECULAR_DYNAMICS, RELAX
from abipy.abio.inputs import AbinitInput
from abipy.electrons.gsr import GsrFile
from abipy.flowtk.psrepos import get_repo_from_name
from abipy.flowtk.utils import Directory, irdvars_for_ext
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.io.abinit.pseudos import PseudoTable
from pymatgen.io.core import InputGenerator, InputSet

from atomate2.abinit.files import (
    fname2ext,
    load_abinit_input,
    load_generator,
    out_to_in,
)
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
)

__all__ = ["AbinitInputSet", "AbinitInputSetGenerator"]

logger = logging.getLogger(__name__)


class AbinitInputSet(InputSet):
    """
    A class to represent a set of Abinit inputs.

    Parameters
    ----------
    abinit_input
        An AbinitInput object.
    """

    def __init__(
        self,
        abinit_input: AbinitInput,
        input_files: Optional[Iterable[Union[str, Path, dict]]] = None,
        link_files: bool = True,
        validation: bool = True,
        generator: Optional[MSONable] = None,
    ):
        self.input_files = input_files
        self.link_files = link_files
        self.validation = validation
        self.generator = generator
        super().__init__(inputs={INPUT_FILE_NAME: abinit_input})

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
        if self.generator:
            gendict = self.generator.as_dict()
            self.inputs["abinit_input_set_generator.json"] = json.dumps(gendict)
        super().write_input(
            directory=directory,
            make_dir=make_dir,
            overwrite=overwrite,
            zip_inputs=zip_inputs,
        )
        del self.inputs["abinit_input.json"]
        if self.generator:
            del self.inputs["abinit_input_set_generator.json"]
        self.indir, self.outdir, self.tmpdir = self.set_workdir(workdir=directory)

        if self.input_files:
            out_to_in(
                out_files=self.input_files,
                indir=self.indir.path,
                link_files=self.link_files,
            )

        if self.validation:
            self.validate()

    def validate(self):
        # Check that all files in the input directory have their corresponding
        # ird variables.
        for filename in os.listdir(self.indir.path):
            ext = fname2ext(filename)
            if ext is None:
                raise ValueError(
                    f"'{filename}' file in input directory does not have a "
                    f"valid abinit extension."
                )
            irdvars = irdvars_for_ext(ext)
            for irdvar, irdval in irdvars.items():
                if irdvar not in self.abinit_input:
                    raise ValueError(
                        f"'{irdvar}' ird variable not set for '{filename}' file."
                    )
                if self.abinit_input[irdvar] != irdval:
                    raise ValueError(
                        f"'{irdvar} {self.abinit_input[irdvar]}' ird variable is wrong for "
                        f"'{filename}' file. Should be '{irdvar} {irdval}'."
                    )

    @property
    def abinit_input(self):
        return self[INPUT_FILE_NAME]

    @staticmethod
    def set_workdir(workdir):
        """Set up the working directory.

        This also sets up and creates standard input, output and temporary
        directories, as well as standard file names for input and output.
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


PrevOutput = namedtuple("PrevOutput", "dirname exts")


@dataclass
class AbinitInputSetGenerator(InputGenerator):
    """A class to generate Abinit input sets."""

    calc_type: str = "abinit_calculation"

    pseudos: Union[str, List[str], PseudoTable] = "ONCVPSP-PBE-SR-PDv0.4:standard"

    extra_abivars: dict = field(default_factory=dict)

    restart_from_deps: Optional[Union[str, tuple]] = None
    prev_outputs_deps: Optional[Union[str, tuple]] = None

    # Register of parameters that have been explicitly set in this
    # AbinitInputSetGenerator. This is used internally when "joining" two
    # generators to know when to take the value of a parameter from the
    # current generator or from the previous one.
    _params_set: set = field(
        default_factory=set, init=False, repr=False, hash=False, compare=False
    )

    # class variables
    params: ClassVar[tuple] = ()
    _tmpcls_params_set: ClassVar[set] = set()

    def __new__(cls, *args, **kwargs):
        """Set up the register of parameters explicitly set.

        Note that due to how dataclasses are implemented, it is not possible
        to perform this in an overridden __init__. A temporary class variable
        is used to register the parameters that are explicitly set. This
        temporary class variable is then set to the instance variable in
        the __post_init__ method, together with the reset of the temporary
        class variable.
        """
        for kwarg in kwargs:
            if kwarg in cls.params:
                cls._tmpcls_params_set.add(kwarg)
        return super().__new__(cls)

    def __post_init__(self):
        """Post init setting of the parameters explicitly set.

        The temporary class variable is also reset (see __new__).
        """
        self._params_set = set(self._tmpcls_params_set)
        self.__class__._tmpcls_params_set = set()

    def __setattr__(self, key, value):
        """Set a given attribute and register that it is explicitly set."""
        if key in self.params:
            self._params_set.add(key)
        super().__setattr__(key, value)

    def _get_parameters(self, kwargs, prev_generator):
        params = {}
        for param in self.params:
            # If the parameter is in the kwargs, take the value from there.
            if param in kwargs:
                val = kwargs[param]
            else:
                # If there is no previous generator, take the value from this
                # generator directly.
                if prev_generator is None:
                    val = self.__getattribute__(param)
                else:
                    # If the parameter is explicitly set in this generator,
                    # use that value. Otherwise, take the value from the previous
                    # generator.
                    if param in self._params_set:
                        val = self.__getattribute__(param)
                    else:
                        val = prev_generator.__getattribute__(param)
            params[param] = val
        # Take extra_abivars from the previous generator if available.
        extra_abivars = (
            prev_generator.extra_abivars if prev_generator is not None else {}
        )
        # Update extra_abivars from this generator.
        extra_abivars.update(self.extra_abivars)
        # Update extra_abivars from kwargs if applicable.
        extra_abivars.update(kwargs.get("extra_abivars", {}))
        return params, extra_abivars

    def _get_generator(self, gen_params, extra_abivars):
        generator = copy.copy(self)
        generator._params_set = set()
        for param in self.params:
            if param in gen_params:
                generator.__setattr__(param, gen_params[param])
        generator.extra_abivars = extra_abivars
        return generator

    def param_is_explicitly_set(self, param):
        """Check if a given parameter has been explicitly set.

        Parameters
        ----------
        param : str
            Name of parameter.

        Raises
        ------
        LookupError
            If the parameter is not registered.
        """
        if param not in self.params:
            raise LookupError(f'Parameter "{param}" is not registered.')
        return param in self._params_set

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
        pseudos = kwargs.get("pseudos", self.pseudos)

        # get the PseudoTable from the PseudoRepo
        if isinstance(pseudos, str):
            # in case a single path to a pseudopotential file has been passed
            if os.path.isfile(pseudos):
                pseudos = [pseudos]
            else:
                pseudo_repo_name, table_name = pseudos.rsplit(":", 1)
                repo = get_repo_from_name(pseudo_repo_name)
                if not repo.is_installed():
                    msg = (
                        f"Pseudo repository {pseudo_repo_name} is not installed in {repo.dirpath}) "
                        f"Use abips.py to install it."
                    )
                    raise RuntimeError(msg)
                pseudos = repo.get_pseudos(table_name)

        restart_from = self.check_format_prev_dirs(restart_from)
        prev_outputs = self.check_format_prev_dirs(prev_outputs)

        prev_generator = None
        all_irdvars = {}
        input_files = []
        if restart_from is not None:
            if self.restart_from_deps is None:
                raise RuntimeError(f"Restart not allowed for {self.__class__.__name__}")
            if len(restart_from) > 1:
                raise RuntimeError("Restart from multiple jobs is not possible.")
            prev_abinit_input = load_abinit_input(restart_from[0])
            prev_generator = load_generator(restart_from[0])
            allow_restart_from = set(self.restart_from_deps[0].split(":")[0].split("|"))
            if len(allow_restart_from.intersection(prev_abinit_input.runlevel)) == 0:
                raise RuntimeError(
                    f"Restart is not allowed. "
                    f'For "{self.__class__.__name__}" input generator, the '
                    f"allowed previous calculations for restart are: "
                    f'{" ".join(allow_restart_from)}'
                )
            if (
                len(
                    {RELAX, ION_RELAX, IONCELL_RELAX, MOLECULAR_DYNAMICS}.intersection(
                        prev_abinit_input.runlevel
                    )
                )
                > 0
            ):
                gsr_path = Directory(
                    os.path.join(restart_from[0], OUTDIR_NAME)
                ).has_abiext("GSR")
                if not gsr_path:
                    raise RuntimeError(
                        "Cannot extract structure from previous directory."
                    )
                try:
                    gsr_file = GsrFile(gsr_path)
                except Exception as exc:
                    msg = "Exception while reading GSR file at %s:\n%s" % (
                        gsr_path,
                        str(exc),
                    )
                    raise RuntimeError(msg)
                structure = gsr_file.structure
                for prop in structure.site_properties:
                    structure.remove_site_property(prop)
            else:
                structure = prev_abinit_input.structure
            # Files for restart (e.g. continue a not yet converged
            # scf/nscf/relax calculation)
            irdvars, files = self.resolve_deps(
                restart_from, deps=self.restart_from_deps
            )
            all_irdvars.update(irdvars)
            input_files.extend(files)

        gen_params, extra_abivars = self._get_parameters(kwargs, prev_generator)

        abinit_input = self.get_abinit_input(
            structure=structure,
            pseudos=pseudos,
            prev_outputs=prev_outputs,
            **gen_params,
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
        abinit_input.set_vars(**extra_abivars)

        if restart_from is not None:
            self.on_restart(abinit_input=abinit_input)

        abinit_input["indata_prefix"] = (f'"{INDATA_PREFIX}"',)
        abinit_input["outdata_prefix"] = (f'"{OUTDATA_PREFIX}"',)
        abinit_input["tmpdata_prefix"] = (f'"{TMPDATA_PREFIX}"',)

        # Get the generator used with all parameters and extra variables combined.
        generator = self._get_generator(gen_params, extra_abivars)

        return AbinitInputSet(
            abinit_input=abinit_input,
            input_files=input_files,
            generator=generator,
        )

    def check_format_prev_dirs(self, prev_dirs):
        """Check and format the prev_dirs (restart or dependency)."""
        if prev_dirs is None:
            return None
        if isinstance(prev_dirs, (str, Path)):
            return [str(prev_dirs)]
        if not isinstance(prev_dirs, (list, tuple)):
            raise RuntimeError(
                "Previous directories should be provided as a list "
                "or tuple of str or a single str."
            )
        for prev_dir in prev_dirs:
            if not isinstance(prev_dir, (str, Path)):
                raise RuntimeError("Previous directory should be a str or a Path.")
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
    def resolve_dep_exts(prev_dir, exts):
        """Return irdvars and corresponding file for a given dependency.

        This method assumes that prev_dir is in the correct format,
        i.e. a directory as a str or Path.
        """
        prev_outdir = Directory(os.path.join(prev_dir, OUTDIR_NAME))
        inp_files = []

        for ext in exts:
            if ext in ("1WF", "1DEN"):
                # Special treatment for 1WF and 1DEN files
                if ext == "1WF":
                    files = prev_outdir.find_1wf_files()
                elif ext == "1DEN":
                    files = prev_outdir.find_1den_files()
                else:
                    raise RuntimeError("Should not occur.")
                if files is not None:
                    inp_files = [f.path for f in files]
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
                    inp_files.append(out_den)
                    break
                last_timden = prev_outdir.find_last_timden_file()
                if last_timden is not None:
                    if last_timden.path.endswith(".nc"):
                        in_file_name = f"{INDATAFILE_PREFIX}_DEN.nc"
                    else:
                        in_file_name = f"{INDATAFILE_PREFIX}_DEN"
                    inp_files.append({last_timden.path: in_file_name})
                    irdvars = irdvars_for_ext("DEN")
                    break
            else:
                inp_file = prev_outdir.has_abiext(ext)
                irdvars = irdvars_for_ext(ext)
                if inp_file:
                    if ext == "WFQ":
                        inp_files.append({inp_file: inp_file.replace("WFQ", "WFK", 1)})
                    else:
                        inp_files.append(inp_file)
                    break
        else:
            msg = f"Cannot find {' or '.join(exts)} file to restart from."
            logger.error(msg)
            raise InitializationError(msg)
        return irdvars, inp_files

    def get_abinit_input(
        self, structure=None, pseudos=None, prev_outputs=None, **kwargs
    ):
        """Get AbinitInput object."""
        raise NotImplementedError

    def on_restart(self, abinit_input):
        """Perform updates of AbinitInput upon restart of a previous calculation."""
