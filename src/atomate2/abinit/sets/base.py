"""Module defining base abinit input set and generator."""

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

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from pymatgen.core.structure import Structure


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
        input_files: Iterable[tuple[str, str]] | None = None,
        link_files: bool = True,
    ) -> None:
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
        """Write Abinit input files to a directory."""
        # TODO: do we allow zipping ? not sure if it really makes sense for abinit as
        #  the abinit input set also sets up links to previous files, sets up the
        #  indir, outdir and tmpdir, ...
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
        indir, _outdir, _tmpdir = self.set_workdir(workdir=directory)

        if self.input_files:
            out_to_in(
                out_files=self.input_files,
                indir=indir.path,
                link_files=self.link_files,
            )

    def validate(self) -> bool:
        """Validate the input set.

        Check that all files in the input directory
        have their corresponding ird variables.
        """
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
    def abinit_input(self) -> AbinitInput:
        """Get the AbinitInput object."""
        return self[INPUT_FILE_NAME]

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

    def remove_vars(self, keys: Iterable[str] | str, strict: bool = True) -> dict:
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

    def runlevel(self) -> set[str]:
        """Get the set of strings defining the calculation type."""
        return self.abinit_input.runlevel

    def set_structure(self, structure: Any) -> Structure:
        """Set the structure for this input set.

        This basically forwards the setting of the structure to the abipy
        AbinitInput object.
        """
        return self.abinit_input.set_structure(structure)

    def deepcopy(self) -> AbinitInputSet:
        """Deep copy of the input set."""
        return copy.deepcopy(self)


def as_pseudo_table(pseudos: str | Sequence[Pseudo]) -> PseudoTable:
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
class AbinitInputGenerator(InputGenerator):
    """
    A class to generate Abinit input sets.

    Parameters
    ----------
    factory
        A callable that generates an AbinitInput or MultiDataset object.
    calc_type
        A short description of the calculation type
    pseudos
        Define the pseudopotentials that should be used for the calculation.
        Can be an instance of a PseudoTable, a list of strings with the paths of
        the pseudopotential files or a string with the name of a PseudoDojo table
        (https://github.com/PseudoDojo/), followed by the accuracy of the pseudos
        in that table, separated by a colon. This requires that the PseudoTable
        is installed in the system.
        Set to None if no pseudopotentials should be set, as coming from a previous
        AbinitInput.
    factory_kwargs
        A dictionary to customize the values for the arguments of the factory
        function.
    user_abinit_settings
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

    factory: Callable
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
            Directory or list/tuple of 1 directory to restart from.
        prev_outputs : str or Path or list or tuple
            Directory or list/tuple of directories needed as dependencies for the
                AbinitInputSet generated.
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
                    f"Previous outputs not allowed for {type(self).__name__}."
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

    @staticmethod
    def check_format_prev_dirs(
        prev_dirs: str | tuple | list | Path | None,
    ) -> list[str] | None:
        """Check and format the prev_dirs (restart or dependency)."""
        if prev_dirs is None:
            return None
        if isinstance(prev_dirs, str | Path):
            return [str(prev_dirs)]
        return [str(prev_dir) for prev_dir in prev_dirs]

    def resolve_deps(
        self, prev_dirs: list[str], deps: str | tuple, check_runlevel: bool = True
    ) -> tuple[dict, list]:
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
                exts = list(dep.split(":")[1].split("|"))
                if not check_runlevel or runlevel.intersection(abinit_input.runlevel):
                    irdvars, inp_files = self.resolve_dep_exts(
                        prev_dir=prev_dir, exts=exts
                    )
                    input_files.extend(inp_files)
                    deps_irdvars.update(irdvars)

        return deps_irdvars, input_files

    def resolve_prev_inputs(
        self, prev_dirs: list[str], prev_inputs_kwargs: dict
    ) -> dict[str, AbinitInput]:
        """
        Find suitable abinit inputs from the previous outputs.

        Also retrieves the final structure from the previous outputs
        and replace it in the selected abinit input.

        This method assumes that prev_dirs is in the correct format, i.e.
        a list of directories as str or Path.
        """
        abinit_inputs = {}
        for prev_dir in prev_dirs:
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
                f"Should have exactly {n_found} previous output. Found {n_required}"
            )

        return abinit_inputs

    @staticmethod
    def _get_in_file_name(out_filepath: str) -> str:
        in_file = os.path.basename(out_filepath)
        in_file = in_file.replace(OUTDATAFILE_PREFIX, INDATAFILE_PREFIX, 1)

        return os.path.basename(in_file).replace("WFQ", "WFK", 1)

    @staticmethod
    def resolve_dep_exts(prev_dir: str, exts: list[str]) -> tuple:
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
                    inp_files = [
                        (f.path, AbinitInputGenerator._get_in_file_name(f.path))
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
        self,
        structure: Structure | None = None,
        pseudos: PseudoTable | None = None,
        prev_outputs: list[str] | None = None,
        abinit_settings: dict | None = None,
        factory_kwargs: dict | None = None,
        kpoints_settings: dict | KSampling | None = None,
        input_index: int | None = None,
    ) -> AbinitInput:
        """Generate the AbinitInput for the input set.

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
                    f"No previous_outputs. Required for {type(self).__name__}."
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

        elif structure is None:
            msg = (
                f"Structure is mandatory for {type(self).__name__} "
                f"generation since no previous output is used."
            )
            raise RuntimeError(msg)

        if not self.prev_outputs_deps and prev_outputs:
            msg = (
                f"Previous outputs not allowed for {type(self).__name__} "
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

    def _set_kpt_vars(
        self,
        abinit_input: AbinitInput | MultiDataset,
        kpoints_settings: dict | KSampling | None,
    ) -> None:
        """
        Update the kpoints variables, according to the options selected.

        Parameters
        ----------
        abinit_input
            An AbinitInput to be updated.
        kpoints_settings
            The options to set the kpoints variable.
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
        Remove the variables whose value is set to None from the AbinitInput.

        Parameters
        ----------
        abinit_input
            An AbinitInput to modify.
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
        """Get the kpoints file."""
        kpoints_updates = {} if kpoints_updates is None else kpoints_updates

        # use user setting if set otherwise default to base config settings
        if self.user_kpoints_settings != {}:
            kconfig = copy.deepcopy(self.user_kpoints_settings)
        elif kpoints_updates:
            kconfig = kpoints_updates
        else:
            return None

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
            # handle line density generation
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
            # handle regular weighted k-point grid generation
            if kconfig.get("grid_density"):
                vasp_kpoints = Kpoints.automatic_density(
                    structure, int(kconfig["grid_density"]), self.force_gamma
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
                    structure, kconfig["reciprocal_density"], self.force_gamma
                )
                base_kpoints = KSampling(
                    mode=KSamplingModes.monkhorst,
                    num_kpts=0,
                    kpts=vasp_kpoints.kpts,
                    kpt_shifts=vasp_kpoints.kpts_shift,
                    comment=vasp_kpoints.comment,
                )
            if explicit:
                sga = SpacegroupAnalyzer(structure, symprec=self.symprec)
                mesh = sga.get_ir_reciprocal_mesh(base_kpoints.kpts[0])
                base_kpoints = KSampling(
                    mode=KSamplingModes.automatic,
                    num_kpts=len(mesh),
                    kpts=[i[0] for i in mesh],
                    kpts_weights=[i[1] for i in mesh],
                    comment="Uniform grid",
                )
            else:
                # if not explicit that means no other options have been specified
                # so we can return the k-points as is
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

        # do some sanity checking
        if not (base_kpoints or added_kpoints):
            raise ValueError("Invalid k-point generation algo.")

        return _combine_kpoints(base_kpoints, added_kpoints)


def _combine_kpoints(*kpoints_objects: KSampling) -> KSampling:
    """Combine k-points files together."""
    kpoints = []
    weights = []

    for kpoints_object in filter(None, kpoints_objects):
        if not kpoints_object.mode == KSamplingModes.automatic:
            raise ValueError(
                "Can only combine kpoints with mode=KSamplingModes.automatic"
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
