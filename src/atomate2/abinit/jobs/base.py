"""Definition of base ABINIT job maker."""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Union

import jobflow
import pseudo_dojo
from abipy.electrons.gsr import GsrFile
from abipy.flowtk import events
from abipy.flowtk.events import AbinitEvent
from abipy.flowtk.utils import Directory, File, irdvars_for_ext
from jobflow import Maker, Response, job
from monty.inspect import all_subclasses
from monty.json import MontyEncoder
from monty.string import is_string
from pymatgen.core.structure import Structure
from pymatgen.io.abinit.pseudos import PseudoTable

from atomate2.abinit.run import run_abinit
from atomate2.abinit.schemas.core import AbinitJobSummary
from atomate2.abinit.sets.base import AbinitInputSet, AbinitInputSetGenerator
from atomate2.abinit.utils.common import (
    HISTORY_JSON,
    INDATA_PREFIX,
    INDIR_NAME,
    INPUT_FILE_NAME,
    LOG_FILE_NAME,
    MPIABORTFILE,
    OUTDATA_PREFIX,
    OUTDIR_NAME,
    OUTPUT_FILE_NAME,
    STDERR_FILE_NAME,
    TMPDIR_NAME,
    InitializationError,
    PostProcessError,
    UnconvergedError,
)
from atomate2.abinit.utils.history import JobHistory
from atomate2.abinit.utils.settings import AbinitAtomateSettings, get_abipy_manager

logger = logging.getLogger(__name__)

__all__ = ["BaseAbinitMaker"]


def as_event_class(event_string):
    """Convert event string into a subclass of AbinitEvent.

    The string can be the class name or the YAML tag.
    """
    if is_string(event_string):
        for c in all_subclasses(AbinitEvent):
            if c.__name__ == event_string or c.yaml_tag == event_string:
                return c
        raise ValueError(f"Cannot find event class associated to {event_string}.")
    raise ValueError(
        f"Cannot convert event_string of type {type(event_string)}. Should be a string."
    )


@dataclass
class BaseAbinitMaker(Maker):
    """
    Base ABINIT job maker.

    Parameters
    ----------
    name : str
        The job name.
    pseudos : list of str, PseudoTable
        The pseudopotentials to use.
    """

    calc_type: str
    name: str = "base abinit job"
    pseudos: Union[
        List[str], PseudoTable
    ] = pseudo_dojo.OfficialDojoTable.from_djson_file(
        os.path.join(
            pseudo_dojo.dojotable_absdir("ONCVPSP-PBE-PDv0.4"), "standard.djson"
        )
    )
    walltime: Optional[int] = None
    input_set_generator: Optional[AbinitInputSetGenerator] = None
    CRITICAL_EVENTS: Sequence[str] = ()
    dependencies: Optional[dict] = None
    extra_abivars: dict = field(default_factory=dict)

    # class variables
    structure_fixed: ClassVar[bool] = True
    DEFAULT_INPUT_SET_GENERATOR: ClassVar[AbinitInputSetGenerator] = NotImplemented

    def __post_init__(self):
        """Process post-init configuration."""
        self.critical_events = [
            as_event_class(ce_name) for ce_name in self.CRITICAL_EVENTS
        ]
        if self.input_set_generator is None:
            self.input_set_generator = self.DEFAULT_INPUT_SET_GENERATOR

    @job
    def make(
        self,
        structure: Optional[Structure] = None,
        prev_outputs: Optional[Any] = None,
        restart_from: Optional[Union[str, jobflow.Job]] = None,
        # abinit_input_set: Optional[AbinitInputSet] = None,
        # previous_abinit_input_set: Optional[AbinitInputSet] = None,
        # restart_info=None,
        history: Optional[JobHistory] = None,
    ) -> Union[jobflow.Flow, jobflow.Job]:
        """
        Return an ABINIT jobflow.Job.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_outputs : TBD
            TODO: make sure the following is ok
            An absolute path as a string (previous directory), a list of absolute paths as strings
            (list of previous directories), a string (previous job uuid), a list of strings (list of previous job
            uuids), an OutputReference or a list of OutputReference objects,
            a dict with keys being a type of previous output and values being one of the previous (absolute
            path or list of absolute paths, uuid or list of uuids, ...), ...
        restart_from : TBD
            An absolute path as a string (previous directory), a string (previous job uuid), an OutputReference,
            TODO: what else ? is an AbinitInputSet ok ?
            TODO: I would say no, because we not only need the input but also the location where it previously ran.
        """
        if structure is None and prev_outputs is None and restart_from is None:
            raise RuntimeError(
                "At least one of structure, prev_outputs or restart_from should be defined."
            )

        self.setup_job(
            structure=structure,
            prev_outputs=prev_outputs,
            restart_from=restart_from,
            # )
            # abinit_input_set=abinit_input_set,
            # previous_abinit_input_set=previous_abinit_input_set,
            # restart_info=restart_info,
            history=history,
        )
        self.run_abinit()
        response = self.job_analysis()

        return response

    def setup_job(
        self,
        structure: Optional[Structure] = None,
        prev_outputs: Optional[Any] = None,
        restart_from=None,
        # restart_info: Optional[RestartInfo] = None,
        # ):
        # previous_abinit_input_set: Optional[AbinitInputSet] = None,
        # restart_info: Optional[RestartInfo] = None,
        history: Optional[JobHistory] = None,
    ):
        """Set up abinit job.

        Parameters
        ----------
        structure
        prev_outputs
        restart_from
        """
        self.start_time = time.time()

        self.structure = structure
        # TODO: transform restart_from and prev_outputs here so that
        #  they are in the format expected by the AbinitInputSet
        self.restart_from = restart_from
        # The previous outputs can be a single previous output, a list of previous outputs,
        # a dict with values being previous outputs.
        self.prev_outputs = prev_outputs
        if isinstance(prev_outputs, AbinitJobSummary):
            self.prev_outputs = [prev_outputs]
        elif isinstance(prev_outputs, list):
            self.prev_outputs = prev_outputs
        elif prev_outputs is None:
            self.prev_outputs = prev_outputs
        else:
            raise NotImplementedError()
        # TODO: see if we keep this restart_info or if we just use the history
        self.history = history or JobHistory()

        # Set up logging
        self.set_logger()

        # Load the atomate settings for abinit to get configuration parameters
        # TODO: how to allow for tuned parameters on a per-job basis ? (similar to fw_spec-passed settings)
        self.settings = AbinitAtomateSettings()
        self.abipy_manager = get_abipy_manager(self.settings)

        # set walltime, if possible
        # TODO: see in set_walltime, where to put this walltime_command
        self.set_walltime()

        # Set up working directory and create input, output and tmp data directories
        self.set_workdir(workdir=os.getcwd())

        # Log information about the start of the job
        self.history.log_start(workdir=self.workdir)

        # Get abinit input
        self.abinit_input_set = self.get_abinit_input_set(
            structure=self.structure,
            prev_outputs=self.prev_outputs,
            restart_from=self.restart_from,
        )

        # TODO: see if we put the resolve_deps and resolve_restart_deps inside the abinit input set.
        #  I think it would make sense. An abinit input set (and, more generally, any input set without previous
        #  dependencies or previous vasp dir or previous "something") can write an input that can then directly
        #  be run just after. It would be nice to have the same for input sets that depends on previous calculations.
        self.resolve_deps(
            restart_from=self.restart_from, prev_outputs=self.prev_outputs
        )

        # if it's the restart of a previous job, perform specific job updates.
        # perform these updates before writing the input, but after creating the dirs.
        if restart_from is not None:
            # TODO: do we need this ?
            self.history.log_restart()
            # "manual" restart from another previous job
            if self.history.is_first_run:
                if isinstance(restart_from, AbinitJobSummary):
                    prev_dir = os.path.join(restart_from.dir_name)
                    self.resolve_restart_deps(prev_dir)
            # automatic restart when unconverged
            else:
                self.resolve_restart_deps(self.history.prev_dir)
        self.abinit_input_set.write_input(
            directory=self.workdir, make_dir=False, overwrite=False
        )

    def job_analysis(self):
        """Perform analysis of abinit job."""
        self.report = None
        try:
            self.report = self.get_event_report()
        except Exception as exc:
            msg = "%s exception while parsing event_report:\n%s" % (self, exc)
            logger.critical(msg)

        output = AbinitJobSummary(
            calc_type=self.calc_type,
            dir_name=os.getcwd(),
            abinit_input_set=self.abinit_input_set,
            structure=self.get_final_structure(),
        )
        response = Response(output=output)

        if self.report is not None:
            # the calculation finished without errors
            if self.report.run_completed:
                self.history.log_end(workdir=self.workdir)
                # Check if the calculation converged.
                # TODO: where do we define whether a given critical event allows for a restart ?
                #  here we seem to assume that we can always restart because it is something unconverged
                #  (be it e.g. scf or relaxation)
                not_ok = self.report.filter_types(self.critical_events)
                if not_ok:
                    self.history.log_unconverged()
                    num_restarts = self.history.num_restarts
                    # num_restarts = (
                    #     self.restart_info.num_restarts if self.restart_info else 0
                    # )
                    if num_restarts < self.settings.MAX_RESTARTS:
                        new_job = self.get_restart_job(output=output)
                        response.replace = new_job
                    else:
                        # TODO: check here if we should stop jobflow or children or if we should throw an error.
                        response.stop_jobflow = True
                        # response.stop_children = True
                        unconverged_error = UnconvergedError(
                            self,
                            msg="Unconverged after {} restarts.".format(num_restarts),
                            abinit_input=self.abinit_input_set.abinit_input,
                            # restart_info=self.restart_info,
                            history=self.history,
                        )
                        response.stored_data = {"error": unconverged_error}
                        raise unconverged_error
                else:
                    # calculation converged
                    # everything is ok. conclude the job
                    # TODO: add convergence of custom parameters (this is used e.g. for dilatmx convergence)
                    response.output.energy = self.get_final_energy()
                    stored_data = self.conclude_task()
                    response.stored_data = stored_data
        else:
            # TODO: add possible fixes here ? (no errors from abinit)
            raise NotImplementedError("")

        return response

    def conclude_task(self):
        """Conclude the task."""
        self.history.log_finalized(self.abinit_input_set.abinit_input)
        stored_data = {
            "report": self.report.as_dict(),
            "finalized": True,
            "history": self.history.as_dict(),
        }
        with open(HISTORY_JSON, "w") as f:
            json.dump(self.history, f, cls=MontyEncoder, indent=4, sort_keys=True)
        return stored_data

    def get_restart_job(self, output):
        """Get new job to restart abinit calculation."""
        logger.info(msg="Getting restart job.")
        structure = None
        if not self.structure_fixed:
            logger.info(msg="Getting final structure to restart from.")
            structure = self.get_final_structure()

        new_job = self.make(
            structure=structure,
            restart_from=output,
            history=self.history,
        )

        return new_job

    def get_event_report(self, source="log"):
        """Get report from abinit calculation.

        This analyzes the main output file for possible Errors or Warnings.
        It will check the presence of an MPIABORTFILE if not output file is found.

        Parameters
        ----------
        source : str
            Type of file to be parsed. Should be "output" or "log".

        Returns
        -------
        EventReport
            Report of the abinit calculation or None if no output file exists.
        """
        ofile = {"output": self.output_file, "log": self.log_file}[source]

        parser = events.EventsParser()

        if not ofile.exists:
            if not self.mpiabort_file.exists:
                return None
            else:
                # ABINIT abort file without log!
                abort_report = parser.parse(self.mpiabort_file.path)
                return abort_report

        try:
            report = parser.parse(ofile.path)

            # Add events found in the ABI_MPIABORTFILE.
            if self.mpiabort_file.exists:
                logger.critical("Found ABI_MPIABORTFILE!")
                abort_report = parser.parse(self.mpiabort_file.path)
                if len(abort_report) == 0:
                    logger.warning("ABI_MPIABORTFILE but empty")
                else:
                    if len(abort_report) != 1:
                        logger.critical("Found more than one event in ABI_MPIABORTFILE")

                    # Add it to the initial report only if it differs
                    # from the last one found in the main log file.
                    last_abort_event = abort_report[-1]
                    if report and last_abort_event != report[-1]:
                        report.append(last_abort_event)
                    else:
                        report.append(last_abort_event)

            return report

        # except parser.Error as exc:
        except Exception as exc:
            # Return a report with an error entry with info on the exception.
            logger.critical(
                "{}: Exception while parsing ABINIT events:\n {}".format(
                    ofile, str(exc)
                )
            )
            return parser.report_exception(ofile.path, exc)

    def set_walltime(self):
        """Set the walltime."""

    def resolve_deps(self, restart_from, prev_outputs):
        """Resolve the dependencies."""
        if not self.dependencies:
            return

        if restart_from is None:
            prev_outputs_job_types = tuple(
                [prev_output.calc_type for prev_output in prev_outputs]
                if prev_outputs
                else []
            )
            for job_type, deps_list in self.dependencies.items():
                n_prev_jobs = prev_outputs_job_types.count(job_type)
                if n_prev_jobs == 0:
                    msg = f'No previous job of type "{job_type}".'
                    logger.error(msg)
                    raise InitializationError(msg)
                elif n_prev_jobs > 1:
                    msg = f'More than 1 previous job of type "{job_type}". Risk of overwriting.'
                    logger.warning(msg)
                self.resolve_deps_per_job_type(
                    [
                        prev_output
                        for prev_output in prev_outputs
                        if prev_output.calc_type == job_type
                    ],
                    deps_list,
                )
        else:
            # Just link everything from the indata folder of the previous run.
            # Files needed for restart will be overwritten
            prev_indata = os.path.join(restart_from.dir_name, INDIR_NAME)
            for f in os.listdir(prev_indata):
                # if the target is already a link, link to the source to avoid many nested levels of linking
                source = os.path.join(prev_indata, f)
                if os.path.islink(source):
                    source = os.readlink(source)
                os.symlink(source, os.path.join(self.workdir, INDIR_NAME, f))

    def resolve_deps_per_job_type(self, prev_outputs, deps_list):
        """Resolve dependencies for specific job type."""
        deps_list = deps_list if isinstance(deps_list, list) else [deps_list]
        for prev_output in prev_outputs:
            for dep in deps_list:
                # TODO: Do we need to keep this here as it is supposed to be passed using the jobflow db ?
                #  this is related to the question on abinit_input AND struture passed together in make.
                #  Do we keep this thing with '@' ?
                # if dep.startswith("@structure"):
                #     self.abinit_input.set_structure(structure=prev_output.structure)
                # if not dep.startswith("@"):
                source_dir = prev_output.dir_name
                self.abinit_input_set.set_vars(irdvars_for_ext(dep))
                if dep == "DDK":
                    raise NotImplementedError
                    # self.link_ddk(source_dir)
                elif dep == "1WF" or dep == "1DEN":
                    raise NotImplementedError
                    # self.link_1ext(dep, source_dir)
                else:
                    self.link_ext(dep, source_dir)

    def link_ext(self, ext, source_dir, strict=True):
        """Link the required files from previous runs in the input data directory.

        It will first try to link the fortran file and then the Netcdf file, if the first is not found.

        Parameters
        ----------
        ext : str
            extension that should be linked.
        source_dir : str
            path to the source directory.
        strict : bool
            whether to raise an exception if the file is missing.

        Returns
        -------
        str
            The path to the generated link. None if strict=False and the file could not be found.
        """
        source = os.path.join(source_dir, OUTDATA_PREFIX + "_" + ext)
        logger.info("Need path {} with ext {}".format(source, ext))
        dest = os.path.join(self.workdir, INDATA_PREFIX + "_" + ext)

        if not os.path.exists(source):
            # Try netcdf file. TODO: this case should be treated in a cleaner way.
            source += ".nc"
            if os.path.exists(source):
                dest += ".nc"

        if not os.path.exists(source):
            if strict:
                msg = "{} is needed by this job but it does not exist".format(source)
                logger.error(msg)
                raise InitializationError(msg)
            return

        # Link path to dest if dest link does not exist.
        # else check that it points to the expected file.
        logger.info("Linking path {} --> {}".format(source, dest))
        if not os.path.exists(dest) or not strict:
            if self.settings.COPY_DEPS:
                shutil.copyfile(source, dest)
            else:
                os.symlink(source, dest)
            return dest
        else:
            # check links but only if we haven't performed the restart.
            # in this case, indeed we may have replaced the file pointer with the
            # previous output file of the present job.
            if (
                not self.settings.COPY_DEPS
                and os.path.realpath(dest) != source
                # and not self.restart_info
            ):
                msg = "dest {} does not point to path {}".format(dest, source)
                logger.error(msg)
                raise InitializationError(msg)

    def resolve_restart_deps(self, prev_dir):
        """Resolve dependencies for a job that is restarted."""
        # To be implemented for specific types of job (a relaxation is not restarted the same way
        # an scf is, or a non scf, or ...)

    @staticmethod
    def set_logger():
        """Set a logger for pymatgen.io.abinit and abipy."""
        # TODO: what to do here ?
        # log_handler = logging.FileHandler('abipy.log')
        # log_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
        # logging.getLogger('pymatgen.io.abinit').addHandler(log_handler)
        # logging.getLogger('abipy').addHandler(log_handler)
        # logging.getLogger('abiflows').addHandler(log_handler)

    def set_workdir(self, workdir):
        """Set up the working directory.

        This also sets up and creates standard input, output and temporary directories, as well as
        standard file names for input and output.
        """
        self.workdir = os.path.abspath(workdir)

        # Files required for the execution.
        self.input_file = File(os.path.join(self.workdir, INPUT_FILE_NAME))
        self.output_file = File(os.path.join(self.workdir, OUTPUT_FILE_NAME))
        self.log_file = File(os.path.join(self.workdir, LOG_FILE_NAME))
        self.stderr_file = File(os.path.join(self.workdir, STDERR_FILE_NAME))

        # This file is produce by Abinit if nprocs > 1 and MPI_ABORT.
        self.mpiabort_file = File(os.path.join(self.workdir, MPIABORTFILE))

        # Directories with input|output|temporary data.
        self.indir = Directory(os.path.join(self.workdir, INDIR_NAME))
        self.outdir = Directory(os.path.join(self.workdir, OUTDIR_NAME))
        self.tmpdir = Directory(os.path.join(self.workdir, TMPDIR_NAME))

        # Create dirs for input, output and tmp data.
        self.indir.makedirs()
        self.outdir.makedirs()
        self.tmpdir.makedirs()

    def get_abinit_input_set(
        self,
        structure: Optional[Structure] = None,
        prev_outputs=None,
        restart_from=None,
    ):
        """Set up AbinitInputSet.

        Parameters
        ----------
        structure : Structure
            Structure of this job.
        prev_outputs : TBD
            TBD
        restart_from : TBD
            restart from a directory, from a previous job, from a previous uuid, from a previous ...
        """
        # if prev_outputs is not None:
        #     raise NotImplementedError('Deal with prev_outputs.')
        # if isinstance(prev_outputs, AbinitJobSummary):
        #     prev_outputs = [prev_outputs]
        # elif prev_outputs is None:
        #     pass
        # else:
        #     raise NotImplementedError()

        gen_kwargs: Dict[str, Any] = {"extra_abivars": self.extra_abivars}

        if restart_from is not None:
            # TODO: depending on what restart_from is, do something
            #  currently, only deal with a previous output reference
            # TODO: if both restart_from and the input_set_generator are set, make an update of the parameters
            #  requires to define what can be updated and how for each input_set_generator.
            # if self.input_set_generator is not None:
            #     raise NotImplementedError('Both input_set_generator and restart_from are not None.')
            if isinstance(restart_from, AbinitInputSet):
                raise NotImplementedError(
                    "Currently, do not allow restarts from an AbinitInputSet."
                )
                # prev_input_set = restart_from
                # if structure is not None:
                #     # TODO: maybe in this case we have to check that both structures are the same and raise
                #     #  only if they differ
                #     raise NotImplementedError(
                #         "Structure is not None and restart_from is an AbinitInputSet."
                #     )
            elif isinstance(restart_from, AbinitJobSummary):
                prev_input_set = restart_from.abinit_input_set
            else:
                raise NotImplementedError(
                    "Implement other restart_from options. "
                    f"Here we try to restart from a {type(restart_from)}"
                )
            if self.history.is_first_run:
                update_params = True
            else:
                update_params = False

            # Here we assume that restart_from is an abinit input set to restart from.
            # TODO: implement restart_from from a directory? Maybe from above ?
            #  (I think we should not allow here to have a job uuid or an OutputReference).
            return self.input_set_generator.get_input_set(
                structure=structure,
                restart_from=prev_input_set,
                prev_outputs=prev_outputs,
                update_params=update_params,
                **gen_kwargs,
            )

        if self.input_set_generator is None and restart_from is None:
            raise RuntimeError(
                "Cannot create abinit input set from structure without input set generator."
            )

        # TODO: deal with prev_outputs
        #  Should be slightly different than restart_from (cannot be "just" a list or dict of input sets or directories)
        return self.input_set_generator.get_input_set(
            structure=structure,
            restart_from=None,
            prev_outputs=prev_outputs,
            update_params=False,
            **gen_kwargs,
        )

    def run_abinit(self):
        """Execute abinit."""
        run_abinit(
            abinit_cmd="abinit",
            mpirun_cmd="mpirun",
            log_file_path=LOG_FILE_NAME,
            stderr_file_path=STDERR_FILE_NAME,
        )

    def remove_restart_vars(self, exts):
        """Remove read variables associated with the extensions.

        This removes the ird* variables associated with the extensions from the current input.
        Useful in case of reset during a restart.
        """
        if not isinstance(exts, (list, tuple)):
            exts = [exts]

        remove_vars = [v for e in exts for v in irdvars_for_ext(e).keys()]
        self.abinit_input_set.remove_vars(remove_vars, strict=False)
        logger.info("Removing variables {} from input".format(remove_vars))

    def out_to_in(self, out_file):
        """Link or copy output file from previous job to the input data directory of this job.

        This will make a link or a copy of the output file to the input data directory of this job
        and rename the file so that ABINIT can read it as an input data file.

        Note that this method also handles WFQ files which are output with a WFQ extension but should be
        read with a WFK extension.

        Parameters
        ----------
        out_file : str
            Output file to be linked or copied to the input data directory.

        Returns
        -------
        str
            The absolute path of the new file in the input data directory.
        """
        in_file = os.path.basename(out_file).replace("out", "in", 1)
        in_file = os.path.basename(in_file).replace("WFQ", "WFK", 1)
        dest = os.path.join(self.indir.path, in_file)

        if os.path.exists(dest) and not os.path.islink(dest):
            logger.warning("Will overwrite %s with %s" % (dest, out_file))

        # if rerunning in the same folder the file should be moved anyway
        if self.settings.COPY_DEPS:
            shutil.copyfile(out_file, dest)
        else:
            # if dest already exists should be overwritten. see also resolve_deps and config_run
            try:
                os.symlink(out_file, dest)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    os.remove(dest)
                    os.symlink(out_file, dest)
                else:
                    raise e

        return dest

    def out_to_in_tim(self, out_file, in_file):
        """Link or copy output file from previous job to the input data directory of this job.

        This will make a link or a copy of the output file to the input data directory of this job
        and rename the file so that ABINIT can read it as an input data file.

        Parameters
        ----------
        out_file : str
            Output file to be linked or copied to the input data directory.

        Returns
        -------
        str
            The absolute path of the new file in the input data directory.
        """
        dest = os.path.join(self.indir.path, in_file)

        if os.path.exists(dest) and not os.path.islink(dest):
            logger.warning("Will overwrite %s with %s" % (dest, out_file))

        if self.settings.COPY_DEPS:
            shutil.copyfile(out_file, dest)
        else:
            # if dest already exists should be overwritten. see also resolve_deps and config_run
            try:
                os.symlink(out_file, dest)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    os.remove(dest)
                    os.symlink(out_file, dest)
                else:
                    raise e

        return dest

    def get_final_structure(self):
        """Get the final structure."""
        # No need to get the structure from the output when it is fixed.
        # This is the case for everything except relaxations and molecular dynamics calculations.
        if self.structure_fixed:
            # When restarts occur, the structure in make is None so we take it from the abinit_input here
            # instead of from self.structure.
            return self.abinit_input_set.abinit_input.structure

        # For relaxations and molecular dynamics, get the structure from the Gsr file.
        try:
            with self.open_gsr() as gsr:
                return gsr.structure
        except AttributeError:
            msg = "Cannot find the GSR file with the final structure."
            logger.error(msg)
            raise PostProcessError(msg)

    def get_final_energy(self):
        """Get the final energy."""
        try:
            with self.open_gsr() as gsr:
                return gsr.energy
        except AttributeError:
            msg = "Cannot find the GSR file with the final energy."
            logger.error(msg)
            raise PostProcessError(msg)

    # TODO: use monty's lazyproperty here ?
    @property
    def gsr_path(self):
        """Get the absolute path of the GSR file. Empty string if file is not present."""
        return self.outdir.has_abiext("GSR")
        # # Lazy property to avoid multiple calls to has_abiext.
        # TODO: There is an issue here when using the lazy property as the _gsr_path is already set
        #  One might unset it at the end of the execution of the job ?
        #  In any case this poses a question about lazy properties (maybe even standard attributes ?)
        # try:
        #     return self._gsr_path
        # except AttributeError:
        #     path = self.outdir.has_abiext("GSR")
        #     if path:
        #         self._gsr_path = path
        #     return path

    def open_gsr(self):
        """Open the GSR.nc file.

        This returns a GsrFile object or raises a PostProcessError exception if the file could not be found or
        is not readable.
        """
        gsr_path = self.gsr_path
        if not gsr_path:
            msg = "No GSR file available for job {} in {}".format(
                self.name, self.outdir
            )
            logger.critical(msg)
            raise PostProcessError(msg)

        # Open the GSR file.
        try:
            return GsrFile(gsr_path)
        except Exception as exc:
            msg = "Exception while reading GSR file at %s:\n%s" % (gsr_path, str(exc))
            logger.critical(msg)
            raise PostProcessError(msg)
