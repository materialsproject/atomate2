"""Definition of base ABINIT job maker."""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import jobflow
import pseudo_dojo
from abipy.abio.inputs import AbinitInput
from abipy.flowtk import events
from abipy.flowtk.events import AbinitEvent
from abipy.flowtk.utils import Directory, File, irdvars_for_ext
from jobflow import Maker, Response, job
from monty.inspect import all_subclasses
from monty.json import MontyEncoder
from monty.string import is_string
from pymatgen.core.structure import Structure
from pymatgen.io.abinit.pseudos import PseudoTable

from atomate2.abinit.inputs.factories import InputGenerator
from atomate2.abinit.run import run_abinit
from atomate2.abinit.schemas.core import AbinitJobSummary
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
    TMPDATA_PREFIX,
    TMPDIR_NAME,
    InitializationError,
    RestartInfo,
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
    input_generator: Optional[InputGenerator] = None
    CRITICAL_EVENTS: Sequence[str] = ()
    # TODO: is this ok to only have namedtuple ?
    #  Do we use namedtuple or do we use dict instead or allow both ?
    #  Do we allow a list or even a single str ?
    dependencies: Optional[dict] = None
    extra_abivars: Optional[dict] = None

    def __post_init__(self):
        """Process post-init configuration."""
        self.critical_events = [
            as_event_class(ce_name) for ce_name in self.CRITICAL_EVENTS
        ]

    @job
    def make(
        self,
        structure: Optional[Structure] = None,
        prev_outputs: Optional[Any] = None,
        abinit_input: Optional[AbinitInput] = None,
        previous_abinit_input: Optional[AbinitInput] = None,
        restart_info=None,
        history=None,
    ) -> Union[jobflow.Flow, jobflow.Job]:
        """
        Return an ABINIT jobflow.Job.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_outputs : str or Path or None
            A previous ABINIT calculation directory.
        """
        if structure is None and abinit_input is None:
            raise RuntimeError(
                "At least one of structure and abinit_input should be defined."
            )
        # TODO: use an "initial_structure" and "final_structure" ?
        if prev_outputs is not None:
            prev_outputs = (
                prev_outputs if isinstance(prev_outputs, list) else [prev_outputs]
            )

        self.setup_job(
            structure=structure,
            prev_outputs=prev_outputs,
            abinit_input=abinit_input,
            previous_abinit_input=previous_abinit_input,
            restart_info=restart_info,
            history=history,
        )
        self.run_abinit()
        response = self.job_analysis()

        return response

    def setup_job(
        self,
        structure: Optional[Structure] = None,
        prev_outputs: Optional[str | Path] = None,
        abinit_input: Optional[AbinitInput] = None,
        previous_abinit_input: Optional[AbinitInput] = None,
        restart_info: Optional[RestartInfo] = None,
        history: Optional[JobHistory] = None,
    ):
        """Set up abinit job.

        Parameters
        ----------
        structure
        prev_outputs
        abinit_input
        previous_abinit_input
        restart_info
        history
        """
        self.start_time = time.time()

        self.structure = structure
        self.restart_info = restart_info
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

        # Get abinit input
        self.set_abinit_input(
            structure=structure,
            prev_outputs=prev_outputs,
            abinit_input=abinit_input,
            previous_abinit_input=previous_abinit_input,
        )

        self.resolve_deps(prev_outputs=prev_outputs)

        # if it's the restart of a previous job, perform specific job updates.
        # perform these updates before writing the input, but after creating the dirs.
        if self.restart_info:
            self.history.log_restart(self.restart_info)
            self.resolve_restart_deps()

        # TODO: currently just in testing (only Si)
        #  transfer this in abipy
        input_str = str(self.abinit_input)
        input_str += (
            '\npseudos "/home/davidwaroquiers/miniconda3/envs/'
            "jobflow_abinit/lib/python3.8/site-packages/"
            'pseudo_dojo/pseudos/ONCVPSP-PBE-PDv0.4/Si/Si.psp8"'
        )

        self.input_file.write(input_str)

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
            abinit_input=self.abinit_input,
        )
        response = Response(output=output)

        if self.report is not None:
            # the calculation finished without errors
            if self.report.run_completed:
                # Check if the calculation converged.
                # TODO: where do we define whether a given critical event allows for a restart ?
                #  here we seem to assume that we can always restart because it is something unconverged
                #  (be it e.g. scf or relaxation)
                not_ok = self.report.filter_types(self.critical_events)
                if not_ok:
                    self.history.log_unconverged()
                    num_restarts = (
                        self.restart_info.num_restarts if self.restart_info else 0
                    )
                    if num_restarts < self.settings.MAX_RESTARTS:
                        new_job = self.get_restart_job()
                        response.replace = new_job
                    else:
                        response.stop_jobflow = True
                        unconverged_error = UnconvergedError(
                            self,
                            msg="Unconverged after {} restarts.".format(num_restarts),
                            abinit_input=self.abinit_input,
                            restart_info=self.restart_info,
                            history=self.history,
                        )
                        response.stored_data = {"error": unconverged_error}
                else:
                    # calculation converged
                    # everything is ok. conclude the job
                    # TODO: add convergence of custom parameters (this is used e.g. for dilatmx convergence)
                    stored_data = self.conclude_task()
                    response.stored_data = stored_data
        else:
            # TODO: add possible fixes here ? (no errors from abinit)
            raise NotImplementedError("")

        return response

    def conclude_task(self):
        """Conclude the task."""
        self.history.log_finalized(self.abinit_input)
        stored_data = {
            "report": self.report.as_dict(),
            "finalized": True,
            "history": self.history.as_dict(),
        }
        with open(HISTORY_JSON, "w") as f:
            json.dump(self.history, f, cls=MontyEncoder, indent=4, sort_keys=True)
        return stored_data

    def get_restart_job(self, reset=False):
        """Get new job to restart abinit calculation."""
        if self.restart_info:
            num_restarts = self.restart_info.num_restarts + 1
        else:
            num_restarts = 0

        self.restart_info = RestartInfo(
            previous_dir=self.workdir, reset=reset, num_restarts=num_restarts
        )

        new_job = self.make(
            structure=self.structure,
            abinit_input=self.abinit_input,
            restart_info=self.restart_info,
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

    def resolve_deps(self, prev_outputs):
        """Resolve the dependencies."""
        if not self.dependencies:
            return

        if not self.restart_info:
            prev_outputs_job_types = (
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
            prev_indata = os.path.join(self.restart_info.previous_dir, INDIR_NAME)
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
                #  this is related to the question on abinit_input AND strutured passed together in make.
                #  Do we keep this thing with '@' ?
                if dep.startswith("@structure"):
                    self.abinit_input.set_structure(structure=prev_output.structure)
                elif not dep.startswith("@"):
                    source_dir = prev_output.dir_name
                    self.abinit_input.set_vars(irdvars_for_ext(dep))
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
            else:
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
            # previous output file of the present task.
            if (
                not self.settings.COPY_DEPS
                and os.path.realpath(dest) != source
                and not self.restart_info
            ):
                msg = "dest {} does not point to path {}".format(dest, source)
                logger.error(msg)
                raise InitializationError(msg)

    def resolve_restart_deps(self):
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

    def set_abinit_input(
        self,
        structure: Optional[Structure] = None,
        prev_outputs: Optional[str | Path] = None,
        abinit_input: Optional[AbinitInput] = None,
        previous_abinit_input: Optional[AbinitInput] = None,
    ):
        """Set up AbinitInput.

        Parameters
        ----------
        structure : Structure
            Structure of this job.
        prev_outputs : list
            List of previous outputs potentially needed to set up the AbinitInput of this job.
        abinit_input : AbinitInput
            Explicit AbinitInput.
        previous_abinit_input : AbinitInput
            Previous AbinitInput object needed for the initialization of the AbinitInput of this job.
        """
        if abinit_input is not None:
            # TODO: what if abinit_input AND structure are set ?
            # Would that be ok to use for example for a restart of a relaxation ?
            # Currently: raise an error if it is the case
            # if structure is not None:
            #     raise NotImplementedError('Both structure and abinit_input are not None.')
            # # TODO: also, what if both abinit_input (in make) and the input_factory (in __init__) are set ?
            # # Currently: raise an error
            # if self.input_generator is not None:
            #     raise NotImplementedError('Both input_factory and abinit_input are not None.')
            self.abinit_input = abinit_input
            return

        if self.input_generator is None and abinit_input is None:
            raise RuntimeError(
                "Cannot create abinit input from structure without input generator."
            )

        # TODO: deal with previous inputs required (see below excerpt from abiflows)
        extra_abivars = self.extra_abivars or {}
        extra_abivars.update(
            {
                "indata_prefix": f'"{INDATA_PREFIX}"',
                "outdata_prefix": f'"{OUTDATA_PREFIX}"',
                "tmpdata_prefix": f'"{TMPDATA_PREFIX}"',
            }
        )
        gen_args = []
        gen_kwargs: Dict[str, Any] = {"extra_abivars": extra_abivars}
        if self.input_generator.structure_required:
            gen_args.append(structure)
            gen_kwargs.update({"pseudos": self.pseudos})
        if self.input_generator.gs_input_required:
            gen_args.append(previous_abinit_input)
        if len(gen_args) != 1:
            raise RuntimeError(
                "Only one positional argument supported right now for input generation. "
                "The positional argument is either a structure or a previous ground-state "
                "input."
            )
        self.abinit_input = self.input_generator.generate_abinit_input(
            *gen_args, **gen_kwargs
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
        self.abinit_input.remove_vars(remove_vars, strict=False)
        logger.info("Removing variables {} from input".format(remove_vars))

    def out_to_in(self, out_file):
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
        in_file = os.path.basename(out_file).replace("out", "in", 1)
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
