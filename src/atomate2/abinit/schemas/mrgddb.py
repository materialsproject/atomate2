"""Core definitions of Abinit calculations documents."""

import logging
import os
import numpy as np
from pathlib import Path
#from typing import Type, TypeVar, Union, Optional, List
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from abipy.abilab import abiopen
from abipy.abio.inputs import AbinitInput
from abipy.flowtk import events
from abipy.flowtk.utils import File
from emmet.core.structure import StructureMetadata
from jobflow.utils import ValueEnum
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure

from atomate2.abinit.files import load_abinit_input
from atomate2.abinit.utils.common import (
    LOG_FILE_NAME,
    MPIABORTFILE,
    OUTPUT_FILE_NAME,
    get_event_report,
    get_final_structure,
)

_T = TypeVar("_T", bound="MrgddbTaskDocument")

logger = logging.getLogger(__name__)


class Status(ValueEnum):
    """Abinit calculation state."""

    # TODO: merge this somewhere with vasp => common calculation schema ?

    SUCCESS = "successful"
    UNCONVERGED = "unconverged"
    FAILED = "failed"


class JobMetadata(BaseModel):
    """Definition of job metadata fields."""

    dir_name: str = Field(None, description="The directory of this job.")
    calc_type: str = Field(None, description="The type of calculation of this job.")


class MrgddbTaskDocument(StructureMetadata):
    """Definition of task document about an Mrgddb Job."""

    state: Status = Field(None, description="State of this job.")
    run_number: int = Field(None, description="Run number of this job.")
    dir_name: str = Field(None, description="The directory of this job.")
    event_report: events.EventReport = Field(
        None, description="Event report of this mrgddb job."
    )
    task_label: str = Field(None, description="The label for this job/task.")
    structure: Structure = Field(None, description="Final structure.")
    dijk: Any = Field(None, description="Conventional SHG tensor in pm/V")
    epsij: Any = Field(None, description="Dielectric tensor")

    @classmethod
    def from_directory(
        cls: Type[_T],
        dir_name: Union[Path, str],
        source: str = "log",
        critical_events=None,
        run_number=1,
        run_status=None,
    ) -> _T:
        """Build MrgddbTaskDocument from directory."""
        # Files required for the job analysis.
        # TODO: See if we can put the AbinitInputFile object here from
        #  abipy.abivars.AbinitInputFile (currently not MSONable)
        # input_file = File(os.path.join(dir_name, INPUT_FILE_NAME))
        output_file = File(os.path.join(dir_name, OUTPUT_FILE_NAME))
        log_file = File(os.path.join(dir_name, LOG_FILE_NAME))
        mpiabort_file = File(os.path.join(dir_name, MPIABORTFILE))
        ofile = {"output": output_file, "log": log_file}[source]

        report = None
        # TODO: How to detect which status it has here ?
        #  UNCONVERGED would be for scf/nscf/relax when it's not yet converged
        #  FAILED should be for a job that failed for other reasons.
        #  What about a job that has been killed by the run_abinit (i.e. before
        #  Slurm or PBS kills it) ?
        state = Status.UNCONVERGED

        try:
            report = get_event_report(ofile=ofile, mpiabort_file=mpiabort_file)
            critical_events_report = report.filter_types(critical_events)
            if not critical_events_report:
                state = Status.SUCCESS

        except Exception as exc:
            msg = f"{cls} exception while parsing event_report:\n{exc}"
            logger.critical(msg)
        
        out_DDB = str(os.path.join(dir_name, "outdata", "out_DDB"))
        with abiopen(out_DDB) as abifile:
            structure = abifile.structure
            dijk = abifile.anaget_nlo(voigt=False)
            epsij = abifile.anaget_epsinf_and_becs()[0]

        doc = cls.from_structure(
            meta_structure=structure,
            include_structure=True,
            dir_name=dir_name,
            event_report=report,
            state=state,
            run_number=run_number,
            structure=structure,
            dijk=dijk,
            epsij=epsij
        )
        return doc
