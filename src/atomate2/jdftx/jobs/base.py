"""Definition of base JDFTx job maker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from jobflow import Maker, Response, job
import logging
logger = logging.getLogger(__name__)

from atomate2.jdftx.sets.base import JdftxInputGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
from pymatgen.core.trajectory import Trajectory
from pymatgen.electronic_structure.bandstructure import (
    BandStructure,
    BandStructureSymmLine,
)

from atomate2.jdftx.files import write_jdftx_input_set
from atomate2.jdftx.run import run_jdftx, should_stop_children
from atomate2.jdftx.schemas.task import TaskDoc

_DATA_OBJECTS = [  # TODO update relevant list for JDFTx
    BandStructure,
    BandStructureSymmLine,
    Trajectory,
    "force_constants",
    "normalmode_eigenvecs",
    "bandstructure",  # FIX: BandStructure is not currently MSONable
]

_INPUT_FILES = [
    "init.in",
    "init.lattice",
    "init.ionpos",
]

# Output files.
_OUTPUT_FILES = [  # TODO finish this list
    "output.out",
    "Ecomponents",
    "wfns",
    "bandProjections",
    "boundCharge",
    "lattice",
    "ionpos",
]


def jdftx_job(method: Callable) -> job:
    """
    Decorate the ``make`` method of JDFTx job makers.

    Parameters
    ----------
    method : callable
        A BaseJdftxMaker.make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate JDFTx jobs.
    """
    return job(method, data=_DATA_OBJECTS, output_schema=TaskDoc)


@dataclass
class BaseJdftxMaker(Maker):
    """
    Base JDFTx job maker.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .JdftxInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_jdftx_input_set`.
    run_jdftx_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_jdftx`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.

    """

    name: str = "base JDFTx job"
    input_set_generator: JdftxInputGenerator = field(
        default_factory=JdftxInputGenerator
    )
    write_input_set_kwargs: dict = field(default_factory=dict)
    run_jdftx_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    def make(self, structure: Structure) -> Response:
        """Run a JDFTx calculation.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.

        Returns
        -------
            Response: A response object containing the output, detours and stop
                commands of the JDFTx run.
        """
        print(structure)
        # write jdftx input files
        write_jdftx_input_set(
            structure, self.input_set_generator, **self.write_input_set_kwargs
        )
        logger.info("Wrote JDFTx input files.")
        # run jdftx
        run_jdftx(**self.run_jdftx_kwargs)

        current_dir = Path.cwd()
        task_doc = get_jdftx_task_document(current_dir, **self.task_document_kwargs)

        stop_children = should_stop_children(task_doc)



        return Response(
            stop_children=stop_children,
            stored_data={},
            output=task_doc,
        )


def get_jdftx_task_document(path: Path | str, **kwargs) -> TaskDoc:
    """Get JDFTx Task Document using atomate2 settings."""
    return TaskDoc.from_directory(path, **kwargs)
