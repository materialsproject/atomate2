"""Definition of base JDFTx job maker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from atomate2.jdftx.schemas.task import TaskDoc
from jobflow import Maker, Response, job
from pymatgen.core.trajectory import Trajectory
from pymatgen.electronic_structure.bandstructure import (
    BandStructure,
    BandStructureSymmLine,
)


from atomate2.jdftx.sets.base import JdftxInputGenerator
from atomate2.jdftx.files import write_jdftx_input_set


from atomate2.jdftx.run import run_jdftx

#if TYPE_CHECKING:
from pymatgen.core import Structure


_DATA_OBJECTS = [ # TODO update relevant list for JDFTx
    BandStructure,
    BandStructureSymmLine,
    Trajectory,
    "force_constants",
    "normalmode_eigenvecs",
    "bandstructure",  # FIX: BandStructure is not currently MSONable
]

_INPUT_FILES = [
    "inputs.in",
    "inputs.lattice",
    "inputs.ionpos",
]

# Output files. Partially from https://www.vasp.at/wiki/index.php/Category:Output_files
_OUTPUT_FILES = [ # TODO finish this list
    "out.log",
    "Ecomponents",
    "wfns",
    "bandProjections",
    "boundCharge",
    "lattice",
    "ionpos",
]




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
 

    """

    name: str = "base JDFTx job"
    input_set_generator: JdftxInputGenerator = field(default_factory=JdftxInputGenerator)
    write_input_set_kwargs: dict = field(default_factory=dict)
    run_jdftx_kwargs: dict = field(default_factory=dict)


    def make(
        self, structure: Structure
    ) -> Response:
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
        # write jdftx input files
        write_jdftx_input_set(
            structure, self.input_set_generator, **self.write_input_set_kwargs
        )

        # run jdftx
        run_jdftx(**self.run_jdftx_kwargs)

        current_dir = Path.cwd()
        files = [str(f) for f in current_dir.glob('*') if f.is_file()]

        return Response(
            stop_children=stop_children,
            stored_data={"custodian": task_doc.custodian},
            output=task_doc,
        )


def get_jdftx_task_document(path: Path | str, **kwargs) -> TaskDoc:
    """Get JDFTx Task Document using atomate2 settings."""

    return TaskDoc.from_directory(path, **kwargs)