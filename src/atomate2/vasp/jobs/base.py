"""Definition of base VASP job maker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from jobflow import Maker, Response, job
from monty.serialization import dumpfn
from monty.shutil import gzip_dir
from pymatgen.core import Structure
from pymatgen.electronic_structure.bandstructure import (
    BandStructure,
    BandStructureSymmLine,
)
from pymatgen.electronic_structure.dos import DOS, CompleteDos, Dos
from pymatgen.io.vasp import Chgcar, Locpot, Wavecar

from atomate2.vasp.files import copy_vasp_outputs, write_vasp_input_set
from atomate2.vasp.run import run_vasp, should_stop_children
from atomate2.vasp.schemas.task import TaskDocument
from atomate2.vasp.sets.base import VaspInputSetGenerator

__all__ = ["BaseVaspMaker", "vasp_job"]


_DATA_OBJECTS = [
    BandStructure,
    BandStructureSymmLine,
    DOS,
    Dos,
    CompleteDos,
    Locpot,
    Chgcar,
    Wavecar,
]


def vasp_job(method: Callable):
    """
    Decorate the ``make`` method of VASP job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.Job` that configures common
    settings for all VASP jobs. For example, it ensures that large data objects
    (band structures, density of states, LOCPOT, CHGCAR, etc) are all stored in the
    atomate2 data store. It also configures the output schema to be a VASP
    :obj:`.TaskDocument`.

    Any makers that return VASP jobs (not flows) should decorate the ``make`` method
    with @vasp_job. For example:

    .. code-block:: python

        class MyVaspMaker(BaseVaspMaker):
            @vasp_job
            def make(structure):
                # code to run VASP job.
                pass

    Parameters
    ----------
    method : callable
        A BaseVaspMaker.make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate VASP jobs.
    """
    return job(method, data=_DATA_OBJECTS, output_schema=TaskDocument)


@dataclass
class BaseVaspMaker(Maker):
    """
    Base VASP job maker.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputSetGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "base vasp job"
    input_set_generator: VaspInputSetGenerator = field(
        default_factory=VaspInputSetGenerator
    )
    write_input_set_kwargs: dict = field(default_factory=dict)
    copy_vasp_kwargs: dict = field(default_factory=dict)
    run_vasp_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
    stop_children_kwargs: dict = field(default_factory=dict)
    write_additional_data: dict = field(default_factory=dict)

    @vasp_job
    def make(self, structure: Structure, prev_vasp_dir: str | Path | None = None):
        """
        Run a VASP calculation.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_vasp_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.
        """
        # copy previous inputs
        from_prev = prev_vasp_dir is not None
        if prev_vasp_dir is not None:
            copy_vasp_outputs(prev_vasp_dir, **self.copy_vasp_kwargs)

        if "from_prev" not in self.write_input_set_kwargs:
            self.write_input_set_kwargs["from_prev"] = from_prev

        # write vasp input files
        write_vasp_input_set(
            structure, self.input_set_generator, **self.write_input_set_kwargs
        )

        # write any additional data
        for filename, data in self.write_additional_data.items():
            dumpfn(data, filename.replace(":", "."))

        # run vasp
        run_vasp(**self.run_vasp_kwargs)

        # parse vasp outputs
        task_doc = TaskDocument.from_directory(Path.cwd(), **self.task_document_kwargs)
        task_doc.task_label = self.name

        # decide whether child jobs should proceed
        stop_children = should_stop_children(task_doc, **self.stop_children_kwargs)

        # gzip folder
        gzip_dir(".")

        return Response(
            stop_children=stop_children,
            stored_data={"custodian": task_doc.custodian},
            output=task_doc,
        )
