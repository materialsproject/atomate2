"""Definition of base CP2K job maker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from jobflow import Maker, Response, job
from monty.serialization import dumpfn
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.alchemy.transmuters import StandardTransmuter
from pymatgen.core.trajectory import Trajectory
from pymatgen.electronic_structure.bandstructure import (
    BandStructure,
    BandStructureSymmLine,
)
from pymatgen.electronic_structure.dos import DOS, CompleteDos, Dos
from pymatgen.io.common import VolumetricData

from atomate2 import SETTINGS
from atomate2.common.files import gzip_files, gzip_output_folder
from atomate2.common.utils import get_transformations
from atomate2.cp2k.files import (
    cleanup_cp2k_outputs,
    copy_cp2k_outputs,
    write_cp2k_input_set,
)
from atomate2.cp2k.run import run_cp2k, should_stop_children
from atomate2.cp2k.schemas.task import TaskDocument
from atomate2.cp2k.sets.base import Cp2kInputGenerator

if TYPE_CHECKING:
    from collections.abc import Callable

    from pymatgen.core import Structure


_DATA_OBJECTS = [
    BandStructure,
    BandStructureSymmLine,
    DOS,
    Dos,
    CompleteDos,
    VolumetricData,
    Trajectory,
]


_FILES_TO_ZIP = ["cp2k.inp", "cp2k.out"]


def cp2k_job(method: Callable) -> job:
    """
    Decorate the ``make`` method of CP2K job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.job` that configures common
    settings for all CP2K jobs. For example, it ensures that large data objects
    (band structures, density of states, Cubes, etc) are all stored in the
    atomate2 data store. It also configures the output schema to be a CP2K
    :obj:`.TaskDocument`.

    Any makers that return CP2K jobs (not flows) should decorate the ``make`` method
    with @cp2k_job. For example:

    .. code-block:: python

        class MyCp2kMaker(BaseCp2kMaker):
            @cp2k_job
            def make(structure):
                # code to run Cp2k job.
                pass

    Parameters
    ----------
    method : callable
        A BaseCp2kMaker.make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate Cp2k jobs.
    """
    return job(method, data=_DATA_OBJECTS, output_schema=TaskDocument)


@dataclass
class BaseCp2kMaker(Maker):
    """
    Base CP2K job maker.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .Cp2kInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_cp2k_input_set`.
    copy_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_cp2k_outputs`.
    copy_cp2k_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_cp2k`.
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
    store_output_data: bool
        Whether the job output (TaskDocument) should be stored in the JobStore through
        the response.
    """

    name: str = "base cp2k job"
    input_set_generator: Cp2kInputGenerator = field(default_factory=Cp2kInputGenerator)
    write_input_set_kwargs: dict = field(default_factory=dict)
    copy_cp2k_kwargs: dict = field(default_factory=dict)
    run_cp2k_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
    stop_children_kwargs: dict = field(default_factory=dict)
    write_additional_data: dict = field(default_factory=dict)
    transformations: tuple[str, ...] = field(default_factory=tuple)
    transformation_params: tuple[dict, ...] | None = None
    store_output_data: bool = True

    @cp2k_job
    def make(
        self, structure: Structure, prev_dir: str | Path | None = None
    ) -> Response:
        """Run a CP2K calculation.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous CP2K calculation directory to copy output files from.
        """
        # Apply transformations if they are present
        if self.transformations:
            transformations = get_transformations(
                self.transformations, self.transformation_params
            )
            ts = TransformedStructure(structure)
            transmuter = StandardTransmuter([ts], transformations)
            structure = transmuter.transformed_structures[-1].final_structure

            # to avoid MongoDB errors, ":" is automatically converted to "."
            t_json = transmuter.transformed_structures[-1]
            self.write_additional_data.setdefault("transformations:json", t_json)

        # copy previous inputs
        from_prev = prev_dir is not None
        if prev_dir is not None:
            copy_cp2k_outputs(prev_dir, **self.copy_cp2k_kwargs)

        # write cp2k input files
        self.write_input_set_kwargs["from_prev"] = from_prev
        write_cp2k_input_set(
            structure, self.input_set_generator, **self.write_input_set_kwargs
        )

        # write any additional data
        for filename, data in self.write_additional_data.items():
            dumpfn(data, filename.replace(":", "."))

        # run cp2k
        run_cp2k(**self.run_cp2k_kwargs)

        # parse cp2k outputs
        task_doc = TaskDocument.from_directory(Path.cwd(), **self.task_document_kwargs)
        task_doc.task_label = self.name

        # decide whether child jobs should proceed
        stop_children = should_stop_children(task_doc, **self.stop_children_kwargs)

        # cleanup files to save disk space
        cleanup_cp2k_outputs(directory=Path.cwd())

        # gzip folder
        gzip_output_folder(
            directory=Path.cwd(),
            setting=SETTINGS.CP2K_ZIP_FILES,
            files_list=_FILES_TO_ZIP,
        )

        if SETTINGS.CP2K_ZIP_FILES == "atomate":
            gzip_files(include_files=_FILES_TO_ZIP, allow_missing=True, force=True)
        elif SETTINGS.CP2K_ZIP_FILES:
            gzip_files(force=True)

        return Response(
            stop_children=stop_children,
            stored_data={"custodian": task_doc.custodian},
            output=task_doc if self.store_output_data else None,
        )
