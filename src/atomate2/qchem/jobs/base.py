"""Definition of a base QChem Maker"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from jobflow import Maker, Response, job
from monty.serialization import dumpfn
from monty.shutil import gzip_dir
from pymatgen.core.structure import Molecule
from pymatgen.io.qchem.inputs import QCInput

from atomate2.qchem.files import copy_qchem_outputs, write_qchem_input_set
from atomate2.qchem.run import run_qchem, should_stop_children
from atomate2.qchem.schemas.task import TaskDocument
from atomate2.qchem.sets.base import QChemInputGenerator

__all__ = ["BaseQChemMaker", "qchem_job"]

# _DATA_OBJECTS = [
#     BandStructure,
#     BandStructureSymmLine,
#     DOS,
#     Dos,
#     CompleteDos,
#     Locpot,
#     Chgcar,
#     Wavecar,
#     Trajectory,
#     "force_constants",
#     "normalmode_eigenvecs",
# ]

# _DATA_OBJECTS = [QCInput]


def qchem_job(method: Callable):
    """
    Decorate the ``make`` method of QChem job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.Job` that configures common
    settings for all QChem jobs. It also configures the output schema to be a QChem
    :obj:`.TaskDocument`.

    Any makers that return QChem jobs (not flows) should decorate the ``make`` method
    with @qchem_job. For example:

    .. code-block:: python

        class MyQChemMaker(BaseQChemMaker):
            @qchem_job
            def make(molecule):
                # code to run QChem job.
                pass

    Parameters
    ----------
    method : callable
        A BaseQChemMaker.make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate QChem jobs.
    """
    return job(method, data=QCInput, output_schema=TaskDocument)


@dataclass
class BaseQChemMaker(Maker):
    """
    Base QChem job maker.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .QChemInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_qchem_input_set`.
    copy_qchem_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_qchem_outputs`.
    run_qchem_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_qchem`.
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

    name: str = "base qchem job"
    input_set_generator: QChemInputGenerator = field(
        default_factory=QChemInputGenerator
    )
    write_input_set_kwargs: dict = field(default_factory=dict)
    copy_qchem_kwargs: dict = field(default_factory=dict)
    run_qchem_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
    stop_children_kwargs: dict = field(default_factory=dict)
    write_additional_data: dict = field(default_factory=dict)

    @qchem_job
    def make(self, molecule: Molecule, prev_qchem_dir: str | Path | None = None):
        """
        Run a QChem calculation.

        Parameters
        ----------
        molecule : Molecule
            A pymatgen molecule object.
        prev_qchem_dir : str or Path or None
            A previous QChem calculation directory to copy output files from.
        """
        # copy previous inputs
        from_prev = prev_qchem_dir is not None
        if prev_qchem_dir is not None:
            copy_qchem_outputs(prev_qchem_dir, **self.copy_qchem_kwargs)

        if "from_prev" not in self.write_input_set_kwargs:
            self.write_input_set_kwargs["from_prev"] = from_prev

        # write qchem input files
        write_qchem_input_set(
            molecule, self.input_set_generator, **self.write_input_set_kwargs
        )

        # write any additional data
        for filename, data in self.write_additional_data.items():
            dumpfn(data, filename.replace(":", "."))

        # run qchem
        run_qchem(**self.run_qchem_kwargs)

        # parse qchem outputs
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
