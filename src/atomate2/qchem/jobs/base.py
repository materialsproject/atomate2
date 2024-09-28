"""Definition of a base QChem Maker."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from emmet.core.qc_tasks import TaskDoc
from jobflow import Maker, Response, job
from monty.serialization import dumpfn
from monty.shutil import gzip_dir
from pymatgen.io.qchem.inputs import QCInput

from atomate2.qchem.files import copy_qchem_outputs
from atomate2.qchem.run import run_qchem, should_stop_children
from atomate2.qchem.sets.base import QCInputGenerator

if TYPE_CHECKING:
    from collections.abc import Callable

    from pymatgen.core.structure import Molecule

logger = logging.getLogger(__name__)


def qchem_job(method: Callable) -> job:
    """
    Decorate the ``make`` method of QChem job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.Job` that configures common
    settings for all QChem jobs. It also configures the output schema to be a QChem
    :obj:`.TaskDoc`.

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
    return job(method, data=QCInput, output_schema=TaskDoc)


@dataclass
class BaseQCMaker(Maker):
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
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
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
    input_set_generator: QCInputGenerator = field(
        default_factory=lambda: QCInputGenerator(
            job_type="sp", scf_algorithm="diis", basis_set="def2-qzvppd"
        )
    )
    write_input_set_kwargs: dict = field(default_factory=dict)
    copy_qchem_kwargs: dict = field(default_factory=dict)
    run_qchem_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
    stop_children_kwargs: dict = field(default_factory=dict)
    write_additional_data: dict = field(default_factory=dict)
    task_type: str | None = None

    @qchem_job
    def make(
        self,
        molecule: Molecule,
        prev_dir: str | Path | None = None,
        prev_qchem_dir: str | Path | None = None,
    ) -> Response:
        """Run a QChem calculation.

        Parameters
        ----------
        molecule : Molecule
            A pymatgen molecule object.
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from.
        prev_qchem_dir (deprecated): str or Path or None
            A previous QChem calculation directory to copy output files from.
        """
        # copy previous inputs
        if prev_qchem_dir is not None:
            logger.warning(
                "`prev_qchem_dir` will be deprecated in a future release. "
                "Please use `prev_dir` instead."
            )
            if prev_dir is not None:
                logger.warning(
                    "You set both `prev_dir` and `prev_qchem_dir`, "
                    "only `prev_dir` will be used."
                )
            else:
                prev_dir = prev_qchem_dir

        if from_prev := (prev_dir is not None):
            copy_qchem_outputs(prev_dir, **self.copy_qchem_kwargs)

        self.write_input_set_kwargs.setdefault("from_prev", from_prev)

        # write qchem input files
        # self.input_set_generator.get_input_set(molecule).write_inputs()
        self.input_set_generator.get_input_set(molecule)
        self.input_set_generator.get_input_set(molecule).write_input(
            directory=Path.cwd()
        )

        # write any additional data
        for filename, data in self.write_additional_data.items():
            dumpfn(data, filename.replace(":", "."))

        # run qchem
        run_qchem(**self.run_qchem_kwargs)

        # parse qchem outputs
        task_doc = TaskDoc.from_directory(Path.cwd(), **self.task_document_kwargs)
        # task_doc.task_label = self.name
        task_doc.task_type = self.name if self.task_type is None else self.task_type

        # decide whether child jobs should proceed
        stop_children = should_stop_children(task_doc, **self.stop_children_kwargs)

        # gzip folder
        gzip_dir(".")

        return Response(
            stop_children=stop_children,
            stored_data={"custodian": task_doc.custodian},
            output=task_doc,
        )
