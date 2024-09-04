"""Definition of base VASP job maker."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING, Callable

from atomate2.jdftx.emmet.jdftx_tasks import TaskDoc
from jobflow import Maker, Response, job
from monty.serialization import dumpfn
from pymatgen.core.trajectory import Trajectory
from pymatgen.electronic_structure.bandstructure import (
    BandStructure,
    BandStructureSymmLine,
)
from pymatgen.electronic_structure.dos import DOS, CompleteDos, Dos
from pymatgen.io.vasp import Chgcar, Locpot, Wavecar

from atomate2 import SETTINGS #TODO we can add JDFTx workflow default settings this way
from atomate2.common.files import gzip_output_folder
from atomate2.jdftx.sets.base import JdftxInputGenerator
from atomate2.vasp.files import copy_vasp_outputs, write_vasp_input_set
from atomate2.vasp.run import run_vasp, should_stop_children

if TYPE_CHECKING:
    from pymatgen.core import Structure


_BADER_EXE_EXISTS = bool(which("bader") or which("bader.exe"))
_CHARGEMOL_EXE_EXISTS = bool(
    which("Chargemol_09_26_2017_linux_parallel")
    or which("Chargemol_09_26_2017_linux_serial")
    or which("chargemol")
)

_DATA_OBJECTS = [ # TODO update relevant list for JDFTx
    BandStructure,
    BandStructureSymmLine,
    DOS,
    Dos,
    CompleteDos,
    Locpot,
    Chgcar,
    Wavecar,
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

# Files to zip: inputs, outputs and additionally generated files
_FILES_TO_ZIP = (
    _INPUT_FILES
    + _OUTPUT_FILES
    + "custodian.json" # probably won't have this output 
)


def jdftx_job(method: Callable) -> job:
    """
    Decorate the ``make`` method of JDFTx job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.Job` that configures common
    settings for all JDFTx jobs. For example, it ensures that large data objects
    (band structures, density of states, LOCPOT, CHGCAR, etc) are all stored in the
    atomate2 data store. It also configures the output schema to be a VASP
    :obj:`.TaskDoc`.

    Any makers that return JDFTx jobs (not flows) should decorate the ``make`` method
    with @jdftx_job. For example:

    .. code-block:: python

        class MyJjdftxMaker(BaseJdftxMaker):
            @jdftx_job
            def make(structure):
                # code to run JDFTx job.
                pass

    Parameters
    ----------
    method : callable
        A BaseJdftxMaker.make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate VASP jobs.
    """
    return job(method, data=_DATA_OBJECTS, output_schema=TaskDoc)


@dataclass
class BaseJdftxMaker(Maker):
    """
    Base VASP job maker.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
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

    name: str = "base JDFTx job"
    input_set_generator: JdftxInputGenerator = field(default_factory=JdftxInputGenerator)
    write_input_set_kwargs: dict = field(default_factory=dict)
    copy_vasp_kwargs: dict = field(default_factory=dict)
    run_vasp_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
    stop_children_kwargs: dict = field(default_factory=dict)
    write_additional_data: dict = field(default_factory=dict)

    @jdftx_job
    def make(
        self, structure: Structure, prev_dir: str | Path | None = None
    ) -> Response:
        """Run a JDFTx calculation.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous JDFTx calculation directory to copy output files from.

        Returns
        -------
            Response: A response object containing the output, detours and stop
                commands of the JDFTx run.
        """
        # copy previous inputs
        from_prev = prev_dir is not None
        if prev_dir is not None:
            copy_vasp_outputs(prev_dir, **self.copy_vasp_kwargs)

        self.write_input_set_kwargs.setdefault("from_prev", from_prev)

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
        task_doc = get_jdftx_task_document(Path.cwd(), **self.task_document_kwargs)
        task_doc.task_label = self.name

        # decide whether child jobs should proceed
        stop_children = should_stop_children(task_doc, **self.stop_children_kwargs)

        # gzip folder
        gzip_output_folder(
            directory=Path.cwd(),
            setting=SETTINGS.VASP_ZIP_FILES,
            files_list=_FILES_TO_ZIP,
        )

        return Response(
            stop_children=stop_children,
            stored_data={"custodian": task_doc.custodian},
            output=task_doc,
        )


def get_jdftx_task_document(path: Path | str, **kwargs) -> TaskDoc:
    """Get JDFTx Task Document using atomate2 settings."""
    # kwargs.setdefault("store_additional_json", SETTINGS.VASP_STORE_ADDITIONAL_JSON)

    # kwargs.setdefault(
    #     "volume_change_warning_tol", SETTINGS.VASP_VOLUME_CHANGE_WARNING_TOL
    # )

    # if SETTINGS.VASP_RUN_BADER:
    #     kwargs.setdefault("run_bader", _BADER_EXE_EXISTS)
    #     if not _BADER_EXE_EXISTS:
    #         warnings.warn(
    #             f"{SETTINGS.VASP_RUN_BADER=} but bader executable not found on path",
    #             stacklevel=1,
    #         )
    # if SETTINGS.VASP_RUN_DDEC6:
    #     # if VASP_RUN_DDEC6 is True but _CHARGEMOL_EXE_EXISTS is False, just silently
    #     # skip running DDEC6
    #     run_ddec6: bool | str = _CHARGEMOL_EXE_EXISTS
    #     if run_ddec6 and isinstance(SETTINGS.DDEC6_ATOMIC_DENSITIES_DIR, str):
    #         # if DDEC6_ATOMIC_DENSITIES_DIR is a string and directory at that path
    #         # exists, use as path to the atomic densities
    #         if Path(SETTINGS.DDEC6_ATOMIC_DENSITIES_DIR).is_dir():
    #             run_ddec6 = SETTINGS.DDEC6_ATOMIC_DENSITIES_DIR
    #         else:
    #             # if the directory doesn't exist, warn the user and skip running DDEC6
    #             warnings.warn(
    #                 f"{SETTINGS.DDEC6_ATOMIC_DENSITIES_DIR=} does not exist, skipping "
    #                 "DDEC6",
    #                 stacklevel=1,
    #             )
    #     kwargs.setdefault("run_ddec6", run_ddec6)

    #     if not _CHARGEMOL_EXE_EXISTS:
    #         warnings.warn(
    #             f"{SETTINGS.VASP_RUN_DDEC6=} but chargemol executable not found on "
    #             "path",
    #             stacklevel=1,
    #         )

    # kwargs.setdefault("store_volumetric_data", SETTINGS.VASP_STORE_VOLUMETRIC_DATA)

    return TaskDoc.from_directory(path, **kwargs)
