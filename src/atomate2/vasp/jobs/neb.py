"""Define NEB VASP jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from glob import glob
from os import mkdir
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from emmet.core.neb import NebTaskDoc
from jobflow import Response, job
from monty.serialization import dumpfn
from pymatgen.io.vasp import Kpoints

from atomate2 import SETTINGS
from atomate2.common.files import gzip_output_folder
from atomate2.common.jobs.neb import NebInterpolation, get_images_from_endpoints
from atomate2.vasp.files import copy_vasp_outputs, write_vasp_input_set
from atomate2.vasp.jobs.base import (
    _DATA_OBJECTS,
    _FILES_TO_ZIP,
    BaseVaspMaker,
    get_vasp_task_document,
)
from atomate2.vasp.run import run_vasp, should_stop_children
from atomate2.vasp.sets.core import NebSetGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from typing_extensions import Self

    from atomate2.vasp.sets.base import VaspInputGenerator


def vasp_neb_job(method: Callable) -> job:
    """
    Decorate the ``make`` method of VASP NEB job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.Job` that configures common
    settings for VASP NEB jobs. For example, it ensures that large data objects
    (band structures, density of states, LOCPOT, CHGCAR, etc) are all stored in the
    atomate2 data store. It also configures the output schema to be a VASP
    :obj:`.NebTaskDoc`.

    Parameters
    ----------
    method : callable
        A BaseVaspMaker.make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate VASP NEB jobs.
    """
    return job(method, data=_DATA_OBJECTS, output_schema=NebTaskDoc)


@dataclass
class NebFromImagesMaker(BaseVaspMaker):
    """
    Maker to create VASP NEB jobs from a set of images.

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

    name: str = "NEB"
    input_set_generator: VaspInputGenerator = field(default_factory=NebSetGenerator)
    run_vasp_kwargs: dict = field(
        default_factory=lambda: {
            "job_type": "neb",
            "vasp_job_kwargs": {
                "output_file": "vasp.out",
                "stderr_file": "std_err.txt",
            },
        }
    )
    lclimb: bool = True
    kpoints_kludge: Kpoints | None = None

    @vasp_neb_job
    def make(
        self,
        images: list[Structure],
        prev_dir: str | Path | None = None,
    ) -> Response:
        """
        Make an NEB job from a list of images.

        Parameters
        ----------
        images : list[Structure]
            A list of NEB images.
        prev_dir : str or Path or None (default)
            A previous directory to copy outputs from.

        """
        num_frames = len(images)
        num_images = num_frames - 2
        self.input_set_generator.num_images = num_images

        for iimage in range(num_frames):
            image_dir = f"{iimage:02}"
            mkdir(image_dir)

            images[iimage].to(f"{image_dir}/POSCAR")

            """
            Bug in VASP 6 compiled with HDF5 support:
            https://www.vasp.at/forum/viewtopic.php?f=3&t=18721&p=23430&hilit=neb+hdf5+images#p23430

            Seems like there's a validation check of whether the
            KPOINTS file used in the head directory
            is the same as in each IMAGE subdirectory.

            Using KSPACING in INCAR gets around this issue; a
            kludge to use KPOINTS is to simply copy it
            to each subdirectory, as we do here
            """
            if isinstance(self.kpoints_kludge, Kpoints):
                self.kpoints_kludge.write_file(f"{image_dir}/KPOINTS")

        # copy previous inputs
        from_prev = prev_dir is not None
        if prev_dir is not None:
            copy_vasp_outputs(prev_dir, **self.copy_vasp_kwargs)

        self.write_input_set_kwargs.setdefault("from_prev", from_prev)

        # write vasp input files
        write_vasp_input_set(
            images[0], self.input_set_generator, **self.write_input_set_kwargs
        )

        # write any additional data
        for filename, data in self.write_additional_data.items():
            dumpfn(data, filename.replace(":", "."))

        # run vasp
        run_vasp(**self.run_vasp_kwargs)

        # parse vasp outputs
        task_doc = get_vasp_task_document(
            Path.cwd(), is_neb=True, **self.task_document_kwargs
        )
        task_doc.task_label = self.name

        # decide whether child jobs should proceed
        stop_children = should_stop_children(task_doc, **self.stop_children_kwargs)

        # gzip folder
        gzip_output_folder(
            directory=Path.cwd(),
            setting=SETTINGS.VASP_ZIP_FILES,
            files_list=_FILES_TO_ZIP,
        )

        for image_dir in glob(str(Path.cwd() / "[0-9][0-9]")):
            gzip_output_folder(
                directory=image_dir,
                setting=SETTINGS.VASP_ZIP_FILES,
                files_list=_FILES_TO_ZIP,
            )

        return Response(
            stop_children=stop_children,
            stored_data={"custodian": task_doc.custodian},
            output=task_doc,
        )


@dataclass
class NebFromEndpointsMaker(NebFromImagesMaker):
    """Maker to create VASP NEB jobs from two endpoints."""

    @vasp_neb_job
    def make(
        self,
        endpoints: tuple[Structure, Structure] | list[Structure],
        num_images: int,
        prev_dir: str | Path = None,
        interpolation_method: NebInterpolation = NebInterpolation.LINEAR,
        **interpolation_kwargs,
    ) -> Self:
        """
        Make an NEB job from a set of endpoints.

        Parameters
        ----------
        endpoints : tuple[Structure,Structure] or list[Structure]
            A set of two endpoints to interpolate NEB images from.
        num_images : int
            The number of images to include in the interpolation.
        prev_dir : str or Path or None (default)
            A previous directory to copy outputs from.
        interpolation_method : .NebInterpolation
            The method to use to interpolate between images.
        **interpolation_kwargs
            kwargs to pass to the interpolation function.
        """
        return super().make(
            get_images_from_endpoints(
                endpoints,
                num_images,
                interpolation_method=interpolation_method,
                **interpolation_kwargs,
            ),
            prev_dir=prev_dir,
        )
