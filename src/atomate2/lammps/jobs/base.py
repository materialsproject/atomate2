"""Base job maker for LAMMPS calculations."""

import glob
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from emmet.core.vasp.task_valid import TaskState
from jobflow import Maker, Response, job
from pymatgen.core import Structure
from pymatgen.io.lammps.generators import (
    BaseLammpsSetGenerator,
    CombinedData,
    LammpsData,
)

from atomate2.common.files import gzip_files
from atomate2.lammps.files import write_lammps_input_set
from atomate2.lammps.run import run_lammps
from atomate2.lammps.schemas.task import LammpsTaskDocument, StoreTrajectoryOption

_DATA_OBJECTS: list[str] = [
    "raw_log_file",
    "inputs",
    "trajectories",
    "dump_files",
]

__all__ = ("BaseLammpsMaker", "lammps_job")


class LammpsRunError(Exception):
    """Custom exception for LAMMPS jobs."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def lammps_job(method: Callable) -> job:
    """Job decorator for LAMMPS jobs."""
    return job(method, data=_DATA_OBJECTS, output_schema=LammpsTaskDocument)


@dataclass
class BaseLammpsMaker(Maker):
    """
    Basic Maker class for LAMMPS jobs.

    name: str
        Name of the job
    input_set_generator: BaseLammpsGenerator
        Input set generator for the job, default is the BaseLammpsSetGenerator.
        Check the sets module for more options on input kwargs.
    write_input_set_kwargs: dict
        Additional kwargs to write_lammps_input_set
    run_lammps_kwargs: dict
        Additional kwargs to run_lammps
    task_document_kwargs: dict
        Additional kwargs to TaskDocument.from_directory
    write_additional_data: dict
        Additional data to write to the job directory
    """

    name: str = "Base LAMMPS job"
    input_set_generator: BaseLammpsSetGenerator = field(
        default_factory=BaseLammpsSetGenerator
    )
    force_field: str | dict | None = field(default=None)
    write_input_set_kwargs: dict = field(default_factory=dict)
    run_lammps_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)
    write_additional_data: LammpsData | CombinedData = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post-initialization warnings for the job."""
        if (
            self.task_document_kwargs.get("store_trajectory", StoreTrajectoryOption.NO)
            != StoreTrajectoryOption.NO
        ):
            warnings.warn(
                "Trajectory data might be large, only store if absolutely necessary. \
                Consider manually parsing the dump files instead.",
                stacklevel=1,
            )

        if self.force_field:
            self.input_set_generator.force_field = self.force_field

    @lammps_job
    def make(
        self,
        input_structure: Structure | Path | LammpsData = None,
        prev_dir: Path | str = None,
    ) -> Response:
        """Run a LAMMPS calculation."""
        if prev_dir:
            restart_files = glob.glob(os.path.join(prev_dir, "*restart*"))
            if len(restart_files) != 1:
                raise FileNotFoundError(
                    "No/More than one restart file found in the previous directory. \
                        If present, it should have the extension '.restart'!"
                )

            self.input_set_generator.update_settings(
                {"read_restart": os.path.join(prev_dir, restart_files[0])}
            )

        if isinstance(input_structure, Path):
            input_structure = LammpsData.from_file(
                input_structure,
                atom_style=self.input_set_generator.settings.get("atom_style", "full"),
            )

        write_lammps_input_set(
            data=input_structure,
            input_set_generator=self.input_set_generator,
            additional_data=self.write_additional_data,
            **self.write_input_set_kwargs,
        )

        run_lammps(**self.run_lammps_kwargs)

        task_doc = LammpsTaskDocument.from_directory(
            os.getcwd(), task_label=self.name, **self.task_document_kwargs
        )

        if task_doc.state == TaskState.ERROR:
            try:
                error = ""
                for index, line in enumerate(task_doc.raw_log_file.split("\n")):
                    if "ERROR" in line:
                        error = error.join(task_doc.raw_log_file.split("\n")[index:])
                        break
            except ValueError:
                error = "could not parse log file"
            raise LammpsRunError(f"Task {task_doc.task_label} failed, error: {error}")

        gzip_files(".")

        return Response(output=task_doc)
