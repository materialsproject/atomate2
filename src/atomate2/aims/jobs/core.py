"""Define all Core FHI-aims jobs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from jobflow import Response, job
from monty.serialization import dumpfn
from pymatgen.io.aims.parsers import read_aims_output
from pymatgen.io.aims.sets.bs import BandStructureSetGenerator, GWSetGenerator
from pymatgen.io.aims.sets.core import (
    RelaxSetGenerator,
    SocketIOSetGenerator,
    StaticSetGenerator,
)

from atomate2 import SETTINGS
from atomate2.aims.files import cleanup_aims_outputs, write_aims_input_set
from atomate2.aims.jobs.base import _FILES_TO_ZIP, BaseAimsMaker
from atomate2.aims.run import run_aims_socket, should_stop_children
from atomate2.aims.schemas.task import AimsTaskDoc
from atomate2.common.files import gzip_output_folder

if TYPE_CHECKING:
    from pymatgen.core import Molecule, Structure
    from pymatgen.io.aims.sets.base import AimsInputGenerator

logger = logging.getLogger(__name__)


@dataclass
class StaticMaker(BaseAimsMaker):
    """Maker to create FHI-aims SCF jobs.

    Parameters
    ----------
    calc_type: str
        The type key for the calculation
    name: str
        The job name
    input_set_generator: .AimsInputGenerator
        The InputGenerator for the calculation
    """

    calc_type: str = "scf"
    name: str = "SCF Calculation"
    input_set_generator: AimsInputGenerator = field(default_factory=StaticSetGenerator)


@dataclass
class RelaxMaker(BaseAimsMaker):
    """Maker to create relaxation calculations.

    Parameters
    ----------
    calc_type: str
        The type key for the calculation
    name: str
        The job name
    input_set_generator: .AimsInputGenerator
        The InputGenerator for the calculation
    """

    calc_type: str = "relax"
    name: str = "Relaxation calculation"
    input_set_generator: AimsInputGenerator = field(default_factory=RelaxSetGenerator)

    @classmethod
    def fixed_cell_relaxation(cls, *args, **kwargs) -> RelaxMaker:
        """Create a fixed cell relaxation maker."""
        return cls(
            input_set_generator=RelaxSetGenerator(*args, relax_cell=False, **kwargs),
            name=cls.name + " (fixed cell)",
        )

    @classmethod
    def full_relaxation(cls, *args, **kwargs) -> RelaxMaker:
        """Create a full relaxation maker."""
        return cls(
            input_set_generator=RelaxSetGenerator(*args, relax_cell=True, **kwargs)
        )


@dataclass
class SocketIOStaticMaker(BaseAimsMaker):
    """Maker for the SocketIO calculator in FHI-aims.

    Parameters
    ----------
    calc_type: str
        The type key for the calculation
    name: str
        The job name
    host: str
        The name of the host to maintain the socket server on
    port: int
        The port number the socket server will listen on
    input_set_generator: .AimsInputGenerator
        The InputGenerator for the calculation
    """

    calc_type: str = "multi_scf"
    name: str = "SCF Calculations Socket"
    host: str = "localhost"
    port: int = 12345
    input_set_generator: AimsInputGenerator = field(
        default_factory=SocketIOSetGenerator
    )

    @job
    def make(
        self,
        structure: list[Structure | Molecule],
        prev_dir: str | Path | None = None,
    ) -> Response:
        """Run multiple FHI-aims calculation with the socket.

        Calculate the properties for multiple structures using the same parameters
        using socket communication to speed up the calculations.

        Parameters
        ----------
        structure : list[Molecule | Structure]
            The list of structure objects to run FHI-aims on
        prev_dir : str or Path or None
            A previous FHI-aims calculation directory to copy output files from.

        Returns
        -------
        The output response for the calculations
        """
        # copy previous inputs
        if not isinstance(structure, list):
            structure = [structure]

        from_prev = prev_dir is not None
        if from_prev:
            hostless_prev_dir = str(prev_dir).split(":")[1]
            images = read_aims_output(f"{hostless_prev_dir}/aims.out")
            if not isinstance(images, Sequence):
                images = [images]

            for img in images:
                img.calc = None

            for ii in range(-1 * len(structure), 0, -1):
                if structure[ii] in images:
                    del structure[ii]

        # write aims input files
        self.write_input_set_kwargs["prev_dir"] = prev_dir
        write_aims_input_set(
            structure[0], self.input_set_generator, **self.write_input_set_kwargs
        )

        # write any additional data
        for filename, data in self.write_additional_data.items():
            dumpfn(data, filename.replace(":", "."))

        # run FHI-aims
        run_aims_socket(structure, **self.run_aims_kwargs)

        # parse FHI-aims outputs
        task_doc = AimsTaskDoc.from_directory(Path.cwd(), **self.task_document_kwargs)
        task_doc.task_label = self.name

        # decide whether child jobs should proceed
        stop_children = should_stop_children(task_doc, **self.stop_children_kwargs)

        # cleanup files to save disk space
        cleanup_aims_outputs(directory=Path.cwd())

        # gzip folder
        gzip_output_folder(
            directory=Path.cwd(),
            setting=SETTINGS.VASP_ZIP_FILES,
            files_list=_FILES_TO_ZIP,
        )

        return Response(
            stop_children=stop_children,
            output=task_doc if self.store_output_data else None,
        )


@dataclass
class BandStructureMaker(BaseAimsMaker):
    """A job Maker for a band structure calculation.

    Parameters
    ----------
    calc_type: str
        The type key for the calculation
    name: str
        The job name
    input_set_generator: .BandStructureSetGenerator
        The InputGenerator for the calculation
    """

    calc_type = "band_structure"
    name: str = "bands"
    input_set_generator: BandStructureSetGenerator = field(
        default_factory=BandStructureSetGenerator
    )


@dataclass
class GWMaker(BaseAimsMaker):
    """A job Maker for a GW band structure calculation.

    Parameters
    ----------
    calc_type: str
        The type key for the calculation
    name: str
        The job name
    input_set_generator: .GWSetGenerator
        The InputGenerator for the calculation
    """

    calc_type = "gw"
    name: str = "GW"
    input_set_generator: GWSetGenerator = field(default_factory=GWSetGenerator)
