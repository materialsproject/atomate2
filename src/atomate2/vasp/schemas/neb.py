""" NEB document schemas specific to VASP. """

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING

from atomate2.common.schemas.neb import NebResult

from emmet.core.tasks import TaskDoc

if TYPE_CHECKING:
    from typing import Sequence
    from typing_extensions import Self

class VaspNebResult(NebResult):
    """ Class for parsing NEB calculations with VASP. """

    @classmethod
    def from_directories(
        cls,
        endpoint_directories : list[str | Path],
        neb_directory : str | Path ,
        **neb_doc_kwargs
    ) -> Self:
        
        image_directories = glob(f"{neb_directory}/[0-9][0-9]")

        endpoint_tasks = [
            TaskDoc.from_directory(endpoint_dir) for endpoint_dir in endpoint_directories
        ]

        return cls(

        )