"""NEB document schemas specific to VASP."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING


from emmet.core.neb import NebMethod, NebTaskDoc
from emmet.core.tasks import TaskDoc

from atomate2.common.schemas.neb import NebResult

if TYPE_CHECKING:
    from typing_extensions import Self


class VaspNebResult(NebResult):
    """Class for parsing NEB calculations with VASP."""

    @classmethod
    def from_directories(
        cls,
        endpoint_directories: list[str | Path],
        neb_directory: str | Path,
        **neb_doc_kwargs,
    ) -> Self:
        """Get NEB analysis from endpoint and image directories."""
        endpoint_tasks = [
            TaskDoc.from_directory(endpoint_dir)
            for endpoint_dir in endpoint_directories
        ]
        neb_task = NebTaskDoc.from_directory(neb_directory)

        properties = {}
        for k in ("structure", "energy"):
            properties[k] = [
                getattr(endpoint_tasks[0].output, k, None),
                *[
                    getattr(calc.output, k, None)
                    for calc in neb_task.image_calculations
                ],
                getattr(endpoint_tasks[1].output, k, None),
            ]

        return cls(
            structures=properties["structure"],
            energies=properties["energy"],
            images = [image_calc.output.structure for image_calc in neb_task.image_calculations],
            method = (
                NebMethod.CLIMBING_IMAGE
                if neb_task.inputs.incar.get("LCLIMB",False)
                else NebMethod.STANDARD
            ),
            **neb_doc_kwargs,
        )
