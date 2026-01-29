"""Schemas for defect documents."""

import logging
from pathlib import Path
from typing import Union

from pydantic import BaseModel, Field
from pymatgen.io.vasp.outputs import WSWQ
from typing_extensions import Self

logger = logging.getLogger(__name__)


class FiniteDifferenceDocument(BaseModel):
    """Collection of computed wavefunction overlap objects.

    Overlaps obtained using a single reference WAVECAR and a list of WAVECARs
    from distorted structures.
    """

    wswqs: list[WSWQ]

    dir_name: str = Field(
        None, description="Directory where the WSWQ calculations are performed"
    )
    ref_dir: str = Field(
        None, description="Directory where the reference W(0) wavefunction comes from"
    )
    distorted_dirs: list[str] = Field(
        None,
        description="Directories where the distorted W(Q) wavefunctions come from",
    )

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        ref_dir: Union[str, Path] | None = None,
        distorted_dirs: list[str] | None = None,
    ) -> Self:
        """Read the FiniteDiff file.

        Parameters
        ----------
        directory : str | Path
            Path to the FiniteDiff directory.
        ref_dir : str | Path
            Directory where the reference W(0) wavefunction comes from.
        distorted_dirs : List[str | Path]
            List of directories where the distorted W(Q) wavefunctions come from.

        Returns
        -------
        FiniteDiffDocument
            FiniteDiffDocument object.
        """
        wswq_dir = Path(directory)
        files = list(Path(wswq_dir).glob("WSWQ.[0-9]*"))
        ordered_files = sorted(files, key=lambda x: int(x.name.split(".")[1]))
        wswq_documents = [WSWQ.from_file(file) for file in ordered_files]

        return cls(
            wswqs=wswq_documents,
            dir_name=str(wswq_dir),
            ref_dir=str(ref_dir),
            distorted_dirs=list(map(str, distorted_dirs)),
        )
