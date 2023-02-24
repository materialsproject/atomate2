"""Schemas for defect documents."""

import logging
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field
from pymatgen.io.vasp.outputs import WSWQ

logger = logging.getLogger(__name__)

__all__ = ["FiniteDifferenceDocument"]


class FiniteDifferenceDocument(BaseModel):
    """Collection of computed wavefunction overlap objects.

    Overlaps obtained using a single reference WAVECAR and a list of WAVECARs
    from distorted structures.
    """

    wswqs: List[WSWQ]

    dir_name: str = Field(
        None, description="Directory where the WSWQ calculations are performed"
    )
    ref_dir: str = Field(
        None, description="Directory where the reference W(0) wavefunction comes from"
    )
    distorted_dirs: List[str] = Field(
        None,
        description="Directories where the distorted W(Q) wavefunctions come from",
    )

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        ref_dir: Optional[Union[str, Path]] = None,
        distorted_dirs: Optional[List[str]] = None,
    ) -> "FiniteDifferenceDocument":
        """
        Read the FiniteDiff file.

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
        wswq_documents = []
        for f in ordered_files:
            wswq_documents.append(WSWQ.from_file(f))

        return cls(
            wswqs=wswq_documents,
            dir_name=str(wswq_dir),
            ref_dir=str(ref_dir),
            distorted_dirs=list(map(str, distorted_dirs)),
        )
