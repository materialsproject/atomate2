"""Flows for calculating dielectric properties."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.common.jobs.dielectric import dielectric_analysis

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Job, OutputReference
    from pymatgen.core.structure import Structure


@dataclass
class BaseDielectricMaker(Maker, ABC):
    """
    Maker to create a flow with a relaxation and a dielectric calculation.

    The dielectric response is sensitive to residual forces, so the structure should
    be well relaxed. The results are collected in a :obj:`.DielectricDocument`.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .Maker or None
        The maker to use for the relaxation. Set to ``None`` to skip the relaxation.
    dielectric_maker : .Maker
        The maker to use for the dielectric calculation.
    """

    name: str = "relax and dielectric"
    relax_maker: Maker | None = None
    dielectric_maker: Maker = None

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """Run a relaxation and then a dielectric calculation.

        Parameters
        ----------
        structure: .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from.

        Returns
        -------
        Flow
            A relax and dielectric flow.
        """
        jobs: list[Job | Flow] = []
        if self.relax_maker is not None:
            relax_job = self.relax_maker.make(structure, prev_dir=prev_dir)
            jobs.append(relax_job)
            structure = relax_job.output.structure
            prev_dir = relax_job.output.dir_name

        dielectric_job = self.dielectric_maker.make(structure, prev_dir=prev_dir)
        analysis = dielectric_analysis(
            dielectric_job.output.structure,
            **self.get_dielectric_tensors(dielectric_job.output),
        )
        jobs += [dielectric_job, analysis]

        return Flow(jobs, analysis.output, name=self.name)

    @staticmethod
    @abstractmethod
    def get_dielectric_tensors(output: OutputReference) -> dict[str, OutputReference]:
        """
        Get the tensors from the output of the dielectric job.

        Parameters
        ----------
        output : .OutputReference
            The output of the dielectric job.

        Returns
        -------
        dict
            Keyword arguments for :obj:`.DielectricDocument.from_tensors`.
        """
