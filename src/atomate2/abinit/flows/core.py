"""Core abinit flow makers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from jobflow import Flow, Maker
from pymatgen.core.structure import Structure

from atomate2.abinit.jobs.base import BaseAbinitMaker
from atomate2.abinit.jobs.core import NonScfMaker, ScfMaker


@dataclass
class LineBandStructureMaker(Maker):
    """
    Maker to generate line abinit band structure.

    This is a static calculation followed by a non-self-consistent field
    calculations.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    scf_maker : .BaseAbinitMaker
        The maker to use for the static calculation.
    bs_maker : .BaseAbinitMaker
        The maker to use for the non-self-consistent field calculations.
    """

    name: str = "line band structure"
    scf_maker: BaseAbinitMaker = ScfMaker()
    bs_maker: BaseAbinitMaker = NonScfMaker()

    def make(
        self, structure: Structure, prev_outputs: Optional[Union[str, Path]] = None
    ):
        """
        Create a line mode band structure flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_dirs : str or Path or None
            One or more previous directories for the calculation.

        Returns
        -------
        Flow
            A line mode band structure flow.
        """
        scf_job = self.scf_maker.make(structure, prev_outputs=prev_outputs)
        line_job = self.bs_maker.make(
            structure=structure,
            prev_outputs=scf_job.output,
            previous_abinit_input=scf_job.output.abinit_input,
        )
        jobs = [scf_job, line_job]
        return Flow(jobs, line_job.output, name=self.name)
