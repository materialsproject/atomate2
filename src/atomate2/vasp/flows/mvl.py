"""Materials Virtual Lab (MVL) VASP flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.vasp.jobs.mvl import MVLGWMaker, MVLNonSCFMaker, MVLStaticMaker

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure

    from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class MVLGWBandStructureMaker(Maker):
    """
    Maker to generate VASP band structures with Materials Virtual Lab GW setup.

    .. warning::
        This workflow is only compatible with the Materials Virtual Lab GW setup,
        and it may require additional benchmarks. Please use with caution.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    gw_maker : .BaseVaspMaker
        The maker to use for the GW calculation.
    """

    name: str = "MVL G0W0 band structure"
    static_maker: BaseVaspMaker = field(default_factory=MVLStaticMaker)
    nscf_maker: BaseVaspMaker = field(default_factory=MVLNonSCFMaker)
    gw_maker: BaseVaspMaker = field(default_factory=MVLGWMaker)

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """
        Create a band structure flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A band structure flow.
        """
        static_job = self.static_maker.make(structure, prev_dir=prev_dir)
        nscf_job = self.nscf_maker.make(
            static_job.output.structure, prev_dir=static_job.output.dir_name
        )
        gw_job = self.gw_maker.make(
            nscf_job.output.structure, prev_dir=nscf_job.output.dir_name
        )
        jobs = [static_job, nscf_job, gw_job]

        outputs = {
            "static": static_job.output,
            "nscf": nscf_job.output,
            "gw": gw_job.output,
        }

        return Flow(jobs, outputs, name=self.name)
