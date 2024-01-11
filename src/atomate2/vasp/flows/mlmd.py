"""MD VASP flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.vasp.jobs.core import MLMDMaker

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure

    from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class MultiMDMaker(Maker):
    """Fake MultiMDMaker before it is merged ..."""

    name: str = "MLFF MD flow"

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
        prev_traj_ids: list[str] | None = None,
    ):
        """Run make."""
        return Flow([])


@dataclass
class MLFFMDMaker(Maker):
    """
    Maker to create MLFF Molecular Dynamics with Vasp.

    This workflow contains an initial HSE static calculation, and then a uniform band
    structure calculation with LOPTICS set. The purpose of the static calculation is
    i) to determine if the material needs magnetism set and ii) to determine the total
    number of bands (the second calculation contains 1.3 * number of bands as the
    initial static) as often the highest bands are not properly converged in VASP.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    static_maker : .BaseVaspMaker
        The maker to use for the static calculation.
    band_structure_maker : .BaseVaspMaker
        The maker to use for the uniform optics calculation.
    """

    name: str = "MLFF MD flow"
    train_maker: BaseVaspMaker = field(default_factory=lambda: MLMDMaker.train())
    refit: bool = True
    train: bool = True
    md_maker: Maker = field(default_factory=lambda: MultiMDMaker())
    # static_maker: BaseVaspMaker = field(default_factory=HSEStaticMaker)
    # band_structure_maker: BaseVaspMaker = field(
    #     default_factory=lambda: HSEBSMaker(
    #         name="hse optics",
    #         input_set_generator=HSEBSSetGenerator(optics=True, mode="uniform"),
    #     )
    # )

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """
        Run a static and then a non-scf optics calculation.

        Parameters
        ----------
        structure: .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A static and nscf with optics flow.
        """
        train_job = self.train_maker.make(structure)
        # static_job = self.static_maker.make(structure, prev_dir=prev_dir)
        # bs_job = self.band_structure_maker.make(
        #     static_job.output.structure, prev_dir=static_job.output.dir_name
        # )
        md_flow = self.md_maker.make(structure)
        return Flow([train_job, md_flow], output=md_flow.output, name=self.name)
