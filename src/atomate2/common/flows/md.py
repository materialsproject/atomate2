"""Flows for molecular dynamics that are shared between codes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.ase.md import AseMDMaker
from atomate2.common.jobs.md import get_md_restart_structure

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Job
    from pymatgen.core import Structure


@dataclass
class ChainedMDMaker(Maker):
    """
    Maker to run one MD as several consecutive MD jobs, with VASP or a force field.

    The first MD job starts from the input structure. Each later one starts from
    the final positions and velocities of the previous one. The thermostat
    variables are not carried over, so they start again from zero in each job.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    md_makers: list[Maker]
        Makers of the MD jobs, in the order of the trajectory. VASP makers and
        ASE makers, such as the force field ones, are supported. An ASE maker
        that is followed by another job must write its trajectory in the ASE
        format, with traj_file set.
    """

    name: str = "chained MD"
    md_makers: list[Maker] = field(default_factory=list)

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """
        Make a flow of consecutive MD jobs.

        Parameters
        ----------
        structure: Structure
            The structure the first MD job starts from. Its magnetic moments, if
            any, are also given to the later MD jobs.
        prev_dir: str | Path | None
            A previous calculation directory, passed to each MD job.

        Returns
        -------
        Flow
            Its output holds the directories and the uuids of the MD jobs, in
            the order of the trajectory.
        """
        jobs: list[Job] = []
        md_jobs: list[Job] = []
        md_structure = structure
        for idx, maker in enumerate(self.md_makers, start=1):
            md_job = maker.make(md_structure, prev_dir=prev_dir)
            if len(self.md_makers) > 1:
                md_job.append_name(f" {idx}/{len(self.md_makers)}")
            jobs.append(md_job)
            md_jobs.append(md_job)
            if idx < len(self.md_makers):
                traj_file = None
                if isinstance(maker, AseMDMaker):
                    if maker.traj_file is None or maker.traj_file_fmt != "ase":
                        raise ValueError(
                            f"{maker.name} must write its trajectory in the ASE "
                            "format, so that the next MD job can continue from it."
                        )
                    traj_file = str(maker.traj_file)
                restart = get_md_restart_structure(
                    md_job.output.dir_name, structure, traj_file
                )
                jobs.append(restart)
                md_structure = restart.output
        output = {
            "dir_names": [md_job.output.dir_name for md_job in md_jobs],
            "uuids": [md_job.uuid for md_job in md_jobs],
        }
        return Flow(jobs, output, name=self.name)
