"""Core flows for OpenMM module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.classical_md.openmm.jobs.core import (
    EnergyMinimizationMaker,
    NPTMaker,
    NVTMaker,
    TempChangeMaker,
)
from atomate2.classical_md.utils import create_list_summing_to

if TYPE_CHECKING:
    from pathlib import Path

    from emmet.core.classical_md import ClassicalMDTaskDocument
    from openff.interchange import Interchange


@dataclass
class AnnealMaker(Maker):
    """A maker class for making annealing workflows.

    Attributes
    ----------
    name : str
        The name of the annealing job. Default is "anneal".
    raise_temp_maker : TempChangeMaker
        The maker for raising the temperature.
        Default is a TempChangeMaker with a target temperature of 400.
    nvt_maker : NVTMaker
        The maker for holding the temperature. Default is an NVTMaker.
    lower_temp_maker : TempChangeMaker
        The maker for lowering the temperature.
        Default is a TempChangeMaker.
    """

    name: str = "anneal"
    tags: list[str] = field(default_factory=list)
    raise_temp_maker: TempChangeMaker = field(
        default_factory=lambda: TempChangeMaker(temperature=400)
    )
    nvt_maker: NVTMaker = field(default_factory=NVTMaker)
    lower_temp_maker: TempChangeMaker = field(default_factory=TempChangeMaker)

    @classmethod
    def from_temps_and_steps(
        cls,
        name: str = "anneal",
        anneal_temp: int = 400,
        final_temp: int = 298,
        n_steps: int | tuple[int, int, int] = 1500000,
        temp_steps: int | tuple[int, int, int] | None = None,
        job_names: tuple[str, str, str] = ("raise temp", "hold temp", "lower temp"),
        **kwargs,
    ) -> AnnealMaker:
        """Create an AnnealMaker from the specified temperatures, steps, and job names.

        Parameters
        ----------
        name : str, optional
            The name of the annealing job. Default is "anneal".
        anneal_temp : int, optional
            The annealing temperature. Default is 400.
        final_temp : int, optional
            The final temperature after annealing. Default is 298.
        n_steps : int or Tuple[int, int, int], optional
            The number of steps for each stage of annealing.
            If an integer is provided, it will be divided into three equal parts.
            If a tuple of three integers is provided, each value represents the
            steps for the corresponding stage. Default is 1500000.
        temp_steps : int or Tuple[int, int, int], optional
            The number of temperature steps for raising and
            lowering the temperature. If an integer is provided, it will be used
            for both stages. If a tuple of three integers is provided, each value
            represents the temperature steps for the corresponding stage.
            Default is None and all jobs will automatically determine temp_steps.
        job_names : Tuple[str, str, str], optional
            The names for the jobs in each stage of annealing.
            Default is ("raise temp", "hold temp", "lower temp").
        **kwargs
            Additional keyword arguments to be passed to the job makers.

        Returns
        -------
        AnnealMaker
            An AnnealMaker instance with the specified parameters.
        """
        if isinstance(n_steps, int):
            n_steps = tuple(create_list_summing_to(n_steps, 3))
        if isinstance(temp_steps, int) or temp_steps is None:
            temp_steps = (temp_steps, temp_steps, temp_steps)

        raise_temp_maker = TempChangeMaker(
            n_steps=n_steps[0],
            name=job_names[0],
            temperature=anneal_temp,
            temp_steps=temp_steps[0],
            **kwargs,
        )
        nvt_maker = NVTMaker(
            n_steps=n_steps[1], name=job_names[1], temperature=anneal_temp, **kwargs
        )
        lower_temp_maker = TempChangeMaker(
            n_steps=n_steps[2],
            name=job_names[2],
            temperature=final_temp,
            temp_steps=temp_steps[2],
            **kwargs,
        )
        return cls(
            name=name,
            raise_temp_maker=raise_temp_maker,
            nvt_maker=nvt_maker,
            lower_temp_maker=lower_temp_maker,
        )

    def make(
        self,
        interchange: Interchange | bytes,
        prev_task: ClassicalMDTaskDocument | None = None,
        output_dir: str | Path | None = None,
    ) -> Flow:
        """Anneal the simulation at the specified temperature.

        Annealing takes place in 3 stages, heating, holding, and cooling.
        After heating, and holding, the system will cool to the temperature in
        prev_task.

        Parameters
        ----------
        interchange : Interchange
            Interchange object containing the system to be annealed.
        prev_task : Optional[ClassicalMDTaskDocument]
            Previous task to use as a starting point for the annealing.
        output_dir : Optional[str]
            Directory to write reporter files to.

        Returns
        -------
        Job
            A OpenMM job containing one npt run.
        """
        raise_temp_job = self.raise_temp_maker.make(
            interchange=interchange,
            prev_task=prev_task,
            output_dir=output_dir,
        )

        nvt_job = self.nvt_maker.make(
            interchange=raise_temp_job.output.interchange,
            prev_task=raise_temp_job.output,
            output_dir=output_dir,
        )

        job_tags = (self.lower_temp_job.tags or []) + (self.tags or []) or None
        self.lower_temp_job.tags = job_tags

        lower_temp_job = self.lower_temp_maker.make(
            interchange=nvt_job.output.interchange,
            prev_task=nvt_job.output,
            output_dir=output_dir,
        )

        return Flow(
            [raise_temp_job, nvt_job, lower_temp_job], output=lower_temp_job.output
        )


@dataclass
class ProductionMaker(Maker):
    """Run a production simulation.

    The production simulation links together energy minimization, NPT equilibration,
    annealing, and NVT production.

    Attributes
    ----------
    name : str
        The name of the production job. Default is "production".
    energy_maker : EnergyMinimizationMaker
        The maker for energy minimization.
    npt_maker : NPTMaker
        The maker for NPT equilibration.
    anneal_maker : AnnealMaker
        The maker for annealing.
    nvt_maker : NVTMaker
        The maker for NVT production.
    """

    name: str = "production"
    tags: list[str] = field(default_factory=list)
    energy_maker: EnergyMinimizationMaker = field(
        default_factory=EnergyMinimizationMaker
    )
    npt_maker: NPTMaker = field(default_factory=NPTMaker)
    anneal_maker: AnnealMaker = field(default_factory=AnnealMaker)
    nvt_maker: NVTMaker = field(default_factory=NVTMaker)

    def make(
        self,
        interchange: Interchange | bytes,
        prev_task: ClassicalMDTaskDocument | None = None,
        output_dir: str | Path | None = None,
    ) -> Flow:
        """Run the production simulation using the provided Interchange object.

        Parameters
        ----------
        interchange : Interchange
            The Interchange object containing the system
            to simulate.
        prev_task : Optional[ClassicalMDTaskDocument]
            The previous task to use as
            a starting point for the production simulation.
        output_dir : Optional[Union[str, Path]]
            The directory to write reporter files to.

        Returns
        -------
        Flow
            A Flow object containing the OpenMM jobs for the simulation.
        """
        energy_job = self.energy_maker.make(
            interchange=interchange,
            prev_task=prev_task,
            output_dir=output_dir,
        )

        pressure_job = self.npt_maker.make(
            interchange=energy_job.output.interchange,
            prev_task=energy_job.output,
            output_dir=output_dir,
        )

        anneal_flow = self.anneal_maker.make(
            interchange=pressure_job.output.interchange,
            prev_task=pressure_job.output,
            output_dir=output_dir,
        )

        self.nvt_maker.tags = (self.nvt_maker.tags or []) + (self.tags or []) or None

        nvt_job = self.nvt_maker.make(
            interchange=anneal_flow.output.interchange,
            prev_task=anneal_flow.output,
            output_dir=output_dir,
        )

        return Flow(
            [
                energy_job,
                pressure_job,
                anneal_flow,
                nvt_job,
            ],
            output=nvt_job.output,
        )
