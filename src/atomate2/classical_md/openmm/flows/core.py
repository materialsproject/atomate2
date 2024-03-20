from typing import Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from jobflow import Maker, Flow
from typing import Tuple

from openff.interchange import Interchange

from atomate2.classical_md.openmm.jobs.core import (
    EnergyMinimizationMaker,
    NPTMaker,
    NVTMaker,
    TempChangeMaker,
)
from atomate2.classical_md.schemas import ClassicalMDTaskDocument


@dataclass
class AnnealMaker(Maker):
    """
    steps : Union[Tuple[int, int, int], int]
    """

    name: str = "anneal"
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
        steps: Union[int, Tuple[int, int, int]] = 1500000,
        temp_steps: Union[int, Tuple[int, int, int]] = 100,
        job_names: Tuple[str, str, str] = ("raise temp", "hold temp", "lower temp"),
        **kwargs,
    ):
        if isinstance(steps, int):
            steps = (steps // 3, steps // 3, steps - 2 * (steps // 3))
        if isinstance(temp_steps, int):
            temp_steps = (temp_steps, temp_steps, temp_steps)

        raise_temp_maker = TempChangeMaker(
            steps=steps[0],
            name=job_names[0],
            temperature=anneal_temp,
            temp_steps=temp_steps[0],
            **kwargs,
        )
        nvt_maker = NVTMaker(
            steps=steps[1], name=job_names[1], temperature=anneal_temp, **kwargs
        )
        lower_temp_maker = TempChangeMaker(
            steps=steps[2],
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
        interchange: Interchange,
        prev_task: Optional[ClassicalMDTaskDocument] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Anneal the simulation at the specified temperature.

        Annealing takes place in 3 stages, heating, holding, and cooling. The three
        elements of steps specify the length of each stage. After heating, and holding,
        the system will cool to its original temperature.

        Parameters
        ----------
        interchange : Interchange
            Interchange object containing the system to be annealed.
        prev_task : Optional[ClassicalMDTaskDocument]
            Previous task to use as a starting point for the annealing.
        output_dir : str
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
    """
    Class for running
    """

    name: str = "production"
    energy_maker: EnergyMinimizationMaker = field(
        default_factory=EnergyMinimizationMaker
    )
    npt_maker: NPTMaker = field(default_factory=NPTMaker)
    anneal_maker: AnnealMaker = field(default_factory=AnnealMaker)
    nvt_maker: NVTMaker = field(default_factory=NVTMaker)

    def make(
        self,
        interchange: Interchange,
        prev_task: Optional[ClassicalMDTaskDocument] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """

        Parameters
        ----------
        interchange : Interchange
            Interchange object containing the system to be annealed.
        prev_task : Optional[ClassicalMDTaskDocument]
            Previous task to use as a starting point for the annealing.
        output_dir : str
            Directory to write reporter files to.
        Returns
        -------

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
