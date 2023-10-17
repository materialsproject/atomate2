from tempfile import TemporaryDirectory

from atomate2.openmm.jobs.energy_minimization_maker import EnergyMinimizationMaker
from atomate2.openmm.jobs.npt_maker import NPTMaker
from atomate2.openmm.jobs.nvt_maker import NVTMaker
from atomate2.openmm.flows.anneal_maker import AnnealMaker
from pymatgen.io.openmm.sets import OpenMMSet
from typing import Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from jobflow import Maker, Flow


@dataclass
class ProductionMaker(Maker):
    """
    Class for running
    """
    name: str = "production"
    energy_maker: EnergyMinimizationMaker = field(default_factory=EnergyMinimizationMaker)
    npt_maker: NPTMaker = field(default_factory=NPTMaker)
    anneal_maker: AnnealMaker = field(default_factory=AnnealMaker)
    nvt_maker: NVTMaker = field(default_factory=NVTMaker)

    def make(self, input_set: OpenMMSet, output_dir: Optional[Union[str, Path]] = None):
        """

        Parameters
        ----------
        input_set : OpenMMSet
            OpenMMSet object instance.
        output_dir : Optional[Union[str, Path]]
            File path to directory for writing simulation output files (e.g. trajectory and state files)

        Returns
        -------

        """
        if output_dir is None:
            # TODO: will temp_dir close properly? When will it close?
            temp_dir = TemporaryDirectory()
            output_dir = temp_dir.name
            output_dir = Path(output_dir)
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        energy_job = self.energy_maker.make(
            input_set=input_set,
            output_dir=output_dir / f"0_{self.energy_maker.name.replace(' ', '_')}"
        )

        pressure_job = self.npt_maker.make(
            input_set=energy_job.output.calculation_output.input_set,
            output_dir=output_dir / f"1_{self.npt_maker.name.replace(' ', '_')}"
        )

        anneal_job = self.anneal_maker.make(
            input_set=pressure_job.output.calculation_output.input_set,
            output_dir=output_dir / f"2_{self.anneal_maker.name.replace(' ', '_')}"
        )

        nvt_job = self.nvt_maker.make(
            input_set=anneal_job.output.calculation_output.input_set,
            output_dir=output_dir / f"3_{self.nvt_maker.name.replace(' ', '_')}"
        )

        my_flow = Flow(
            [
                energy_job,
                pressure_job,
                anneal_job,
                nvt_job,
            ],
            output={"log": nvt_job},
        )

        return my_flow
