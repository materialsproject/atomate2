from dataclasses import dataclass, field

from jobflow import Flow, Maker
from pymatgen.core import Structure

from atomate2.lammps.jobs.base import BaseLammpsMaker
from atomate2.lammps.jobs.core import LammpsNVTMaker, LammpsNPTMaker

from copy import deepcopy

import warnings


@dataclass
class MeltQuenchThermalizeMaker(Maker):
    # potentially remove and replace with single job
    name: str = "melt-quench-thermalize"
    melt_maker: BaseLammpsMaker = field(default_factory=LammpsNPTMaker)
    quench_maker: BaseLammpsMaker = field(default_factory=LammpsNPTMaker)
    thermalize_maker: BaseLammpsMaker = field(default_factory=LammpsNVTMaker)

    def make(self, structure: Structure):
        melt = self.melt_maker.make(structure)
        quench = self.quench_maker.make(melt.output.structure)
        thermalize = self.thermalize_maker.make(quench.output.structure)
        return Flow([melt, quench, thermalize], name=self.name)
    
    @classmethod
    def from_temperature_steps(
        cls,
        start_temperature: float = 300,
        melt_temperature: float = 3000,
        quench_temperature: float = 300,
        n_steps_melt: int = 10000,
        n_steps_quench: int = 10000,
        n_steps_thermalize: int = 10000,
        npt_maker : LammpsNPTMaker = None,
        nvt_maker : LammpsNVTMaker = None
    ) -> "MeltQuenchThermalizeMaker":
        if nvt_maker is None:
            warnings.warn("No NVT maker provided, using NPT maker for thermalize.")
        if npt_maker is None:
            raise ValueError("NPT maker must be provided.")
        
        melt_maker = deepcopy(npt_maker)
        melt_maker.name = "melt"
        melt_maker.input_set_generator.update_settings({"temperature" : [start_temperature, melt_temperature],
                                                                "nsteps" : n_steps_melt})

        quench_maker = deepcopy(npt_maker)
        quench_maker.name = "quench"
        quench_maker.input_set_generator.update_settings({"temperature" : [melt_temperature, quench_temperature],
                                                                "nsteps" : n_steps_quench})

        thermalize_maker = deepcopy(nvt_maker) if nvt_maker else deepcopy(npt_maker)
        thermalize_maker.name = "thermalize"
        thermalize_maker.input_set_generator.update_settings({"nsteps" : n_steps_thermalize,
                                                                    "temperature" : [quench_temperature, quench_temperature]})
        return cls(
            melt_maker=melt_maker,
            quench_maker=quench_maker,
            thermalize_maker=thermalize_maker,
        )