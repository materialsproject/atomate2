"""Core LAMMPS flows."""

from copy import deepcopy
from dataclasses import dataclass, field

from jobflow import Flow, Maker
from pymatgen.core import Structure

from atomate2.lammps.jobs.base import BaseLammpsMaker
from atomate2.lammps.jobs.core import LammpsNPTMaker, LammpsNVTMaker


@dataclass
class MeltQuenchThermalizeMaker(Maker):
    """Melt -> Quench -> Thermalize flow maker."""

    name: str = "melt-quench-thermalize"
    melt_maker: BaseLammpsMaker = field(default_factory=LammpsNPTMaker)
    quench_maker: BaseLammpsMaker = field(default_factory=LammpsNPTMaker)
    thermalize_maker: BaseLammpsMaker = field(default_factory=LammpsNVTMaker)

    def make(self, structure: Structure) -> Flow:
        """Make the flow for melting, quenching, and thermalizing a structure."""
        melt = self.melt_maker.make(structure)
        quench = self.quench_maker.make(
            melt.output.structure, prev_dir=melt.output.dir_name
        )
        thermalize = self.thermalize_maker.make(
            quench.output.structure, prev_dir=quench.output.dir_name
        )
        return Flow([melt, quench, thermalize], name=self.name)

    @classmethod
    def from_temperature_steps(
        cls,
        npt_maker: LammpsNPTMaker,
        nvt_maker: LammpsNVTMaker = None,
        start_temperature: float = 300,
        melt_temperature: float = 3000,
        quench_temperature: float = 300,
        n_steps_melt: int = 10000,
        n_steps_quench: int = 10000,
        n_steps_thermalize: int = 10000,
    ) -> "MeltQuenchThermalizeMaker":
        """Make a melt-quench-thermalize flow maker from temperature and steps."""
        melt_maker = deepcopy(npt_maker)
        melt_maker.name = "melt"
        melt_maker.input_set_generator.update_settings(
            {
                "start_temp": start_temperature,
                "end_temp": melt_temperature,
                "nsteps": n_steps_melt,
            }
        )

        quench_maker = deepcopy(npt_maker)
        quench_maker.name = "quench"
        quench_maker.input_set_generator.update_settings(
            {
                "start_temp": melt_temperature,
                "end_temp": quench_temperature,
                "nsteps": n_steps_quench,
            }
        )

        thermalize_maker = deepcopy(nvt_maker) if nvt_maker else deepcopy(npt_maker)
        thermalize_maker.name = "thermalize"
        thermalize_maker.input_set_generator.update_settings(
            {
                "start_temp": quench_temperature,
                "end_temp": quench_temperature,
                "nsteps": n_steps_thermalize,
            }
        )
        return cls(
            melt_maker=melt_maker,
            quench_maker=quench_maker,
            thermalize_maker=thermalize_maker,
        )
