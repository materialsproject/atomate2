"""Flows adapted from MPMorph *link to origin github repo*"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Response

from atomate2.common.flows.mpmorph import (
    FastQuenchMaker,
    SlowQuenchMaker,
    MPMorphMDMaker,
    EquilibriumVolumeMaker,
)

from atomate2.forcefields.md import ForceFieldMDMaker
from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker

from atomate2.common.jobs.eos import MPMorphEVPostProcess, MPMorphPVPostProcess

from atomate2.forcefields.jobs import (
    ForceFieldRelaxMaker,
    ForceFieldStaticMaker,
    CHGNetRelaxMaker,
    CHGNetStaticMaker,
    M3GNetRelaxMaker,
    M3GNetStaticMaker,
    MACERelaxMaker,
    MACEStaticMaker,
    LJRelaxMaker,
    LJStaticMaker,
)
from atomate2.forcefields.md import (
    ForceFieldMDMaker,
    CHGNetMDMaker,
    M3GNetMDMaker,
    MACEMDMaker,
    LJMDMaker,
)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pathlib import Path
    from jobflow import Flow, Maker


@dataclass
class MPMorphMLFFMDMaker(MPMorphMDMaker):
    """ML ForceField MPMorph flow for volume equilibration, quench, and production runs via molecular dynamics

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    (optional) for the volume followed by quench (optional) from high temperature to low temperature
    and finally a production run(s) at a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : int = 300
        Temperature of the equilibrium volume search and production run in Kelvin, default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : ForceFieldMDMaker
        MDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    quench_tempature_setup: dict = None
        Only needed for SlowQuenchMaker. Setup for slow quenching the structure from high temperature to low temperature
        Example:
        quench_tempature_setup ={
            "start_temp": 3000, # Starting temperature for quench
            "end_temp": 500, # Ending temperature for quench
            "temp_step": 500, # Temperature step for quench
            "nsteps": 1000, # Number of steps for quench
        }
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher; inherits from EquilibriumVolumeMaker and ForceFieldMDMaker (MLFF)
    production_md_maker : ForceFieldMDMaker
        MDMaker to generate the production run(s); inherits from ForceFieldMDMaker (MLFF)
    relax_maker: ForceFieldRelaxMaker = None
        Used for FastQuench only. Check out atomate2.forcefields.jobs and all available ForceFieldRelaxMaker.
    static_maker: ForceFieldStaticMaker = None
        Used for FastQuench only. Check out atomate2.forcefields.jobs and all available ForceFieldStaticMaker.
    quench_maker :  SlowQuenchMaker or FastQuenchMaker or None
        SlowQuenchMaker - MDMaker that quenchs structure from high temperature to low temperature
        FastQuenchMaker - DoubleRelaxMaker + Static that "quenchs" structure at 0K
    """

    name: str = "MP Morph MLFF MD Maker"
    temperature: int = 300
    end_temp: int = 300  # DEPRECATED: Unusable for MLFF
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    quench_tempature_setup: dict | None = None

    md_maker: ForceFieldMDMaker | None = field(default_factory=ForceFieldMDMaker)
    convergence_md_maker: EquilibriumVolumeMaker | None = None
    production_md_maker: ForceFieldMDMaker = field(default_factory=ForceFieldMDMaker)

    relax_maker: ForceFieldRelaxMaker | None = None
    static_maker: ForceFieldStaticMaker | None = None
    quench_maker: FastQuenchMaker | SlowQuenchMaker | None = None

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        self.md_maker = self.md_maker.update_kwargs(
            update={
                "name": "MLFF MD Maker",
                "temperature": self.temperature,
                "nsteps": self.steps_convergence,
            },
            class_filter=ForceFieldMDMaker,
        )
        self.convergence_md_maker = EquilibriumVolumeMaker(
            name="MP Morph MLFF Equilibrium Volume Maker",
            md_maker=self.md_maker,
            postprocessor=MPMorphEVPostProcess(),
        )  # TODO: check EV versus PV

        self.production_md_maker = self.md_maker.update_kwargs(
            update=dict(
                name="Production Run MLFF MD Maker",
                temperature=self.temperature,
                nsteps=self.steps_total_production,
            )
        )
        print(type(self.quench_maker))
        if self.quench_maker is not None:
            if isinstance(self.quench_maker, SlowQuenchMaker):
                self.quench_maker = SlowQuenchMaker(
                    md_maker=self.md_maker,
                    quench_tempature_setup=self.quench_tempature_setup,
                )
            elif isinstance(self.quench_maker, FastQuenchMaker):
                self.quench_maker = FastQuenchMaker(
                    relax_maker=self.relax_maker,
                    static_maker=self.static_maker,
                )
        return super().make(structure=structure, prev_dir=prev_dir)


@dataclass
class MPMorphLJMDMaker(MPMorphMDMaker):
    """Lennard Jones ForceField MPMorph flow for volume equilibration, quench, and production runs via molecular dynamics

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    (optional) for the volume followed by quench (optional) from high temperature to low temperature
    and finally a production run(s) at a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : int = 300
        Temperature of the equilibrium volume search and production run in Kelvin, default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : LJMDMaker
        LJMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    quench_tempature_setup: dict = None
        Only needed for SlowQuenchMaker. Setup for slow quenching the structure from high temperature to low temperature
        Example:
        quench_tempature_setup ={
            "start_temp": 3000, # Starting temperature for quench
            "end_temp": 500, # Ending temperature for quench
            "temp_step": 500, # Temperature step for quench
            "nsteps": 1000, # Number of steps for quench
        }
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher; inherits from EquilibriumVolumeMaker and ForceFieldMDMaker
    production_md_maker : LJMDMaker
        LJMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    relax_maker: LJRelaxMaker = None
        Used for FastQuench only.
    static_maker: LJStaticMaker = None
        Used for FastQuench only.
    quench_maker :  SlowQuenchMaker or FastQuenchMaker or None
        SlowQuenchMaker - MDMaker that quenchs structure from high temperature to low temperature
        FastQuenchMaker - DoubleRelaxMaker + Static that "quenchs" structure at 0K
    """

    name: str = "MP Morph LJ MD Maker"
    temperature: int = 300
    end_temp: int = 300  # DEPRECATED: Unusable for MLFF
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    quench_tempature_setup: dict | None = None

    md_maker: LJMDMaker | None = field(default_factory=LJMDMaker)
    convergence_md_maker: EquilibriumVolumeMaker | None = None
    production_md_maker: LJMDMaker = field(default_factory=LJMDMaker)

    relax_maker: LJRelaxMaker | None = None
    static_maker: LJStaticMaker | None = None
    quench_maker: FastQuenchMaker | SlowQuenchMaker | None = None

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        self.md_maker = self.md_maker.update_kwargs(
            update={
                "name": "LJ MD Maker",
                "temperature": self.temperature,
                "nsteps": self.steps_convergence,
            },
            class_filter=ForceFieldMDMaker,
        )
        self.convergence_md_maker = EquilibriumVolumeMaker(
            name="MP Morph LJ Equilibrium Volume Maker",
            md_maker=self.md_maker,
            postprocessor=MPMorphEVPostProcess(),
        )  # TODO: check EV versus PV

        self.production_md_maker = self.md_maker.update_kwargs(
            update=dict(
                name="Production Run LJ MD Maker",
                temperature=self.temperature,
                nsteps=self.steps_total_production,
            )
        )
        print(type(self.quench_maker))
        if self.quench_maker is not None:
            if isinstance(self.quench_maker, SlowQuenchMaker):
                self.quench_maker = SlowQuenchMaker(
                    md_maker=self.md_maker,
                    quench_tempature_setup=self.quench_tempature_setup,
                )
            elif isinstance(self.quench_maker, FastQuenchMaker):
                self.quench_maker = FastQuenchMaker(
                    relax_maker=self.relax_maker,
                    static_maker=self.static_maker,
                )
        return super().make(structure=structure, prev_dir=prev_dir)


@dataclass
class MPMorphCHGNetMDMaker(MPMorphMDMaker):
    """CHGNet ForceField MPMorph flow for volume equilibration, quench, and production runs via molecular dynamics

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    (optional) for the volume followed by quench (optional) from high temperature to low temperature
    and finally a production run(s) at a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : int = 300
        Temperature of the equilibrium volume search and production run in Kelvin, default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : CHGNetMDMaker
        CHGNetMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    quench_tempature_setup: dict = None
        Only needed for SlowQuenchMaker. Setup for slow quenching the structure from high temperature to low temperature
        Example:
        quench_tempature_setup ={
            "start_temp": 3000, # Starting temperature for quench
            "end_temp": 500, # Ending temperature for quench
            "temp_step": 500, # Temperature step for quench
            "nsteps": 1000, # Number of steps for quench
        }
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher; inherits from EquilibriumVolumeMaker and ForceFieldMDMaker
    production_md_maker : CHGNetMDMaker
        CHGNetMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    relax_maker: CHGNetRelaxMaker = None
        Used for FastQuench only.
    static_maker: CHGNetStaticMaker = None
        Used for FastQuench only.
    quench_maker :  SlowQuenchMaker or FastQuenchMaker or None
        SlowQuenchMaker - MDMaker that quenchs structure from high temperature to low temperature
        FastQuenchMaker - DoubleRelaxMaker + Static that "quenchs" structure at 0K
    """

    name: str = "MP Morph CHGNet MD Maker"
    temperature: int = 300
    end_temp: int = 300  # DEPRECATED: Unusable for MLFF
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    quench_tempature_setup: dict | None = None

    md_maker: CHGNetMDMaker | None = field(default_factory=CHGNetMDMaker)
    convergence_md_maker: EquilibriumVolumeMaker | None = None
    production_md_maker: CHGNetMDMaker = field(default_factory=CHGNetMDMaker)

    relax_maker: CHGNetRelaxMaker | None = None
    static_maker: CHGNetStaticMaker | None = None
    quench_maker: FastQuenchMaker | SlowQuenchMaker | None = None

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        self.md_maker = self.md_maker.update_kwargs(
            update={
                "name": "CHGNet MD Maker",
                "temperature": self.temperature,
                "nsteps": self.steps_convergence,
            },
            class_filter=ForceFieldMDMaker,
        )
        self.convergence_md_maker = EquilibriumVolumeMaker(
            name="MP Morph CHGNet Equilibrium Volume Maker",
            md_maker=self.md_maker,
            postprocessor=MPMorphEVPostProcess(),
        )  # TODO: check EV versus PV

        self.production_md_maker = self.md_maker.update_kwargs(
            update=dict(
                name="Production Run CHGNet MD Maker",
                temperature=self.temperature,
                nsteps=self.steps_total_production,
            )
        )
        print(type(self.quench_maker))
        if self.quench_maker is not None:
            if isinstance(self.quench_maker, SlowQuenchMaker):
                self.quench_maker = SlowQuenchMaker(
                    md_maker=self.md_maker,
                    quench_tempature_setup=self.quench_tempature_setup,
                )
            elif isinstance(self.quench_maker, FastQuenchMaker):
                self.quench_maker = FastQuenchMaker(
                    relax_maker=self.relax_maker,
                    static_maker=self.static_maker,
                )
        return super().make(structure=structure, prev_dir=prev_dir)


@dataclass
class MPMorphM3GNetMDMaker(MPMorphMDMaker):
    """M3GNet ForceField MPMorph flow for volume equilibration, quench, and production runs via molecular dynamics

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    (optional) for the volume followed by quench (optional) from high temperature to low temperature
    and finally a production run(s) at a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : int = 300
        Temperature of the equilibrium volume search and production run in Kelvin, default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : M3GNetMDMaker
        M3GNetMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    quench_tempature_setup: dict = None
        Only needed for SlowQuenchMaker. Setup for slow quenching the structure from high temperature to low temperature
        Example:
        quench_tempature_setup ={
            "start_temp": 3000, # Starting temperature for quench
            "end_temp": 500, # Ending temperature for quench
            "temp_step": 500, # Temperature step for quench
            "nsteps": 1000, # Number of steps for quench
        }
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher; inherits from EquilibriumVolumeMaker and ForceFieldMDMaker
    production_md_maker : M3GNetMDMaker
        M3GNetMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    relax_maker: M3GNetRelaxMaker = None
        Used for FastQuench only.
    static_maker: M3GNetStaticMaker = None
        Used for FastQuench only.
    quench_maker :  SlowQuenchMaker or FastQuenchMaker or None
        SlowQuenchMaker - MDMaker that quenchs structure from high temperature to low temperature
        FastQuenchMaker - DoubleRelaxMaker + Static that "quenchs" structure at 0K
    """

    name: str = "MP Morph M3GNet MD Maker"
    temperature: int = 300
    end_temp: int = 300  # DEPRECATED: Unusable for MLFF
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    quench_tempature_setup: dict | None = None

    md_maker: M3GNetMDMaker | None = field(default_factory=M3GNetMDMaker)
    convergence_md_maker: EquilibriumVolumeMaker | None = None
    production_md_maker: M3GNetMDMaker = field(default_factory=M3GNetMDMaker)

    relax_maker: M3GNetRelaxMaker | None = None
    static_maker: M3GNetStaticMaker | None = None
    quench_maker: FastQuenchMaker | SlowQuenchMaker | None = None

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        self.md_maker = self.md_maker.update_kwargs(
            update={
                "name": "M3GNet MD Maker",
                "temperature": self.temperature,
                "nsteps": self.steps_convergence,
            },
            class_filter=ForceFieldMDMaker,
        )
        self.convergence_md_maker = EquilibriumVolumeMaker(
            name="MP Morph M3GNet Equilibrium Volume Maker",
            md_maker=self.md_maker,
            postprocessor=MPMorphEVPostProcess(),
        )  # TODO: check EV versus PV

        self.production_md_maker = self.md_maker.update_kwargs(
            update=dict(
                name="Production Run M3GNet MD Maker",
                temperature=self.temperature,
                nsteps=self.steps_total_production,
            )
        )
        print(type(self.quench_maker))
        if self.quench_maker is not None:
            if isinstance(self.quench_maker, SlowQuenchMaker):
                self.quench_maker = SlowQuenchMaker(
                    md_maker=self.md_maker,
                    quench_tempature_setup=self.quench_tempature_setup,
                )
            elif isinstance(self.quench_maker, FastQuenchMaker):
                self.quench_maker = FastQuenchMaker(
                    relax_maker=self.relax_maker,
                    static_maker=self.static_maker,
                )
        return super().make(structure=structure, prev_dir=prev_dir)


@dataclass
class MPMorphMACEMDMaker(MPMorphMDMaker):
    """MACE ForceField MPMorph flow for volume equilibration, quench, and production runs via molecular dynamics

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    (optional) for the volume followed by quench (optional) from high temperature to low temperature
    and finally a production run(s) at a given temperature.

    Check atomate2.common.flows.mpmorph for MPMorphMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperature : int = 300
        Temperature of the equilibrium volume search and production run in Kelvin, default 300K
    steps_convergence: int | None = None
        Defaults to 5000 steps unless specified
    md_maker : MACEMDMaker
        MACEMDMaker to generate the molecular dynamics jobs specifically for MLFF MDs
    steps_total_production: int = 10000
        Total number of steps for the production run(s); default 10000 steps
    quench_tempature_setup: dict = None
        Only needed for SlowQuenchMaker. Setup for slow quenching the structure from high temperature to low temperature
        Example:
        quench_tempature_setup ={
            "start_temp": 3000, # Starting temperature for quench
            "end_temp": 500, # Ending temperature for quench
            "temp_step": 500, # Temperature step for quench
            "nsteps": 1000, # Number of steps for quench
        }
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher; inherits from EquilibriumVolumeMaker and ForceFieldMDMaker
    production_md_maker : MACEMDMaker
        MACEMDMaker to generate the production run(s); inherits from ForceFieldMDMaker
    relax_maker: MACERelaxMaker = None
        Used for FastQuench only.
    static_maker: MACEStaticMaker = None
        Used for FastQuench only.
    quench_maker :  SlowQuenchMaker or FastQuenchMaker or None
        SlowQuenchMaker - MDMaker that quenchs structure from high temperature to low temperature
        FastQuenchMaker - DoubleRelaxMaker + Static that "quenchs" structure at 0K
    """

    name: str = "MP Morph MACE MD Maker"
    temperature: int = 300
    end_temp: int = 300  # DEPRECATED: Unusable for MLFF
    steps_convergence: int = 5000
    steps_total_production: int = 10000

    quench_tempature_setup: dict | None = None

    md_maker: MACEMDMaker | None = field(default_factory=MACEMDMaker)
    convergence_md_maker: EquilibriumVolumeMaker | None = None
    production_md_maker: MACEMDMaker = field(default_factory=MACEMDMaker)

    relax_maker: MACERelaxMaker | None = None
    static_maker: MACEStaticMaker | None = None
    quench_maker: FastQuenchMaker | SlowQuenchMaker | None = None

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        self.md_maker = self.md_maker.update_kwargs(
            update={
                "name": "MACE MD Maker",
                "temperature": self.temperature,
                "nsteps": self.steps_convergence,
            },
            class_filter=ForceFieldMDMaker,
        )
        self.convergence_md_maker = EquilibriumVolumeMaker(
            name="MP Morph MACE Equilibrium Volume Maker",
            md_maker=self.md_maker,
            postprocessor=MPMorphEVPostProcess(),
        )  # TODO: check EV versus PV

        self.production_md_maker = self.md_maker.update_kwargs(
            update=dict(
                name="Production Run MACE MD Maker",
                temperature=self.temperature,
                nsteps=self.steps_total_production,
            )
        )
        print(type(self.quench_maker))
        if self.quench_maker is not None:
            if isinstance(self.quench_maker, SlowQuenchMaker):
                self.quench_maker = SlowQuenchMaker(
                    md_maker=self.md_maker,
                    quench_tempature_setup=self.quench_tempature_setup,
                )
            elif isinstance(self.quench_maker, FastQuenchMaker):
                self.quench_maker = FastQuenchMaker(
                    relax_maker=self.relax_maker,
                    static_maker=self.static_maker,
                )
        return super().make(structure=structure, prev_dir=prev_dir)
