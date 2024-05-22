"""Jobs that compose MPMorph flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.common.flows.mpmorph import FastQuenchMaker, SlowQuenchMaker
from atomate2.vasp.jobs.md import MDMaker
from atomate2.vasp.jobs.mp import (
    MPMetaGGARelaxMaker,
    MPMetaGGAStaticMaker,
    MPPreRelaxMaker,
)
from atomate2.vasp.powerups import update_user_incar_settings
from atomate2.vasp.sets.mpmorph import MPMorphMDSetGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Flow, Job
    from pymatgen.core import Structure

    from atomate2.vasp.jobs.base import BaseVaspMaker
    from atomate2.vasp.sets.base import VaspInputGenerator


@dataclass
class BaseMPMorphMDMaker(MDMaker):
    """
    Maker to create VASP molecular dynamics jobs for amorphous materials.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputSetGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "MPMorph MD Maker"
    input_set_generator: VaspInputGenerator = field(
        default_factory=MPMorphMDSetGenerator
    )


@dataclass
class SlowQuenchVaspMaker(SlowQuenchMaker):
    """Slow quench from high to low temperature structures using VASP MDMaker.

    Quenches a provided structure with a molecular dynamics run from a
    desired high temperature to a desired low temperature.
    Flow creates a series of MD runs that holds at a certain temperature
    and initiates the following MD run at a lower temperature (step-wise
    temperature MD runs).

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    md_maker :  BaseMPMorphMDMaker
        MDMaker to generate the molecular dynamics jobs specifically for MPMorph AIMD;
        inherits  from MDMaker (VASP)
    quench_start_temperature : float = 3000
        Starting temperature for quench; default 3000K
    quench_end_temperature : float = 500
        Ending temperature for quench; default 500K
    quench_temperature_step : int = 500
        Temperature step for quench; default 500K drop
    quench_n_steps : int = 1000
        Number of steps for quench; default 1000 steps
    """

    name: str = "vasp slow quench"
    md_maker: BaseVaspMaker = field(default_factory=BaseMPMorphMDMaker)

    def call_md_maker(
        self,
        structure: Structure,
        temp: float | tuple[float, float],
        prev_dir: str | Path | None = None,
    ) -> Flow | Job:
        """Call the MD maker to create the MD jobs for VASP Only."""
        begining_temp, ending_temp = temp if isinstance(temp, tuple) else (temp, temp)

        self.md_maker = update_user_incar_settings(
            flow=self.md_maker,
            incar_updates={
                "TEBEG": begining_temp,
                "TEEND": ending_temp,
                "NSW": self.quench_n_steps,
            },
        )
        self.md_maker = self.md_maker.update_kwargs(
            update={"name": f"Vasp Slow Quench MD Maker {temp}K"}
        )
        return self.md_maker.make(structure=structure, prev_dir=prev_dir)


@dataclass
class FastQuenchVaspMaker(FastQuenchMaker):
    """Fast quench from high temperature to 0K structures with VASP.

    Quenches a provided structure with a single (or double) relaxation
    and a static calculation at 0K.
    NOTE: Same as MPMetaGGADoubleRelaxMaker.
    This is built for consistency with MPMorph flows.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker :  ForceFieldRelaxMaker
        Relax Maker
    relax_maker2 :  ForceFieldRelaxMaker
        Relax Maker for a second relaxation; useful for tighter convergence
    static_maker : ForceFieldStaticMaker
        Static Maker

    """

    name: str = "Vasp fast quench"
    relax_maker: MPPreRelaxMaker = field(default_factory=MPPreRelaxMaker)
    relax_maker2: MPMetaGGARelaxMaker = field(
        default_factory=lambda: MPMetaGGARelaxMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")}
        )
    )
    static_maker: MPMetaGGAStaticMaker = field(
        default_factory=lambda: MPMetaGGAStaticMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")}
        )
    )
