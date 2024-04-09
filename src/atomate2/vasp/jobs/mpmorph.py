""" Jobs that compose MPMorph flows. """

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.vasp.jobs.md import MDMaker
from atomate2.vasp.sets.mpmorph import MPMorphMDSetGenerator

from atomate2.vasp.jobs.mp import (
    MPMetaGGARelaxMaker,
    MPMetaGGAStaticMaker,
    MPPreRelaxMaker,
)

from atomate2.common.flows.mpmorph import SlowQuenchMaker
from atomate2.vasp.powerups import update_user_incar_settings

if TYPE_CHECKING:
    from atomate2.vasp.sets.base import VaspInputGenerator
    from atomate2.vasp.jobs.base import BaseVaspMaker


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
    """Slow quench flow for quenching high temperature structures to low temperature using VASP MDMaker.

    Quench's a provided structure with a molecular dyanmics run from a desired high temperature to
    a desired low temperature. Flow creates a series of MD runs that holds at a certain temperature
    and initiates the following MD run at a lower temperature (step-wise temperature MD runs).
    Adapted from MPMorph Workflow.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    md_maker :  BaseMPMorphMDMaker
        MDMaker to generate the molecular dynamics jobs specifically for MPMorph AIMD; inherits  from MDMaker (VASP)
    quench_start_temperature : int = 3000
        Starting temperature for quench; default 3000K
    quench_end_temperature : int = 500
        Ending temperature for quench; default 500K
    quench_temperature_step : int = 500
        Temperature step for quench; default 500K drop
    quench_nsteps : int = 1000
        Number of steps for quench; default 1000 steps
    """

    name: str = "vasp slow quench"
    md_maker: BaseVaspMaker = field(default_factory=BaseMPMorphMDMaker)

    def call_md_maker(self, structure, prev_dir, temp, nsteps):
        """Call the MD maker to create the MD jobs for VASP Only."""
        self.md_maker = update_user_incar_settings(
            flow=self.md_maker,
            incar_updates={
                "TEBEG": temp,
                "TEEND": temp,  # Equilibrium volume search is only at single temperature (temperature sweep not allowed)
                "NSW": nsteps,
            },
        )
        return self.md_maker.make(
            structure=structure, prev_dir=prev_dir, temperature=temp, nsteps=nsteps
        )


@dataclass
class FastQuenchVaspMDMaker(SlowQuenchMaker):
    """Fast quench flow for quenching high temperature structures to 0K with MLFF.

    Quench's a provided structure with a single (or double) relaxation and a static calculation at 0K.
    Adapted from MPMorph Workflow. NOTE: Same as MPMetaGGADoubleRelaxMaker. This is built for consistency
    with MPMorph flows.

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
        default_factory=MPMetaGGARelaxMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")}
        )
    )
    static_maker: MPMetaGGAStaticMaker = field(
        default_factory=MPMetaGGAStaticMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")}
        )
    )
