"""Core jobs for running ABINIT calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from abipy.flowtk.events import (
    AbinitCriticalWarning,
    NscfConvergenceWarning,
    RelaxConvergenceWarning,
    ScfConvergenceWarning,
)

from atomate2.abinit.jobs.base import BaseAbinitMaker, abinit_job
from atomate2.abinit.sets.core import (
    LineNonSCFSetGenerator,
    NonSCFSetGenerator,
    NscfWfqSetGenerator,
    RelaxSetGenerator,
    StaticSetGenerator,
    UniformNonSCFSetGenerator,
)
from atomate2.abinit.utils.history import JobHistory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jobflow import Job
    from pymatgen.core.structure import Structure

    from atomate2.abinit.sets.base import AbinitInputGenerator
    from atomate2.abinit.utils.history import JobHistory

logger = logging.getLogger(__name__)

__all__ = [
    "LineNonSCFMaker",
    "NonSCFMaker",
    "RelaxMaker",
    "StaticMaker",
    "UniformNonSCFMaker",
    "WfqMaker",
]


@dataclass
class StaticMaker(BaseAbinitMaker):
    """
    Maker to create ABINIT self-consistent field (SCF) jobs.

    This maker generates static SCF calculations where the electronic
    structure is converged to self-consistency at fixed ionic positions.

    Parameters
    ----------
    calc_type : str
        The calculation type identifier. Default is "scf".
    name : str
        The job name. Default is "Scf calculation".
    input_set_generator : AbinitInputGenerator
        Generator for ABINIT input files. Defaults to StaticSetGenerator.
    """

    calc_type: str = "scf"
    name: str = "Scf calculation"
    input_set_generator: AbinitInputGenerator = field(
        default_factory=StaticSetGenerator
    )

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        ScfConvergenceWarning,
    )


@dataclass
class LineNonSCFMaker(BaseAbinitMaker):
    """
    Maker to create non-SCF calculations along high-symmetry k-point paths.

    This maker generates non-self-consistent field calculations along
    high-symmetry lines in the Brillouin zone, typically used for computing
    band structures. Requires a previous SCF calculation to provide the
    self-consistent density.

    Parameters
    ----------
    calc_type : str
        The calculation type identifier. Default is "nscf_line".
    name : str
        The job name. Default is "Line non-Scf calculation".
    input_set_generator : AbinitInputGenerator
        Generator for ABINIT input files. Defaults to LineNonSCFSetGenerator.
    """

    calc_type: str = "nscf_line"
    name: str = "Line non-Scf calculation"
    input_set_generator: AbinitInputGenerator = field(
        default_factory=LineNonSCFSetGenerator
    )

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        NscfConvergenceWarning,
    )


@dataclass
class UniformNonSCFMaker(BaseAbinitMaker):
    """
    Maker to create non-SCF calculations with uniform k-point meshes.

    This maker generates non-self-consistent field calculations with uniform
    k-point sampling, typically used for computing density of states.
    Requires a previous SCF calculation to provide the self-consistent density.

    Parameters
    ----------
    calc_type : str
        The calculation type identifier. Default is "nscf_uniform".
    name : str
        The job name. Default is "Uniform non-Scf calculation".
    input_set_generator : AbinitInputGenerator
        Generator for ABINIT input files. Defaults to UniformNonSCFSetGenerator.
    """

    calc_type: str = "nscf_uniform"
    name: str = "Uniform non-Scf calculation"
    input_set_generator: AbinitInputGenerator = field(
        default_factory=UniformNonSCFSetGenerator
    )

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        NscfConvergenceWarning,
    )


@dataclass
class NonSCFMaker(BaseAbinitMaker):
    """
    Maker to create non-SCF calculations with configurable k-point sampling.

    This is a flexible non-self-consistent field maker that can generate
    either line-mode or uniform k-point calculations depending on the mode
    parameter. It's useful when you need to switch between calculation types
    dynamically.

    Parameters
    ----------
    calc_type : str
        The calculation type identifier. Default is "nscf".
    name : str
        The job name. Default is "non-Scf calculation".
    input_set_generator : AbinitInputGenerator
        Generator for ABINIT input files. Defaults to NonSCFSetGenerator.
    """

    calc_type: str = "nscf"
    name: str = "non-Scf calculation"

    input_set_generator: AbinitInputGenerator = field(
        default_factory=NonSCFSetGenerator
    )

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        NscfConvergenceWarning,
    )

    @abinit_job
    def make(
        self,
        structure: Structure | None = None,
        prev_outputs: str | list[str] | None = None,
        restart_from: str | list[str] | None = None,
        history: JobHistory | None = None,
        mode: str = "uniform",
    ) -> Job:
        """
        Create a non-SCF ABINIT job with configurable k-point sampling.

        Parameters
        ----------
        structure : Structure or None
            A pymatgen Structure object. At least one of structure,
            prev_outputs, or restart_from must be provided.
        prev_outputs : str or list[str] or None
            Path(s) to previous SCF calculation directories to use the
            self-consistent density.
        restart_from : str or list[str] or None
            Path(s) to previous calculation directories to restart from.
        history : JobHistory or None
            Job history tracking previous runs and restarts.
        mode : str
            K-point sampling mode. Options are:
            - "line": K-points along high-symmetry lines (band structure).
            - "uniform": Uniform k-point mesh (density of states).
            Default is "uniform".

        Returns
        -------
        Job
            A jobflow Job for the non-SCF calculation.
        """
        self.input_set_generator.mode = mode

        return super().make.original(
            self,
            structure=structure,
            prev_outputs=prev_outputs,
            restart_from=restart_from,
            history=history,
        )


@dataclass
class WfqMaker(NonSCFMaker):
    """
    Maker for wavefunctions at q-points (WFQ) calculations.

    WFQ calculations compute wavefunctions on a k-point grid shifted by a
    q-point. These are required for phonon calculations when the q-point
    mesh is not commensurate with the ground-state k-point mesh.

    Parameters
    ----------
    calc_type : str
        The calculation type identifier. Default is "wfq".
    name : str
        The job name. Default is "WFQ nscf Calculation".
    input_set_generator : AbinitInputGenerator
        Generator for ABINIT input files. Defaults to NscfWfqSetGenerator.
    """

    calc_type: str = "wfq"
    name: str = "WFQ nscf Calculation"
    input_set_generator: AbinitInputGenerator = field(
        default_factory=NscfWfqSetGenerator
    )

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        NscfConvergenceWarning,
    )

    @abinit_job
    def make(
        self,
        structure: Structure | None = None,
        prev_outputs: str | list[str] | None = None,
        restart_from: str | list[str] | None = None,
        history: JobHistory | None = None,
        mode: str = "uniform",
        qpt: list | tuple | None = None,
    ) -> Job:
        """
        Create a WFQ ABINIT job.

        Parameters
        ----------
        structure : Structure or None
            A pymatgen Structure object. At least one of structure,
            prev_outputs, or restart_from must be provided.
        prev_outputs : str or list[str] or None
            Path(s) to previous SCF calculation directories.
        restart_from : str or list[str] or None
            Path(s) to previous calculation directories to restart from.
        history : JobHistory or None
            Job history tracking previous runs and restarts.
        mode : str
            K-point sampling mode. Default is "uniform".
        qpt : list or tuple or None
            The q-point used to shift the k-point grid (e.g., [0.25, 0, 0]).
            Must be provided for WFQ calculations.

        Returns
        -------
        Job
            A jobflow Job for the WFQ calculation.
        """
        self.input_set_generator.factory_kwargs.update({"qpt": qpt})

        return super().make.original(
            self,
            structure=structure,
            prev_outputs=prev_outputs,
            restart_from=restart_from,
            history=history,
            mode=mode,
        )


@dataclass
class RelaxMaker(BaseAbinitMaker):
    """
    Maker to create structural relaxation calculations.

    This maker performs geometry optimization where atomic positions and/or
    lattice parameters are relaxed to minimize forces and stresses.

    Parameters
    ----------
    calc_type : str
        The calculation type identifier. Default is "relax".
    input_set_generator : AbinitInputGenerator
        Generator for ABINIT input files. Defaults to RelaxSetGenerator.
    name : str
        The job name. Default is "Relaxation calculation".
    """

    calc_type: str = "relax"
    input_set_generator: AbinitInputGenerator = field(default_factory=RelaxSetGenerator)
    name: str = "Relaxation calculation"

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        RelaxConvergenceWarning,
    )

    @classmethod
    def ionic_relaxation(cls, *args, **kwargs) -> RelaxMaker:
        """
        Create a RelaxMaker for ionic-only relaxation.

        This classmethod creates a maker that relaxes only atomic positions
        while keeping the cell parameters fixed.

        Parameters
        ----------
        *args
            Positional arguments passed to RelaxSetGenerator.
        **kwargs
            Keyword arguments passed to RelaxSetGenerator. The relax_cell
            parameter will be set to False.

        Returns
        -------
        RelaxMaker
            A RelaxMaker configured for ionic-only relaxation.
        """
        return cls(
            input_set_generator=RelaxSetGenerator(*args, relax_cell=False, **kwargs),
            name="Relaxation calculation (ions only)",
        )

    @classmethod
    def full_relaxation(cls, *args, **kwargs) -> RelaxMaker:
        """
        Create a RelaxMaker for full structural relaxation.

        This classmethod creates a maker that relaxes both atomic positions
        and cell parameters (shape and volume).

        Parameters
        ----------
        *args
            Positional arguments passed to RelaxSetGenerator.
        **kwargs
            Keyword arguments passed to RelaxSetGenerator. The relax_cell
            parameter will be set to True.

        Returns
        -------
        RelaxMaker
            A RelaxMaker configured for full structural relaxation.
        """
        return cls(
            input_set_generator=RelaxSetGenerator(*args, relax_cell=True, **kwargs)
        )
