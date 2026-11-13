"""Module defining core ABINIT input set generators."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from abipy.abio.factories import (
    dos_from_gsinput,
    ebands_from_gsinput,
    ion_ioncell_relax_input,
    nscf_from_gsinput,
    scf_for_phonons,
    scf_input,
    wfq_nscf_from_gsinput,
)
from abipy.abio.input_tags import MOLECULAR_DYNAMICS, NSCF, RELAX, SCF
from pymatgen.analysis.structure_matcher import StructureMatcher

from atomate2.abinit.sets.base import AbinitInputGenerator
from atomate2.abinit.utils.common import get_final_structure
from atomate2.utils.path import strip_hostname

if TYPE_CHECKING:
    from collections.abc import Callable

    from abipy.abio.inputs import AbinitInput
    from pymatgen.core import Structure
    from pymatgen.io.abinit import PseudoTable
    from pymatgen.io.abinit.abiobjects import KSampling


logger = logging.getLogger(__name__)

__all__ = [
    "GS_RESTART_FROM_DEPS",
    "LineNonSCFSetGenerator",
    "NonSCFSetGenerator",
    "NscfWfqSetGenerator",
    "PhononsStaticSetGenerator",
    "RelaxSetGenerator",
    "ShgStaticSetGenerator",
    "StaticSetGenerator",
    "UniformNonSCFSetGenerator",
]

GS_RESTART_FROM_DEPS = (f"{SCF}|{RELAX}|{MOLECULAR_DYNAMICS}:WFK|DEN",)


@dataclass
class StaticSetGenerator(AbinitInputGenerator):
    """
    Generator for static self-consistent field (SCF) calculations.

    This class generates input sets for ground-state ABINIT calculations
    without ionic or cell relaxation.

    Attributes
    ----------
    calc_type : str
        Type of calculation. Default is "static".
    factory : Callable
        Factory function for generating the input. Default is scf_input.
    restart_from_deps : tuple
        Dependencies for restarting calculations. Default is GS_RESTART_FROM_DEPS.
    """

    calc_type: str = "static"
    factory: Callable = scf_input
    restart_from_deps: tuple = GS_RESTART_FROM_DEPS

    def get_abinit_input(
        self,
        structure: Structure | None = None,
        pseudos: PseudoTable | None = None,
        prev_outputs: list[str] | None = None,
        abinit_settings: dict | None = None,
        factory_kwargs: dict | None = None,
        kpoints_settings: dict | KSampling | None = None,
        input_index: int | None = None,
    ) -> AbinitInput:
        """
        Generate an AbinitInput for a static SCF calculation.

        This method disables relaxation-related variables to ensure a
        static calculation, even when restarting from a relaxation run.

        Parameters
        ----------
        structure : Structure or None
            Pymatgen Structure object. Default is None.
        pseudos : PseudoTable or None
            Pseudopotential table. Default is None.
        prev_outputs : list[str] or None
            List of previous output directories. Default is None.
        abinit_settings : dict or None
            Additional ABINIT keywords to set. Default is None.
        factory_kwargs : dict or None
            Additional factory keywords. Default is None.
        kpoints_settings : dict or KSampling or None
            K-points settings. Default is None.
        input_index : int or None
            Index for MultiDataset selection. Default is None.

        Returns
        -------
        AbinitInput
            An AbinitInput object for a static SCF calculation.
        """
        # Disable relaxation options in case they are present (from a restart)
        scf_abinit_settings = {
            "ionmov": None,
            "optcell": None,
            "ntime": None,
        }
        if abinit_settings:
            scf_abinit_settings.update(abinit_settings)

        return super().get_abinit_input(
            structure=structure,
            pseudos=pseudos,
            prev_outputs=prev_outputs,
            abinit_settings=scf_abinit_settings,
            factory_kwargs=factory_kwargs,
            kpoints_settings=kpoints_settings,
        )


@dataclass
class ShgStaticSetGenerator(StaticSetGenerator):
    """
    Generator for static SCF input sets optimized for DFPT SHG calculations.

    This class generates static SCF input sets specifically adapted for
    second harmonic generation (SHG) calculations using density functional
    perturbation theory (DFPT).

    Attributes
    ----------
    factory : Callable
        Factory function for generating the input. Default is scf_for_phonons.
    factory_kwargs : dict
        Default factory arguments including no smearing, unpolarized spin mode,
        and kppa=3000.
    user_abinit_settings : dict
        Default ABINIT settings including nstep=500, toldfe=1e-22, autoparal=1,
        and npfft=1.
    """

    factory: Callable = scf_for_phonons
    factory_kwargs: dict = field(
        default_factory=lambda: {
            "smearing": "nosmearing",
            "spin_mode": "unpolarized",
            "kppa": 3000,
        }
    )

    user_abinit_settings: dict = field(
        default_factory=lambda: {
            "nstep": 500,
            "toldfe": 1e-22,
            "autoparal": 1,
            "npfft": 1,
        }
    )


@dataclass
class PhononsStaticSetGenerator(StaticSetGenerator):
    """
    Generator for static SCF input sets optimized for DFPT phonon calculations.

    This class generates static SCF input sets specifically adapted for
    phonon calculations using density functional perturbation theory (DFPT).

    Attributes
    ----------
    factory : Callable
        Factory function for generating the input. Default is scf_for_phonons.
    factory_kwargs : dict
        Default factory arguments including kppa=3000.
    user_abinit_settings : dict
        Default ABINIT settings including nstep=500.
    """

    factory: Callable = scf_for_phonons
    factory_kwargs: dict = field(
        default_factory=lambda: {
            "kppa": 3000,
        }
    )

    user_abinit_settings: dict = field(
        default_factory=lambda: {
            "nstep": 500,
        }
    )


@dataclass
class NonSCFSetGenerator(AbinitInputGenerator):
    """
    Generator for ABINIT non-self-consistent field (non-SCF) input sets.

    This class generates input sets for non-SCF calculations that require
    a previous SCF calculation for the density.

    Attributes
    ----------
    calc_type : str
        Type of calculation. Default is "nscf".
    factory : Callable
        Factory function for generating the input. Default is nscf_from_gsinput.
    pseudos : str or list[str] or PseudoTable or None
        Pseudopotentials specification. Default is None.
    restart_from_deps : tuple
        Dependencies for restarting calculations. Default is (f"{NSCF}:WFK",).
    prev_outputs_deps : tuple
        Dependencies from previous calculations. Default is (f"{SCF}:DEN",).
    nbands_factor : float
        Factor to multiply the number of bands from the previous calculation.
        Default is 1.2.
    factory_prev_inputs_kwargs : dict or None
        Mapping of factory arguments to previous calculation types.
        Default is {"gs_input": (SCF,)}.
    """

    calc_type: str = "nscf"
    factory: Callable = nscf_from_gsinput
    pseudos: str | list[str] | PseudoTable | None = None
    restart_from_deps: tuple = (f"{NSCF}:WFK",)
    prev_outputs_deps: tuple = (f"{SCF}:DEN",)
    nbands_factor: float = 1.2

    factory_prev_inputs_kwargs: dict | None = field(
        default_factory=lambda: {"gs_input": (SCF,)}
    )

    def get_abinit_input(
        self,
        structure: Structure | None = None,
        pseudos: PseudoTable | None = None,
        prev_outputs: list[str] | None = None,
        abinit_settings: dict | None = None,
        factory_kwargs: dict | None = None,
        kpoints_settings: dict | KSampling | None = None,
        input_index: int | None = None,
    ) -> AbinitInput:
        """
        Generate an AbinitInput for a non-SCF calculation.

        Automatically determines the number of bands based on the previous
        SCF calculation and the nbands_factor.

        Parameters
        ----------
        structure : Structure or None
            Pymatgen Structure object. Default is None.
        pseudos : PseudoTable or None
            Pseudopotential table. Default is None.
        prev_outputs : list[str] or None
            List of previous output directories. Default is None.
        abinit_settings : dict or None
            Additional ABINIT keywords to set. Default is None.
        factory_kwargs : dict or None
            Additional factory keywords. Default is None.
        kpoints_settings : dict or KSampling or None
            K-points settings. Default is None.
        input_index : int or None
            Index for MultiDataset selection. Default is None.

        Returns
        -------
        AbinitInput
            An AbinitInput object for a non-SCF calculation.
        """
        factory_kwargs = dict(factory_kwargs) if factory_kwargs else {}
        factory_kwargs["nband"] = self._get_nband(prev_outputs)

        return super().get_abinit_input(
            structure=structure,
            pseudos=pseudos,
            prev_outputs=prev_outputs,
            abinit_settings=abinit_settings,
            factory_kwargs=factory_kwargs,
            kpoints_settings=kpoints_settings,
        )

    def _get_nband(self, prev_outputs: list[str] | None) -> int:
        """
        Calculate the number of bands for the non-SCF calculation.

        Retrieves the number of bands from the previous SCF calculation and
        multiplies by nbands_factor.

        Parameters
        ----------
        prev_outputs : list[str] or None
            List of previous output directories.

        Returns
        -------
        int
            Number of bands to use in the non-SCF calculation.

        Raises
        ------
        RuntimeError
            If the number of previous outputs is not exactly one.
        """
        abinit_inputs = self.resolve_prev_inputs(
            prev_outputs, self.factory_prev_inputs_kwargs
        )
        if len(abinit_inputs) != 1:
            raise RuntimeError(
                f"Should have exactly one previous output. Found {len(abinit_inputs)}"
            )
        previous_abinit_input = next(iter(abinit_inputs.values()))
        n_band = previous_abinit_input.get(
            "nband",
            previous_abinit_input.structure.num_valence_electrons(
                previous_abinit_input.pseudos
            ),
        )
        return int(np.ceil(n_band * self.nbands_factor))


@dataclass
class LineNonSCFSetGenerator(NonSCFSetGenerator):
    """
    Generator for ABINIT non-SCF input sets along high-symmetry lines.

    This class generates input sets for band structure calculations along
    high-symmetry lines in the Brillouin zone.

    Attributes
    ----------
    calc_type : str
        Type of calculation. Default is "nscf_line".
    factory : Callable
        Factory function for generating the input. Default is ebands_from_gsinput.
    """

    calc_type: str = "nscf_line"
    factory: Callable = ebands_from_gsinput


@dataclass
class UniformNonSCFSetGenerator(NonSCFSetGenerator):
    """
    Generator for ABINIT non-SCF input sets with uniform k-point sampling.

    This class generates input sets for density of states (DOS) calculations
    with uniform k-point grids.

    Attributes
    ----------
    calc_type : str
        Type of calculation. Default is "nscf_uniform".
    factory : Callable
        Factory function for generating the input. Default is dos_from_gsinput.
    """

    calc_type: str = "nscf_uniform"
    factory: Callable = dos_from_gsinput


@dataclass
class NscfWfqSetGenerator(NonSCFSetGenerator):
    """
    Generator for non-SCF input sets with k-point grid shifted by q.

    This class generates input sets for non-SCF calculations with a k-point
    grid shifted by a phonon wave vector q.

    Attributes
    ----------
    calc_type : str
        Type of calculation. Default is "wfq".
    factory : Callable
        Factory function for generating the input. Default is wfq_nscf_from_gsinput.
    prev_outputs_deps : tuple
        Dependencies from previous calculations.
        Default is (f"{SCF}:DEN", f"{SCF}:WFK").
    nbands_factor : float
        Factor to multiply the number of bands. Default is 1.0.
    """

    calc_type: str = "wfq"
    factory: Callable = wfq_nscf_from_gsinput
    prev_outputs_deps: tuple = (f"{SCF}:DEN", f"{SCF}:WFK")
    nbands_factor: float = 1.0


@dataclass
class RelaxSetGenerator(AbinitInputGenerator):
    """
    Generator for structure relaxation calculations.

    This class generates input sets for ionic and/or cell relaxation
    calculations in ABINIT.

    Attributes
    ----------
    calc_type : str
        Type of calculation. Default is "relaxation".
    factory : Callable
        Factory function for generating the input. Default is ion_ioncell_relax_input.
    restart_from_deps : tuple
        Dependencies for restarting calculations. Default is GS_RESTART_FROM_DEPS.
    prev_outputs_deps : tuple
        Dependencies from previous calculations. Default is GS_RESTART_FROM_DEPS.
    relax_cell : bool
        Whether to relax the cell in addition to ionic positions. Default is True.
    tolmxf : float
        Tolerance on the maximum force. Default is 5e-5.
    """

    calc_type: str = "relaxation"
    factory: Callable = ion_ioncell_relax_input
    restart_from_deps: tuple = GS_RESTART_FROM_DEPS
    prev_outputs_deps: tuple = GS_RESTART_FROM_DEPS
    relax_cell: bool = True
    tolmxf: float = 5e-5

    def get_abinit_input(
        self,
        structure: Structure | None = None,
        pseudos: PseudoTable | None = None,
        prev_outputs: list[str] | None = None,
        abinit_settings: dict | None = None,
        factory_kwargs: dict | None = None,
        kpoints_settings: dict | KSampling | None = None,
        input_index: int | None = None,
    ) -> AbinitInput:
        """
        Generate an AbinitInput for a relaxation calculation.

        Sets the force tolerance (tolmxf) and determines the appropriate index
        in the MultiDataset based on whether cell relaxation is enabled.

        Parameters
        ----------
        structure : Structure or None
            Pymatgen Structure object. Default is None.
        pseudos : PseudoTable or None
            Pseudopotential table. Default is None.
        prev_outputs : list[str] or None
            List of previous output directories. Default is None.
        abinit_settings : dict or None
            Additional ABINIT keywords to set. Default is None.
        factory_kwargs : dict or None
            Additional factory keywords. Default is None.
        kpoints_settings : dict or KSampling or None
            K-points settings. Default is None.
        input_index : int or None
            Index for MultiDataset selection. If None, automatically determined
            based on relax_cell setting. Default is None.

        Returns
        -------
        AbinitInput
            An AbinitInput object for a relaxation calculation.
        """
        abinit_settings = abinit_settings or {}
        # Note: Consider moving tolmxf setting to the factory function
        abinit_settings["tolmxf"] = self.tolmxf
        if input_index is None:
            input_index = 1 if self.relax_cell else 0

        # Handle the case when no structure is provided or when both a
        # structure and a previous output are provided
        if prev_outputs is not None:
            # Note: Consider using FileClient for remote file handling
            prev_dir = strip_hostname(prev_outputs[-1])
            final_structure = get_final_structure(prev_dir)
            if structure is not None and final_structure != structure:
                if not StructureMatcher().fit(final_structure, structure):
                    logger.warning(
                        "The structure you provided is different from the one "
                        "of the previous output. "
                        "We will go ahead with the one you provided."
                    )
                else:
                    logger.warning(
                        "Both the structure you provided and the one "
                        "from the previous output are deemed to be "
                        "the same. We will go ahead with the one you provided."
                    )
            else:
                structure = final_structure

        return super().get_abinit_input(
            structure=structure,
            pseudos=pseudos,
            prev_outputs=prev_outputs,
            abinit_settings=abinit_settings,
            factory_kwargs=factory_kwargs,
            kpoints_settings=kpoints_settings,
            input_index=input_index,
        )
