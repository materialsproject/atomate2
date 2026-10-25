"""Flow for Raman spectra with VASP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from jobflow import Flow, Maker

from atomate2.utils.raman import get_encut_from_potcar
from atomate2.vasp.flows.phonons import PhononMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.raman import (
    RamanDielectricMaker,
    calculate_raman_tensors,
    get_gamma_normal_modes,
    run_raman_displacements,
)
from atomate2.vasp.powerups import update_user_incar_settings

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from emmet.core.math import Matrix3D
    from jobflow import Job, OutputReference
    from pymatgen.core import Structure


@dataclass
class RamanMaker(Maker):
    """
    Raman spectra from finite differences of the dielectric tensor.

    Relaxation and phonons (PhononMaker), Gamma normal modes with symmetry-based
    mode selection, LEPSILON runs at -/+ step along each mode, Raman tensors.

    Parameters
    ----------
    name : str
        Name of the flow.
    phonon_maker : PhononMaker
        Relaxation and phonon maker; force constants are always stored.
    raman_maker : BaseVaspMaker
        Maker for the displaced dielectric runs.
    step_size : float
        Displacement along each mode in Angstrom.
    mode_selection : str
        "raman_active" or "all" optical modes.
    degeneracy_tolerance : float
        Frequency tolerance in THz for degenerate modes.
    lorentzian_width : float or None
        Line width in cm^-1 of the stored spectrum.
    encut_enmax_factor : float or None
        ENCUT = factor * max(POTCAR ENMAX) for every VASP job of the flow.
        None keeps the input set defaults.
    bandgap_thresholds : tuple or None
        Band-gap limits (eV) for choosing Rk of the dielectric runs from the band
        gap of the phonon flow (static, else Born, else relaxation run).
        None keeps the Rk of raman_maker.
    kpoints_lengths : tuple
        Rk for band gaps below, between and above the thresholds.
    """

    name: str = "raman"
    phonon_maker: PhononMaker = field(
        default_factory=lambda: PhononMaker(store_force_constants=True)
    )
    raman_maker: BaseVaspMaker = field(default_factory=RamanDielectricMaker)
    step_size: float = 0.005
    mode_selection: Literal["raman_active", "all"] = "raman_active"
    degeneracy_tolerance: float = 1e-3
    lorentzian_width: float | None = 4.0
    encut_enmax_factor: float | None = 1.3
    bandgap_thresholds: tuple[float, float] | None = (1.0, 2.0)
    kpoints_lengths: tuple[float, float, float] = (40.0, 30.0, 20.0)

    def __post_init__(self) -> None:
        """Ensure the phonon flow stores force constants."""
        self.phonon_maker.store_force_constants = True

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
        modes: Sequence[int] | None = None,
        mp_id: str | None = None,
        encut: float | None = None,
        bandgap: float | None = None,
        born: list[Matrix3D] | None = None,
        epsilon_static: Matrix3D | None = None,
        supercell_matrix: Matrix3D | None = None,
    ) -> Flow:
        """
        Make the Raman flow.

        Parameters
        ----------
        structure : Structure
            Input structure.
        prev_dir : str or Path or None
            Previous VASP directory for the relaxation.
        modes : list of int or None
            Explicit 0-based mode indices; overrides mode_selection.
        mp_id : str or None
            Materials Project id stored in the output.
        encut : float or None
            Explicit ENCUT for all jobs; overrides encut_enmax_factor.
        bandgap : float or None
            Band gap in eV for the Rk rule instead of the phonon-flow value.
        born, epsilon_static : optional
            Passed to the phonon flow to skip its Born-charge run.
        supercell_matrix : optional
            Supercell matrix for the phonon flow.
        """
        phonon_flow = self.phonon_maker.make(
            structure,
            prev_dir=prev_dir,
            born=born,
            epsilon_static=epsilon_static,
            supercell_matrix=supercell_matrix,
        )
        ph_out = phonon_flow.output

        modes_job = get_gamma_normal_modes(
            structure=ph_out.structure,
            force_constants=ph_out.force_constants,
            supercell_matrix=ph_out.supercell_matrix,
            primitive_matrix=ph_out.primitive_matrix,
            symprec=self.phonon_maker.symprec,
            step_size=self.step_size,
            mode_selection=self.mode_selection,
            modes=modes,
            degeneracy_tolerance=self.degeneracy_tolerance,
        )
        modes_job.name = "raman normal modes"

        if bandgap is None and self.bandgap_thresholds is not None:
            bandgap = self._get_bandgap_reference(phonon_flow)
        disp_job = run_raman_displacements(
            modes_data=modes_job.output,
            raman_maker=self.raman_maker,
            bandgap=bandgap,
            bandgap_thresholds=self.bandgap_thresholds,
            kpoints_lengths=self.kpoints_lengths,
        )
        disp_job.name = "raman displacements"

        analysis_job = calculate_raman_tensors(
            modes_data=modes_job.output,
            raman_data=disp_job.output,
            mp_id=mp_id,
            lorentzian_width=self.lorentzian_width,
        )
        analysis_job.name = "raman analysis"

        flow = Flow(
            [phonon_flow, modes_job, disp_job, analysis_job],
            output=analysis_job.output,
            name=f"{structure.composition.reduced_formula} {self.name}",
        )

        if encut is None and self.encut_enmax_factor is not None:
            encut = get_encut_from_potcar(
                structure,
                self.raman_maker.input_set_generator,
                self.encut_enmax_factor,
            )
        if encut is not None:
            flow = update_user_incar_settings(flow, {"ENCUT": encut})
        return flow

    def _get_bandgap_reference(self, phonon_flow: Flow | Job) -> OutputReference | None:
        """Return the band gap of the static, Born or last relaxation job."""
        jobs = (
            [j for j, _ in phonon_flow.iterflow()]
            if isinstance(phonon_flow, Flow)
            else [phonon_flow]
        )
        vasp_jobs = [j for j in jobs if isinstance(j.maker, BaseVaspMaker)]
        preferred = [
            getattr(self.phonon_maker, attr, None)
            for attr in ("static_energy_maker", "born_maker")
        ]
        for maker in preferred:
            for j in vasp_jobs:
                if maker is not None and j.name == maker.name:
                    return j.output.output.bandgap
        relax = [j for j in vasp_jobs if "relax" in j.name]
        return relax[-1].output.output.bandgap if relax else None
