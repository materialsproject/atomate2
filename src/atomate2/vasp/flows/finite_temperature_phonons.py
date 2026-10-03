"""Define the VASP maker for finite-temperature phonons."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING

from atomate2.common.flows.finite_temperature_phonons import (
    BaseFiniteTemperaturePhononMaker,
)
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import DielectricMaker, TightRelaxMaker
from atomate2.vasp.jobs.md import MDMaker
from atomate2.vasp.jobs.phonons import PhononDisplacementMaker
from atomate2.vasp.sets.core import (
    LangevinMDSetGenerator,
    MDSetGenerator,
    StaticSetGenerator,
    TightRelaxSetGenerator,
)

if TYPE_CHECKING:
    from jobflow import Maker

    from atomate2.vasp.jobs.base import BaseVaspMaker

# The relaxation, the MD and the phonon displacement calculations use the same
# smearing and k-point density.
_ELECTRONIC_INCAR = {"ISMEAR": 0, "SIGMA": 0.05}
_KPOINTS = {"reciprocal_density": 100}

# The MD only samples the displacements. The forces of the fit come from the
# phonon displacement calculations, which use tighter settings.
_MD_INCAR = {
    "ENCUT": 500,
    "EDIFF": 1e-5,
    "PREC": "Normal",
    "ALGO": "Normal",
    "LREAL": "Auto",
    "NBLOCK": 1,
    "LWAVE": False,
    "LCHARG": False,
    **_ELECTRONIC_INCAR,
}
_STATIC_INCAR = {
    "IBRION": 2,
    "ISIF": 3,
    "NSW": 0,
    "ENCUT": 600,
    "ENAUG": 1360,
    "EDIFF": 1e-7,
    "PREC": "Accurate",
    "ALGO": "Normal",
    "LASPH": True,
    "NELM": 200,
    "LREAL": False,
    "LAECHG": False,
    "LCHARG": False,
    "LWAVE": False,
    **_ELECTRONIC_INCAR,
}
_RELAX_INCAR = {
    key: _STATIC_INCAR[key] for key in ("ENCUT", "ENAUG", "PREC", "LASPH")
} | _ELECTRONIC_INCAR

# INCAR tags of the MD that the flow sets
_MD_FLOW_TAGS = (
    "IBRION",
    "ISIF",
    "LANGEVIN_GAMMA",
    "MDALGO",
    "NSW",
    "POTIM",
    "SMASS",
    "TEBEG",
    "TEEND",
)


def _get_relax_maker() -> DoubleRelaxMaker:
    """Get a tight relaxation with the basis set and smearing of the statics."""
    return DoubleRelaxMaker.from_relax_maker(
        TightRelaxMaker(
            input_set_generator=TightRelaxSetGenerator(
                user_incar_settings=dict(_RELAX_INCAR),
                user_kpoints_settings=dict(_KPOINTS),
            )
        )
    )


def _get_md_maker() -> MDMaker:
    """Get the MD maker, whose temperature, steps and thermostat the flow sets."""
    return MDMaker(
        input_set_generator=MDSetGenerator(
            user_incar_settings=dict(_MD_INCAR),
            user_kpoints_settings=dict(_KPOINTS),
        )
    )


def _get_phonon_displacement_maker() -> PhononDisplacementMaker:
    """Get the static maker for the forces on the snapshots."""
    return PhononDisplacementMaker(
        input_set_generator=StaticSetGenerator(
            user_incar_settings=dict(_STATIC_INCAR),
            user_kpoints_settings=dict(_KPOINTS),
            auto_ispin=True,
        )
    )


@dataclass
class FiniteTemperaturePhononMaker(BaseFiniteTemperaturePhononMaker):
    """
    Maker for effective harmonic phonons at a finite temperature with VASP.

    The relaxation, the MD and the phonon displacement calculations use VASP.
    The functional, the spin settings and +U follow the atomate2 VASP defaults.
    All three use ISMEAR = 0 with SIGMA = 0.05 and the same reciprocal_density
    setting of 100. The relaxation is a double tight relaxation with ENCUT =
    600 eV, ENAUG = 1360 eV and PREC = Accurate, as in the phonon displacement
    calculations. These also use EDIFF = 1e-7. The MD uses ENCUT = 500 eV,
    EDIFF = 1e-5 and PREC = Normal, and writes every step to XDATCAR with
    NBLOCK = 1. The Nose-Hoover thermostat uses MDALGO = 2 and SMASS = 0. The
    Langevin thermostat uses MDALGO = 3 with LANGEVIN_GAMMA = 10 ps^-1 for each
    species. VASP draws the initial velocities of the first MD job. The MD and
    the phonon displacement calculations start from the magnetic moments of the
    relaxed structure, if it has any. They all get the same prev_dir, the
    relaxation directory by default. auto_ispin of the MD and of the phonon
    displacement calculations sets ISPIN from it.

    See :obj:`.BaseFiniteTemperaturePhononMaker` for the workflow.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    bulk_relax_maker: .Maker | None
        Maker for the relaxation of the unit cell. None skips the relaxation.
    born_maker: .BaseVaspMaker | None
        Maker for the Born effective charges and the dielectric tensor, used
        for the non-analytical correction. It is a DielectricMaker by default,
        as in the VASP pheasy phonon workflow. None skips it.
    md_maker: .MDMaker
        Maker for the MD, with an :obj:`.MDSetGenerator`. Its
        user_incar_settings must not contain IBRION, ISIF, LANGEVIN_GAMMA,
        MDALGO, NSW, POTIM, SMASS, TEBEG or TEEND, since the flow sets them.
        NBLOCK must be 1 if it is set.
    phonon_displacement_maker: .BaseVaspMaker
        Maker for the static calculations on the snapshots and the undisplaced
        supercell.
    code: str
        Code of the phonon displacement calculations.
    md_code: str
        Code of the MD.
    """

    bulk_relax_maker: Maker | None = field(default_factory=_get_relax_maker)
    born_maker: BaseVaspMaker | None = field(default_factory=DielectricMaker)
    md_maker: Maker = field(default_factory=_get_md_maker)
    phonon_displacement_maker: Maker = field(
        default_factory=_get_phonon_displacement_maker
    )
    code: str = "vasp"
    md_code: str = "vasp"

    @property
    def prev_calc_dir_argname(self) -> str | None:
        """Name of the argument that passes prev_dir to the phonon displacement maker.

        Returns
        -------
        str
        """
        return "prev_dir"

    def get_md_maker(self, n_steps: int) -> MDMaker:
        """
        Get the VASP MD maker of one MD job.

        Parameters
        ----------
        n_steps: int
            Number of MD steps of the job.

        Returns
        -------
        MDMaker
        """
        md_maker = self.md_maker
        if not isinstance(md_maker, MDMaker) or not isinstance(
            md_maker.input_set_generator, MDSetGenerator
        ):
            raise TypeError(
                "The MD maker must be a VASP MDMaker with an MDSetGenerator."
            )
        generator = md_maker.input_set_generator
        user_incar = generator.user_incar_settings
        if tags := sorted(set(_MD_FLOW_TAGS) & set(user_incar)):
            raise ValueError(f"The flow sets {tags}. Remove them from the MD maker.")
        if int(user_incar.get("NBLOCK", 1)) != 1:
            raise ValueError(
                "The MD must write every step to XDATCAR, so NBLOCK must be 1."
            )
        if isinstance(generator, LangevinMDSetGenerator):
            if self.thermostat != "langevin":
                raise ValueError(
                    "The MD maker has a Langevin thermostat. Set thermostat to "
                    "'langevin'."
                )
        elif self.thermostat == "langevin":
            init_fields = [f.name for f in fields(generator) if f.init]
            generator = LangevinMDSetGenerator(
                **{name: getattr(generator, name) for name in init_fields}
            )
        generator = replace(
            generator,
            ensemble="nvt",
            start_temp=self.temperature,
            end_temp=self.temperature,
            nsteps=n_steps,
            time_step=self.md_time_step,
        )
        return replace(md_maker, input_set_generator=generator)
