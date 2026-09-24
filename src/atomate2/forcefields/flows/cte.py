"""Define the force field thermal expansion maker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from atomate2.common.flows.cte import BaseCTEMaker
from atomate2.forcefields.flows.elastic import ElasticMaker
from atomate2.forcefields.flows.pheasy import PhononMaker
from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker

if TYPE_CHECKING:
    from typing_extensions import Self

    from atomate2.forcefields import MLFF

_DEFAULT_FORCE_FIELD = "MACE-MP-0"


def _get_makers(
    force_field_name: str | MLFF | dict, calculator_kwargs: dict | None = None
) -> dict:
    """Get the relaxation, phonon and elastic makers for one force field."""
    calculator: dict[str, Any] = {
        "force_field_name": force_field_name,
        "calculator_kwargs": dict(calculator_kwargs or {}),
    }
    # the relaxation settings are those of the force field elastic flow
    return {
        "bulk_relax_maker": ForceFieldRelaxMaker(
            relax_cell=True,
            relax_kwargs={"fmax": 0.00001},
            fix_symmetry=True,
            **calculator,
        ),
        "phonon_maker": PhononMaker(
            min_length=12.0,
            bulk_relax_maker=None,
            static_energy_maker=None,
            phonon_displacement_maker=ForceFieldStaticMaker(**calculator),
            cal_anhar_fcs=True,
            displacement_anhar=0.03,
            anhar_fit_methods=("one-shot",),
        ),
        "elastic_maker": ElasticMaker(
            bulk_relax_maker=None,
            elastic_relax_maker=ForceFieldRelaxMaker(
                relax_cell=False,
                relax_kwargs={"fmax": 0.00001},
                fix_symmetry=True,
                **calculator,
            ),
        ),
    }


@dataclass
class CTEMaker(BaseCTEMaker):
    """
    Maker to calculate the thermal expansion with a force field and pheasy.

    A tight relaxation is performed first. The relaxed structure is then passed
    to the pheasy phonon flow, which fits the second- and third-order force
    constants, and to the elastic flow. Neither flow relaxes the structure
    again. Finally, phono3py gives the mode Grueneisen tensors, and the thermal
    expansion tensor follows from them and the elastic tensor. The frequencies
    are not renormalized with temperature.

    By default, all steps use MACE-MP-0. Use :obj:`from_force_field_name` to
    run every step with another force field. The phonon flow fits the force
    constants with the one-shot method from randomly displaced supercells with
    0.03 A displacements, and builds the supercells with min_length=12.0.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    bulk_relax_maker: .ForceFieldRelaxMaker or None
        A maker to perform a tight relaxation on the bulk. Set to None to skip
        the relaxation.
    phonon_maker: .PhononMaker
        The pheasy phonon maker. It must have cal_anhar_fcs=True,
        use_symmetrized_structure=None and bulk_relax_maker=None. Its
        anhar_fit_methods set which force constants are used for the thermal
        expansion.
    elastic_maker: .ElasticMaker
        Maker for the elastic tensor. It must have bulk_relax_maker=None.
    temperatures: list[float]
        Temperatures in K.
    mesh: tuple[int, int, int] | float
        q-point mesh for the mode Grueneisen tensors, or a q-point density used
        as kppa in pymatgen's Kpoints.automatic_density for the unit cell.
    tol_imaginary_modes: float
        If a frequency on the mesh is below -tol_imaginary_modes in THz, a
        warning is raised and the thermal expansion of that fit is not computed.
    min_frequency: float
        Modes below this frequency in THz are left out of the thermal expansion.
    """

    name: str = "cte"
    bulk_relax_maker: ForceFieldRelaxMaker | None = field(
        default_factory=lambda: _get_makers(_DEFAULT_FORCE_FIELD)["bulk_relax_maker"]
    )
    phonon_maker: PhononMaker = field(
        default_factory=lambda: _get_makers(_DEFAULT_FORCE_FIELD)["phonon_maker"]
    )
    elastic_maker: ElasticMaker = field(
        default_factory=lambda: _get_makers(_DEFAULT_FORCE_FIELD)["elastic_maker"]
    )

    @classmethod
    def from_force_field_name(
        cls,
        force_field_name: str | MLFF | dict,
        calculator_kwargs: dict | None = None,
        **kwargs,
    ) -> Self:
        """
        Create a thermal expansion maker that uses one force field for all steps.

        Parameters
        ----------
        force_field_name: str or .MLFF or dict
            The name of the force field.
        calculator_kwargs: dict or None
            Keyword arguments passed to the force field calculator.
        **kwargs
            Further keyword arguments passed to CTEMaker. A maker given here
            replaces the one built for the force field.

        Returns
        -------
        CTEMaker
        """
        return cls(**{**_get_makers(force_field_name, calculator_kwargs), **kwargs})

    @property
    def prev_calc_dir_argname(self) -> None:
        """Name of the argument that passes the previous calculation directory."""
        return
