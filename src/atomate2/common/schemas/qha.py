"""Schemas for phonon documents."""

import logging
from typing import Optional

import numpy as np
from emmet.core.structure import StructureMetadata
from phonopy.api_qha import PhonopyQHA
from pydantic import Field
from pymatgen.core import Structure
from typing_extensions import Self

logger = logging.getLogger(__name__)


class PhononQHADoc(StructureMetadata, extra="allow"):  # type: ignore[call-arg]
    """Collection of all data produced by the phonon workflow."""

    structure: Optional[Structure] = Field(
        None, description="Structure of Materials Project."
    )

    temperatures: Optional[list[float]] = Field(
        None,
        description="temperatures at which the vibrational"
        " part of the free energies"
        " and other properties have been computed",
    )

    bulk_modulus: Optional[list[float]] = Field(None, description="")
    thermal_expansion: Optional[list[float]] = Field(None, description="")
    helmholtz_volume: Optional[list[list[float]]] = Field(None, description="")
    volume_temperature: Optional[list[float]] = Field(None, description="")
    gibbs_temperature: Optional[list[float]] = Field(None, description="")
    bulk_modulus_temperature: Optional[list[float]] = Field(None, description="")
    heat_capacity_P_numerical: Optional[list[float]] = Field(None, description="")
    heat_capacity_P_polyfit: Optional[list[float]] = Field(None, description="")
    gruneisen_temperature: Optional[list[float]] = Field(None, description="")
    pressure: Optional[float] = Field(
        None, description="Pressure at GPA at which Gibb's energy was computed"
    )
    t_max: Optional[float] = Field(
        None,
        description="Maximum temperature in K until which Free energy volume curves are evaluated",
    )

    @classmethod
    def from_phonon_runs(
        cls,
        structure: Structure,
        volumes,
        temperatures,
        electronic_energies,
        free_energies,
        heat_capacities,
        entropies,
        t_max=None,
        pressure=None,
        **kwargs,
    ) -> Self:
        """Generate qha results.

        Parameters
        ----------
        structure: Structure object
        **kwargs:
            additional arguments
        """
        # put this into a schema and use this information there
        # generate plots and save the data in a schema
        qha = PhonopyQHA(
            volumes=np.array(volumes),
            electronic_energies=np.array(electronic_energies),
            temperatures=np.array(temperatures),
            free_energy=np.array(free_energies),
            cv=np.array(heat_capacities),
            entropy=np.array(entropies),
            t_max=t_max,
            pressure=pressure,
        )

        # create some plots here
        qha.plot_helmholtz_volume().savefig("helmholtz_volume.eps")
        qha.plot_volume_temperature().savefig("volume_temperature.eps")
        qha.plot_thermal_expansion().savefig("thermal_expansion.eps")
        qha.plot_gibbs_temperature().savefig("gibbs_temperature.eps")
        qha.plot_bulk_modulus_temperature().savefig("bulk_modulus_temperature.eps")
        qha.plot_heat_capacity_P_numerical().savefig("heat_capacity_P_numerical.eps")
        # qha.plot_heat_capacity_P_polyfit().savefig("heat_capacity_P_polyfit.eps")
        qha.plot_gruneisen_temperature().savefig("gruneisen_temperature.eps")

        return cls.from_structure(
            structure=structure,
            meta_structure=structure,
            bulk_modulus=qha.bulk_modulus,
            thermal_expansion=qha.thermal_expansion,
            helmholtz_volume=qha.helmholtz_volume,
            volume_temperature=qha.volume_temperature,
            gibbs_temperature=qha.gibbs_temperature,
            bulk_modulus_temperature=qha.bulk_modulus_temperature,
            heat_capacity_P_numerical=qha.heat_capacity_P_numerical,
            gruneisen_temperature=qha.gruneisen_temperature,
            pressure=pressure,
            t_max=t_max,
        )
