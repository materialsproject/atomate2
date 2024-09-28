"""Schemas for qha documents."""

import logging
from typing import Optional, Union

import numpy as np
from emmet.core.structure import StructureMetadata
from phonopy.api_qha import PhonopyQHA
from pydantic import Field
from pymatgen.core import Structure
from typing_extensions import Self

logger = logging.getLogger(__name__)


class PhononQHADoc(StructureMetadata, extra="allow"):  # type: ignore[call-arg]
    """Collection of all data produced by the qha workflow."""

    structure: Optional[Structure] = Field(
        None, description="Structure of Materials Project."
    )

    temperatures: Optional[list[float]] = Field(
        None,
        description="temperatures at which the vibrational"
        " part of the free energies"
        " and other properties have been computed",
    )

    bulk_modulus: Optional[float] = Field(
        None, description="Bulk modulus in GPa computed without phonon contribution."
    )
    thermal_expansion: Optional[list[float]] = Field(
        None,
        description="Thermal expansion coefficients at temperatures."
        "Shape=(temperatures, ). ",
    )
    helmholtz_volume: Optional[list[list[float]]] = Field(
        None,
        description="Free energies at temperatures and volumes."
        "shape (temperatures, volumes)",
    )
    volume_temperature: Optional[list[float]] = Field(
        None,
        description="Volumes in Angstrom^3 at temperatures.Shape: (temperatures, )",
    )
    gibbs_temperature: Optional[list[float]] = Field(
        None,
        description="Gibbs free energies in eV at temperatures."
        "Shape: (temperatures, )",
    )
    bulk_modulus_temperature: Optional[list[float]] = Field(
        None,
        description="Bulk modulus in GPa  at temperature.Shape: (temperatures, )",
    )
    heat_capacity_p_numerical: Optional[list[float]] = Field(
        None,
        description="Heat capacities in J/K/mol at constant pressure at temperatures."
        "Shape: (temperatures, )",
    )
    gruneisen_temperature: Optional[list[float]] = Field(
        None,
        description="Gruneisen parameters at temperatures.Shape: (temperatures, )",
    )
    pressure: Optional[float] = Field(
        None, description="Pressure in GPA at which Gibb's energy was computed"
    )
    t_max: Optional[float] = Field(
        None,
        description="Maximum temperature in K up to"
        " which free energy volume curves are evaluated",
    )
    volumes: Optional[list[float]] = Field(None, description="Volumes in Angstrom^3.")
    free_energies: Optional[list[list[float]]] = Field(
        None,
        description="List of free energies in J/mol for per formula unit. "
        "Shape: (temperatuers, volumes)",
    )
    heat_capacities: Optional[list[list[float]]] = Field(
        None,
        description="List of heat capacities in J/K/mol  per formula unit. "
        "Shape: (temperatuers, volumes)",
    )
    entropies: Optional[list[list[float]]] = Field(
        None,
        description="List of entropies in J/(K*mol) per formula unit. "
        "Shape: (temperatuers, volumes) ",
    )
    formula_units: Optional[int] = Field(None, description="Formula units")

    @classmethod
    def from_phonon_runs(
        cls,
        structure: Structure,
        volumes: list[float],
        temperatures: list[float],
        electronic_energies: list[list[float]],
        free_energies: list[list[float]],
        heat_capacities: list[list[float]],
        entropies: list[list[float]],
        t_max: float = None,
        pressure: float = None,
        formula_units: Union[int, None] = None,
        eos_type: str = "vinet",
        **kwargs,
    ) -> Self:
        """Generate qha results.

        Parameters
        ----------
        structure: Structure object
        volumes: list of floats
        temperatures: list of floats
        electronic_energies: list of list of floats
        free_energies: list of list of floats
        heat_capacities: list of list of floats
        entropies: list of list of floats
        t_max: float
        pressure: float
        eos_type: string
            determines eos type used for the fit
        kwargs: dict
            Additional keywords to pass to this method

        Returns
        -------
        .PhononQHADoc
        """
        import warnings

        with warnings.catch_warnings():
            # Phonopy messes with the warnings
            # Turns all warnings into errors

            qha = PhonopyQHA(
                volumes=np.array(volumes),
                electronic_energies=np.array(electronic_energies),
                temperatures=np.array(temperatures),
                free_energy=np.array(free_energies),
                cv=np.array(heat_capacities),
                entropy=np.array(entropies),
                t_max=t_max,
                pressure=pressure,
                eos=eos_type,
            )

        # create some plots here
        # add kwargs to change the names and file types
        qha.plot_helmholtz_volume().savefig(
            f"{kwargs.get('helmholtz_volume_filename', 'helmholtz_volume')}"
            f".{kwargs.get('plot_type', 'pdf')}"
        )
        qha.plot_volume_temperature().savefig(
            f"{kwargs.get('volume_temperature_plot', 'volume_temperature')}"
            f".{kwargs.get('plot_type', 'pdf')}"
        )
        qha.plot_thermal_expansion().savefig(
            f"{kwargs.get('thermal_expansion_plot', 'thermal_expansion')}"
            f".{kwargs.get('plot_type', 'pdf')}"
        )
        qha.plot_gibbs_temperature().savefig(
            f"{kwargs.get('gibbs_temperature_plot', 'gibbs_temperature')}"
            f".{kwargs.get('plot_type', 'pdf')}"
        )
        qha.plot_bulk_modulus_temperature().savefig(
            f"{kwargs.get('bulk_modulus_plot', 'bulk_modulus_temperature')}"
            f".{kwargs.get('plot_type', 'pdf')}"
        )
        qha.plot_heat_capacity_P_numerical().savefig(
            f"{kwargs.get('heat_capacity_plot', 'heat_capacity_P_numerical')}"
            f".{kwargs.get('plot_type', 'pdf')}"
        )
        # qha.plot_heat_capacity_P_polyfit().savefig("heat_capacity_P_polyfit.eps")
        qha.plot_gruneisen_temperature().savefig(
            f"{kwargs.get('gruneisen_temperature_plot', 'gruneisen_temperature')}"
            f".{kwargs.get('plot_type', 'pdf')}"
        )

        qha.write_helmholtz_volume(
            filename=kwargs.get("helmholtz_volume_datafile", "helmholtz_volume.dat")
        )
        qha.write_helmholtz_volume_fitted(
            filename=kwargs.get(
                "helmholtz_volume_fitted_datafile", "helmholtz_volume_fitted.dat"
            ),
            thin_number=kwargs.get("thin_number", 10),
        )
        qha.write_volume_temperature(
            filename=kwargs.get("volume_temperature_datafile", "volume_temperature.dat")
        )
        qha.write_thermal_expansion(
            filename=kwargs.get("thermal_expansion_datafile", "thermal_expansion.dat")
        )
        qha.write_gibbs_temperature(
            filename=kwargs.get("gibbs_temperature_datafile", "gibbs_temperature.dat")
        )
        qha.write_gruneisen_temperature(
            filename=kwargs.get(
                "gruneisen_temperature_datafile", "gruneisen_temperature.dat"
            )
        )
        qha.write_heat_capacity_P_numerical(
            filename=kwargs.get(
                "heat_capacity_datafile", "heat_capacity_P_numerical.dat"
            )
        )
        qha.write_gruneisen_temperature(
            filename=kwargs.get(
                "gruneisen_temperature_datafile", "gruneisen_temperature.dat"
            )
        )

        # write files as well - might be easier for plotting

        return cls.from_structure(
            structure=structure,
            meta_structure=structure,
            bulk_modulus=qha.bulk_modulus[0],  # all bulk moduli are the same
            # (if electronic effects are not treated)
            thermal_expansion=qha.thermal_expansion,
            helmholtz_volume=qha.helmholtz_volume,
            volume_temperature=qha.volume_temperature,
            gibbs_temperature=qha.gibbs_temperature,
            bulk_modulus_temperature=qha.bulk_modulus_temperature,
            heat_capacity_p_numerical=qha.heat_capacity_P_numerical,
            gruneisen_temperature=qha.gruneisen_temperature,
            pressure=pressure,
            t_max=t_max,
            temperatures=temperatures,
            volumes=volumes,
            free_energies=np.array(np.array(free_energies) * 1000.0).tolist(),
            heat_capacities=heat_capacities,
            entropies=entropies,
            formula_units=formula_units,
        )
