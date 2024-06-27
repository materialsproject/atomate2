"""Schemas for phonon documents."""

import copy
import logging
from pathlib import Path
from typing import Optional, Union
from jobflow import Flow, Response, job
from phonopy import Phonopy
from phonopy.api_qha import PhonopyQHA

import numpy as np
from emmet.core.math import Matrix3D
from emmet.core.structure import StructureMetadata
from monty.json import MSONable
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon
from phonopy.units import VaspToTHz
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.io.phonopy import (get_ph_bs_symm_line, get_ph_dos, get_phonopy_structure, get_pmg_structure, )
from pymatgen.io.vasp import Kpoints
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.symmetry.kpath import KPathSeek
from typing_extensions import Self

from atomate2.aims.utils.units import omegaToTHz

logger = logging.getLogger(__name__)


class PhononQHADoc(StructureMetadata, extra="allow"):  # type: ignore[call-arg]
    """Collection of all data produced by the phonon workflow."""

    structure: Optional[Structure] = Field(None, description="Structure of Materials Project.")

    temperatures: Optional[list[float]] = Field(None, description="temperatures at which the vibrational"
                                                                " part of the free energies"
                                                                " and other properties have been computed", )

    # bulk_modulus: Optional[list[float]] = Field(None, description="")
    # thermal_expansion: Optional[list[float]] = Field(None, description="")
    # helmholtz_volume: Optional[list[float]] = Field(None, description="")
    # volume_temperature: Optional[list[float]] = Field(None, description="")
    # gibbs_temperature: Optional[list[float]] = Field(None, description="")
    # bulk_modulus_temperature: Optional[list[float]] = Field(None, description="")
    # heat_capacity_P_numerical: Optional[list[float]] = Field(None, description="")
    # heat_capacity_P_polyfit: Optional[list[float]] = Field(None, description="")
    # gruneisen_temperature: Optional[list[float]] = Field(None, description="")



    @classmethod
    def from_phonon_runs(cls, structure: Structure,
                         volumes,
                         temperatures, electronic_energies,
                         free_energies,
                         heat_capacities,
                         entropies,
                         stresses,
                         **kwargs, ) -> Self:
        """Generate qha results.

        Parameters
        ----------
        structure: Structure object
        **kwargs:
            additional arguments
        """
        # TODO: figure out how to treat the stresses!

        # put this into a schema and use this information there
        # generate plots and save the data in a schema
        qha = PhonopyQHA(volumes=np.array(volumes), electronic_energies=np.array(electronic_energies),
                         temperatures=np.array(temperatures), free_energy=np.array(free_energies),
                         cv=np.array(heat_capacities), entropy=np.array(entropies))

        # create some plots here
        qha.plot_helmholtz_volume().savefig("helmholtz_volume.eps")
        qha.plot_volume_temperature().savefig("volume_temperature.eps")
        qha.plot_thermal_expansion().savefig("thermal_expansion.eps")
        qha.plot_gibbs_temperature().savefig("gibbs_temperature.eps")
        qha.plot_bulk_modulus_temperature().savefig("bulk_modulus_temperature.eps")
        qha.plot_heat_capacity_P_numerical().savefig("heat_capacity_P_numerical.eps")
        qha.plot_heat_capacity_P_polyfit().savefig("heat_capacity_P_polyfit.eps")
        qha.plot_gruneisen_temperature().savefig("gruneisen_temperature.eps")



        return cls.from_structure(structure=structure,
                                  meta_structure=structure,
                                  # bulk_modulus = qha.bulk_modulus,
                                  # thermal_expansion=qha.thermal_expansion,
                                  # helmholtz_volume=qha.helmholtz_volume,
                                  # volume_temperature=qha.volume_temperature,
                                  # gibbs_temperature=qha.gibbs_temperature,
                                  # bulk_modulus_temperature=qha.bulk_modulus_temperature,
                                  # heat_capacity_P_numerical=qha.heat_capacity_P_numerical,
                                  # heat_capacity_P_polyfit=qha.heat_capacity_P_polyfit,
                                  # gruneisen_temperature=qha.gruneisen_temperature,


                                  ) # TODO: should return some output doc  # have to think about how it should look like  # need to check

