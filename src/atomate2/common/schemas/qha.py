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

    # free_energies: Optional[list[float]] = Field(
    #     None,
    #     description="vibrational part of the free energies in J/mol per "
    #     "formula unit for temperatures in temperature_list",
    # )
    #
    # heat_capacities: Optional[list[float]] = Field(
    #     None,
    #     description="heat capacities in J/K/mol per "
    #     "formula unit for temperatures in temperature_list",
    # )
    #
    # internal_energies: Optional[list[float]] = Field(
    #     None,
    #     description="internal energies in J/mol per "
    #     "formula unit for temperatures in temperature_list",
    # )
    # entropies: Optional[list[float]] = Field(
    #     None,
    #     description="entropies in J/(K*mol) per formula unit"
    #     "for temperatures in temperature_list ",
    # )
    #
    # temperatures: Optional[list[int]] = Field(
    #     None,
    #     description="temperatures at which the vibrational"
    #     " part of the free energies"
    #     " and other properties have been computed",
    # )
    #
    # total_dft_energy: Optional[float] = Field("total DFT energy per formula unit in eV")
    #
    # stress: Optional[Matrix3D] = Field("The stress on the structure.")
    #
    # volume_per_formula_unit: Optional[float] = Field("volume per formula unit in Angstrom**3")
    #
    # formula_units: Optional[int] = Field("Formula units per cell")
    #
    # has_imaginary_modes: Optional[bool] = Field(
    #     None, description="if true, structure has imaginary modes"
    # )
    #
    # # needed, e.g. to compute Grueneisen parameter etc
    # force_constants: Optional[ForceConstants] = Field(
    #     None, description="Force constants between every pair of atoms in the structure"
    # )
    #
    # born: Optional[list[Matrix3D]] = Field(
    #     None,
    #     description="born charges as computed from phonopy. Only for symmetrically "
    #     "different atoms",
    # )
    #
    # epsilon_static: Optional[Matrix3D] = Field(
    #     None, description="The high-frequency dielectric constant"
    # )
    #
    # supercell_matrix: Matrix3D = Field("matrix describing the supercell")
    # primitive_matrix: Matrix3D = Field(
    #     "matrix describing relationship to primitive cell"
    # )
    #
    # code: str = Field("String describing the code for the computation")
    #
    # phonopy_settings: PhononComputationalSettings = Field(
    #     "Field including settings for Phonopy"
    # )
    #
    # thermal_displacement_data: Optional[ThermalDisplacementData] = Field(
    #     "Includes all data of the computation of the thermal displacements"
    # )
    #
    # jobdirs: Optional[PhononJobDirs] = Field(
    #     "Field including all relevant job directories"
    # )
    #
    # uuids: Optional[PhononUUIDs] = Field("Field including all relevant uuids")

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
        # put this into a schema and use this information there
        # generate plots and save the data in a schema
        qha = PhonopyQHA(volumes=np.array(volumes), electronic_energies=np.array(electronic_energies),
                         temperatures=np.array(temperatures), free_energy=np.array(free_energies),
                         cv=np.array(heat_capacities), entropy=np.array(entropies))

        qha.plot_helmholtz_volume().savefig("helmholtzvolume.eps")
        # qha.plot_volume_temperature().show()
        qha.plot_thermal_expansion().savefig("thermalexpansion.eps")
        # plot = qha.plot_volume_expansion()
        # if plot:
        #     plot.show()
        # qha.plot_gibbs_temperature().show()
        # qha.plot_bulk_modulus_temperature().show()
        # qha.plot_heat_capacity_P_numerical().show()
        # qha.plot_heat_capacity_P_polyfit().show()
        # qha.plot_gruneisen_temperature().show()
        print(qha.thermal_expansion)

        return cls.from_structure(structure=structure,
                                  meta_structure=structure) # TODO: should return some output doc  # have to think about how it should look like  # need to check

