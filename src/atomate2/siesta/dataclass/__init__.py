"""
Dataclass modules for SIESTA FDF parameters.

This package provides 28 dataclass modules organized by parameter category,
enabling type-safe, validated parameter handling for SIESTA calculations.

Basic Usage
-----------
Import dataclasses directly from the package:

.. code-block:: python

    from atomate2.siesta.dataclass import (
        RealSpaceGridParameters,
        BasisSetsAndProjectors,
        MolecularDynamicsAndRelaxation,
    )

    # Query unit information
    unit = RealSpaceGridParameters.get_field_unit("mesh_cutoff")
    # Returns: "Ry"

Module Organization
-------------------
**28 registered modules** across **4 tiers**:

**Basic Tier (7 modules)** - Always enabled:
    - GeneralSystemDescriptors
    - Pseudopotentials
    - BasisSetsAndProjectors
    - KPointSampling
    - ExchangeCorrelationFunctionals
    - SpinSettings
    - RealSpaceGridParameters

**Intermediate Tier (6 modules)**:
    - SCFLoopParameters
    - ElectronicStructureCalculationOptions
    - MolecularDynamicsAndRelaxation
    - GeneralConstraints
    - ExternalControlAndScripting
    - ChemicalAnalysis

**Advanced Tier (9 modules)**:
    - PhononCalculations
    - OpticalProperties
    - DensityOfStatesAndBandStructure
    - DFTU
    - ChargeDipoleElectricField
    - Grids
    - Wannier90
    - AuxiliaryForceField
    - Denchar

**Expert Tier (6 modules)**:
    - ParallelOptions
    - SolversAndPerformanceOptions
    - EfficiencyOptions
    - HamiltonianAndOverlapParameters
    - NetcdfOptions
    - RTTDDFT

Base Classes
------------
All dataclasses inherit from :class:`FDFDataclass`, which provides:

- FDF parameter registration and validation
- Unit metadata extraction
- Case-insensitive parameter matching
- Automatic FDF generation
"""

# Base class
from atomate2.siesta.dataclass.auxiliary_force_field import AuxiliaryForceField
from atomate2.siesta.dataclass.base import FDFDataclass
from atomate2.siesta.dataclass.basis_sets_and_projectors import BasisSetsAndProjectors
from atomate2.siesta.dataclass.charge_dipole_electric_field import (
    ChargeDipoleElectricField,
)
from atomate2.siesta.dataclass.chemical_analysis import ChemicalAnalysis
from atomate2.siesta.dataclass.denchar import Denchar
from atomate2.siesta.dataclass.density_of_states_and_band_structure import (
    DensityOfStatesAndBandStructure,
)
from atomate2.siesta.dataclass.dftu import DFTU
from atomate2.siesta.dataclass.efficiency_options import EfficiencyOptions
from atomate2.siesta.dataclass.electronic_structure_calculation_options import (
    ElectronicStructureCalculationOptions,
)
from atomate2.siesta.dataclass.exchange_correlation_functionals import (
    ExchangeCorrelationFunctionals,
)
from atomate2.siesta.dataclass.external_control_and_scripting import (
    ExternalControlAndScripting,
)
from atomate2.siesta.dataclass.general_constraints import GeneralConstraints

# Basic tier (7 modules)
from atomate2.siesta.dataclass.general_system_descriptors import (
    GeneralSystemDescriptors,
)
from atomate2.siesta.dataclass.grids import Grids
from atomate2.siesta.dataclass.hamiltonian_and_overlap_parameters import (
    HamiltonianAndOverlapParameters,
)
from atomate2.siesta.dataclass.kpoint_sampling import KPointSampling
from atomate2.siesta.dataclass.molecular_dynamics_and_relaxation import (
    MolecularDynamicsAndRelaxation,
)
from atomate2.siesta.dataclass.netcdf_options import NetcdfOptions
from atomate2.siesta.dataclass.optical_properties import OpticalProperties

# Expert tier (6 modules)
from atomate2.siesta.dataclass.parallel_options import ParallelOptions

# Advanced tier (9 modules)
from atomate2.siesta.dataclass.phonon_calculations import PhononCalculations
from atomate2.siesta.dataclass.pseudopotentials import Pseudopotentials
from atomate2.siesta.dataclass.real_space_grid_parameters import RealSpaceGridParameters
from atomate2.siesta.dataclass.rttddft import RTTDDFT

# Intermediate tier (6 modules)
from atomate2.siesta.dataclass.scf_loop_parameters import SCFLoopParameters
from atomate2.siesta.dataclass.solvers_and_performance_options import (
    SolversAndPerformanceOptions,
)
from atomate2.siesta.dataclass.spin_settings import SpinSettings

# Not registered (used internally or special cases)
from atomate2.siesta.dataclass.structural_information import (
    StructuralInformationVersion1,
    StructuralInformationVersion2,
)
from atomate2.siesta.dataclass.wannier90 import Wannier90

__all__ = [  # noqa: RUF022  intentionally grouped by SIESTA parameter tier
    # Base class
    "FDFDataclass",
    # Basic tier (7)
    "GeneralSystemDescriptors",
    "Pseudopotentials",
    "BasisSetsAndProjectors",
    "KPointSampling",
    "ExchangeCorrelationFunctionals",
    "SpinSettings",
    "RealSpaceGridParameters",
    # Intermediate tier (6)
    "SCFLoopParameters",
    "ElectronicStructureCalculationOptions",
    "MolecularDynamicsAndRelaxation",
    "GeneralConstraints",
    "ExternalControlAndScripting",
    "ChemicalAnalysis",
    # Advanced tier (9)
    "PhononCalculations",
    "OpticalProperties",
    "DensityOfStatesAndBandStructure",
    "DFTU",
    "ChargeDipoleElectricField",
    "Grids",
    "Wannier90",
    "AuxiliaryForceField",
    "Denchar",
    # Expert tier (6)
    "ParallelOptions",
    "SolversAndPerformanceOptions",
    "EfficiencyOptions",
    "HamiltonianAndOverlapParameters",
    "NetcdfOptions",
    "RTTDDFT",
    # Not registered (internal use)
    "StructuralInformationVersion1",
    "StructuralInformationVersion2",
]
