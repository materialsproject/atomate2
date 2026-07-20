"""Test all dataclass files for proper initialization."""

import pytest

dataclass_modules = [
    "auxiliary_force_field",
    "basis_sets_and_projectors",
    "charge_dipole_electric_field",
    "chemical_analysis",
    "denchar",
    "density_of_states_and_band_structure",
    "dftu",
    "efficiency_options",
    "electronic_structure_calculation_options",
    "exchange_correlation_functionals",
    "external_control_and_scripting",
    "general_constraints",
    "general_system_descriptors",
    "grids",
    "hamiltonian_and_overlap_parameters",
    "kpoint_sampling",
    "molecular_dynamics_and_relaxation",
    "netcdf_options",
    "optical_properties",
    "parallel_options",
    "phonon_calculations",
    "pseudopotentials",
    "real_space_grid_parameters",
    "registry",
    "rttddft",
    "scf_loop_parameters",
    "solvers_and_performance_options",
    "spin_settings",
    "structural_information",
    "wannier90",
]


@pytest.mark.parametrize("module_name", dataclass_modules)
def test_dataclass_import(module_name):
    """Test that each dataclass module can be imported."""
    module = __import__(f"atomate2.siesta.dataclass.{module_name}", fromlist=[""])
    assert module is not None


@pytest.mark.parametrize("module_name", dataclass_modules)
def test_dataclass_instantiation(module_name):
    """Test that dataclasses in each module can be instantiated with defaults."""
    module = __import__(f"atomate2.siesta.dataclass.{module_name}", fromlist=[""])

    # Find all dataclasses in module
    dataclasses = []
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        attr = getattr(module, attr_name)
        if hasattr(attr, "__dataclass_fields__"):
            # Skip DataclassModule - it's a registry class, not a user-facing dataclass
            if attr_name == "DataclassModule":
                continue
            dataclasses.append((attr_name, attr))

    # Test instantiation
    for cls_name, cls in dataclasses:
        try:
            instance = cls()
            assert instance is not None, f"Failed to instantiate {cls_name}"
        except Exception as e:
            pytest.fail(f"Failed to instantiate {cls_name}: {e}")
