"""Utility modules for SIESTA input set generation."""

# Import ALL functions from core (main utilities from old utils.py)
# Import from basis_builder
from atomate2.siesta.sets.utils.basis_builder import (
    PAOBasisSpecies,
    PAOShell,
    create_pao_basis,
)
from atomate2.siesta.sets.utils.core import (
    ase_v2_to_pymatgen,
    get_default_initial_magnetic_moments,
    get_magnetic_structure_info,
    pymatgen_to_ase,
    pymatgen_to_ase_v2,
    pymatgen_to_sisl,
    read_outvars,
    set_magnetic_ordering,
    siesta_fdf_to_json,
    write_parameter_evolution_log,
)

# Import from per_atom_basis
from atomate2.siesta.sets.utils.per_atom_basis import (
    apply_diffuse_basis_to_surface,
    apply_per_atom_basis,
    create_per_atom_basis_dict,
    detect_surface_atoms,
)

__all__ = [
    # Core utilities (from old utils.py)
    "pymatgen_to_ase",
    "pymatgen_to_ase_v2",
    "ase_v2_to_pymatgen",
    "pymatgen_to_sisl",
    "get_default_initial_magnetic_moments",
    "set_magnetic_ordering",
    "get_magnetic_structure_info",
    "read_outvars",
    "siesta_fdf_to_json",
    "write_parameter_evolution_log",
    # Basis builder (Phase 4)
    "PAOShell",
    "PAOBasisSpecies",
    "create_pao_basis",
    # Per-atom basis (Phase 3)
    "apply_per_atom_basis",
    "create_per_atom_basis_dict",
    # Surface atom detection (diffuse orbitals)
    "detect_surface_atoms",
    "apply_diffuse_basis_to_surface",
]
