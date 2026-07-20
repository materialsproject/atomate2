"""Spin configuration utilities for molecular calculations.

This module provides automatic spin state detection for common molecules used in
electrocatalysis and general chemistry. It determines whether spin-polarized DFT
is needed and provides initial magnetic moments for SIESTA calculations.

CRITICAL: For open-shell species (O₂, O, OH, NO), spin polarization is REQUIRED
or SIESTA will converge to the wrong (closed-shell) ground state!
"""

from __future__ import annotations


def get_siesta_spin_config(formula: str) -> dict[str, bool | dict | float | int | None]:
    """
    Auto-detect SIESTA spin configuration for common molecules.

    Provides spin-polarized settings and initial magnetic moments for molecules
    commonly encountered in:
    - Electrocatalysis (O₂, H₂O, OH, OOH intermediates)
    - Thermochemistry (CO₂, CO, N₂, H₂, etc.)
    - Atmospheric chemistry (NO, NO₂, O₃, etc.)
    - Organic radicals

    **WARNING**: For open-shell species (O₂, O, OH, NO), you MUST initialize
    spin density or SIESTA will converge to the wrong (closed-shell) state!

    Parameters
    ----------
    formula : str
        Chemical formula (e.g., "O2", "H2O", "CO2")
        Case-insensitive, accepts various formats: O2, o2, H2O, h2o

    Returns
    -------
    dict
        {
            "spin_polarized": bool,
                True if spin polarization needed, False otherwise
            "init_magnetic_moments": dict or None,
                Element-to-moment mapping (in μB) or None for closed-shell
                Example: {"O": 1.0} means each O atom has 1 μB moment
            "total_spin_moment": float,
                Total magnetic moment of the molecule (μB)
            "n_unpaired_electrons": int,
                Number of unpaired electrons (for SIESTA TotalSpin parameter)
            "fix_spin": bool,
                Whether to use FixSpin in SIESTA (recommended for O₂, radicals)
        }

    Examples
    --------
    >>> config = get_siesta_spin_config("O2")
    >>> config
    {
        'spin_polarized': True,
        'init_magnetic_moments': {'O': 1.0},
        'total_spin_moment': 2.0,
        'n_unpaired_electrons': 2,
        'fix_spin': True
    }

    >>> config = get_siesta_spin_config("H2O")
    >>> config
    {
        'spin_polarized': False,
        'init_magnetic_moments': None,
        'total_spin_moment': 0.0,
        'n_unpaired_electrons': 0,
        'fix_spin': False
    }

    >>> config = get_siesta_spin_config("OH")
    >>> config
    {
        'spin_polarized': True,
        'init_magnetic_moments': {'O': 1.0},
        'total_spin_moment': 1.0,
        'n_unpaired_electrons': 1,
        'fix_spin': True
    }

    Notes
    -----
    **Spin States and Ground States**:
    - O₂: Triplet ³Σg⁻ (S=1, 2 unpaired e⁻) - MUST use spin polarization!
    - O: Triplet ³P (S=1, 2 unpaired e⁻)
    - OH: Doublet ²Π (S=1/2, 1 unpaired e⁻)
    - NO: Doublet ²Π (S=1/2, 1 unpaired e⁻)
    - H₂O, CO₂, N₂, H₂, CO: Singlet (S=0, all paired)

    **SIESTA Implementation**:
    - `init_magnetic_moments` → `structure.add_site_property("magmom", ...)`
    - SIESTA converts to `DM.InitSpin` block automatically
    - `fix_spin=True` → Add `FixSpin: True` to SIESTA input (constrains total spin)
    - For O₂: Without FixSpin, may converge to wrong singlet state!

    **DM.InitSpin vs. FixSpin**:
    - `DM.InitSpin`: INITIALIZES magnetic moments (can change during SCF)
    - `FixSpin`: CONSTRAINS total spin to remain fixed
    - For radicals (O₂, O, OH, NO): Recommend both!

    References
    ----------
    - Nørskov et al., J. Phys. Chem. B 2004, 108, 17886 (CHE model for electrocatalysis)
    - Rossmeisl et al., J. Electroanal. Chem. 2007, 607, 83 (ORR/OER spin states)
    - Mulliken, R. S., J. Chem. Phys. 1955, 23, 1997 (O₂ ground state, triplet)
    """
    # Normalize formula (case-insensitive)
    formula_normalized = formula.strip()

    # Database of spin configurations for common molecules
    spin_config_db: dict[str, dict[str, bool | dict | float | int | None]] = {
        # ===== PARAMAGNETIC MOLECULES (MUST initialize magnetic moments!) =====
        # WARNING: DM.InitSpin only INITIALIZES - use FixSpin to CONSTRAIN!
        "O2": {
            "spin_polarized": True,
            "init_magnetic_moments": {"O": 1.0},  # Each O: 1 μB spin-up, triplet ³Σg⁻
            "total_spin_moment": 2.0,  # Total magnetic moment: 2 μB
            "n_unpaired_electrons": 2,  # 2 unpaired electrons
            "fix_spin": True,  # Recommend fixing total spin for O₂!
        },
        "O": {
            "spin_polarized": True,
            "init_magnetic_moments": {"O": 2.0},  # Triplet ³P ground state
            "total_spin_moment": 2.0,  # 2 μB
            "n_unpaired_electrons": 2,
            "fix_spin": True,
        },
        "OH": {
            "spin_polarized": True,
            "init_magnetic_moments": {"O": 1.0},  # Doublet ²Π
            "total_spin_moment": 1.0,  # 1 μB
            "n_unpaired_electrons": 1,
            "fix_spin": True,
        },
        "OOH": {
            "spin_polarized": True,
            "init_magnetic_moments": {"O": 0.5},  # Doublet, moment distributed
            "total_spin_moment": 1.0,  # 1 μB total
            "n_unpaired_electrons": 1,
            "fix_spin": True,
        },
        "NO": {
            "spin_polarized": True,
            "init_magnetic_moments": {"N": 1.0},  # Doublet ²Π
            "total_spin_moment": 1.0,  # 1 μB
            "n_unpaired_electrons": 1,
            "fix_spin": True,
        },
        "NO2": {
            "spin_polarized": True,
            "init_magnetic_moments": {"N": 1.0},  # Doublet
            "total_spin_moment": 1.0,  # 1 μB
            "n_unpaired_electrons": 1,
            "fix_spin": True,
        },
        # ===== DIAMAGNETIC MOLECULES (closed shell, no initialization needed) =====
        "H2O": {
            "spin_polarized": False,
            "init_magnetic_moments": None,
            "total_spin_moment": 0.0,
            "n_unpaired_electrons": 0,
            "fix_spin": False,
        },
        "N2": {
            "spin_polarized": False,
            "init_magnetic_moments": None,
            "total_spin_moment": 0.0,
            "n_unpaired_electrons": 0,
            "fix_spin": False,
        },
        "CO2": {
            "spin_polarized": False,
            "init_magnetic_moments": None,
            "total_spin_moment": 0.0,
            "n_unpaired_electrons": 0,
            "fix_spin": False,
        },
        "CO": {
            "spin_polarized": False,
            "init_magnetic_moments": None,
            "total_spin_moment": 0.0,
            "n_unpaired_electrons": 0,
            "fix_spin": False,
        },
        "H2": {
            "spin_polarized": False,
            "init_magnetic_moments": None,
            "total_spin_moment": 0.0,
            "n_unpaired_electrons": 0,
            "fix_spin": False,
        },
        "CH4": {
            "spin_polarized": False,
            "init_magnetic_moments": None,
            "total_spin_moment": 0.0,
            "n_unpaired_electrons": 0,
            "fix_spin": False,
        },
        "C2H4": {
            "spin_polarized": False,
            "init_magnetic_moments": None,
            "total_spin_moment": 0.0,
            "n_unpaired_electrons": 0,
            "fix_spin": False,
        },
        "NH3": {
            "spin_polarized": False,
            "init_magnetic_moments": None,
            "total_spin_moment": 0.0,
            "n_unpaired_electrons": 0,
            "fix_spin": False,
        },
        # ===== BULK PRODUCTS (metal peroxides/oxides - closed shell) =====
        "Li2O2": {
            "spin_polarized": False,
            "init_magnetic_moments": None,
            "total_spin_moment": 0.0,
            "n_unpaired_electrons": 0,
            "fix_spin": False,
        },
        "Na2O2": {
            "spin_polarized": False,
            "init_magnetic_moments": None,
            "total_spin_moment": 0.0,
            "n_unpaired_electrons": 0,
            "fix_spin": False,
        },
        "K2O2": {
            "spin_polarized": False,
            "init_magnetic_moments": None,
            "total_spin_moment": 0.0,
            "n_unpaired_electrons": 0,
            "fix_spin": False,
        },
        "Li2O": {
            "spin_polarized": False,
            "init_magnetic_moments": None,
            "total_spin_moment": 0.0,
            "n_unpaired_electrons": 0,
            "fix_spin": False,
        },
        "Na2O": {
            "spin_polarized": False,
            "init_magnetic_moments": None,
            "total_spin_moment": 0.0,
            "n_unpaired_electrons": 0,
            "fix_spin": False,
        },
    }

    # Try exact match first (case-sensitive)
    if formula_normalized in spin_config_db:
        return spin_config_db[formula_normalized]

    # Try case-insensitive match
    formula_upper = formula_normalized.upper()
    for key in spin_config_db:
        if key.upper() == formula_upper:
            return spin_config_db[key]

    # Default: assume closed-shell if not in database
    # (Conservative choice - user can override if needed)
    default_config: dict[str, bool | dict | float | int | None] = {
        "spin_polarized": False,
        "init_magnetic_moments": None,
        "total_spin_moment": 0.0,
        "n_unpaired_electrons": 0,
        "fix_spin": False,
    }

    return default_config
